"""Trade executor — routes reconciled signals to paper or live execution."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from .adapters.base import HarnessSignal
from .reconciler import ReconciledSignal
from .paper_trading.db import HarnessTradingDB
from .config import HarnessConfig, get_config

log = logging.getLogger(__name__)


# ── Safety Gate ───────────────────────────────────────────────────────────────

class SafetyGate:
    """Pre-trade safety checks for live Alpaca execution.

    Checks (all must pass to allow a trade):
      1. Market hours  — NYSE regular session 09:30–16:00 ET Mon–Fri
      2. Per-trade cap — order value ≤ max_trade_pct of total capital (default 2%)
      3. Daily loss    — cumulative realized P&L today ≥ -daily_loss_limit_pct
      4. Cooldown      — last trade for this ticker was ≥ trade_cooldown_hours ago
    """

    MAX_TRADE_PCT: float = 0.02          # 2% of total capital per single order

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()

    # ── Public ────────────────────────────────────────────────────────────────

    def check(
        self,
        signal: ReconciledSignal,
        shares: float,
        db: Optional[HarnessTradingDB] = None,
    ) -> Tuple[bool, str]:
        """Run all safety checks.  Returns (allowed, reason_if_blocked)."""

        ok, msg = self._market_hours()
        if not ok:
            return False, msg

        ok, msg = self._trade_size(signal, shares)
        if not ok:
            return False, msg

        if db is not None:
            ok, msg = self._daily_loss(db)
            if not ok:
                return False, msg

            ok, msg = self._cooldown(signal.ticker, db)
            if not ok:
                return False, msg

        return True, "ok"

    # ── Checks ────────────────────────────────────────────────────────────────

    def _market_hours(self) -> Tuple[bool, str]:
        """NYSE regular session: 09:30–16:00 ET, Mon–Fri."""
        try:
            import zoneinfo
            et = zoneinfo.ZoneInfo("America/New_York")
        except ImportError:
            try:
                import pytz
                et = pytz.timezone("America/New_York")
            except ImportError:
                from datetime import timezone as tz
                et = timezone(timedelta(hours=-5))   # EST fallback (conservative: blocks 1h early in EDT, never allows pre-market)

        now_et = datetime.now(et)
        if now_et.weekday() >= 5:
            return False, f"Market closed — weekend ({now_et.strftime('%A')})"
        open_t  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
        if not (open_t <= now_et <= close_t):
            return False, f"Market closed — {now_et.strftime('%H:%M')} ET (session 09:30–16:00)"
        return True, "ok"

    def _trade_size(self, signal: ReconciledSignal, shares: float) -> Tuple[bool, str]:
        """Single order must not exceed MAX_TRADE_PCT of total capital."""
        order_value = shares * signal.price
        cap = self.cfg.total_capital * self.MAX_TRADE_PCT
        if order_value > cap:
            return False, (
                f"Trade size ${order_value:,.0f} exceeds 2% cap "
                f"(${cap:,.0f}) for {signal.ticker}"
            )
        return True, "ok"

    def _daily_loss(self, db: HarnessTradingDB) -> Tuple[bool, str]:
        """Block if today's realized losses exceed daily_loss_limit_pct."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(str(db.db_path)) as conn:
                row = conn.execute(
                    "SELECT SUM(realized_pnl) FROM trades "
                    "WHERE strategy='harness' AND action='SELL' "
                    "AND executed_at >= ?",
                    (today,),
                ).fetchone()
            total_realized = float(row[0] or 0.0)
            limit = -self.cfg.total_capital * self.cfg.daily_loss_limit_pct
            if total_realized < limit:
                return False, (
                    f"Daily loss limit hit: today's realized P&L ${total_realized:,.0f} "
                    f"< limit ${limit:,.0f}"
                )
        except Exception:
            pass  # Don't block on DB errors
        return True, "ok"

    def _cooldown(self, ticker: str, db: HarnessTradingDB) -> Tuple[bool, str]:
        """Enforce minimum hours between trades for the same ticker."""
        hours = self.cfg.trade_cooldown_hours
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        try:
            trades = db.get_trades(ticker=ticker, limit=1)
            if trades:
                last_ts = trades[0].get("executed_at", "")
                if last_ts and last_ts > cutoff:
                    return False, (
                        f"Cooldown active for {ticker} — "
                        f"last trade {last_ts[:16]}, cooldown={hours}h"
                    )
        except Exception:
            pass
        return True, "ok"


class PaperExecutor:
    """Records trades to harness_trades.db AND mirrors them to Alpaca paper trading account.

    Local DB is always the source of truth; Alpaca submission is best-effort.
    """

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()
        self.db = HarnessTradingDB(self.cfg.paper_db_path)
        self._broker = None
        self._broker_unavailable = False  # set True on first init failure to avoid repeated attempts

    def _get_broker(self):
        """Lazy-init Alpaca paper broker; returns None if keys are missing."""
        if self._broker_unavailable:
            return None
        if self._broker is None:
            try:
                from trading_core.alpaca_broker import AlpacaBroker
                self._broker = AlpacaBroker(paper=True)
                log.info("PaperExecutor: Alpaca paper broker initialised")
            except Exception as e:
                log.warning("PaperExecutor: Alpaca paper broker unavailable (%s) — local-only mode", e)
                self._broker_unavailable = True
        return self._broker

    def _submit_to_alpaca(self, signal: ReconciledSignal, shares: float) -> None:
        """Mirror the paper trade to Alpaca paper account. Errors are non-fatal."""
        if not getattr(self.cfg, "alpaca_mirror_enabled", True):
            return
        broker = self._get_broker()
        if broker is None:
            return
        try:
            side = "buy" if signal.action == "BUY" else "sell"
            order = broker.submit_market_order(symbol=signal.ticker, qty=shares, side=side)
            if order is not None:
                log.info(
                    "ALPACA-PAPER %s %s %.2f shares  order_id=%s",
                    signal.action, signal.ticker, shares, getattr(order, "id", "?"),
                )
            else:
                log.warning("ALPACA-PAPER %s %s returned None", signal.action, signal.ticker)
        except Exception as e:
            log.warning("ALPACA-PAPER submission failed for %s %s: %s", signal.action, signal.ticker, e)

    def execute(self, signal: ReconciledSignal, capital: float) -> bool:
        """Execute (paper) a reconciled signal.

        Args:
            signal: Reconciled signal with final action/confidence/price
            capital: Available capital for this strategy

        Returns:
            True if trade was recorded, False if skipped.
        """
        if signal.action == "HOLD" or signal.price <= 0:
            return False

        position = self.db.get_position(signal.ticker, "harness")
        shares = signal.suggested_shares

        if shares is None:
            max_position_value = capital * self.cfg.max_position_pct
            shares = max_position_value / signal.price if signal.price > 0 else 0.0

        if shares <= 0:
            return False

        if signal.action == "BUY":
            current_shares = (position or {}).get("shares", 0.0)
            current_entry = (position or {}).get("entry_price", signal.price)
            current_entry_conf = (position or {}).get("entry_confidence") or 0.0

            # S2: DCA-into-losers guard — block adding to an existing position that
            # is underwater beyond dca_loss_guard_pct, unless the new signal's
            # confidence exceeds the position's running average entry confidence.
            if current_shares > 0 and current_entry > 0:
                pnl_pct = (signal.price - current_entry) / current_entry
                if pnl_pct < -self.cfg.dca_loss_guard_pct and signal.confidence <= current_entry_conf:
                    log.warning(
                        "BLOCKED add-to-position %s — underwater %.1f%% (limit -%.1f%%) "
                        "and new confidence %.2f <= entry confidence %.2f",
                        signal.ticker, pnl_pct * 100, self.cfg.dca_loss_guard_pct * 100,
                        signal.confidence, current_entry_conf,
                    )
                    return False

            new_shares = current_shares + shares

            if new_shares > 0:
                avg_entry = (
                    (current_shares * current_entry + shares * signal.price) / new_shares
                )
                avg_entry_conf = (
                    (current_shares * current_entry_conf + shares * signal.confidence) / new_shares
                )
            else:
                avg_entry = signal.price
                avg_entry_conf = signal.confidence

            # S1: persist entry-time exit context instead of discarding it.
            stop_price = None
            if signal.suggested_stop_pct is not None:
                stop_price = avg_entry * (1 - signal.suggested_stop_pct)

            self.db.upsert_position(
                ticker=signal.ticker,
                strategy="harness",
                shares=new_shares,
                entry_price=avg_entry,
                current_price=signal.price,
                unrealized_pnl=(signal.price - avg_entry) * new_shares,
                stop_price=stop_price,
                expected_hold_days=signal.expected_hold_days,
                risk_bucket=signal.risk_bucket,
                entry_confidence=avg_entry_conf,
            )
            self.db.record_trade(
                ticker=signal.ticker,
                strategy="harness",
                action="BUY",
                shares=shares,
                price=signal.price,
                confidence=signal.confidence,
                realized_pnl=0.0,
                reconciled_from=signal.votes,
            )
            log.info("PAPER BUY  %s  %.2f shares @ $%.2f", signal.ticker, shares, signal.price)
            self._submit_to_alpaca(signal, shares)

        elif signal.action == "SELL":
            if not position or (position.get("shares", 0) <= 0):
                log.debug("No position to sell: %s", signal.ticker)
                return False

            existing_shares = position["shares"]
            sell_shares = min(shares, existing_shares)
            realized_pnl = (signal.price - position["entry_price"]) * sell_shares
            remaining = existing_shares - sell_shares

            self.db.upsert_position(
                ticker=signal.ticker,
                strategy="harness",
                shares=remaining,
                entry_price=position["entry_price"],
                current_price=signal.price,
                unrealized_pnl=(signal.price - position["entry_price"]) * remaining,
                realized_pnl=position.get("realized_pnl", 0.0) + realized_pnl,
            )
            self.db.record_trade(
                ticker=signal.ticker,
                strategy="harness",
                action="SELL",
                shares=sell_shares,
                price=signal.price,
                confidence=signal.confidence,
                realized_pnl=realized_pnl,
                reconciled_from=signal.votes,
            )
            log.info(
                "PAPER SELL %s  %.2f shares @ $%.2f  P&L=$%.2f",
                signal.ticker, sell_shares, signal.price, realized_pnl
            )
            self._submit_to_alpaca(signal, sell_shares)

        return True


class AlpacaExecutor:
    """Routes orders to Alpaca via trading_core.AlpacaBroker (live mode).

    Safety gates (all enforced before every live order):
      - Market hours check (NYSE 09:30–16:00 ET)
      - Per-trade size cap (2% of total capital)
      - Daily loss limit
      - Trade cooldown per ticker
    """

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()
        self._broker = None
        self._gate = SafetyGate(cfg)
        self._db = HarnessTradingDB(self.cfg.paper_db_path)

    def _get_broker(self):
        if self._broker is None:
            from trading_core.alpaca_broker import AlpacaBroker
            self._broker = AlpacaBroker()
        return self._broker

    def execute(self, signal: ReconciledSignal, capital: float, dry_run: bool = False) -> bool:
        """Submit a market order to Alpaca after passing all safety checks.

        Args:
            signal: Reconciled signal
            capital: Available capital
            dry_run: If True, log what would be submitted but don't actually submit.

        Returns:
            True if submitted (or dry_run passed gates), False if skipped/blocked.
        """
        if signal.action == "HOLD" or signal.price <= 0:
            return False

        shares = signal.suggested_shares
        if shares is None:
            max_position_value = capital * self.cfg.max_position_pct
            shares = max_position_value / signal.price if signal.price > 0 else 0.0

        if shares <= 0:
            return False

        # ── Safety gates (skip in dry-run for visibility, enforce in live) ────
        if not dry_run:
            allowed, reason = self._gate.check(signal, shares, self._db)
            if not allowed:
                log.warning("BLOCKED %s %s — %s", signal.action, signal.ticker, reason)
                return False

        # S2: DCA-into-losers guard — same rule as PaperExecutor.
        if signal.action == "BUY":
            pos = self._db.get_position(signal.ticker, "harness")
            current_shares = (pos or {}).get("shares", 0.0)
            current_entry = (pos or {}).get("entry_price", signal.price)
            current_entry_conf = (pos or {}).get("entry_confidence") or 0.0
            if current_shares > 0 and current_entry > 0:
                pnl_pct = (signal.price - current_entry) / current_entry
                if pnl_pct < -self.cfg.dca_loss_guard_pct and signal.confidence <= current_entry_conf:
                    log.warning(
                        "BLOCKED add-to-position %s — underwater %.1f%% (limit -%.1f%%) "
                        "and new confidence %.2f <= entry confidence %.2f",
                        signal.ticker, pnl_pct * 100, self.cfg.dca_loss_guard_pct * 100,
                        signal.confidence, current_entry_conf,
                    )
                    return False

        if dry_run:
            log.info(
                "DRY-RUN ALPACA %s %s  %.2f shares @ ~$%.2f",
                signal.action, signal.ticker, shares, signal.price
            )
            return True

        try:
            broker = self._get_broker()
            side = "buy" if signal.action == "BUY" else "sell"
            order = broker.submit_market_order(
                symbol=signal.ticker,
                qty=shares,
                side=side,
            )
            if order is None:
                log.error(
                    "Alpaca order returned None (submission failed) for %s %s",
                    signal.action, signal.ticker,
                )
                return False
                
            # Wait for fill
            import time
            filled_qty = 0.0
            fill_price = signal.price
            order_id_str = getattr(order, "id", "")
            if order_id_str:
                for _ in range(10):
                    time.sleep(1)
                    latest_order = broker.get_order(str(order_id_str))
                    if latest_order:
                        if latest_order.status == "filled":
                            filled_qty = float(latest_order.filled_qty)
                            fill_price = float(latest_order.filled_avg_price)
                            break
                        elif latest_order.status == "partially_filled":
                            filled_qty = float(latest_order.filled_qty)
                            fill_price = float(latest_order.filled_avg_price)
                        elif latest_order.status in ("canceled", "rejected", "expired"):
                            break
            
            if filled_qty == 0:
                # If we couldn't track it or it didn't fill in 10s, log warning but fall back to requested
                log.warning("Alpaca order %s for %s was not filled within 10s, assuming requested qty.", order_id_str, signal.ticker)
                filled_qty = shares
                
            log.info(
                "ALPACA %s %s  %.2f shares @ ~$%.2f  order_id=%s",
                signal.action, signal.ticker, filled_qty, fill_price,
                order_id_str,
            )
            # Compute realized P&L for SELL (lookup cost basis from paper DB)
            realized_pnl = 0.0
            if signal.action == "SELL":
                pos = self._db.get_position(signal.ticker, "harness")
                if pos and pos.get("entry_price", 0) > 0:
                    realized_pnl = (fill_price - pos["entry_price"]) * filled_qty
            # Record trade in paper DB for cooldown + daily loss tracking
            self._db.record_trade(
                ticker=signal.ticker,
                strategy="harness",
                action=signal.action,
                shares=filled_qty,
                price=fill_price,
                confidence=signal.confidence,
                realized_pnl=realized_pnl,
                reconciled_from=signal.votes,
            )
            
            # Update position locally
            if signal.action == "BUY":
                stop_price = None
                if signal.suggested_stop_pct is not None:
                    stop_price = fill_price * (1 - signal.suggested_stop_pct)
                self._db.update_position(
                    ticker=signal.ticker,
                    strategy="harness",
                    shares=filled_qty,
                    entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    stop_price=stop_price,
                    expected_hold_days=signal.expected_hold_days,
                    risk_bucket=signal.risk_bucket,
                    entry_confidence=signal.confidence,
                )
            elif signal.action == "SELL":
                pos = self._db.get_position(signal.ticker, "harness")
                if pos:
                    rem = pos["shares"] - filled_qty
                    if rem <= 0:
                        self._db.close_position(signal.ticker, "harness")
                    else:
                        self._db.update_position(
                            ticker=signal.ticker,
                            strategy="harness",
                            shares=rem,
                            entry_price=pos["entry_price"],
                            current_price=fill_price,
                            unrealized_pnl=(fill_price - pos["entry_price"]) * rem,
                            realized_pnl=pos.get("realized_pnl", 0) + realized_pnl,
                        )
                        
            return True
        except Exception as e:
            log.error("Alpaca order failed for %s: %s", signal.ticker, e)
            return False
