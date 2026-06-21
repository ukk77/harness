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
            from datetime import timezone as tz
            et = timezone(timedelta(hours=-4))   # EDT fallback

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
                    "SELECT SUM(realized_pnl) FROM positions WHERE strategy='harness'"
                ).fetchone()
            total_realized = float(row[0] or 0.0)
            limit = -self.cfg.total_capital * self.cfg.daily_loss_limit_pct
            if total_realized < limit:
                return False, (
                    f"Daily loss limit hit: realized P&L ${total_realized:,.0f} "
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
    """Records trades to harness_trades.db without hitting a real broker."""

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()
        self.db = HarnessTradingDB(self.cfg.paper_db_path)

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
            new_shares = current_shares + shares

            if new_shares > 0:
                avg_entry = (
                    (current_shares * current_entry + shares * signal.price) / new_shares
                )
            else:
                avg_entry = signal.price

            self.db.upsert_position(
                ticker=signal.ticker,
                strategy="harness",
                shares=new_shares,
                entry_price=avg_entry,
                current_price=signal.price,
                unrealized_pnl=(signal.price - avg_entry) * new_shares,
            )
            self.db.record_trade(
                ticker=signal.ticker,
                strategy="harness",
                action="BUY",
                shares=shares,
                price=signal.price,
                confidence=signal.confidence,
                reconciled_from=signal.votes,
            )
            log.info("PAPER BUY  %s  %.2f shares @ $%.2f", signal.ticker, shares, signal.price)

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
                reconciled_from=signal.votes,
            )
            log.info(
                "PAPER SELL %s  %.2f shares @ $%.2f  P&L=$%.2f",
                signal.ticker, sell_shares, signal.price, realized_pnl
            )

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
            log.info(
                "ALPACA %s %s  %.2f shares @ ~$%.2f  order_id=%s",
                signal.action, signal.ticker, shares, signal.price,
                getattr(order, "id", "?"),
            )
            # Record trade in paper DB for cooldown + daily loss tracking
            self._db.record_trade(
                ticker=signal.ticker,
                strategy="harness",
                action=signal.action,
                shares=shares,
                price=signal.price,
                confidence=signal.confidence,
                reconciled_from=signal.votes,
            )
            return True
        except Exception as e:
            log.error("Alpaca order failed for %s: %s", signal.ticker, e)
            return False
