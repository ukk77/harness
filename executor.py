"""Trade executor — routes reconciled signals to paper or live execution."""
from __future__ import annotations

import logging
from typing import Dict, Optional

from .adapters.base import HarnessSignal
from .reconciler import ReconciledSignal
from .paper_trading.db import HarnessTradingDB
from .config import HarnessConfig, get_config

log = logging.getLogger(__name__)


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
    """Routes orders to Alpaca via trading_core.AlpacaBroker (live mode)."""

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()
        self._broker = None

    def _get_broker(self):
        if self._broker is None:
            from trading_core.alpaca_broker import AlpacaBroker
            self._broker = AlpacaBroker()
        return self._broker

    def execute(self, signal: ReconciledSignal, capital: float, dry_run: bool = False) -> bool:
        """Submit a market order to Alpaca.

        Args:
            signal: Reconciled signal
            capital: Available capital
            dry_run: If True, log what would be submitted but don't actually submit.

        Returns:
            True if submitted (or dry_run), False if skipped.
        """
        if signal.action == "HOLD" or signal.price <= 0:
            return False

        shares = signal.suggested_shares
        if shares is None:
            max_position_value = capital * self.cfg.max_position_pct
            shares = max_position_value / signal.price if signal.price > 0 else 0.0

        if shares <= 0:
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
            broker.submit_market_order(
                symbol=signal.ticker,
                qty=shares,
                side=side,
            )
            log.info(
                "ALPACA %s %s  %.2f shares @ ~$%.2f",
                signal.action, signal.ticker, shares, signal.price
            )
            return True
        except Exception as e:
            log.error("Alpaca order failed for %s: %s", signal.ticker, e)
            return False
