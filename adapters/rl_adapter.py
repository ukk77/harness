"""RL Strategy adapter — wraps rl_strategy.signals.generator."""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .base import BaseAdapter, HarnessSignal

_TRADING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRADING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_ROOT))

log = logging.getLogger(__name__)

_CACHE_TTL = 60.0  # seconds — refresh Alpaca account once per signal cycle


class RLAdapter(BaseAdapter):
    """Wraps the RL strategy signal generator."""

    source = "rl"

    def __init__(self, models_dir: Optional[str] = None):
        self._models_dir = Path(models_dir) if models_dir else _TRADING_ROOT / "models"
        self._generators: dict = {}
        self._cached_cash: Optional[float] = None
        self._cached_positions: Dict[str, float] = {}  # ticker -> shares
        self._cache_ts: float = 0.0

    # ── Account state ─────────────────────────────────────────────────────────

    def _refresh_account(self) -> None:
        """Fetch cash and positions from Alpaca paper account. Falls back to harness DB."""
        # Primary: Alpaca paper account
        try:
            from trading_core.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(paper=True)
            info = broker.get_account_info()
            self._cached_cash = float(info["cash"])
            alpaca_positions = broker.get_positions()
            self._cached_positions = {
                sym: float(pos["shares"])
                for sym, pos in alpaca_positions.items()
            }
            self._cache_ts = time.time()
            log.debug("[rl_adapter] Account refreshed from Alpaca: cash=%.2f, positions=%d",
                      self._cached_cash, len(self._cached_positions))
            return
        except Exception as e:
            log.warning("[rl_adapter] Alpaca unavailable (%s) — falling back to harness DB", e)

        # Fallback: estimate cash from harness paper DB
        try:
            from harness.paper_trading.db import HarnessTradingDB
            from harness.config import get_config
            cfg = get_config()
            db = HarnessTradingDB(cfg.paper_db_path)
            positions = db.get_all_positions()
            position_cost = sum(p["shares"] * p["entry_price"] for p in positions)
            realized_pnl = db.total_realized_pnl()
            self._cached_cash = 100_000.0 - position_cost + realized_pnl
            self._cached_positions = {
                p["ticker"]: float(p["shares"])
                for p in positions
                if p["shares"] > 0
            }
            self._cache_ts = time.time()
            log.debug("[rl_adapter] Account refreshed from harness DB: cash=%.2f", self._cached_cash)
            return
        except Exception as e:
            log.warning("[rl_adapter] Harness DB fallback failed (%s) — using $100k default", e)

        self._cached_cash = 100_000.0
        self._cached_positions = {}
        self._cache_ts = time.time()

    def invalidate_cache(self) -> None:
        """Force a fresh Alpaca fetch on the next signal generation call."""
        self._cache_ts = 0.0

    def _ensure_fresh(self) -> None:
        if self._cached_cash is None or (time.time() - self._cache_ts) > _CACHE_TTL:
            self._refresh_account()

    def _available_cash(self) -> float:
        self._ensure_fresh()
        return self._cached_cash or 100_000.0

    def _position_shares(self, ticker: str) -> float:
        self._ensure_fresh()
        return self._cached_positions.get(ticker, 0.0)

    # ── Generator ─────────────────────────────────────────────────────────────

    def _get_generator(self, ticker: str):
        """Lazy-load and cache the signal generator per ticker."""
        if ticker not in self._generators:
            from rl_strategy.signals.generator import create_generator
            model_path = self._models_dir / f"{ticker}_ppo.zip"
            if not model_path.exists():
                raise FileNotFoundError(f"No RL model for {ticker}")
            gen = create_generator(ticker, str(model_path))
            if gen is None:
                raise RuntimeError(f"Failed to create RL generator for {ticker}")
            self._generators[ticker] = gen
        return self._generators[ticker]

    def _generate(self, ticker: str) -> HarnessSignal:
        gen = self._get_generator(ticker)

        # Sync env state from real account so the agent observes accurate context
        env = gen.agent.env
        env.cash = self._available_cash()
        env.position_shares = self._position_shares(ticker)

        from rl_strategy.data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        _, current_state = engineer.get_latest_features(ticker, lookback=100)

        if current_state is None:
            return self._hold(ticker, reason="No feature data")

        # Build observation with actual position flag from real account
        position_flag = 1.0 if env.position_shares > 0 else 0.0
        obs = np.append(current_state.values.astype(np.float32), [position_flag])
        sig = gen.generate_signal(observation=obs)

        action = sig.action
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        return HarnessSignal(
            ticker=ticker,
            timestamp=sig.timestamp,
            action=action,
            confidence=float(sig.confidence),
            source=self.source,
            price=float(sig.price) if sig.price else 0.0,
            suggested_shares=None,  # executor sizes from allocated capital
            reason=None,
        )
