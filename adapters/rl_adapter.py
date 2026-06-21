"""RL Strategy adapter — wraps rl_strategy.signals.generator."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseAdapter, HarnessSignal

_TRADING_ROOT = Path(__file__).resolve().parents[3]
if str(_TRADING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_ROOT))


class RLAdapter(BaseAdapter):
    """Wraps the RL strategy signal generator."""

    source = "rl"

    def __init__(self, models_dir: Optional[str] = None):
        self._models_dir = Path(models_dir) if models_dir else _TRADING_ROOT / "models"
        self._generators: dict = {}

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

        from rl_strategy.data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        _, current_state = engineer.get_latest_features(ticker, lookback=100)

        if current_state is None:
            return self._hold(ticker, reason="No feature data")

        obs = current_state.values
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
            suggested_shares=float(sig.shares) if sig.shares else None,
            reason=None,
        )
