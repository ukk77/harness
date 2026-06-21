"""Volatility Breakout adapter — wraps volatility_breakout.signals.generator."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from .base import BaseAdapter, HarnessSignal

_TRADING_ROOT = Path(__file__).resolve().parents[3]
_RISK_BACKEND = _TRADING_ROOT / "risk_calculator" / "backend"
for _p in [str(_TRADING_ROOT), str(_RISK_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class VBAdapter(BaseAdapter):
    """Wraps the Volatility Breakout strategy signal generator."""

    source = "vb"

    def __init__(self):
        self._cfg = None
        self._ohlcv_cache: dict = {}

    def _get_cfg(self):
        if self._cfg is None:
            from volatility_breakout.config import VolatilityBreakoutConfig
            self._cfg = VolatilityBreakoutConfig()
        return self._cfg

    def _load_ohlcv(self, ticker: str):
        if ticker not in self._ohlcv_cache:
            from app.services.market_data import fetch_ohlcv
            cfg = self._get_cfg()
            lookback = getattr(cfg, "lookback_days", 365)
            self._ohlcv_cache[ticker] = fetch_ohlcv(ticker, lookback)
        return self._ohlcv_cache[ticker]

    def _generate(self, ticker: str) -> HarnessSignal:
        from volatility_breakout.signals.generator import generate_signal

        cfg = self._get_cfg()
        ohlc = self._load_ohlcv(ticker)

        if ohlc is None or ohlc.empty:
            return self._hold(ticker, reason="No OHLCV data")

        sig = generate_signal(ticker, ohlc, cfg)

        raw_action = str(sig.action)
        if hasattr(raw_action, "value"):
            raw_action = raw_action.value
        action = raw_action if raw_action in ("BUY", "SELL", "HOLD") else "HOLD"

        price = float(sig.price) if sig.price else 0.0

        confidence = float(sig.kelly_fraction) if sig.kelly_fraction else 0.0
        if action == "BUY" and confidence == 0.0:
            confidence = 0.5
        if action == "HOLD":
            confidence = 0.0

        return HarnessSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            source=self.source,
            price=price,
            suggested_shares=None,
            reason=sig.reason,
        )
