"""Mean Reversion adapter — wraps mean_reversion.signals.generator."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseAdapter, HarnessSignal

_TRADING_ROOT = Path(__file__).resolve().parents[3]
_RISK_BACKEND = _TRADING_ROOT / "risk_calculator" / "backend"
for _p in [str(_TRADING_ROOT), str(_RISK_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ACTION_NORM = {
    "BUY": "BUY",
    "SELL": "SELL",
    "SHORT": "SELL",
    "COVER": "BUY",
    "PARTIAL_SELL": "SELL",
    "PARTIAL_COVER": "BUY",
    "HOLD": "HOLD",
}


class MRAdapter(BaseAdapter):
    """Wraps the Mean Reversion strategy signal generator."""

    source = "mr"

    def __init__(self):
        self._cfg = None
        self._ohlcv_cache: dict = {}

    def _get_cfg(self):
        if self._cfg is None:
            from mean_reversion.config import MeanReversionConfig
            self._cfg = MeanReversionConfig()
        return self._cfg

    def _load_ohlcv(self, ticker: str):
        if ticker not in self._ohlcv_cache:
            from app.services.market_data import fetch_ohlcv
            cfg = self._get_cfg()
            self._ohlcv_cache[ticker] = fetch_ohlcv(ticker, cfg.lookback_days)
        return self._ohlcv_cache[ticker]

    def _generate(self, ticker: str) -> HarnessSignal:
        from mean_reversion.signals.generator import generate_signal

        cfg = self._get_cfg()
        ohlc = self._load_ohlcv(ticker)

        if ohlc is None or ohlc.empty:
            return self._hold(ticker, reason="No OHLCV data")

        sig = generate_signal(ticker, ohlc, cfg)

        raw_action = str(sig.action)
        action = _ACTION_NORM.get(raw_action, "HOLD")
        price_col = "Close" if "Close" in ohlc.columns else "close"
        price = float(ohlc[price_col].iloc[-1]) if price_col in ohlc.columns else 0.0

        confidence = float(sig.filtered_strength) if sig.filtered_strength else 0.0
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
