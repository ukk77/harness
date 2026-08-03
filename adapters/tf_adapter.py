"""Trend Following adapter — wraps trend_following.signals.generator."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseAdapter, HarnessSignal

_TRADING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRADING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_ROOT))

_ACTION_NORM = {
    "BUY": "BUY",
    "SELL": "SELL",
    "SHORT": "SELL",
    "COVER": "BUY",
    "HOLD": "HOLD",
}


class TFAdapter(BaseAdapter):
    """Wraps the Trend Following strategy signal generator."""

    source = "tf"

    def __init__(self):
        self._cfg = None
        self._ohlcv_cache: dict = {}

    def _get_cfg(self):
        if self._cfg is None:
            from trend_following.config import TrendFollowingConfig
            self._cfg = TrendFollowingConfig()
        return self._cfg

    def _load_ohlcv(self, ticker: str):
        if ticker not in self._ohlcv_cache:
            from trading_core.market_data import fetch_ohlcv
            cfg = self._get_cfg()
            self._ohlcv_cache[ticker] = fetch_ohlcv(ticker, cfg.lookback_days)
        return self._ohlcv_cache[ticker]

    def _generate(self, ticker: str) -> HarnessSignal:
        from trend_following.signals.generator import generate_signal

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

        # S1: derive exit-context metadata from the strategy's own stop config
        # rather than discarding it — see Master Spec § 10 S1. TF has no native
        # max-hold-days concept, so expected_hold_days is left unset (honest gap).
        suggested_stop_pct: Optional[float] = None
        if sig.atr_stop is not None and price > 0:
            suggested_stop_pct = abs(price - float(sig.atr_stop)) / price

        return HarnessSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            source=self.source,
            price=price,
            suggested_shares=None,
            reason=sig.reason,
            suggested_stop_pct=suggested_stop_pct,
        )
