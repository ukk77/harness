"""Market Regime Detector.

Classifies the current market into one of four regimes using SPY OHLCV data:

  BULL_TREND   — price above 200-SMA, ADX > 25, positive momentum
  BEAR_TREND   — price below 200-SMA, ADX > 25, negative momentum
  HIGH_VOL     — VIX-proxy (20-day realised vol) > 30% annualised
  RANGE_BOUND  — all other conditions (low volatility, no trend)

Used by harness/allocator.py to scale strategy capital weights per regime.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class Regime(str, Enum):
    BULL_TREND  = "bull_trend"
    BEAR_TREND  = "bear_trend"
    HIGH_VOL    = "high_vol"
    RANGE_BOUND = "range_bound"


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ADX (simplified)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr   = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx    = (100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _realised_vol_ann(close: pd.Series, window: int = 20) -> Optional[float]:
    """Annualised realised volatility using last *window* daily returns."""
    rets = close.pct_change().dropna()
    if len(rets) < window:
        return None
    return float(rets.tail(window).std() * np.sqrt(252))


def detect_regime(ohlcv: pd.DataFrame, previous_regime: Optional[Regime] = None) -> Regime:
    """Classify the current market regime from a daily OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume]
               (case-insensitive — accepts both TitleCase and lowercase).
               Index must be datetime-like and sorted ascending.
               Minimum 200 rows recommended for reliable 200-SMA.
        previous_regime: The regime identified in the previous run (for hysteresis).

    Returns:
        Regime enum value.
    """
    # Normalise column names to lowercase so this works with any source
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    if len(close) < 50:
        return Regime.RANGE_BOUND

    # 1. Realised vol check (highest priority — elevated vol overrides trend calls)
    rv = _realised_vol_ann(close, window=20)
    hv_threshold = 0.28 if previous_regime == Regime.HIGH_VOL else 0.30
    if rv is not None and rv > hv_threshold:
        return Regime.HIGH_VOL

    # 2. 200-SMA trend filter
    sma200 = _sma(close, 200).iloc[-1]
    price  = float(close.iloc[-1])
    above_200 = price > sma200 if not np.isnan(sma200) else None

    # 3. ADX trend strength
    adx_series = _adx(high, low, close)
    adx_val = float(adx_series.iloc[-1]) if len(adx_series) > 0 else 0.0
    
    is_trend_prev = previous_regime in (Regime.BULL_TREND, Regime.BEAR_TREND)
    trend_threshold = 20 if is_trend_prev else 25
    trending = adx_val > trend_threshold

    # 4. Momentum (20-day return)
    momentum = float(close.pct_change(20).iloc[-1]) if len(close) >= 21 else 0.0

    if trending and above_200 is True and momentum > 0:
        return Regime.BULL_TREND
    if trending and above_200 is False and momentum < 0:
        return Regime.BEAR_TREND
    return Regime.RANGE_BOUND
