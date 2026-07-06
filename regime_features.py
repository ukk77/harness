"""Feature engineering for the learned regime classifier (Phase 4, A4).

Shared between training (`scripts/train_regime.py`) and live inference
(`regime_model.py` via `regime.py`) so features are computed **identically**
in both places — no train/inference skew.

Design constraint (documented decision): the Kaggle training dataset
(`harness/data/archive_extracted/stock_market_regimes_2000_2026.csv`) has
columns for fed_funds_rate / unemployment_rate / cpi / 10y_treasury /
2y_treasury, but there is no live source for any of these anywhere in the
codebase (no FRED integration). Training on them would silently break at
inference time. They are therefore NOT used as model features. Only two
feature groups are used, both of which have a live-fetchable equivalent:

  1. SPY OHLC-derived technicals (ADX, 200-SMA distance %, momentum, realised
     vol) — computed from real SPY OHLC in both training (yfinance history)
     and inference (existing `trading_core.market_data.fetch_ohlcv("SPY")`
     Parquet cache).
  2. VIX level — from the Kaggle dataset's `vix` column at training time;
     live-fetched via yfinance `^VIX` at inference time, with a graceful
     fallback to the realised-vol proxy (same one the heuristic already
     uses) if the live fetch fails.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Canonical feature order — must match at train time and inference time.
FEATURE_COLUMNS = [
    "realised_vol_20d",
    "adx_14",
    "sma200_dist_pct",
    "mom_20d",
    "mom_5d",
    "rsi_14",
    "vix",
]


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ADX (simplified) — identical implementation to regime.py's heuristic."""
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


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _realised_vol_ann(close: pd.Series, window: int = 20) -> pd.Series:
    rets = close.pct_change()
    return rets.rolling(window, min_periods=window).std() * np.sqrt(252)


def compute_price_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the price-technical feature columns for every row of a daily OHLC DataFrame.

    Args:
        ohlcv: DataFrame with columns [open, high, low, close] (case-insensitive),
               index datetime-like and sorted ascending.

    Returns:
        DataFrame indexed the same as `ohlcv`, with columns:
        realised_vol_20d, adx_14, sma200_dist_pct, mom_20d, mom_5d, rsi_14
        (NaN for rows before enough history has accumulated).
    """
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    sma200 = _sma(close, 200)

    out = pd.DataFrame(index=df.index)
    out["realised_vol_20d"] = _realised_vol_ann(close, window=20)
    out["adx_14"] = _adx(high, low, close, period=14)
    out["sma200_dist_pct"] = (close - sma200) / sma200
    out["mom_20d"] = close.pct_change(20)
    out["mom_5d"] = close.pct_change(5)
    out["rsi_14"] = _rsi(close, period=14)
    return out


def fetch_vix_live(fallback_realised_vol: Optional[float] = None) -> Optional[float]:
    """Fetch the latest VIX close via yfinance. Never raises.

    Args:
        fallback_realised_vol: if the live fetch fails, and this is provided
            (annualised realised vol as a fraction, e.g. 0.22), return
            `fallback_realised_vol * 100` as a VIX-level proxy (same scale
            the heuristic already treats as a VIX proxy). Otherwise returns
            None on failure.

    Returns:
        Latest VIX close (float) or a realised-vol-based proxy, or None.
    """
    try:
        import yfinance as yf
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        if not vix_hist.empty:
            return float(vix_hist["Close"].iloc[-1])
    except Exception as e:
        log.warning("[regime_features] Live VIX fetch failed: %s", e)

    if fallback_realised_vol is not None:
        return float(fallback_realised_vol) * 100.0
    return None


def build_live_feature_vector(ohlcv: pd.DataFrame) -> Optional[dict]:
    """Build a single feature dict for the *latest* row of live SPY OHLCV data.

    Used at inference time by regime_model.RegimeClassifier. Returns None if
    there isn't enough history to compute a reliable feature vector (mirrors
    the heuristic's `len(close) < 50` guard in regime.py).
    """
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    if len(df) < 50:
        return None

    price_feats = compute_price_features(df)
    latest = price_feats.iloc[-1]

    if latest.isna().any():
        # Not enough history for a fully-populated vector (e.g. < 200 rows for SMA200).
        return None

    vix = fetch_vix_live(fallback_realised_vol=latest["realised_vol_20d"])
    if vix is None:
        return None

    return {
        "realised_vol_20d": float(latest["realised_vol_20d"]),
        "adx_14": float(latest["adx_14"]),
        "sma200_dist_pct": float(latest["sma200_dist_pct"]),
        "mom_20d": float(latest["mom_20d"]),
        "mom_5d": float(latest["mom_5d"]),
        "rsi_14": float(latest["rsi_14"]),
        "vix": float(vix),
    }
