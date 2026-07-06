"""
Standalone test for Phase 4 A4: regime_features.py.

Run directly (no pytest required):
    python test_regime_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from harness.regime_features import (
    compute_price_features,
    fetch_vix_live,
    build_live_feature_vector,
    FEATURE_COLUMNS,
)

PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def _make_ohlc(n: int, trend: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily OHLC. trend=0 -> flat/random walk; trend>0 -> uptrend."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=trend, scale=0.01, size=n)
    close = 100 * np.cumprod(1 + rets)
    high = close * 1.005
    low = close * 0.995
    open_ = close * 1.0
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_compute_price_features_shape():
    print("\n--- test_compute_price_features_shape ---")
    df = _make_ohlc(300)
    feats = compute_price_features(df)
    expected_cols = {"realised_vol_20d", "adx_14", "sma200_dist_pct", "mom_20d", "mom_5d", "rsi_14"}
    check("compute_price_features returns expected columns", set(feats.columns) == expected_cols, f"got {set(feats.columns)}")
    check("compute_price_features preserves row count", len(feats) == len(df))
    check("last row is fully populated (enough history)", not feats.iloc[-1].isna().any())
    check("early rows are NaN (insufficient history for SMA200)", feats.iloc[10].isna().any())


def test_flat_price_low_adx_low_vol():
    print("\n--- test_flat_price_low_adx_low_vol ---")
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    flat = pd.Series(100.0, index=idx)
    df = pd.DataFrame({"open": flat, "high": flat * 1.0001, "low": flat * 0.9999, "close": flat}, index=idx)
    feats = compute_price_features(df)
    last = feats.iloc[-1]
    check("flat price -> near-zero realised vol", abs(last["realised_vol_20d"]) < 1e-6, f"got {last['realised_vol_20d']}")
    check("flat price -> near-zero ADX", last["adx_14"] < 1.0, f"got {last['adx_14']}")
    check("flat price -> near-zero momentum", abs(last["mom_20d"]) < 1e-6, f"got {last['mom_20d']}")


def test_uptrend_positive_momentum():
    print("\n--- test_uptrend_positive_momentum ---")
    df = _make_ohlc(300, trend=0.003)
    feats = compute_price_features(df)
    last = feats.iloc[-1]
    check("uptrend -> positive 20d momentum", last["mom_20d"] > 0, f"got {last['mom_20d']}")
    check("uptrend -> positive SMA200 distance", last["sma200_dist_pct"] > 0, f"got {last['sma200_dist_pct']}")


def test_fetch_vix_live_fallback_on_failure(monkeypatch_module=None):
    print("\n--- test_fetch_vix_live_fallback_on_failure ---")
    import harness.regime_features as rf

    class _BrokenTicker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            raise RuntimeError("network down")

    import yfinance
    original = yfinance.Ticker
    yfinance.Ticker = _BrokenTicker
    try:
        result = rf.fetch_vix_live(fallback_realised_vol=0.22)
        check("VIX fetch failure falls back to realised_vol*100", result == 22.0, f"got {result}")

        result_none = rf.fetch_vix_live(fallback_realised_vol=None)
        check("VIX fetch failure with no fallback returns None", result_none is None, f"got {result_none}")
    finally:
        yfinance.Ticker = original


def test_build_live_feature_vector_insufficient_history():
    print("\n--- test_build_live_feature_vector_insufficient_history ---")
    df = _make_ohlc(30)  # < 50 rows
    result = build_live_feature_vector(df)
    check("insufficient history (<50 rows) returns None", result is None, f"got {result}")


def test_build_live_feature_vector_valid_shape():
    print("\n--- test_build_live_feature_vector_valid_shape ---")
    import harness.regime_features as rf

    # Stub out the live VIX fetch so this test has no network dependency.
    original_fetch = rf.fetch_vix_live
    rf.fetch_vix_live = lambda fallback_realised_vol=None: 18.5
    try:
        df = _make_ohlc(300, trend=0.001)
        result = build_live_feature_vector(df)
        check("valid history returns a feature dict", result is not None)
        if result is not None:
            check("feature dict has exactly FEATURE_COLUMNS keys", set(result.keys()) == set(FEATURE_COLUMNS), f"got {set(result.keys())}")
            check("feature dict vix matches stub", result["vix"] == 18.5, f"got {result['vix']}")
            check("all feature values are finite floats", all(np.isfinite(v) for v in result.values()))
    finally:
        rf.fetch_vix_live = original_fetch


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 A4 — regime_features.py Tests")
    print("=" * 60)

    test_compute_price_features_shape()
    test_flat_price_low_adx_low_vol()
    test_uptrend_positive_momentum()
    test_fetch_vix_live_fallback_on_failure()
    test_build_live_feature_vector_insufficient_history()
    test_build_live_feature_vector_valid_shape()

    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 60)

    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
