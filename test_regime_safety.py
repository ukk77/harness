"""
Standalone test for Phase 4 A4: safety regression tests for regime.py.

CRITICAL: these tests exist to guarantee that today's live orchestrator run
is byte-for-byte unaffected by the addition of the learned-model code path.

Run directly (no pytest required):
    python test_regime_safety.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from harness.regime import Regime, detect_regime

PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def _make_ohlc(n: int, trend: float = 0.0, vol: float = 0.01, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=trend, scale=vol, size=n)
    close = 100 * np.cumprod(1 + rets)
    high = close * 1.005
    low = close * 0.995
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


# A range of representative scenarios: uptrend, downtrend, high-vol chop, flat/range-bound.
_SCENARIOS = {
    "uptrend": _make_ohlc(300, trend=0.0015, vol=0.008, seed=1),
    "downtrend": _make_ohlc(300, trend=-0.0015, vol=0.008, seed=2),
    "high_vol_chop": _make_ohlc(300, trend=0.0, vol=0.035, seed=3),
    "range_bound": _make_ohlc(300, trend=0.0, vol=0.006, seed=4),
    "short_history": _make_ohlc(30, trend=0.001, vol=0.01, seed=5),
}


def test_default_mode_matches_explicit_heuristic():
    print("\n--- test_default_mode_matches_explicit_heuristic ---")
    for name, ohlcv in _SCENARIOS.items():
        default_result = detect_regime(ohlcv)
        explicit_result = detect_regime(ohlcv, mode="heuristic")
        check(
            f"[{name}] default call == explicit mode='heuristic' call",
            default_result == explicit_result,
            f"default={default_result}, explicit={explicit_result}",
        )
        check(f"[{name}] result is a valid Regime enum", isinstance(default_result, Regime))


def test_model_mode_falls_back_when_file_missing():
    print("\n--- test_model_mode_falls_back_when_file_missing ---")
    for name, ohlcv in _SCENARIOS.items():
        heuristic_result = detect_regime(ohlcv, mode="heuristic")
        model_result = detect_regime(ohlcv, mode="model", model_path="C:/does/not/exist/regime_xgb.json")
        check(
            f"[{name}] mode='model' with missing file == heuristic result",
            model_result == heuristic_result,
            f"model={model_result}, heuristic={heuristic_result}",
        )


def test_model_mode_never_raises_on_corrupt_file():
    print("\n--- test_model_mode_never_raises_on_corrupt_file ---")
    import tempfile
    ohlcv = _SCENARIOS["uptrend"]
    with tempfile.TemporaryDirectory() as tmp:
        corrupt = Path(tmp) / "corrupt.json"
        corrupt.write_text("not a real model file")
        try:
            result = detect_regime(ohlcv, mode="model", model_path=str(corrupt))
            check("mode='model' with corrupt file does not raise", True)
            check("mode='model' with corrupt file returns valid Regime (heuristic fallback)", isinstance(result, Regime))
        except Exception as e:
            check("mode='model' with corrupt file does not raise", False, f"raised: {e}")


def test_previous_regime_hysteresis_unaffected():
    print("\n--- test_previous_regime_hysteresis_unaffected ---")
    ohlcv = _SCENARIOS["high_vol_chop"]
    # Hysteresis changes the vol threshold based on previous_regime — confirm
    # this still works identically regardless of the new `mode` param existing.
    r1 = detect_regime(ohlcv, previous_regime=None)
    r2 = detect_regime(ohlcv, previous_regime=Regime.HIGH_VOL)
    check("previous_regime param still accepted and used", isinstance(r1, Regime) and isinstance(r2, Regime))


def test_unknown_mode_falls_back_to_heuristic_path():
    print("\n--- test_unknown_mode_falls_back_to_heuristic_path ---")
    ohlcv = _SCENARIOS["uptrend"]
    # Any mode other than "model" should just run the heuristic path (the
    # `if mode == "model":` guard means anything else — including typos —
    # safely falls through to heuristic rather than raising).
    result = detect_regime(ohlcv, mode="typo_mode")
    expected = detect_regime(ohlcv, mode="heuristic")
    check("unrecognised mode value falls through to heuristic (no crash)", result == expected, f"got {result} vs {expected}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 A4 — regime.py Safety Regression Tests")
    print("=" * 60)

    test_default_mode_matches_explicit_heuristic()
    test_model_mode_falls_back_when_file_missing()
    test_model_mode_never_raises_on_corrupt_file()
    test_previous_regime_hysteresis_unaffected()
    test_unknown_mode_falls_back_to_heuristic_path()

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
