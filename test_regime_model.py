"""
Standalone test for Phase 4 A4: regime_model.py.

Run directly (no pytest required):
    python test_regime_model.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from harness.regime import Regime
from harness.regime_features import FEATURE_COLUMNS
from harness.regime_model import RegimeClassifier, save_model, _LABEL_ORDER

PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def _dummy_feature_dict() -> dict:
    return {col: 0.1 for col in FEATURE_COLUMNS}


def test_missing_model_file():
    print("\n--- test_missing_model_file ---")
    clf = RegimeClassifier(model_path="C:/does/not/exist/regime_xgb.json")
    check("load() returns False for missing file", clf.load() is False)
    check("predict() returns None when not loaded", clf.predict(_dummy_feature_dict()) is None)
    check("predict_proba() returns None when not loaded", clf.predict_proba(_dummy_feature_dict()) is None)


def test_corrupt_model_file():
    print("\n--- test_corrupt_model_file ---")
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = Path(tmp) / "corrupt.json"
        bad_path.write_text("{ not valid xgboost model }")
        clf = RegimeClassifier(model_path=str(bad_path))
        check("load() returns False for corrupt file", clf.load() is False)
        check("predict() returns None for corrupt file", clf.predict(_dummy_feature_dict()) is None)


def test_no_model_path():
    print("\n--- test_no_model_path ---")
    clf = RegimeClassifier(model_path=None)
    check("load() returns False when model_path=None", clf.load() is False)


def test_save_and_load_roundtrip():
    print("\n--- test_save_and_load_roundtrip ---")
    import xgboost as xgb

    label_values = [r.value for r in _LABEL_ORDER]
    # Tiny synthetic training set — one clearly-separable sample per class.
    rng = np.random.default_rng(0)
    X = []
    y = []
    for i, _ in enumerate(label_values):
        for _ in range(20):
            row = rng.normal(loc=i * 5.0, scale=0.1, size=len(FEATURE_COLUMNS))
            X.append(row)
            y.append(i)
    X = np.array(X)
    y = np.array(y)

    model = xgb.XGBClassifier(n_estimators=10, max_depth=2, objective="multi:softprob", num_class=len(label_values))
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = str(Path(tmp) / "test_model.json")
        save_model(model, model_path, metadata={"test": True})

        meta_path = Path(model_path).with_suffix(".meta.json")
        check("save_model writes model file", Path(model_path).exists())
        check("save_model writes sidecar metadata file", meta_path.exists())

        clf = RegimeClassifier(model_path=model_path)
        check("load() returns True for valid saved model", clf.load() is True)

        # Predict on a sample near class 0's cluster center.
        features = {col: 0.0 for col in FEATURE_COLUMNS}
        proba = clf.predict_proba(features)
        check("predict_proba returns a dict over all 4 regimes", proba is not None and set(proba.keys()) == set(_LABEL_ORDER))
        if proba:
            check("predict_proba values sum to ~1.0", abs(sum(proba.values()) - 1.0) < 1e-4, f"got {sum(proba.values())}")

        predicted = clf.predict(features)
        check("predict() returns a valid Regime member", predicted in _LABEL_ORDER, f"got {predicted}")
        check("predict() on class-0 cluster center returns class 0's label", predicted == _LABEL_ORDER[0], f"got {predicted}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 A4 — regime_model.py Tests")
    print("=" * 60)

    test_missing_model_file()
    test_corrupt_model_file()
    test_no_model_path()
    test_save_and_load_roundtrip()

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
