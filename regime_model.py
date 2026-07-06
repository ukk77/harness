"""Learned regime classifier (Phase 4, A4).

Wraps an XGBoost multi-class classifier that predicts `regime.Regime` from
the feature vector defined in `regime_features.FEATURE_COLUMNS`.

Drop-in replacement path: `regime.detect_regime(..., mode="model")` loads a
`RegimeClassifier` and falls back to the heuristic if the model file is
missing/corrupt — this module never raises out to the caller.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .regime import Regime
from .regime_features import FEATURE_COLUMNS

log = logging.getLogger(__name__)

# Fixed encode/decode order — must match training. Persisted alongside the
# model file so a mismatched artifact is detectable rather than silently wrong.
_LABEL_ORDER = [
    Regime.BULL_TREND,
    Regime.BEAR_TREND,
    Regime.HIGH_VOL,
    Regime.RANGE_BOUND,
]


class RegimeClassifier:
    """Loads a trained XGBoost model and exposes predict / predict_proba.

    Usage:
        clf = RegimeClassifier(model_path="models/regime_xgb.json")
        if clf.load():
            regime = clf.predict(feature_dict)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._loaded = False

    def load(self) -> bool:
        """Load the model file. Returns True on success, False on any failure
        (missing file, corrupt file, version mismatch) — never raises."""
        if self._loaded:
            return True
        if self.model_path is None or not self.model_path.exists():
            log.info("[regime_model] No model file at %s — will use heuristic fallback.", self.model_path)
            return False
        try:
            import xgboost as xgb
            self._model = xgb.XGBClassifier()
            self._model.load_model(str(self.model_path))
            self._loaded = True
            return True
        except Exception as e:
            log.warning("[regime_model] Failed to load model at %s: %s — using heuristic fallback.", self.model_path, e)
            self._model = None
            self._loaded = False
            return False

    def _feature_row(self, features: dict):
        return [[features[col] for col in FEATURE_COLUMNS]]

    def predict_proba(self, features: dict) -> Optional[Dict[Regime, float]]:
        """Return a probability dict over all 4 regimes, or None if the model isn't loaded."""
        if not self._loaded or self._model is None:
            return None
        try:
            row = self._feature_row(features)
            proba = self._model.predict_proba(row)[0]
            return {label: float(p) for label, p in zip(_LABEL_ORDER, proba)}
        except Exception as e:
            log.warning("[regime_model] predict_proba failed: %s", e)
            return None

    def predict(self, features: dict) -> Optional[Regime]:
        """Return the argmax regime, or None if the model isn't loaded / prediction failed."""
        proba = self.predict_proba(features)
        if proba is None:
            return None
        return max(proba, key=proba.get)


def save_model(model, model_path: str, metadata: Optional[dict] = None) -> None:
    """Persist a trained XGBoost model + a sidecar metadata JSON (feature order,
    label order, training info) for reproducibility / debugging."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "label_order": [r.value for r in _LABEL_ORDER],
    }
    if metadata:
        meta.update(metadata)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
