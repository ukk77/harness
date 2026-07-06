"""Train the learned regime classifier (Phase 4, A4).

Data sources:
  - Labels + VIX: harness/data/archive_extracted/stock_market_regimes_2000_2026.csv
    (Kaggle: mafaqbhatti/stock-market-regimes-20002026), filtered to ticker == '^GSPC'.
    Regime labels are sanity-checked against known market history in the Phase 4
    Step-2 plan doc before being trusted for training.
  - Price technicals (ADX, SMA-distance, momentum, RSI, realised vol): computed from
    REAL SPY OHLC fetched via yfinance for the same date range — NOT from the Kaggle
    dataset's close/returns columns — so training features are computed by the exact
    same code path (`regime_features.compute_price_features`) used at live inference
    time against `trading_core.market_data.fetch_ohlcv("SPY")`.

Label mapping (5 Kaggle labels -> 4 Regime enum values; documented decision,
see plans/phase4_harness.md):
    Bull      -> BULL_TREND
    Bear      -> BEAR_TREND
    Sideways  -> RANGE_BOUND
    Crisis    -> HIGH_VOL
    High-volatility -> dropped (negligible: 0 rows for ^GSPC in the dataset)

Split: chronological. Last 12 months = holdout (no shuffling — time series).

Usage:
    python -m harness.scripts.train_regime
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_regime")

_HARNESS_DIR = Path(__file__).resolve().parents[1]
_TRADING_ROOT = _HARNESS_DIR.parent
if str(_TRADING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_ROOT))

from harness.regime import Regime  # noqa: E402
from harness.regime_features import compute_price_features, FEATURE_COLUMNS  # noqa: E402
from harness.regime_model import save_model, _LABEL_ORDER  # noqa: E402

DATASET_CSV = _HARNESS_DIR / "data" / "archive_extracted" / "stock_market_regimes_2000_2026.csv"
MODEL_OUT = _HARNESS_DIR.parent / "models" / "regime_xgb.json"

_LABEL_MAP = {
    "Bull": Regime.BULL_TREND,
    "Bear": Regime.BEAR_TREND,
    "Sideways": Regime.RANGE_BOUND,
    "Crisis": Regime.HIGH_VOL,
    # "High-volatility" intentionally dropped — see module docstring.
}

HOLDOUT_MONTHS = 12


def load_labels_and_vix() -> pd.DataFrame:
    """Load ^GSPC rows from the Kaggle dataset: date, mapped regime, confidence, vix."""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {DATASET_CSV}. "
            "Expected the extracted Kaggle CSV (see plans/phase4_harness.md)."
        )
    df = pd.read_csv(DATASET_CSV, usecols=[
        "date", "ticker", "regime_label", "regime_confidence", "vix",
    ])
    df = df[df["ticker"] == "^GSPC"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    df = df[df["regime_label"].isin(_LABEL_MAP.keys())].copy()
    df["regime"] = df["regime_label"].map(_LABEL_MAP)
    df = df.set_index("date").sort_index()
    return df[["regime", "regime_confidence", "vix"]]


def fetch_spy_ohlc_history(start: str, end: str) -> pd.DataFrame:
    """Fetch real daily SPY OHLC via yfinance for the training date range."""
    import yfinance as yf
    df = yf.download("SPY", start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no SPY data — check network connectivity.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).normalize()
    return df[["open", "high", "low", "close", "volume"]]


def build_training_frame() -> pd.DataFrame:
    labels = load_labels_and_vix()
    start = labels.index.min().strftime("%Y-%m-%d")
    end = (labels.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("Fetching SPY OHLC history %s -> %s via yfinance...", start, end)
    spy = fetch_spy_ohlc_history(start, end)
    log.info("Fetched %d SPY daily bars.", len(spy))

    price_feats = compute_price_features(spy)

    merged = price_feats.join(labels, how="inner")
    merged["vix"] = merged["vix"].astype(float)

    before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLUMNS + ["regime"])
    log.info("Merged frame: %d rows -> %d after dropping NaN (insufficient history, e.g. <200d for SMA200).",
              before, len(merged))
    return merged


def chronological_split(df: pd.DataFrame, holdout_months: int = HOLDOUT_MONTHS):
    cutoff = df.index.max() - pd.DateOffset(months=holdout_months)
    train = df[df.index <= cutoff]
    holdout = df[df.index > cutoff]
    log.info("Train: %d rows (%s -> %s)", len(train), train.index.min().date(), train.index.max().date())
    log.info("Holdout: %d rows (%s -> %s)", len(holdout), holdout.index.min().date(), holdout.index.max().date())
    return train, holdout


def train_model(train_df: pd.DataFrame):
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    label_order_values = [r.value for r in _LABEL_ORDER]
    encoder = LabelEncoder()
    encoder.fit(label_order_values)

    X = train_df[FEATURE_COLUMNS].values
    y = encoder.transform(train_df["regime"].astype(str).values)
    sample_weight = train_df["regime_confidence"].clip(lower=0.05).values

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(label_order_values),
        eval_metric="mlogloss",
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, encoder


def evaluate(model, encoder, holdout_df: pd.DataFrame) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X = holdout_df[FEATURE_COLUMNS].values
    y_true = encoder.transform(holdout_df["regime"].astype(str).values)
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=encoder.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    log.info("Holdout accuracy: %.4f", acc)
    log.info("Classification report:\n%s", classification_report(y_true, y_pred, target_names=encoder.classes_, zero_division=0))
    log.info("Confusion matrix (rows=true, cols=pred, order=%s):\n%s", list(encoder.classes_), cm)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "label_order": list(encoder.classes_)}


def main():
    df = build_training_frame()
    train_df, holdout_df = chronological_split(df)

    if len(train_df) < 200 or len(holdout_df) < 20:
        raise RuntimeError(
            f"Not enough data to train reliably (train={len(train_df)}, holdout={len(holdout_df)})."
        )

    model, encoder = train_model(train_df)
    metrics = evaluate(model, encoder, holdout_df)

    metadata = {
        "train_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "train_range": [str(train_df.index.min().date()), str(train_df.index.max().date())],
        "holdout_range": [str(holdout_df.index.min().date()), str(holdout_df.index.max().date())],
        "holdout_accuracy": metrics["accuracy"],
        "source_dataset": "kaggle:mafaqbhatti/stock-market-regimes-20002026 (^GSPC rows) + yfinance SPY OHLC",
    }
    save_model(model, str(MODEL_OUT), metadata=metadata)
    log.info("Model saved to %s (metadata: %s)", MODEL_OUT, MODEL_OUT.with_suffix(".meta.json"))


if __name__ == "__main__":
    main()
