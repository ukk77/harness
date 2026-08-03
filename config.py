"""Harness configuration.

Ticker ownership:
  MR  — uses MeanReversionConfig().tickers  (range-bound stocks/ETFs)
  TF  — uses TrendFollowingConfig().tickers  (trending stocks/ETFs)
  VB  — uses VolatilityBreakoutConfig().tickers  (high-beta breakout stocks)
  RL  — uses only tickers with a trained model AND backtest mean_sharpe >= rl_min_sharpe

  HarnessConfig.tickers = union of all four (used for data collection jobs).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

_TRADING_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _TRADING_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
    
log = logging.getLogger(__name__)


def _load_rl_tickers(models_dir: str, results_dir: str, min_sharpe: float = 0.5) -> List[str]:
    """Return RL tickers that have a trained model AND backtest mean_sharpe >= min_sharpe."""
    models = Path(models_dir)
    results = Path(results_dir)
    good: List[str] = []
    for model_file in sorted(models.glob("*_ppo.zip")):
        ticker = model_file.stem.replace("_ppo", "")
        result_file = results / f"{ticker}_backtest.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text())
            episodes = data.get("episodes", [])
            if not episodes:
                continue
            mean_sharpe = sum(e.get("sharpe_ratio", 0.0) for e in episodes) / len(episodes)
            if mean_sharpe >= min_sharpe:
                good.append(ticker)
        except Exception:
            pass
    return good


@dataclass
class HarnessConfig:
    """Master configuration for the trading harness."""

    # ── Capital ──────────────────────────────────────────────────────────────
    total_capital: float = 100_000.0
    max_position_pct: float = 0.10      # Max 10% of capital per ticker
    max_strategy_pct: float = 0.50      # Max 50% of capital to one strategy

    # ── Strategies ───────────────────────────────────────────────────────────
    strategies: List[str] = field(default_factory=lambda: ["rl", "mr", "tf", "vb"])
    strategy_weights: Optional[Dict[str, float]] = None   # None = Sharpe-weighted auto

    # ── Reconciliation ───────────────────────────────────────────────────────
    reconciliation_mode: str = "confidence_weighted"
    min_confidence_to_act: float = 0.55

    # ── Per-strategy tickers (each strategy trades its own universe) ─────────
    # MR / TF / VB read from their own configs at runtime (do not override here).
    # RL tickers are filtered by backtest quality.
    rl_min_sharpe: float = 0.5          # Minimum mean episode Sharpe for RL to trade a ticker

    # ── Union ticker list (data collection, health checks) ───────────────────
    # Populated by get_config() by merging all per-strategy lists.
    tickers: List[str] = field(default_factory=list)

    # ── Execution ────────────────────────────────────────────────────────────
    execution_mode: str = "paper"       # "paper" | "live"
    paper_db_path: str = field(default_factory=lambda: os.environ.get(
        "HARNESS_DB_PATH", str(_TRADING_ROOT / "harness" / "harness_trades.db")))

    # ── Services — overridable via env vars for Docker/cloud ─────────────────
    sentiment_api_url: str = field(default_factory=lambda: os.environ.get(
        "SENTIMENT_API_URL", "http://localhost:8000"))
    risk_api_url: str = field(default_factory=lambda: os.environ.get(
        "RISK_API_URL", "http://localhost:8100"))

    # ── Paths — all overridable via env vars for Docker/cloud ─────────────────
    trading_root: str = field(default_factory=lambda: os.environ.get(
        "TRADING_ROOT", str(_TRADING_ROOT)))
    models_dir: str = field(default_factory=lambda: os.environ.get(
        "MODELS_DIR", str(_TRADING_ROOT / "models")))
    market_data_dir: str = field(default_factory=lambda: os.environ.get(
        "MARKET_DATA_DIR", str(_TRADING_ROOT / "market_data" / "hourly")))
    results_dir: str = field(default_factory=lambda: os.environ.get(
        "RESULTS_DIR", str(_TRADING_ROOT / "results")))
    logs_dir: str = field(default_factory=lambda: os.environ.get(
        "LOGS_DIR", str(_TRADING_ROOT / "logs")))

    # ── Risk controls ────────────────────────────────────────────────────────
    daily_loss_limit_pct: float = 0.02
    trade_cooldown_hours: int = 4
    circuit_breaker_drawdown_pct: float = 0.05  # Halt trading if portfolio DD > 5% today
    circuit_breaker_extreme_bearish: bool = True # Halt trading if regime is bear + high vol
    # Annualized realized-vol threshold (20d) that, combined with a bearish
    # SPY trend (below 200-SMA + negative 20d momentum), trips the circuit
    # breaker above. Below the heuristic's own HIGH_VOL classification cutoff
    # (~0.28-0.30) so genuinely elevated-but-not-extreme bear conditions are
    # still caught, without firing on every ordinary bear trend.
    circuit_breaker_bear_vol_threshold: float = 0.22

    # PaperExecutor mirrors every fill to the real Alpaca paper account by
    # default. Set False for hermetic/offline runs (e.g. tests) that should
    # not make network calls to an external service (§ 10 I2).
    alpaca_mirror_enabled: bool = True

    # ── DCA-into-losers guard (§ 10 S2) ──────────────────────────────────────
    # Block adding to an existing position when it is underwater beyond this
    # threshold, UNLESS the new signal's confidence exceeds the position's
    # running average entry confidence (i.e. genuinely higher conviction).
    dca_loss_guard_pct: float = 0.05

    # ── Regime detection (Phase 4, A4) ──────────────────────────────────────
    regime_mode: str = "model"            # "model" (learned XGBoost classifier) | "heuristic" (fallback)
    regime_soft_blend: bool = False       # If True, allocator blends _REGIME_MULTIPLIERS by predict_proba
    regime_model_path: str = field(default_factory=lambda: os.environ.get(
        "REGIME_MODEL_PATH", str(_TRADING_ROOT / "models" / "regime_xgb.json")))

    # ── LLM Run Summary (Phase 4, A2) ────────────────────────────────────────
    summary_mode: str = "rag"            # "none" (default) | "llm" | "rag"
    llm_provider: str = field(default_factory=lambda: os.environ.get(
        "RAG_LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: os.environ.get(
        "LLM_MODEL", "llama3.2:3b"))
    llm_base_url: str = field(default_factory=lambda: os.environ.get(
        "LLM_BASE_URL", "http://localhost:11434"))

    # ── RAG Service (Phase 4, A9) ────────────────────────────────────────────
    rag_service_url: str = field(default_factory=lambda: os.environ.get(
        "RAG_SERVICE_URL", "http://localhost:8200"))
    rag_enrichment_enabled: bool = False   # signal-enrichment hook (gate; default off)
    rag_ingest_on_collect: bool = True     # trigger POST /api/ingest after data_collection

    # ── Schedule ─────────────────────────────────────────────────────────────
    # data_collection runs: 08:00, 11:00, 14:00, 17:00 ET
    data_collection_times: List[str] = field(
        default_factory=lambda: ["08:00", "11:00", "14:00", "17:00"]
    )
    # signal_generation runs: 08:30 – 17:30 ET, every hour
    signal_generation_start: str = "08:30"
    signal_generation_end: str = "17:30"


_config: Optional[HarnessConfig] = None


def get_config() -> HarnessConfig:
    """Return (and lazily build) the singleton HarnessConfig."""
    global _config
    if _config is not None:
        return _config

    cfg = HarnessConfig()

    # ── Build per-strategy ticker lists ──────────────────────────────────────
    try:
        from mean_reversion.config import MeanReversionConfig
        mr_tickers = list(MeanReversionConfig().tickers)
    except Exception:
        mr_tickers = []

    try:
        from trend_following.config import TrendFollowingConfig
        tf_tickers = list(TrendFollowingConfig().tickers)
    except Exception:
        tf_tickers = []

    try:
        from volatility_breakout.config import VolatilityBreakoutConfig
        vb_tickers = list(VolatilityBreakoutConfig().tickers)
    except Exception:
        vb_tickers = []

    rl_tickers = _load_rl_tickers(cfg.models_dir, cfg.results_dir, cfg.rl_min_sharpe)
    if not rl_tickers:
        # Fallback: any ticker with a trained model
        rl_tickers = [
            f.stem.replace("_ppo", "")
            for f in sorted(Path(cfg.models_dir).glob("*_ppo.zip"))
        ]
        log.warning(
            "No RL tickers met Sharpe threshold %.1f — using all %d trained models",
            cfg.rl_min_sharpe, len(rl_tickers),
        )

    # Store per-strategy lists on config for use by adapters / data collection
    cfg._mr_tickers = mr_tickers
    cfg._tf_tickers = tf_tickers
    cfg._vb_tickers = vb_tickers
    cfg._rl_tickers = rl_tickers

    # Union = all unique tickers across every strategy
    cfg.tickers = sorted(set(mr_tickers + tf_tickers + vb_tickers + rl_tickers))

    _config = cfg
    return _config
