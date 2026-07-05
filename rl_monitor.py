"""RL model performance monitor and auto-retrain trigger.

Tracks per-ticker Sharpe ratios from backtest results and triggers retraining
when the rolling mean Sharpe falls below the configured ``rl_min_sharpe``
threshold.  A persistent history file keeps the last N backtest mean Sharpes
so the check is trend-aware, not just a single snapshot.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harness.config import HarnessConfig, get_config

log = logging.getLogger(__name__)


DEFAULT_BACKTEST_WINDOW = 3
DEFAULT_EPISODE_WINDOW = 5


@dataclass
class RLModelStatus:
    ticker: str
    model_exists: bool
    backtest_exists: bool
    latest_episode_sharpe: Optional[float] = None
    rolling_mean_sharpe: Optional[float] = None
    history_count: int = 0
    degraded: bool = False
    retrained: bool = False
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)


def _results_dir(cfg: HarnessConfig) -> Path:
    return Path(cfg.trading_root) / "results"


def _history_path(cfg: HarnessConfig) -> Path:
    return _results_dir(cfg) / "rl_sharpe_history.json"


def _backtest_path(cfg: HarnessConfig, ticker: str) -> Path:
    return _results_dir(cfg) / f"{ticker}_backtest.json"


def _model_path(cfg: HarnessConfig, ticker: str) -> Path:
    return Path(cfg.models_dir) / f"{ticker}_ppo.zip"


def _load_backtest_episode_sharpes(
    cfg: HarnessConfig, ticker: str
) -> List[Tuple[int, float]]:
    """Return [(episode_id, sharpe_ratio), ...] from the latest backtest file."""
    path = _backtest_path(cfg, ticker)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        return [
            (int(ep.get("episode_id", idx)), float(ep.get("sharpe_ratio", 0.0)))
            for idx, ep in enumerate(episodes)
            if ep.get("sharpe_ratio") is not None
        ]
    except Exception as exc:
        log.warning("Could not read backtest Sharpe for %s: %s", ticker, exc)
        return []


def _load_backtest_mean_sharpe(cfg: HarnessConfig, ticker: str) -> Optional[float]:
    """Return the aggregate mean_sharpe from the latest backtest file, if present."""
    path = _backtest_path(cfg, ticker)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("episode_metrics", {}).get("mean_sharpe")
    except Exception as exc:
        log.warning("Could not read mean_sharpe for %s: %s", ticker, exc)
        return None


def load_sharpe_history(cfg: HarnessConfig) -> Dict[str, List[Dict]]:
    """Load persisted Sharpe history: {ticker: [{at, mean_sharpe}, ...]}."""
    path = _history_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load RL Sharpe history: %s", exc)
        return {}


def save_sharpe_history(cfg: HarnessConfig, history: Dict[str, List[Dict]]) -> None:
    """Persist the Sharpe history JSON."""
    path = _history_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")


def record_backtest_sharpe(
    cfg: HarnessConfig,
    ticker: str,
    mean_sharpe: Optional[float] = None,
    timestamp: Optional[str] = None,
) -> None:
    """Append a backtest mean-Sharpe snapshot to the rolling history.

    If ``mean_sharpe`` is not provided, it is read from the existing backtest
    JSON for the ticker.  This is safe to call after every backtest run.
    """
    if mean_sharpe is None:
        mean_sharpe = _load_backtest_mean_sharpe(cfg, ticker)
    if mean_sharpe is None:
        return

    history = load_sharpe_history(cfg)
    entry = {
        "at": timestamp or (datetime.utcnow().isoformat() + "Z"),
        "mean_sharpe": float(mean_sharpe),
    }
    history.setdefault(ticker, []).append(entry)
    save_sharpe_history(cfg, history)


def _compute_rolling_sharpe(
    backtest_sharpes: List[Tuple[int, float]],
    history: List[Dict],
    backtest_window: int,
    episode_window: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (latest_episode_sharpe, rolling_mean_sharpe).

    The rolling mean prefers the mean of the last ``backtest_window`` backtest
    snapshots in the history.  If history is empty, it falls back to the mean
    of the last ``episode_window`` episodes from the current backtest file.
    """
    latest_episode_sharpe = backtest_sharpes[-1][1] if backtest_sharpes else None

    recent_backtests = [
        float(entry["mean_sharpe"])
        for entry in history[-backtest_window:]
        if "mean_sharpe" in entry
    ]
    if recent_backtests:
        rolling_mean = sum(recent_backtests) / len(recent_backtests)
    else:
        recent_episodes = [s for _, s in backtest_sharpes[-episode_window:]]
        rolling_mean = (
            sum(recent_episodes) / len(recent_episodes) if recent_episodes else None
        )

    return latest_episode_sharpe, rolling_mean


def _check_ticker(
    cfg: HarnessConfig,
    ticker: str,
    min_sharpe: float,
    backtest_window: int,
    episode_window: int,
) -> RLModelStatus:
    """Evaluate a single RL ticker and return its status."""
    status = RLModelStatus(ticker=ticker, model_exists=False, backtest_exists=False)
    status.model_exists = _model_path(cfg, ticker).exists()
    status.backtest_exists = _backtest_path(cfg, ticker).exists()

    if not status.model_exists:
        status.degraded = True
        status.details["reason"] = "no trained model"
        return status

    if not status.backtest_exists:
        # Model exists but has never been backtested -> needs a backtest, not retrain
        status.degraded = False
        status.details["reason"] = "model exists but no backtest results"
        return status

    try:
        episode_sharpes = _load_backtest_episode_sharpes(cfg, ticker)
        history = load_sharpe_history(cfg).get(ticker, [])
        status.history_count = len(history)

        latest_episode_sharpe, rolling_mean = _compute_rolling_sharpe(
            episode_sharpes, history, backtest_window, episode_window
        )
        status.latest_episode_sharpe = latest_episode_sharpe
        status.rolling_mean_sharpe = rolling_mean

        status.details["backtest_episodes"] = len(episode_sharpes)

        # Degraded if either current episode or rolling mean is below threshold
        below_current = (
            latest_episode_sharpe is not None and latest_episode_sharpe < min_sharpe
        )
        below_rolling = (
            rolling_mean is not None and rolling_mean < min_sharpe
        )
        status.degraded = below_current or below_rolling

        reasons = []
        if below_current:
            reasons.append(
                f"latest episode Sharpe {latest_episode_sharpe:.2f} < {min_sharpe:.2f}"
            )
        if below_rolling:
            reasons.append(
                f"rolling Sharpe {rolling_mean:.2f} < {min_sharpe:.2f}"
            )
        status.details["reason"] = (
            "; ".join(reasons) if reasons else "Sharpe above threshold"
        )
    except Exception as exc:
        status.error = str(exc)
        status.degraded = False
        status.details["reason"] = f"check failed: {exc}"

    return status


def run_retrain_check(
    cfg: Optional[HarnessConfig] = None,
    min_sharpe: Optional[float] = None,
    backtest_window: int = DEFAULT_BACKTEST_WINDOW,
    episode_window: int = DEFAULT_EPISODE_WINDOW,
    auto_retrain: bool = False,
    retrain_timesteps: int = 50000,
    tickers: Optional[List[str]] = None,
) -> List[RLModelStatus]:
    """Check all (or selected) RL models and optionally retrain degraded ones.

    Args:
        cfg: Harness config (loaded from env if None).
        min_sharpe: Threshold below which a model is considered degraded.
            Defaults to ``cfg.rl_min_sharpe``.
        backtest_window: Number of historical backtests to include in the rolling mean.
        episode_window: Fallback window of episodes from the latest backtest.
        auto_retrain: If True, call ``rl_strategy.agent.train.train_single_ticker``
            for every degraded ticker.
        retrain_timesteps: Timesteps to use when auto-retraining.
        tickers: Optional list of tickers to check. Defaults to ``cfg._rl_tickers``.

    Returns:
        List of ``RLModelStatus`` objects, one per ticker checked.
    """
    cfg = cfg or get_config()
    min_sharpe = min_sharpe if min_sharpe is not None else cfg.rl_min_sharpe
    tickers = tickers or cfg._rl_tickers or []

    results: List[RLModelStatus] = []
    for ticker in tickers:
        status = _check_ticker(
            cfg, ticker, min_sharpe, backtest_window, episode_window
        )
        results.append(status)

    if auto_retrain:
        from rl_strategy.agent.train import train_single_ticker

        for status in results:
            if not status.degraded:
                continue
            log.info("Auto-retraining %s (timesteps=%d)", status.ticker, retrain_timesteps)
            try:
                train_single_ticker(status.ticker, timesteps=retrain_timesteps)
                status.retrained = True
                # Record the evaluation mean return isn't a Sharpe; schedule a backtest
                # run separately to update the rolling Sharpe history.
            except Exception as exc:
                log.error("Auto-retrain failed for %s: %s", status.ticker, exc)
                status.error = f"retrain failed: {exc}"

    # Persist today's backtest mean_sharpes into history so the next check has
    # more data points.  Do this even for non-degraded models.
    for status in results:
        if status.backtest_exists and not status.error:
            record_backtest_sharpe(cfg, status.ticker)

    return results


def save_retrain_report(
    cfg: HarnessConfig,
    results: List[RLModelStatus],
    min_sharpe: float,
    auto_retrain: bool,
) -> Path:
    """Write a JSON report of the retrain check to ``results/retrain_check_YYYYMMDD.json``."""
    results_dir = _results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"retrain_check_{timestamp}.json"

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "min_sharpe": min_sharpe,
        "auto_retrain": auto_retrain,
        "degraded_count": sum(1 for r in results if r.degraded),
        "retrained_count": sum(1 for r in results if r.retrained),
        "models": [
            {
                "ticker": r.ticker,
                "model_exists": r.model_exists,
                "backtest_exists": r.backtest_exists,
                "latest_episode_sharpe": r.latest_episode_sharpe,
                "rolling_mean_sharpe": r.rolling_mean_sharpe,
                "history_count": r.history_count,
                "degraded": r.degraded,
                "retrained": r.retrained,
                "error": r.error,
                "details": r.details,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Retrain check report saved: %s", out_path)
    return out_path


def print_retrain_report(results: List[RLModelStatus], min_sharpe: float) -> None:
    """Print a formatted table of the retrain check results."""
    print(f"\n{'='*72}")
    print(f"RL AUTO-RETRAIN CHECK  (threshold: Sharpe >= {min_sharpe:.2f})")
    print(f"{'='*72}")
    print(
        f"  {'Ticker':<8} {'Status':<10} {'Rolling':>8} {'Latest':>8} "
        f"{'Hist':>5} {'Reason / Detail'}"
    )
    print("  " + "-" * 68)

    for r in results:
        if r.error:
            status = "ERROR"
            detail = r.error[:50]
        elif r.degraded:
            status = "DEGRADED" if not r.retrained else "RETRAINED"
            detail = r.details.get("reason", "")
        elif not r.model_exists:
            status = "NO MODEL"
            detail = "train required"
        elif not r.backtest_exists:
            status = "NO BACKTEST"
            detail = "run backtest first"
        else:
            status = "OK"
            detail = r.details.get("reason", "")

        rolling_str = f"{r.rolling_mean_sharpe:>7.2f}" if r.rolling_mean_sharpe is not None else "   N/A"
        latest_str = f"{r.latest_episode_sharpe:>7.2f}" if r.latest_episode_sharpe is not None else "   N/A"
        print(
            f"  {r.ticker:<8} {status:<10} {rolling_str:>8} {latest_str:>8} "
            f"{r.history_count:>5} {detail}"
        )

    degraded = sum(1 for r in results if r.degraded)
    retrained = sum(1 for r in results if r.retrained)
    print("  " + "-" * 68)
    print(f"  Summary: {degraded} degraded, {retrained} auto-retrained, {len(results)} checked\n")
