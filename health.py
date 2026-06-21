"""Pre-flight health checker for the trading harness.

Validates:
  - Sentiment Analysis API reachable
  - Risk Calculator API reachable
  - RL model files present for configured tickers
  - Market data parquet files not stale (< 25h old)
  - All strategy packages importable
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .config import HarnessConfig, get_config


@dataclass
class HealthResult:
    name: str
    status: str          # "OK" | "WARN" | "FAIL"
    message: str
    latency_ms: Optional[float] = None


@dataclass
class HealthReport:
    timestamp: datetime = field(default_factory=datetime.now)
    results: List[HealthResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.status != "FAIL" for r in self.results)

    @property
    def warnings(self) -> List[HealthResult]:
        return [r for r in self.results if r.status == "WARN"]

    @property
    def failures(self) -> List[HealthResult]:
        return [r for r in self.results if r.status == "FAIL"]


class HealthChecker:
    """Validates all harness dependencies before running."""

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()

    def run(self) -> HealthReport:
        report = HealthReport()
        report.results.extend(self._check_apis())
        report.results.extend(self._check_models())
        report.results.extend(self._check_market_data())
        report.results.extend(self._check_strategy_imports())
        return report

    # ── API checks ────────────────────────────────────────────────────────────

    def _check_apis(self) -> List[HealthResult]:
        results = []
        for name, url in [
            ("sentiment_analysis", f"{self.cfg.sentiment_api_url}/api/health"),
            ("risk_calculator", f"{self.cfg.risk_api_url}/api/health"),
        ]:
            t0 = time.time()
            try:
                resp = requests.get(url, timeout=15)
                latency = (time.time() - t0) * 1000
                if resp.status_code == 200:
                    results.append(HealthResult(name, "OK", f"UP ({latency:.0f}ms)", latency))
                else:
                    results.append(HealthResult(name, "WARN", f"HTTP {resp.status_code}", latency))
            except Exception as e:
                results.append(HealthResult(name, "WARN", f"Unreachable: {e}"))
        return results

    # ── Model file checks ─────────────────────────────────────────────────────

    def _check_models(self) -> List[HealthResult]:
        results = []
        models_dir = Path(self.cfg.models_dir)
        missing = []
        present = 0
        for ticker in self.cfg.tickers:
            model_path = models_dir / f"{ticker}_ppo.zip"
            if model_path.exists():
                present += 1
            else:
                missing.append(ticker)

        if not missing:
            results.append(HealthResult(
                "rl_models", "OK",
                f"All {present} models present"
            ))
        elif present == 0:
            results.append(HealthResult(
                "rl_models", "FAIL",
                f"No models found in {models_dir}"
            ))
        else:
            results.append(HealthResult(
                "rl_models", "WARN",
                f"{present}/{present+len(missing)} models present. Missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
            ))
        return results

    # ── Market data freshness ─────────────────────────────────────────────────

    def _check_market_data(self) -> List[HealthResult]:
        results = []
        data_dir = Path(self.cfg.market_data_dir)
        if not data_dir.exists():
            results.append(HealthResult("market_data", "FAIL", f"Directory not found: {data_dir}"))
            return results

        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            results.append(HealthResult("market_data", "FAIL", "No parquet files found"))
            return results

        cutoff = datetime.now() - timedelta(hours=25)
        stale = []
        for f in parquet_files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                stale.append(f.stem)

        if not stale:
            results.append(HealthResult(
                "market_data", "OK",
                f"{len(parquet_files)} files, all fresh"
            ))
        else:
            results.append(HealthResult(
                "market_data", "WARN",
                f"{len(stale)}/{len(parquet_files)} files stale (>25h): {', '.join(stale[:5])}"
            ))
        return results

    # ── Strategy package imports ──────────────────────────────────────────────

    def _check_strategy_imports(self) -> List[HealthResult]:
        results = []
        checks = [
            ("rl_strategy", "rl_strategy.config"),
            ("mean_reversion", "mean_reversion.config"),
            ("trend_following", "trend_following.config"),
            ("volatility_breakout", "volatility_breakout.config"),
        ]
        for name, module in checks:
            try:
                __import__(module)
                results.append(HealthResult(name, "OK", "Importable"))
            except ImportError as e:
                results.append(HealthResult(name, "WARN", f"Import failed: {e}"))
        return results


def print_health_report(report: HealthReport) -> None:
    """Print health report in a formatted table."""
    status_icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}
    print(f"\n{'='*65}")
    print(f"HARNESS HEALTH CHECK — {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"{'Check':<25} {'Status':<8} {'Details'}")
    print("-" * 65)
    for r in report.results:
        icon = status_icon.get(r.status, "?")
        print(f"  {icon} {r.name:<23} {r.status:<8} {r.message}")
    print("-" * 65)

    if report.ok:
        print(f"\n  [OK] All checks passed. Ready to run.\n")
    else:
        fails = len(report.failures)
        warns = len(report.warnings)
        print(f"\n  {fails} failure(s), {warns} warning(s). Review above.\n")
