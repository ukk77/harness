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
from datetime import datetime, timedelta, timezone
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

    def __init__(self, cfg: Optional[HarnessConfig] = None, request_timeout: float = 15.0):
        self.cfg = cfg or get_config()
        self.request_timeout = request_timeout

    def run(self) -> HealthReport:
        report = HealthReport()
        report.results.extend(self._check_apis())
        report.results.extend(self._check_rag())
        report.results.extend(self._check_models())
        report.results.extend(self._check_market_data())
        report.results.extend(self._check_strategy_imports())
        return report

    # ── API checks ────────────────────────────────────────────────────────────

    def _check_apis(self) -> List[HealthResult]:
        results = []
        for name, url in [
            ("sentiment_analysis", f"{self.cfg.sentiment_api_url}"),
            ("risk_calculator", f"{self.cfg.risk_api_url}"),
        ]:
            t0 = time.time()
            try:
                # 1. Check reachability
                health_url = f"{url}/api/health"
                resp = requests.get(health_url, timeout=self.request_timeout)
                latency = (time.time() - t0) * 1000
                if resp.status_code == 200:
                    results.append(HealthResult(f"{name}_api", "OK", f"UP ({latency:.0f}ms)", latency))
                else:
                    results.append(HealthResult(f"{name}_api", "WARN", f"HTTP {resp.status_code}", latency))
                    continue
                
                # 2. Check data freshness on a benchmark ticker
                bench = "SPY"
                hist_url = f"{url}/api/history/{bench}?limit=1"
                resp = requests.get(hist_url, timeout=min(self.request_timeout, 5.0))
                if resp.status_code == 200:
                    data = resp.json()
                    snapshots = data.get("snapshots", [])
                    if snapshots:
                        latest = snapshots[0]
                        # Risk calculator uses "as_of", sentiment uses "captured_at" or "as_of"
                        captured_at_str = latest.get("captured_at") or latest.get("as_of")
                        if captured_at_str:
                            try:
                                # Parse ISO format, handle Z
                                if captured_at_str.endswith("Z"):
                                    captured_at_str = captured_at_str[:-1] + "+00:00"
                                captured_at = datetime.fromisoformat(captured_at_str)
                                age_hours = (datetime.now(captured_at.tzinfo) - captured_at).total_seconds() / 3600
                                if age_hours > 4:
                                    results.append(HealthResult(f"{name}_freshness", "WARN", f"Stale data: {age_hours:.1f}h old for {bench}"))
                                else:
                                    results.append(HealthResult(f"{name}_freshness", "OK", f"Fresh: {age_hours:.1f}h old"))
                            except Exception as e:
                                results.append(HealthResult(f"{name}_freshness", "WARN", f"Could not parse timestamp: {e}"))
                        else:
                            results.append(HealthResult(f"{name}_freshness", "WARN", "No timestamp found in snapshot"))
                    else:
                        results.append(HealthResult(f"{name}_freshness", "WARN", f"No history found for {bench}"))
                else:
                    results.append(HealthResult(f"{name}_freshness", "WARN", f"History API HTTP {resp.status_code}"))
                    
            except Exception as e:
                results.append(HealthResult(f"{name}_api", "FAIL", f"Unreachable: {type(e).__name__}"))
        return results

    # ── RAG service check (non-critical — WARN if down, never FAIL) ───────────

    def _check_rag(self) -> List[HealthResult]:
        """Check rag_service reachability. Degraded (WARN) if down — not in critical path."""
        t0 = time.time()
        try:
            url = getattr(self.cfg, "rag_service_url", "http://localhost:8200")
            resp = requests.get(f"{url}/api/health", timeout=5)
            latency = (time.time() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")
                doc_count = data.get("doc_count", 0)
                if status == "healthy":
                    return [HealthResult("rag_service", "OK", f"UP ({latency:.0f}ms, {doc_count} docs)")]
                else:
                    return [HealthResult("rag_service", "WARN", f"{status} — {data.get('detail', '')} ({latency:.0f}ms)")]
            else:
                return [HealthResult("rag_service", "WARN", f"HTTP {resp.status_code} (degraded — not critical)")]
        except Exception as e:
            return [HealthResult("rag_service", "WARN", f"Unreachable (degraded — not critical): {type(e).__name__}")]

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

        cutoff = datetime.now(timezone.utc) - timedelta(hours=25)
        stale = []
        missing_freshness = []
        for f in parquet_files:
            freshness_file = f.with_suffix(".freshness")
            if freshness_file.exists():
                try:
                    ts_str = freshness_file.read_text().strip()
                    if ts_str.endswith("Z"):
                        ts_str = ts_str[:-1] + "+00:00"
                    last_update = datetime.fromisoformat(ts_str)
                    if last_update < cutoff:
                        stale.append(f.stem)
                except Exception:
                    missing_freshness.append(f.stem)
            else:
                missing_freshness.append(f.stem)

        if not stale and not missing_freshness:
            results.append(HealthResult(
                "market_data", "OK",
                f"{len(parquet_files)} files, all fresh"
            ))
        else:
            err_msg = ""
            if stale:
                err_msg += f"{len(stale)}/{len(parquet_files)} files stale (>25h): {', '.join(stale[:5])}. "
            if missing_freshness:
                err_msg += f"{len(missing_freshness)} files missing freshness metadata."
            results.append(HealthResult(
                "market_data", "WARN",
                err_msg.strip()
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
    status_icon = {"OK": "[OK]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
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
