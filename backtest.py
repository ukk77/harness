"""Harness cross-strategy backtester.

Runs all 4 strategy backtests on the same date range, then simulates all
4 reconciliation modes to find which produces the best risk-adjusted returns.

Usage:
    from harness.backtest import HarnessBacktester
    bt = HarnessBacktester()
    bt.run()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_TRADING_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR  = _TRADING_ROOT / "results"
_RESULTS_DIR.mkdir(exist_ok=True)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class StrategyBacktestResult:
    strategy: str
    tickers: List[str]
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    pnl: float = 0.0            # absolute P&L in dollars
    alpha: float = 0.0          # excess return vs cash+3% hurdle
    error: Optional[str] = None


@dataclass
class ReconciliationModeResult:
    mode: str
    tickers_acted: int = 0
    tickers_held: int = 0
    conflicts_blocked: int = 0
    consensus_count: int = 0
    estimated_sharpe: float = 0.0


@dataclass
class HarnessBacktestReport:
    run_at: str
    start_date: str
    end_date: str
    capital: float
    strategy_results: List[StrategyBacktestResult] = field(default_factory=list)
    reconciliation_results: List[ReconciliationModeResult] = field(default_factory=list)
    best_strategy: Optional[str] = None
    best_recon_mode: Optional[str] = None
    recommendation: str = ""


# ── Backtester ────────────────────────────────────────────────────────────────

class HarnessBacktester:
    """Runs per-strategy backtests and compares all reconciliation modes."""

    def __init__(self, capital: float = 100_000.0, lookback_days: int = 180):
        from harness.config import get_config
        self.cfg = get_config()
        self.capital = capital
        self.lookback_days = lookback_days
        self.end_date   = datetime.today().strftime("%Y-%m-%d")
        self.start_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, strategies: Optional[List[str]] = None) -> HarnessBacktestReport:
        strats = strategies or ["rl", "mr", "tf", "vb"]
        report = HarnessBacktestReport(
            run_at=datetime.now().isoformat(),
            start_date=self.start_date,
            end_date=self.end_date,
            capital=self.capital,
        )

        total = len(strats)
        for i, strat in enumerate(strats, 1):
            pct = i / total * 100
            print(f"\r  [{i}/{total}] {pct:5.1f}%  Running {strat.upper()} backtest...", end="", flush=True)
            result = self._run_strategy(strat)
            report.strategy_results.append(result)
            if result.error:
                print(f"\r  [{i}/{total}] {strat.upper():<4}  ERROR: {result.error[:60]}")
            else:
                print(
                    f"\r  [{i}/{total}] {strat.upper():<4}  "
                    f"Sharpe={result.sharpe:5.2f}  "
                    f"CAGR={result.cagr:+.1f}%  "
                    f"MaxDD={result.max_drawdown:.1f}%  "
                    f"WinRate={result.win_rate:.0f}%  "
                    f"Trades={result.num_trades}"
                )

        print()

        # Reconciliation mode comparison using signals from a sample run
        print("  Comparing reconciliation modes on sample signals...")
        report.reconciliation_results = self._compare_recon_modes()

        # Best strategy by Sharpe
        valid = [r for r in report.strategy_results if r.error is None and r.sharpe > 0]
        if valid:
            best = max(valid, key=lambda r: r.sharpe)
            report.best_strategy = best.strategy

        # Best recon mode by estimated Sharpe
        if report.reconciliation_results:
            best_recon = max(report.reconciliation_results, key=lambda r: r.estimated_sharpe)
            report.best_recon_mode = best_recon.mode

        report.recommendation = self._make_recommendation(report)
        self._save(report)
        return report

    # ── Per-strategy backtests ────────────────────────────────────────────────

    def _run_strategy(self, strategy: str) -> StrategyBacktestResult:
        try:
            if strategy == "rl":
                return self._run_rl()
            elif strategy == "mr":
                return self._run_mr()
            elif strategy == "tf":
                return self._run_tf()
            elif strategy == "vb":
                return self._run_vb()
        except Exception as e:
            log.exception("Backtest failed for %s", strategy)
            return StrategyBacktestResult(strategy=strategy, tickers=[], error=str(e))
        return StrategyBacktestResult(strategy=strategy, tickers=[], error="unknown")

    def _run_rl(self) -> StrategyBacktestResult:
        """Aggregate RL backtest results from saved JSON files."""
        tickers = self.cfg._rl_tickers
        sharpes, returns, drawdowns, win_rates, trades_list = [], [], [], [], []

        for ticker in tickers:
            result_file = _RESULTS_DIR / f"{ticker}_backtest.json"
            if not result_file.exists():
                continue
            try:
                data = json.loads(result_file.read_text())
                episodes = data.get("episodes", [])
                if not episodes:
                    continue
                ep_sharpes  = [e.get("sharpe_ratio", 0)    for e in episodes]
                ep_returns  = [e.get("total_return_pct", 0) for e in episodes]
                ep_dds      = [e.get("max_drawdown_pct", 0) for e in episodes]
                ep_trades   = [e.get("num_trades", 0)       for e in episodes]
                sharpes.append(sum(ep_sharpes) / len(ep_sharpes))
                returns.append(sum(ep_returns) / len(ep_returns))
                drawdowns.append(sum(ep_dds)   / len(ep_dds))
                trades_list.append(sum(ep_trades))
                # Win rate from trade_metrics sub-dict
                wr = (data.get("trade_metrics") or {}).get("win_rate")
                if wr is not None:
                    win_rates.append(float(wr) * 100)
            except Exception:
                pass

        def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

        # RL P&L: mean_return across tickers (returns are in %, convert to $)
        avg_return_pct = _avg(returns)
        pnl_est = self.capital * avg_return_pct / 100.0

        return StrategyBacktestResult(
            strategy="rl",
            tickers=tickers,
            total_return=avg_return_pct,
            cagr=avg_return_pct / (self.lookback_days / 365) if returns else 0.0,
            sharpe=_avg(sharpes),
            max_drawdown=_avg(drawdowns),
            win_rate=_avg(win_rates),
            pnl=pnl_est,
            alpha=0.0,  # RL engine doesn't compute alpha vs benchmark
            num_trades=sum(trades_list),
        )

    # ── OHLC loader shared by MR / TF / VB ───────────────────────────────────

    def _load_ohlc(self, tickers: List[str], interval: str = "daily") -> Dict:
        """Load OHLC DataFrames from parquet cache for a list of tickers."""
        data_dir = _TRADING_ROOT / "market_data" / interval
        result = {}
        for ticker in tickers:
            f = data_dir / f"{ticker}.parquet"
            if not f.exists():
                continue
            try:
                df = pd.read_parquet(f)
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)  # naive UTC
                # Rename columns to Title Case expected by backtest engines
                col_map = {c: c.capitalize() for c in df.columns}
                df = df.rename(columns=col_map)
                if self.start_date:
                    df = df[df.index >= pd.Timestamp(self.start_date)]
                if self.end_date:
                    df = df[df.index <= pd.Timestamp(self.end_date)]
                if not df.empty:
                    result[ticker] = df
            except Exception as e:
                log.debug("OHLC load failed %s: %s", ticker, e)
        return result

    def _extract_metrics(self, summary, strategy: str, tickers: List[str]) -> StrategyBacktestResult:
        """Extract normalised metrics from a BacktestSummary object.

        Handles the actual key names used by MR/TF/VB engines:
          sharpe, cagr_pct, max_drawdown_pct, win_rate_pct, total_trades
        """
        m = getattr(summary, "portfolio_metrics", None) or {}

        def _g(key, alt=0.0):
            v = m.get(key, alt)
            return float(v) if v is not None else alt

        total_trades = int(_g("total_trades"))
        # Fallback: count from per-ticker results
        if total_trades == 0:
            for r in getattr(summary, "results", {}).values():
                t = getattr(r, "trades_df", None)
                if t is not None and not t.empty:
                    total_trades += len(t)

        return StrategyBacktestResult(
            strategy=strategy,
            tickers=tickers,
            total_return=_g("total_return_pct"),
            cagr=_g("cagr_pct"),
            sharpe=_g("sharpe"),
            sortino=_g("sortino"),
            max_drawdown=_g("max_drawdown_pct"),
            win_rate=_g("win_rate_pct"),
            num_trades=total_trades,
            pnl=_g("profit_loss"),
            alpha=_g("alpha_vs_cash_plus_3_pct"),
        )

    def _run_mr(self) -> StrategyBacktestResult:
        from mean_reversion.backtest.engine import run_backtest
        from mean_reversion.config import MeanReversionConfig
        tickers = self.cfg._mr_tickers
        ticker_ohlc = self._load_ohlc(tickers, "daily")
        if not ticker_ohlc:
            return StrategyBacktestResult(strategy="mr", tickers=tickers, error="No OHLC data found")
        cfg = MeanReversionConfig()
        summary = run_backtest(cfg, ticker_ohlc, {}, start_date=self.start_date, end_date=self.end_date)
        return self._extract_metrics(summary, "mr", list(ticker_ohlc.keys()))

    def _run_tf(self) -> StrategyBacktestResult:
        from trend_following.backtest.engine import run_backtest
        from trend_following.config import TrendFollowingConfig
        tickers = self.cfg._tf_tickers
        ticker_ohlc = self._load_ohlc(tickers, "daily")
        if not ticker_ohlc:
            return StrategyBacktestResult(strategy="tf", tickers=tickers, error="No OHLC data found")
        cfg = TrendFollowingConfig()
        summary = run_backtest(cfg, ticker_ohlc, {}, start_date=self.start_date, end_date=self.end_date)
        return self._extract_metrics(summary, "tf", list(ticker_ohlc.keys()))

    def _run_vb(self) -> StrategyBacktestResult:
        from volatility_breakout.backtest.engine import run_backtest
        from volatility_breakout.config import VolatilityBreakoutConfig
        tickers = self.cfg._vb_tickers
        ticker_ohlc = self._load_ohlc(tickers, "daily")
        if not ticker_ohlc:
            return StrategyBacktestResult(strategy="vb", tickers=tickers, error="No OHLC data found")
        cfg = VolatilityBreakoutConfig()
        summary = run_backtest(cfg, ticker_ohlc, {}, start_date=self.start_date, end_date=self.end_date)
        return self._extract_metrics(summary, "vb", list(ticker_ohlc.keys()))

    # ── Reconciliation mode comparison ────────────────────────────────────────

    def _compare_recon_modes(self) -> List[ReconciliationModeResult]:
        """Run the orchestrator on a sample of tickers for each recon mode and compare."""
        from harness.orchestrator import Orchestrator
        from harness.reconciler import SignalReconciler

        sample_tickers = sorted(set(
            self.cfg._rl_tickers[:3] +
            self.cfg._mr_tickers[:3] +
            self.cfg._tf_tickers[:3]
        ))[:8]

        # Collect raw signals once, reuse across all modes
        orch = Orchestrator(self.cfg)
        try:
            raw = orch.run(tickers=sample_tickers)
        except Exception as e:
            log.warning("Orchestrator sample run failed: %s", e)
            return []

        modes = ["confidence_weighted", "majority_vote", "rl_priority", "consensus_only"]
        results = []
        for mode in modes:
            try:
                # Temporarily override mode on a copy of config
                import copy
                cfg_copy = copy.copy(self.cfg)
                cfg_copy.reconciliation_mode = mode
                recon = SignalReconciler(cfg_copy)
                reconciled = recon.reconcile_all(raw)
                acted      = sum(1 for s in reconciled.values() if s.action != "HOLD")
                held       = sum(1 for s in reconciled.values() if s.action == "HOLD")
                conflicts  = sum(1 for s in reconciled.values() if getattr(s, "conflict", False))
                consensus  = sum(
                    1 for ticker, signals in raw.items()
                    if len({s.action for s in signals if s.action != "HOLD"}) == 1
                    and len(signals) >= 2
                )
                # Proxy Sharpe: average confidence of actionable signals * diversity bonus
                confs = [s.confidence for s in reconciled.values() if s.action != "HOLD"]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                diversity = acted / len(reconciled) if reconciled else 0.0
                est_sharpe = avg_conf * (1 + diversity * 0.5)
                results.append(ReconciliationModeResult(
                    mode=mode,
                    tickers_acted=acted,
                    tickers_held=held,
                    conflicts_blocked=conflicts,
                    consensus_count=consensus,
                    estimated_sharpe=round(est_sharpe, 3),
                ))
            except Exception as e:
                log.warning("Recon mode %s failed: %s", mode, e)
                results.append(ReconciliationModeResult(mode=mode, estimated_sharpe=0.0))

        return results

    # ── Recommendation ────────────────────────────────────────────────────────

    def _make_recommendation(self, report: HarnessBacktestReport) -> str:
        lines = []
        valid = [r for r in report.strategy_results if r.error is None]
        if not valid:
            return "No backtest data available — run strategy backtests first."

        best = max(valid, key=lambda r: r.sharpe) if valid else None
        current_recon = self.cfg.reconciliation_mode
        best_recon = report.best_recon_mode

        if best:
            lines.append(
                f"Best individual strategy: {best.strategy.upper()} "
                f"(Sharpe={best.sharpe:.2f}, CAGR={best.cagr:+.1f}%)"
            )

        if best_recon and best_recon != current_recon:
            lines.append(
                f"Consider switching reconciliation mode from "
                f"'{current_recon}' to '{best_recon}' for better signal quality."
            )
        else:
            lines.append(f"Current reconciliation mode '{current_recon}' is optimal.")

        # Live readiness check
        live_ready = best and best.sharpe >= 1.0 and best.max_drawdown <= 20.0
        if live_ready:
            lines.append(
                "Platform appears READY for live trading "
                f"(best Sharpe={best.sharpe:.2f} >= 1.0, MaxDD={best.max_drawdown:.1f}% <= 20%)."
            )
        else:
            reasons = []
            if best and best.sharpe < 1.0:
                reasons.append(f"Sharpe {best.sharpe:.2f} < 1.0")
            if best and best.max_drawdown > 20.0:
                reasons.append(f"MaxDD {best.max_drawdown:.1f}% > 20%")
            lines.append(f"NOT recommended for live trading yet: {', '.join(reasons)}.")

        return "  ".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, report: HarnessBacktestReport) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _RESULTS_DIR / f"harness_backtest_{ts}.json"
        data = {
            "run_at": report.run_at,
            "start_date": report.start_date,
            "end_date": report.end_date,
            "capital": report.capital,
            "best_strategy": report.best_strategy,
            "best_recon_mode": report.best_recon_mode,
            "recommendation": report.recommendation,
            "strategies": [
                {
                    "strategy": r.strategy,
                    "tickers_count": len(r.tickers),
                    "total_return_pct": round(r.total_return, 2),
                    "cagr_pct": round(r.cagr, 2),
                    "sharpe": round(r.sharpe, 3),
                    "sortino": round(r.sortino, 3),
                    "max_drawdown_pct": round(r.max_drawdown, 2),
                    "win_rate_pct": round(r.win_rate, 1),
                    "num_trades": r.num_trades,
                    "error": r.error,
                }
                for r in report.strategy_results
            ],
            "reconciliation_modes": [
                {
                    "mode": r.mode,
                    "tickers_acted": r.tickers_acted,
                    "tickers_held": r.tickers_held,
                    "conflicts_blocked": r.conflicts_blocked,
                    "consensus_count": r.consensus_count,
                    "estimated_sharpe": r.estimated_sharpe,
                }
                for r in report.reconciliation_results
            ],
        }
        out.write_text(json.dumps(data, indent=2))
        log.info("Harness backtest saved: %s", out)
        return out
