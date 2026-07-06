"""Signal orchestrator — collects signals from all strategies in parallel.

Each strategy runs against its own ticker universe:
  RL  → cfg._rl_tickers  (filtered by backtest Sharpe)
  MR  → cfg._mr_tickers
  TF  → cfg._tf_tickers
  VB  → cfg._vb_tickers

Returns: { ticker → [HarnessSignal, ...] } covering the union of all tickers.
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from .adapters.base import BaseAdapter, HarnessSignal
from .adapters.rl_adapter import RLAdapter
from .adapters.mr_adapter import MRAdapter
from .adapters.tf_adapter import TFAdapter
from .adapters.vb_adapter import VBAdapter
from .config import HarnessConfig, get_config
from .regime import Regime, detect_regime, get_regime_probs
from .allocator import CapitalAllocator
from .paper_trading.db import save_regime_log

log = logging.getLogger(__name__)
_TRADING_ROOT = Path(__file__).resolve().parent.parent


def _progress(done: int, total: int, label: str = "") -> None:
    """Print an in-place progress bar to stdout."""
    pct = done / total * 100 if total else 0
    bar_len = 30
    filled = int(bar_len * done // max(total, 1))
    bar = "#" * filled + "-" * (bar_len - filled)
    suffix = f" {label}" if label else ""
    print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}{suffix}   ", end="", flush=True)
    if done >= total:
        print()  # newline on completion


class Orchestrator:
    """Collects signals from all enabled strategies, each on its own ticker universe."""

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()
        self._adapters: Dict[str, BaseAdapter] = {}
        self._init_adapters()

    def _init_adapters(self) -> None:
        adapter_map = {
            "rl": lambda: RLAdapter(models_dir=self.cfg.models_dir),
            "mr": MRAdapter,
            "tf": TFAdapter,
            "vb": VBAdapter,
        }
        for name in self.cfg.strategies:
            if name in adapter_map:
                try:
                    self._adapters[name] = adapter_map[name]()
                    log.debug("Initialized %s adapter", name)
                except Exception as e:
                    log.warning("Failed to initialize %s adapter: %s", name, e)

    def _tickers_for(self, strategy: str) -> List[str]:
        """Return the ticker list for a given strategy."""
        attr = f"_{strategy}_tickers"
        return getattr(self.cfg, attr, None) or self.cfg.tickers

    def _sync_from_alpaca(self) -> None:
        """Sync open positions and cash from Alpaca at the start of each signal cycle.

        1. Updates harness_trades.db positions to match Alpaca exactly.
        2. Invalidates the RLAdapter cache so it re-reads the fresh account state.
        """
        try:
            from .paper_trading.db import HarnessTradingDB
            n = HarnessTradingDB(self.cfg.paper_db_path).sync_from_alpaca()
            log.info("[orchestrator] Alpaca sync: %d open positions updated", n)
            print(f"  Alpaca sync: {n} open position(s) refreshed")
        except Exception as e:
            log.warning("[orchestrator] Alpaca sync failed (%s) — continuing with local DB state", e)
            print(f"  Alpaca sync: unavailable ({e})")

        # Invalidate RLAdapter cache so it re-fetches account state on first use
        rl_adapter = self._adapters.get("rl")
        if rl_adapter is not None and hasattr(rl_adapter, "invalidate_cache"):
            rl_adapter.invalidate_cache()
            log.debug("[orchestrator] RLAdapter account cache invalidated")

    def _intraday_drawdown(self) -> Optional[float]:
        """Return today's drawdown as a positive fraction (e.g. 0.05 = -5% on the day).

        Primary source: Alpaca equity vs last_equity (previous market close) — this is
        the true intraday NAV change and excludes unrealized losses carried from prior days.
        Fallback: today's REALIZED PnL from the harness DB (never all-time unrealized).
        Returns None if drawdown cannot be determined (circuit breaker then does not trip).
        """
        # Primary: Alpaca intraday equity change
        try:
            from trading_core.alpaca_broker import AlpacaBroker
            info = AlpacaBroker(paper=(self.cfg.execution_mode != "live")).get_account_info()
            equity = info.get("equity")
            last_equity = info.get("last_equity")
            if equity is not None and last_equity and last_equity > 0:
                intraday_return = (equity - last_equity) / last_equity
                dd = -intraday_return if intraday_return < 0 else 0.0
                log.info("[orchestrator] Intraday NAV: equity=%.2f last_equity=%.2f drawdown=%.2f%%",
                         equity, last_equity, dd * 100)
                return dd
        except Exception as e:
            log.warning("[orchestrator] Alpaca intraday drawdown unavailable (%s) — using realized PnL fallback", e)

        # Fallback: today's realized PnL only (exclude all-time unrealized)
        try:
            from datetime import date
            from pathlib import Path
            from .paper_trading.db import _conn
            today = date.today().isoformat()
            with _conn(Path(self.cfg.paper_db_path)) as conn:
                row = conn.execute(
                    "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades "
                    "WHERE DATE(executed_at) = ?",
                    (today,),
                ).fetchone()
            realized_today = float(row[0]) if row and row[0] is not None else 0.0
            nav = self.cfg.total_capital
            dd = -realized_today / nav if realized_today < 0 and nav > 0 else 0.0
            log.info("[orchestrator] Intraday drawdown (realized fallback): %.2f%%", dd * 100)
            return dd
        except Exception as e:
            log.warning("[orchestrator] Could not determine intraday drawdown: %s", e)
            return None

    def run(
        self,
        tickers: Optional[List[str]] = None,
        max_workers: int = 8,
    ) -> Dict[str, List[HarnessSignal]]:
        """Collect signals for each strategy's own ticker universe (or override list).

        Returns:
            { "AAPL": [rl_signal, mr_signal, ...], ... }
        """
        # Sync positions and cash from Alpaca before any signals are generated
        self._sync_from_alpaca()

        # D1/D2: Detect market regime using SPY and compute regime-aware allocation
        circuit_breaker_tripped = False
        regime = Regime.RANGE_BOUND
        allocation = None
        try:
            from trading_core.market_data import fetch_ohlcv
            
            # Fetch previous regime
            previous_regime = None
            try:
                from .paper_trading.unified_reader import summary as db_summary
                db_stats = db_summary(days=1) # look at today's stats or just last log
                from .paper_trading.db import HarnessTradingDB
                db = HarnessTradingDB(self.cfg.paper_db_path)
                with db.get_connection() as conn:
                    row = conn.execute("SELECT regime FROM allocation_history ORDER BY id DESC LIMIT 1").fetchone()
                    if row and row[0]:
                        try:
                            previous_regime = Regime(row[0])
                        except Exception:
                            pass
            except Exception:
                pass
                
            spy_ohlcv = fetch_ohlcv("SPY", lookback_days=365)
            regime = detect_regime(
                spy_ohlcv,
                previous_regime=previous_regime,
                mode=self.cfg.regime_mode,
                model_path=self.cfg.regime_model_path,
            )
            regime_probs = None
            if self.cfg.regime_soft_blend:
                regime_probs = get_regime_probs(spy_ohlcv, model_path=self.cfg.regime_model_path)
            allocation = CapitalAllocator(self.cfg).allocate_for_regime(regime, regime_probs=regime_probs)
            log.info("[orchestrator] Regime=%s | DetectionMode=%s | AllocationMode=%s",
                     regime.value, self.cfg.regime_mode, allocation.mode)
            print(f"  Regime: {regime.value.upper()} [{self.cfg.regime_mode}]  |  Allocation: " +
                  "  ".join(f"{a.strategy}={a.capital:,.0f}" for a in allocation.allocations))
            
            # Circuit breaker check 1: Regime
            if self.cfg.circuit_breaker_extreme_bearish and regime == Regime.BEAR_TREND:
                # Basic check, ideally we check for HIGH_VOL + BEAR_TREND
                pass
                
            # Circuit breaker check 2: True intraday drawdown
            dd_pct = self._intraday_drawdown()
            if dd_pct is not None and dd_pct > self.cfg.circuit_breaker_drawdown_pct:
                log.error("CIRCUIT BREAKER TRIGGERED: Intraday Drawdown %.2f%% > %.2f%% limit", dd_pct*100, self.cfg.circuit_breaker_drawdown_pct*100)
                print(f"\n[bold red]CIRCUIT BREAKER TRIGGERED: Intraday Drawdown {dd_pct*100:.2f}% > {self.cfg.circuit_breaker_drawdown_pct*100:.2f}% limit[/bold red]")
                print("  New entries (BUY/SHORT) blocked — exits (SELL/COVER) remain active.\n")
                circuit_breaker_tripped = True
                
            # D3: Log regime + allocation to harness_trades.db
            save_regime_log(
                regime=regime.value,
                allocation_mode=allocation.mode,
                allocations=[{
                    "strategy": a.strategy,
                    "capital": round(a.capital, 2),
                    "weight": round(a.weight, 4),
                    "sharpe": a.sharpe,
                } for a in allocation.allocations],
            )
        except Exception as exc:
            log.warning("[orchestrator] Regime detection failed: %s", exc)
        # Build (ticker, strategy, adapter) task list
        if tickers:
            # Explicit override — run all active strategies against the given tickers
            tasks = [
                (ticker, name, adapter)
                for ticker in tickers
                for name, adapter in self._adapters.items()
            ]
        else:
            # Each strategy uses its own universe
            tasks = [
                (ticker, name, adapter)
                for name, adapter in self._adapters.items()
                for ticker in self._tickers_for(name)
            ]

        # Collect all unique tickers appearing in tasks
        all_tickers = sorted({t for t, _, _ in tasks})
        results: Dict[str, List[HarnessSignal]] = {t: [] for t in all_tickers}

        total = len(tasks)
        done_count = 0
        lock = Lock()

        t0 = time.time()
        print(f"\n  Collecting signals: {len(self._adapters)} strategies, {len(all_tickers)} tickers, {total} calls")
        _progress(0, total)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(adapter.get_signal, ticker): (ticker, name)
                for ticker, name, adapter in tasks
            }
            for future in as_completed(futures):
                ticker, strategy = futures[future]
                try:
                    sig = future.result()
                    with lock:
                        results[ticker].append(sig)
                        done_count += 1
                        _progress(done_count, total, f"{strategy}:{ticker} -> {sig.action}")
                    log.debug("%s:%s -> %s (conf=%.2f)", strategy, ticker, sig.action, sig.confidence)
                except Exception as e:
                    log.error("Error %s:%s - %s", strategy, ticker, e)
                    with lock:
                        results[ticker].append(HarnessSignal(
                            ticker=ticker,
                            timestamp=datetime.now(),
                            action="HOLD",
                            confidence=0.0,
                            source=strategy,
                            price=0.0,
                            reason=f"Error: {e}",
                        ))
                        done_count += 1
                        _progress(done_count, total, f"{strategy}:{ticker} -> ERROR")

        elapsed = time.time() - t0
        total_signals = sum(len(v) for v in results.values())
        total_errors = sum(1 for v in results.values() for s in v if s.reason and s.reason.startswith("Error"))
        print(f"  Done - {total_signals} signals ({total_errors} degraded) in {elapsed:.1f}s")
        log.info("Orchestrator: %d signals (%d degraded), %d tickers, %.1fs", total_signals, total_errors, len(all_tickers), elapsed)

        # Circuit breaker: demote any new-entry signals, leave exits intact
        if circuit_breaker_tripped:
            blocked = 0
            for sig_list in results.values():
                for sig in sig_list:
                    if sig.action in ("BUY", "SHORT"):
                        sig.action = "HOLD"
                        sig.confidence = 0.0
                        sig.reason = f"blocked:circuit_breaker | {sig.reason}"
                        blocked += 1
            if blocked:
                log.info("[orchestrator] Circuit breaker blocked %d entry signal(s); exits preserved", blocked)
                print(f"  Circuit breaker: blocked {blocked} entry signal(s). SELL/COVER signals preserved.")

        self._log_signals(results)
        return results

    def _log_signals(self, results: Dict[str, List[HarnessSignal]]) -> None:
        """Append signal summary to logs/harness_signals.log."""
        log_path = Path(self.cfg.logs_dir) / "harness_signals.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n--- {ts} ---\n")
                for ticker, signals in sorted(results.items()):
                    for sig in signals:
                        f.write(
                            f"{ts}  {sig.source:<4}  {ticker:<8}  "
                            f"{sig.action:<4}  conf={sig.confidence:.2f}"
                            f"  price={sig.price:.2f}\n"
                        )
        except Exception as e:
            log.warning("Could not write signal log: %s", e)
