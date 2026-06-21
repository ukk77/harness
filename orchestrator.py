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

log = logging.getLogger(__name__)
_TRADING_ROOT = Path(__file__).resolve().parent.parent


def _progress(done: int, total: int, label: str = "") -> None:
    """Print an in-place progress bar to stdout."""
    pct = done / total * 100 if total else 0
    bar_len = 30
    filled = int(bar_len * done // max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
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

    def run(
        self,
        tickers: Optional[List[str]] = None,
        max_workers: int = 8,
    ) -> Dict[str, List[HarnessSignal]]:
        """Collect signals for each strategy's own ticker universe (or override list).

        Returns:
            { "AAPL": [rl_signal, mr_signal, ...], ... }
        """
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
                        _progress(done_count, total, f"{strategy}:{ticker} → {sig.action}")
                    log.debug("%s:%s → %s (conf=%.2f)", strategy, ticker, sig.action, sig.confidence)
                except Exception as e:
                    log.error("Error %s:%s — %s", strategy, ticker, e)
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
                        _progress(done_count, total, f"{strategy}:{ticker} → ERROR")

        elapsed = time.time() - t0
        total_signals = sum(len(v) for v in results.values())
        print(f"  Done — {total_signals} signals in {elapsed:.1f}s")
        log.info("Orchestrator: %d signals, %d tickers, %.1fs", total_signals, len(all_tickers), elapsed)

        self._log_signals(results)
        return results

    def _log_signals(self, results: Dict[str, List[HarnessSignal]]) -> None:
        """Append signal summary to logs/harness_signals.log."""
        log_path = Path(self.cfg.logs_dir) / "harness_signals.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "a") as f:
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
