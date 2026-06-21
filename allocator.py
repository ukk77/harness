"""Capital allocator — distributes capital across strategies based on Sharpe.

Modes:
  equal          — simple equal split across active strategies
  sharpe_weighted — allocate proportionally to each strategy's mean backtest Sharpe
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import HarnessConfig, get_config

log = logging.getLogger(__name__)

_TRADING_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StrategyAllocation:
    strategy: str
    capital: float
    weight: float
    sharpe: Optional[float]


@dataclass
class AllocationResult:
    total_capital: float
    allocations: List[StrategyAllocation]
    mode: str

    def get(self, strategy: str) -> float:
        for a in self.allocations:
            if a.strategy == strategy:
                return a.capital
        return 0.0

    def position_cap(self, ticker: str, cfg: HarnessConfig) -> float:
        """Max $ per ticker across the whole portfolio."""
        return cfg.total_capital * cfg.max_position_pct


class CapitalAllocator:
    """Allocates harness capital across strategies."""

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()

    def allocate(self, mode: str = "sharpe_weighted") -> AllocationResult:
        if mode == "equal":
            return self._equal_split()
        elif mode == "sharpe_weighted":
            return self._sharpe_weighted()
        else:
            log.warning("Unknown allocation mode '%s', falling back to equal", mode)
            return self._equal_split()

    # ── Allocation modes ──────────────────────────────────────────────────────

    def _equal_split(self) -> AllocationResult:
        strategies = self.cfg.strategies
        n = len(strategies)
        per_strategy = self.cfg.total_capital / n
        allocations = [
            StrategyAllocation(s, per_strategy, 1.0 / n, None)
            for s in strategies
        ]
        return AllocationResult(self.cfg.total_capital, allocations, "equal")

    def _sharpe_weighted(self) -> AllocationResult:
        """Allocate capital proportional to each strategy's mean backtest Sharpe.

        Reads from results/*_backtest.json (RL) and results/*_evaluation.json.
        Falls back to equal split if no results found.
        """
        sharpes: Dict[str, float] = {}
        for strategy in self.cfg.strategies:
            sharpes[strategy] = self._mean_sharpe(strategy)

        total_sharpe = sum(max(s, 0) for s in sharpes.values())

        if total_sharpe == 0:
            log.warning("No Sharpe data found; falling back to equal split")
            return self._equal_split()

        allocations = []
        for strategy in self.cfg.strategies:
            s = max(sharpes.get(strategy, 0), 0)
            weight = s / total_sharpe
            # Enforce max_strategy_pct cap
            weight = min(weight, self.cfg.max_strategy_pct)
            capital = self.cfg.total_capital * weight
            allocations.append(StrategyAllocation(strategy, capital, weight, sharpes[strategy]))

        # Re-normalise after capping
        total_alloc = sum(a.capital for a in allocations)
        if total_alloc > 0:
            for a in allocations:
                a.capital = (a.capital / total_alloc) * self.cfg.total_capital
                a.weight = a.capital / self.cfg.total_capital

        return AllocationResult(self.cfg.total_capital, allocations, "sharpe_weighted")

    def _mean_sharpe(self, strategy: str) -> float:
        """Read mean Sharpe from result JSON files for the given strategy."""
        results_dir = Path(self.cfg.results_dir)
        if not results_dir.exists():
            return 0.0

        sharpes = []

        if strategy == "rl":
            for f in results_dir.glob("*_backtest.json"):
                try:
                    data = json.loads(f.read_text())
                    s = data.get("episode_metrics", {}).get("mean_sharpe")
                    if s is not None:
                        sharpes.append(float(s))
                except Exception:
                    pass

        elif strategy in ("mr", "tf", "vb"):
            prefix = {"mr": "mr", "tf": "tf", "vb": "vb"}.get(strategy, strategy)
            for f in results_dir.glob(f"{prefix}_*.json"):
                try:
                    data = json.loads(f.read_text())
                    s = data.get("sharpe_ratio") or data.get("sharpe")
                    if s is not None:
                        sharpes.append(float(s))
                except Exception:
                    pass

            # Fallback: per-ticker evaluation files named {TICKER}_evaluation.json
            if not sharpes:
                for f in results_dir.glob("*_evaluation.json"):
                    try:
                        data = json.loads(f.read_text())
                        if data.get("strategy") == strategy:
                            s = data.get("metrics", {}).get("sharpe")
                            if s is not None:
                                sharpes.append(float(s))
                    except Exception:
                        pass

        return float(sum(sharpes) / len(sharpes)) if sharpes else 0.0
