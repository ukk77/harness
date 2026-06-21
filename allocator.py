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
from .regime import Regime

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


# Regime multipliers per strategy.
# Values < 1.0 reduce capital for that strategy in this regime;
# values > 1.0 increase it (subject to max_strategy_pct cap after re-normalisation).
_REGIME_MULTIPLIERS: Dict[Regime, Dict[str, float]] = {
    Regime.BULL_TREND: {
        "mr":  0.6,   # mean-reversion underperforms in strong trends
        "tf":  1.4,   # trend-following excels
        "vb":  1.2,   # breakouts are more reliable in bull markets
        "rl":  1.0,
    },
    Regime.BEAR_TREND: {
        "mr":  0.8,
        "tf":  1.3,   # short-side trend following still relevant
        "vb":  0.7,   # fewer clean breakouts in bear markets
        "rl":  1.0,
    },
    Regime.HIGH_VOL: {
        "mr":  1.2,   # mean-reversion can capture overshoots
        "tf":  0.7,   # trends break down in high volatility
        "vb":  0.9,
        "rl":  0.8,   # RL agents trained on normal vol may struggle
    },
    Regime.RANGE_BOUND: {
        "mr":  1.4,   # ideal regime for mean-reversion
        "tf":  0.6,
        "vb":  0.8,
        "rl":  1.0,
    },
}


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

    def allocate_for_regime(self, regime: Regime) -> AllocationResult:
        """Sharpe-weighted allocation scaled by regime-specific multipliers.

        1. Compute base Sharpe-weighted allocation.
        2. Apply _REGIME_MULTIPLIERS for the current regime.
        3. Re-normalise weights to 1.0, enforce max_strategy_pct cap.

        Returns an AllocationResult with mode set to "regime:<regime_value>".
        """
        base = self._sharpe_weighted()
        multipliers = _REGIME_MULTIPLIERS.get(regime, {})

        adjusted: List[StrategyAllocation] = []
        for a in base.allocations:
            mult = multipliers.get(a.strategy, 1.0)
            adjusted.append(StrategyAllocation(
                strategy=a.strategy,
                capital=a.capital * mult,
                weight=a.weight * mult,
                sharpe=a.sharpe,
            ))

        # Re-normalise
        total_w = sum(a.weight for a in adjusted)
        if total_w > 0:
            for a in adjusted:
                a.weight = min(a.weight / total_w, self.cfg.max_strategy_pct)

        # Second pass: re-normalise after cap
        total_w2 = sum(a.weight for a in adjusted)
        for a in adjusted:
            if total_w2 > 0:
                a.weight = a.weight / total_w2
            a.capital = self.cfg.total_capital * a.weight

        log.info(
            "[allocator] Regime=%s | %s",
            regime.value,
            " | ".join(f"{a.strategy}={a.capital:,.0f}" for a in adjusted),
        )
        return AllocationResult(
            total_capital=self.cfg.total_capital,
            allocations=adjusted,
            mode=f"regime:{regime.value}",
        )

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
