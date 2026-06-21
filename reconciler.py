"""Signal reconciler — resolves conflicting signals across strategies.

Four pluggable modes:
  confidence_weighted — weighted vote by confidence score (default)
  majority_vote       — most-voted action wins
  rl_priority         — RL signal overrides if confidence > threshold
  consensus_only      — only act when ≥ 3/4 strategies agree
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .adapters.base import HarnessSignal
from .config import HarnessConfig, get_config

log = logging.getLogger(__name__)


@dataclass
class ReconciledSignal:
    """Final resolved signal for one ticker after reconciliation."""
    ticker: str
    timestamp: datetime
    action: str           # "BUY" | "SELL" | "HOLD"
    confidence: float     # aggregate confidence
    price: float
    suggested_shares: Optional[float]
    votes: Dict[str, str]             # { "rl": "BUY", "mr": "SELL", ... }
    vote_confidences: Dict[str, float]
    mode_used: str
    conflict: bool        # True when strategies disagree
    reason: str

    def __str__(self) -> str:
        votes_str = "  ".join(f"{k}={v}" for k, v in self.votes.items())
        conflict_tag = " [CONFLICT]" if self.conflict else ""
        return (
            f"{self.ticker:<8} {self.action:<4} conf={self.confidence:.2f}"
            f"{conflict_tag}  [{votes_str}]"
        )


class SignalReconciler:
    """Resolves per-ticker signal lists into a single ReconciledSignal."""

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()

    def reconcile_all(
        self,
        signals_map: Dict[str, List[HarnessSignal]],
    ) -> Dict[str, ReconciledSignal]:
        """Reconcile signals for every ticker.

        Returns:
            { "AAPL": ReconciledSignal, ... }
        """
        return {
            ticker: self.reconcile(signals)
            for ticker, signals in signals_map.items()
        }

    def reconcile(self, signals: List[HarnessSignal]) -> ReconciledSignal:
        """Reconcile a list of signals (one per strategy) for one ticker."""
        if not signals:
            raise ValueError("Empty signal list")

        ticker = signals[0].ticker
        mode = self.cfg.reconciliation_mode

        mode_fn = {
            "confidence_weighted": self._confidence_weighted,
            "majority_vote": self._majority_vote,
            "rl_priority": self._rl_priority,
            "consensus_only": self._consensus_only,
        }.get(mode, self._confidence_weighted)

        action, confidence = mode_fn(signals)

        # Apply minimum confidence threshold
        if confidence < self.cfg.min_confidence_to_act and action != "HOLD":
            action = "HOLD"
            confidence = 0.0

        votes = {s.source: s.action for s in signals}
        vote_confs = {s.source: s.confidence for s in signals}
        active_actions = [s.action for s in signals if s.action != "HOLD"]
        conflict = len(set(active_actions)) > 1 if active_actions else False

        price = next((s.price for s in signals if s.price > 0), 0.0)
        shares = next((s.suggested_shares for s in signals if s.suggested_shares), None)

        vote_str = " | ".join(
            f"{s.source}={s.action}({s.confidence:.2f})" for s in signals
        )
        reason = f"mode={mode} | {vote_str}"

        result = ReconciledSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            price=price,
            suggested_shares=shares,
            votes=votes,
            vote_confidences=vote_confs,
            mode_used=mode,
            conflict=conflict,
            reason=reason,
        )

        if conflict:
            log.info("Conflict on %s: %s -> %s (conf=%.2f)", ticker, votes, action, confidence)

        return result

    # ── Reconciliation modes ──────────────────────────────────────────────────

    def _confidence_weighted(self, signals: List[HarnessSignal]) -> tuple[str, float]:
        """Weighted average of actions by confidence score."""
        buy_score = sum(s.confidence for s in signals if s.action == "BUY")
        sell_score = sum(s.confidence for s in signals if s.action == "SELL")
        hold_score = sum(s.confidence for s in signals if s.action == "HOLD")

        total = buy_score + sell_score + hold_score
        if total == 0:
            return "HOLD", 0.0

        if buy_score >= sell_score and buy_score >= hold_score:
            return "BUY", buy_score / total
        elif sell_score >= buy_score and sell_score >= hold_score:
            return "SELL", sell_score / total
        else:
            return "HOLD", hold_score / total

    def _majority_vote(self, signals: List[HarnessSignal]) -> tuple[str, float]:
        """Action with the most votes wins."""
        vote_counts: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        vote_conf: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

        for s in signals:
            vote_counts[s.action] = vote_counts.get(s.action, 0) + 1
            vote_conf[s.action] = vote_conf.get(s.action, 0.0) + s.confidence

        winner = max(vote_counts, key=lambda a: (vote_counts[a], vote_conf[a]))
        n = len(signals)
        confidence = vote_conf[winner] / n if n > 0 else 0.0
        return winner, confidence

    def _rl_priority(self, signals: List[HarnessSignal]) -> tuple[str, float]:
        """RL signal overrides if its confidence >= threshold (0.55), else fall back to weighted."""
        rl_signals = [s for s in signals if s.source == "rl"]
        if rl_signals:
            rl = rl_signals[0]
            if rl.confidence >= self.cfg.min_confidence_to_act:
                return rl.action, rl.confidence
        return self._confidence_weighted(signals)

    def _consensus_only(self, signals: List[HarnessSignal]) -> tuple[str, float]:
        """Only act when ≥ 3/4 strategies agree on the same non-HOLD action."""
        n = len(signals)
        threshold = max(3, int(n * 0.75))
        vote_counts: Dict[str, int] = {}
        vote_conf: Dict[str, float] = {}

        for s in signals:
            if s.action != "HOLD":
                vote_counts[s.action] = vote_counts.get(s.action, 0) + 1
                vote_conf[s.action] = vote_conf.get(s.action, 0.0) + s.confidence

        for action, count in vote_counts.items():
            if count >= threshold:
                confidence = vote_conf[action] / count
                return action, confidence

        return "HOLD", 0.0
