"""Exit-sweep — enforces per-position stop-loss and max-hold-days regardless of
the reconciler's confidence gate (Master Spec § 10 S5).

Prior to this module, `suggested_stop_pct` / `expected_hold_days` were computed
by the adapters and persisted to the `positions` table (S1), but nothing ever
read them back to force an exit — a position could sit past its stop or its
expected hold horizon indefinitely if no strategy happened to emit a fresh
SELL signal for it. This closes that gap: every signal_generation cycle,
`sweep_exits()` scans open harness positions and overrides the reconciled
signal to a forced SELL wherever a stop or hold-horizon breach is detected.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .adapters.base import HarnessSignal
from .reconciler import ReconciledSignal
from .paper_trading.db import HarnessTradingDB
from .config import HarnessConfig, get_config

log = logging.getLogger(__name__)


def _parse_entry_time(entry_time: Optional[str]) -> Optional[datetime]:
    if not entry_time:
        return None
    try:
        dt = datetime.fromisoformat(entry_time)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _resolve_current_price(
    ticker: str,
    reconciled: Dict[str, ReconciledSignal],
    raw_signals: Dict[str, List[HarnessSignal]],
) -> Optional[float]:
    """Prefer the reconciled signal's price; fall back to any raw strategy signal."""
    rec = reconciled.get(ticker)
    if rec is not None and rec.price and rec.price > 0:
        return rec.price
    for sig in raw_signals.get(ticker, []) or []:
        if sig.price and sig.price > 0:
            return sig.price
    return None


def sweep_exits(
    reconciled: Dict[str, ReconciledSignal],
    raw_signals: Dict[str, List[HarnessSignal]],
    cfg: Optional[HarnessConfig] = None,
) -> Dict[str, int]:
    """Force a SELL for any open harness position whose stop or hold horizon is breached.

    Mutates `reconciled` in place (overrides or inserts a ReconciledSignal for the
    breached ticker with action="SELL", confidence=1.0 so it always clears the
    min_confidence_to_act gate downstream).

    Returns:
        {"stops_triggered": int, "holds_expired": int} — used by cli.py to feed
        accurate per-run counters into run_summary (§ 10 I3).
    """
    cfg = cfg or get_config()
    db = HarnessTradingDB(cfg.paper_db_path)
    positions = [
        p for p in db.get_all_positions()
        if p.get("strategy") == "harness" and (p.get("shares") or 0) > 0
    ]
    counts = {"stops_triggered": 0, "holds_expired": 0}
    if not positions:
        return counts

    now = datetime.now(timezone.utc)

    for pos in positions:
        ticker = pos["ticker"]
        stop_price = pos.get("stop_price")
        expected_hold_days = pos.get("expected_hold_days")
        entry_time = _parse_entry_time(pos.get("entry_time"))

        current_price = _resolve_current_price(ticker, reconciled, raw_signals)
        if current_price is None:
            continue

        breach_reason: Optional[str] = None
        breach_kind: Optional[str] = None
        if stop_price and stop_price > 0 and current_price <= stop_price:
            breach_reason = f"stop_breach(price={current_price:.2f}<=stop={stop_price:.2f})"
            breach_kind = "stops_triggered"
        elif expected_hold_days and expected_hold_days > 0 and entry_time is not None:
            held_days = (now - entry_time).days
            if held_days >= expected_hold_days:
                breach_reason = f"hold_expired(held={held_days}d>=max={expected_hold_days}d)"
                breach_kind = "holds_expired"

        if breach_reason is None:
            continue

        existing = reconciled.get(ticker)
        votes = existing.votes if existing else {}
        vote_confs = existing.vote_confidences if existing else {}
        sources = existing.strategy_sources if existing else ["exit_sweep"]

        reconciled[ticker] = ReconciledSignal(
            ticker=ticker,
            timestamp=now,
            action="SELL",
            confidence=1.0,
            price=current_price,
            suggested_shares=pos.get("shares"),
            suggested_stop_pct=None,
            expected_hold_days=None,
            risk_bucket=pos.get("risk_bucket"),
            strategy_sources=sources,
            votes=votes,
            vote_confidences=vote_confs,
            mode_used="exit_sweep",
            conflict=False,
            reason=f"blocked:exit_sweep | {breach_reason}",
        )
        counts[breach_kind] += 1
        log.warning("[exit_sweep] Forced SELL %s — %s", ticker, breach_reason)

    return counts
