"""Unified DB reader — aggregates positions and trades across all strategy DBs.

Reads from all 5 paper trading databases:
  harness_trades.db              (harness-executed unified trades)
  mean_reversion/mr_paper_trades.db
  trend_following/paper_trades.db
  volatility_breakout/paper_trading/vb_paper_trades.db
  rl_strategy/paper_trades.db

Strategy DBs are read-only from here — they are still owned by each strategy.
The harness DB is the authoritative source for harness-executed trades.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_TRADING_ROOT = Path(__file__).resolve().parents[2]

_DB_MAP: Dict[str, Path] = {
    "harness": _TRADING_ROOT / "harness" / "harness_trades.db",
    "mr":      _TRADING_ROOT / "mean_reversion" / "mr_paper_trades.db",
    "tf":      _TRADING_ROOT / "trend_following" / "paper_trades.db",
    "vb":      _TRADING_ROOT / "volatility_breakout" / "paper_trading" / "vb_paper_trades.db",
    "rl":      _TRADING_ROOT / "rl_strategy" / "paper_trades.db",
}


@dataclass
class UnifiedPosition:
    ticker: str
    strategy: str
    shares: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class UnifiedTrade:
    ticker: str
    strategy: str
    action: str
    shares: float
    price: float
    executed_at: str
    pnl: Optional[float]
    reason: Optional[str]


def _safe_conn(db_path: Path) -> Optional[sqlite3.Connection]:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def get_all_positions() -> List[UnifiedPosition]:
    """Read open positions from all strategy DBs plus the harness DB."""
    results: List[UnifiedPosition] = []

    # ── harness DB ────────────────────────────────────────────────────────────
    conn = _safe_conn(_DB_MAP["harness"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, strategy, shares, entry_price, current_price, "
                "unrealized_pnl, realized_pnl FROM positions WHERE shares > 0"
            ).fetchall()
            for r in rows:
                results.append(UnifiedPosition(
                    ticker=r["ticker"], strategy=r["strategy"],
                    shares=r["shares"], avg_cost=r["entry_price"],
                    current_price=r["current_price"],
                    unrealized_pnl=r["unrealized_pnl"],
                    realized_pnl=r["realized_pnl"],
                ))
        finally:
            conn.close()

    # ── mean_reversion ────────────────────────────────────────────────────────
    conn = _safe_conn(_DB_MAP["mr"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, shares, avg_cost FROM paper_positions WHERE shares > 0"
            ).fetchall()
            for r in rows:
                results.append(UnifiedPosition(
                    ticker=r["ticker"], strategy="mr",
                    shares=float(r["shares"]), avg_cost=float(r["avg_cost"]),
                    current_price=float(r["avg_cost"]),
                    unrealized_pnl=0.0, realized_pnl=0.0,
                ))
        finally:
            conn.close()

    # ── trend_following ───────────────────────────────────────────────────────
    conn = _safe_conn(_DB_MAP["tf"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, shares, avg_cost FROM paper_positions WHERE shares > 0"
            ).fetchall()
            for r in rows:
                results.append(UnifiedPosition(
                    ticker=r["ticker"], strategy="tf",
                    shares=float(r["shares"]), avg_cost=float(r["avg_cost"]),
                    current_price=float(r["avg_cost"]),
                    unrealized_pnl=0.0, realized_pnl=0.0,
                ))
        finally:
            conn.close()

    # ── volatility_breakout ───────────────────────────────────────────────────
    conn = _safe_conn(_DB_MAP["vb"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, shares, avg_cost FROM positions WHERE shares > 0"
            ).fetchall()
            for r in rows:
                results.append(UnifiedPosition(
                    ticker=r["ticker"], strategy="vb",
                    shares=float(r["shares"]), avg_cost=float(r["avg_cost"]),
                    current_price=float(r["avg_cost"]),
                    unrealized_pnl=0.0, realized_pnl=0.0,
                ))
        finally:
            conn.close()

    # ── rl_strategy ───────────────────────────────────────────────────────────
    conn = _safe_conn(_DB_MAP["rl"])
    if conn:
        try:
            # RL DB may use different schema — probe columns
            cols = {r[1] for r in conn.execute("PRAGMA table_info(positions)")}
            if "shares" in cols and "avg_cost" in cols:
                rows = conn.execute(
                    "SELECT ticker, shares, avg_cost FROM positions WHERE shares > 0"
                ).fetchall()
                for r in rows:
                    results.append(UnifiedPosition(
                        ticker=r["ticker"], strategy="rl",
                        shares=float(r["shares"]), avg_cost=float(r["avg_cost"]),
                        current_price=float(r["avg_cost"]),
                        unrealized_pnl=0.0, realized_pnl=0.0,
                    ))
        except Exception:
            pass
        finally:
            conn.close()

    return results


def get_all_trades(limit_per_db: int = 50) -> List[UnifiedTrade]:
    """Read recent trades from all strategy DBs plus the harness DB."""
    results: List[UnifiedTrade] = []

    # harness
    conn = _safe_conn(_DB_MAP["harness"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, strategy, action, shares, price, executed_at, NULL as pnl, NULL as reason "
                "FROM trades ORDER BY executed_at DESC LIMIT ?", (limit_per_db,)
            ).fetchall()
            for r in rows:
                results.append(UnifiedTrade(
                    ticker=r["ticker"], strategy=r["strategy"],
                    action=r["action"], shares=float(r["shares"]),
                    price=float(r["price"]), executed_at=r["executed_at"],
                    pnl=None, reason=None,
                ))
        finally:
            conn.close()

    # mr
    conn = _safe_conn(_DB_MAP["mr"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, action, shares, price, executed_at, pnl, reason "
                "FROM paper_trades ORDER BY executed_at DESC LIMIT ?", (limit_per_db,)
            ).fetchall()
            for r in rows:
                results.append(UnifiedTrade(
                    ticker=r["ticker"], strategy="mr",
                    action=r["action"], shares=float(r["shares"]),
                    price=float(r["price"]), executed_at=r["executed_at"],
                    pnl=r["pnl"], reason=r["reason"],
                ))
        finally:
            conn.close()

    # tf
    conn = _safe_conn(_DB_MAP["tf"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, action, shares, price, executed_at, pnl, reason "
                "FROM paper_trades ORDER BY executed_at DESC LIMIT ?", (limit_per_db,)
            ).fetchall()
            for r in rows:
                results.append(UnifiedTrade(
                    ticker=r["ticker"], strategy="tf",
                    action=r["action"], shares=float(r["shares"]),
                    price=float(r["price"]), executed_at=r["executed_at"],
                    pnl=r["pnl"], reason=r["reason"],
                ))
        finally:
            conn.close()

    # vb
    conn = _safe_conn(_DB_MAP["vb"])
    if conn:
        try:
            rows = conn.execute(
                "SELECT ticker, action, shares, price, executed_at, reason "
                "FROM trades ORDER BY executed_at DESC LIMIT ?", (limit_per_db,)
            ).fetchall()
            for r in rows:
                results.append(UnifiedTrade(
                    ticker=r["ticker"], strategy="vb",
                    action=r["action"], shares=float(r["shares"]),
                    price=float(r["price"]), executed_at=r["executed_at"],
                    pnl=None, reason=r["reason"],
                ))
        finally:
            conn.close()

    results.sort(key=lambda t: t.executed_at, reverse=True)
    return results


def summary(days: Optional[int] = None) -> Dict[str, Any]:
    """Return a dict summary of positions and P&L across all DBs."""
    positions = get_all_positions()
    open_count = len(positions)
    total_unrealized = sum(p.unrealized_pnl for p in positions)
    total_realized = sum(p.realized_pnl for p in positions)
    by_strategy: Dict[str, int] = {}
    for p in positions:
        by_strategy[p.strategy] = by_strategy.get(p.strategy, 0) + 1

    return {
        "open_positions": open_count,
        "by_strategy": by_strategy,
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_realized_pnl": round(total_realized, 2),
    }
