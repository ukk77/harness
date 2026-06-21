"""Unified harness paper trading database.

Single harness_trades.db tracks all positions and trades across every strategy.
Schema mirrors the plan exactly.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "harness_trades.db"


def _conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables if they don't exist."""
    db_path = db_path or _DEFAULT_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                strategy        TEXT    NOT NULL,
                shares          REAL    NOT NULL DEFAULT 0,
                entry_price     REAL    NOT NULL DEFAULT 0,
                current_price   REAL    NOT NULL DEFAULT 0,
                entry_time      TEXT    NOT NULL,
                unrealized_pnl  REAL    NOT NULL DEFAULT 0,
                realized_pnl    REAL    NOT NULL DEFAULT 0,
                UNIQUE(ticker, strategy)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                strategy        TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                shares          REAL    NOT NULL,
                price           REAL    NOT NULL,
                confidence      REAL,
                executed_at     TEXT    NOT NULL,
                reconciled_from TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_positions_ticker  ON positions(ticker);
            CREATE INDEX IF NOT EXISTS idx_trades_ticker     ON trades(ticker);
            CREATE INDEX IF NOT EXISTS idx_trades_executed   ON trades(executed_at);

            CREATE TABLE IF NOT EXISTS regime_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at       TEXT    NOT NULL,
                regime          TEXT    NOT NULL,
                allocation_mode TEXT    NOT NULL,
                allocations_json TEXT   NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_regime_log_date ON regime_log(logged_at);
        """)
        conn.commit()


def save_regime_log(
    regime: str,
    allocation_mode: str,
    allocations: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> None:
    """Persist a regime + allocation snapshot to regime_log table."""
    db_path = db_path or _DEFAULT_DB
    init_db(db_path)
    logged_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO regime_log (logged_at, regime, allocation_mode, allocations_json) "
            "VALUES (?, ?, ?, ?)",
            (logged_at, regime, allocation_mode, json.dumps(allocations)),
        )
        conn.commit()


class HarnessTradingDB:
    """Interface to the unified harness paper trading database."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        init_db(self.db_path)

    # ── Positions ─────────────────────────────────────────────────────────────

    def upsert_position(
        self,
        ticker: str,
        strategy: str,
        shares: float,
        entry_price: float,
        current_price: float,
        unrealized_pnl: float,
        realized_pnl: float = 0.0,
    ) -> None:
        with _conn(self.db_path) as conn:
            conn.execute("""
                INSERT INTO positions
                    (ticker, strategy, shares, entry_price, current_price, entry_time, unrealized_pnl, realized_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, strategy) DO UPDATE SET
                    shares        = excluded.shares,
                    current_price = excluded.current_price,
                    unrealized_pnl = excluded.unrealized_pnl,
                    realized_pnl  = excluded.realized_pnl
            """, (ticker, strategy, shares, entry_price, current_price,
                  datetime.utcnow().isoformat(), unrealized_pnl, realized_pnl))
            conn.commit()

    def get_position(self, ticker: str, strategy: str) -> Optional[Dict[str, Any]]:
        with _conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE ticker=? AND strategy=?",
                (ticker, strategy)
            ).fetchone()
        return dict(row) if row else None

    def get_all_positions(self) -> List[Dict[str, Any]]:
        with _conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE shares > 0 ORDER BY ticker, strategy"
            ).fetchall()
        return [dict(r) for r in rows]

    def close_position(self, ticker: str, strategy: str, realized_pnl: float) -> None:
        with _conn(self.db_path) as conn:
            conn.execute("""
                UPDATE positions SET shares=0, unrealized_pnl=0, realized_pnl=realized_pnl+?
                WHERE ticker=? AND strategy=?
            """, (realized_pnl, ticker, strategy))

    def sync_from_alpaca(self, paper: bool = True) -> int:
        """Overwrite harness positions to exactly match Alpaca's open positions.

        Alpaca is the source of truth.  Any ticker not in Alpaca is zeroed out.
        Returns the number of positions synced.
        """
        try:
            from trading_core.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(paper=paper)
            alpaca_positions = broker.get_positions()   # { symbol: {...} }
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Alpaca sync skipped: %s", e)
            return 0

        with _conn(self.db_path) as conn:
            # Zero out all current harness positions
            conn.execute("UPDATE positions SET shares=0, unrealized_pnl=0 WHERE strategy='harness'")
            # Upsert each Alpaca position
            for symbol, pos in alpaca_positions.items():
                shares = float(pos["shares"])
                avg_entry = float(pos["avg_entry_price"])
                mkt_val = float(pos["market_value"])
                unreal = float(pos["unrealized_pl"])
                conn.execute("""
                    INSERT INTO positions
                        (ticker, strategy, shares, entry_price, current_price, entry_time, unrealized_pnl, realized_pnl)
                    VALUES (?, 'harness', ?, ?, ?, ?, ?, 0.0)
                    ON CONFLICT(ticker, strategy) DO UPDATE SET
                        shares         = excluded.shares,
                        entry_price    = excluded.entry_price,
                        current_price  = excluded.current_price,
                        unrealized_pnl = excluded.unrealized_pnl
                """, (
                    symbol, shares, avg_entry,
                    mkt_val / shares if shares else avg_entry,
                    datetime.utcnow().isoformat(), unreal,
                ))
            conn.commit()

        return len(alpaca_positions)

    # ── Trades ────────────────────────────────────────────────────────────────

    def record_trade(
        self,
        ticker: str,
        strategy: str,
        action: str,
        shares: float,
        price: float,
        confidence: float = 0.0,
        reconciled_from: Optional[Dict] = None,
    ) -> None:
        with _conn(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades
                    (ticker, strategy, action, shares, price, confidence, executed_at, reconciled_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, strategy, action, shares, price, confidence,
                datetime.utcnow().isoformat(),
                json.dumps(reconciled_from) if reconciled_from else None,
            ))
            conn.commit()

    def get_trades(
        self,
        ticker: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if ticker:
            query += " AND ticker=?"
            params.append(ticker)
        if strategy:
            query += " AND strategy=?"
            params.append(strategy)
        query += " ORDER BY executed_at DESC LIMIT ?"
        params.append(limit)
        with _conn(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def total_realized_pnl(self) -> float:
        with _conn(self.db_path) as conn:
            row = conn.execute("SELECT SUM(realized_pnl) FROM positions").fetchone()
        return float(row[0] or 0.0)

    def total_unrealized_pnl(self) -> float:
        with _conn(self.db_path) as conn:
            row = conn.execute("SELECT SUM(unrealized_pnl) FROM positions WHERE shares>0").fetchone()
        return float(row[0] or 0.0)
