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
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker              TEXT    NOT NULL,
                strategy            TEXT    NOT NULL,
                shares              REAL    NOT NULL DEFAULT 0,
                entry_price         REAL    NOT NULL DEFAULT 0,
                current_price       REAL    NOT NULL DEFAULT 0,
                entry_time          TEXT    NOT NULL,
                unrealized_pnl      REAL    NOT NULL DEFAULT 0,
                realized_pnl        REAL    NOT NULL DEFAULT 0,
                stop_price          REAL,
                expected_hold_days  INTEGER,
                risk_bucket         TEXT,
                entry_confidence    REAL,
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
                realized_pnl    REAL    NOT NULL DEFAULT 0.0,
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

            CREATE TABLE IF NOT EXISTS signal_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at       TEXT    NOT NULL,
                ticker          TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                confidence      REAL,
                conflict        INTEGER NOT NULL DEFAULT 0,
                regime          TEXT,
                n_votes_buy     INTEGER DEFAULT 0,
                n_votes_sell    INTEGER DEFAULT 0,
                n_votes_hold    INTEGER DEFAULT 0,
                strategy_votes  TEXT,
                entry_price     REAL,
                vol_20d         REAL,
                adx             REAL,
                sma200_dist     REAL,
                rsi             REAL,
                momentum_20d    REAL
            );

            CREATE INDEX IF NOT EXISTS idx_signal_log_ticker ON signal_log(ticker);
            CREATE INDEX IF NOT EXISTS idx_signal_log_date   ON signal_log(logged_at);

            CREATE TABLE IF NOT EXISTS run_summary (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at               TEXT    NOT NULL,
                positions_opened        INTEGER NOT NULL DEFAULT 0,
                positions_closed        INTEGER NOT NULL DEFAULT 0,
                stops_triggered         INTEGER NOT NULL DEFAULT 0,
                holds_expired           INTEGER NOT NULL DEFAULT 0,
                circuit_breaker_blocked INTEGER NOT NULL DEFAULT 0,
                aggregate_exposure_pct  REAL,
                max_single_position_pct REAL,
                total_unrealized_pnl    REAL,
                today_realized_pnl      REAL
            );

            CREATE INDEX IF NOT EXISTS idx_run_summary_date ON run_summary(logged_at);
        """)
        conn.commit()
        # Migration: add realized_pnl column if it doesn't exist (for existing DBs)
        try:
            conn.execute("ALTER TABLE trades ADD COLUMN realized_pnl REAL NOT NULL DEFAULT 0.0")
            conn.commit()
        except Exception:
            pass
        # Migration: add S1 exit-context columns to positions if missing (for existing DBs)
        for col_def in (
            "stop_price REAL",
            "expected_hold_days INTEGER",
            "risk_bucket TEXT",
            "entry_confidence REAL",
        ):
            try:
                conn.execute(f"ALTER TABLE positions ADD COLUMN {col_def}")
                conn.commit()
            except Exception:
                pass


def save_signal_log(
    signals: Dict[str, Any],
    regime: Optional[str] = None,
    regime_features: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Persist reconciled signal feature vectors to signal_log for A3 meta-labeler training.

    Args:
        signals: dict of {ticker: ReconciledSignal} from reconciler.
        regime: current regime string (e.g. 'bear_trend').
        regime_features: dict with keys vol_20d, adx, sma200_dist, rsi, momentum_20d
                         computed from SPY OHLCV (same for all tickers per run).
        db_path: override DB path; defaults to cfg-controlled path passed by caller.
    """
    db_path = db_path or _DEFAULT_DB
    init_db(db_path)
    logged_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    feats = regime_features or {}

    rows = []
    for ticker, sig in signals.items():
        votes: Dict[str, str] = getattr(sig, "votes", {}) or {}
        n_buy = sum(1 for v in votes.values() if str(v).upper() == "BUY")
        n_sell = sum(1 for v in votes.values() if str(v).upper() == "SELL")
        n_hold = sum(1 for v in votes.values() if str(v).upper() == "HOLD")
        rows.append((
            logged_at,
            ticker,
            str(getattr(sig, "action", "HOLD")),
            getattr(sig, "confidence", None),
            1 if getattr(sig, "conflict", False) else 0,
            regime,
            n_buy,
            n_sell,
            n_hold,
            json.dumps(votes) if votes else None,
            getattr(sig, "price", None),
            feats.get("vol_20d"),
            feats.get("adx"),
            feats.get("sma200_dist"),
            feats.get("rsi"),
            feats.get("momentum_20d"),
        ))

    if not rows:
        return
    with _conn(db_path) as conn:
        conn.executemany(
            "INSERT INTO signal_log "
            "(logged_at, ticker, action, confidence, conflict, regime, "
            "n_votes_buy, n_votes_sell, n_votes_hold, strategy_votes, entry_price, "
            "vol_20d, adx, sma200_dist, rsi, momentum_20d) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def save_run_summary(
    positions_opened: int = 0,
    positions_closed: int = 0,
    stops_triggered: int = 0,
    holds_expired: int = 0,
    circuit_breaker_blocked: int = 0,
    aggregate_exposure_pct: Optional[float] = None,
    max_single_position_pct: Optional[float] = None,
    total_unrealized_pnl: Optional[float] = None,
    today_realized_pnl: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Persist per-run accumulation/observability counters (§ 10 I3).

    Called once at the end of each signal_generation cycle so drift (exposure
    creeping up, stops never triggering, etc.) is visible in the DB rather than
    only in scrollback logs.
    """
    db_path = db_path or _DEFAULT_DB
    init_db(db_path)
    logged_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO run_summary "
            "(logged_at, positions_opened, positions_closed, stops_triggered, holds_expired, "
            "circuit_breaker_blocked, aggregate_exposure_pct, max_single_position_pct, "
            "total_unrealized_pnl, today_realized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                logged_at, positions_opened, positions_closed, stops_triggered, holds_expired,
                circuit_breaker_blocked, aggregate_exposure_pct, max_single_position_pct,
                total_unrealized_pnl, today_realized_pnl,
            ),
        )
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
        stop_price: Optional[float] = None,
        expected_hold_days: Optional[int] = None,
        risk_bucket: Optional[str] = None,
        entry_confidence: Optional[float] = None,
    ) -> None:
        """Insert or update a position.

        S1 note: stop_price / expected_hold_days / risk_bucket / entry_confidence are
        entry-time exit-context fields. When not explicitly passed (e.g. on a partial
        SELL update), COALESCE preserves whatever was previously stored rather than
        wiping it out. Also fixes a prior bug where entry_price was never updated on
        conflict — DCA average-cost recalculation was silently discarded.
        """
        with _conn(self.db_path) as conn:
            conn.execute("""
                INSERT INTO positions
                    (ticker, strategy, shares, entry_price, current_price, entry_time, unrealized_pnl, realized_pnl,
                     stop_price, expected_hold_days, risk_bucket, entry_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, strategy) DO UPDATE SET
                    shares             = excluded.shares,
                    entry_price        = excluded.entry_price,
                    current_price      = excluded.current_price,
                    unrealized_pnl     = excluded.unrealized_pnl,
                    realized_pnl       = excluded.realized_pnl,
                    stop_price         = COALESCE(excluded.stop_price, positions.stop_price),
                    expected_hold_days = COALESCE(excluded.expected_hold_days, positions.expected_hold_days),
                    risk_bucket        = COALESCE(excluded.risk_bucket, positions.risk_bucket),
                    entry_confidence   = COALESCE(excluded.entry_confidence, positions.entry_confidence)
            """, (ticker, strategy, shares, entry_price, current_price,
                  datetime.utcnow().isoformat(), unrealized_pnl, realized_pnl,
                  stop_price, expected_hold_days, risk_bucket, entry_confidence))
            conn.commit()

    # Backward-compat alias used by AlpacaExecutor (previously called a
    # non-existent `update_position` method — see § 10 S1 fix).
    def update_position(self, *args, **kwargs) -> None:
        self.upsert_position(*args, **kwargs)

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

    def close_position(self, ticker: str, strategy: str, realized_pnl: float = 0.0) -> None:
        with _conn(self.db_path) as conn:
            conn.execute("""
                UPDATE positions SET shares=0, unrealized_pnl=0, realized_pnl=realized_pnl+?,
                       stop_price=NULL, expected_hold_days=NULL, risk_bucket=NULL, entry_confidence=NULL
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
        realized_pnl: float = 0.0,
        reconciled_from: Optional[Dict] = None,
    ) -> None:
        with _conn(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades
                    (ticker, strategy, action, shares, price, confidence, realized_pnl, executed_at, reconciled_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, strategy, action, shares, price, confidence, realized_pnl,
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

    def today_realized_pnl(self) -> float:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with _conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT SUM(realized_pnl) FROM trades WHERE action='SELL' AND executed_at LIKE ?",
                (f"{today}%",),
            ).fetchone()
        return float(row[0] or 0.0)

    def total_unrealized_pnl(self) -> float:
        with _conn(self.db_path) as conn:
            row = conn.execute("SELECT SUM(unrealized_pnl) FROM positions WHERE shares>0").fetchone()
        return float(row[0] or 0.0)
