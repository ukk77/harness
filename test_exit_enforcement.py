"""
Behavioral smoke tests for Master Spec § 10 I1.

These exist because Phase 2 items #16 ("enrich ReconciledSignal with stop/
horizon metadata") and #18 ("circuit breaker/kill switch") were marked "Done"
in the roadmap while the actual runtime behavior was a no-op — metadata was
computed then discarded, and the bear-regime circuit breaker was a literal
`pass`. These tests assert BEHAVIOR (a breached stop actually closes the
position; the circuit breaker actually demotes entries), not just that a
field or config flag exists.

Run directly (no pytest required):
    python test_exit_enforcement.py
"""
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from harness.adapters.base import HarnessSignal
from harness.reconciler import ReconciledSignal
from harness.orchestrator import apply_circuit_breaker, should_trip_bear_vol_circuit_breaker
from harness.exit_sweep import sweep_exits
from harness.paper_trading.db import HarnessTradingDB
from harness.config import HarnessConfig

PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def _make_signal(ticker: str, action: str, confidence: float = 0.8, source: str = "mr") -> HarnessSignal:
    return HarnessSignal(
        ticker=ticker, timestamp=datetime.now(), action=action,
        confidence=confidence, source=source, price=100.0,
    )


def _make_reconciled(ticker: str, action: str, price: float, confidence: float = 0.8) -> ReconciledSignal:
    return ReconciledSignal(
        ticker=ticker, timestamp=datetime.now(), action=action, confidence=confidence,
        price=price, suggested_shares=None, suggested_stop_pct=None, expected_hold_days=None,
        risk_bucket=None, strategy_sources=["mr"], votes={"mr": action},
        vote_confidences={"mr": confidence}, mode_used="confidence_weighted",
        conflict=False, reason="test",
    )


def _make_synthetic_ohlcv(daily_drift: float, daily_vol: float, n: int = 260, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic daily OHLCV series with a given drift/vol so we can
    deterministically exercise the bear+high-vol circuit breaker condition
    without needing live SPY data."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(daily_drift, daily_vol, n)
    close = 100 * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * 1.001, "high": close * 1.005,
        "low": close * 0.995, "close": close,
    }, index=idx)


def test_bear_vol_breaker_trips_only_on_bear_and_high_vol():
    print("\n--- test_bear_vol_breaker_trips_only_on_bear_and_high_vol ---")
    threshold = 0.22

    bear_high_vol = _make_synthetic_ohlcv(daily_drift=-0.004, daily_vol=0.03)
    tripped, reason = should_trip_bear_vol_circuit_breaker(bear_high_vol, threshold)
    check("trips on bear trend + elevated vol", tripped is True, f"reason={reason}")

    calm_bear = _make_synthetic_ohlcv(daily_drift=-0.004, daily_vol=0.003)
    tripped, reason = should_trip_bear_vol_circuit_breaker(calm_bear, threshold)
    check("does NOT trip on ordinary (calm) bear trend", tripped is False, f"reason={reason}")

    volatile_bull = _make_synthetic_ohlcv(daily_drift=0.004, daily_vol=0.03)
    tripped, reason = should_trip_bear_vol_circuit_breaker(volatile_bull, threshold)
    check("does NOT trip on high-vol bull trend", tripped is False, f"reason={reason}")


def test_circuit_breaker_demotes_buy_short_preserves_sell():
    print("\n--- test_circuit_breaker_demotes_buy_short_preserves_sell ---")
    results = {
        "AAPL": [_make_signal("AAPL", "BUY")],
        "MSFT": [_make_signal("MSFT", "SHORT")],
        "TSLA": [_make_signal("TSLA", "SELL")],
        "NVDA": [_make_signal("NVDA", "HOLD")],
    }
    blocked = apply_circuit_breaker(results)
    check("blocked count == 2 (BUY + SHORT)", blocked == 2, f"got {blocked}")
    check("AAPL BUY demoted to HOLD", results["AAPL"][0].action == "HOLD")
    check("AAPL confidence zeroed", results["AAPL"][0].confidence == 0.0)
    check("AAPL reason tagged with circuit_breaker", "circuit_breaker" in results["AAPL"][0].reason)
    check("MSFT SHORT demoted to HOLD", results["MSFT"][0].action == "HOLD")
    check("TSLA SELL preserved (exits not blocked)", results["TSLA"][0].action == "SELL")
    check("NVDA HOLD unaffected", results["NVDA"][0].action == "HOLD")


def test_stop_breach_forces_sell_and_closes_position():
    print("\n--- test_stop_breach_forces_sell_and_closes_position ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "test_harness.db"
        db = HarnessTradingDB(str(db_path))
        # Open a position with an entry-time stop_price of 95.0
        db.upsert_position(
            ticker="AAPL", strategy="harness", shares=10, entry_price=100.0,
            current_price=100.0, unrealized_pnl=0.0, stop_price=95.0,
        )
        cfg = HarnessConfig(paper_db_path=str(db_path))

        # Price has dropped below the stop
        reconciled = {"AAPL": _make_reconciled("AAPL", "HOLD", price=90.0)}
        raw_signals = {"AAPL": [_make_signal("AAPL", "HOLD")]}
        counts = sweep_exits(reconciled, raw_signals, cfg)

        check("stops_triggered == 1", counts["stops_triggered"] == 1, f"got {counts}")
        check("reconciled action forced to SELL", reconciled["AAPL"].action == "SELL")
        check("reconciled confidence == 1.0 (clears the gate)", reconciled["AAPL"].confidence == 1.0)
        check("reason tagged with exit_sweep", "exit_sweep" in reconciled["AAPL"].reason)

        # Simulate the executor actually closing the position and verify the
        # DB reflects a closed position afterward.
        db.close_position("AAPL", "harness", realized_pnl=(90.0 - 100.0) * 10)
        pos = db.get_position("AAPL", "harness")
        check("position shares == 0 after close", pos["shares"] == 0)
        check("stop_price cleared after close", pos["stop_price"] is None)


def test_hold_expiry_forces_sell():
    print("\n--- test_hold_expiry_forces_sell ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "test_harness.db"
        db = HarnessTradingDB(str(db_path))
        db.upsert_position(
            ticker="MU", strategy="harness", shares=5, entry_price=50.0,
            current_price=50.0, unrealized_pnl=0.0, expected_hold_days=5,
        )
        # Manually backdate entry_time to 10 days ago (past the 5-day hold).
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        conn.execute("UPDATE positions SET entry_time=? WHERE ticker='MU'", (old_ts,))
        conn.commit()
        conn.close()

        cfg = HarnessConfig(paper_db_path=str(db_path))
        reconciled = {"MU": _make_reconciled("MU", "HOLD", price=52.0)}
        raw_signals = {"MU": [_make_signal("MU", "HOLD")]}
        counts = sweep_exits(reconciled, raw_signals, cfg)

        check("holds_expired == 1", counts["holds_expired"] == 1, f"got {counts}")
        check("reconciled action forced to SELL", reconciled["MU"].action == "SELL")


def test_no_breach_leaves_reconciled_untouched():
    print("\n--- test_no_breach_leaves_reconciled_untouched ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "test_harness.db"
        db = HarnessTradingDB(str(db_path))
        db.upsert_position(
            ticker="AAPL", strategy="harness", shares=10, entry_price=100.0,
            current_price=105.0, unrealized_pnl=50.0, stop_price=90.0, expected_hold_days=20,
        )
        cfg = HarnessConfig(paper_db_path=str(db_path))
        reconciled = {"AAPL": _make_reconciled("AAPL", "HOLD", price=105.0)}
        raw_signals = {"AAPL": [_make_signal("AAPL", "HOLD")]}
        counts = sweep_exits(reconciled, raw_signals, cfg)

        check("no stops triggered when price above stop", counts["stops_triggered"] == 0)
        check("no holds expired when within horizon", counts["holds_expired"] == 0)
        check("reconciled action unchanged", reconciled["AAPL"].action == "HOLD")


if __name__ == "__main__":
    print("=" * 60)
    print("Master Spec § 10 I1 — Exit Enforcement Behavioral Smoke Tests")
    print("=" * 60)

    test_bear_vol_breaker_trips_only_on_bear_and_high_vol()
    test_circuit_breaker_demotes_buy_short_preserves_sell()
    test_stop_breach_forces_sell_and_closes_position()
    test_hold_expiry_forces_sell()
    test_no_breach_leaves_reconciled_untouched()

    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 60)

    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
