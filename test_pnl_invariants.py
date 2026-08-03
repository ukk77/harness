"""
P&L / exit invariant integration test (Master Spec § 10 I2).

Runs a mini signal cycle — reconcile -> exit-sweep -> execute — against a
temp harness DB and asserts platform-level invariants that should hold after
every run:
  1. sell_count > 0 when a stop-breached position exists (exits actually fire)
  2. every open position's market value <= max_position_pct * total_capital
  3. no position is ever added to unconditionally while deeply underwater
     without a qualifying confidence override (S2 DCA guard holds)

This is the single test that would have caught the "DCA into losers" and
"stop metadata computed but never enforced" issues immediately, rather than
relying on live paper-trading logs to reveal accumulation drift after the fact.

Run directly (no pytest required):
    python test_pnl_invariants.py
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.adapters.base import HarnessSignal
from harness.reconciler import SignalReconciler
from harness.exit_sweep import sweep_exits
from harness.executor import PaperExecutor
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


def _sig(ticker, action, confidence, price, source="mr"):
    return HarnessSignal(
        ticker=ticker, timestamp=datetime.now(), action=action,
        confidence=confidence, source=source, price=price,
    )


def test_mini_signal_cycle_invariants():
    print("\n--- test_mini_signal_cycle_invariants ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "test_harness.db"
        cfg = HarnessConfig(
            paper_db_path=str(db_path),
            total_capital=100_000.0,
            max_position_pct=0.10,
            min_confidence_to_act=0.55,
            alpaca_mirror_enabled=False,
        )
        db = HarnessTradingDB(str(db_path))

        # Seed: MU position deeply underwater with a stop price already breached.
        db.upsert_position(
            ticker="MU", strategy="harness", shares=50, entry_price=120.0,
            current_price=120.0, unrealized_pnl=0.0, stop_price=110.0,
            entry_confidence=0.9,
        )

        # Raw signals for this cycle: MU's strategies still say BUY (would normally
        # DCA into the loser) and AAPL has a fresh, healthy BUY.
        raw_signals = {
            "MU": [_sig("MU", "BUY", 0.6, price=95.0)],   # price below stop=110
            "AAPL": [_sig("AAPL", "BUY", 0.7, price=180.0)],
        }

        reconciler = SignalReconciler(cfg)
        reconciled = reconciler.reconcile_all(raw_signals)

        # Exit-sweep should override MU to a forced SELL regardless of the BUY vote.
        counts = sweep_exits(reconciled, raw_signals, cfg)
        check("exit_sweep triggered a stop on MU", counts["stops_triggered"] == 1, f"got {counts}")
        check("MU reconciled action is SELL after sweep", reconciled["MU"].action == "SELL")

        executor = PaperExecutor(cfg)
        sell_count = 0
        buy_count = 0
        for ticker, rec in reconciled.items():
            if rec.action == "HOLD" or rec.confidence < cfg.min_confidence_to_act:
                continue
            if executor.execute(rec, capital=cfg.total_capital):
                if rec.action == "SELL":
                    sell_count += 1
                elif rec.action == "BUY":
                    buy_count += 1

        # ── Invariant 1: exits actually fire ──────────────────────────────────
        check("sell_count > 0", sell_count > 0, f"sell_count={sell_count}")

        # ── Invariant 2: MU's DCA-into-loser BUY was blocked, not executed ────
        # (MU was already forced to SELL by the sweep, so it can never reach the
        # BUY branch this cycle — but we additionally verify the position was
        # actually closed, not silently left open with a phantom add.)
        mu_pos = db.get_position("MU", "harness")
        check("MU position closed (shares == 0) after forced SELL", mu_pos["shares"] == 0, f"got {mu_pos}")

        # ── Invariant 3: every open position respects max_position_pct ───────
        all_positions = db.get_all_positions()
        cap_limit = cfg.total_capital * cfg.max_position_pct
        violations = []
        for pos in all_positions:
            mkt_value = pos["shares"] * pos["current_price"]
            if mkt_value > cap_limit * 1.0001:  # small float tolerance
                violations.append((pos["ticker"], mkt_value))
        check(
            "no open position exceeds max_position_pct * capital",
            not violations,
            f"violations={violations}, cap_limit={cap_limit}",
        )


def test_dca_guard_blocks_add_to_underwater_position():
    print("\n--- test_dca_guard_blocks_add_to_underwater_position ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "test_harness.db"
        cfg = HarnessConfig(paper_db_path=str(db_path), dca_loss_guard_pct=0.05, alpaca_mirror_enabled=False)
        db = HarnessTradingDB(str(db_path))

        # Existing position underwater by 15%, high entry confidence.
        db.upsert_position(
            ticker="XLV", strategy="harness", shares=20, entry_price=100.0,
            current_price=100.0, unrealized_pnl=0.0, entry_confidence=0.9,
        )

        executor = PaperExecutor(cfg)
        from harness.reconciler import ReconciledSignal
        weak_buy = ReconciledSignal(
            ticker="XLV", timestamp=datetime.now(), action="BUY", confidence=0.6,
            price=85.0, suggested_shares=5, suggested_stop_pct=None, expected_hold_days=None,
            risk_bucket=None, strategy_sources=["mr"], votes={"mr": "BUY"},
            vote_confidences={"mr": 0.6}, mode_used="confidence_weighted",
            conflict=False, reason="test",
        )
        executed = executor.execute(weak_buy, capital=cfg.total_capital)
        check("weak-confidence add-to-loser is blocked", executed is False, f"executed={executed}")

        pos = db.get_position("XLV", "harness")
        check("XLV shares unchanged (still 20)", pos["shares"] == 20, f"got {pos['shares']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Master Spec § 10 I2 — P&L / Exit Invariant Integration Tests")
    print("=" * 60)

    test_mini_signal_cycle_invariants()
    test_dca_guard_blocks_add_to_underwater_position()

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
