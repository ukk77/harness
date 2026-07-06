"""
Standalone test for Phase 4 A4: allocator.py soft-blend (regime_probs) support.

Run directly (no pytest required):
    python test_allocator_soft_blend.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.config import HarnessConfig
from harness.allocator import CapitalAllocator
from harness.regime import Regime

PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def _empty_results_cfg(**overrides) -> HarnessConfig:
    """Config pointing at an empty results dir so _sharpe_weighted() falls
    back to equal split deterministically (no dependency on real backtest files)."""
    tmp = tempfile.mkdtemp()
    cfg = HarnessConfig(results_dir=tmp, **overrides)
    return cfg


def test_default_hard_pick_unchanged():
    print("\n--- test_default_hard_pick_unchanged ---")
    cfg = _empty_results_cfg()  # regime_soft_blend defaults to False
    allocator = CapitalAllocator(cfg)

    result_no_probs = allocator.allocate_for_regime(Regime.BULL_TREND)
    check("mode is hard-pick 'regime:bull_trend' with no probs", result_no_probs.mode == "regime:bull_trend", f"got {result_no_probs.mode}")

    # Even if probs ARE passed, soft_blend defaults to False -> still hard pick.
    probs = {Regime.BULL_TREND: 0.7, Regime.BEAR_TREND: 0.1, Regime.HIGH_VOL: 0.1, Regime.RANGE_BOUND: 0.1}
    result_with_probs_but_disabled = allocator.allocate_for_regime(Regime.BULL_TREND, regime_probs=probs)
    check(
        "mode stays hard-pick when regime_soft_blend=False even with probs given",
        result_with_probs_but_disabled.mode == "regime:bull_trend",
        f"got {result_with_probs_but_disabled.mode}",
    )
    # Allocations should be identical between these two calls (probs ignored).
    caps_a = {a.strategy: a.capital for a in result_no_probs.allocations}
    caps_b = {a.strategy: a.capital for a in result_with_probs_but_disabled.allocations}
    check("allocations identical regardless of probs when disabled", caps_a == caps_b, f"{caps_a} vs {caps_b}")


def test_soft_blend_active_when_enabled_and_probs_given():
    print("\n--- test_soft_blend_active_when_enabled_and_probs_given ---")
    cfg = _empty_results_cfg(regime_soft_blend=True)
    allocator = CapitalAllocator(cfg)

    probs = {Regime.BULL_TREND: 0.7, Regime.BEAR_TREND: 0.1, Regime.HIGH_VOL: 0.1, Regime.RANGE_BOUND: 0.1}
    result = allocator.allocate_for_regime(Regime.BULL_TREND, regime_probs=probs)
    check("mode is 'regime_soft:bull_trend' when enabled with probs", result.mode == "regime_soft:bull_trend", f"got {result.mode}")

    hard_result = CapitalAllocator(_empty_results_cfg()).allocate_for_regime(Regime.BULL_TREND)
    soft_caps = {a.strategy: a.capital for a in result.allocations}
    hard_caps = {a.strategy: a.capital for a in hard_result.allocations}
    check(
        "soft-blend allocation differs from hard-pick allocation for mixed probs",
        soft_caps != hard_caps,
        f"soft={soft_caps} hard={hard_caps}",
    )

    total = sum(a.capital for a in result.allocations)
    check("soft-blend allocations still sum to total_capital", abs(total - cfg.total_capital) < 1.0, f"got {total}")


def test_soft_blend_disabled_when_no_probs_even_if_flag_true():
    print("\n--- test_soft_blend_disabled_when_no_probs_even_if_flag_true ---")
    cfg = _empty_results_cfg(regime_soft_blend=True)
    allocator = CapitalAllocator(cfg)
    result = allocator.allocate_for_regime(Regime.BEAR_TREND, regime_probs=None)
    check(
        "flag=True but regime_probs=None -> falls back to hard-pick",
        result.mode == "regime:bear_trend",
        f"got {result.mode}",
    )


def test_max_strategy_pct_cap_enforced_in_soft_blend():
    print("\n--- test_max_strategy_pct_cap_enforced_in_soft_blend ---")
    cfg = _empty_results_cfg(regime_soft_blend=True, max_strategy_pct=0.5)
    allocator = CapitalAllocator(cfg)
    # Extreme probability concentration in BULL_TREND (tf gets 1.4x there).
    probs = {Regime.BULL_TREND: 1.0, Regime.BEAR_TREND: 0.0, Regime.HIGH_VOL: 0.0, Regime.RANGE_BOUND: 0.0}
    result = allocator.allocate_for_regime(Regime.BULL_TREND, regime_probs=probs)
    for a in result.allocations:
        check(f"strategy '{a.strategy}' weight <= max_strategy_pct", a.weight <= cfg.max_strategy_pct + 1e-6, f"got {a.weight}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 A4 — allocator.py Soft-Blend Tests")
    print("=" * 60)

    test_default_hard_pick_unchanged()
    test_soft_blend_active_when_enabled_and_probs_given()
    test_soft_blend_disabled_when_no_probs_even_if_flag_true()
    test_max_strategy_pct_cap_enforced_in_soft_blend()

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
