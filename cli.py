"""Harness CLI — unified entry point for the trading platform.

Jobs:
    data_collection   — refresh sentiment + risk snapshots for all tickers
                        Scheduled: 08:00, 11:00, 14:00, 17:00 ET
    signal_generation — collect signals → reconcile → allocate → execute
                        Scheduled: 08:30 – 17:30 ET, every hour

Other commands:
    status            — health check all services, models, data
    report            — unified P&L + positions + strategy comparison
    positions         — open positions across all strategy DBs
    schedule          — register both Task Scheduler jobs (Windows)

Usage (from trading/ root):
    python -m harness.cli status
    python -m harness.cli data_collection
    python -m harness.cli signal_generation [--dry-run] [--ticker AAPL]
    python -m harness.cli report
    python -m harness.cli positions
    python -m harness.cli schedule
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

_TRADING_ROOT = Path(__file__).resolve().parents[1]
_RISK_BACKEND = _TRADING_ROOT / "risk_calculator" / "backend"
for _p in [str(_TRADING_ROOT), str(_RISK_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _setup_logging(job: str) -> logging.Logger:
    """Configure root + file logging for a job run."""
    from harness.config import get_config
    logs_dir = Path(get_config().logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{job}_{ts}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger("harness")
    logger.info("Log file: %s", log_file)
    return logger


def _section(title: str) -> None:
    width = 72
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _step(n: int, total: int, label: str) -> None:
    print(f"\n[{n}/{total}] {label}")


# ── cmd_status ────────────────────────────────────────────────────────────────

def cmd_status(args) -> None:
    """Health check all services, models, and data."""
    log = _setup_logging("status")
    from harness.health import HealthChecker, print_health_report
    from harness.config import get_config

    cfg = get_config()
    checker = HealthChecker(cfg)
    report = checker.run()
    print_health_report(report)
    log.info("Health check complete — ok=%s", report.ok)
    sys.exit(0 if report.ok else 1)


# ── cmd_data_collection ───────────────────────────────────────────────────────

def cmd_data_collection(args) -> None:
    """Collect sentiment + risk snapshots for every ticker in the union universe.

    Scheduled: 08:00, 11:00, 14:00, 17:00 ET daily.
    """
    log = _setup_logging("data_collection")
    from harness.config import get_config

    cfg = get_config()
    tickers = cfg.tickers
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _section(f"DATA COLLECTION — {ts_str}  ({len(tickers)} tickers)")
    log.info("Starting data_collection for %d tickers", len(tickers))

    # ── Step 1: Sentiment batch ───────────────────────────────────────────────
    _step(1, 2, f"Sentiment snapshots ({len(tickers)} tickers)")
    sent_ok = 0
    sent_fail = 0
    for i, ticker in enumerate(tickers, 1):
        pct = i / len(tickers) * 100
        print(f"\r  {i:>3}/{len(tickers)}  [{pct:5.1f}%]  {ticker:<8}", end="", flush=True)
        try:
            import requests
            resp = requests.post(
                f"{cfg.sentiment_api_url}/api/analyze",
                json={"ticker": ticker, "max_articles": 10},
                timeout=30,
            )
            if resp.status_code == 200:
                sent_ok += 1
                log.debug("Sentiment OK: %s", ticker)
            else:
                sent_fail += 1
                log.warning("Sentiment HTTP %d: %s", resp.status_code, ticker)
        except Exception as e:
            sent_fail += 1
            log.warning("Sentiment error %s: %s", ticker, e)
    print(f"\r  Sentiment: {sent_ok} OK  {sent_fail} failed ({len(tickers)} tickers)          ")
    log.info("Sentiment collection: %d ok, %d failed", sent_ok, sent_fail)

    # ── Step 2: Risk batch ────────────────────────────────────────────────────
    _step(2, 2, f"Risk snapshots ({len(tickers)} tickers)")
    risk_ok = 0
    risk_fail = 0
    for i, ticker in enumerate(tickers, 1):
        pct = i / len(tickers) * 100
        print(f"\r  {i:>3}/{len(tickers)}  [{pct:5.1f}%]  {ticker:<8}", end="", flush=True)
        try:
            import requests
            resp = requests.post(
                f"{cfg.risk_api_url}/api/risk",
                json={"ticker": ticker, "lookback_days": 252},
                timeout=60,
            )
            if resp.status_code == 200:
                risk_ok += 1
                log.debug("Risk OK: %s", ticker)
            else:
                risk_fail += 1
                log.warning("Risk HTTP %d: %s", resp.status_code, ticker)
        except Exception as e:
            risk_fail += 1
            log.warning("Risk error %s: %s", ticker, e)
    print(f"\r  Risk:      {risk_ok} OK  {risk_fail} failed ({len(tickers)} tickers)          ")
    log.info("Risk collection: %d ok, %d failed", risk_ok, risk_fail)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"  DATA COLLECTION COMPLETE")
    print(f"  Sentiment: {sent_ok}/{len(tickers)} OK")
    print(f"  Risk:      {risk_ok}/{len(tickers)} OK")
    log.info("data_collection complete")


# ── cmd_signal_generation ─────────────────────────────────────────────────────

def cmd_signal_generation(args) -> None:
    """Full signal cycle: collect → reconcile → allocate → execute → report.

    Each strategy runs against its own ticker universe.
    Sends signals to Alpaca (live) or paper DB (paper mode).
    Scheduled: hourly 08:30–17:30 ET.
    """
    log = _setup_logging("signal_generation")
    from harness.config import get_config
    from harness.orchestrator import Orchestrator
    from harness.reconciler import SignalReconciler
    from harness.allocator import CapitalAllocator
    from harness.executor import PaperExecutor, AlpacaExecutor
    from harness.reporter import Reporter

    cfg = get_config()
    if args.mode:
        cfg.reconciliation_mode = args.mode

    tickers_override = [args.ticker.upper()] if getattr(args, "ticker", None) else None
    dry_run = getattr(args, "dry_run", False)
    mode_label = "DRY-RUN" if dry_run else cfg.execution_mode.upper()
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    universe_str = (
        f"  RL:  {len(getattr(cfg, '_rl_tickers', []))} tickers\n"
        f"  MR:  {len(getattr(cfg, '_mr_tickers', []))} tickers\n"
        f"  TF:  {len(getattr(cfg, '_tf_tickers', []))} tickers\n"
        f"  VB:  {len(getattr(cfg, '_vb_tickers', []))} tickers\n"
        f"  Union: {len(cfg.tickers)} unique tickers"
    )

    _section(f"SIGNAL GENERATION — {ts_str}  [{mode_label}]")
    if tickers_override:
        print(f"  Ticker override: {tickers_override}")
    else:
        print(universe_str)
    print(f"  Reconciliation: {cfg.reconciliation_mode}")
    log.info("signal_generation start — mode=%s dry_run=%s", mode_label, dry_run)

    STEPS = 5

    # ── Step 1: Collect ───────────────────────────────────────────────────────
    _step(1, STEPS, "Collecting signals from all strategies (parallel)")
    t0 = time.time()
    orchestrator = Orchestrator(cfg)
    raw_signals = orchestrator.run(tickers=tickers_override)
    log.info("Step 1 done in %.1fs — %d tickers with signals",
             time.time() - t0, len(raw_signals))

    # ── Step 2: Reconcile ─────────────────────────────────────────────────────
    _step(2, STEPS, "Reconciling signals")
    t0 = time.time()
    reconciler = SignalReconciler(cfg)
    reconciled = reconciler.reconcile_all(raw_signals)
    actionable = sum(1 for r in reconciled.values() if r.action != "HOLD")
    conflicts  = sum(1 for r in reconciled.values() if r.conflict)
    print(f"  Actionable: {actionable}  |  Conflicts blocked: {conflicts}  |  HOLD: {len(reconciled)-actionable}")
    log.info("Step 2 done in %.1fs — %d actionable, %d conflicts",
             time.time() - t0, actionable, conflicts)

    # ── Step 3: Allocate capital ──────────────────────────────────────────────
    _step(3, STEPS, "Computing capital allocation (Sharpe-weighted)")
    t0 = time.time()
    allocator = CapitalAllocator(cfg)
    allocation = allocator.allocate(mode="sharpe_weighted")
    harness_capital = sum(
        allocation.get(s) for s in ["rl", "mr", "tf", "vb"]
    ) or cfg.total_capital
    print(f"  Total capital allocated: ${harness_capital:,.0f}")
    for s in ["rl", "mr", "tf", "vb"]:
        print(f"    {s.upper():<4}: ${allocation.get(s):>10,.0f}")
    log.info("Step 3 done in %.1fs — capital $%.0f", time.time() - t0, harness_capital)

    # ── Step 4: Execute ───────────────────────────────────────────────────────
    action_word = "Simulating (dry-run)" if dry_run else (
        "Sending to Alpaca" if cfg.execution_mode == "live" else "Recording (paper)"
    )
    _step(4, STEPS, action_word)
    t0 = time.time()

    # Sync paper DB from Alpaca before executing so positions reflect reality
    if not dry_run:
        from harness.paper_trading.db import HarnessTradingDB
        _db = HarnessTradingDB(cfg.paper_db_path)
        synced = _db.sync_from_alpaca(paper=(cfg.execution_mode != "live"))
        print(f"  Synced {synced} position(s) from Alpaca → paper DB")
        log.info("Alpaca sync: %d positions loaded into paper DB", synced)

    if cfg.execution_mode == "live" and not dry_run:
        executor = AlpacaExecutor(cfg)
    else:
        executor = PaperExecutor(cfg)

    to_execute = [
        (ticker, rec) for ticker, rec in sorted(reconciled.items())
        if rec.action != "HOLD" and rec.confidence >= cfg.min_confidence_to_act
    ]
    executed = 0
    skipped = 0
    total_exec = len(to_execute)

    for i, (ticker, rec) in enumerate(to_execute, 1):
        pct = i / max(total_exec, 1) * 100
        print(f"\r  [{i:>3}/{total_exec}]  {pct:5.1f}%  {ticker:<8} {rec.action:<4} conf={rec.confidence:.2f}   ",
              end="", flush=True)
        if dry_run:
            log.info("DRY-RUN: %s %s @ $%.2f conf=%.2f", rec.action, ticker, rec.price, rec.confidence)
            executed += 1
        else:
            try:
                if executor.execute(rec, harness_capital):
                    executed += 1
                    log.info("EXECUTED: %s %s @ $%.2f", rec.action, ticker, rec.price)
                else:
                    skipped += 1
                    log.debug("SKIPPED: %s %s (no position / insufficient capital)", rec.action, ticker)
            except Exception as e:
                skipped += 1
                log.error("EXECUTE ERROR: %s %s — %s", rec.action, ticker, e)

    if total_exec > 0:
        print()
    print(f"  Executed: {executed}  |  Skipped: {skipped}  |  Dry-run: {dry_run}")
    log.info("Step 4 done in %.1fs — executed=%d skipped=%d", time.time() - t0, executed, skipped)

    # ── Step 5: Report ────────────────────────────────────────────────────────
    _step(5, STEPS, "Generating run report")
    reporter = Reporter(cfg)
    reporter.print_signal_summary(reconciled, raw_signals)
    reporter.print_positions()
    reporter.save_run_report(reconciled, raw_signals, executed, skipped, dry_run)
    log.info("Step 5 done — report saved")

    _section("SIGNAL GENERATION COMPLETE")
    print(f"  Trades executed : {executed}")
    print(f"  Skipped/no-pos  : {skipped}")
    print(f"  Conflicts blocked: {conflicts}")
    print(f"  Mode            : {mode_label}")
    print()
    log.info("signal_generation complete — executed=%d", executed)


# ── cmd_report ────────────────────────────────────────────────────────────────

def cmd_report(args) -> None:
    """Print unified P&L, positions, recent trades, and strategy comparison."""
    _setup_logging("report")
    from harness.reporter import Reporter
    from harness.config import get_config
    reporter = Reporter(get_config())
    reporter.print_full_report()


# ── cmd_positions ─────────────────────────────────────────────────────────────

def cmd_positions(args) -> None:
    """Show all open positions across all strategy DBs."""
    _setup_logging("positions")
    from harness.reporter import Reporter
    from harness.config import get_config
    reporter = Reporter(get_config())
    reporter.print_positions()
    reporter.print_recent_trades()


# ── cmd_schedule ──────────────────────────────────────────────────────────────

def cmd_schedule(args) -> None:
    """Register both Task Scheduler jobs on Windows:
      - HarnessDataCollection  : 08:00, 11:00, 14:00, 17:00 ET daily
      - HarnessSignalGeneration: 08:30 → 17:30 ET hourly, Mon–Fri
    """
    import subprocess, textwrap
    python_exe = sys.executable

    jobs = [
        {
            "name": "HarnessDataCollection",
            "desc": "Harness — sentiment + risk data collection (4x daily)",
            "command": f"{python_exe}",
            "args": "-m harness.cli data_collection",
            "triggers": [
                ("08:00", "daily"),
                ("11:00", "daily"),
                ("14:00", "daily"),
                ("17:00", "daily"),
            ],
        },
        {
            "name": "HarnessSignalGeneration",
            "desc": "Harness — signal generation + execution (hourly 08:30–17:30)",
            "command": f"{python_exe}",
            "args": "-m harness.cli signal_generation",
            "triggers": [
                (f"{h:02d}:30", "weekday") for h in range(8, 18)
            ],
        },
    ]

    for job in jobs:
        # Build a trigger per time — Windows Task Scheduler XML supports multiple triggers
        trigger_xml = ""
        for time_str, freq in job["triggers"]:
            h, m = time_str.split(":")
            trigger_xml += f"""
      <CalendarTrigger>
        <StartBoundary>2026-01-05T{h}:{m}:00</StartBoundary>
        <Enabled>true</Enabled>
        <ScheduleByWeek>
          <WeeksInterval>1</WeeksInterval>
          <DaysOfWeek>
            <Monday/><Tuesday/><Wednesday/><Thursday/><Friday/>
          </DaysOfWeek>
        </ScheduleByWeek>
      </CalendarTrigger>"""

        xml = textwrap.dedent(f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{job['desc']}</Description>
  </RegistrationInfo>
  <Triggers>{trigger_xml}
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{job['command']}</Command>
      <Arguments>{job['args']}</Arguments>
      <WorkingDirectory>{_TRADING_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>""").strip()

        xml_path = _TRADING_ROOT / f"{job['name']}.xml"
        xml_path.write_text(xml, encoding="utf-16")
        try:
            subprocess.run(
                ["schtasks", "/Create", "/TN", job["name"], "/XML", str(xml_path), "/F"],
                check=True, capture_output=True, text=True,
            )
            print(f"  [OK] Task registered: {job['name']}")
        except subprocess.CalledProcessError as e:
            print(f"  [!]  schtasks failed for {job['name']}: {e.stderr.strip()}")
            print(f"       Import manually: schtasks /Create /TN {job['name']} /XML {xml_path} /F")

    print("\n  Both jobs registered. View in Task Scheduler → Task Scheduler Library.\n")


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Trading Platform Harness — unified orchestration layer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Health check all services, models, and data")

    sub.add_parser("data_collection",
                   help="Collect sentiment + risk snapshots (runs at 08:00/11:00/14:00/17:00 ET)")

    p_sg = sub.add_parser("signal_generation",
                           help="Signal collection → reconcile → allocate → execute (hourly 08:30–17:30 ET)")
    p_sg.add_argument("--dry-run", action="store_true",
                      help="Show signals without executing trades")
    p_sg.add_argument("--ticker", default=None,
                      help="Run for a single ticker only (overrides per-strategy universes)")
    p_sg.add_argument("--mode", default=None,
                      choices=["confidence_weighted", "majority_vote", "rl_priority", "consensus_only"],
                      help="Override reconciliation mode")

    sub.add_parser("report", help="Print unified P&L, positions, and strategy comparison")
    sub.add_parser("positions", help="Show all open positions across all strategy DBs")
    sub.add_parser("schedule", help="Register Windows Task Scheduler jobs")

    # Legacy alias: 'run' → signal_generation for backwards compatibility
    p_run = sub.add_parser("run", help="Alias for signal_generation")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--ticker", default=None)
    p_run.add_argument("--mode", default=None,
                       choices=["confidence_weighted", "majority_vote", "rl_priority", "consensus_only"])

    return parser.parse_args()


def main():
    args = _parse_args()
    dispatch = {
        "status": cmd_status,
        "data_collection": cmd_data_collection,
        "signal_generation": cmd_signal_generation,
        "run": cmd_signal_generation,   # legacy alias
        "report": cmd_report,
        "positions": cmd_positions,
        "schedule": cmd_schedule,
    }
    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
