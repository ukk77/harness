"""Harness CLI — unified entry point for the trading platform.

ALL production commands run through this CLI. Individual strategy CLIs
are for development/debugging only.

Scheduled jobs (cron / Task Scheduler):
    data_collection   — refresh sentiment + risk snapshots (08:00/11:00/14:00/17:00 ET)
    signal_generation — collect signals → reconcile → allocate → execute (hourly 08:30-17:30 ET)

Orchestration:
    status            — health check all services, models, data
    report            — unified P&L + positions + strategy comparison
    positions         — open positions across all strategies
    backtest          — compare all strategies on same historical period
    logs              — tail/aggregate logs from all services
    schedule          — register Task Scheduler jobs (Windows; use cron in cloud)

Data & Models:
    data              — trigger market data pipeline refresh (daily parquet cache)
    train             — retrain all RL models
    retrain-check     — check model degradation, retrain if needed

Usage (from trading/ root):
    python -m harness.cli status
    python -m harness.cli data [--interval 1d|1h|both] [--force-full]
    python -m harness.cli data_collection
    python -m harness.cli signal_generation [--dry-run] [--ticker AAPL]
    python -m harness.cli train [--ticker AAPL] [--timesteps 100000]
    python -m harness.cli retrain-check
    python -m harness.cli report
    python -m harness.cli positions
    python -m harness.cli backtest [--days 365]
    python -m harness.cli logs [--lines 100]
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


# ── Ticker → Company name lookup (used by sentiment /api/analyze) ────────────
_COMPANY_NAMES: dict = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation", "AMD": "Advanced Micro Devices", "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.", "NFLX": "Netflix Inc.",
    "UBER": "Uber Technologies", "PLTR": "Palantir Technologies", "ASML": "ASML Holding",
    "AVGO": "Broadcom Inc.", "LITE": "Lumentum Holdings", "MU": "Micron Technology",
    "NVTS": "Navitas Semiconductor", "SMCI": "Super Micro Computer",
    "JPM": "JPMorgan Chase", "V": "Visa Inc.", "MA": "Mastercard Inc.",
    "BRK.B": "Berkshire Hathaway", "XLF": "Financial Select Sector SPDR",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "BLK": "BlackRock Inc.",
    "LLY": "Eli Lilly and Company", "UNH": "UnitedHealth Group", "JNJ": "Johnson & Johnson",
    "MRK": "Merck & Co.", "XLV": "Health Care Select Sector SPDR",
    "ABBV": "AbbVie Inc.", "GILD": "Gilead Sciences",
    "CAT": "Caterpillar Inc.", "BA": "Boeing Company", "LMT": "Lockheed Martin",
    "GE": "GE Aerospace", "NUE": "Nucor Corporation", "XLB": "Materials Select Sector SPDR",
    "FCX": "Freeport-McMoRan", "MP": "MP Materials", "RTX": "RTX Corporation",
    "XOM": "Exxon Mobil", "VST": "Vistra Corp", "GLD": "SPDR Gold Shares",
    "XLE": "Energy Select Sector SPDR", "EQT": "EQT Corporation",
    "KMI": "Kinder Morgan", "WMB": "Williams Companies",
    "USAR": "US AR ETF", "UUUU": "Energy Fuels Inc.",
    "COST": "Costco Wholesale", "HD": "Home Depot", "WMT": "Walmart Inc.",
    "MCD": "McDonald's Corporation", "XLP": "Consumer Staples Select Sector SPDR",
    "BABA": "Alibaba Group", "NB": "NioCorp Developments",
    "COIN": "Coinbase Global", "MARA": "MARA Holdings", "MSTR": "MicroStrategy",
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust", "IWM": "iShares Russell 2000 ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF", "SQQQ": "ProShares UltraPro Short QQQ",
    "VIX": "CBOE Volatility Index", "XLU": "Utilities Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR", "XLK": "Technology Select Sector SPDR",
    "GEV": "GE Vernova",
}


def _company_name(ticker: str) -> str:
    """Return company name for ticker, falling back to '<ticker> Inc.' if unknown."""
    return _COMPANY_NAMES.get(ticker.upper(), f"{ticker} Inc.")


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
            resp = None
            last_error = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        f"{cfg.sentiment_api_url}/api/analyze",
                        json={"ticker": ticker, "company_name": _company_name(ticker)},
                        timeout=60,
                    )
                    last_error = None
                    break
                except requests.RequestException as e:
                    last_error = e
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
            if last_error is not None:
                raise last_error
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
        print(f"  Synced {synced} position(s) from Alpaca -> paper DB")
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


# ── Data command ─────────────────────────────────────────────────────────────

def cmd_data(args) -> None:
    """Trigger market data pipeline refresh (incremental parquet update)."""
    log = _setup_logging("data")
    import subprocess

    interval = getattr(args, "interval", "both")
    force_full = getattr(args, "force_full", False)
    tickers = getattr(args, "tickers", None)

    cmd = [sys.executable, "-m", "data_pipeline.data_ingestion",
           "--interval", interval]
    if force_full:
        cmd.append("--force-full")
    if tickers:
        cmd += ["--tickers"] + tickers

    print(f"  Running data pipeline — interval={interval} force={force_full}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  Data pipeline complete in {elapsed:.1f}s")
        log.info("Data pipeline complete in %.1fs", elapsed)
    else:
        print(f"  Data pipeline FAILED (exit {result.returncode})")
        log.error("Data pipeline failed with exit %d", result.returncode)


# ── Train command ─────────────────────────────────────────────────────────────

def cmd_train(args) -> None:
    """Retrain RL models — all tickers or a single one."""
    log = _setup_logging("train")
    from rl_strategy.agent.train import train_single_ticker, train_all_tickers
    from harness.config import get_config

    ticker = getattr(args, "ticker", None)
    timesteps = getattr(args, "timesteps", 100_000)
    cfg = get_config()

    if ticker:
        tickers = [ticker.upper()]
    else:
        tickers = cfg._rl_tickers

    print(f"  Training {len(tickers)} RL model(s) — {timesteps:,} timesteps each")
    t0 = time.time()
    for t in tickers:
        print(f"    Training {t}...", end="", flush=True)
        try:
            train_single_ticker(t, timesteps=timesteps)
            print(" done")
            log.info("Trained %s", t)
        except Exception as e:
            print(f" FAILED: {e}")
            log.error("Train failed for %s: %s", t, e)
    print(f"  Training complete in {time.time()-t0:.1f}s")


# ── Retrain-check command ─────────────────────────────────────────────────────

def cmd_retrain_check(args) -> None:
    """Check all RL models for performance degradation; retrain if needed."""
    log = _setup_logging("retrain_check")
    import json
    from harness.config import get_config

    cfg = get_config()
    results_dir = Path(__file__).resolve().parents[1] / "results"
    models_dir = Path(cfg.models_dir)
    degradation_threshold = 0.15  # 15% P&L drop triggers retrain

    needs_retrain = []
    print(f"  Checking {len(cfg._rl_tickers)} RL models for degradation...")
    for ticker in cfg._rl_tickers:
        eval_file = results_dir / f"{ticker}_evaluation.json"
        backtest_file = results_dir / f"{ticker}_backtest.json"
        model_file = models_dir / f"{ticker}_ppo.zip"

        if not model_file.exists():
            print(f"    {ticker:<8} NO MODEL — will train")
            needs_retrain.append(ticker)
            continue

        if not eval_file.exists() or not backtest_file.exists():
            print(f"    {ticker:<8} NO EVAL DATA — skip")
            continue

        try:
            eval_data = json.loads(eval_file.read_text())
            bt_data = json.loads(backtest_file.read_text())
            train_mean = eval_data.get("metrics", {}).get("mean_return", 0)
            bt_mean = bt_data.get("trade_metrics", {}).get("mean_trade_pnl", 0)
            # Simple heuristic: if latest backtest mean return is 15%+ below evaluation
            ratio = (bt_mean / train_mean) if train_mean and abs(train_mean) > 0.01 else 1.0
            degraded = ratio < (1 - degradation_threshold)
            status = "DEGRADED" if degraded else "OK"
            print(f"    {ticker:<8} {status}  train={train_mean:.4f} bt={bt_mean:.4f} ratio={ratio:.2f}")
            if degraded:
                needs_retrain.append(ticker)
        except Exception as e:
            print(f"    {ticker:<8} CHECK FAILED: {e}")

    if needs_retrain:
        print(f"\n  Retraining {len(needs_retrain)} model(s): {needs_retrain}")
        from rl_strategy.agent.train import train_single_ticker
        for t in needs_retrain:
            print(f"    Retraining {t}...", end="", flush=True)
            try:
                train_single_ticker(t)
                print(" done")
                log.info("Retrained %s", t)
            except Exception as e:
                print(f" FAILED: {e}")
                log.error("Retrain failed %s: %s", t, e)
    else:
        print("  All models healthy — no retraining needed.")


# ── Logs command ──────────────────────────────────────────────────────────────

def cmd_logs(args) -> None:
    """Tail and aggregate recent log lines from all harness log files."""
    lines = getattr(args, "lines", 50)
    logs_dir = Path(__file__).resolve().parents[1] / "logs"

    log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        print("No log files found in logs/")
        return

    print(f"\n  Last {lines} lines per log file  ({len(log_files)} files)\n")
    for lf in log_files[:6]:  # cap at 6 most recent files
        print(f"  {'='*60}")
        print(f"  {lf.name}")
        print(f"  {'='*60}")
        try:
            content = lf.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in content[-lines:]:
                print(f"  {line}")
        except Exception as e:
            print(f"  [error reading file: {e}]")
        print()


# ── Backtest command ─────────────────────────────────────────────────────────

def cmd_backtest(args) -> None:
    """Run per-strategy backtests and compare all reconciliation modes."""
    log = _setup_logging("backtest")
    from harness.backtest import HarnessBacktester

    days = getattr(args, "days", 180)
    strats = getattr(args, "strategy", None)
    strats = [strats] if strats else None

    _section(f"HARNESS BACKTEST — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Period   : last {days} days")
    print(f"  Capital  : $100,000")
    print(f"  Strategies: {', '.join(strats or ['rl','mr','tf','vb']).upper()}")
    print()

    bt = HarnessBacktester(lookback_days=days)
    t0 = time.time()
    report = bt.run(strategies=strats)
    elapsed = time.time() - t0

    print()
    print(f"{'='*90}")
    print("STRATEGY COMPARISON")
    print(f"{'='*90}")
    print(f"  {'Strategy':<10} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>8} "
          f"{'WinRate%':>10} {'P&L ($)':>11} {'Alpha%':>8} {'Trades':>8} {'Status'}")
    print("  " + "-"*87)
    for r in sorted(report.strategy_results, key=lambda x: x.sharpe, reverse=True):
        star = " [*]" if r.strategy == report.best_strategy else ""
        if r.error:
            print(f"  {r.strategy.upper():<10} {'ERROR':>7}  {r.error[:50]}")
        else:
            alpha_str = f"{r.alpha:>+7.1f}" if r.alpha != 0.0 else f"{'N/A':>7}"
            print(
                f"  {r.strategy.upper():<10} {r.sharpe:>7.2f} {r.sortino:>8.2f} {r.cagr:>+7.1f} "
                f"{r.max_drawdown:>8.1f} {r.win_rate:>10.1f} {r.pnl:>+11,.0f} {alpha_str} {r.num_trades:>8}{star}"
            )
    print()

    if report.reconciliation_results:
        print(f"{'='*72}")
        print("RECONCILIATION MODE COMPARISON")
        print(f"{'='*72}")
        print(f"  {'Mode':<24} {'Acted':>6} {'Held':>6} {'Conflicts':>10} {'Consensus':>10} {'Est.Sharpe':>12}")
        print("  " + "-"*65)
        for r in sorted(report.reconciliation_results, key=lambda x: x.estimated_sharpe, reverse=True):
            star = " [*]" if r.mode == report.best_recon_mode else ""
            print(
                f"  {r.mode:<24} {r.tickers_acted:>6} {r.tickers_held:>6} "
                f"{r.conflicts_blocked:>10} {r.consensus_count:>10} {r.estimated_sharpe:>12.3f}{star}"
            )
        print()

    print(f"{'='*72}")
    print("RECOMMENDATION")
    print(f"{'='*72}")
    print(f"  {report.recommendation}")
    print()
    print(f"  Completed in {elapsed:.1f}s  |  Report saved to results/harness_backtest_*.json")
    log.info("Backtest complete in %.1fs — best=%s recon=%s",
             elapsed, report.best_strategy, report.best_recon_mode)


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Trading Platform Harness — unified orchestration layer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Health check all services, models, and data")

    p_data = sub.add_parser("data", help="Trigger market data pipeline refresh (parquet cache)")
    p_data.add_argument("--interval", choices=["1d", "1h", "both"], default="both",
                        help="Interval to update (default: both)")
    p_data.add_argument("--force-full", action="store_true",
                        help="Re-download full history instead of incremental")
    p_data.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker list")

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

    p_train = sub.add_parser("train", help="Retrain RL models (all tickers or one)")
    p_train.add_argument("--ticker", default=None, help="Single ticker (default: all)")
    p_train.add_argument("--timesteps", type=int, default=100_000,
                         help="Training timesteps per model (default: 100000)")

    sub.add_parser("retrain-check",
                   help="Check all RL models for degradation, retrain if needed")

    p_logs = sub.add_parser("logs", help="Tail recent log files from all services")
    p_logs.add_argument("--lines", type=int, default=50,
                        help="Lines per file to show (default: 50)")

    p_bt = sub.add_parser("backtest", help="Run per-strategy backtests and compare reconciliation modes")
    p_bt.add_argument("--days", type=int, default=180, help="Lookback period in days (default: 180)")
    p_bt.add_argument("--strategy", choices=["rl", "mr", "tf", "vb"], default=None,
                      help="Run only one strategy (default: all 4)")

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
        "data": cmd_data,
        "data_collection": cmd_data_collection,
        "signal_generation": cmd_signal_generation,
        "run": cmd_signal_generation,   # legacy alias
        "report": cmd_report,
        "positions": cmd_positions,
        "schedule": cmd_schedule,
        "backtest": cmd_backtest,
        "train": cmd_train,
        "retrain-check": cmd_retrain_check,
        "logs": cmd_logs,
    }
    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
