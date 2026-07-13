"""Harness CLI — unified entry point for the trading platform.

ALL production commands run through this CLI. Individual strategy CLIs
are for development/debugging only.

Scheduled jobs (cron / Task Scheduler):
    data_collection   — refresh sentiment + risk snapshots (08:00/11:00/14:00/17:00 ET)
    signal_generation — collect signals → reconcile → allocate → execute (hourly 08:30-17:30 ET)

Orchestration:
    status            — health check all services, models, data
    dashboard         — unified health + Alpaca + positions + trades + data + RL models + regime
    report            — unified P&L + positions + strategy comparison
    positions         — open positions across all strategies
    backtest          — compare all strategies on same historical period
    logs              — tail/aggregate logs from all services
    schedule          — register Task Scheduler jobs (Windows; use cron in cloud)

User-only research:
    comprehensive TICKERS... — deep-dive analysis on tickers (auth-protected)

Data & Models:
    data              — trigger market data pipeline refresh (daily parquet cache)
    train             — retrain all RL models
    retrain-check     — check rolling Sharpe degradation, retrain if needed

Usage (from trading/ root):
    python -m harness.cli status
    python -m harness.cli dashboard [--trade-limit 10] [--max-age-hours 25] [--health-timeout 5]
    python -m harness.cli data [--interval 1d|1h|both] [--force-full]
    python -m harness.cli data_collection
    python -m harness.cli signal_generation [--dry-run] [--ticker AAPL]
    python -m harness.cli train [--ticker AAPL] [--timesteps 100000]
    python -m harness.cli retrain-check [--auto-retrain] [--min-sharpe 0.5]
    python -m harness.cli report
    python -m harness.cli positions
    python -m harness.cli backtest [--days 365]
    python -m harness.cli logs [--lines 100]
    python -m harness.cli schedule
    python -m harness.cli comprehensive AAPL [MSFT TSLA ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Set HARNESS_CONTROLLED flag to disable strategy-level paper trading execution
os.environ["HARNESS_CONTROLLED"] = "1"

from datetime import datetime
from pathlib import Path

_TRADING_ROOT = Path(__file__).resolve().parents[1]
_RISK_BACKEND = _TRADING_ROOT / "risk_calculator" / "backend"
for _p in [str(_TRADING_ROOT), str(_RISK_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that tolerates flush errors on non-standard stdout wrappers."""

    def flush(self) -> None:
        try:
            super().flush()
        except OSError:
            # Some execution environments (e.g., backgrounded pseudo-terminals)
            # raise EINVAL on flush; the message has already been emitted.
            pass


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
            _SafeStreamHandler(sys.stdout),
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


# ── cmd_dashboard ───────────────────────────────────────────────────────────────

def cmd_dashboard(args) -> None:
    """Unified dashboard: health, Alpaca, positions, trades, data, RL models, regime."""
    log = _setup_logging("dashboard")
    from harness.config import get_config
    from harness.dashboard import build_dashboard, print_dashboard, save_dashboard

    cfg = get_config()
    dashboard = build_dashboard(
        cfg=cfg,
        trade_limit=getattr(args, "trade_limit", 10),
        max_age_hours=getattr(args, "max_age_hours", 25.0),
        health_timeout=getattr(args, "health_timeout", 5.0),
    )
    print_dashboard(dashboard)
    if not getattr(args, "no_save", False):
        out_path = save_dashboard(dashboard, cfg)
        print(f"  Dashboard JSON saved: {out_path}")
    log.info("Dashboard complete")


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


# ── Service auto-start ───────────────────────────────────────────────────────

_SVC_DEFS = [
    {
        "name": "sentiment_analysis",
        "health_url_tmpl": "{sentiment_api_url}/api/health",
        "ready_url_tmpl": "{sentiment_api_url}/",
        "python": str(_TRADING_ROOT / "sentiment_analysis" / "venv" / "Scripts" / "python.exe"),
        "args": ["-m", "app.main"],
        "cwd": str(_TRADING_ROOT / "sentiment_analysis" / "backend"),
    },
    {
        "name": "risk_calculator",
        "health_url_tmpl": "{risk_api_url}/api/health",
        "ready_url_tmpl": "{risk_api_url}/",
        "python": str(_TRADING_ROOT / "risk_calculator" / "venv" / "Scripts" / "python.exe"),
        "args": ["-m", "backend.app.main"],
        "cwd": str(_TRADING_ROOT / "risk_calculator"),
    },
]


def _ensure_services_up(cfg, log: logging.Logger) -> None:
    """Start sentiment and risk services if not running; wait up to 90s for readiness."""
    import subprocess
    import requests as _req

    started = []
    for svc in _SVC_DEFS:
        health_url = svc["health_url_tmpl"].format(
            sentiment_api_url=cfg.sentiment_api_url,
            risk_api_url=cfg.risk_api_url,
        )
        ready_url = svc["ready_url_tmpl"].format(
            sentiment_api_url=cfg.sentiment_api_url,
            risk_api_url=cfg.risk_api_url,
        )
        try:
            _req.get(ready_url, timeout=20)  # any response = port is bound = service is up
            log.info("%s already up", svc["name"])
            continue
        except Exception:
            pass

        python_exe = svc["python"]
        if not Path(python_exe).exists():
            python_exe = sys.executable
            log.warning("%s venv not found, falling back to %s", svc["name"], python_exe)

        log.info("Starting %s ...", svc["name"])
        print(f"  Starting {svc['name']} ...")
        from harness.config import get_config as _get_cfg
        _logs_dir = Path(_get_cfg().logs_dir)
        _logs_dir.mkdir(parents=True, exist_ok=True)
        _svc_log = _logs_dir / f"{svc['name']}_service.log"
        _svc_out = open(_svc_log, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [python_exe] + svc["args"],
            cwd=svc["cwd"],
            stdout=_svc_out,
            stderr=_svc_out,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        started.append((svc["name"], ready_url, proc))

    if not started:
        return

    print(f"  Waiting for {len(started)} service(s) to be ready", end="", flush=True)
    deadline = time.time() + 150
    pending = list(started)
    while pending and time.time() < deadline:
        time.sleep(3)
        print(".", end="", flush=True)
        still_waiting = []
        for name, url, proc in pending:
            try:
                _req.get(url, timeout=20)  # any response = ready
                log.info("%s ready", name)
                continue
            except Exception:
                pass
            still_waiting.append((name, url, proc))
        pending = still_waiting
    print()

    for name, url, _proc in pending:
        log.warning("%s did not become ready within 150s — proceeding anyway", name)


# ── cmd_data_collection ───────────────────────────────────────────────────────

def cmd_data_collection(args) -> None:
    """Collect sentiment + risk snapshots for every ticker in the union universe.

    Scheduled: 08:00, 11:00, 14:00, 17:00 ET daily.
    """
    log = _setup_logging("data_collection")
    from harness.config import get_config

    cfg = get_config()
    _ensure_services_up(cfg, log)
    tickers = cfg.tickers
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _section(f"DATA COLLECTION — {ts_str}  ({len(tickers)} tickers)")
    log.info("Starting data_collection for %d tickers", len(tickers))

    import requests  # ensure available for both batches below

    # ── Step 1: Sentiment batch ───────────────────────────────────────────────
    _step(1, 2, f"Sentiment snapshots ({len(tickers)} tickers)")
    sent_ok = 0
    sent_fail = 0
    for i, ticker in enumerate(tickers, 1):
        pct = i / len(tickers) * 100
        print(f"\r  {i:>3}/{len(tickers)}  [{pct:5.1f}%]  {ticker:<8}", end="", flush=True)
        try:
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
            resp = None
            last_error = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        f"{cfg.risk_api_url}/api/risk",
                        json={"ticker": ticker, "company_name": _company_name(ticker), "lookback_days": 252},
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

    # ── Step 3: RAG ingest (incremental, non-blocking) ────────────────────────
    if getattr(cfg, "rag_ingest_on_collect", True):
        print("  Triggering RAG ingestion (incremental)...", end="", flush=True)
        try:
            from harness.rag_client import RAGClient
            rag = RAGClient(base_url=cfg.rag_service_url)
            result = rag.ingest()
            if result:
                total_added = result.get("total_docs_added", 0)
                print(f"\r  RAG ingest: +{total_added} docs                  ")
                log.info("RAG ingest complete: +%d docs", total_added)
            else:
                print(f"\r  RAG ingest: skipped (rag_service unavailable)    ")
                log.debug("RAG ingest skipped — rag_service not reachable")
        except Exception as _rag_exc:
            print(f"\r  RAG ingest: failed (non-blocking)                ")
            log.warning("RAG ingest failed (non-blocking): %s", _rag_exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"  DATA COLLECTION COMPLETE")
    print(f"  Sentiment: {sent_ok}/{len(tickers)} OK")
    print(f"  Risk:      {risk_ok}/{len(tickers)} OK")
    log.info("data_collection complete")


# ── cmd_ask ───────────────────────────────────────────────────────────────────

def cmd_ask(args) -> None:
    """Query the RAG market-intelligence layer with a free-text question."""
    _setup_logging("ask")
    from harness.config import get_config
    cfg = get_config()
    query = " ".join(args.query)
    ticker = getattr(args, "ticker", None)
    days = getattr(args, "days", 30)

    # Auto-detect ticker from query if not explicitly provided
    if not ticker:
        import re
        words = re.findall(r'[A-Za-z]+', query.upper())
        for word in words:
            if word in cfg.tickers:
                ticker = word
                break

    from harness.rag_client import RAGClient
    client = RAGClient(base_url=cfg.rag_service_url)

    if not client.is_up():
        print(f"  ERROR: rag_service is not reachable at {cfg.rag_service_url}")
        print("  Start it with: python -m app.main  (from rag_service/)")
        return

    from datetime import datetime, timedelta, timezone
    date_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d") if days else None

    result = client.ask(query=query, ticker=ticker, date_from=date_from)
    if not result:
        print("  rag_service returned no answer.")
        return

    print(f"\n{'═'*72}")
    print(f"  QUESTION: {query}")
    if ticker:
        print(f"  Ticker filter: {ticker}")
    print(f"{'═'*72}")
    print(result.get("answer", "(no answer)"))
    print(f"\n{'─'*72}")
    sources = result.get("sources", [])
    if sources:
        print(f"  Sources ({len(sources)}):")
        for s in sources[:5]:
            print(f"    [{s.get('source','?')}] {s.get('ticker','?')} {s.get('date','?')} — {s.get('text','')[:80]}...")
    print(f"{'─'*72}")
    print(f"  Model: {result.get('model_used', '?')}")


# ── cmd_comprehensive ─────────────────────────────────────────────────────────

def cmd_comprehensive(args) -> None:
    """Run comprehensive deep-dive analysis on one or more tickers.

    This is a user-only harness command that calls the
    /api/comprehensive endpoint on the sentiment_analysis service.
    """
    log = _setup_logging("comprehensive")
    from harness.config import get_config

    cfg = get_config()
    _ensure_services_up(cfg, log)

    tickers = [t.upper() for t in getattr(args, "tickers", [])]
    if not tickers:
        print("ERROR: At least one ticker is required.")
        print("Usage: python -m harness.cli comprehensive <TICKER> [TICKER...]")
        sys.exit(1)

    import json
    import requests

    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _section(f"COMPREHENSIVE ANALYSIS — {ts_str}  ({len(tickers)} tickers)")
    log.info("Starting comprehensive analysis for %d tickers: %s", len(tickers), tickers)

    all_reports = []
    for ticker in tickers:
        print(f"\n  Analyzing {ticker}...")
        try:
            resp = requests.post(
                f"{cfg.sentiment_api_url}/api/comprehensive",
                json={"ticker": ticker},
                timeout=120,
            )
            if resp.status_code == 503:
                print(f"  ERROR {ticker}: Service unavailable — sentiment_analysis service may be down.")
                log.error("Service unavailable for %s", ticker)
                continue
            resp.raise_for_status()
            report = resp.json()
            all_reports.append(report)
            _print_comprehensive_report(report)
            log.info("Comprehensive analysis OK: %s", ticker)
        except requests.RequestException as e:
            print(f"  ERROR {ticker}: {e}")
            log.error("Comprehensive analysis failed for %s: %s", ticker, e)

    # Save JSON report
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"comprehensive_{ts_file}.json"
    try:
        out_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
        print(f"\n  JSON report saved: {out_path}")
        log.info("Report saved to %s", out_path)
    except Exception as e:
        log.warning("Could not save report: %s", e)

    _section("COMPREHENSIVE ANALYSIS COMPLETE")


def _print_comprehensive_report(report: dict) -> None:
    """Pretty-print a comprehensive analysis report."""
    ticker = report.get("ticker", "UNKNOWN")
    current = report.get("current_data", {})
    deep = report.get("deep_research", {})
    news = report.get("news_check", {})

    width = 72
    print(f"\n  {'='*width}")
    print(f"  {ticker} — Comprehensive Report")
    print(f"  {'='*width}")

    # Current Data
    print(f"\n  [CURRENT DATA]")
    if current.get("error"):
        print(f"    Error: {current['error']}")
    else:
        price = current.get("price")
        change = current.get("change")
        change_pct = current.get("change_percent")
        price_str = f"${price:.2f}" if price is not None else "N/A"
        change_str = ""
        if change is not None and change_pct is not None:
            change_str = f" ({change:+.2f}, {change_pct:+.2f}%)"
        print(f"    Price         : {price_str}{change_str}")
        print(f"    Bid / Ask     : {current.get('bid', 'N/A')} / {current.get('ask', 'N/A')}")
        print(f"    Spread        : {current.get('spread', 'N/A')}")
        print(f"    Volume        : {current.get('volume', 'N/A')}  (vs avg {current.get('average_volume', 'N/A')})")
        print(f"    Volume ratio  : {current.get('volume_vs_average', 'N/A')}")
        print(f"    52w range     : ${current.get('fifty_two_week_low', 'N/A')} - ${current.get('fifty_two_week_high', 'N/A')}")
        print(f"    Position      : {current.get('position_in_52w_range_percent', 'N/A')}% through 52w range")
        print(f"    Market cap    : ${current.get('market_cap_millions', 'N/A')}M")

    # Deep Research
    print(f"\n  [DEEP RESEARCH]")
    if deep.get("error"):
        print(f"    Error: {deep['error']}")
    else:
        ops = deep.get("business_operations", {})
        if ops.get("company_name"):
            print(f"    Company       : {ops['company_name']}")
        if ops.get("business_summary"):
            summary = ops["business_summary"]
            print(f"    Business      : {summary[:240]}{'...' if len(summary) > 240 else ''}")

        fin = deep.get("financials", {})
        print(f"\n    Latest Q revenue : ${fin.get('latest_quarter_revenue_millions', 'N/A')}M")
        print(f"    Latest Q earnings: ${fin.get('latest_quarter_net_income_millions', 'N/A')}M")
        print(f"    Revenue YoY      : {fin.get('revenue_yoy_growth_percent', 'N/A')}%")
        print(f"    Earnings YoY     : {fin.get('earnings_yoy_growth_percent', 'N/A')}%")
        print(f"    Revenue trend    : {fin.get('revenue_trend', 'N/A')}")
        print(f"    Earnings trend   : {fin.get('earnings_trend', 'N/A')}")

        bs = deep.get("balance_sheet", {})
        print(f"\n    Debt          : ${bs.get('total_debt_millions', 'N/A')}M")
        print(f"    Cash          : ${bs.get('cash_millions', 'N/A')}M")
        print(f"    Net debt      : ${bs.get('net_debt_millions', 'N/A')}M")
        print(f"    Debt/Equity   : {bs.get('debt_to_equity', 'N/A')}")
        print(f"    Current ratio : {bs.get('current_ratio', 'N/A')}")

        metrics = deep.get("key_metrics", {})
        print(f"\n    Trailing P/E  : {metrics.get('trailing_pe', 'N/A')}")
        print(f"    Forward P/E   : {metrics.get('forward_pe', 'N/A')}")
        print(f"    PEG           : {metrics.get('peg_ratio', 'N/A')}")
        print(f"    P/B           : {metrics.get('price_to_book', 'N/A')}")
        print(f"    Profit margin : {metrics.get('profit_margin', 'N/A')}")
        print(f"    Beta          : {metrics.get('beta', 'N/A')}")

        comp = deep.get("competitive_landscape", {})
        print(f"\n    Sector        : {comp.get('sector', 'N/A')} / {comp.get('industry', 'N/A')}")
        print(f"    Size tier     : {comp.get('size_tier', 'N/A')}")
        print(f"    Assessment    : {comp.get('relative_assessment', 'N/A')}")

        growth = deep.get("future_growth", {})
        print(f"\n    Earnings growth  : {growth.get('earnings_growth', 'N/A')}")
        print(f"    Revenue growth   : {growth.get('revenue_growth', 'N/A')}")
        print(f"    Analyst target   : ${growth.get('analyst_target_mean', 'N/A')} (upside {growth.get('upside_to_target_percent', 'N/A')}%)")
        if growth.get("what_has_to_go_right"):
            print(f"    Growth drivers:")
            for item in growth["what_has_to_go_right"]:
                print(f"      - {item}")

        risks = deep.get("risks", [])
        if risks:
            print(f"\n    Key risks:")
            for risk in risks:
                print(f"      - {risk}")

    # News Check
    print(f"\n  [NEWS CHECK]")
    if news.get("error"):
        print(f"    Error: {news['error']}")
    else:
        articles = news.get("articles", [])
        print(f"    Material recent articles: {news.get('material_recent_count', 0)} / {len(articles)} shown")
        for art in articles[:5]:
            age = art.get("age_days")
            age_str = f"({age}d ago)" if age is not None else ""
            print(f"    - {art.get('title', '')[:70]} {age_str}")
            print(f"      {art.get('source', '')} — {art.get('summary', '')[:100]}...")


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
    _ensure_services_up(cfg, log)
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
    shadow_mode = getattr(args, "shadow_mode", None)
    reconciled = reconciler.reconcile_all(raw_signals, shadow_mode=shadow_mode)
    actionable = sum(1 for r in reconciled.values() if r.action != "HOLD")
    conflicts  = sum(1 for r in reconciled.values() if r.conflict)
    print(f"  Actionable: {actionable}  |  Conflicts blocked: {conflicts}  |  HOLD: {len(reconciled)-actionable}")
    log.info("Step 2 done in %.1fs — %d actionable, %d conflicts",
             time.time() - t0, actionable, conflicts)

    # A3-Data: persist signal feature vectors for future meta-labeler training
    try:
        from harness.paper_trading.db import save_signal_log
        from pathlib import Path as _Path
        save_signal_log(
            signals=reconciled,
            regime=getattr(orchestrator, "_last_regime", None),
            regime_features=getattr(orchestrator, "_last_regime_features", None),
            db_path=_Path(cfg.paper_db_path),
        )
        log.debug("signal_log: %d rows written", len(reconciled))
    except Exception as _sl_exc:
        log.warning("signal_log write failed (non-blocking): %s", _sl_exc)

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

    # ── Step 3.5: Portfolio Risk Check (non-blocking background thread) ─────────
    import threading
    import requests as _requests
    _step(3.5, STEPS, "Portfolio risk check")
    risk_blocked = False

    def _portfolio_risk_thread():
        try:
            from harness.paper_trading.unified_reader import get_all_positions
            positions = get_all_positions()
            open_positions = [p for p in positions if p.shares > 0]
            if not open_positions:
                return
            total_value = sum(p.shares * (p.current_price or p.avg_cost) for p in open_positions)
            holdings = [
                {
                    "ticker": p.ticker,
                    "weight": round(p.shares * (p.current_price or p.avg_cost) / total_value, 6),
                }
                for p in open_positions
            ]
            resp = _requests.post(
                f"{cfg.risk_api_url}/api/risk/portfolio",
                json={"holdings": holdings, "benchmark": "SPY", "lookback_days": 252},
                timeout=180,
            )
            if resp.status_code == 200:
                p_risk = resp.json()
                comp_score = p_risk.get("portfolio_composite_score", 0)
                div_ratio = p_risk.get("diversification_ratio", 0) or 0
                bucket = p_risk.get("portfolio_risk_bucket", "N/A")
                log.info("Portfolio risk — score=%.1f bucket=%s diversification=%.2f",
                         comp_score, bucket, div_ratio)
                if comp_score > 75:
                    log.warning("High portfolio risk score (>75): %.1f", comp_score)
            else:
                log.warning("Portfolio risk HTTP %d", resp.status_code)
        except Exception as exc:
            log.warning("Portfolio risk check failed: %s", exc)

    _t = threading.Thread(target=_portfolio_risk_thread, daemon=True)
    _t.start()
    print("  Portfolio risk check running in background (results in log)")

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

    # ── SentimentAnalysisService (AtLogon — keeps port 8000 up before crons fire) ──
    sentiment_xml = textwrap.dedent(f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Sentiment Analysis FastAPI service — keeps localhost:8000 running for harness data collection</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{_TRADING_ROOT / 'sentiment_analysis' / 'venv' / 'Scripts' / 'python.exe'}</Command>
      <Arguments>-m app.main</Arguments>
      <WorkingDirectory>{_TRADING_ROOT / 'sentiment_analysis' / 'backend'}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>""").strip()

    svc_xml_path = _TRADING_ROOT / "SentimentAnalysisService.xml"
    svc_xml_path.write_text(sentiment_xml, encoding="utf-16")
    try:
        subprocess.run(
            ["schtasks", "/Create", "/TN", "SentimentAnalysisService", "/XML", str(svc_xml_path), "/F"],
            check=True, capture_output=True, text=True,
        )
        print("  [OK] Task registered: SentimentAnalysisService")
    except subprocess.CalledProcessError as e:
        print(f"  [!]  schtasks failed for SentimentAnalysisService: {e.stderr.strip()}")
        print(f"       Import manually: schtasks /Create /TN SentimentAnalysisService /XML {svc_xml_path} /F")

    # ── Remove legacy per-strategy tasks from Task Scheduler and local disk ──
    _OLD_TASKS = [
        "TradingHourlyCron",
        "TradingSentimentRisk_0800",
        "TradingSentimentRisk_1100",
        "TradingSentimentRisk_1400",
        "TradingDailyPipeline",
    ]
    _OLD_LOCAL_FILES = [
        _TRADING_ROOT / "temp_task.xml",
    ]

    print("\n  Removing legacy scheduled tasks...")
    for task in _OLD_TASKS:
        try:
            subprocess.run(
                ["schtasks", "/End", "/TN", task],
                capture_output=True, text=True,
            )
            result = subprocess.run(
                ["schtasks", "/Delete", "/TN", task, "/F"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"  [OK] Deleted task: {task}")
            else:
                print(f"  [--] Task not found (already removed): {task}")
        except Exception as ex:
            print(f"  [!]  Could not delete {task}: {ex}")

    for path in _OLD_LOCAL_FILES:
        try:
            if path.exists():
                path.unlink()
                print(f"  [OK] Deleted local file: {path.name}")
        except Exception as ex:
            print(f"  [!]  Could not delete {path.name}: {ex}")

    # ── RagMarketIntelligenceService (AtLogon — keeps port 8200 up) ─────────────
    rag_xml = textwrap.dedent(f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>RAG Market Intelligence FastAPI service — keeps localhost:8200 running for harness RAG enrichment and cmd_ask</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{sys.executable}</Command>
      <Arguments>-m app.main</Arguments>
      <WorkingDirectory>{_TRADING_ROOT / 'rag_service'}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>""").strip()

    rag_xml_path = _TRADING_ROOT / "RagMarketIntelligenceService.xml"
    rag_xml_path.write_text(rag_xml, encoding="utf-16")
    try:
        subprocess.run(
            ["schtasks", "/Create", "/TN", "RagMarketIntelligenceService", "/XML", str(rag_xml_path), "/F"],
            check=True, capture_output=True, text=True,
        )
        print("  [OK] Task registered: RagMarketIntelligenceService")
    except subprocess.CalledProcessError as e:
        print(f"  [!]  schtasks failed for RagMarketIntelligenceService: {e.stderr.strip()}")
        print(f"       Import manually: schtasks /Create /TN RagMarketIntelligenceService /XML {rag_xml_path} /F")

    print("\n  Harness tasks registered. Legacy tasks removed.")
    print("  View in Task Scheduler → Task Scheduler Library.\n")


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
    """Check all RL models for rolling Sharpe degradation; retrain if needed.

    Uses the RL backtest history to compute a rolling mean Sharpe.  If the
    rolling mean or the latest episode Sharpe falls below ``rl_min_sharpe``,
    the model is flagged for retraining.
    """
    log = _setup_logging("retrain_check")
    from harness.config import get_config
    from harness.rl_monitor import (
        run_retrain_check,
        save_retrain_report,
        print_retrain_report,
    )

    cfg = get_config()
    min_sharpe = getattr(args, "min_sharpe", None) or cfg.rl_min_sharpe
    auto_retrain = getattr(args, "auto_retrain", False)
    retrain_timesteps = getattr(args, "retrain_timesteps", 50000)
    backtest_window = getattr(args, "backtest_window", 3)
    episode_window = getattr(args, "episode_window", 5)
    ticker = getattr(args, "ticker", None)
    tickers = [ticker.upper()] if ticker else cfg._rl_tickers

    print(f"  Checking {len(tickers)} RL model(s) against Sharpe threshold {min_sharpe:.2f}...")
    results = run_retrain_check(
        cfg=cfg,
        min_sharpe=min_sharpe,
        backtest_window=backtest_window,
        episode_window=episode_window,
        auto_retrain=auto_retrain,
        retrain_timesteps=retrain_timesteps,
        tickers=tickers,
    )
    print_retrain_report(results, min_sharpe)

    report_path = save_retrain_report(cfg, results, min_sharpe, auto_retrain)
    print(f"  Report saved: {report_path}")
    log.info("Retrain check complete — degraded=%d, retrained=%d",
             sum(1 for r in results if r.degraded),
             sum(1 for r in results if r.retrained))



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

    p_dash = sub.add_parser("dashboard",
                             help="Unified health + Alpaca + positions + trades + data + RL + regime")
    p_dash.add_argument("--trade-limit", type=int, default=10,
                        help="Number of recent trades to show (default: 10)")
    p_dash.add_argument("--max-age-hours", type=float, default=25.0,
                        help="Max age for fresh market data in hours (default: 25)")
    p_dash.add_argument("--health-timeout", type=float, default=5.0,
                        help="Timeout for service health checks in seconds (default: 5)")
    p_dash.add_argument("--no-save", action="store_true",
                        help="Print dashboard without saving JSON")

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
    p_sg.add_argument("--shadow-mode", default=None,
                      choices=["confidence_weighted", "majority_vote", "rl_priority", "consensus_only"],
                      help="Run an A/B test with a shadow reconciliation mode and log divergences")

    sub.add_parser("report", help="Print unified P&L, positions, and strategy comparison")
    sub.add_parser("positions", help="Show all open positions across all strategy DBs")
    sub.add_parser("schedule", help="Register Windows Task Scheduler jobs")

    p_train = sub.add_parser("train", help="Retrain RL models (all tickers or one)")
    p_train.add_argument("--ticker", default=None, help="Single ticker (default: all)")
    p_train.add_argument("--timesteps", type=int, default=100_000,
                         help="Training timesteps per model (default: 100000)")

    p_retrain = sub.add_parser("retrain-check",
                                help="Check RL models for Sharpe degradation, retrain if needed")
    p_retrain.add_argument("--ticker", default=None,
                           help="Check a single RL ticker instead of all configured tickers")
    p_retrain.add_argument("--min-sharpe", type=float, default=None,
                           help=f"Sharpe threshold (default: cfg.rl_min_sharpe)")
    p_retrain.add_argument("--backtest-window", type=int, default=3,
                           help="Backtests to include in rolling mean Sharpe (default: 3)")
    p_retrain.add_argument("--episode-window", type=int, default=5,
                           help="Fallback episode window for rolling Sharpe (default: 5)")
    p_retrain.add_argument("--auto-retrain", action="store_true",
                           help="Automatically retrain degraded models")
    p_retrain.add_argument("--retrain-timesteps", type=int, default=50000,
                           help="Timesteps for auto-retraining (default: 50000)")

    p_logs = sub.add_parser("logs", help="Tail recent log files from all services")
    p_logs.add_argument("--lines", type=int, default=50,
                        help="Lines per file to show (default: 50)")

    p_bt = sub.add_parser("backtest", help="Run per-strategy backtests and compare reconciliation modes")
    p_bt.add_argument("--days", type=int, default=180, help="Lookback period in days (default: 180)")
    p_bt.add_argument("--strategy", choices=["rl", "mr", "tf", "vb"], default=None,
                      help="Run only one strategy (default: all 4)")

    p_comp = sub.add_parser("comprehensive",
                            help="Run auth-protected deep-dive analysis on tickers")
    p_comp.add_argument("tickers", nargs="+", help="One or more ticker symbols")

    p_ask = sub.add_parser("ask",
                           help="Query RAG market-intelligence layer with a free-text question")
    p_ask.add_argument("query", nargs="+", help="Free-text query (e.g. \"what is the recent sentiment for NVDA?\")") 
    p_ask.add_argument("--ticker", default=None, help="Filter results to a specific ticker")
    p_ask.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30)")

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
        "dashboard": cmd_dashboard,
        "data": cmd_data,
        "data_collection": cmd_data_collection,
        "comprehensive": cmd_comprehensive,
        "ask": cmd_ask,
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
