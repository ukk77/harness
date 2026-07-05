"""Unified health + positions + data dashboard for the trading harness.

Provides a single CLI view that aggregates:
  - Service health checks (sentiment, risk, models, data, strategy imports)
  - Alpaca account status (equity, buying power, day change)
  - Unified open positions across all 5 paper-trading DBs
  - Recent cross-strategy trades
  - Market data freshness for the parquet cache
  - RL model status and rolling Sharpe
  - Latest detected market regime and capital allocation
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.config import HarnessConfig, get_config
from harness.health import HealthChecker, HealthReport, HealthResult
from harness.paper_trading.unified_reader import (
    UnifiedPosition,
    UnifiedTrade,
    get_all_positions,
    get_all_trades,
    summary as positions_summary,
)
from harness.rl_monitor import (
    _load_backtest_mean_sharpe,
    _model_path,
    load_sharpe_history,
)

log = logging.getLogger(__name__)


@dataclass
class DashboardSection:
    title: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Dashboard:
    timestamp: str
    health: Optional[HealthReport] = None
    alpaca: Optional[Dict[str, Any]] = None
    alpaca_error: Optional[str] = None
    positions_summary: Optional[Dict[str, Any]] = None
    positions: List[UnifiedPosition] = field(default_factory=list)
    recent_trades: List[UnifiedTrade] = field(default_factory=list)
    data_freshness: List[Dict[str, Any]] = field(default_factory=list)
    rl_models: List[Dict[str, Any]] = field(default_factory=list)
    regime: Optional[Dict[str, Any]] = None
    regime_error: Optional[str] = None


def _check_data_freshness(
    cfg: HarnessConfig, max_age_hours: float = 25.0
) -> List[Dict[str, Any]]:
    """Return freshness status for every parquet file in the hourly data dir."""
    data_dir = Path(cfg.market_data_dir)
    if not data_dir.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    rows: List[Dict[str, Any]] = []
    for parquet in sorted(data_dir.glob("*.parquet")):
        ticker = parquet.stem
        freshness_file = parquet.with_suffix(".freshness")
        if not freshness_file.exists():
            rows.append({
                "ticker": ticker,
                "status": "UNKNOWN",
                "age_hours": None,
                "message": "no freshness sidecar",
            })
            continue
        try:
            ts_str = freshness_file.read_text(encoding="utf-8").strip()
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            last_update = datetime.fromisoformat(ts_str)
            age_hours = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
            status = "STALE" if last_update < cutoff else "OK"
            rows.append({
                "ticker": ticker,
                "status": status,
                "age_hours": round(age_hours, 1),
                "message": f"{age_hours:.1f}h old",
            })
        except Exception as exc:
            rows.append({
                "ticker": ticker,
                "status": "ERROR",
                "age_hours": None,
                "message": f"parse error: {exc}",
            })
    return rows


def _check_rl_models(cfg: HarnessConfig) -> List[Dict[str, Any]]:
    """Return status rows for each configured RL ticker."""
    rows: List[Dict[str, Any]] = []
    history = load_sharpe_history(cfg)
    for ticker in cfg._rl_tickers or []:
        model_exists = _model_path(cfg, ticker).exists()
        mean_sharpe = _load_backtest_mean_sharpe(cfg, ticker)
        hist = history.get(ticker, [])
        recent = [entry["mean_sharpe"] for entry in hist[-3:] if "mean_sharpe" in entry]
        rolling = sum(recent) / len(recent) if recent else None

        status = "OK"
        if not model_exists:
            status = "NO MODEL"
        elif mean_sharpe is None:
            status = "NO BACKTEST"
        elif rolling is not None and rolling < cfg.rl_min_sharpe:
            status = "DEGRADED"
        elif mean_sharpe < cfg.rl_min_sharpe:
            status = "DEGRADED"

        rows.append({
            "ticker": ticker,
            "status": status,
            "model_exists": model_exists,
            "mean_sharpe": round(mean_sharpe, 2) if mean_sharpe is not None else None,
            "rolling_sharpe": round(rolling, 2) if rolling is not None else None,
            "history_count": len(hist),
            "threshold": cfg.rl_min_sharpe,
        })
    return rows


def _read_latest_regime(cfg: HarnessConfig) -> Optional[Dict[str, Any]]:
    """Read the latest regime entry from the harness DB."""
    try:
        from harness.paper_trading.db import _conn
        db_path = Path(cfg.paper_db_path)
        if not db_path.exists():
            return None
        with _conn(db_path) as conn:
            row = conn.execute(
                "SELECT logged_at, regime, allocation_mode, allocations_json "
                "FROM regime_log ORDER BY logged_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "logged_at": row["logged_at"],
            "regime": row["regime"],
            "allocation_mode": row["allocation_mode"],
            "allocations": json.loads(row["allocations_json"]),
        }
    except Exception as exc:
        log.warning("Could not read latest regime: %s", exc)
        return None


def build_dashboard(
    cfg: Optional[HarnessConfig] = None,
    include_health: bool = True,
    include_alpaca: bool = True,
    include_positions: bool = True,
    include_trades: bool = True,
    include_data_freshness: bool = True,
    include_rl_models: bool = True,
    include_regime: bool = True,
    max_age_hours: float = 25.0,
    trade_limit: int = 10,
    health_timeout: float = 5.0,
) -> Dashboard:
    """Build a unified dashboard object from all available sources."""
    cfg = cfg or get_config()
    dashboard = Dashboard(timestamp=datetime.utcnow().isoformat() + "Z")

    if include_health:
        try:
            dashboard.health = HealthChecker(cfg, request_timeout=health_timeout).run()
        except Exception as exc:
            log.warning("Health check failed: %s", exc)
            dashboard.health = HealthReport(results=[HealthResult("health", "FAIL", str(exc))])

    if include_alpaca:
        try:
            from trading_core.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(paper=(cfg.execution_mode != "live"))
            dashboard.alpaca = broker.get_account_info()
            # Enrich with day change percentage
            last_equity = dashboard.alpaca.get("last_equity") or 0
            equity = dashboard.alpaca.get("equity") or 0
            if last_equity and last_equity > 0:
                dashboard.alpaca["day_change_pct"] = (equity - last_equity) / last_equity * 100
        except Exception as exc:
            dashboard.alpaca_error = str(exc)

    if include_positions:
        try:
            dashboard.positions = get_all_positions()
            dashboard.positions_summary = positions_summary()
        except Exception as exc:
            log.warning("Could not load positions: %s", exc)

    if include_trades:
        try:
            dashboard.recent_trades = get_all_trades(limit_per_db=trade_limit)[:trade_limit]
        except Exception as exc:
            log.warning("Could not load trades: %s", exc)

    if include_data_freshness:
        try:
            dashboard.data_freshness = _check_data_freshness(cfg, max_age_hours)
        except Exception as exc:
            log.warning("Could not check data freshness: %s", exc)

    if include_rl_models:
        try:
            dashboard.rl_models = _check_rl_models(cfg)
        except Exception as exc:
            log.warning("Could not check RL models: %s", exc)

    if include_regime:
        try:
            dashboard.regime = _read_latest_regime(cfg)
        except Exception as exc:
            dashboard.regime_error = str(exc)

    return dashboard


def _icon(status: str) -> str:
    return {"OK": "[OK]", "WARN": "[WARN]", "FAIL": "[FAIL]", "DEGRADED": "[DEGRADED]"}.get(status, "[?]")


def print_dashboard(dashboard: Dashboard) -> None:
    """Print a formatted dashboard to stdout."""
    ts = datetime.fromisoformat(dashboard.timestamp.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*78}")
    print(f"  HARNESS DASHBOARD  {ts}")
    print(f"{'='*78}")

    # ── Health ────────────────────────────────────────────────────────────────
    print("\n  [Health Checks]")
    if dashboard.health:
        for r in dashboard.health.results:
            print(f"    {_icon(r.status)} {r.name:<24} {r.status:<8} {r.message}")
    else:
        print("    unavailable")

    # ── Alpaca ───────────────────────────────────────────────────────────────
    print("\n  [Alpaca Account]")
    if dashboard.alpaca:
        a = dashboard.alpaca
        equity = a.get("equity", 0)
        last_equity = a.get("last_equity", 0)
        day_change = equity - last_equity if last_equity else 0
        day_pct = a.get("day_change_pct", 0)
        sign = "+" if day_change >= 0 else ""
        print(f"    Equity:        ${equity:>12,.2f}")
        print(f"    Buying power:  ${a.get('buying_power', 0):>12,.2f}")
        print(f"    Cash:          ${a.get('cash', 0):>12,.2f}")
        print(f"    Day change:    ${day_change:>+12,.2f}  ({sign}{day_pct:.2f}%)")
    elif dashboard.alpaca_error:
        print(f"    unavailable — {dashboard.alpaca_error}")
    else:
        print("    unavailable")

    # ── Regime ───────────────────────────────────────────────────────────────
    print("\n  [Market Regime & Allocation]")
    if dashboard.regime:
        r = dashboard.regime
        print(f"    Regime: {r['regime'].upper()}  ({r['logged_at']})")
        print(f"    Mode:   {r['allocation_mode']}")
        alloc = r.get("allocations", [])
        if alloc:
            print("    " + "  ".join(
                f"{a['strategy'].upper()}=${a['capital']:,.0f}({a['weight']*100:.0f}%)"
                for a in alloc
            ))
    elif dashboard.regime_error:
        print(f"    unavailable — {dashboard.regime_error}")
    else:
        print("    no regime log found")

    # ── Positions ────────────────────────────────────────────────────────────
    print("\n  [Open Positions]")
    if dashboard.positions:
        print(
            f"    {'Ticker':<8} {'Strat':<8} {'Shares':>10} {'Entry':>10} "
            f"{'Mkt':>10} {'Unreal P&L':>12} {'Real P&L':>12}"
        )
        print("    " + "-" * 72)
        for p in dashboard.positions:
            mkt = p.shares * p.current_price if p.current_price else 0
            print(
                f"    {p.ticker:<8} {p.strategy:<8} {p.shares:>10.2f} ${p.avg_cost:>9.2f} "
                f"${mkt:>9.2f} {p.unrealized_pnl:>+11.2f} {p.realized_pnl:>+11.2f}"
            )
        s = dashboard.positions_summary or {}
        print("    " + "-" * 72)
        print(f"    Total open: {s.get('open_positions', 0)}  |  "
              f"Unrealized: {s.get('total_unrealized_pnl', 0):+.2f}  |  "
              f"Realized: {s.get('total_realized_pnl', 0):+.2f}")
    else:
        print("    no open positions")

    # ── Recent trades ──────────────────────────────────────────────────────────
    print("\n  [Recent Trades]")
    if dashboard.recent_trades:
        print(
            f"    {'Time':<20} {'Strat':<6} {'Ticker':<8} {'Action':<6} "
            f"{'Shares':>10} {'Price':>10} {'P&L':>12}"
        )
        print("    " + "-" * 74)
        for t in dashboard.recent_trades:
            pnl_str = (
                f"{t.pnl:>+.2f}" if t.pnl is not None else " " * 12
            )
            print(
                f"    {t.executed_at[:19]:<20} {t.strategy:<6} {t.ticker:<8} "
                f"{t.action:<6} {t.shares:>10.2f} ${t.price:>9.2f} {pnl_str:>12}"
            )
    else:
        print("    no recent trades")

    # ── Data freshness ─────────────────────────────────────────────────────────
    print("\n  [Market Data Freshness]")
    if dashboard.data_freshness:
        stale = [r for r in dashboard.data_freshness if r["status"] != "OK"]
        ok = [r for r in dashboard.data_freshness if r["status"] == "OK"]
        print(f"    {len(ok)} fresh, {len(stale)} stale/unknown/error  ({len(dashboard.data_freshness)} files)")
        if stale:
            for r in stale[:10]:
                print(f"    {_icon(r['status'])} {r['ticker']:<8} {r['status']:<8} {r['message']}")
    else:
        print("    no market data found")

    # ── RL models ─────────────────────────────────────────────────────────────
    print("\n  [RL Models]")
    if dashboard.rl_models:
        print(f"    {'Ticker':<8} {'Status':<12} {'Rolling':>8} {'Latest':>8} {'Threshold':>10}")
        print("    " + "-" * 48)
        for r in dashboard.rl_models:
            rolling_str = f"{r['rolling_sharpe']:>7.2f}" if r['rolling_sharpe'] is not None else "N/A"
            latest_str = f"{r['mean_sharpe']:>7.2f}" if r['mean_sharpe'] is not None else "N/A"
            print(
                f"    {r['ticker']:<8} {_icon(r['status'])} {r['status']:<8} "
                f"{rolling_str:>8} {latest_str:>8} {r['threshold']:>10.2f}"
            )
    else:
        print("    no RL models configured")

    print(f"\n{'='*78}\n")


def save_dashboard(
    dashboard: Dashboard,
    cfg: Optional[HarnessConfig] = None,
) -> Path:
    """Serialize the dashboard to ``daily_results/dashboard_YYYYMMDD.json``."""
    cfg = cfg or get_config()
    out_dir = Path(cfg.trading_root) / "daily_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"dashboard_{timestamp}.json"

    payload = {
        "timestamp": dashboard.timestamp,
        "health": {
            "ok": dashboard.health.ok if dashboard.health else None,
            "results": [
                {"name": r.name, "status": r.status, "message": r.message}
                for r in (dashboard.health.results if dashboard.health else [])
            ],
        } if dashboard.health else None,
        "alpaca": dashboard.alpaca,
        "alpaca_error": dashboard.alpaca_error,
        "positions_summary": dashboard.positions_summary,
        "positions": [
            {
                "ticker": p.ticker,
                "strategy": p.strategy,
                "shares": p.shares,
                "avg_cost": p.avg_cost,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "realized_pnl": p.realized_pnl,
            }
            for p in dashboard.positions
        ],
        "recent_trades": [
            {
                "ticker": t.ticker,
                "strategy": t.strategy,
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "executed_at": t.executed_at,
                "pnl": t.pnl,
            }
            for t in dashboard.recent_trades
        ],
        "data_freshness": dashboard.data_freshness,
        "rl_models": dashboard.rl_models,
        "regime": dashboard.regime,
        "regime_error": dashboard.regime_error,
    }

    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Dashboard saved: %s", out_path)
    return out_path
