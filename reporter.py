"""Unified reporter — formats signal summaries, positions, and strategy comparison."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .adapters.base import HarnessSignal
from .reconciler import ReconciledSignal
from .paper_trading.db import HarnessTradingDB
from .paper_trading.unified_reader import get_all_positions, get_all_trades, summary as db_summary
from .config import HarnessConfig, get_config

_TRADING_ROOT = Path(__file__).resolve().parents[1]


class Reporter:
    """Formats and prints all harness output."""

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()
        self.db = HarnessTradingDB(self.cfg.paper_db_path)

    # ── Signal summary ────────────────────────────────────────────────────────

    def print_signal_summary(
        self,
        reconciled: Dict[str, ReconciledSignal],
        raw_signals: Optional[Dict[str, List[HarnessSignal]]] = None,
    ) -> None:
        strategies = self.cfg.strategies
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*72}")
        print(f"HARNESS — SIGNAL SUMMARY  {ts}")
        print(f"{'='*72}\n")

        header = f"{'Ticker':<8}"
        for s in strategies:
            header += f"  {s.upper():<6}"
        header += f"  {'FINAL':<6}  Conf"
        print(header)
        print("-" * 72)

        executed = held = conflicts = 0
        for ticker in sorted(reconciled.keys()):
            rec = reconciled[ticker]
            row = f"{ticker:<8}"
            for s in strategies:
                vote = rec.votes.get(s, "-")
                conf = rec.vote_confidences.get(s, 0.0)
                cell = f"{vote:<4}" if vote != "HOLD" else "HOLD"
                row += f"  {cell:<6}"
            tag = ""
            if rec.conflict:
                tag = " ⚡"
                conflicts += 1
            elif rec.action != "HOLD":
                tag = " ✓"
            final_tag = f"{rec.action}{tag}"
            row += f"  {final_tag:<8}  {rec.confidence:.2f}"
            print(row)

            if rec.action != "HOLD":
                executed += 1
            else:
                held += 1

        print("-" * 72)
        print(f"\nExecuted: {executed}  |  Hold: {held}  |  Conflicts: {conflicts}\n")

    # ── Positions (unified across all 5 DBs) ───────────────────────────────────

    def print_positions(self) -> None:
        positions = get_all_positions()   # reads harness + mr + tf + vb + rl DBs
        s = db_summary()

        print(f"\n{'='*72}")
        print("UNIFIED POSITIONS  (all strategies)")
        print(f"{'='*72}\n")

        if not positions:
            print("  No open positions across any strategy.\n")
            return

        by_strat = s["by_strategy"]
        strat_summary = "  ".join(f"{k.upper()}:{v}" for k, v in sorted(by_strat.items()))
        print(f"  Open: {s['open_positions']}  [{strat_summary}]")
        print()
        print(f"  {'Ticker':<8}  {'Strat':<8}  {'Shares':>10}  {'Entry':>8}  {'Unreal P&L':>12}  {'Real P&L':>10}")
        print("  " + "-" * 66)
        for p in positions:
            unreal_str = f"+${p.unrealized_pnl:.2f}" if p.unrealized_pnl >= 0 else f"-${abs(p.unrealized_pnl):.2f}"
            real_str   = f"+${p.realized_pnl:.2f}"   if p.realized_pnl   >= 0 else f"-${abs(p.realized_pnl):.2f}"
            print(
                f"  {p.ticker:<8}  {p.strategy:<8}  "
                f"{p.shares:>10.2f}  ${p.avg_cost:>7.2f}  "
                f"{unreal_str:>12}  {real_str:>10}"
            )
        print("  " + "-" * 66)
        total_u = s['total_unrealized_pnl']
        total_r = s['total_realized_pnl']
        u_str = f"+${total_u:.2f}" if total_u >= 0 else f"-${abs(total_u):.2f}"
        r_str = f"+${total_r:.2f}" if total_r >= 0 else f"-${abs(total_r):.2f}"
        print(f"\n  Unrealized P&L: {u_str}   Realized P&L: {r_str}\n"
        )

    # ── Recent trades (unified across all 5 DBs) ─────────────────────────────

    def print_recent_trades(self, limit: int = 20) -> None:
        trades = get_all_trades(limit_per_db=limit)
        trades = trades[:limit]

        print(f"\n{'='*72}")
        print(f"RECENT TRADES  (last {limit} across all strategies)")
        print(f"{'='*72}\n")

        if not trades:
            print("  No trades recorded yet.\n")
            return

        print(f"  {'Time':<20}  {'Strat':<6}  {'Ticker':<8}  {'Action':<5}  {'Shares':>8}  {'Price':>8}  {'P&L':>10}")
        print("  " + "-" * 72)
        for t in trades:
            pnl_str = f"+${t.pnl:.2f}" if t.pnl and t.pnl >= 0 else (f"-${abs(t.pnl):.2f}" if t.pnl else "   —")
            print(
                f"  {t.executed_at[:19]:<20}  {t.strategy:<6}  {t.ticker:<8}  "
                f"{t.action:<5}  {t.shares:>8.2f}  ${t.price:>7.2f}  {pnl_str:>10}"
            )
        print()

    # ── Strategy comparison ───────────────────────────────────────────────────

    def print_strategy_comparison(self) -> None:
        results_dir = Path(self.cfg.results_dir)
        rows = []

        if results_dir.exists():
            sharpes = []
            returns = []
            win_rates = []
            trade_counts = []
            for f in sorted(results_dir.glob("*_backtest.json")):
                try:
                    data = json.loads(f.read_text())
                    em = data.get("episode_metrics", {})
                    sharpes.append(em.get("mean_sharpe", 0.0))
                    returns.append(em.get("mean_return", 0.0))
                    win_rates.append(data.get("trade_metrics", {}).get("win_rate", 0.0) * 100)
                    trade_counts.append(data.get("total_trades", 0))
                except Exception:
                    pass
            if sharpes:
                rows.append({
                    "strategy": "rl_strategy",
                    "trades": sum(trade_counts),
                    "win_rate": sum(win_rates) / len(win_rates) if win_rates else 0,
                    "pnl": sum(returns),
                    "sharpe": sum(sharpes) / len(sharpes),
                })

        if not rows:
            print("\n  No strategy comparison data available yet.\n")
            return

        rows.sort(key=lambda r: r["sharpe"], reverse=True)

        print(f"\n{'='*72}")
        print("STRATEGY COMPARISON (backtest results)")
        print(f"{'='*72}\n")
        print(f"  {'Strategy':<20}  {'Trades':>8}  {'Win Rate':>10}  {'P&L':>12}  {'Sharpe':>8}")
        print("  " + "-" * 68)
        for r in rows:
            tag = "  ⭐ BEST" if r == rows[0] else ""
            print(
                f"  {r['strategy']:<20}  {r['trades']:>8}  "
                f"{r['win_rate']:>9.1f}%  ${r['pnl']:>10.0f}  {r['sharpe']:>8.2f}{tag}"
            )
        print()

    # ── Full report ───────────────────────────────────────────────────────────

    def print_full_report(
        self,
        reconciled: Optional[Dict[str, ReconciledSignal]] = None,
        raw_signals: Optional[Dict[str, List[HarnessSignal]]] = None,
    ) -> None:
        if reconciled:
            self.print_signal_summary(reconciled, raw_signals)
        self.print_positions()
        self.print_recent_trades()
        self.print_strategy_comparison()

    # ── Save run report to file ───────────────────────────────────────────────

    def save_run_report(
        self,
        reconciled: Dict[str, ReconciledSignal],
        raw_signals: Dict[str, List[HarnessSignal]],
        executed: int,
        skipped: int,
        dry_run: bool,
    ) -> None:
        """Save a JSON run summary to logs/run_reports/."""
        from datetime import datetime
        ts = datetime.now()
        logs_dir = Path(self.cfg.logs_dir) / "run_reports"
        logs_dir.mkdir(parents=True, exist_ok=True)
        report_path = logs_dir / f"signal_generation_{ts.strftime('%Y%m%d_%H%M%S')}.json"

        signals_out = {}
        for ticker, rec in reconciled.items():
            signals_out[ticker] = {
                "action": rec.action,
                "confidence": round(rec.confidence, 4),
                "price": round(rec.price, 4),
                "conflict": rec.conflict,
                "votes": rec.votes,
                "vote_confidences": {k: round(v, 4) for k, v in rec.vote_confidences.items()},
            }

        report = {
            "run_at": ts.isoformat(),
            "dry_run": dry_run,
            "executed": executed,
            "skipped": skipped,
            "total_tickers": len(reconciled),
            "actionable": sum(1 for r in reconciled.values() if r.action != "HOLD"),
            "conflicts": sum(1 for r in reconciled.values() if r.conflict),
            "signals": signals_out,
        }

        try:
            report_path.write_text(json.dumps(report, indent=2))
            print(f"\n  Run report saved: {report_path}")
        except Exception as e:
            print(f"\n  WARNING: Could not save run report: {e}")
