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
                tag = " !"
                conflicts += 1
            elif rec.action != "HOLD":
                tag = " [OK]"
            elif rec.action == "HOLD" and rec.confidence == 0.0:
                # Check if it was an error
                is_degraded = False
                if any(v.startswith("Error") for v in rec.votes.values()):
                    is_degraded = True
                elif raw_signals:
                    # Check raw signals for Error
                    if any(sig.reason is not None and "Error" in sig.reason for signals in raw_signals.values() for sig in signals if sig.ticker == ticker and sig.source in rec.votes):
                        is_degraded = True
                
                if is_degraded:
                    tag = " [DEGRADED]"
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
            pnl_str = f"+${t.pnl:.2f}" if t.pnl and t.pnl >= 0 else (f"-${abs(t.pnl):.2f}" if t.pnl else "   -")
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
            # Prefer the latest harness_backtest_*.json which contains all 4 strategies
            harness_files = sorted(results_dir.glob("harness_backtest_*.json"), reverse=True)
            if harness_files:
                try:
                    data = json.loads(harness_files[0].read_text())
                    for s in data.get("strategies", []):
                        if s.get("error"):
                            continue
                        rows.append({
                            "strategy": s["strategy"],
                            "trades": s.get("num_trades", 0),
                            "win_rate": s.get("win_rate_pct", 0.0),
                            "pnl": s.get("total_return_pct", 0.0),
                            "sharpe": s.get("sharpe", 0.0),
                        })
                except Exception:
                    pass

            # Fall back to RL per-ticker backtest files if no harness backtest exists
            if not rows:
                sharpes, returns, win_rates, trade_counts = [], [], [], []
                for f in sorted(results_dir.glob("*_backtest.json")):
                    if f.name.startswith("harness_"):
                        continue
                    try:
                        data = json.loads(f.read_text())
                        em = data.get("episode_metrics", {})
                        s = em.get("mean_sharpe")
                        if s is not None:
                            sharpes.append(float(s))
                            returns.append(em.get("mean_return", 0.0))
                            win_rates.append(data.get("trade_metrics", {}).get("win_rate", 0.0) * 100)
                            trade_counts.append(data.get("total_trades", 0))
                    except Exception:
                        pass
                if sharpes:
                    rows.append({
                        "strategy": "rl",
                        "trades": sum(trade_counts),
                        "win_rate": sum(win_rates) / len(win_rates) if win_rates else 0,
                        "pnl": sum(returns),
                        "sharpe": sum(sharpes) / len(sharpes),
                    })

        if not rows:
            print("\n  No strategy comparison data available yet. Run: harness backtest\n")
            return

        rows.sort(key=lambda r: r["sharpe"], reverse=True)

        print(f"\n{'='*72}")
        print("STRATEGY COMPARISON (backtest results)")
        print(f"{'='*72}\n")
        print(f"  {'Strategy':<20}  {'Trades':>8}  {'Win Rate':>10}  {'Return%':>10}  {'Sharpe':>8}")
        print("  " + "-" * 68)
        for r in rows:
            tag = "  [BEST]" if r == rows[0] else ""
            print(
                f"  {r['strategy']:<20}  {r['trades']:>8}  "
                f"{r['win_rate']:>9.1f}%  {r['pnl']:>+9.1f}%  {r['sharpe']:>8.2f}{tag}"
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

    # ── Regime log helper ─────────────────────────────────────────────────────

    def _read_latest_regime_log(self) -> dict:
        """Return the most recent regime_log row as a plain dict.

        Keys: ``regime``, ``allocation_mode``, ``allocations`` (list of dicts),
        ``allocation_summary`` (compact string).  All values default to empty
        strings / empty lists if the table cannot be read.
        """
        result = {
            "regime": "",
            "allocation_mode": "",
            "allocations": [],
            "allocation_summary": "",
        }
        try:
            import sqlite3 as _sqlite3
            with _sqlite3.connect(str(self.db.db_path)) as conn:
                row = conn.execute(
                    "SELECT regime, allocation_mode, allocations_json "
                    "FROM regime_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if row:
                allocations = json.loads(row[2] or "[]")
                result["regime"] = row[0]
                result["allocation_mode"] = row[1]
                result["allocations"] = allocations
                result["allocation_summary"] = "  ".join(
                    f"{a['strategy'].upper()}=${a['capital']:,.0f}"
                    for a in allocations
                )
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[reporter] Could not read regime_log: %s", exc
            )
        return result

    # ── LLM run summary ───────────────────────────────────────────────────────

    def summarize_run(self, report: dict) -> Optional[str]:
        """Generate a 2-3 paragraph plain-English narrative for this run.

        Reads regime + allocation from the report dict (populated by
        ``save_run_report`` before this is called).  Fetches open-position
        counts directly from the DB.  Returns ``None`` on any failure — the
        run is never blocked.

        Only called when ``cfg.summary_mode == 'llm'``.
        """
        if self.cfg.summary_mode != "llm":
            return None

        open_count = 0
        unreal_pnl = 0.0
        try:
            from .paper_trading.unified_reader import summary as _db_summary
            s = _db_summary()
            open_count = s.get("open_positions", 0)
            unreal_pnl = s.get("total_unrealized_pnl", 0.0)
        except Exception:
            pass

        signals = report.get("signals", {})
        buys = sorted(
            [(t, d["confidence"]) for t, d in signals.items() if d["action"] == "BUY"],
            key=lambda x: -x[1],
        )[:5]
        sells = sorted(
            [(t, d["confidence"]) for t, d in signals.items() if d["action"] == "SELL"],
            key=lambda x: -x[1],
        )[:5]
        buy_str  = ", ".join(f"{t}({c:.2f})" for t, c in buys)  or "none"
        sell_str = ", ".join(f"{t}({c:.2f})" for t, c in sells) or "none"

        regime_label    = report.get("regime", "unknown").upper()
        alloc_summary   = report.get("allocation_summary", "unavailable")
        actionable      = report.get("actionable", 0)
        total_tickers   = report.get("total_tickers", 0)
        conflicts       = report.get("conflicts", 0)
        holds           = total_tickers - actionable - conflicts
        pnl_str         = f"${unreal_pnl:+,.0f}"

        prompt = (
            "You are an operator reviewing a live/paper algorithmic trading system run. "
            "Write a concise 2-3 paragraph plain-English narrative covering: "
            "(1) what the current market regime means and how capital has been allocated, "
            "(2) which signals were generated and the key trades taken, "
            "(3) the current portfolio state and any notable observations or risks.\n\n"
            "Run data:\n"
            f"- Time         : {report.get('run_at', 'unknown')}  "
            f"({'dry-run' if report.get('dry_run') else 'live/paper'})\n"
            f"- Regime       : {regime_label}\n"
            f"- Allocation   : {alloc_summary}\n"
            f"- Signals      : {actionable} executed | {holds} HOLD | {conflicts} conflicts blocked\n"
            f"- Top BUYs     : {buy_str}\n"
            f"- Top SELLs    : {sell_str}\n"
            f"- Open positions: {open_count}  (unrealized P&L: {pnl_str})\n\n"
            "Be factual and concise. Do not invent numbers not present above. "
            "Do not repeat raw numbers verbatim — synthesise them into readable prose."
        )

        from .llm_client import call_llm
        return call_llm(
            prompt,
            provider=self.cfg.llm_provider,
            model=self.cfg.llm_model,
            base_url=self.cfg.llm_base_url,
        )

    # ── RAG-grounded run summary (A9) ─────────────────────────────────────────

    def enrich_with_context(self, report: dict) -> Optional[str]:
        """Generate a RAG-grounded operator narrative via rag_service.

        Called when ``cfg.summary_mode == 'rag'``.  Supersedes A2 LLM narrative
        when rag_service is deployed.  Returns ``None`` silently on any failure.
        """
        if self.cfg.summary_mode != "rag":
            return None
        try:
            from .rag_client import RAGClient
            client = RAGClient(base_url=self.cfg.rag_service_url)
            if not client.is_up():
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "[reporter] rag_service unreachable at %s — skipping RAG narrative",
                    self.cfg.rag_service_url,
                )
                return None
            return client.summarize(report)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[reporter] enrich_with_context failed (non-blocking): %s", exc
            )
            return None

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

        regime_info = self._read_latest_regime_log()
        if regime_info["regime"]:
            report["regime"] = regime_info["regime"]
            report["allocation_mode"] = regime_info["allocation_mode"]
            report["allocation_summary"] = regime_info["allocation_summary"]

        narrative: Optional[str] = None
        if self.cfg.summary_mode == "llm":
            narrative = self.summarize_run(report)
            if narrative:
                report["llm_summary"] = narrative
            else:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "[reporter] LLM summary skipped — no response from %s/%s",
                    self.cfg.llm_provider, self.cfg.llm_model,
                )
        elif self.cfg.summary_mode == "rag":
            narrative = self.enrich_with_context(report)
            if narrative:
                report["rag_summary"] = narrative
            else:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "[reporter] RAG summary skipped — rag_service unavailable or failed"
                )

        summary_key = "rag_summary" if self.cfg.summary_mode == "rag" else "llm_summary"
        summary_label = "RAG RUN SUMMARY" if self.cfg.summary_mode == "rag" else "LLM RUN SUMMARY"

        try:
            report_path.write_text(json.dumps(report, indent=2))
            print(f"\n  Run report saved: {report_path}")
            if report.get(summary_key):
                print(f"\n{'─'*72}")
                print(f"  {summary_label}")
                print(f"{'─'*72}")
                print(report[summary_key])
                print(f"{'─'*72}")
        except Exception as e:
            print(f"\n  WARNING: Could not save run report: {e}")
