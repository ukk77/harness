"""Tests for A2 — LLM Run Summary.

Run directly (no pytest required):
    python harness/test_llm_summary.py

Covers:
  - llm_client.call_llm: unknown provider, connection error, HTTP error, happy-path (mocked)
  - Reporter.summarize_run: mode=none no-op, LLM failure returns None, success returns text
  - Reporter.save_run_report: no llm_summary when mode=none; llm_summary in JSON when mode=llm
  - Regime enrichment always added to JSON when regime_log has data
  - LLM failure still saves the report (run is never blocked)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


# ── helpers ───────────────────────────────────────────────────────────────────


def _minimal_reconciled() -> dict:
    buy = SimpleNamespace(
        action="BUY", confidence=1.0, price=100.0, conflict=False,
        votes={"rl": "BUY"}, vote_confidences={"rl": 1.0},
    )
    hold = SimpleNamespace(
        action="HOLD", confidence=0.0, price=200.0, conflict=False,
        votes={"rl": "HOLD"}, vote_confidences={"rl": 0.0},
    )
    return {"AAPL": buy, "MSFT": hold}


def _make_reporter(summary_mode: str = "none", tmp_db: str | None = None):
    from harness.reporter import Reporter
    from harness.config import HarnessConfig
    cfg = HarnessConfig()
    cfg.summary_mode = summary_mode
    cfg.llm_provider = "ollama"
    cfg.llm_model = "llama3.2:3b"
    cfg.llm_base_url = "http://localhost:11434"
    if tmp_db:
        cfg.paper_db_path = tmp_db
    return Reporter(cfg=cfg)


# ── llm_client tests ──────────────────────────────────────────────────────────


def test_call_llm_unknown_provider():
    print("\n--- test_call_llm_unknown_provider ---")
    from harness.llm_client import call_llm
    result = call_llm("hello", provider="unknown_xyz")
    check("unknown provider returns None", result is None)


def test_call_llm_ollama_connection_error():
    print("\n--- test_call_llm_ollama_connection_error ---")
    from harness.llm_client import call_llm
    with patch("requests.post", side_effect=ConnectionError("refused")):
        result = call_llm("hello", provider="ollama", model="x", base_url="http://localhost:11434")
    check("connection error returns None (no raise)", result is None)


def test_call_llm_ollama_http_error():
    print("\n--- test_call_llm_ollama_http_error ---")
    import requests
    from harness.llm_client import call_llm
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
    with patch("requests.post", return_value=mock_resp):
        result = call_llm("hello", provider="ollama")
    check("HTTP error returns None (no raise)", result is None)


def test_call_llm_ollama_success():
    print("\n--- test_call_llm_ollama_success ---")
    from harness.llm_client import call_llm
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": "  bear trend narrative  "}}
    with patch("requests.post", return_value=mock_resp):
        result = call_llm("prompt", provider="ollama", model="llama3.2:3b")
    check("successful call returns stripped text", result == "bear trend narrative",
          f"got: {result!r}")


def test_call_llm_openai_missing_package():
    print("\n--- test_call_llm_openai_missing_package ---")
    from harness.llm_client import call_llm
    with patch.dict("sys.modules", {"openai": None}):
        result = call_llm("prompt", provider="openai", model="gpt-4o-mini")
    check("missing openai package returns None (no raise)", result is None)


# ── Reporter.summarize_run tests ──────────────────────────────────────────────


def test_summarize_run_mode_none():
    print("\n--- test_summarize_run_mode_none ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reporter = _make_reporter(summary_mode="none", tmp_db=str(Path(tmp) / "t.db"))
        result = reporter.summarize_run({"regime": "bear_trend", "signals": {}})
    check("summary_mode=none returns None (no LLM call)", result is None)


def test_summarize_run_llm_failure():
    print("\n--- test_summarize_run_llm_failure ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reporter = _make_reporter(summary_mode="llm", tmp_db=str(Path(tmp) / "t.db"))
        with patch("harness.llm_client.call_llm", return_value=None):
            result = reporter.summarize_run({"regime": "bear_trend", "signals": {}, "run_at": "t"})
    check("LLM failure in summarize_run returns None (no raise)", result is None)


def test_summarize_run_success():
    print("\n--- test_summarize_run_success ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        reporter = _make_reporter(summary_mode="llm", tmp_db=str(Path(tmp) / "t.db"))
        with patch("harness.llm_client.call_llm", return_value="The market is in a bear trend."):
            result = reporter.summarize_run({
                "regime": "bear_trend",
                "allocation_summary": "RL=$26,316  MR=$21,053",
                "actionable": 10, "total_tickers": 20, "conflicts": 1,
                "signals": {
                    "QQQ": {"action": "BUY", "confidence": 1.0},
                    "TLT": {"action": "SELL", "confidence": 0.9},
                },
                "run_at": "2026-07-10T20:00:00",
                "dry_run": True,
            })
    check("summarize_run returns LLM text on success",
          result == "The market is in a bear trend.", f"got: {result!r}")


# ── Reporter.save_run_report tests ────────────────────────────────────────────


def test_save_run_report_no_summary_mode():
    print("\n--- test_save_run_report_no_summary_mode ---")
    from harness.reporter import Reporter
    from harness.config import HarnessConfig
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = HarnessConfig()
        cfg.summary_mode = "none"
        cfg.logs_dir = str(Path(tmp) / "logs")
        cfg.paper_db_path = str(Path(tmp) / "harness_trades.db")
        reporter = Reporter(cfg=cfg)
        reporter.save_run_report(_minimal_reconciled(), {}, executed=1, skipped=0, dry_run=True)
        reports = list((Path(tmp) / "logs" / "run_reports").glob("*.json"))
        check("exactly one report file saved", len(reports) == 1, f"found {len(reports)}")
        if reports:
            data = json.loads(reports[0].read_text())
            check("no llm_summary key when mode=none", "llm_summary" not in data)
            check("executed count correct", data["executed"] == 1, f"got {data.get('executed')}")


def test_save_run_report_with_llm_summary():
    print("\n--- test_save_run_report_with_llm_summary ---")
    from harness.reporter import Reporter
    from harness.config import HarnessConfig
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = HarnessConfig()
        cfg.summary_mode = "llm"
        cfg.llm_provider = "ollama"
        cfg.llm_model = "llama3.2:3b"
        cfg.llm_base_url = "http://localhost:11434"
        cfg.logs_dir = str(Path(tmp) / "logs")
        cfg.paper_db_path = str(Path(tmp) / "harness_trades.db")
        reporter = Reporter(cfg=cfg)
        with patch("harness.llm_client.call_llm", return_value="Bearish conditions persist."):
            reporter.save_run_report(_minimal_reconciled(), {}, executed=1, skipped=0, dry_run=True)
        reports = list((Path(tmp) / "logs" / "run_reports").glob("*.json"))
        check("report saved", len(reports) == 1)
        if reports:
            data = json.loads(reports[0].read_text())
            check("llm_summary key present in JSON", "llm_summary" in data)
            check("llm_summary text correct",
                  data.get("llm_summary") == "Bearish conditions persist.",
                  f"got: {data.get('llm_summary')!r}")


def test_save_run_report_regime_enrichment():
    print("\n--- test_save_run_report_regime_enrichment ---")
    from harness.reporter import Reporter
    from harness.config import HarnessConfig
    from harness.paper_trading.db import init_db, save_regime_log
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "harness_trades.db"
        cfg = HarnessConfig()
        cfg.summary_mode = "none"
        cfg.logs_dir = str(Path(tmp) / "logs")
        cfg.paper_db_path = str(db_path)
        init_db(db_path)
        save_regime_log(
            regime="bear_trend",
            allocation_mode="regime:bear_trend",
            allocations=[{"strategy": "rl", "capital": 26316.0, "weight": 0.26, "sharpe": 1.2}],
            db_path=db_path,
        )
        reporter = Reporter(cfg=cfg)
        reporter.save_run_report(_minimal_reconciled(), {}, executed=1, skipped=0, dry_run=True)
        reports = list((Path(tmp) / "logs" / "run_reports").glob("*.json"))
        check("report saved", len(reports) == 1)
        if reports:
            data = json.loads(reports[0].read_text())
            check("regime key added to JSON", data.get("regime") == "bear_trend",
                  f"got {data.get('regime')!r}")
            check("allocation_summary contains strategy label",
                  "RL=" in data.get("allocation_summary", ""),
                  f"got {data.get('allocation_summary')!r}")


def test_save_run_report_llm_failure_still_saves():
    print("\n--- test_save_run_report_llm_failure_still_saves ---")
    from harness.reporter import Reporter
    from harness.config import HarnessConfig
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = HarnessConfig()
        cfg.summary_mode = "llm"
        cfg.llm_provider = "ollama"
        cfg.llm_model = "llama3.2:3b"
        cfg.logs_dir = str(Path(tmp) / "logs")
        cfg.paper_db_path = str(Path(tmp) / "harness_trades.db")
        reporter = Reporter(cfg=cfg)
        with patch("harness.llm_client.call_llm", return_value=None):
            reporter.save_run_report(_minimal_reconciled(), {}, executed=1, skipped=0, dry_run=True)
        reports = list((Path(tmp) / "logs" / "run_reports").glob("*.json"))
        check("report still saved even when LLM returns None", len(reports) == 1)
        if reports:
            data = json.loads(reports[0].read_text())
            check("no llm_summary key when LLM fails", "llm_summary" not in data)
            check("executed count still correct when LLM fails",
                  data.get("executed") == 1, f"got {data.get('executed')}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 A2 — LLM Run Summary Tests")
    print("=" * 60)

    test_call_llm_unknown_provider()
    test_call_llm_ollama_connection_error()
    test_call_llm_ollama_http_error()
    test_call_llm_ollama_success()
    test_call_llm_openai_missing_package()

    test_summarize_run_mode_none()
    test_summarize_run_llm_failure()
    test_summarize_run_success()

    test_save_run_report_no_summary_mode()
    test_save_run_report_with_llm_summary()
    test_save_run_report_regime_enrichment()
    test_save_run_report_llm_failure_still_saves()

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
