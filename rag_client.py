"""Thin HTTP client for the RAG service (rag_service :8200).

All methods degrade gracefully — ConnectionError or any HTTP failure returns None
without raising, so the harness pipeline is never blocked by RAG unavailability.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT = 30


class RAGClient:
    """HTTP client for rag_service endpoints."""

    def __init__(self, base_url: str = "http://localhost:8200", timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Optional[Dict[str, Any]]:
        """GET /api/health — returns dict or None if unreachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=self.timeout)
            return resp.json() if resp.status_code == 200 else None
        except Exception as e:
            log.debug("rag_service health check failed: %s", e)
            return None

    def ingest(self, sources: Optional[List[str]] = None, incremental: bool = True) -> Optional[Dict[str, Any]]:
        """POST /api/ingest — trigger incremental ingestion. Returns response dict or None."""
        payload = {"incremental": incremental}
        if sources:
            payload["sources"] = sources
        try:
            resp = requests.post(f"{self.base_url}/api/ingest", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning("rag_service ingest failed (non-blocking): %s", e)
            return None

    def ask(
        self,
        query: str,
        ticker: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """POST /api/ask — returns answer dict or None if unavailable."""
        payload: Dict[str, Any] = {"query": query}
        if ticker:
            payload["ticker"] = ticker
        if date_from:
            payload["date_from"] = date_from
        if date_to:
            payload["date_to"] = date_to
        if top_k:
            payload["top_k"] = top_k
        try:
            resp = requests.post(f"{self.base_url}/api/ask", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning("rag_service ask failed: %s", e)
            return None

    def summarize(self, run_report: Dict[str, Any]) -> Optional[str]:
        """POST /api/summarize — returns narrative string or None if unavailable."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/summarize",
                json={"run_report": run_report},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("narrative")
        except Exception as e:
            log.warning("rag_service summarize failed (non-blocking): %s", e)
            return None

    def get_context(self, ticker: str, days: int = 7) -> Optional[Dict[str, Any]]:
        """GET /api/context/{ticker} — returns context dict or None."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/context/{ticker}",
                params={"days": days},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.debug("rag_service context(%s) failed: %s", ticker, e)
            return None

    def is_up(self) -> bool:
        """Quick reachability check."""
        h = self.health()
        return h is not None and h.get("status") in ("healthy", "degraded")
