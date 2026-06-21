"""Base adapter interface and HarnessSignal dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class HarnessSignal:
    """Unified signal schema across all strategies."""
    ticker: str
    timestamp: datetime
    action: str              # "BUY" | "SELL" | "HOLD"
    confidence: float        # 0.0 – 1.0
    source: str              # "rl" | "mr" | "tf" | "vb"
    price: float
    suggested_shares: Optional[float] = None
    reason: Optional[str] = None

    def __str__(self) -> str:
        shares_str = f"{self.suggested_shares:.2f}sh" if self.suggested_shares else "N/A"
        return (
            f"[{self.source.upper()}:{self.ticker}] "
            f"{self.action:<4} conf={self.confidence:.2f} "
            f"@ ${self.price:.2f}  {shares_str}"
        )

    @property
    def is_actionable(self) -> bool:
        return self.action != "HOLD"


class BaseAdapter(ABC):
    """Abstract base for strategy adapters."""

    source: str = "unknown"

    def get_signal(self, ticker: str) -> HarnessSignal:
        """Return a HarnessSignal for *ticker*, falling back to HOLD on any error."""
        try:
            return self._generate(ticker)
        except Exception as e:
            return self._hold(ticker, reason=f"Error: {e}")

    @abstractmethod
    def _generate(self, ticker: str) -> HarnessSignal:
        """Strategy-specific signal generation. May raise."""
        ...

    def _hold(self, ticker: str, price: float = 0.0, reason: str = "HOLD") -> HarnessSignal:
        return HarnessSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            action="HOLD",
            confidence=0.0,
            source=self.source,
            price=price,
            reason=reason,
        )
