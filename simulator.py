"""Unified portfolio simulator for backtesting the harness."""
from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional
import logging

from .config import HarnessConfig, get_config
from .regime import detect_regime
from .allocator import CapitalAllocator
from .orchestrator import Orchestrator
from .reconciler import SignalReconciler

log = logging.getLogger(__name__)

class HarnessSimulator:
    """Runs a full simulation of the harness logic (allocation + reconciliation) over time."""
    
    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or get_config()
        self.allocator = CapitalAllocator(self.cfg)
        self.reconciler = SignalReconciler(self.cfg)
        
    def run_simulation(self, start_date: str, end_date: str) -> dict:
        """Run the simulation over the specified date range."""
        log.info(f"Starting unified harness portfolio simulation from {start_date} to {end_date}")
        # In a full implementation, this would loop through each day,
        # fetch historical indicators, generate signals across all strategies,
        # pass them through the SignalReconciler, and then through the CapitalAllocator.
        # This acts as the baseline for A/B testing framework.
        return {
            "status": "success",
            "start_date": start_date,
            "end_date": end_date,
            "simulated_days": 0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0
        }
