"""Data models for the historical VaR framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ScenarioShift:
    """
    Absolute or relative market data changes for a single historical scenario date.
    - yield_curve_shifts: curve_id -> array of absolute changes per tenor
    - vol_shifts:         surface_id -> scalar absolute change (parallel shift)
    - fx_shifts:          fx_pair_id -> percentage change in spot
    - commodity_shifts:   commodity_id -> percentage change in nearest futures price
    """
    scenario_date: date
    yield_curve_shifts: Dict[str, np.ndarray] = field(default_factory=dict)
    vol_shifts: Dict[str, float] = field(default_factory=dict)
    fx_shifts: Dict[str, float] = field(default_factory=dict)
    commodity_shifts: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    scenario_date: date
    trade_pnls: Dict[str, float] = field(default_factory=dict)  # trade_id -> P&L
    portfolio_pnl: float = 0.0


@dataclass
class VaRResult:
    confidence_level: float           # e.g. 0.95 or 0.99
    lookback_days: int
    var_amount: float                 # Negative number (loss)
    expected_shortfall: float         # CVaR — average loss beyond VaR (negative)
    position_vars: Dict[str, float] = field(default_factory=dict)  # trade_id -> VaR
    scenario_pnls: List[float] = field(default_factory=list)       # Full sorted P&L distribution
