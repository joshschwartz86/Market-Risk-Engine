"""Shared data structures and abstract base class for all pricers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

from ..common.calendar import Calendar
from ..layer1_market_data.models import (
    CommodityCurve, FXRate, VolSurface, YieldCurve,
)
from ..layer2_portfolio.models import TradeUnion


@dataclass
class MarketSnapshot:
    """
    Immutable bundle of all market data for a single valuation date.
    Passed read-only to every pricer and used as the base for scenario shifts.
    """
    as_of_date: date
    yield_curves: Dict[str, YieldCurve] = field(default_factory=dict)
    vol_surfaces: Dict[str, VolSurface] = field(default_factory=dict)
    fx_rates: Dict[str, FXRate] = field(default_factory=dict)
    commodity_curves: Dict[str, CommodityCurve] = field(default_factory=dict)
    calendars: Dict[str, Calendar] = field(default_factory=dict)


@dataclass
class PricingResult:
    trade_id: str
    npv: float                          # Net present value in domestic/pricing currency
    currency: str
    pv01: Optional[float] = None        # DV01: NPV change per +1bp parallel shift
    vega: Optional[float] = None        # NPV change per +1% absolute vol shift
    delta: Optional[float] = None       # For FX/commodity: NPV change per unit move
    error: Optional[str] = None         # Non-None if pricing failed


class PricingEngine(ABC):
    """Abstract base class for all per-instrument-type pricers."""

    @abstractmethod
    def price(self, trade: TradeUnion, market: MarketSnapshot) -> PricingResult:
        """Price a single trade against a market snapshot."""
        ...
