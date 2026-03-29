from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np


@dataclass
class YieldCurve:
    """Continuously-compounded zero-rate curve for a single currency / index."""

    currency: str
    curve_name: str
    as_of_date: date
    tenors: List[float]       # year fractions
    zero_rates: List[float]   # continuously-compounded rates
    day_count: str = "ACT360"
    interpolation: str = "cubic_spline"

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.zero_rates):
            raise ValueError("tenors and zero_rates must have the same length.")
        if self.tenors != sorted(self.tenors):
            raise ValueError("tenors must be sorted in ascending order.")


@dataclass
class VolSurface:
    """Implied-volatility surface grid (strikes × expiries)."""

    asset_id: str
    as_of_date: date
    strikes: List[float]      # absolute strike or moneyness values
    expiries: List[float]     # year fractions
    vols: np.ndarray          # shape (len(expiries), len(strikes))
    vol_type: str = "lognormal"   # "lognormal" or "normal"
    sabr_params: Optional[Dict[float, Dict[str, float]]] = None
    # sabr_params keyed by expiry: {"alpha", "beta", "rho", "nu"}

    def __post_init__(self) -> None:
        expected = (len(self.expiries), len(self.strikes))
        if self.vols.shape != expected:
            raise ValueError(
                f"vols shape {self.vols.shape} does not match "
                f"(len(expiries)={len(self.expiries)}, len(strikes)={len(self.strikes)})."
            )


@dataclass
class FXRate:
    """Spot FX rate and forward-point term structure."""

    base_currency: str
    quote_currency: str
    as_of_date: date
    spot: float
    tenors: List[float] = field(default_factory=list)          # year fractions
    forward_points: List[float] = field(default_factory=list)  # in pips
    pip_factor: float = 10_000.0

    @property
    def pair(self) -> str:
        return f"{self.base_currency}{self.quote_currency}"

    def forward_rate(self, tenor: float) -> float:
        """Linear-interpolate forward points to get the outright forward rate."""
        if not self.tenors:
            return self.spot
        fps = np.interp(tenor, self.tenors, self.forward_points)
        return self.spot + fps / self.pip_factor


@dataclass
class CommodityCurve:
    """Futures-price term structure for a commodity."""

    commodity_id: str
    as_of_date: date
    maturities: List[float]        # year fractions to each futures expiry
    futures_prices: List[float]    # corresponding futures prices
    unit: str = ""                 # e.g. "USD/bbl", "USD/MMBtu"

    def __post_init__(self) -> None:
        if len(self.maturities) != len(self.futures_prices):
            raise ValueError("maturities and futures_prices must have the same length.")

    def price_at(self, t: float) -> float:
        """Linear-interpolate the futures price at maturity t."""
        return float(np.interp(t, self.maturities, self.futures_prices))
