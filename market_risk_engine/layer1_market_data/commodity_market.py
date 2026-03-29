"""Commodity futures curve utilities."""
from __future__ import annotations

import numpy as np

from .models import CommodityCurve
from ..common.exceptions import MarketDataError


def implied_convenience_yield(
    curve: CommodityCurve,
    risk_free_rate: float,
    storage_cost: float = 0.0,
) -> list[float]:
    """
    Infer the convenience yield δ(t) from the futures curve via:
        F(t) = S * exp((r + c - δ) * t)
    => δ(t) = r + c - ln(F(t)/S) / t
    where S = F(0) (shortest maturity used as spot proxy).
    """
    if len(curve.maturities) < 2:
        raise MarketDataError("Need at least two maturities to infer convenience yield.")

    spot = curve.futures_prices[0]
    yields = []
    for t, F in zip(curve.maturities[1:], curve.futures_prices[1:]):
        if t <= 0 or F <= 0 or spot <= 0:
            yields.append(0.0)
            continue
        cy = risk_free_rate + storage_cost - np.log(F / spot) / t
        yields.append(float(cy))
    return yields


def roll_adjusted_price(curve: CommodityCurve, t: float) -> float:
    """Interpolate (linearly) the futures price at any maturity t."""
    return curve.price_at(t)
