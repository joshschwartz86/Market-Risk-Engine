"""FX spot and forward rate utilities."""
from __future__ import annotations

import math
from typing import Optional

from .models import FXRate
from .yield_curve import YieldCurveInterpolator
from ..common.exceptions import MarketDataError


def implied_forward(
    spot: float,
    base_interp: YieldCurveInterpolator,
    quote_interp: YieldCurveInterpolator,
    t: float,
) -> float:
    """
    Compute the no-arbitrage FX forward via covered interest rate parity:
        F = S * DF_base(t) / DF_quote(t)
    where DF_base discounts in the base currency and DF_quote in the quote currency.
    """
    df_base = base_interp.discount_factor(t)
    df_quote = quote_interp.discount_factor(t)
    if df_quote == 0:
        raise MarketDataError("Quote currency discount factor is zero.")
    return spot * df_base / df_quote


def cross_rate(
    fx_ab: FXRate,     # base=A, quote=B
    fx_cb: FXRate,     # base=C, quote=B
) -> float:
    """
    Compute the A/C spot cross rate given two pairs sharing the B quote currency.
    A/C = (A/B) / (C/B)
    """
    if fx_ab.quote_currency != fx_cb.quote_currency:
        raise MarketDataError(
            f"Cannot compute cross rate: {fx_ab.pair} and {fx_cb.pair} "
            "do not share a common quote currency."
        )
    return fx_ab.spot / fx_cb.spot
