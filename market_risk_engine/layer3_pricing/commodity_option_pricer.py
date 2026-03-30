"""Commodity Futures Option pricer using Black-76 model."""
from __future__ import annotations

import math

import scipy.stats as st

from ..common.date_utils import year_fraction
from ..common.enums import OptionType
from ..common.exceptions import PricingError
from ..layer1_market_data.vol_surface import VolSurfaceInterpolator
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import CommodityFuturesOption
from .base import MarketSnapshot, PricingEngine, PricingResult


def _black76(F: float, K: float, sigma: float, T: float,
             df: float, qty: float, opt_type: OptionType) -> float:
    """
    Black-76 option on a futures price.
    F  = current futures price
    K  = strike
    df = discount factor to option expiry
    """
    if T <= 0:
        intrinsic = max(F - K, 0.0) if opt_type == OptionType.CALL else max(K - F, 0.0)
        return intrinsic * df * qty

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if opt_type == OptionType.CALL:
        return df * qty * (F * st.norm.cdf(d1) - K * st.norm.cdf(d2))
    else:  # PUT
        return df * qty * (K * st.norm.cdf(-d2) - F * st.norm.cdf(-d1))


class CommodityFuturesOptionPricer(PricingEngine):
    """Price options on commodity futures using Black-76."""

    def price(self, trade: CommodityFuturesOption,  # type: ignore[override]
              market: MarketSnapshot) -> PricingResult:
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency="USD", error=str(exc)
            )

    def _price(self, trade: CommodityFuturesOption,
               market: MarketSnapshot) -> PricingResult:
        if trade.commodity_curve_id not in market.commodity_curves:
            raise PricingError(f"Missing commodity curve '{trade.commodity_curve_id}'.")
        if trade.vol_surface_id not in market.vol_surfaces:
            raise PricingError(f"Missing vol surface '{trade.vol_surface_id}'.")
        if trade.discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing discount curve '{trade.discount_curve_id}'.")

        comm = market.commodity_curves[trade.commodity_curve_id]
        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        vol_interp = VolSurfaceInterpolator(market.vol_surfaces[trade.vol_surface_id])
        today = market.as_of_date

        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        option_expiry = cal.adjust(trade.option_expiry, trade.business_day_convention) if cal else trade.option_expiry
        futures_maturity = cal.adjust(trade.futures_maturity, trade.business_day_convention) if cal else trade.futures_maturity
        T = year_fraction(today, option_expiry, "ACT365")
        t_fut = year_fraction(today, futures_maturity, "ACT365")
        F = comm.price_at(max(t_fut, 1e-6))
        df = disc.discount_factor(max(T, 1e-6))
        sigma = vol_interp.get_vol(max(T, 1e-4), trade.strike)

        npv = _black76(F, trade.strike, sigma, max(T, 0.0), df,
                       trade.notional_quantity, trade.option_type)

        # Vega
        npv_bumped = _black76(F, trade.strike, sigma + 0.01, max(T, 0.0), df,
                               trade.notional_quantity, trade.option_type)
        vega = npv_bumped - npv

        # Delta
        delta_sign = 1.0 if trade.option_type == OptionType.CALL else -1.0
        if T > 0 and sigma > 0:
            d1 = (math.log(F / trade.strike) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
            delta = delta_sign * df * st.norm.cdf(delta_sign * d1) * trade.notional_quantity
        else:
            delta = 0.0

        return PricingResult(
            trade_id=trade.trade_id,
            npv=npv,
            currency="USD",
            delta=delta,
            vega=vega,
        )
