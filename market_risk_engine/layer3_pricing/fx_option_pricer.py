"""FX Option pricer using the Garman-Kohlhagen model."""
from __future__ import annotations

import math

import scipy.stats as st

from ..common.date_utils import year_fraction
from ..common.enums import OptionType
from ..common.exceptions import PricingError
from ..layer1_market_data.vol_surface import VolSurfaceInterpolator
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import FXOption
from .base import MarketSnapshot, PricingEngine, PricingResult


def _garman_kohlhagen(S: float, K: float, r_d: float, r_f: float,
                      sigma: float, T: float, opt_type: OptionType) -> float:
    """
    Garman-Kohlhagen FX option value per unit of notional.
    S     = spot rate (quote per base)
    K     = strike
    r_d   = annually-compounded domestic (quote) risk-free rate
    r_f   = annually-compounded foreign (base) risk-free rate
    sigma = lognormal vol
    T     = time to expiry in years
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if opt_type == OptionType.CALL else max(K - S, 0.0)
        return intrinsic

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (math.log1p(r_d) - math.log1p(r_f) + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if opt_type == OptionType.CALL:
        return (S * (1.0 + r_f) ** (-T) * st.norm.cdf(d1)
                - K * (1.0 + r_d) ** (-T) * st.norm.cdf(d2))
    else:  # PUT
        return (K * (1.0 + r_d) ** (-T) * st.norm.cdf(-d2)
                - S * (1.0 + r_f) ** (-T) * st.norm.cdf(-d1))


class FXOptionPricer(PricingEngine):
    """Price vanilla FX options using the Garman-Kohlhagen formula."""

    def price(self, trade: FXOption, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.quote_currency, error=str(exc)
            )

    def _price(self, trade: FXOption, market: MarketSnapshot) -> PricingResult:
        pair = f"{trade.base_currency}{trade.quote_currency}"
        if pair not in market.fx_rates:
            raise PricingError(f"FX rate '{pair}' not in market snapshot.")
        if trade.vol_surface_id not in market.vol_surfaces:
            raise PricingError(f"Missing vol surface '{trade.vol_surface_id}'.")

        fx = market.fx_rates[pair]
        base_disc = YieldCurveInterpolator(market.yield_curves[trade.base_discount_curve_id])
        quote_disc = YieldCurveInterpolator(market.yield_curves[trade.quote_discount_curve_id])
        vol_interp = VolSurfaceInterpolator(market.vol_surfaces[trade.vol_surface_id])
        today = market.as_of_date

        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        expiry = cal.adjust(trade.expiry_date, trade.business_day_convention) if cal else trade.expiry_date
        T = year_fraction(today, expiry, "ACT365")
        r_f = base_disc.zero_rate(T) if T > 0 else 0.0
        r_d = quote_disc.zero_rate(T) if T > 0 else 0.0

        sigma = vol_interp.get_vol(max(T, 1e-4), trade.strike)
        S = fx.spot

        unit_value = _garman_kohlhagen(S, trade.strike, r_d, r_f, sigma, T,
                                       trade.option_type)
        npv = unit_value * trade.notional_base

        # Delta (spot delta in quote currency)
        sqrt_T = math.sqrt(max(T, 1e-9))
        d1 = (math.log(S / trade.strike) + (math.log1p(r_d) - math.log1p(r_f) + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        if trade.option_type == OptionType.CALL:
            delta = (1.0 + r_f) ** (-T) * st.norm.cdf(d1) * trade.notional_base
        else:
            delta = -(1.0 + r_f) ** (-T) * st.norm.cdf(-d1) * trade.notional_base

        # Vega
        unit_bumped = _garman_kohlhagen(S, trade.strike, r_d, r_f, sigma + 0.01, T,
                                        trade.option_type)
        vega = (unit_bumped - unit_value) * trade.notional_base

        return PricingResult(
            trade_id=trade.trade_id,
            npv=npv,
            currency=trade.quote_currency,
            delta=delta,
            vega=vega,
        )
