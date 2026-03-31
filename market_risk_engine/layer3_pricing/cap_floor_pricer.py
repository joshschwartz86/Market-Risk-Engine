"""Cap and Floor pricer: Black-76 (lognormal) and Bachelier (normal) models."""
from __future__ import annotations

import math
from datetime import date

import scipy.stats as st

from ..common.date_utils import generate_schedule, year_fraction
from ..common.enums import OptionType
from ..common.exceptions import PricingError
from ..layer1_market_data.vol_surface import VolSurfaceInterpolator
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import CapFloor
from .base import MarketSnapshot, PricingEngine, PricingResult


def _sdf(disc: YieldCurveInterpolator, t: float, spread: float) -> float:
    """Spread-adjusted discount factor (annual compounding): DF(t) * (1+s)^(-t)."""
    if t <= 0.0:
        return 1.0
    df = disc.discount_factor(t)
    if spread == 0.0:
        return df
    return df * (1.0 + spread) ** (-t)


def _black_option(F: float, K: float, sigma: float, T: float,
                  df: float, tau: float, notional: float,
                  opt_type: OptionType) -> float:
    """
    Black-76 caplet/floorlet value.
    F    = forward rate for the accrual period
    K    = strike rate
    sigma= lognormal vol
    T    = time to expiry (year fraction)
    df   = discount factor to payment date
    tau  = accrual fraction for the period
    """
    if T <= 0 or sigma <= 0:
        # Intrinsic value only
        intrinsic = max(F - K, 0.0) if opt_type == OptionType.CAP else max(K - F, 0.0)
        return intrinsic * tau * df * notional

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if opt_type == OptionType.CAP:
        pv = df * tau * notional * (F * st.norm.cdf(d1) - K * st.norm.cdf(d2))
    else:  # FLOOR
        pv = df * tau * notional * (K * st.norm.cdf(-d2) - F * st.norm.cdf(-d1))
    return pv


def _bachelier_option(F: float, K: float, sigma: float, T: float,
                      df: float, tau: float, notional: float,
                      opt_type: OptionType) -> float:
    """
    Bachelier (normal) caplet/floorlet value.
    F     = forward rate for the accrual period
    K     = strike rate
    sigma = normal (absolute) volatility
    T     = time to expiry (year fraction)
    df    = discount factor to payment date
    tau   = accrual fraction for the period
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if opt_type == OptionType.CAP else max(K - F, 0.0)
        return intrinsic * tau * df * notional

    sqrt_T = math.sqrt(T)
    d = (F - K) / (sigma * sqrt_T)

    if opt_type == OptionType.CAP:
        val = (F - K) * st.norm.cdf(d) + sigma * sqrt_T * st.norm.pdf(d)
    else:  # FLOOR
        val = (K - F) * st.norm.cdf(-d) + sigma * sqrt_T * st.norm.pdf(d)

    return df * tau * notional * val


class CapFloorPricer(PricingEngine):
    """Sum caplets/floorlets to price a cap or floor; supports Black-76 and Bachelier."""

    def price(self, trade: CapFloor, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.currency, error=str(exc)
            )

    def _price(self, trade: CapFloor, market: MarketSnapshot) -> PricingResult:
        if trade.discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing discount curve '{trade.discount_curve_id}'.")
        if trade.vol_surface_id not in market.vol_surfaces:
            raise PricingError(f"Missing vol surface '{trade.vol_surface_id}'.")

        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        fwd_interp = YieldCurveInterpolator(market.yield_curves[trade.forward_curve_id])
        vol_interp = VolSurfaceInterpolator(market.vol_surfaces[trade.vol_surface_id])
        today = market.as_of_date

        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        schedule = generate_schedule(trade.effective_date, trade.maturity_date,
                                     trade.payment_frequency,
                                     calendar=cal,
                                     convention=trade.business_day_convention)
        prev = trade.effective_date
        total_pv = 0.0
        total_vega = 0.0

        s_d = trade.discount_spread
        s_f = trade.forward_spread
        use_bachelier = trade.vol_model.lower() == "bachelier"
        # Vega bump: 1bp for normal vol, 1% for lognormal vol
        vega_bump = 0.0001 if use_bachelier else 0.01

        for pay_date in schedule:
            if pay_date <= today:
                prev = pay_date
                continue
            t_prev = max(0.0, year_fraction(today, prev, trade.day_count))
            t_pay = year_fraction(today, pay_date, trade.day_count)
            tau = year_fraction(prev, pay_date, trade.day_count)

            # Additive forward spread applied on top of each period's forward rate
            F = fwd_interp.forward_rate(
                t_prev if t_prev > 1e-6 else 1e-6, t_pay
            ) + s_f
            T_exp = t_prev  # expiry is the start of the accrual period
            # Spread-adjusted discount factor to payment date
            df = _sdf(disc, t_pay, s_d)
            sigma = vol_interp.get_vol(T_exp if T_exp > 1e-6 else 1e-4, trade.strike)

            if use_bachelier:
                pv = _bachelier_option(F, trade.strike, sigma, T_exp, df, tau,
                                       trade.notional, trade.option_type)
                pv_bumped = _bachelier_option(F, trade.strike, sigma + vega_bump,
                                              T_exp, df, tau,
                                              trade.notional, trade.option_type)
            else:
                pv = _black_option(F, trade.strike, sigma, T_exp, df, tau,
                                   trade.notional, trade.option_type)
                pv_bumped = _black_option(F, trade.strike, sigma + vega_bump,
                                          T_exp, df, tau,
                                          trade.notional, trade.option_type)
            total_pv += pv
            total_vega += pv_bumped - pv

            prev = pay_date

        return PricingResult(
            trade_id=trade.trade_id,
            npv=total_pv,
            currency=trade.currency,
            vega=total_vega,
        )
