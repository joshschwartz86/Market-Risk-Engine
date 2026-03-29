"""Cap and Floor pricer using Black's model on each caplet/floorlet."""
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


class CapFloorPricer(PricingEngine):
    """Sum Black-76 caplets/floorlets to price a cap or floor."""

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

        schedule = generate_schedule(trade.effective_date, trade.maturity_date,
                                     trade.payment_frequency)
        prev = trade.effective_date
        total_pv = 0.0
        total_vega = 0.0

        for pay_date in schedule:
            if pay_date <= today:
                prev = pay_date
                continue
            t_prev = max(0.0, year_fraction(today, prev, trade.day_count))
            t_pay = year_fraction(today, pay_date, trade.day_count)
            tau = year_fraction(prev, pay_date, trade.day_count)

            F = fwd_interp.forward_rate(
                t_prev if t_prev > 1e-6 else 1e-6, t_pay
            )
            T_exp = t_prev  # expiry is the start of the accrual period
            df = disc.discount_factor(t_pay)
            sigma = vol_interp.get_vol(T_exp if T_exp > 1e-6 else 1e-4, trade.strike)

            pv = _black_option(F, trade.strike, sigma, T_exp, df, tau,
                               trade.notional, trade.option_type)
            total_pv += pv

            # Vega: reprice with +1% vol bump
            pv_bumped = _black_option(F, trade.strike, sigma + 0.01, T_exp, df, tau,
                                      trade.notional, trade.option_type)
            total_vega += pv_bumped - pv

            prev = pay_date

        return PricingResult(
            trade_id=trade.trade_id,
            npv=total_pv,
            currency=trade.currency,
            vega=total_vega,
        )
