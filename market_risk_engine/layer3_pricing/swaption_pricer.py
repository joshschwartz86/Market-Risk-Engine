"""Swaption pricer: Black and Bachelier models."""
from __future__ import annotations

import math

import scipy.stats as st

from ..common.date_utils import year_fraction
from ..common.enums import OptionType
from ..common.exceptions import PricingError
from ..layer1_market_data.vol_surface import VolSurfaceInterpolator
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import Swaption
from .base import MarketSnapshot, PricingEngine, PricingResult


def _sdf(disc: YieldCurveInterpolator, t: float, spread: float) -> float:
    """Spread-adjusted discount factor (annual compounding): DF(t) * (1+s)^(-t)."""
    if t <= 0.0:
        return 1.0
    df = disc.discount_factor(t)
    if spread == 0.0:
        return df
    return df * (1.0 + spread) ** (-t)


class SwaptionPricer(PricingEngine):
    """Price European swaptions using Black or Bachelier (normal) model."""

    def price(self, trade: Swaption, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.currency, error=str(exc)
            )

    def _price(self, trade: Swaption, market: MarketSnapshot) -> PricingResult:
        if trade.discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing discount curve '{trade.discount_curve_id}'.")
        if trade.vol_surface_id not in market.vol_surfaces:
            raise PricingError(f"Missing vol surface '{trade.vol_surface_id}'.")

        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        vol_interp = VolSurfaceInterpolator(market.vol_surfaces[trade.vol_surface_id])
        today = market.as_of_date

        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        option_expiry = cal.adjust(trade.option_expiry, trade.business_day_convention) if cal else trade.option_expiry
        T_exp = year_fraction(today, option_expiry, "ACT365")
        t_start = year_fraction(today, trade.underlying_start, "ACT365")
        t_end = year_fraction(today, trade.underlying_maturity, "ACT365")

        if T_exp <= 0:
            return PricingResult(trade_id=trade.trade_id, npv=0.0, currency=trade.currency)

        # Resolve per-leg frequency overrides (fall back to payment_frequency)
        fixed_freq = trade.fixed_payment_frequency or trade.payment_frequency

        # Annuity using spread-adjusted discount factors
        s_d = trade.discount_spread
        annuity = self._annuity(disc, t_start, t_end, fixed_freq, s_d)

        # Forward swap rate from first principles using spread-adjusted DFs,
        # then add the additive forward spread
        df_start = _sdf(disc, max(t_start, 1e-9), s_d)
        df_end = _sdf(disc, t_end, s_d)
        F = ((df_start - df_end) / annuity if annuity > 0 else 0.0) + trade.forward_spread

        sigma = vol_interp.get_vol(T_exp, trade.strike, forward=F)

        if trade.vol_model.lower() == "bachelier":
            npv = self._bachelier(annuity, F, trade.strike, sigma, T_exp,
                                  trade.option_type, trade.notional)
        else:  # default: black
            npv = self._black(annuity, F, trade.strike, sigma, T_exp,
                              trade.option_type, trade.notional)

        # Vega: bump vol by +1%
        sigma_bumped = sigma + 0.01
        if trade.vol_model.lower() == "bachelier":
            npv_bumped = self._bachelier(annuity, F, trade.strike, sigma_bumped,
                                         T_exp, trade.option_type, trade.notional)
        else:
            npv_bumped = self._black(annuity, F, trade.strike, sigma_bumped,
                                     T_exp, trade.option_type, trade.notional)
        vega = npv_bumped - npv

        return PricingResult(
            trade_id=trade.trade_id,
            npv=npv,
            currency=trade.currency,
            vega=vega,
        )

    # ------------------------------------------------------------------
    def _annuity(self, disc: YieldCurveInterpolator,
                 t_start: float, t_end: float, frequency: str,
                 discount_spread: float = 0.0) -> float:
        freq_map = {"MONTHLY": 12, "QUARTERLY": 4, "SEMIANNUAL": 2, "ANNUAL": 1}
        n_per_year = freq_map[frequency.upper()]
        dt = 1.0 / n_per_year
        annuity = 0.0
        t = t_start + dt
        while t <= t_end + 1e-9:
            annuity += dt * _sdf(disc, t, discount_spread)
            t += dt
        return annuity

    def _black(self, annuity: float, F: float, K: float, sigma: float,
               T: float, opt_type: OptionType, notional: float) -> float:
        if sigma <= 0 or T <= 0:
            intrinsic = max(F - K, 0.0) if opt_type == OptionType.PAYER else max(K - F, 0.0)
            return notional * annuity * intrinsic
        sqrt_T = math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if opt_type == OptionType.PAYER:
            return notional * annuity * (F * st.norm.cdf(d1) - K * st.norm.cdf(d2))
        else:  # RECEIVER
            return notional * annuity * (K * st.norm.cdf(-d2) - F * st.norm.cdf(-d1))

    def _bachelier(self, annuity: float, F: float, K: float, sigma: float,
                   T: float, opt_type: OptionType, notional: float) -> float:
        """Bachelier (normal) model for swaptions."""
        if sigma <= 0 or T <= 0:
            intrinsic = max(F - K, 0.0) if opt_type == OptionType.PAYER else max(K - F, 0.0)
            return notional * annuity * intrinsic
        sqrt_T = math.sqrt(T)
        d = (F - K) / (sigma * sqrt_T)
        if opt_type == OptionType.PAYER:
            val = (F - K) * st.norm.cdf(d) + sigma * sqrt_T * st.norm.pdf(d)
        else:
            val = (K - F) * st.norm.cdf(-d) + sigma * sqrt_T * st.norm.pdf(d)
        return notional * annuity * val
