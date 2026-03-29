"""Interest Rate Swap pricer using discounted cash flow."""
from __future__ import annotations

import math
from datetime import date

from ..common.date_utils import generate_schedule, year_fraction
from ..common.exceptions import PricingError
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import IRS, PayReceive
from .base import MarketSnapshot, PricingEngine, PricingResult


class IRSPricer(PricingEngine):
    """Price a vanilla fixed-for-floating interest rate swap."""

    def price(self, trade: IRS, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.currency, error=str(exc)
            )

    def _price_legs(self, trade: IRS, market: MarketSnapshot) -> tuple[float, float]:
        """Return (fixed_pv, float_pv) without computing greeks."""
        if trade.discount_curve_id not in market.yield_curves:
            raise PricingError(
                f"Discount curve '{trade.discount_curve_id}' not in market snapshot."
            )
        if trade.forward_curve_id not in market.yield_curves:
            raise PricingError(
                f"Forward curve '{trade.forward_curve_id}' not in market snapshot."
            )
        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        fwd = YieldCurveInterpolator(market.yield_curves[trade.forward_curve_id])
        today = market.as_of_date
        fixed_pv = self._fixed_leg_pv(trade, disc, today)
        float_pv = self._float_leg_pv(trade, disc, fwd, today)
        return fixed_pv, float_pv

    def _price(self, trade: IRS, market: MarketSnapshot) -> PricingResult:
        fixed_pv, float_pv = self._price_legs(trade, market)

        # PAY fixed means we pay fixed, receive float → NPV = float_pv − fixed_pv
        if trade.pay_receive == PayReceive.PAY:
            npv = float_pv - fixed_pv
        else:
            npv = fixed_pv - float_pv

        pv01 = self._compute_pv01(trade, market, npv)

        return PricingResult(
            trade_id=trade.trade_id,
            npv=npv,
            currency=trade.currency,
            pv01=pv01,
        )

    def _fixed_leg_pv(self, trade: IRS, disc: YieldCurveInterpolator,
                      today: date) -> float:
        schedule = generate_schedule(trade.effective_date, trade.maturity_date,
                                     trade.payment_frequency)
        prev = trade.effective_date
        pv = 0.0
        for pay_date in schedule:
            if pay_date <= today:
                prev = pay_date
                continue
            tau = year_fraction(prev, pay_date, trade.fixed_day_count)
            t = year_fraction(today, pay_date, trade.fixed_day_count)
            df = disc.discount_factor(t)
            pv += trade.fixed_rate * tau * df * trade.notional
            prev = pay_date
        return pv

    def _float_leg_pv(self, trade: IRS, disc: YieldCurveInterpolator,
                      fwd: YieldCurveInterpolator, today: date) -> float:
        """
        Float leg PV using the standard approximation:
            PV = Σ  fwd_rate(t_{i-1}, t_i) × τ_i × DF(t_i) × N
        """
        schedule = generate_schedule(trade.effective_date, trade.maturity_date,
                                     trade.payment_frequency)
        prev = trade.effective_date
        pv = 0.0
        for pay_date in schedule:
            if pay_date <= today:
                prev = pay_date
                continue
            t_prev = max(0.0, year_fraction(today, prev, trade.float_day_count))
            t_pay = year_fraction(today, pay_date, trade.float_day_count)
            tau = year_fraction(prev, pay_date, trade.float_day_count)
            fwd_rate = fwd.forward_rate(t_prev, t_pay) if t_prev > 0 else fwd.forward_rate(0.0001, t_pay)
            df = disc.discount_factor(t_pay)
            pv += fwd_rate * tau * df * trade.notional
            prev = pay_date
        return pv

    def _compute_pv01(self, trade: IRS, market: MarketSnapshot,
                      base_npv: float) -> float:
        """Approximate DV01: reprice with +1bp parallel shift on the discount curve."""
        bump = 0.0001
        bumped_curves = dict(market.yield_curves)
        orig = market.yield_curves[trade.discount_curve_id]
        from ..layer1_market_data.models import YieldCurve
        bumped_curves[trade.discount_curve_id] = YieldCurve(
            currency=orig.currency,
            curve_name=orig.curve_name,
            as_of_date=orig.as_of_date,
            tenors=orig.tenors,
            zero_rates=[r + bump for r in orig.zero_rates],
            day_count=orig.day_count,
            interpolation=orig.interpolation,
        )
        bumped_market = MarketSnapshot(
            as_of_date=market.as_of_date,
            yield_curves=bumped_curves,
            vol_surfaces=market.vol_surfaces,
            fx_rates=market.fx_rates,
            commodity_curves=market.commodity_curves,
        )
        fixed_pv, float_pv = self._price_legs(trade, bumped_market)
        bumped_npv = float_pv - fixed_pv if trade.pay_receive == PayReceive.PAY else fixed_pv - float_pv
        return bumped_npv - base_npv
