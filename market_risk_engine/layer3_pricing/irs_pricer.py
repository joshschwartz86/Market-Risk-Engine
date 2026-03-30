"""Interest Rate Swap pricers: vanilla IRS, amortizing IRS, and float-float (basis) swap."""
from __future__ import annotations

import math
from datetime import date
from typing import Callable

from ..common.date_utils import generate_schedule, year_fraction
from ..common.exceptions import PricingError
from ..layer1_market_data.models import YieldCurve
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import AmortizingIRS, FloatFloatSwap, IRS, PayReceive
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
        fixed_pv = self._fixed_leg_pv(trade, disc, today, market)
        float_pv = self._float_leg_pv(trade, disc, fwd, today, market)
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
                      today: date, market: MarketSnapshot = None) -> float:
        cal = market.calendars.get(trade.calendar_name) if (market and trade.calendar_name) else None
        schedule = generate_schedule(trade.effective_date, trade.maturity_date,
                                     trade.payment_frequency,
                                     calendar=cal,
                                     convention=trade.business_day_convention)
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
                      fwd: YieldCurveInterpolator, today: date,
                      market: MarketSnapshot = None) -> float:
        """
        Float leg PV using the standard approximation:
            PV = Σ  fwd_rate(t_{i-1}, t_i) × τ_i × DF(t_i) × N
        """
        cal = market.calendars.get(trade.calendar_name) if (market and trade.calendar_name) else None
        schedule = generate_schedule(trade.effective_date, trade.maturity_date,
                                     trade.payment_frequency,
                                     calendar=cal,
                                     convention=trade.business_day_convention)
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
            calendars=market.calendars,
        )
        fixed_pv, float_pv = self._price_legs(trade, bumped_market)
        bumped_npv = float_pv - fixed_pv if trade.pay_receive == PayReceive.PAY else fixed_pv - float_pv
        return bumped_npv - base_npv


# ---------------------------------------------------------------------------
# Shared helper: price a single floating leg
# ---------------------------------------------------------------------------

def _float_leg_pv(
    disc: YieldCurveInterpolator,
    fwd: YieldCurveInterpolator,
    effective_date: date,
    maturity_date: date,
    frequency: str,
    day_count: str,
    today: date,
    notional_fn: Callable[[date], float],
    spread: float = 0.0,
    calendar=None,
    convention=None,
) -> float:
    """
    Generic floating-leg PV.

    ``notional_fn(payment_date)`` returns the outstanding notional for the
    accrual period ending on that date — allows constant or amortizing schedules.
    ``spread`` is an additive rate spread applied to every period.
    """
    from ..common.enums import BusinessDayConvention as _BDC
    _conv = convention if convention is not None else _BDC.MODIFIED_FOLLOWING
    schedule = generate_schedule(effective_date, maturity_date, frequency,
                                 calendar=calendar, convention=_conv)
    prev = effective_date
    pv = 0.0
    for pay_date in schedule:
        if pay_date <= today:
            prev = pay_date
            continue
        t_prev = max(0.0, year_fraction(today, prev, day_count))
        t_pay = year_fraction(today, pay_date, day_count)
        tau = year_fraction(prev, pay_date, day_count)
        fwd_rate = fwd.forward_rate(t_prev if t_prev > 1e-6 else 1e-6, t_pay)
        df = disc.discount_factor(t_pay)
        notional = notional_fn(pay_date)
        pv += (fwd_rate + spread) * tau * df * notional
        prev = pay_date
    return pv


def _fixed_leg_pv(
    disc: YieldCurveInterpolator,
    effective_date: date,
    maturity_date: date,
    frequency: str,
    day_count: str,
    today: date,
    fixed_rate: float,
    notional_fn: Callable[[date], float],
    calendar=None,
    convention=None,
) -> float:
    """Generic fixed-leg PV with support for amortizing notionals."""
    from ..common.enums import BusinessDayConvention as _BDC
    _conv = convention if convention is not None else _BDC.MODIFIED_FOLLOWING
    schedule = generate_schedule(effective_date, maturity_date, frequency,
                                 calendar=calendar, convention=_conv)
    prev = effective_date
    pv = 0.0
    for pay_date in schedule:
        if pay_date <= today:
            prev = pay_date
            continue
        tau = year_fraction(prev, pay_date, day_count)
        t = year_fraction(today, pay_date, day_count)
        df = disc.discount_factor(t)
        notional = notional_fn(pay_date)
        pv += fixed_rate * tau * df * notional
        prev = pay_date
    return pv


# ---------------------------------------------------------------------------
# Amortizing IRS pricer
# ---------------------------------------------------------------------------

class AmortizingIRSPricer(PricingEngine):
    """
    Price a fixed-for-floating swap whose notional steps down (or up) over time.

    Identical to ``IRSPricer`` except each accrual period uses the notional
    returned by ``trade.notional_at(payment_date)``.
    """

    def price(self, trade: AmortizingIRS, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.currency, error=str(exc),
            )

    def _compute_npv(self, trade: AmortizingIRS, market: MarketSnapshot) -> float:
        if trade.discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing discount curve '{trade.discount_curve_id}'.")
        if trade.forward_curve_id not in market.yield_curves:
            raise PricingError(f"Missing forward curve '{trade.forward_curve_id}'.")
        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        fwd = YieldCurveInterpolator(market.yield_curves[trade.forward_curve_id])
        today = market.as_of_date
        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        conv = trade.business_day_convention
        notional_fn = trade.notional_at
        fixed_pv = _fixed_leg_pv(
            disc, trade.effective_date, trade.maturity_date,
            trade.payment_frequency, trade.fixed_day_count, today,
            trade.fixed_rate, notional_fn,
            calendar=cal, convention=conv,
        )
        float_pv = _float_leg_pv(
            disc, fwd, trade.effective_date, trade.maturity_date,
            trade.payment_frequency, trade.float_day_count, today, notional_fn,
            calendar=cal, convention=conv,
        )
        return float_pv - fixed_pv if trade.pay_receive == PayReceive.PAY else fixed_pv - float_pv

    def _price(self, trade: AmortizingIRS, market: MarketSnapshot) -> PricingResult:
        npv = self._compute_npv(trade, market)
        pv01 = self._compute_pv01(trade, market, npv)
        return PricingResult(
            trade_id=trade.trade_id, npv=npv,
            currency=trade.currency, pv01=pv01,
        )

    def _compute_pv01(self, trade: AmortizingIRS, market: MarketSnapshot,
                      base_npv: float) -> float:
        bump = 0.0001
        orig = market.yield_curves[trade.discount_curve_id]
        bumped_curves = dict(market.yield_curves)
        bumped_curves[trade.discount_curve_id] = YieldCurve(
            currency=orig.currency, curve_name=orig.curve_name,
            as_of_date=orig.as_of_date, tenors=orig.tenors,
            zero_rates=[r + bump for r in orig.zero_rates],
            day_count=orig.day_count, interpolation=orig.interpolation,
        )
        bumped_market = MarketSnapshot(
            as_of_date=market.as_of_date, yield_curves=bumped_curves,
            vol_surfaces=market.vol_surfaces, fx_rates=market.fx_rates,
            commodity_curves=market.commodity_curves, calendars=market.calendars,
        )
        return self._compute_npv(trade, bumped_market) - base_npv


# ---------------------------------------------------------------------------
# Float-float (basis) swap pricer
# ---------------------------------------------------------------------------

class FloatFloatSwapPricer(PricingEngine):
    """
    Price a floating-for-floating (basis) swap.

    Each leg is priced as a generic floating leg:
      PV_leg = Σ (fwd_rate(t_{i-1}, t_i) + spread) × τ_i × DF(t_i) × N

    NPV = PV_leg2 − PV_leg1   when pay_receive == PAY (pay leg 1, receive leg 2).
    """

    def price(self, trade: FloatFloatSwap, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.currency, error=str(exc),
            )

    def _compute_npv(self, trade: FloatFloatSwap, market: MarketSnapshot) -> float:
        for curve_id in (trade.discount_curve_id,
                         trade.leg1_forward_curve_id,
                         trade.leg2_forward_curve_id):
            if curve_id not in market.yield_curves:
                raise PricingError(f"Missing curve '{curve_id}'.")
        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        fwd1 = YieldCurveInterpolator(market.yield_curves[trade.leg1_forward_curve_id])
        fwd2 = YieldCurveInterpolator(market.yield_curves[trade.leg2_forward_curve_id])
        today = market.as_of_date
        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        conv = trade.business_day_convention
        notional_fn = lambda _: trade.notional  # noqa: E731
        leg1_pv = _float_leg_pv(
            disc, fwd1, trade.effective_date, trade.maturity_date,
            trade.leg1_frequency, trade.leg1_day_count, today,
            notional_fn, spread=trade.leg1_spread,
            calendar=cal, convention=conv,
        )
        leg2_pv = _float_leg_pv(
            disc, fwd2, trade.effective_date, trade.maturity_date,
            trade.leg2_frequency, trade.leg2_day_count, today,
            notional_fn, spread=trade.leg2_spread,
            calendar=cal, convention=conv,
        )
        return leg2_pv - leg1_pv if trade.pay_receive == PayReceive.PAY else leg1_pv - leg2_pv

    def _price(self, trade: FloatFloatSwap, market: MarketSnapshot) -> PricingResult:
        npv = self._compute_npv(trade, market)
        pv01 = self._compute_pv01(trade, market, npv)
        return PricingResult(
            trade_id=trade.trade_id, npv=npv,
            currency=trade.currency, pv01=pv01,
        )

    def _compute_pv01(self, trade: FloatFloatSwap, market: MarketSnapshot,
                      base_npv: float) -> float:
        bump = 0.0001
        orig = market.yield_curves[trade.discount_curve_id]
        bumped_curves = dict(market.yield_curves)
        bumped_curves[trade.discount_curve_id] = YieldCurve(
            currency=orig.currency, curve_name=orig.curve_name,
            as_of_date=orig.as_of_date, tenors=orig.tenors,
            zero_rates=[r + bump for r in orig.zero_rates],
            day_count=orig.day_count, interpolation=orig.interpolation,
        )
        bumped_market = MarketSnapshot(
            as_of_date=market.as_of_date, yield_curves=bumped_curves,
            vol_surfaces=market.vol_surfaces, fx_rates=market.fx_rates,
            commodity_curves=market.commodity_curves, calendars=market.calendars,
        )
        return self._compute_npv(trade, bumped_market) - base_npv
