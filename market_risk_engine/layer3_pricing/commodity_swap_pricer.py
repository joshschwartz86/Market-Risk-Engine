"""Commodity Swap pricer: discounted futures strip vs fixed leg."""
from __future__ import annotations

from ..common.date_utils import generate_schedule, year_fraction
from ..common.enums import PayReceive
from ..common.exceptions import PricingError
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import CommoditySwap
from .base import MarketSnapshot, PricingEngine, PricingResult


class CommoditySwapPricer(PricingEngine):
    """
    Fixed leg: Σ fixed_price × Q × τ_i × DF(t_i)
    Float leg: Σ F_futures(t_i) × Q × τ_i × DF(t_i)
    where F_futures(t_i) is interpolated from the commodity futures curve.

    PAY fixed means NPV = float_pv − fixed_pv.
    """

    def price(self, trade: CommoditySwap, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency="USD", error=str(exc)
            )

    def _price(self, trade: CommoditySwap, market: MarketSnapshot) -> PricingResult:
        if trade.commodity_curve_id not in market.commodity_curves:
            raise PricingError(f"Missing commodity curve '{trade.commodity_curve_id}'.")
        if trade.discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing discount curve '{trade.discount_curve_id}'.")

        comm = market.commodity_curves[trade.commodity_curve_id]
        disc = YieldCurveInterpolator(market.yield_curves[trade.discount_curve_id])
        today = market.as_of_date

        schedule = generate_schedule(trade.effective_date, trade.maturity_date,
                                     trade.payment_frequency)
        prev = trade.effective_date
        fixed_pv = 0.0
        float_pv = 0.0

        for pay_date in schedule:
            if pay_date <= today:
                prev = pay_date
                continue
            tau = year_fraction(prev, pay_date, "ACT360")
            t = year_fraction(today, pay_date, "ACT360")
            df = disc.discount_factor(t)
            # Use the futures price at the midpoint of the accrual period as proxy
            t_mid = max(year_fraction(today, prev, "ACT360") + tau / 2, 1e-6)
            F_futures = comm.price_at(t_mid)

            fixed_pv += trade.fixed_price * trade.notional_quantity * tau * df
            float_pv += F_futures * trade.notional_quantity * tau * df
            prev = pay_date

        if trade.pay_receive == PayReceive.PAY:
            npv = float_pv - fixed_pv
        else:
            npv = fixed_pv - float_pv

        return PricingResult(
            trade_id=trade.trade_id,
            npv=npv,
            currency="USD",
        )
