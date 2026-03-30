"""FX Forward pricer using covered interest-rate parity."""
from __future__ import annotations

from ..common.date_utils import year_fraction
from ..common.enums import PayReceive
from ..common.exceptions import PricingError
from ..layer1_market_data.yield_curve import YieldCurveInterpolator
from ..layer2_portfolio.models import FXForward
from .base import MarketSnapshot, PricingEngine, PricingResult


class FXForwardPricer(PricingEngine):
    """
    NPV = Notional_base × (F_theoretical − F_contractual) × DF_quote(T)

    Sign convention: PAY base (buy base forward) is long the forward.
    """

    def price(self, trade: FXForward, market: MarketSnapshot) -> PricingResult:  # type: ignore[override]
        try:
            return self._price(trade, market)
        except Exception as exc:
            return PricingResult(
                trade_id=trade.trade_id, npv=float("nan"),
                currency=trade.quote_currency, error=str(exc)
            )

    def _price(self, trade: FXForward, market: MarketSnapshot) -> PricingResult:
        pair = f"{trade.base_currency}{trade.quote_currency}"
        if pair not in market.fx_rates:
            raise PricingError(f"FX rate '{pair}' not in market snapshot.")
        if trade.base_discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing base discount curve '{trade.base_discount_curve_id}'.")
        if trade.quote_discount_curve_id not in market.yield_curves:
            raise PricingError(f"Missing quote discount curve '{trade.quote_discount_curve_id}'.")

        fx = market.fx_rates[pair]
        base_disc = YieldCurveInterpolator(market.yield_curves[trade.base_discount_curve_id])
        quote_disc = YieldCurveInterpolator(market.yield_curves[trade.quote_discount_curve_id])
        today = market.as_of_date

        cal = market.calendars.get(trade.calendar_name) if trade.calendar_name else None
        delivery = cal.adjust(trade.delivery_date, trade.business_day_convention) if cal else trade.delivery_date
        T = year_fraction(today, delivery, "ACT360")
        if T <= 0:
            return PricingResult(trade_id=trade.trade_id, npv=0.0,
                                 currency=trade.quote_currency)

        df_base = base_disc.discount_factor(T)
        df_quote = quote_disc.discount_factor(T)
        S = fx.spot
        F_theo = S * df_base / df_quote

        # PAY means we agreed to pay base and receive quote at F_contractual
        # Value = notional_base × (F_theo − F_contractual) × DF_quote
        sign = 1.0 if trade.pay_receive == PayReceive.PAY else -1.0
        npv = sign * trade.notional_base * (F_theo - trade.forward_rate_contractual) * df_quote

        delta = sign * trade.notional_base * df_base  # sensitivity to spot

        return PricingResult(
            trade_id=trade.trade_id,
            npv=npv,
            currency=trade.quote_currency,
            delta=delta,
        )
