"""Parse an XML portfolio file into trade dataclass instances."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional

from lxml import etree

from ..common.enums import BusinessDayConvention, OptionType, PayReceive
from ..common.exceptions import PortfolioParseError
from .models import (
    AmortizingIRS, CapFloor, CommodityFuturesOption, CommoditySwap,
    FloatFloatSwap, FXForward, FXOption, IRS, Portfolio, Swaption,
)


def _date(el: etree._Element, tag: str) -> date:
    text = el.findtext(tag)
    if text is None:
        raise PortfolioParseError(f"Missing required element <{tag}>.")
    return date.fromisoformat(text.strip())


def _float(el: etree._Element, tag: str) -> float:
    text = el.findtext(tag)
    if text is None:
        raise PortfolioParseError(f"Missing required element <{tag}>.")
    return float(text.strip())


def _str(el: etree._Element, tag: str) -> str:
    text = el.findtext(tag)
    if text is None:
        raise PortfolioParseError(f"Missing required element <{tag}>.")
    return text.strip()


def _opt_str(el: etree._Element, tag: str) -> Optional[str]:
    text = el.findtext(tag)
    return text.strip() if text else None


def _bdc(el: etree._Element) -> BusinessDayConvention:
    """Read <BusinessDayConvention>; defaults to MODIFIED_FOLLOWING if absent."""
    text = el.findtext("BusinessDayConvention")
    return BusinessDayConvention(text.strip()) if text else BusinessDayConvention.MODIFIED_FOLLOWING


# ---------------------------------------------------------------------------
# Per-type parsers
# ---------------------------------------------------------------------------

def _parse_irs(el: etree._Element, trade_id: str, ns_id: Optional[str]) -> IRS:
    return IRS(
        trade_id=trade_id,
        currency=_str(el, "Currency"),
        notional=_float(el, "Notional"),
        effective_date=_date(el, "EffectiveDate"),
        maturity_date=_date(el, "MaturityDate"),
        fixed_rate=_float(el, "FixedRate"),
        fixed_day_count=_str(el, "FixedDayCount"),
        float_index=_str(el, "FloatIndex"),
        float_day_count=_str(el, "FloatDayCount"),
        payment_frequency=_str(el, "PaymentFrequency"),
        pay_receive=PayReceive(_str(el, "PayReceive")),
        discount_curve_id=_str(el, "DiscountCurveId"),
        forward_curve_id=_str(el, "ForwardCurveId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_capfloor(el: etree._Element, trade_id: str, ns_id: Optional[str]) -> CapFloor:
    return CapFloor(
        trade_id=trade_id,
        currency=_str(el, "Currency"),
        notional=_float(el, "Notional"),
        effective_date=_date(el, "EffectiveDate"),
        maturity_date=_date(el, "MaturityDate"),
        strike=_float(el, "Strike"),
        option_type=OptionType(_str(el, "OptionType")),
        float_index=_str(el, "FloatIndex"),
        day_count=_str(el, "DayCount"),
        payment_frequency=_str(el, "PaymentFrequency"),
        vol_surface_id=_str(el, "VolSurfaceId"),
        discount_curve_id=_str(el, "DiscountCurveId"),
        forward_curve_id=_str(el, "ForwardCurveId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_swaption(el: etree._Element, trade_id: str, ns_id: Optional[str]) -> Swaption:
    return Swaption(
        trade_id=trade_id,
        currency=_str(el, "Currency"),
        notional=_float(el, "Notional"),
        option_expiry=_date(el, "OptionExpiry"),
        underlying_start=_date(el, "UnderlyingStart"),
        underlying_maturity=_date(el, "UnderlyingMaturity"),
        strike=_float(el, "Strike"),
        option_type=OptionType(_str(el, "OptionType")),
        vol_model=_str(el, "VolModel"),
        vol_surface_id=_str(el, "VolSurfaceId"),
        discount_curve_id=_str(el, "DiscountCurveId"),
        forward_curve_id=_str(el, "ForwardCurveId"),
        payment_frequency=el.findtext("PaymentFrequency", "SEMIANNUAL").strip(),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_fxforward(el: etree._Element, trade_id: str, ns_id: Optional[str]) -> FXForward:
    return FXForward(
        trade_id=trade_id,
        base_currency=_str(el, "BaseCurrency"),
        quote_currency=_str(el, "QuoteCurrency"),
        notional_base=_float(el, "NotionalBase"),
        delivery_date=_date(el, "DeliveryDate"),
        forward_rate_contractual=_float(el, "ForwardRateContractual"),
        pay_receive=PayReceive(_str(el, "PayReceive")),
        base_discount_curve_id=_str(el, "BaseDiscountCurveId"),
        quote_discount_curve_id=_str(el, "QuoteDiscountCurveId"),
        fx_rate_id=_str(el, "FxRateId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_fxoption(el: etree._Element, trade_id: str, ns_id: Optional[str]) -> FXOption:
    return FXOption(
        trade_id=trade_id,
        base_currency=_str(el, "BaseCurrency"),
        quote_currency=_str(el, "QuoteCurrency"),
        notional_base=_float(el, "NotionalBase"),
        expiry_date=_date(el, "ExpiryDate"),
        delivery_date=_date(el, "DeliveryDate"),
        strike=_float(el, "Strike"),
        option_type=OptionType(_str(el, "OptionType")),
        vol_surface_id=_str(el, "VolSurfaceId"),
        base_discount_curve_id=_str(el, "BaseDiscountCurveId"),
        quote_discount_curve_id=_str(el, "QuoteDiscountCurveId"),
        fx_rate_id=_str(el, "FxRateId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_commodity_swap(el: etree._Element, trade_id: str,
                           ns_id: Optional[str]) -> CommoditySwap:
    return CommoditySwap(
        trade_id=trade_id,
        commodity_id=_str(el, "CommodityId"),
        notional_quantity=_float(el, "NotionalQuantity"),
        effective_date=_date(el, "EffectiveDate"),
        maturity_date=_date(el, "MaturityDate"),
        fixed_price=_float(el, "FixedPrice"),
        pay_receive=PayReceive(_str(el, "PayReceive")),
        payment_frequency=_str(el, "PaymentFrequency"),
        commodity_curve_id=_str(el, "CommodityCurveId"),
        discount_curve_id=_str(el, "DiscountCurveId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_commodity_futures_option(el: etree._Element, trade_id: str,
                                     ns_id: Optional[str]) -> CommodityFuturesOption:
    return CommodityFuturesOption(
        trade_id=trade_id,
        commodity_id=_str(el, "CommodityId"),
        notional_quantity=_float(el, "NotionalQuantity"),
        futures_maturity=_date(el, "FuturesMaturity"),
        option_expiry=_date(el, "OptionExpiry"),
        strike=_float(el, "Strike"),
        option_type=OptionType(_str(el, "OptionType")),
        vol_surface_id=_str(el, "VolSurfaceId"),
        discount_curve_id=_str(el, "DiscountCurveId"),
        commodity_curve_id=_str(el, "CommodityCurveId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_amortizing_irs(el: etree._Element, trade_id: str,
                           ns_id: Optional[str]) -> AmortizingIRS:
    """
    XML structure for amortizing notional schedule::

        <NotionalSchedule>
            <Entry date="2024-07-15" notional="8000000"/>
            <Entry date="2025-01-15" notional="6000000"/>
            ...
        </NotionalSchedule>
    """
    sched_el = el.find("NotionalSchedule")
    notional_schedule = []
    if sched_el is not None:
        for entry in sched_el.findall("Entry"):
            entry_date = date.fromisoformat(entry.get("date", "").strip())
            entry_notional = float(entry.get("notional", "0").strip())
            notional_schedule.append((entry_date, entry_notional))

    return AmortizingIRS(
        trade_id=trade_id,
        currency=_str(el, "Currency"),
        initial_notional=_float(el, "InitialNotional"),
        notional_schedule=notional_schedule,
        effective_date=_date(el, "EffectiveDate"),
        maturity_date=_date(el, "MaturityDate"),
        fixed_rate=_float(el, "FixedRate"),
        fixed_day_count=_str(el, "FixedDayCount"),
        float_index=_str(el, "FloatIndex"),
        float_day_count=_str(el, "FloatDayCount"),
        payment_frequency=_str(el, "PaymentFrequency"),
        pay_receive=PayReceive(_str(el, "PayReceive")),
        discount_curve_id=_str(el, "DiscountCurveId"),
        forward_curve_id=_str(el, "ForwardCurveId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


def _parse_float_float_swap(el: etree._Element, trade_id: str,
                             ns_id: Optional[str]) -> FloatFloatSwap:
    return FloatFloatSwap(
        trade_id=trade_id,
        currency=_str(el, "Currency"),
        notional=_float(el, "Notional"),
        effective_date=_date(el, "EffectiveDate"),
        maturity_date=_date(el, "MaturityDate"),
        leg1_index=_str(el, "Leg1Index"),
        leg1_day_count=_str(el, "Leg1DayCount"),
        leg1_frequency=_str(el, "Leg1Frequency"),
        leg1_forward_curve_id=_str(el, "Leg1ForwardCurveId"),
        leg1_spread=float(el.findtext("Leg1Spread", "0").strip()),
        leg2_index=_str(el, "Leg2Index"),
        leg2_day_count=_str(el, "Leg2DayCount"),
        leg2_frequency=_str(el, "Leg2Frequency"),
        leg2_forward_curve_id=_str(el, "Leg2ForwardCurveId"),
        leg2_spread=float(el.findtext("Leg2Spread", "0").strip()),
        pay_receive=PayReceive(_str(el, "PayReceive")),
        discount_curve_id=_str(el, "DiscountCurveId"),
        netting_set_id=ns_id,
        calendar_name=_opt_str(el, "CalendarName"),
        business_day_convention=_bdc(el),
    )


_PARSERS = {
    "IRS": _parse_irs,
    "AmortizingIRS": _parse_amortizing_irs,
    "FloatFloatSwap": _parse_float_float_swap,
    "CapFloor": _parse_capfloor,
    "Swaption": _parse_swaption,
    "FXForward": _parse_fxforward,
    "FXOption": _parse_fxoption,
    "CommoditySwap": _parse_commodity_swap,
    "CommodityFuturesOption": _parse_commodity_futures_option,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_portfolio(
    xml_path: str,
    schema_path: Optional[str] = None,
) -> Portfolio:
    """
    Parse an XML portfolio file and return a Portfolio object.

    Args:
        xml_path: Path to the portfolio XML file.
        schema_path: Optional path to the XSD schema for validation.
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    if schema_path:
        schema_doc = etree.parse(schema_path)
        schema = etree.XMLSchema(schema_doc)
        if not schema.validate(tree):
            errors = "\n".join(str(e) for e in schema.error_log)
            raise PortfolioParseError(f"Portfolio XML failed schema validation:\n{errors}")

    portfolio_id = root.get("id", "UNKNOWN")
    as_of_str = root.get("asOfDate")
    as_of = date.fromisoformat(as_of_str) if as_of_str else date.today()

    # Build netting set map: trade_id -> netting_set_id
    trade_to_ns: Dict[str, str] = {}
    for ns_el in root.findall("NettingSet"):
        ns_id = ns_el.get("id", "")
        for ref in ns_el.findall("TradeRef"):
            if ref.text:
                trade_to_ns[ref.text.strip()] = ns_id

    portfolio = Portfolio(portfolio_id=portfolio_id, as_of_date=as_of)

    for trade_el in root.findall("Trade"):
        trade_id = trade_el.get("id", "")
        trade_type = trade_el.get("type", "")
        ns_id = trade_to_ns.get(trade_id)

        parser_fn = _PARSERS.get(trade_type)
        if parser_fn is None:
            raise PortfolioParseError(
                f"Unsupported trade type '{trade_type}' for trade '{trade_id}'."
            )
        trade = parser_fn(trade_el, trade_id, ns_id)
        portfolio.add_trade(trade)

    return portfolio
