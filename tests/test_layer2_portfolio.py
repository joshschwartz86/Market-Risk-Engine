"""Unit tests for Layer 2 — Portfolio Import."""
import os
from datetime import date

import pytest

from market_risk_engine.layer2_portfolio.models import IRS, Portfolio, PayReceive
from market_risk_engine.layer2_portfolio.xml_parser import parse_portfolio
from market_risk_engine.layer2_portfolio.portfolio import summary, group_by_type


SAMPLE_XML = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample", "portfolio_sample.xml"
)
SCHEMA_XSD = os.path.join(
    os.path.dirname(__file__), "..", "data", "schemas", "portfolio_schema.xsd"
)


def test_parse_portfolio_trade_count():
    portfolio = parse_portfolio(SAMPLE_XML)
    assert len(portfolio.trades) == 8  # 2 IRS + 1 Swaption + 1 Cap + 1 FXFwd + 1 FXOpt + 1 CommSwap + 1 CommOpt


def test_parse_portfolio_types():
    portfolio = parse_portfolio(SAMPLE_XML)
    counts = summary(portfolio)
    assert counts["IRS"] == 2
    assert counts["Swaption"] == 1
    assert counts["CapFloor"] == 1
    assert counts["FXForward"] == 1
    assert counts["FXOption"] == 1
    assert counts["CommoditySwap"] == 1
    assert counts["CommodityFuturesOption"] == 1


def test_parse_portfolio_as_of_date():
    portfolio = parse_portfolio(SAMPLE_XML)
    assert portfolio.as_of_date == date(2024, 1, 15)


def test_parse_portfolio_netting_sets():
    portfolio = parse_portfolio(SAMPLE_XML)
    assert "NS_RATES_001" in portfolio.netting_sets
    assert len(portfolio.netting_sets["NS_RATES_001"]) == 4


def test_parse_portfolio_irs_fields():
    portfolio = parse_portfolio(SAMPLE_XML)
    irs = portfolio.get_trade("IRS_001")
    assert irs is not None
    assert isinstance(irs, IRS)
    assert irs.currency == "USD"
    assert abs(irs.fixed_rate - 0.0425) < 1e-9
    assert irs.pay_receive == PayReceive.PAY
    assert irs.notional == 10_000_000.0


def test_schema_validation():
    portfolio = parse_portfolio(SAMPLE_XML, schema_path=SCHEMA_XSD)
    assert portfolio is not None


def test_portfolio_add_trade():
    p = Portfolio(portfolio_id="TEST", as_of_date=date(2024, 1, 15))
    trade = IRS(
        trade_id="T1", currency="USD", notional=1_000_000,
        effective_date=date(2024, 1, 15), maturity_date=date(2029, 1, 15),
        fixed_rate=0.04, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL", pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
        netting_set_id="NS1",
    )
    p.add_trade(trade)
    assert len(p.trades) == 1
    assert "NS1" in p.netting_sets
