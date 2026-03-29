"""Unit tests for Layer 3 — Pricing."""
import math
from datetime import date

import numpy as np
import pytest

from market_risk_engine.common.enums import OptionType, PayReceive
from market_risk_engine.layer1_market_data.models import (
    CommodityCurve, FXRate, VolSurface, YieldCurve,
)
from market_risk_engine.layer2_portfolio.models import (
    CapFloor, CommodityFuturesOption, CommoditySwap,
    FXForward, FXOption, IRS, Swaption,
)
from market_risk_engine.layer3_pricing.base import MarketSnapshot
from market_risk_engine.layer3_pricing.dispatcher import PricingDispatcher
from market_risk_engine.layer3_pricing.irs_pricer import IRSPricer
from market_risk_engine.layer3_pricing.fx_option_pricer import _garman_kohlhagen
from market_risk_engine.layer3_pricing.commodity_option_pricer import _black76


TODAY = date(2024, 1, 15)

# Flat 5% USD SOFR curve
USD_SOFR = YieldCurve(
    currency="USD", curve_name="USD_SOFR", as_of_date=TODAY,
    tenors=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    zero_rates=[0.05] * 8,
)

EUR_ESTR = YieldCurve(
    currency="EUR", curve_name="EUR_ESTR", as_of_date=TODAY,
    tenors=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    zero_rates=[0.038] * 6,
)

EURUSD = FXRate(
    base_currency="EUR", quote_currency="USD",
    as_of_date=TODAY, spot=1.0875,
    tenors=[0.25, 0.5, 1.0],
    forward_points=[38.0, 75.5, 148.0],
)

USD_SWPTN_VOL = VolSurface(
    asset_id="USD_SWAPTION", as_of_date=TODAY,
    strikes=[0.03, 0.04, 0.045, 0.05, 0.06],
    expiries=[0.25, 0.5, 1.0, 2.0],
    vols=np.full((4, 5), 0.245),
)

EURUSD_VOL = VolSurface(
    asset_id="EURUSD", as_of_date=TODAY,
    strikes=[1.00, 1.05, 1.0875, 1.12, 1.16],
    expiries=[0.25, 0.5, 1.0],
    vols=np.full((3, 5), 0.07),
)

WTI_CURVE = CommodityCurve(
    commodity_id="WTI", as_of_date=TODAY,
    maturities=[0.083, 0.25, 0.5, 1.0],
    futures_prices=[72.45, 71.95, 70.80, 69.20],
)

WTI_VOL = VolSurface(
    asset_id="WTI_VOL", as_of_date=TODAY,
    strikes=[60.0, 65.0, 70.0, 72.45, 75.0, 80.0],
    expiries=[0.083, 0.25],
    vols=np.full((2, 6), 0.28),
)

MARKET = MarketSnapshot(
    as_of_date=TODAY,
    yield_curves={"USD_SOFR": USD_SOFR, "EUR_ESTR": EUR_ESTR},
    vol_surfaces={
        "USD_SWAPTION": USD_SWPTN_VOL,
        "EURUSD": EURUSD_VOL,
        "WTI_VOL": WTI_VOL,
    },
    fx_rates={"EURUSD": EURUSD},
    commodity_curves={"WTI": WTI_CURVE},
)


# ---------------------------------------------------------------------------
# IRS
# ---------------------------------------------------------------------------

def test_par_irs_npv_near_zero():
    """A par swap (fixed rate = par rate) should have NPV ≈ 0."""
    from market_risk_engine.layer1_market_data.yield_curve import YieldCurveInterpolator
    interp = YieldCurveInterpolator(USD_SOFR)
    par = interp.par_swap_rate(0.0, 5.0, "SEMIANNUAL")

    trade = IRS(
        trade_id="PAR_IRS",
        currency="USD", notional=10_000_000,
        effective_date=date(2024, 1, 15), maturity_date=date(2029, 1, 15),
        fixed_rate=par,
        fixed_day_count="30360", float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL", pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    pricer = IRSPricer()
    result = pricer.price(trade, MARKET)
    assert result.error is None
    # Par rate from analytic formula uses exact 0.5y spacing; actual schedule uses
    # calendar dates + 30360 day count — allow up to $50k on $10M notional.
    assert abs(result.npv) < 50_000, f"Par swap NPV too large: {result.npv}"


def test_irs_pay_vs_receive_symmetry():
    """Pay-fixed and receive-fixed positions in the same swap should sum to zero."""
    common = dict(
        currency="USD", notional=1_000_000,
        effective_date=date(2024, 1, 15), maturity_date=date(2026, 1, 15),
        fixed_rate=0.05, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL",
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    pay_trade = IRS(trade_id="PAY", pay_receive=PayReceive.PAY, **common)
    recv_trade = IRS(trade_id="RCV", pay_receive=PayReceive.RECEIVE, **common)
    pricer = IRSPricer()
    npv_pay = pricer.price(pay_trade, MARKET).npv
    npv_rcv = pricer.price(recv_trade, MARKET).npv
    assert abs(npv_pay + npv_rcv) < 1.0, f"Pay+Receive sum: {npv_pay + npv_rcv}"


# ---------------------------------------------------------------------------
# FX Option — put-call parity
# ---------------------------------------------------------------------------

def test_garman_kohlhagen_put_call_parity():
    """C - P = S*exp(-rf*T) - K*exp(-rd*T)"""
    S, K, rd, rf, sigma, T = 1.0875, 1.0875, 0.05, 0.038, 0.07, 0.5
    call = _garman_kohlhagen(S, K, rd, rf, sigma, T, OptionType.CALL)
    put = _garman_kohlhagen(S, K, rd, rf, sigma, T, OptionType.PUT)
    parity = S * math.exp(-rf * T) - K * math.exp(-rd * T)
    assert abs((call - put) - parity) < 1e-9


def test_fx_option_npv_positive():
    trade = FXOption(
        trade_id="FXO", base_currency="EUR", quote_currency="USD",
        notional_base=1_000_000,
        expiry_date=date(2024, 7, 15), delivery_date=date(2024, 7, 17),
        strike=1.0875, option_type=OptionType.CALL,
        vol_surface_id="EURUSD",
        base_discount_curve_id="EUR_ESTR",
        quote_discount_curve_id="USD_SOFR",
        fx_rate_id="EURUSD",
    )
    dispatcher = PricingDispatcher()
    result = dispatcher.price_trade(trade, MARKET)
    assert result.error is None
    assert result.npv > 0


# ---------------------------------------------------------------------------
# Black-76 commodity option
# ---------------------------------------------------------------------------

def test_black76_put_call_parity():
    """Black-76: C - P = DF * (F - K)"""
    F, K, sigma, T, df = 72.45, 72.45, 0.28, 0.25, 0.988
    call = _black76(F, K, sigma, T, df, 1.0, OptionType.CALL)
    put = _black76(F, K, sigma, T, df, 1.0, OptionType.PUT)
    assert abs((call - put) - df * (F - K)) < 1e-9


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_dispatcher_prices_full_portfolio():
    from market_risk_engine.layer2_portfolio.xml_parser import parse_portfolio
    import os
    xml = os.path.join(os.path.dirname(__file__), "..", "data", "sample",
                       "portfolio_sample.xml")
    portfolio = parse_portfolio(xml)
    dispatcher = PricingDispatcher()
    results = dispatcher.price_portfolio(portfolio, MARKET)
    assert len(results) == len(portfolio.trades)
    # At least half should price without error
    successes = [r for r in results if r.error is None]
    assert len(successes) >= len(results) // 2
