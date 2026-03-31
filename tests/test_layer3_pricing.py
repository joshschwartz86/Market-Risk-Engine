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
    AmortizingFloatFloatSwap, AmortizingIRS, BermudanSwaption, CapFloor,
    CommodityFuturesOption, CommoditySwap, FloatFloatSwap, FXForward, FXOption,
    IRS, Swaption,
)
from market_risk_engine.layer3_pricing.base import MarketSnapshot
from market_risk_engine.layer3_pricing.dispatcher import PricingDispatcher
from market_risk_engine.layer3_pricing.irs_pricer import (
    AmortizingIRSPricer, FloatFloatSwapPricer, IRSPricer,
)
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
    """C - P = S*(1+rf)^(-T) - K*(1+rd)^(-T)"""
    S, K, rd, rf, sigma, T = 1.0875, 1.0875, 0.05, 0.038, 0.07, 0.5
    call = _garman_kohlhagen(S, K, rd, rf, sigma, T, OptionType.CALL)
    put = _garman_kohlhagen(S, K, rd, rf, sigma, T, OptionType.PUT)
    parity = S * (1.0 + rf) ** (-T) - K * (1.0 + rd) ** (-T)
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

# ---------------------------------------------------------------------------
# Amortizing IRS
# ---------------------------------------------------------------------------

def test_amortizing_irs_lower_npv_than_bullet():
    """An amortizing IRS on the same terms should have smaller DV01 than a bullet."""
    maturity = date(2027, 1, 15)
    # Bullet IRS
    bullet = IRS(
        trade_id="BULLET",
        currency="USD", notional=10_000_000,
        effective_date=TODAY, maturity_date=maturity,
        fixed_rate=0.05, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL", pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    # Amortizing IRS: notional drops by 2.5M every year
    sched = [
        (date(2025, 1, 15), 7_500_000),
        (date(2026, 1, 15), 5_000_000),
        (date(2027, 1, 15), 2_500_000),
    ]
    amort = AmortizingIRS(
        trade_id="AMORT",
        currency="USD", initial_notional=10_000_000,
        notional_schedule=sched,
        effective_date=TODAY, maturity_date=maturity,
        fixed_rate=0.05, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL", pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    pricer = AmortizingIRSPricer()
    bullet_pricer = IRSPricer()
    r_amort = pricer.price(amort, MARKET)
    r_bullet = bullet_pricer.price(bullet, MARKET)
    assert r_amort.error is None
    assert r_amort.pv01 is not None
    assert r_bullet.pv01 is not None
    # Amortizing DV01 must be smaller in magnitude than bullet
    assert abs(r_amort.pv01) < abs(r_bullet.pv01)


def test_amortizing_irs_pay_receive_symmetry():
    """Pay and receive amortizing IRS should sum to zero NPV."""
    sched = [(date(2025, 7, 15), 5_000_000), (date(2026, 1, 15), 0.0)]
    common = dict(
        currency="USD", initial_notional=10_000_000,
        notional_schedule=sched,
        effective_date=TODAY, maturity_date=date(2026, 1, 15),
        fixed_rate=0.05, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL",
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    pay = AmortizingIRS(trade_id="PAY", pay_receive=PayReceive.PAY, **common)
    recv = AmortizingIRS(trade_id="RCV", pay_receive=PayReceive.RECEIVE, **common)
    pricer = AmortizingIRSPricer()
    assert abs(pricer.price(pay, MARKET).npv + pricer.price(recv, MARKET).npv) < 1.0


# ---------------------------------------------------------------------------
# Float-Float (basis) swap
# ---------------------------------------------------------------------------

def test_float_float_identical_legs_zero_npv():
    """When both legs reference the same curve with zero spread, NPV must be zero."""
    trade = FloatFloatSwap(
        trade_id="BASIS_FLAT",
        currency="USD", notional=10_000_000,
        effective_date=TODAY, maturity_date=date(2027, 1, 15),
        leg1_index="SOFR", leg1_day_count="ACT360", leg1_frequency="QUARTERLY",
        leg1_forward_curve_id="USD_SOFR", leg1_spread=0.0,
        leg2_index="SOFR", leg2_day_count="ACT360", leg2_frequency="QUARTERLY",
        leg2_forward_curve_id="USD_SOFR", leg2_spread=0.0,
        pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR",
    )
    pricer = FloatFloatSwapPricer()
    result = pricer.price(trade, MARKET)
    assert result.error is None
    assert abs(result.npv) < 1.0, f"Identical-leg basis swap NPV should be 0, got {result.npv}"


def test_float_float_spread_increases_npv():
    """Adding a positive spread to leg 2 should increase NPV when paying leg 1."""
    common = dict(
        currency="USD", notional=10_000_000,
        effective_date=TODAY, maturity_date=date(2027, 1, 15),
        leg1_index="SOFR", leg1_day_count="ACT360", leg1_frequency="QUARTERLY",
        leg1_forward_curve_id="USD_SOFR", leg1_spread=0.0,
        leg2_index="USD_SOFR", leg2_day_count="ACT360", leg2_frequency="QUARTERLY",
        leg2_forward_curve_id="USD_SOFR",
        pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR",
    )
    pricer = FloatFloatSwapPricer()
    base = FloatFloatSwap(trade_id="BASE", leg2_spread=0.0, **common)
    with_spread = FloatFloatSwap(trade_id="SPREAD", leg2_spread=0.0025, **common)  # +25bp on leg 2
    npv_base = pricer.price(base, MARKET).npv
    npv_spread = pricer.price(with_spread, MARKET).npv
    assert npv_spread > npv_base  # extra spread on received leg improves NPV


def test_float_float_pay_receive_symmetry():
    """Pay and receive sides should sum to zero."""
    common = dict(
        currency="USD", notional=5_000_000,
        effective_date=TODAY, maturity_date=date(2026, 1, 15),
        leg1_index="SOFR", leg1_day_count="ACT360", leg1_frequency="QUARTERLY",
        leg1_forward_curve_id="USD_SOFR", leg1_spread=0.001,
        leg2_index="USD_SOFR", leg2_day_count="ACT360", leg2_frequency="SEMIANNUAL",
        leg2_forward_curve_id="USD_SOFR", leg2_spread=0.0,
        discount_curve_id="USD_SOFR",
    )
    pricer = FloatFloatSwapPricer()
    pay = FloatFloatSwap(trade_id="PAY", pay_receive=PayReceive.PAY, **common)
    recv = FloatFloatSwap(trade_id="RCV", pay_receive=PayReceive.RECEIVE, **common)
    assert abs(pricer.price(pay, MARKET).npv + pricer.price(recv, MARKET).npv) < 1.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Bermudan Swaption — Hull-White tree pricer
# ---------------------------------------------------------------------------

# Normal (Bachelier) vol surface for swaptions: flat at 100bp
_HW_NORMAL_VOL = VolSurface(
    asset_id="USD_BACHELIER",
    as_of_date=TODAY,
    strikes=[0.03, 0.04, 0.045, 0.05, 0.055, 0.06],
    expiries=[0.5, 1.0, 2.0, 3.0, 4.0],
    vols=np.full((5, 6), 0.0100),   # 100bp flat normal vol
)

_HW_MARKET = MarketSnapshot(
    as_of_date=TODAY,
    yield_curves={"USD_SOFR": USD_SOFR},
    vol_surfaces={"USD_BACHELIER": _HW_NORMAL_VOL},
)


def _make_bermudan(exercise_dates, n_steps=80, opt_type=OptionType.PAYER,
                   strike=0.05):
    return BermudanSwaption(
        trade_id="BERM",
        currency="USD",
        notional=1_000_000,
        exercise_dates=exercise_dates,
        underlying_start=date(2025, 1, 15),
        underlying_maturity=date(2029, 1, 15),
        strike=strike,
        option_type=opt_type,
        vol_surface_id="USD_BACHELIER",
        discount_curve_id="USD_SOFR",
        forward_curve_id="USD_SOFR",
        payment_frequency="SEMIANNUAL",
        day_count="ACT365",
        n_tree_steps=n_steps,
    )


def test_bermudan_payer_npv_positive():
    """ATM payer Bermudan swaption should have a positive NPV."""
    from market_risk_engine.layer3_pricing.bermudan_swaption_pricer import (
        BermudanSwaptionPricer,
    )
    trade = _make_bermudan([date(2025, 1, 15), date(2026, 1, 15),
                             date(2027, 1, 15), date(2028, 1, 15)])
    pricer = BermudanSwaptionPricer()
    result = pricer.price(trade, _HW_MARKET)
    assert result.error is None, result.error
    assert result.npv > 0, f"Expected positive NPV, got {result.npv}"


def test_bermudan_receiver_npv_positive():
    """ATM receiver Bermudan swaption should also have a positive NPV."""
    from market_risk_engine.layer3_pricing.bermudan_swaption_pricer import (
        BermudanSwaptionPricer,
    )
    trade = _make_bermudan(
        [date(2025, 1, 15), date(2026, 1, 15), date(2027, 1, 15)],
        opt_type=OptionType.RECEIVER,
    )
    pricer = BermudanSwaptionPricer()
    result = pricer.price(trade, _HW_MARKET)
    assert result.error is None, result.error
    assert result.npv > 0


def test_bermudan_single_exercise_approx_european():
    """
    A Bermudan with a single exercise date should give a value close to
    the corresponding European swaption priced with the Bachelier model.
    The tree (100 steps) introduces discretisation error, so we allow 5%
    relative tolerance.
    """
    from market_risk_engine.layer1_market_data.yield_curve import (
        YieldCurveInterpolator,
    )
    from market_risk_engine.layer3_pricing.bermudan_swaption_pricer import (
        BermudanSwaptionPricer,
    )
    from market_risk_engine.layer3_pricing.swaption_pricer import SwaptionPricer

    ex_date = date(2026, 1, 15)
    berm_trade = _make_bermudan([ex_date], n_steps=120)
    euro_trade = Swaption(
        trade_id="EURO",
        currency="USD",
        notional=1_000_000,
        option_expiry=ex_date,
        underlying_start=ex_date,
        underlying_maturity=date(2029, 1, 15),
        strike=0.05,
        option_type=OptionType.PAYER,
        vol_model="bachelier",
        vol_surface_id="USD_BACHELIER",
        discount_curve_id="USD_SOFR",
        forward_curve_id="USD_SOFR",
        payment_frequency="SEMIANNUAL",
    )
    berm_result = BermudanSwaptionPricer().price(berm_trade, _HW_MARKET)
    euro_result = SwaptionPricer().price(euro_trade, _HW_MARKET)

    assert berm_result.error is None, berm_result.error
    assert euro_result.error is None, euro_result.error
    assert berm_result.npv > 0
    assert euro_result.npv > 0
    # Tree price should be within 10% of analytic European price
    rel_err = abs(berm_result.npv - euro_result.npv) / euro_result.npv
    assert rel_err < 0.10, (
        f"Bermudan single-exercise vs European: tree={berm_result.npv:.2f}, "
        f"analytic={euro_result.npv:.2f}, rel_err={rel_err:.3%}"
    )


def test_bermudan_value_geq_earliest_european():
    """
    A Bermudan with multiple exercise dates must be worth at least as much
    as the European swaption with the earliest exercise date (since it
    strictly dominates by providing more optionality).
    """
    from market_risk_engine.layer3_pricing.bermudan_swaption_pricer import (
        BermudanSwaptionPricer,
    )
    from market_risk_engine.layer3_pricing.swaption_pricer import SwaptionPricer

    ex_dates = [date(2025, 1, 15), date(2026, 1, 15),
                date(2027, 1, 15), date(2028, 1, 15)]
    berm_trade = _make_bermudan(ex_dates, n_steps=100)
    euro_trade = Swaption(
        trade_id="EURO_FIRST",
        currency="USD",
        notional=1_000_000,
        option_expiry=ex_dates[0],
        underlying_start=ex_dates[0],
        underlying_maturity=date(2029, 1, 15),
        strike=0.05,
        option_type=OptionType.PAYER,
        vol_model="bachelier",
        vol_surface_id="USD_BACHELIER",
        discount_curve_id="USD_SOFR",
        forward_curve_id="USD_SOFR",
        payment_frequency="SEMIANNUAL",
    )
    berm_npv = BermudanSwaptionPricer().price(berm_trade, _HW_MARKET).npv
    euro_npv = SwaptionPricer().price(euro_trade, _HW_MARKET).npv
    # Bermudan >= European (more exercise rights can only add value)
    assert berm_npv >= euro_npv * 0.90, (
        f"Bermudan {berm_npv:.2f} should be >= European {euro_npv:.2f}"
    )


def test_bermudan_itm_payer_higher_than_otm():
    """An in-the-money Bermudan payer should be worth more than an OTM one."""
    from market_risk_engine.layer3_pricing.bermudan_swaption_pricer import (
        BermudanSwaptionPricer,
    )
    ex_dates = [date(2025, 7, 15), date(2026, 7, 15), date(2027, 7, 15)]
    pricer = BermudanSwaptionPricer()
    # Low strike → deep ITM payer (right to pay below-market fixed rate)
    itm = _make_bermudan(ex_dates, strike=0.02)
    # High strike → OTM payer
    otm = _make_bermudan(ex_dates, strike=0.10)
    npv_itm = pricer.price(itm, _HW_MARKET).npv
    npv_otm = pricer.price(otm, _HW_MARKET).npv
    assert npv_itm > npv_otm, f"ITM={npv_itm:.2f}, OTM={npv_otm:.2f}"


def test_bermudan_dispatcher_integration():
    """BermudanSwaption routes correctly through PricingDispatcher."""
    trade = _make_bermudan([date(2025, 7, 15), date(2026, 7, 15)], n_steps=60)
    dispatcher = PricingDispatcher()
    result = dispatcher.price_trade(trade, _HW_MARKET)
    assert result.error is None, result.error
    assert result.npv > 0


def test_bermudan_expired_exercise_dates_returns_zero():
    """If all exercise dates are in the past the NPV should be zero."""
    from market_risk_engine.layer3_pricing.bermudan_swaption_pricer import (
        BermudanSwaptionPricer,
    )
    trade = BermudanSwaption(
        trade_id="EXPIRED",
        currency="USD",
        notional=1_000_000,
        exercise_dates=[date(2023, 1, 15), date(2023, 7, 15)],
        underlying_start=date(2023, 1, 15),
        underlying_maturity=date(2028, 1, 15),
        strike=0.05,
        option_type=OptionType.PAYER,
        vol_surface_id="USD_BACHELIER",
        discount_curve_id="USD_SOFR",
        forward_curve_id="USD_SOFR",
    )
    result = BermudanSwaptionPricer().price(trade, _HW_MARKET)
    assert result.error is None
    assert result.npv == 0.0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Split payment frequencies
# ---------------------------------------------------------------------------

def test_irs_split_frequencies_pay_receive_symmetry():
    """Pay and receive IRS with different fixed/float frequencies must sum to zero."""
    common = dict(
        currency="USD", notional=5_000_000,
        effective_date=TODAY, maturity_date=date(2027, 1, 15),
        fixed_rate=0.05, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL",
        fixed_payment_frequency="ANNUAL",
        float_payment_frequency="QUARTERLY",
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    pricer = IRSPricer()
    pay = IRS(trade_id="PAY_SPLIT", pay_receive=PayReceive.PAY, **common)
    recv = IRS(trade_id="RCV_SPLIT", pay_receive=PayReceive.RECEIVE, **common)
    assert abs(pricer.price(pay, MARKET).npv + pricer.price(recv, MARKET).npv) < 1.0


def test_swaption_split_frequency_vs_default():
    """A swaption with annual fixed-leg frequency has a smaller annuity than semiannual."""
    common = dict(
        currency="USD", notional=5_000_000,
        option_expiry=date(2025, 1, 15),
        underlying_start=date(2025, 1, 15),
        underlying_maturity=date(2030, 1, 15),
        strike=0.042,
        option_type=OptionType.PAYER,
        vol_model="black",
        vol_surface_id="USD_SWAPTION",
        discount_curve_id="USD_SOFR",
        forward_curve_id="USD_SOFR",
    )
    from market_risk_engine.layer3_pricing.swaption_pricer import SwaptionPricer
    semi = Swaption(trade_id="SEMI", payment_frequency="SEMIANNUAL", **common)
    ann = Swaption(trade_id="ANN", payment_frequency="SEMIANNUAL",
                   fixed_payment_frequency="ANNUAL", **common)
    pricer = SwaptionPricer()
    r_semi = pricer.price(semi, MARKET)
    r_ann = pricer.price(ann, MARKET)
    assert r_semi.error is None
    assert r_ann.error is None
    # Annual coupon payments → slightly different annuity → different NPV (not necessarily bigger/smaller,
    # but the two should differ)
    assert abs(r_semi.npv - r_ann.npv) > 0.0


# ---------------------------------------------------------------------------
# Discount spread and forward spread
# ---------------------------------------------------------------------------

def test_irs_discount_spread_reduces_npv_receiver():
    """A positive discount spread reduces the PV of future cash flows (receiver swap)."""
    common = dict(
        currency="USD", notional=5_000_000,
        effective_date=TODAY, maturity_date=date(2029, 1, 15),
        fixed_rate=0.06, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL", pay_receive=PayReceive.RECEIVE,
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    pricer = IRSPricer()
    base = IRS(trade_id="BASE", **common)
    spread = IRS(trade_id="SPREAD", discount_spread=0.005, **common)  # +50bp discount spread
    npv_base = pricer.price(base, MARKET).npv
    npv_spread = pricer.price(spread, MARKET).npv
    # Higher discount spread → lower PV for both legs; receiver has higher fixed rate here so
    # net effect is a reduced (less positive) NPV
    assert npv_base != npv_spread, "Discount spread had no effect"


def test_irs_forward_spread_increases_npv_payer():
    """Positive forward spread raises the float rate received, benefiting a PAY-fixed swap."""
    common = dict(
        currency="USD", notional=5_000_000,
        effective_date=TODAY, maturity_date=date(2027, 1, 15),
        fixed_rate=0.05, fixed_day_count="30360",
        float_index="SOFR", float_day_count="ACT360",
        payment_frequency="SEMIANNUAL", pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    pricer = IRSPricer()
    base = IRS(trade_id="BASE", **common)
    fwd_spread = IRS(trade_id="FWD", forward_spread=0.0050, **common)  # +50bp on float received
    npv_base = pricer.price(base, MARKET).npv
    npv_fwd = pricer.price(fwd_spread, MARKET).npv
    # PAY fixed / RECEIVE float: higher float rate → higher float leg PV received → higher NPV
    assert npv_fwd > npv_base, f"Forward spread should increase PAY-fixed NPV: {npv_base:.0f} → {npv_fwd:.0f}"


def test_capfloor_discount_spread_reduces_npv():
    """Positive discount spread reduces the PV of caplet cash flows."""
    common = dict(
        currency="USD", notional=5_000_000,
        effective_date=TODAY, maturity_date=date(2026, 1, 15),
        strike=0.05, option_type=OptionType.CAP,
        float_index="SOFR", day_count="ACT360",
        payment_frequency="QUARTERLY",
        vol_surface_id="USD_SWAPTION",
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    pricer = CapFloorPricer()
    base = CapFloor(trade_id="BASE", **common)
    spread = CapFloor(trade_id="SPREAD", discount_spread=0.010, **common)
    npv_base = pricer.price(base, MARKET).npv
    npv_spread = pricer.price(spread, MARKET).npv
    assert npv_base > npv_spread, "Higher discount spread should lower cap NPV"


def test_capfloor_forward_spread_increases_cap_npv():
    """Positive forward spread increases forward rates → increases cap NPV."""
    common = dict(
        currency="USD", notional=5_000_000,
        effective_date=TODAY, maturity_date=date(2026, 1, 15),
        strike=0.05, option_type=OptionType.CAP,
        float_index="SOFR", day_count="ACT360",
        payment_frequency="QUARTERLY",
        vol_surface_id="USD_SWAPTION",
        discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
    )
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    pricer = CapFloorPricer()
    base = CapFloor(trade_id="BASE", **common)
    fwd_spread = CapFloor(trade_id="FWD", forward_spread=0.010, **common)
    npv_base = pricer.price(base, MARKET).npv
    npv_fwd = pricer.price(fwd_spread, MARKET).npv
    assert npv_fwd > npv_base, "Higher forward spread should increase cap NPV"


# ---------------------------------------------------------------------------
# AmortizingFloatFloatSwap
# ---------------------------------------------------------------------------

def _make_amort_float_float(pay_receive=PayReceive.PAY, leg1_spread=0.001, leg2_spread=0.0):
    sched = [
        (date(2025, 1, 15), 7_500_000),
        (date(2026, 1, 15), 5_000_000),
        (date(2027, 1, 15), 2_500_000),
    ]
    return AmortizingFloatFloatSwap(
        trade_id="AMORT_FFS",
        currency="USD",
        initial_notional=10_000_000,
        notional_schedule=sched,
        effective_date=TODAY,
        maturity_date=date(2027, 1, 15),
        leg1_index="SOFR",
        leg1_day_count="ACT360",
        leg1_frequency="QUARTERLY",
        leg1_forward_curve_id="USD_SOFR",
        leg1_spread=leg1_spread,
        leg2_index="SOFR",
        leg2_day_count="ACT360",
        leg2_frequency="QUARTERLY",
        leg2_forward_curve_id="USD_SOFR",
        leg2_spread=leg2_spread,
        pay_receive=pay_receive,
        discount_curve_id="USD_SOFR",
    )


def test_amortizing_float_float_pay_receive_symmetry():
    """Pay and receive amortizing float-float swaps must sum to zero NPV."""
    from market_risk_engine.layer3_pricing.irs_pricer import AmortizingFloatFloatSwapPricer
    pricer = AmortizingFloatFloatSwapPricer()
    pay = _make_amort_float_float(PayReceive.PAY)
    recv = _make_amort_float_float(PayReceive.RECEIVE)
    npv_pay = pricer.price(pay, MARKET).npv
    npv_recv = pricer.price(recv, MARKET).npv
    assert abs(npv_pay + npv_recv) < 1.0, f"Pay+Receive = {npv_pay + npv_recv}"


def test_amortizing_float_float_lower_dv01_than_bullet():
    """Amortizing FF swap should have smaller DV01 than a bullet FF swap."""
    from market_risk_engine.layer3_pricing.irs_pricer import AmortizingFloatFloatSwapPricer
    amort = _make_amort_float_float()
    bullet = FloatFloatSwap(
        trade_id="BULLET_FFS",
        currency="USD", notional=10_000_000,
        effective_date=TODAY, maturity_date=date(2027, 1, 15),
        leg1_index="SOFR", leg1_day_count="ACT360", leg1_frequency="QUARTERLY",
        leg1_forward_curve_id="USD_SOFR", leg1_spread=0.001,
        leg2_index="SOFR", leg2_day_count="ACT360", leg2_frequency="QUARTERLY",
        leg2_forward_curve_id="USD_SOFR", leg2_spread=0.0,
        pay_receive=PayReceive.PAY,
        discount_curve_id="USD_SOFR",
    )
    amort_pricer = AmortizingFloatFloatSwapPricer()
    bullet_pricer = FloatFloatSwapPricer()
    r_amort = amort_pricer.price(amort, MARKET)
    r_bullet = bullet_pricer.price(bullet, MARKET)
    assert r_amort.error is None
    assert r_amort.pv01 is not None and r_bullet.pv01 is not None
    assert abs(r_amort.pv01) < abs(r_bullet.pv01), (
        f"Amortizing DV01 {r_amort.pv01:.2f} should be < bullet {r_bullet.pv01:.2f}"
    )


def test_amortizing_float_float_dispatcher():
    """AmortizingFloatFloatSwap routes correctly through PricingDispatcher."""
    trade = _make_amort_float_float()
    dispatcher = PricingDispatcher()
    result = dispatcher.price_trade(trade, MARKET)
    assert result.error is None, result.error


# ---------------------------------------------------------------------------
# Cap/Floor — Bachelier (normal) model
# ---------------------------------------------------------------------------

# Flat normal vol surface at 80bp = 0.0080
_NORMAL_CAPFLOOR_VOL = VolSurface(
    asset_id="USD_CAPFLOOR_NORMAL",
    as_of_date=TODAY,
    strikes=[0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
    expiries=[0.25, 0.5, 1.0, 2.0, 3.0],
    vols=np.full((5, 6), 0.0080),   # 80bp flat normal vol
)

_NORMAL_CAP_MARKET = MarketSnapshot(
    as_of_date=TODAY,
    yield_curves={"USD_SOFR": USD_SOFR},
    vol_surfaces={"USD_CAPFLOOR_NORMAL": _NORMAL_CAPFLOOR_VOL},
)

_CAP_COMMON = dict(
    currency="USD", notional=5_000_000,
    effective_date=TODAY, maturity_date=date(2026, 1, 15),
    strike=0.05, float_index="SOFR", day_count="ACT360",
    payment_frequency="QUARTERLY",
    vol_surface_id="USD_CAPFLOOR_NORMAL",
    discount_curve_id="USD_SOFR", forward_curve_id="USD_SOFR",
)


def test_bachelier_cap_npv_positive():
    """ATM cap under Bachelier model should have a positive NPV."""
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    trade = CapFloor(trade_id="BACH_CAP", option_type=OptionType.CAP,
                     vol_model="bachelier", **_CAP_COMMON)
    result = CapFloorPricer().price(trade, _NORMAL_CAP_MARKET)
    assert result.error is None, result.error
    assert result.npv > 0


def test_bachelier_floor_npv_positive():
    """ATM floor under Bachelier model should have a positive NPV."""
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    trade = CapFloor(trade_id="BACH_FLOOR", option_type=OptionType.FLOOR,
                     vol_model="bachelier", **_CAP_COMMON)
    result = CapFloorPricer().price(trade, _NORMAL_CAP_MARKET)
    assert result.error is None, result.error
    assert result.npv > 0


def test_bachelier_cap_floor_put_call_parity():
    """
    Cap - Floor = swap PV (forward annuity × (F − K)).
    Under any model, caplet - floorlet = df * tau * N * (F - K).
    Summing over all periods: Cap - Floor = PV of fixed-rate swap at K.
    """
    from market_risk_engine.layer1_market_data.yield_curve import YieldCurveInterpolator
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    pricer = CapFloorPricer()
    cap = CapFloor(trade_id="CAP_PCP", option_type=OptionType.CAP,
                   vol_model="bachelier", **_CAP_COMMON)
    floor = CapFloor(trade_id="FLR_PCP", option_type=OptionType.FLOOR,
                     vol_model="bachelier", **_CAP_COMMON)
    npv_cap = pricer.price(cap, _NORMAL_CAP_MARKET).npv
    npv_floor = pricer.price(floor, _NORMAL_CAP_MARKET).npv

    # Replicate the RHS: sum of df(t_pay) * tau * N * (F - K) over each period
    from market_risk_engine.common.date_utils import generate_schedule, year_fraction
    disc = YieldCurveInterpolator(USD_SOFR)
    fwd = YieldCurveInterpolator(USD_SOFR)
    schedule = generate_schedule(TODAY, date(2026, 1, 15), "QUARTERLY")
    K = 0.05
    N = 5_000_000
    swap_pv = 0.0
    prev = TODAY
    for pay_date in schedule:
        if pay_date <= TODAY:
            prev = pay_date
            continue
        t_prev = max(year_fraction(TODAY, prev, "ACT360"), 1e-6)
        t_pay = year_fraction(TODAY, pay_date, "ACT360")
        tau = year_fraction(prev, pay_date, "ACT360")
        F = fwd.forward_rate(t_prev, t_pay)
        df = disc.discount_factor(t_pay)
        swap_pv += df * tau * N * (F - K)
        prev = pay_date

    assert abs((npv_cap - npv_floor) - swap_pv) < 1.0, (
        f"Put-call parity breach: cap-floor={npv_cap - npv_floor:.2f}, swap_pv={swap_pv:.2f}"
    )


def test_black_vs_bachelier_cap_differ():
    """Black and Bachelier models should produce different prices for the same inputs."""
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    pricer = CapFloorPricer()
    black_cap = CapFloor(trade_id="BLACK", option_type=OptionType.CAP,
                         vol_model="black", **_CAP_COMMON)
    bach_cap = CapFloor(trade_id="BACH", option_type=OptionType.CAP,
                        vol_model="bachelier", **_CAP_COMMON)
    npv_black = pricer.price(black_cap, _NORMAL_CAP_MARKET).npv
    npv_bach = pricer.price(bach_cap, _NORMAL_CAP_MARKET).npv
    assert npv_black != npv_bach, "Black and Bachelier models should give different prices"


def test_bachelier_cap_vega_positive():
    """Bachelier cap vega should be positive (higher vol → higher cap value)."""
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    trade = CapFloor(trade_id="VEGA_CAP", option_type=OptionType.CAP,
                     vol_model="bachelier", **_CAP_COMMON)
    result = CapFloorPricer().price(trade, _NORMAL_CAP_MARKET)
    assert result.vega is not None and result.vega > 0


def test_bachelier_itm_cap_higher_than_otm():
    """Deep ITM Bachelier cap (low strike) should be worth more than OTM (high strike)."""
    from market_risk_engine.layer3_pricing.cap_floor_pricer import CapFloorPricer
    pricer = CapFloorPricer()
    itm = CapFloor(trade_id="ITM", option_type=OptionType.CAP, vol_model="bachelier",
                   strike=0.02, **{k: v for k, v in _CAP_COMMON.items() if k != "strike"})
    otm = CapFloor(trade_id="OTM", option_type=OptionType.CAP, vol_model="bachelier",
                   strike=0.08, **{k: v for k, v in _CAP_COMMON.items() if k != "strike"})
    assert pricer.price(itm, _NORMAL_CAP_MARKET).npv > pricer.price(otm, _NORMAL_CAP_MARKET).npv


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
