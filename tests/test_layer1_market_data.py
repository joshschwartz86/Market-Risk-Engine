"""Unit tests for Layer 1 — Market Data."""
import math
from datetime import date

import numpy as np
import pytest

from market_risk_engine.layer1_market_data.models import (
    CommodityCurve, FXRate, VolSurface, YieldCurve,
)
from market_risk_engine.layer1_market_data.yield_curve import (
    BootstrapInstrument, YieldCurveBuilder, YieldCurveInterpolator,
)
from market_risk_engine.layer1_market_data.vol_surface import (
    SABRCalibrator, VolSurfaceInterpolator,
)


# ---------------------------------------------------------------------------
# YieldCurve model
# ---------------------------------------------------------------------------

def test_yield_curve_discount_factor_consistency():
    """DF(t) must equal (1+z)^(-t) for a flat curve."""
    yc = YieldCurve(
        currency="USD", curve_name="TEST",
        as_of_date=date(2024, 1, 15),
        tenors=[0.5, 1.0, 2.0, 5.0, 10.0],
        zero_rates=[0.05, 0.05, 0.05, 0.05, 0.05],
    )
    interp = YieldCurveInterpolator(yc)
    for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
        df = interp.discount_factor(t)
        expected = (1.05) ** (-t)
        assert abs(df - expected) < 1e-8, f"DF mismatch at t={t}"


def test_yield_curve_forward_rate():
    """Forward rate from a flat 5% curve should also be 5%."""
    yc = YieldCurve(
        currency="USD", curve_name="TEST",
        as_of_date=date(2024, 1, 15),
        tenors=[1.0, 2.0, 5.0],
        zero_rates=[0.05, 0.05, 0.05],
    )
    interp = YieldCurveInterpolator(yc)
    fwd = interp.forward_rate(1.0, 2.0)
    # forward_rate returns simply-compounded; for flat 5% annual curve
    # the simply-compounded forward rate equals exactly 0.05
    assert abs(fwd - 0.05) < 1e-8


def test_yield_curve_sorted_tenors():
    with pytest.raises(ValueError):
        YieldCurve(
            currency="USD", curve_name="TEST",
            as_of_date=date(2024, 1, 15),
            tenors=[2.0, 1.0],  # Not sorted
            zero_rates=[0.05, 0.05],
        )


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def test_bootstrap_deposits():
    builder = YieldCurveBuilder(date(2024, 1, 15), "USD", "USD_SOFR")
    for t, r in [(0.25, 0.053), (0.5, 0.052), (1.0, 0.051)]:
        builder.add_instrument(BootstrapInstrument("deposit", t, r))
    yc = builder.bootstrap()
    assert len(yc.tenors) == 3
    assert abs(yc.zero_rates[0] - 0.053) < 1e-9


def test_bootstrap_swap_par_rate_zero_npv():
    """A swap bootstrapped from a par rate should price at zero NPV."""
    builder = YieldCurveBuilder(date(2024, 1, 15), "USD", "USD_TEST")
    builder.add_instrument(BootstrapInstrument("deposit", 0.25, 0.053))
    builder.add_instrument(BootstrapInstrument("deposit", 0.5, 0.052))
    builder.add_instrument(BootstrapInstrument("swap", 2.0, 0.049))
    yc = builder.bootstrap()
    interp = YieldCurveInterpolator(yc)
    # Par rate from bootstrapped curve should match the input par rate
    par = interp.par_swap_rate(0.0, 2.0, "SEMIANNUAL")
    assert abs(par - 0.049) < 5e-4


# ---------------------------------------------------------------------------
# Vol surface
# ---------------------------------------------------------------------------

def test_vol_surface_shape():
    with pytest.raises(ValueError):
        VolSurface(
            asset_id="TEST",
            as_of_date=date(2024, 1, 15),
            strikes=[0.04, 0.05],
            expiries=[0.5, 1.0],
            vols=np.zeros((3, 2)),  # Wrong shape
        )


def test_vol_surface_bilinear_interpolation():
    vs = VolSurface(
        asset_id="TEST",
        as_of_date=date(2024, 1, 15),
        strikes=[0.04, 0.05, 0.06],
        expiries=[0.5, 1.0],
        vols=np.array([[0.20, 0.21, 0.22],
                        [0.18, 0.19, 0.20]]),
    )
    interp = VolSurfaceInterpolator(vs)
    # Exact grid point
    assert abs(interp.get_vol(0.5, 0.05) - 0.21) < 1e-10
    # Interpolated
    v = interp.get_vol(0.75, 0.05)
    assert 0.19 < v < 0.21


# ---------------------------------------------------------------------------
# FX Rate
# ---------------------------------------------------------------------------

def test_fx_forward_rate():
    fx = FXRate(
        base_currency="EUR", quote_currency="USD",
        as_of_date=date(2024, 1, 15),
        spot=1.0875,
        tenors=[0.25, 0.5, 1.0],
        forward_points=[38.0, 75.5, 148.0],
        pip_factor=10_000.0,
    )
    fwd = fx.forward_rate(0.25)
    assert abs(fwd - (1.0875 + 38.0 / 10_000.0)) < 1e-9


# ---------------------------------------------------------------------------
# Commodity curve
# ---------------------------------------------------------------------------

def test_commodity_curve_interpolation():
    cc = CommodityCurve(
        commodity_id="WTI",
        as_of_date=date(2024, 1, 15),
        maturities=[0.25, 0.5, 1.0],
        futures_prices=[72.0, 71.0, 69.5],
    )
    price = cc.price_at(0.75)
    assert 69.5 < price < 72.0
