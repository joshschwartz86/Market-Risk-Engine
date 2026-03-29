"""Yield curve bootstrapping and interpolation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from ..common.date_utils import year_fraction
from ..common.exceptions import MarketDataError
from .models import YieldCurve


# ---------------------------------------------------------------------------
# Bootstrap input instrument
# ---------------------------------------------------------------------------

@dataclass
class BootstrapInstrument:
    """Single market quote used to bootstrap the zero curve."""

    instrument_type: str   # "deposit", "fra", "swap"
    maturity: float        # year fraction to maturity
    rate: float            # market rate (par rate for swaps, deposit rate, fra rate)
    day_count: str = "ACT360"
    payment_frequency: str = "SEMIANNUAL"  # relevant for swaps


# ---------------------------------------------------------------------------
# Bootstrapper
# ---------------------------------------------------------------------------

class YieldCurveBuilder:
    """Bootstrap a zero-rate curve from a set of market instruments."""

    def __init__(self, as_of_date: date, currency: str, curve_name: str) -> None:
        self._as_of = as_of_date
        self._currency = currency
        self._curve_name = curve_name
        self._instruments: List[BootstrapInstrument] = []

    def add_instrument(self, instrument: BootstrapInstrument) -> None:
        self._instruments.append(instrument)

    def bootstrap(self) -> YieldCurve:
        """Return the bootstrapped YieldCurve."""
        instruments = sorted(self._instruments, key=lambda i: i.maturity)
        tenors: List[float] = []
        zero_rates: List[float] = []

        for inst in instruments:
            t = inst.maturity
            if inst.instrument_type == "deposit":
                z = inst.rate  # treated as already a continuously-compounded rate
                tenors.append(t)
                zero_rates.append(z)
            elif inst.instrument_type in ("fra", "future"):
                z = inst.rate
                tenors.append(t)
                zero_rates.append(z)
            elif inst.instrument_type == "swap":
                z = self._bootstrap_swap(t, inst.rate, inst.payment_frequency,
                                         tenors, zero_rates)
                tenors.append(t)
                zero_rates.append(z)
            else:
                raise MarketDataError(
                    f"Unknown instrument type: {inst.instrument_type}"
                )

        if not tenors:
            raise MarketDataError("No instruments supplied for bootstrapping.")

        return YieldCurve(
            currency=self._currency,
            curve_name=self._curve_name,
            as_of_date=self._as_of,
            tenors=tenors,
            zero_rates=zero_rates,
        )

    def _bootstrap_swap(
        self,
        maturity: float,
        par_rate: float,
        frequency: str,
        existing_tenors: List[float],
        existing_rates: List[float],
    ) -> float:
        """Find the zero rate at `maturity` such that the par swap NPV = 0."""
        freq_map = {"MONTHLY": 12, "QUARTERLY": 4, "SEMIANNUAL": 2, "ANNUAL": 1}
        n_per_year = freq_map[frequency.upper()]
        dt = 1.0 / n_per_year

        def _interp_df(t: float, extra_t: Optional[float] = None,
                       extra_z: Optional[float] = None) -> float:
            ts = list(existing_tenors)
            zs = list(existing_rates)
            if extra_t is not None and extra_z is not None:
                ts.append(extra_t)
                zs.append(extra_z)
            if not ts:
                return 1.0
            z = float(np.interp(t, ts, zs))
            return math.exp(-z * t)

        def npv(z_guess: float) -> float:
            pv_fixed = 0.0
            t = dt
            while t <= maturity + 1e-9:
                df = _interp_df(t, maturity, z_guess)
                pv_fixed += par_rate * dt * df
                t += dt
            pv_float = 1.0 - _interp_df(maturity, maturity, z_guess)
            return pv_fixed - pv_float

        try:
            z_solution = brentq(npv, -0.20, 0.50, xtol=1e-10)
        except ValueError as exc:
            raise MarketDataError(
                f"Could not bootstrap zero rate at {maturity:.4f}y: {exc}"
            ) from exc
        return z_solution


# ---------------------------------------------------------------------------
# Interpolator
# ---------------------------------------------------------------------------

class YieldCurveInterpolator:
    """Interpolate a YieldCurve to provide discount factors and forward rates."""

    def __init__(self, curve: YieldCurve) -> None:
        self._curve = curve
        t = np.array(curve.tenors, dtype=float)
        z = np.array(curve.zero_rates, dtype=float)

        if curve.interpolation == "cubic_spline" and len(t) >= 3:
            self._spline = CubicSpline(t, z, extrapolate=True)
            self._mode = "spline"
        elif curve.interpolation == "log_linear":
            # interpolate in log-discount-factor space
            self._log_dfs = -z * t
            self._t = t
            self._mode = "log_linear"
        else:
            self._t = t
            self._z = z
            self._mode = "linear"

    def zero_rate(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if self._mode == "spline":
            return float(self._spline(t))
        if self._mode == "log_linear":
            log_df = float(np.interp(t, self._t, self._log_dfs))
            return -log_df / t
        return float(np.interp(t, self._t, self._z))

    def discount_factor(self, t: float) -> float:
        if t <= 0:
            return 1.0
        z = self.zero_rate(t)
        return math.exp(-z * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simply-compounded forward rate between t1 and t2."""
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1.")
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (df1 / df2 - 1.0) / (t2 - t1)

    def par_swap_rate(self, start: float, end: float, frequency: str) -> float:
        """Compute the par fixed rate for a swap from `start` to `end`."""
        freq_map = {"MONTHLY": 12, "QUARTERLY": 4, "SEMIANNUAL": 2, "ANNUAL": 1}
        n_per_year = freq_map[frequency.upper()]
        dt = 1.0 / n_per_year

        annuity = 0.0
        t = start + dt
        while t <= end + 1e-9:
            annuity += dt * self.discount_factor(t)
            t += dt

        df_start = self.discount_factor(start)
        df_end = self.discount_factor(end)
        if annuity == 0:
            raise MarketDataError("Zero annuity — cannot compute par rate.")
        return (df_start - df_end) / annuity
