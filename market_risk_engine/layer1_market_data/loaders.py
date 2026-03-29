"""Load market data from CSV files into domain model objects."""
from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.exceptions import MarketDataError
from .models import CommodityCurve, FXRate, VolSurface, YieldCurve


# ---------------------------------------------------------------------------
# Yield curves
# ---------------------------------------------------------------------------

def load_yield_curves(path: str, as_of: date) -> Dict[str, YieldCurve]:
    """
    Load yield curves from a CSV with columns:
      as_of_date, currency, curve_name, tenor_yr, zero_rate, day_count

    Returns a dict keyed by curve_name.
    """
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    df = df[df["as_of_date"].dt.date == as_of]
    if df.empty:
        raise MarketDataError(f"No yield curve data for {as_of} in {path}.")

    curves: Dict[str, YieldCurve] = {}
    for (currency, curve_name, day_count), grp in df.groupby(
        ["currency", "curve_name", "day_count"]
    ):
        grp = grp.sort_values("tenor_yr")
        curves[curve_name] = YieldCurve(
            currency=str(currency),
            curve_name=str(curve_name),
            as_of_date=as_of,
            tenors=grp["tenor_yr"].tolist(),
            zero_rates=grp["zero_rate"].tolist(),
            day_count=str(day_count),
        )
    return curves


# ---------------------------------------------------------------------------
# Volatility surfaces
# ---------------------------------------------------------------------------

def load_vol_surfaces(path: str, as_of: date) -> Dict[str, VolSurface]:
    """
    Load vol surfaces from a CSV with columns:
      as_of_date, surface_id, expiry_yr, strike, vol, vol_type

    Returns a dict keyed by surface_id.
    """
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    df = df[df["as_of_date"].dt.date == as_of]
    if df.empty:
        raise MarketDataError(f"No vol surface data for {as_of} in {path}.")

    surfaces: Dict[str, VolSurface] = {}
    for (surface_id, vol_type), grp in df.groupby(["surface_id", "vol_type"]):
        expiries = sorted(grp["expiry_yr"].unique().tolist())
        strikes = sorted(grp["strike"].unique().tolist())
        vols = np.zeros((len(expiries), len(strikes)), dtype=float)
        exp_idx = {e: i for i, e in enumerate(expiries)}
        str_idx = {s: i for i, s in enumerate(strikes)}
        for _, row in grp.iterrows():
            vols[exp_idx[row["expiry_yr"]], str_idx[row["strike"]]] = row["vol"]

        surfaces[str(surface_id)] = VolSurface(
            asset_id=str(surface_id),
            as_of_date=as_of,
            strikes=strikes,
            expiries=expiries,
            vols=vols,
            vol_type=str(vol_type),
        )
    return surfaces


# ---------------------------------------------------------------------------
# FX rates
# ---------------------------------------------------------------------------

def load_fx_rates(path: str, as_of: date) -> Dict[str, FXRate]:
    """
    Load FX rates from a CSV with columns:
      as_of_date, base_ccy, quote_ccy, spot, tenor_yr, forward_points, pip_factor

    Returns a dict keyed by "<base><quote>" pair, e.g. "EURUSD".
    """
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    df = df[df["as_of_date"].dt.date == as_of]
    if df.empty:
        raise MarketDataError(f"No FX rate data for {as_of} in {path}.")

    fx_rates: Dict[str, FXRate] = {}
    for (base, quote), grp in df.groupby(["base_ccy", "quote_ccy"]):
        pip_factor = float(grp["pip_factor"].iloc[0])
        spot = float(grp["spot"].iloc[0])
        # Rows with a tenor (forward points)
        fwd_rows = grp.dropna(subset=["tenor_yr", "forward_points"])
        fwd_rows = fwd_rows.sort_values("tenor_yr")
        tenors = fwd_rows["tenor_yr"].tolist() if not fwd_rows.empty else []
        fps = fwd_rows["forward_points"].tolist() if not fwd_rows.empty else []

        pair = f"{base}{quote}"
        fx_rates[pair] = FXRate(
            base_currency=str(base),
            quote_currency=str(quote),
            as_of_date=as_of,
            spot=spot,
            tenors=tenors,
            forward_points=fps,
            pip_factor=pip_factor,
        )
    return fx_rates


# ---------------------------------------------------------------------------
# Commodity curves
# ---------------------------------------------------------------------------

def load_commodity_curves(path: str, as_of: date) -> Dict[str, CommodityCurve]:
    """
    Load commodity futures curves from a CSV with columns:
      as_of_date, commodity_id, maturity_yr, futures_price, unit

    Returns a dict keyed by commodity_id.
    """
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    df = df[df["as_of_date"].dt.date == as_of]
    if df.empty:
        raise MarketDataError(f"No commodity curve data for {as_of} in {path}.")

    curves: Dict[str, CommodityCurve] = {}
    for commodity_id, grp in df.groupby("commodity_id"):
        grp = grp.sort_values("maturity_yr")
        unit = str(grp["unit"].iloc[0]) if "unit" in grp.columns else ""
        curves[str(commodity_id)] = CommodityCurve(
            commodity_id=str(commodity_id),
            as_of_date=as_of,
            maturities=grp["maturity_yr"].tolist(),
            futures_prices=grp["futures_price"].tolist(),
            unit=unit,
        )
    return curves


# ---------------------------------------------------------------------------
# Historical scenario data
# ---------------------------------------------------------------------------

def load_historical_scenarios(path: str) -> pd.DataFrame:
    """
    Load historical scenario time series.

    Expected columns:
      scenario_date, factor_id, factor_type, tenor_key, value

    Returns the full DataFrame (all dates).
    """
    df = pd.read_csv(path, parse_dates=["scenario_date"])
    df["tenor_key"] = df["tenor_key"].fillna("spot")
    return df
