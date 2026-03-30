"""Load market data from CSV files into domain model objects."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.date_utils import add_tenor, year_fraction
from ..common.exceptions import MarketDataError
from .models import CommodityCurve, FXRate, VolSurface, YieldCurve
from .yield_curve import BootstrapInstrument, YieldCurveBuilder


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


def load_curve_file(csv_path: str, curve_name: str, meta: dict) -> YieldCurve:
    """
    Load a single yield-curve CSV file in the daily-delivery format::

        Tenor,Rate,Date
        1D,3.772627,3/12/2026
        1W,3.78241,3/12/2026
        ...

    Parameters
    ----------
    csv_path  : path to the CSV file.
    curve_name: logical name for this curve (e.g. ``"USD_SOFR"``).
    meta      : metadata dict for this curve.  Keys:

        currency          – ISO currency code, e.g. ``"USD"``
        day_count         – ``ACT360``, ``ACT365``, ``30360``, ``ACTACT``
                            (default: ``"ACT360"``)
        rate_type         – ``"zero"`` or ``"par"`` (default: ``"zero"``)
        rate_basis        – ``"percent"``, ``"bps"``, or ``"decimal"``
                            (default: ``"percent"``)
        interpolation     – ``cubic_spline``, ``log_linear``, ``linear``
                            (default: ``"cubic_spline"``)
        deposit_cutoff_yr – *par only* — tenors ≤ this year-fraction are
                            bootstrapped as deposits; longer as swaps
                            (default: ``1.0``)
        payment_frequency – *par only* — swap coupon frequency
                            (default: ``"SEMIANNUAL"``)
    """
    df = pd.read_csv(csv_path)
    missing = {"Tenor", "Rate", "Date"} - set(df.columns)
    if missing:
        raise MarketDataError(
            f"Missing columns {missing} in {csv_path}. Expected: Tenor, Rate, Date."
        )

    # Parse curve date — accept M/D/YYYY or YYYY-MM-DD
    raw_date = str(df["Date"].dropna().iloc[0]).strip()
    try:
        as_of = datetime.strptime(raw_date, "%m/%d/%Y").date()
    except ValueError:
        try:
            as_of = date.fromisoformat(raw_date)
        except ValueError:
            raise MarketDataError(
                f"Cannot parse date '{raw_date}' in {csv_path}. "
                "Expected M/D/YYYY or YYYY-MM-DD."
            )

    currency     = meta["currency"]
    day_count    = meta.get("day_count", "ACT360")
    rate_type    = meta.get("rate_type", "zero")
    rate_basis   = meta.get("rate_basis", "percent")
    interpolation = meta.get("interpolation", "cubic_spline")

    _rate_factors = {"percent": 0.01, "bps": 0.0001, "decimal": 1.0}
    if rate_basis not in _rate_factors:
        raise MarketDataError(
            f"Unknown rate_basis '{rate_basis}'. Expected: percent, bps, decimal."
        )
    rate_factor = _rate_factors[rate_basis]

    # Convert tenor strings → year-fraction + decimal rate
    pairs: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        tenor_str = str(row["Tenor"]).strip()
        maturity_date = add_tenor(as_of, tenor_str)
        t = year_fraction(as_of, maturity_date, day_count)
        r = float(row["Rate"]) * rate_factor
        pairs.append((t, r))
    pairs.sort()

    if rate_type == "zero":
        return YieldCurve(
            currency=currency,
            curve_name=curve_name,
            as_of_date=as_of,
            tenors=[p[0] for p in pairs],
            zero_rates=[p[1] for p in pairs],
            day_count=day_count,
            interpolation=interpolation,
        )

    if rate_type == "par":
        deposit_cutoff = float(meta.get("deposit_cutoff_yr", 1.0))
        frequency = meta.get("payment_frequency", "SEMIANNUAL")
        builder = YieldCurveBuilder(as_of, currency, curve_name)
        for t, r in pairs:
            inst_type = "deposit" if t <= deposit_cutoff else "swap"
            builder.add_instrument(
                BootstrapInstrument(
                    instrument_type=inst_type,
                    maturity=t,
                    rate=r,
                    day_count=day_count,
                    payment_frequency=frequency,
                )
            )
        yc = builder.bootstrap()
        yc.interpolation = interpolation
        return yc

    raise MarketDataError(
        f"Unknown rate_type '{rate_type}'. Expected: zero or par."
    )


def load_curve_directory(dir_path: str, as_of: date) -> Dict[str, YieldCurve]:
    """
    Load all yield curves from a directory that contains:

    * One or more daily curve CSVs named ``{file_prefix}{YYYYMMDD}.csv``
    * A ``curve_metadata.json`` file describing each curve

    The metadata file is a JSON object keyed by logical curve name.  Example::

        {
          "USD_SOFR": {
            "description": "USD SOFR zero curve",
            "currency": "USD",
            "day_count": "ACT360",
            "rate_type": "zero",
            "rate_basis": "percent",
            "interpolation": "cubic_spline",
            "file_prefix": "sofr_curve"
          }
        }

    Returns a dict keyed by curve name.
    """
    meta_path = Path(dir_path) / "curve_metadata.json"
    if not meta_path.exists():
        raise MarketDataError(
            f"curve_metadata.json not found in {dir_path}. "
            "Create one to describe each curve's currency, day_count, etc."
        )

    with open(meta_path, encoding="utf-8") as f:
        all_meta: dict = json.load(f)

    date_str = as_of.strftime("%Y%m%d")
    curves: Dict[str, YieldCurve] = {}

    for curve_name, meta in all_meta.items():
        prefix = meta.get("file_prefix", curve_name.lower())
        filename = f"{prefix}{date_str}.csv"
        file_path = Path(dir_path) / filename
        if not file_path.exists():
            raise MarketDataError(
                f"No curve file found for '{curve_name}' on {as_of}: "
                f"expected {file_path}"
            )
        curves[curve_name] = load_curve_file(str(file_path), curve_name, meta)

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
