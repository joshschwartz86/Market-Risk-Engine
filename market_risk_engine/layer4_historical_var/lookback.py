"""Lookback window management for historical VaR."""
from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd


def extract_window(df: pd.DataFrame, as_of: date, lookback_days: int) -> pd.DataFrame:
    """
    Return rows from `df` whose scenario_date falls within the lookback window
    ending on (but not including) as_of.

    Parameters
    ----------
    df : DataFrame with a 'scenario_date' column (datetime or date)
    as_of : reference date (today)
    lookback_days : number of calendar days to look back
    """
    df = df.copy()
    df["scenario_date"] = pd.to_datetime(df["scenario_date"]).dt.date
    cutoff = as_of
    mask = (df["scenario_date"] < cutoff)
    window = df[mask].sort_values("scenario_date")
    # Take the most recent `lookback_days` unique dates
    unique_dates = sorted(window["scenario_date"].unique())
    if len(unique_dates) > lookback_days:
        unique_dates = unique_dates[-lookback_days:]
    return window[window["scenario_date"].isin(unique_dates)].reset_index(drop=True)


def get_scenario_dates(df: pd.DataFrame, as_of: date,
                       lookback_days: int) -> List[date]:
    """Return the list of distinct scenario dates within the lookback window."""
    windowed = extract_window(df, as_of, lookback_days)
    return sorted(windowed["scenario_date"].unique().tolist())
