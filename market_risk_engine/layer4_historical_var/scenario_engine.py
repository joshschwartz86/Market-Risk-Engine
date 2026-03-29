"""Historical scenario construction, market shifting, and P&L computation."""
from __future__ import annotations

import copy
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..common.exceptions import MarketDataError
from ..layer1_market_data.models import (
    CommodityCurve, FXRate, VolSurface, YieldCurve,
)
from ..layer3_pricing.base import MarketSnapshot, PricingResult
from ..layer3_pricing.dispatcher import PricingDispatcher
from ..layer2_portfolio.models import Portfolio
from .lookback import extract_window, get_scenario_dates
from .models import ScenarioResult, ScenarioShift


class HistoricalScenarioEngine:
    """
    Build historical scenarios from time-series market data and compute P&L.

    The historical_data DataFrame is expected to have the columns:
      scenario_date, factor_id, factor_type, tenor_key, value
    """

    def __init__(self, historical_data: pd.DataFrame,
                 lookback_window: int = 250) -> None:
        self._data = historical_data.copy()
        self._data["scenario_date"] = pd.to_datetime(
            self._data["scenario_date"]
        ).dt.date
        self._lookback = lookback_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_scenarios(self, as_of: date) -> List[ScenarioShift]:
        """
        Compute daily absolute/percentage changes for each market factor
        in the lookback window ending on `as_of`.
        """
        windowed = extract_window(self._data, as_of, self._lookback)
        scenario_dates = sorted(windowed["scenario_date"].unique())

        # Build a pivot: (scenario_date, factor_id, tenor_key) -> value
        pivot = (
            windowed.set_index(["scenario_date", "factor_id", "tenor_key"])["value"]
        )

        shifts: List[ScenarioShift] = []
        prev_values: Dict = {}

        for s_date in scenario_dates:
            day_data = self._data[self._data["scenario_date"] == s_date]
            shift = ScenarioShift(scenario_date=s_date)

            # Group by factor
            for (fid, ftype), grp in day_data.groupby(["factor_id", "factor_type"]):
                key = str(fid)
                for _, row in grp.iterrows():
                    tenor = str(row["tenor_key"])
                    val = float(row["value"])
                    lookup = (key, tenor)
                    prev = prev_values.get(lookup)
                    if prev is not None:
                        if str(ftype) == "yield_curve":
                            arr = shift.yield_curve_shifts.setdefault(key, np.array([]))
                            shift.yield_curve_shifts[key] = np.append(
                                shift.yield_curve_shifts[key], val - prev
                            )
                        elif str(ftype) == "fx_spot":
                            shift.fx_shifts[key] = (val - prev) / prev if prev != 0 else 0.0
                        elif str(ftype) == "commodity":
                            shift.commodity_shifts[key] = (val - prev) / prev if prev != 0 else 0.0
                        elif str(ftype) == "vol":
                            shift.vol_shifts[key] = val - prev
                    prev_values[lookup] = val

            shifts.append(shift)

        return shifts

    def apply_scenario(
        self, base_market: MarketSnapshot, shift: ScenarioShift
    ) -> MarketSnapshot:
        """Return a new MarketSnapshot with the scenario shift applied."""
        # Yield curves — absolute rate shifts
        new_ycs = {}
        for cid, yc in base_market.yield_curves.items():
            rate_shifts = shift.yield_curve_shifts.get(cid)
            if rate_shifts is not None and len(rate_shifts) == len(yc.zero_rates):
                new_rates = [r + s for r, s in zip(yc.zero_rates, rate_shifts)]
            else:
                # Partial or missing shift — apply mean shift if available
                if rate_shifts is not None and len(rate_shifts) > 0:
                    mean_shift = float(np.mean(rate_shifts))
                    new_rates = [max(0.0001, r + mean_shift) for r in yc.zero_rates]
                else:
                    new_rates = list(yc.zero_rates)
            new_ycs[cid] = YieldCurve(
                currency=yc.currency,
                curve_name=yc.curve_name,
                as_of_date=yc.as_of_date,
                tenors=list(yc.tenors),
                zero_rates=[max(0.0001, r) for r in new_rates],
                day_count=yc.day_count,
                interpolation=yc.interpolation,
            )

        # Vol surfaces — parallel shift
        new_vols = {}
        for sid, vs in base_market.vol_surfaces.items():
            v_shift = shift.vol_shifts.get(sid, 0.0)
            new_vols[sid] = VolSurface(
                asset_id=vs.asset_id,
                as_of_date=vs.as_of_date,
                strikes=list(vs.strikes),
                expiries=list(vs.expiries),
                vols=np.clip(vs.vols + v_shift, 0.001, None),
                vol_type=vs.vol_type,
            )

        # FX rates — percentage shift on spot
        new_fx = {}
        for pid, fx in base_market.fx_rates.items():
            pct = shift.fx_shifts.get(pid, 0.0)
            new_fx[pid] = FXRate(
                base_currency=fx.base_currency,
                quote_currency=fx.quote_currency,
                as_of_date=fx.as_of_date,
                spot=fx.spot * (1.0 + pct),
                tenors=list(fx.tenors),
                forward_points=list(fx.forward_points),
                pip_factor=fx.pip_factor,
            )

        # Commodity curves — percentage shift on all prices
        new_comm = {}
        for cid, cc in base_market.commodity_curves.items():
            pct = shift.commodity_shifts.get(cid, 0.0)
            new_comm[cid] = CommodityCurve(
                commodity_id=cc.commodity_id,
                as_of_date=cc.as_of_date,
                maturities=list(cc.maturities),
                futures_prices=[max(0.01, p * (1.0 + pct)) for p in cc.futures_prices],
                unit=cc.unit,
            )

        return MarketSnapshot(
            as_of_date=base_market.as_of_date,
            yield_curves=new_ycs,
            vol_surfaces=new_vols,
            fx_rates=new_fx,
            commodity_curves=new_comm,
        )

    def compute_scenario_pnl(
        self,
        portfolio: Portfolio,
        base_market: MarketSnapshot,
        shifted_market: MarketSnapshot,
        dispatcher: PricingDispatcher,
    ) -> ScenarioResult:
        """Full revaluation P&L for a single shifted market."""
        base_results = {r.trade_id: r for r in dispatcher.price_portfolio(portfolio, base_market)}
        shifted_results = dispatcher.price_portfolio(portfolio, shifted_market)

        trade_pnls: Dict[str, float] = {}
        for res in shifted_results:
            base = base_results.get(res.trade_id)
            base_npv = base.npv if base and base.error is None else 0.0
            shifted_npv = res.npv if res.error is None else base_npv
            trade_pnls[res.trade_id] = shifted_npv - base_npv

        portfolio_pnl = sum(trade_pnls.values())
        return ScenarioResult(
            scenario_date=shifted_market.as_of_date,
            trade_pnls=trade_pnls,
            portfolio_pnl=portfolio_pnl,
        )

    def run_all_scenarios(
        self,
        portfolio: Portfolio,
        base_market: MarketSnapshot,
        dispatcher: PricingDispatcher,
        as_of: date,
    ) -> List[ScenarioResult]:
        """Build all scenarios and compute P&L for each."""
        shifts = self.build_scenarios(as_of)
        results = []
        for shift in shifts:
            shifted = self.apply_scenario(base_market, shift)
            result = self.compute_scenario_pnl(portfolio, base_market, shifted, dispatcher)
            result.scenario_date = shift.scenario_date
            results.append(result)
        return results
