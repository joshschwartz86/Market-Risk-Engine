"""
Expected Exposure (EE), Peak Exposure (PE), and EPE calculator.

Design:
  - Outer loop: time steps
  - Inner operation: vectorised across all paths at each time step
  - At each (path, time_step) reconstruct a MarketSnapshot from simulated values
    and call PricingDispatcher to get the netting-set MtM.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from ..common.exceptions import SimulationError
from ..layer1_market_data.models import (
    CommodityCurve, FXRate, VolSurface, YieldCurve,
)
from ..layer2_portfolio.models import Portfolio, TradeUnion
from ..layer3_pricing.base import MarketSnapshot
from ..layer3_pricing.dispatcher import PricingDispatcher
from .models import ExposureProfile, RiskFactorSpec, SimulationConfig, SimulationPaths
from .netting_set import apply_netting, get_netting_set_trades

PEAK_PERCENTILE = 0.95


class ExposureCalculator:
    """
    Compute exposure profiles for netting sets under Monte Carlo simulation.

    Parameters
    ----------
    config : SimulationConfig
    dispatcher : PricingDispatcher (shared with Layers 3/4)
    """

    def __init__(self, config: SimulationConfig,
                 dispatcher: PricingDispatcher) -> None:
        self._config = config
        self._dispatcher = dispatcher

    def compute_exposure(
        self,
        netting_set_id: str,
        trades: List[TradeUnion],
        paths: SimulationPaths,
        base_market: MarketSnapshot,
        factor_specs: List[RiskFactorSpec],
    ) -> ExposureProfile:
        """
        For each path and time step:
          1. Reconstruct a MarketSnapshot from the simulated factor values.
          2. Price all trades in the netting set.
          3. Apply netting: exposure = max(sum(MtMs), 0).

        Returns EE(t), PE(t), EPE scalar.
        """
        n_paths, n_steps, n_factors = paths.paths.shape
        time_grid = paths.time_grid
        factor_index = {fid: i for i, fid in enumerate(paths.factor_ids)}

        mtm_paths = np.zeros((n_paths, n_steps), dtype=float)

        # Pre-compute base NPVs once (MtM at t=0)
        base_npvs = {
            r.trade_id: (r.npv if r.error is None else 0.0)
            for r in self._dispatcher.price_portfolio(
                _make_dummy_portfolio(trades, netting_set_id), base_market
            )
        }

        for t_idx in range(n_steps):
            for p_idx in range(n_paths):
                factor_values = {
                    fid: float(paths.paths[p_idx, t_idx, factor_index[fid]])
                    for fid in paths.factor_ids
                }
                sim_market = self._reconstruct_market(
                    base_market, factor_specs, factor_values
                )
                trade_mtms = []
                for trade in trades:
                    result = self._dispatcher.price_trade(trade, sim_market)
                    base_npv = base_npvs.get(trade.trade_id, 0.0)
                    incremental = (result.npv - base_npv) if result.error is None else 0.0
                    trade_mtms.append(base_npv + incremental)
                mtm_paths[p_idx, t_idx] = apply_netting(trade_mtms)

        positive_exposure = np.maximum(mtm_paths, 0.0)
        ee = positive_exposure.mean(axis=0)
        pe = np.percentile(positive_exposure, PEAK_PERCENTILE * 100, axis=0)
        epe = float(np.trapz(ee, time_grid) / (time_grid[-1] - time_grid[0])
                    if time_grid[-1] > time_grid[0] else ee.mean())

        return ExposureProfile(
            netting_set_id=netting_set_id,
            time_grid=time_grid,
            expected_exposure=ee,
            peak_exposure=pe,
            expected_positive_exposure=epe,
            mtm_paths=mtm_paths,
        )

    def _reconstruct_market(
        self,
        base: MarketSnapshot,
        factor_specs: List[RiskFactorSpec],
        factor_values: Dict[str, float],
    ) -> MarketSnapshot:
        """
        Build a new MarketSnapshot by scaling the base market with simulated values.

        Convention per factor_type:
          "rate"      : shift all tenors of the named yield curve by (value - initial)
          "fx"        : scale spot of the named FX pair by (value / initial)
          "commodity" : scale all futures prices by (value / initial)
          "vol"       : shift all vols on the named surface by (value - initial)
        """
        spec_map = {s.factor_id: s for s in factor_specs}

        new_ycs = dict(base.yield_curves)
        new_vols = dict(base.vol_surfaces)
        new_fx = dict(base.fx_rates)
        new_comm = dict(base.commodity_curves)

        for fid, val in factor_values.items():
            spec = spec_map.get(fid)
            if spec is None:
                continue
            ftype = spec.factor_type.lower()

            if ftype == "rate" and fid in new_ycs:
                orig = new_ycs[fid]
                shift = val - spec.initial_value
                new_ycs[fid] = YieldCurve(
                    currency=orig.currency,
                    curve_name=orig.curve_name,
                    as_of_date=orig.as_of_date,
                    tenors=list(orig.tenors),
                    zero_rates=[max(0.0001, r + shift) for r in orig.zero_rates],
                    day_count=orig.day_count,
                    interpolation=orig.interpolation,
                )

            elif ftype == "fx" and fid in new_fx:
                orig = new_fx[fid]
                scale = val / spec.initial_value if spec.initial_value != 0 else 1.0
                new_fx[fid] = FXRate(
                    base_currency=orig.base_currency,
                    quote_currency=orig.quote_currency,
                    as_of_date=orig.as_of_date,
                    spot=orig.spot * scale,
                    tenors=list(orig.tenors),
                    forward_points=[fp * scale for fp in orig.forward_points],
                    pip_factor=orig.pip_factor,
                )

            elif ftype == "commodity" and fid in new_comm:
                orig = new_comm[fid]
                scale = val / spec.initial_value if spec.initial_value != 0 else 1.0
                new_comm[fid] = CommodityCurve(
                    commodity_id=orig.commodity_id,
                    as_of_date=orig.as_of_date,
                    maturities=list(orig.maturities),
                    futures_prices=[max(0.01, p * scale) for p in orig.futures_prices],
                    unit=orig.unit,
                )

            elif ftype == "vol" and fid in new_vols:
                orig = new_vols[fid]
                shift = val - spec.initial_value
                new_vols[fid] = VolSurface(
                    asset_id=orig.asset_id,
                    as_of_date=orig.as_of_date,
                    strikes=list(orig.strikes),
                    expiries=list(orig.expiries),
                    vols=np.clip(orig.vols + shift, 0.001, None),
                    vol_type=orig.vol_type,
                )

        return MarketSnapshot(
            as_of_date=base.as_of_date,
            yield_curves=new_ycs,
            vol_surfaces=new_vols,
            fx_rates=new_fx,
            commodity_curves=new_comm,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_dummy_portfolio(trades: List[TradeUnion],
                           netting_set_id: str) -> Portfolio:
    """Wrap a list of trades in a minimal Portfolio for dispatcher calls."""
    from datetime import date as date_
    p = Portfolio(portfolio_id=netting_set_id, as_of_date=date_.today())
    for t in trades:
        p.trades.append(t)
    return p
