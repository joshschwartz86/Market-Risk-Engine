"""VaR and Expected Shortfall calculation from scenario results."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .models import ScenarioResult, VaRResult


class VaRCalculator:
    """
    Compute historical VaR and Expected Shortfall from a list of ScenarioResult objects.
    """

    def compute_var(
        self,
        scenario_results: List[ScenarioResult],
        confidence_level: float = 0.99,
    ) -> VaRResult:
        """
        Compute portfolio-level VaR at the given confidence level.
        VaR is the loss that is not exceeded with probability `confidence_level`.
        Returned as a negative number (a loss).
        """
        pnls = np.array([r.portfolio_pnl for r in scenario_results], dtype=float)
        pnls_sorted = np.sort(pnls)

        # VaR: the (1 - confidence_level) quantile of the loss distribution
        var_amount = float(np.percentile(pnls_sorted, (1.0 - confidence_level) * 100.0))

        # Expected Shortfall: mean of losses worse than VaR
        tail_mask = pnls_sorted <= var_amount
        es = float(np.mean(pnls_sorted[tail_mask])) if tail_mask.any() else var_amount

        return VaRResult(
            confidence_level=confidence_level,
            lookback_days=len(scenario_results),
            var_amount=var_amount,
            expected_shortfall=es,
            scenario_pnls=pnls_sorted.tolist(),
        )

    def compute_portfolio_var(
        self,
        scenario_results: List[ScenarioResult],
        confidence_levels: List[float] = None,
    ) -> Dict[float, VaRResult]:
        """Compute VaR at multiple confidence levels."""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        return {cl: self.compute_var(scenario_results, cl) for cl in confidence_levels}

    def marginal_var(
        self,
        scenario_results: List[ScenarioResult],
        trade_id: str,
        confidence_level: float = 0.99,
    ) -> float:
        """
        Component VaR of a single trade:
        Difference between portfolio VaR and portfolio VaR without the trade.
        """
        portfolio_var = self.compute_var(scenario_results, confidence_level).var_amount

        # Rebuild scenario results without the trade
        reduced_results = [
            ScenarioResult(
                scenario_date=r.scenario_date,
                trade_pnls={k: v for k, v in r.trade_pnls.items() if k != trade_id},
                portfolio_pnl=sum(
                    v for k, v in r.trade_pnls.items() if k != trade_id
                ),
            )
            for r in scenario_results
        ]
        reduced_var = self.compute_var(reduced_results, confidence_level).var_amount

        return portfolio_var - reduced_var

    def position_level_vars(
        self,
        scenario_results: List[ScenarioResult],
        confidence_level: float = 0.99,
    ) -> Dict[str, float]:
        """Compute standalone VaR for each position."""
        if not scenario_results:
            return {}
        all_ids = list(scenario_results[0].trade_pnls.keys())
        result = {}
        for tid in all_ids:
            pnls = np.array([r.trade_pnls.get(tid, 0.0) for r in scenario_results])
            result[tid] = float(
                np.percentile(pnls, (1.0 - confidence_level) * 100.0)
            )
        return result
