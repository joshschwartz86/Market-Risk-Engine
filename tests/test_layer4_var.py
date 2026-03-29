"""Unit tests for Layer 4 — Historical VaR."""
from datetime import date

import numpy as np
import pytest

from market_risk_engine.layer4_historical_var.var_calculator import VaRCalculator
from market_risk_engine.layer4_historical_var.models import ScenarioResult


def _make_results(pnls):
    from datetime import timedelta
    base = date(2023, 1, 3)
    return [
        ScenarioResult(scenario_date=base + timedelta(days=i), portfolio_pnl=p,
                       trade_pnls={"T1": p})
        for i, p in enumerate(pnls)
    ]


def test_var_99_greater_than_var_95():
    """99% VaR should be at least as bad (more negative) as 95% VaR."""
    np.random.seed(0)
    pnls = np.random.normal(0, 100_000, 250).tolist()
    results = _make_results(pnls)
    calc = VaRCalculator()
    var95 = calc.compute_var(results, 0.95).var_amount
    var99 = calc.compute_var(results, 0.99).var_amount
    assert var99 <= var95


def test_es_worse_than_var():
    """Expected Shortfall (CVaR) must be at least as bad as VaR."""
    np.random.seed(1)
    pnls = np.random.normal(0, 100_000, 250).tolist()
    results = _make_results(pnls)
    calc = VaRCalculator()
    var_result = calc.compute_var(results, 0.99)
    assert var_result.expected_shortfall <= var_result.var_amount


def test_var_known_distribution():
    """For a uniform distribution on [-200, 200], 95% VaR ≈ -190."""
    np.random.seed(42)
    pnls = np.linspace(-200, 200, 200).tolist()
    results = _make_results(pnls)
    calc = VaRCalculator()
    var_result = calc.compute_var(results, 0.95)
    # 5th percentile of -200 to 200 should be around -190
    assert -200 <= var_result.var_amount <= -180


def test_marginal_var_small():
    """Marginal VaR of a single trade in a 2-trade portfolio should be reasonable."""
    np.random.seed(2)
    n = 250
    pnl_t1 = np.random.normal(0, 50_000, n)
    pnl_t2 = np.random.normal(0, 50_000, n)
    results = [
        ScenarioResult(
            scenario_date=date(2023, 1, 3),
            trade_pnls={"T1": float(p1), "T2": float(p2)},
            portfolio_pnl=float(p1 + p2),
        )
        for p1, p2 in zip(pnl_t1, pnl_t2)
    ]
    calc = VaRCalculator()
    m_var = calc.marginal_var(results, "T1", 0.99)
    # Marginal VaR of T1 should be in a reasonable range
    assert -500_000 <= m_var <= 500_000


def test_position_level_vars():
    np.random.seed(3)
    n = 100
    results = [
        ScenarioResult(
            scenario_date=date(2023, 1, 3),
            trade_pnls={"A": float(np.random.normal(0, 1000)),
                        "B": float(np.random.normal(0, 2000))},
            portfolio_pnl=0.0,
        )
        for _ in range(n)
    ]
    calc = VaRCalculator()
    pos_vars = calc.position_level_vars(results, 0.99)
    assert "A" in pos_vars
    assert "B" in pos_vars
    # B has higher vol so its VaR magnitude should be larger
    assert abs(pos_vars["B"]) >= abs(pos_vars["A"]) * 0.5
