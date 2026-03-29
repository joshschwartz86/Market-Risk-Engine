"""Unit tests for Layer 5 — CCR Simulation."""
import math
from datetime import date

import numpy as np
import pytest

from market_risk_engine.layer5_ccr_simulation.correlation import CorrelationManager
from market_risk_engine.layer5_ccr_simulation.models import (
    RiskFactorSpec, SimulationConfig,
)
from market_risk_engine.layer5_ccr_simulation.risk_factor_sim import RiskFactorSimulator


# ---------------------------------------------------------------------------
# CorrelationManager
# ---------------------------------------------------------------------------

def test_identity_is_psd():
    cm = CorrelationManager(["r1", "r2", "r3"])
    assert cm.validate_positive_semidefinite()


def test_cholesky_reconstructs_matrix():
    cm = CorrelationManager(["r1", "r2"])
    cm.set_correlation("r1", "r2", 0.7)
    L = cm.cholesky()
    reconstructed = L @ L.T
    original = cm.get_matrix()
    assert np.allclose(reconstructed, original, atol=1e-10)


def test_nearest_psd_fixes_non_psd():
    cm = CorrelationManager(["r1", "r2", "r3"])
    # Deliberately set a non-PSD matrix
    cm._matrix = np.array([[1.0, 0.9, 0.9],
                            [0.9, 1.0, 0.9],
                            [0.9, 0.9, 1.0]])
    # This is actually PSD, use a worse one
    cm._matrix = np.array([[1.0, 1.1, 0.0],
                            [1.1, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
    result = cm.nearest_psd()
    eigenvalues = np.linalg.eigvalsh(result)
    assert np.all(eigenvalues >= -1e-6)


# ---------------------------------------------------------------------------
# RiskFactorSimulator — GBM
# ---------------------------------------------------------------------------

def test_gbm_initial_distribution():
    """GBM paths at t→0 should stay near initial value."""
    config = SimulationConfig(n_paths=1000, n_time_steps=5,
                               time_horizon=0.1, seed=0)
    spec = RiskFactorSpec(
        factor_id="S", factor_type="fx",
        initial_value=1.0875, drift=0.0, volatility=0.07,
        process="gbm",
    )
    sim = RiskFactorSimulator(config)
    paths = sim.simulate([spec])
    # First time step paths should be close to initial value
    first_step = paths.paths[:, 0, 0]
    assert abs(first_step.mean() - 1.0875) < 0.05


def test_gbm_positive_paths():
    """GBM paths should always be positive."""
    config = SimulationConfig(n_paths=500, n_time_steps=10,
                               time_horizon=5.0, seed=42)
    spec = RiskFactorSpec(
        factor_id="WTI", factor_type="commodity",
        initial_value=72.45, drift=0.0, volatility=0.30,
        process="gbm",
    )
    sim = RiskFactorSimulator(config)
    paths = sim.simulate([spec])
    assert np.all(paths.paths > 0)


# ---------------------------------------------------------------------------
# RiskFactorSimulator — Vasicek
# ---------------------------------------------------------------------------

def test_vasicek_mean_reversion():
    """Long-run mean of Vasicek paths should be near theta."""
    theta = 0.05
    config = SimulationConfig(n_paths=5000, n_time_steps=1,
                               time_horizon=20.0, seed=7)
    spec = RiskFactorSpec(
        factor_id="r", factor_type="rate",
        initial_value=0.02, drift=theta, volatility=0.01,
        mean_reversion_speed=0.5, process="vasicek",
    )
    sim = RiskFactorSimulator(config)
    paths = sim.simulate([spec])
    long_run_mean = paths.paths[:, -1, 0].mean()
    assert abs(long_run_mean - theta) < 0.005


# ---------------------------------------------------------------------------
# Correlated simulation
# ---------------------------------------------------------------------------

def test_correlated_paths_correlation():
    """Empirical correlation of simulated paths should be near the specified rho."""
    rho = 0.80
    config = SimulationConfig(n_paths=10_000, n_time_steps=1,
                               time_horizon=1.0, seed=99)
    specs = [
        RiskFactorSpec("S1", "fx", 1.0, 0.0, 0.1, process="gbm"),
        RiskFactorSpec("S2", "fx", 1.0, 0.0, 0.1, process="gbm"),
    ]
    corr = np.array([[1.0, rho], [rho, 1.0]])
    sim = RiskFactorSimulator(config)
    paths = sim.simulate(specs, correlation_matrix=corr)
    log_r1 = np.log(paths.paths[:, 0, 0])
    log_r2 = np.log(paths.paths[:, 0, 1])
    empirical_rho = np.corrcoef(log_r1, log_r2)[0, 1]
    assert abs(empirical_rho - rho) < 0.05
