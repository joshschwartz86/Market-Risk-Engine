"""Data models for the CCR Monte Carlo simulation framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SimulationConfig:
    n_paths: int = 10_000
    n_time_steps: int = 50
    time_horizon: float = 10.0            # Years
    time_grid: Optional[List[float]] = None  # Auto-generated if None
    seed: Optional[int] = None

    def get_time_grid(self) -> np.ndarray:
        if self.time_grid is not None:
            return np.array(self.time_grid, dtype=float)
        return np.linspace(0.0, self.time_horizon, self.n_time_steps + 1)[1:]


@dataclass
class RiskFactorSpec:
    """Specification for a single stochastic risk factor."""
    factor_id: str
    factor_type: str                  # "rate", "fx", "commodity", "vol"
    initial_value: float
    drift: float = 0.0                # mu (GBM) or mean-reversion level theta (Vasicek/CIR)
    volatility: float = 0.01
    mean_reversion_speed: float = 0.0  # kappa for Vasicek / CIR (0 = GBM)
    process: str = "gbm"              # "gbm", "vasicek", "cir"


@dataclass
class SimulationPaths:
    """Output of RiskFactorSimulator.simulate()."""
    factor_ids: List[str]
    time_grid: np.ndarray              # shape (n_time_steps,)
    paths: np.ndarray                  # shape (n_paths, n_time_steps, n_factors)


@dataclass
class ExposureProfile:
    """Expected and peak exposure profile for a single netting set."""
    netting_set_id: str
    time_grid: np.ndarray
    expected_exposure: np.ndarray       # EE(t) = mean(max(MtM, 0))
    peak_exposure: np.ndarray           # PE(t) = 95th percentile of positive exposure
    expected_positive_exposure: float   # EPE = time-average of EE
    mtm_paths: np.ndarray               # shape (n_paths, n_time_steps)
