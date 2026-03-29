"""Stochastic risk factor simulation: GBM, Vasicek, CIR."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..common.exceptions import SimulationError
from .models import RiskFactorSpec, SimulationConfig, SimulationPaths


class RiskFactorSimulator:
    """
    Simulate correlated risk factor paths using Euler discretisation.

    Supported processes:
      gbm      : dS = mu*S*dt + sigma*S*dW
      vasicek  : dr = kappa*(theta - r)*dt + sigma*dW
      cir      : dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config

    def simulate(
        self,
        factor_specs: List[RiskFactorSpec],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> SimulationPaths:
        """
        Parameters
        ----------
        factor_specs : one spec per risk factor
        correlation_matrix : n_factors x n_factors PSD matrix (identity if None)

        Returns
        -------
        SimulationPaths with shape (n_paths, n_time_steps, n_factors)
        """
        cfg = self._config
        rng = np.random.default_rng(cfg.seed)
        time_grid = cfg.get_time_grid()
        n_steps = len(time_grid)
        n_paths = cfg.n_paths
        n_factors = len(factor_specs)

        if correlation_matrix is None:
            L = np.eye(n_factors)
        else:
            if correlation_matrix.shape != (n_factors, n_factors):
                raise SimulationError(
                    f"Correlation matrix shape {correlation_matrix.shape} does not "
                    f"match number of factors {n_factors}."
                )
            try:
                L = np.linalg.cholesky(correlation_matrix)
            except np.linalg.LinAlgError:
                reg = correlation_matrix + 1e-8 * np.eye(n_factors)
                L = np.linalg.cholesky(reg)

        # Paths array: (n_paths, n_steps, n_factors)
        paths = np.zeros((n_paths, n_steps, n_factors), dtype=float)

        # Initial values
        x0 = np.array([spec.initial_value for spec in factor_specs], dtype=float)

        t_prev = 0.0
        x_cur = np.tile(x0, (n_paths, 1))  # (n_paths, n_factors)

        for step_idx, t in enumerate(time_grid):
            dt = t - t_prev
            if dt <= 0:
                paths[:, step_idx, :] = x_cur
                t_prev = t
                continue

            # Correlated standard normals: (n_paths, n_factors)
            Z_indep = rng.standard_normal((n_paths, n_factors))
            Z_corr = Z_indep @ L.T

            x_next = np.zeros_like(x_cur)
            for j, spec in enumerate(factor_specs):
                x_next[:, j] = self._step(
                    x_cur[:, j], dt, Z_corr[:, j], spec
                )

            paths[:, step_idx, :] = x_next
            x_cur = x_next
            t_prev = t

        return SimulationPaths(
            factor_ids=[s.factor_id for s in factor_specs],
            time_grid=time_grid,
            paths=paths,
        )

    # ------------------------------------------------------------------
    def _step(self, x: np.ndarray, dt: float,
              Z: np.ndarray, spec: RiskFactorSpec) -> np.ndarray:
        sqrt_dt = math.sqrt(dt)
        if spec.process == "gbm":
            return x * np.exp(
                (spec.drift - 0.5 * spec.volatility ** 2) * dt
                + spec.volatility * sqrt_dt * Z
            )
        if spec.process == "vasicek":
            kappa = spec.mean_reversion_speed
            theta = spec.drift
            sigma = spec.volatility
            return (x * math.exp(-kappa * dt)
                    + theta * (1.0 - math.exp(-kappa * dt))
                    + sigma * sqrt_dt * Z)
        if spec.process == "cir":
            kappa = spec.mean_reversion_speed
            theta = spec.drift
            sigma = spec.volatility
            x_pos = np.maximum(x, 0.0)
            return np.maximum(
                x_pos + kappa * (theta - x_pos) * dt
                + sigma * np.sqrt(x_pos) * sqrt_dt * Z,
                0.0,
            )
        raise SimulationError(f"Unknown stochastic process: '{spec.process}'")
