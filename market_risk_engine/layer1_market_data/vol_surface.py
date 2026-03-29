"""Volatility surface interpolation and SABR model calibration."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from ..common.exceptions import CalibrationError
from .models import VolSurface


# ---------------------------------------------------------------------------
# SABR model
# ---------------------------------------------------------------------------

def _sabr_vol(F: float, K: float, T: float,
              alpha: float, beta: float, rho: float, nu: float) -> float:
    """Hagan et al. (2002) SABR lognormal implied volatility approximation."""
    if abs(F - K) < 1e-12:
        FK = F
        log_fk = 0.0
    else:
        FK = math.sqrt(F * K)
        log_fk = math.log(F / K)

    FK_beta = FK ** (1.0 - beta)
    z = (nu / alpha) * FK_beta * log_fk
    x_z: float
    if abs(z) < 1e-9:
        x_z = 1.0
    else:
        disc = math.sqrt(1.0 - 2.0 * rho * z + z * z)
        x_z = z / math.log((disc + z - rho) / (1.0 - rho))

    A = alpha / (FK_beta * (1.0 + ((1.0 - beta) ** 2 / 24.0) * log_fk ** 2
                             + ((1.0 - beta) ** 4 / 1920.0) * log_fk ** 4))
    B = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / FK_beta ** 2
        + (rho * beta * nu * alpha) / (4.0 * FK_beta)
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    ) * T

    return A * x_z * B


class SABRCalibrator:
    """Calibrate SABR parameters to a single expiry slice."""

    def calibrate_slice(
        self,
        forward: float,
        expiry: float,
        strikes: List[float],
        market_vols: List[float],
        beta: float = 0.5,
    ) -> Dict[str, float]:
        """
        Returns {"alpha": ..., "beta": ..., "rho": ..., "nu": ...}.
        beta is fixed; alpha, rho, nu are calibrated.
        """
        market_vols = np.array(market_vols, dtype=float)

        def objective(params: np.ndarray) -> float:
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or rho <= -1 or rho >= 1:
                return 1e10
            total = 0.0
            for K, mv in zip(strikes, market_vols):
                try:
                    model_v = _sabr_vol(forward, K, expiry, alpha, beta, rho, nu)
                except Exception:
                    return 1e10
                total += (model_v - mv) ** 2
            return total

        atm_vol = float(np.interp(forward, strikes, market_vols))
        x0 = np.array([atm_vol * (forward ** (1.0 - beta)), 0.0, 0.3])
        bounds = [(1e-6, None), (-0.999, 0.999), (1e-6, None)]

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-14})
        if not res.success and res.fun > 1e-6:
            raise CalibrationError(
                f"SABR calibration did not converge for expiry {expiry}: {res.message}"
            )

        alpha, rho, nu = res.x
        return {"alpha": float(alpha), "beta": float(beta),
                "rho": float(rho), "nu": float(nu)}

    def implied_vol(
        self, forward: float, strike: float, expiry: float,
        alpha: float, beta: float, rho: float, nu: float
    ) -> float:
        return _sabr_vol(forward, strike, expiry, alpha, beta, rho, nu)


# ---------------------------------------------------------------------------
# Surface interpolator
# ---------------------------------------------------------------------------

class VolSurfaceInterpolator:
    """Bilinear interpolation across the expiry–strike grid; SABR if calibrated."""

    def __init__(self, surface: VolSurface) -> None:
        self._surface = surface
        self._calibrator = SABRCalibrator() if surface.sabr_params else None

    def get_vol(self, expiry: float, strike: float,
                forward: Optional[float] = None) -> float:
        """
        Return the implied vol for a given expiry (year fraction) and strike.
        Uses SABR if `sabr_params` are present for the nearest expiry slice;
        falls back to bilinear interpolation on the grid.
        """
        s = self._surface
        if s.sabr_params and forward is not None:
            # Find nearest calibrated expiry
            expiries = np.array(sorted(s.sabr_params.keys()))
            idx = int(np.argmin(np.abs(expiries - expiry)))
            nearest_exp = float(expiries[idx])
            p = s.sabr_params[nearest_exp]
            return self._calibrator.implied_vol(
                forward, strike, nearest_exp,
                p["alpha"], p["beta"], p["rho"], p["nu"]
            )

        # Bilinear interpolation on the grid
        exp_arr = np.array(s.expiries, dtype=float)
        str_arr = np.array(s.strikes, dtype=float)
        vols = s.vols  # shape (n_expiry, n_strike)

        # Clamp to grid boundaries
        expiry_c = float(np.clip(expiry, exp_arr[0], exp_arr[-1]))
        strike_c = float(np.clip(strike, str_arr[0], str_arr[-1]))

        # Find surrounding indices for expiry
        ei = int(np.searchsorted(exp_arr, expiry_c, side="right")) - 1
        ei = max(0, min(ei, len(exp_arr) - 2))
        si = int(np.searchsorted(str_arr, strike_c, side="right")) - 1
        si = max(0, min(si, len(str_arr) - 2))

        # Interpolation weights
        t_e = (expiry_c - exp_arr[ei]) / (exp_arr[ei + 1] - exp_arr[ei] + 1e-15)
        t_s = (strike_c - str_arr[si]) / (str_arr[si + 1] - str_arr[si] + 1e-15)

        v00 = vols[ei, si]
        v10 = vols[ei + 1, si]
        v01 = vols[ei, si + 1]
        v11 = vols[ei + 1, si + 1]

        return float(
            (1 - t_e) * (1 - t_s) * v00
            + t_e * (1 - t_s) * v10
            + (1 - t_e) * t_s * v01
            + t_e * t_s * v11
        )
