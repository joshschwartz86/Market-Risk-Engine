"""Correlation matrix management with nearest-PSD projection (Higham 2002)."""
from __future__ import annotations

from typing import List

import numpy as np

from ..common.exceptions import CorrelationMatrixError


class CorrelationManager:
    """
    Build and validate a correlation matrix for a set of risk factors.
    Defaults to identity (uncorrelated) for any unspecified pair.
    """

    def __init__(self, factor_ids: List[str]) -> None:
        self._ids = list(factor_ids)
        n = len(self._ids)
        self._matrix = np.eye(n, dtype=float)
        self._idx = {fid: i for i, fid in enumerate(self._ids)}

    @property
    def factor_ids(self) -> List[str]:
        return list(self._ids)

    def set_correlation(self, factor_i: str, factor_j: str, rho: float) -> None:
        if abs(rho) > 1.0:
            raise CorrelationMatrixError(f"Correlation must be in [-1, 1], got {rho}.")
        i = self._idx[factor_i]
        j = self._idx[factor_j]
        self._matrix[i, j] = rho
        self._matrix[j, i] = rho

    def get_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def validate_positive_semidefinite(self) -> bool:
        eigenvalues = np.linalg.eigvalsh(self._matrix)
        return bool(np.all(eigenvalues >= -1e-8))

    def nearest_psd(self, epsilon: float = 1e-8) -> np.ndarray:
        """
        Return the nearest positive semi-definite matrix using Higham (2002).
        Modifies in place and returns the corrected matrix.
        """
        A = self._matrix.copy()
        n = A.shape[0]
        delta_S = np.zeros_like(A)
        Y = A.copy()

        for _ in range(200):
            R = Y - delta_S
            # Symmetric polar factor
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.maximum(eigvals, epsilon)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T
            delta_S = X - R
            # Project onto unit diagonal
            np.fill_diagonal(X, 1.0)
            Y = X.copy()
            if np.max(np.abs(Y - A)) < 1e-10:
                break

        self._matrix = Y
        return Y

    def cholesky(self) -> np.ndarray:
        """
        Return the lower-triangular Cholesky factor L such that L @ L.T = C.
        If the matrix is not PSD, the nearest PSD is used first.
        """
        if not self.validate_positive_semidefinite():
            self.nearest_psd()
        try:
            return np.linalg.cholesky(self._matrix)
        except np.linalg.LinAlgError:
            # Regularise slightly
            reg = self._matrix + 1e-8 * np.eye(self._matrix.shape[0])
            return np.linalg.cholesky(reg)
