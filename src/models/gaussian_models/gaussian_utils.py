import numpy as np

from src.helpers import MathHelper as mh


class GaussianUtils:
    @staticmethod
    def compute_mu_and_sigma(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return mh.compute_mu_and_sigma(X)

    @staticmethod
    def inv_matrix(matrix: np.ndarray) -> np.ndarray | None:
        if matrix is None:
            return None
        return mh.inv_matrix(matrix)

    @staticmethod
    def log_det_matrix(matrix: np.ndarray) -> float | None:
        if matrix is None:
            return None
        return mh.log_det_matrix(matrix)

    @staticmethod
    def calculate_probability_distribution(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        inv_cov = mh.inv_matrix(sigma)
        log_det_cov = mh.log_det_matrix(sigma)
        return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * log_det_cov - 0.5 * (
                (x - mu) * (inv_cov @ (x - mu))).sum(0)
