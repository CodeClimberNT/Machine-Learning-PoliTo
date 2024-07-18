from typing import Optional
import numpy as np

from src.helpers import MathHelper as mh
from src.models.gaussian_models.base_gaussian_model import BaseGaussianModel


class MultivariateGaussianModel(BaseGaussianModel):
    """
    Multivariate Gaussian (MVG) class to represent and compute properties
    of a multivariate normal distribution.

    Attributes:
    ----------
    mu_ : np.ndarray
        Mean vector of the Gaussian distribution.
    sigma_ : np.ndarray
        Covariance matrix of the Gaussian distribution.
    inv_sigma_ : np.ndarray
        Inverse of the covariance matrix.
    M_ : int
        Dimensionality of the Gaussian distribution.
    log_det_sigma_ : float
        Logarithm of the determinant of the covariance matrix.
    const_ : float
        Precomputed constant part of the log PDF equation.
    """

    def __init__(
        self, *, mu: np.ndarray | None = None, sigma: np.ndarray | None = None
    ) -> None:
        super().__init__()
        self.h_params = {}
        self.mu_: np.ndarray | None = mu
        self.sigma_: np.ndarray | None = sigma
        self.inv_sigma_: np.ndarray | None = self.utils.inv_matrix(sigma)
        self.M_: int = 0 if mu is None else len(mu)
        self.log_det_sigma_: float | None = self.utils.log_det_matrix(sigma)
        self.const_: Optional[float] = None
        if mu is not None and sigma is not None:
            self._calculate_constant_part()

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "MultivariateGaussianModel":  # type: ignore
        """
        Fit the MVG model to the data in X and computes the mean vector
        and covariance matrix.

        Parameters:
        ----------
        X : np.ndarray
            Data to fit the MVG model.
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y)

        self.mu_, self.sigma_ = self.utils.compute_mu_and_sigma(X)
        # if self.sigma_.ndim == 1:
        #     self.sigma_ = np.diag(self.sigma_)
        self.inv_sigma_ = self.utils.inv_matrix(self.sigma_)
        self.M_ = len(self.mu_)
        self.log_det_sigma_ = self.utils.log_det_matrix(self.sigma_)
        self._calculate_constant_part()
        self._calculate_h_params(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Method predict not implemented")

    def _calculate_h_params(self, X):
        for c in self.classes:
            mu_, cov_ = self.utils.compute_mu_and_sigma(X[:, self.y == c])
            self.h_params[c] = {"mean_": mu_, "sigma_": cov_}

    def pdf_GAU_ND(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function (PDF) for the given data.

        Parameters:
        ----------
        X : np.ndarray
            Data for which to compute the PDF.

        Returns:
        -------
        np.ndarray
            The probability densities.
        """
        return np.exp(self.logpdf_GAU_ND(X))

    def logpdf_GAU_ND(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log of the probability density function (PDF) for the given data.

        Parameters:
        ----------
        X : np.ndarray
            Data for which to compute the log PDF.

        Returns:
        -------
        np.ndarray
            The log probability densities.
        """
        if isinstance(X, (int, float)):
            X = np.array([[X]])

        result = [self._logpdf_GAU_ND(X[:, i : i + 1]) for i in range(X.shape[1])]
        return np.array(result).ravel()

    def _logpdf_GAU_ND(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the log PDF for a single data point.

        Parameters:
        ----------
        x : np.ndarray
            Single data point for which to compute the log PDF.

        Returns:
        -------
        float
            The log probability density.
        """
        x = mh.v_col(x)
        x_mu = x - self.mu_
        exponent = -0.5 * (x_mu.T @ self.inv_sigma_ @ x_mu).ravel()

        if self.const_ is None:
            raise ValueError(
                "The constant part of the log PDF equation is not computed."
            )
        return self.const_ + exponent

    def _calculate_constant_part(self) -> None:
        """
        Calculate the constant part of the log PDF equation.
        """
        if self.log_det_sigma_ is None:
            raise ValueError("The covariance matrix is singular.")
        self.const_ = -0.5 * self.M_ * np.log(2 * np.pi) - 0.5 * self.log_det_sigma_
