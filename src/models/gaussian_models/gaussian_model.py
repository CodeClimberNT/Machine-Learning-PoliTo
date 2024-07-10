import numpy as np
import scipy

from src.helpers import MathHelper as mh
from src.models.base_model import BaseModel


class GaussianModel(BaseModel):
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

    def __init__(self, *, mu: np.ndarray = None, sigma: np.ndarray = None):
        super().__init__()
        self.mu_ = mu
        self.sigma_ = sigma
        self.inv_sigma_ = None if sigma is None else mh.inv_matrix(sigma)
        self.M_ = 0 if mu is None else len(mu)
        self.log_det_sigma_ = None if sigma is None else mh.log_det_matrix(sigma)
        self.const_ = None
        if mu is not None and sigma is not None:
            self._calculate_const()

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'GaussianModel':
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
        if self.mu_ is None or self.sigma_ is None:
            self.mu_, self.sigma_ = mh.compute_mu_and_sigma(X)
            if self.sigma_.ndim == 1:
                self.sigma_ = np.diag(self.sigma_)
            self.inv_sigma_ = mh.inv_matrix(self.sigma_)
            self.M_ = len(self.mu_)
            self.log_det_sigma_ = mh.log_det_matrix(self.sigma_)
            self._calculate_const()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the log probability density for the given data.

        Parameters:
        ----------
        X : np.ndarray
            Data for which to predict the log probability density.

        Returns:
        -------
        np.ndarray
            The log probability densities.
        """
        return self.logpdf_GAU_ND(X).sum()

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

        result = [self._logpdf_GAU_ND(X[:, i:i + 1]) for i in range(X.shape[1])]
        return np.array(result).ravel()

    def _logpdf_GAU_ND(self, x: np.ndarray) -> float:
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
        return self.const_ + exponent

    def _calculate_const(self) -> None:
        """
        Calculate the constant part of the log PDF equation.
        """
        self.const_ = -0.5 * self.M_ * np.log(2 * np.pi) - 0.5 * self.log_det_sigma_

    @staticmethod
    def calculate_probability_distribution(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        inv_cov = mh.inv_matrix(sigma)
        log_det_cov = mh.log_det_matrix(sigma)
        return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * log_det_cov - 0.5 * ((x - mu) * (inv_cov @ (x - mu))).sum(
            0)

    def _compute_log_likelihood(self, X_val: np.ndarray) -> np.ndarray:
        S = np.zeros((self.num_classes, X_val.shape[1]))
        for c in range(self.num_classes):
            # compute mean and covariance for each class
            x_c = self.x_train[:, self.y_train == c]
            mu_c, sigma_c = mh.compute_mu_and_sigma(x_c)
            S[c, :] = GaussianModel.calculate_probability_distribution(X_val, mu_c, sigma_c)
        return S

    def compute_log_posterior(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the matrix of joint densities SJoint for all test samples and classes.

        Parameters:
        ----------
        X_test : np.ndarray
            Test dataset.

        Returns:
        -------
        np.ndarray
            The matrix of joint densities SJoint.
        """
        log_likelihood = self._compute_log_likelihood(X)
        S_joint = self._compute_SJoint(log_likelihood, np.ones(self.num_classes) / float(self.num_classes))
        S_marginal = mh.v_row(scipy.special.logsumexp(S_joint, axis=0))
        return S_joint - S_marginal

    def _compute_SJoint(self, log_likelihood, prior_prob) -> np.ndarray:
        return log_likelihood + mh.v_col(np.log(prior_prob))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the mean accuracy of the MVG model.

        Parameters:
        ----------
        X : np.ndarray
            Data to predict.
        y : np.ndarray
            Target values.

        Returns:
        -------
        float
            Mean accuracy of the MVG model.
        """
        return np.mean(self.predict(X) == y)
