import numpy as np

from helper import MathHelper as mh


class MVG:
    """
    Multivariate Gaussian (MVG) class represents a multivariate Gaussian distribution.

    Attributes:
        mu (np.ndarray): Mean vector of the Gaussian distribution.
        sigma (np.ndarray): Covariance matrix of the Gaussian distribution.
        inv_sigma (np.ndarray): Inverse of the covariance matrix.
        M (int): Dimensionality of the Gaussian distribution.
        log_det_sigma (float): Logarithm of the determinant of the covariance matrix.
    """

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        """
        Initializes a new instance of the MVG class.

        Args:
            mu (np.ndarray): Mean vector of the Gaussian distribution.
            sigma (np.ndarray): Covariance matrix of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma
        self.inv_sigma = mh.inv_matrix(sigma)
        self.M = len(mu)
        self.log_det_sigma = mh.log_det_matrix(sigma)
        self.__calculate_const()

    def pdf_GAU_ND(self, X: np.ndarray[np.ndarray]) -> np.ndarray:
        """
        Computes the probability density function (PDF) of the multivariate Gaussian distribution.

        Args:
            X (np.ndarray): Input data with shape (M, N), where M is the dimensionality and N is the number of samples.

        Returns:
            np.ndarray: Array of PDF values for each input sample.
        """
        return np.exp(self.logpdf_GAU_ND(X))

    def logpdf_GAU_ND(self, X: np.ndarray[np.ndarray]) -> np.ndarray:
        """
        Computes the logarithm of the probability density function (PDF) of the multivariate Gaussian distribution.

        Args:
            X (np.ndarray): Input data with shape (M, N), where M is the dimensionality and N is the number of samples.

        Returns:
            np.ndarray: Array of logarithm of PDF values for each input sample.
        """
        if isinstance(X, (int, float)):
            return self.__logpdf_GAU_ND(np.array([X]))

        # X have a shape of (M, N)

        result = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            result[i] = self.__logpdf_GAU_ND(X[:, i])

        return result

    def __logpdf_GAU_ND(self, x: np.ndarray) -> float:
        """
        Computes the logarithm of the probability density function (PDF) of the multivariate Gaussian distribution for a single input sample.

        Args:
            x (np.ndarray): Input sample with shape (M, 1), where M is the dimensionality.

        Returns:
            float: Logarithm of the PDF value for the input sample.
        """
        x = mh.v_col(x)
        x_mu: np.ndarray = x - self.mu
        exponent = -0.5 * (x_mu.T @ self.inv_sigma @ x_mu)
        return self.const + exponent

    def __calculate_const(self) -> float:
        """
        Calculates the constant term used in the logarithm of the probability density function (PDF) of the multivariate Gaussian distribution.

        Returns:
            float: Constant term used in the logarithm of the PDF.
        """
        self.const = -0.5 * self.M * np.log(2 * np.pi) - 0.5 * self.log_det_sigma
