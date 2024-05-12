import numpy as np

from helper import MathHelper as mh


class MVG:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.inv_sigma = mh.inv_matrix(sigma)
        self.M = len(mu)
        self.log_det_sigma = mh.log_det_matrix(sigma)
        self.__calculate_const()

    def pdf_GAU_ND(self, X: np.ndarray[np.ndarray]) -> np.ndarray:
        return np.exp(self.logpdf_GAU_ND(X))

    def logpdf_GAU_ND(self, X: np.ndarray[np.ndarray]) -> np.ndarray:
        if isinstance(X, (int, float)):
            return self.__logpdf_GAU_ND(np.array([X]))
        
        # X have a shape of (M, N)

        result = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            result[i] = self.__logpdf_GAU_ND(X[:, i])

        return result
    

    def __logpdf_GAU_ND(self, x: np.ndarray) -> float:
        x = mh.v_col(x)
        x_mu: np.ndarray = x - self.mu
        exponent = -0.5 * (x_mu.T @ self.inv_sigma @ x_mu)
        return self.const + exponent

    def __calculate_const(self) -> float:
        self.const = -0.5 * self.M * np.log(2 * np.pi) - 0.5 * self.log_det_sigma
