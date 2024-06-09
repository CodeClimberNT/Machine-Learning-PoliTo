import numpy as np

from helper import MathHelper as mh


class MVG:
    def __init__(
            self, X: np.ndarray = None, mu: np.ndarray = None, sigma: np.ndarray = None
    ) -> None:
        if X is None and (mu is None or sigma is None):
            print(
                "No value was passed, to create a MVG object please run the fit method."
            )
            return
        elif X is not None and (mu is not None or sigma is not None):
            raise ValueError("You can't pass X and mu/sigma at the same time.")
        elif X is not None and (mu is None and sigma is None):
            mu, sigma = mh.compute_mu_and_sigma(X)

        self.mu, self.sigma = mu, sigma
        self.inv_sigma = mh.inv_matrix(self.sigma)
        self.M = len(self.mu)
        self.log_det_sigma = mh.log_det_matrix(self.sigma)
        self.__calculate_const()

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.mu, self.sigma = mh.compute_mu_and_sigma(X)
        self.inv_sigma = mh.inv_matrix(self.sigma)
        self.M = len(self.mu)
        self.log_det_sigma = mh.log_det_matrix(self.sigma)
        self.__calculate_const()
        return self.mu, self.sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logpdf_GAU_ND(X).sum()

    def pdf_GAU_ND(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf_GAU_ND(X))

    def logpdf_GAU_ND(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, (int, float)):
            return self.__logpdf_GAU_ND(np.array([X]))

        # X have a shape of (M, N)
        # result = np.zeros(X.shape[1])
        # for i in range(X.shape[1]):
        #     result[i] = self.__logpdf_GAU_ND(X[:, i:i+1])

        result = [self.__logpdf_GAU_ND(X[:, i:i + 1]) for i in range(X.shape[1])]

        result = np.array(result).ravel()
        return result

    def __logpdf_GAU_ND(self, x: np.ndarray) -> float:
        x = mh.v_col(x)
        x_mu: np.ndarray = x - self.mu
        exponent = -0.5 * (x_mu.T @ self.inv_sigma @ x_mu).ravel()
        return self.const + exponent

    def __calculate_const(self) -> float:
        self.const = -0.5 * self.M * np.log(2 * np.pi) - 0.5 * self.log_det_sigma
