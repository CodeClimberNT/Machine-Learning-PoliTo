import numpy as np

from helper import MathHelper as mh


class MVG:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.M = len(mu)
        self.log_det_sigma = mh.log_det_matrix(sigma)
        self.__calculate_const()

    def logpdf_GAU_ND(self, x):
        raise NotImplementedError

    def __calculate_const(self) -> float:

        self.const = 1 / ((2 * np.pi) ** (self.d / 2) * self.det_sigma**0.5)
