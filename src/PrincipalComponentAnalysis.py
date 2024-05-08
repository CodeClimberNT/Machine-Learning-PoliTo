import numpy as np

from helper import MathHelper as mh


class PCA:
    def __init__(self, *, m: int = 2):
        self.set_dimensions(m)

    def set_dimensions(self, m: int) -> None:
        self.m = m

    def set_train_data(self, D: np.ndarray, L: np.ndarray) -> np.ndarray:
        self.D = D
        self.L = L

        self.mu = mh.v_col(np.mean(D, axis=1))
        self.num_classes = len(np.unique(L))

    def fit(self) -> None:
        self.get_m_components()

    def get_m_components(self) -> np.ndarray:
        C = mh.cv_matrix(self.D)
        self.U, _, _ = np.linalg.svd(C)
        self.P = self.U[:, 0 : self.m]

        return self.P

    def get_projected_matrix(self) -> np.ndarray:
        if self.D is None:
            raise ValueError("Train data is not set")

        if self.P is None:
            self.get_m_components()

        return np.dot(self.P.T, self.D)

    def predict(self, D: np.ndarray) -> np.ndarray:
        return np.dot(self.P.T, D)
