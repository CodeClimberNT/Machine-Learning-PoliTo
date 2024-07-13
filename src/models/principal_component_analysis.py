import numpy as np

from src.helpers import MathHelper as mh


class PrincipalComponentAnalysis:
    def __init__(self, *, m: int = 2):
        self.P = None
        self.U = None
        self.num_classes = None
        self.mu = None
        self.L = None
        self.D = None
        self.m = None
        self.set_dimensions(m)

    def set_dimensions(self, m: int) -> None:
        self.m = m

    def set_train_data(self, D: np.ndarray, L: np.ndarray) -> None:
        self.D = D
        self.L = L

        self.mu = mh.v_col(np.mean(D, axis=1))
        self.num_classes = len(np.unique(L))

    def fit(self, x, y) -> 'PrincipalComponentAnalysis':
        self.set_train_data(x, y)
        self.get_m_components()
        return self

    def get_m_components(self) -> np.ndarray:
        C = mh.cv_matrix(self.D)
        self.U, _, _ = np.linalg.svd(C)
        self.P = self.U[:, 0: self.m]

        return self.P

    def take_n_components(self, n: int) -> np.ndarray:
        return self.U[:, n]

    def get_projected_matrix(self) -> np.ndarray:
        if self.D is None:
            raise ValueError("Train data is not set")

        if self.P is None:
            self.get_m_components()

        return np.dot(self.P.T, self.D)

    def predict(self, D: np.ndarray) -> np.ndarray:
        return np.dot(self.P.T, D)

    def predict_custom_dir(self, *, U: np.ndarray = None, D: np.ndarray = None) -> np.ndarray:
        if U is None or D is None:
            raise ValueError("Direction or Data not set")
        return np.dot(U.T, D)

    def transform(self, D: np.ndarray) -> np.ndarray:
        return np.dot(self.P.T, D)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x)
