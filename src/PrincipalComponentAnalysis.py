import numpy as np

from helper import MathHelper as mh


class pca:
    def __init__(self, m: int = 2):
        self._set_dimensions(m)

    def _cv_matrix(self, matrix: np.ndarray) -> np.ndarray:
        centered_matrix: np.ndarray = self._center_matrix(matrix)

        return np.dot(centered_matrix, centered_matrix.T) / float(matrix.shape[0])

    def _center_matrix(self, matrix: np.ndarray) -> np.ndarray:

        mu = matrix.mean(1).reshape(-1, 1)

        return matrix - mu

    def projection_matrix(self, D: np.ndarray, m: int) -> np.ndarray:

        C = self._cv_matrix(D)
        U, _, _ = np.linalg.svd(C)

        P = U[:, 0:m]

        return np.dot(P.T, D)

    def _set_dimensions(self, m: int) -> None:
        self.m = m

    def set_train_data(self, DTrain: np.ndarray, *, m: int) -> np.ndarray:
        self.DTrain = DTrain

        self.mu = mh.v_col(np.mean(DTrain, axis=1))

    def fit(self):
        C = np.dot((self.DTrain - self.mu), (self.DTrain - self.mu).T) / float(self.DTrain.shape[1])
        U, _, _ = np.linalg.svd(C)
        P = U[:, 0:self.m]

        return P

    def apply_pca(P: np.ndarray, D: np.ndarray) -> np.ndarray:

        return np.dot(P.T, D)
