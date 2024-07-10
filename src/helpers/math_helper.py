import numpy as np


class MathHelper:
    @staticmethod
    def v_col(x: np.ndarray) -> np.ndarray:
        return x.reshape((x.size, 1))

    @staticmethod
    def v_row(x: np.ndarray) -> np.ndarray:
        return x.reshape((1, x.size))

    @staticmethod
    def compute_mu(matrix: np.ndarray) -> np.ndarray:
        return MathHelper.v_col(matrix.mean(1))

    @staticmethod
    def center_matrix(matrix: np.ndarray) -> np.ndarray:
        mu = MathHelper.compute_mu(matrix)

        return matrix - mu

    @staticmethod
    def cv_matrix(matrix: np.ndarray) -> np.ndarray:
        centered_matrix: np.ndarray = MathHelper.center_matrix(matrix)
        # determine the main dimension of the matrix
        len_matrix = (
            float(matrix.shape[0])
            if float(matrix.shape[0]) > float(matrix.shape[1])
            else float(matrix.shape[1])
        )
        return np.dot(centered_matrix, centered_matrix.T) / len_matrix

    @staticmethod
    def compute_mu_and_sigma(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return MathHelper.compute_mu(matrix), MathHelper.cv_matrix(matrix)

    @staticmethod
    def inv_matrix(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(matrix)

    @staticmethod
    def det_matrix(matrix: np.ndarray) -> float:
        return np.linalg.det(matrix)

    @staticmethod
    def log_det_matrix(matrix: np.ndarray) -> float:
        return np.linalg.slogdet(matrix)[1]

    @staticmethod
    def mean(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.mean(matrix, axis=axis)

    @staticmethod
    def var(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.var(matrix, axis=axis)
