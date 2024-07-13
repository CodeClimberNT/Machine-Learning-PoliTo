from typing import Tuple

import numpy as np
import scipy.linalg

from src.helpers import MathHelper as mh


class LinearDiscriminantAnalysis:
    def __init__(self, solver: str = "svd", m: int = 2):
        self.num_classes = None
        self.labels = None
        self.mu = None
        self.y = None
        self.m = None
        self.x = None
        self.solver = None
        self.threshold = None
        self.set_solver(solver)
        self.set_dimensions(m)
        self.last_predicted = None

    def set_solver(self, solver: str):
        valid_solver = self.get_valid_solver()
        if solver not in valid_solver:
            raise ValueError(f"Invalid solver. Choose one from: {valid_solver}")
        else:
            self.solver = solver
        return self

    @staticmethod
    def get_valid_solver() -> tuple[str, str]:
        return "svd", "eigh"

    def set_dimensions(self, m: int) -> None:
        self.m = m

    def set_train_data(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> None:
        self.x = x
        self.y = y
        self.mu = mh.v_col(np.mean(x, axis=1))
        self.labels = list(np.unique(y))
        self.num_classes = len(np.unique(y))

    def fit(self, x, y) -> 'LinearDiscriminantAnalysis':
        self.set_train_data(x, y)
        self.solve()
        self.get_projected_matrix()
        self.calculate_threshold()
        return self

    def _compute_SB(self) -> np.ndarray:
        Sb = 0
        for c in np.unique(self.y):
            Dc = self.x[:, self.y == c]
            mu_c = mh.v_col(np.mean(Dc, axis=1))
            Sb += np.dot((mu_c - self.mu), (mu_c - self.mu).T) * Dc.shape[1]
        return Sb

    def _compute_SW(self) -> np.ndarray:
        Sw = 0

        for c in np.unique(self.y):
            Dc = self.x[:, self.y == c]
            mu_c = mh.v_col(np.mean(Dc, axis=1))
            Sw += np.dot((Dc - mu_c), (Dc - mu_c).T)
        return Sw / self.x.shape[1]

    def solve(self) -> np.ndarray | None:
        if self.solver == "svd":
            return self.solve_svd()
        elif self.solver == "eigh":
            return self.solve_eigh()
        else:
            raise ValueError("Invalid solver")

    def solve_svd(self):
        Sw: np.ndarray = self._compute_SW()
        U1, s, _ = np.linalg.svd(Sw)
        P1: np.ndarray = np.dot(U1 * mh.v_row(1.0 / (s ** 0.5)), U1.T)

        Sb: np.ndarray = self._compute_SB()
        Sbt: np.ndarray = np.dot(P1, np.dot(Sb, P1.T))
        self.U, _, _ = np.linalg.svd(Sbt)
        P2: np.ndarray = self.U[:, 0: self.m]
        self.W: np.ndarray = np.dot(P2.T, P1).T
        return self.W

    def solve_eigh(self) -> np.ndarray:
        Sb: np.ndarray = self._compute_SB()
        Sw: np.ndarray = self._compute_SW()
        _, self.U = scipy.linalg.eigh(Sb, Sw)
        self.W: np.ndarray = self.U[:, 0: self.m]
        return self.W

    def get_projected_matrix(self):
        if hasattr(self, "W") is False:
            raise ValueError("W is not set. Run fit method first")

        self.projected_matrix: np.ndarray = self.W.T @ self.x
        if (
                self.projected_matrix[0, self.y == self.labels[0]].mean()
                > self.projected_matrix[0, self.y == self.labels[1]].mean()
        ):
            self.W = -self.W
            self.projected_matrix = self.W.T @ self.x

        return self.projected_matrix

    def calculate_threshold(self) -> float:
        if hasattr(self, "projected_matrix") is False or self.projected_matrix is None:
            raise ValueError("Projected matrix is not set. Run fit method first")

        self.threshold: float = (
                                        self.projected_matrix[0, self.y == self.labels[0]].mean()
                                        + self.projected_matrix[0, self.y == self.labels[1]].mean()
                                ) / 2.0

        return self.threshold

    def set_threshold(self, threshold: float) -> float:
        self.threshold = threshold
        return self.threshold

    def optimize_threshold(
            self, *, steps: np.int64, show_threshold: bool = True
    ) -> float:
        if hasattr(self, "projected_matrix") is False or self.projected_matrix is None:
            raise ValueError("Projected matrix is not set. Run fit method first")
        if hasattr(self, "threshold") is False or self.threshold is None:
            self.calculate_threshold()

        best_threshold = self.threshold
        _, best_error_rate = self.validate(self.x, self.y)

        for threshold in np.linspace(
                self.projected_matrix.min(), self.projected_matrix.max(), steps
        ):
            predicted = np.zeros(shape=self.projected_matrix.shape, dtype=np.int32)
            predicted[self.projected_matrix >= threshold] = self.labels[1]
            predicted[self.projected_matrix < threshold] = self.labels[0]

            error_rate = self.get_error_rate(predicted, self.y)
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                best_threshold = threshold
        if best_threshold != self.threshold:
            if show_threshold:
                print(f"Found better threshold: {best_threshold}")
                print(f"New error rate: {best_error_rate * 100}")

            self.threshold = best_threshold

        return self.threshold

    def validate(
            self,
            x_val: np.ndarray,
            y_val: np.ndarray,
            *,
            threshold: float = None,
            show_results: bool = False,
    ) -> tuple[np.ndarray, float]:

        if threshold:
            return self.validate_custom_threshold(x_val, y_val, threshold=threshold, show_results=show_results)

        if hasattr(self, "W") is False:
            raise ValueError("W is not set. Run fit method first")

        if hasattr(self, "threshold") is False or not self.threshold:
            self.calculate_threshold()

        predicted = self.__get_predicted(x_val, self.threshold)

        error_rate = self.get_error_rate(predicted, y_val)
        if show_results and y_val is not None:
            self.show_error_rate(predicted, y_val, error_rate)

        return predicted, error_rate

    def validate_custom_threshold(
            self,
            x_val: np.ndarray,
            y_val: np.ndarray,
            *,
            threshold: float = None,
            show_results: bool = False,
    ) -> tuple[np.ndarray, float]:
        if hasattr(self, "W") is False:
            raise ValueError("W is not set. Run fit method first")

        if y_val is None:
            raise ValueError("Validation data not set")

        if threshold is None:
            raise ValueError("Threshold not set")

        predicted = self.__get_predicted(x_val, threshold)

        error_rate = self.get_error_rate(predicted, y_val)
        if show_results and y_val is not None:
            self.show_error_rate(predicted, y_val, error_rate)

        return predicted, error_rate

    def __get_predicted(self, x_val: np.ndarray, threshold: float) -> np.ndarray:
        projected = self.transform(x_val)

        predicted = np.zeros(shape=projected.shape, dtype=np.int32)
        predicted[projected >= threshold] = self.labels[1]
        predicted[projected < threshold] = self.labels[0]

        return predicted

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.W.T @ x

    def show_error_rate(
            self, predicted: np.ndarray, labels: np.ndarray, error_rate: float
    ) -> None:
        print(
            f"Error rate: {(predicted != labels).sum()}/{(labels.size)} => {(error_rate * 100):.1f}%"
        )

    def get_error_rate(
            self,
            predicted: np.ndarray,
            labels: np.ndarray,
            *,
            show_values: bool = False,
    ) -> float:
        error_rate: float = (predicted != labels).sum() / float(labels.size)

        if show_values:
            show_values(predicted, labels, error_rate)

        return error_rate

    def take_n_components(self, n: int) -> np.ndarray:
        return self.U[:, n]

    def predict_custom_dir(
            self, *, U: np.ndarray = None, x: np.ndarray = None
    ) -> np.ndarray:
        if U is None or x is None:
            raise ValueError("Direction or Data not set")
        return np.dot(U.T, x)
