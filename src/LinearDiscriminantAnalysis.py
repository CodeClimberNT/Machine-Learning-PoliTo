import numpy as np
import scipy.linalg

from helper import MathHelper as mh


class LDA:
    def __init__(self, solver: str = "svd", m: int = 2):
        self.set_solver(solver)
        self.set_dimensions(m)

    def set_solver(self, solver: str):
        valid_solver = self.get_valid_solver()
        if solver not in valid_solver:
            raise ValueError(f"Invalid solver. Choose one from: {valid_solver}")
        else:
            self.solver = solver
        return self

    def get_valid_solver(self) -> tuple[str]:
        return ("svd", "eigh")

    def set_dimensions(self, m: int) -> None:
        self.m = m

    def set_train_data(
        self,
        DTrain: np.ndarray,
        LTrain: np.ndarray,
    ) -> None:
        self.DTrain = DTrain
        self.LTrain = LTrain
        self.mu = mh.v_col(np.mean(DTrain, axis=1))

        self.num_classes = len(np.unique(LTrain))

    def fit(self):
        self.solve()
        self.get_projected_matrix()

    def _compute_SB(self):
        Sb = 0
        for c in np.unique(self.LTrain):
            Dc = self.DTrain[:, self.LTrain == c]
            mu_c = mh.v_col(np.mean(Dc, axis=1))
            Sb += np.dot((mu_c - self.mu), (mu_c - self.mu).T) * Dc.shape[1]
        return Sb

    def _compute_SW(self):
        Sw = 0

        for c in np.unique(self.LTrain):
            Dc = self.DTrain[:, self.LTrain == c]
            mu_c = mh.v_col(np.mean(Dc, axis=1))
            Sw += np.dot((Dc - mu_c), (Dc - mu_c).T)
        return Sw / self.DTrain.shape[1]

    def solve(self) -> np.ndarray:
        if self.solver == "svd":
            return self.solve_svd()
        elif self.solver == "eigh":
            return self.solve_eigh()
        else:
            raise ValueError("Invalid solver")

    def solve_svd(self):
        Sw = self._compute_SW()
        U, s, _ = np.linalg.svd(Sw)
        P1 = np.dot(U * mh.v_row(1.0 / (s**0.5)), U.T)

        Sb = self._compute_SB()
        Sbt = np.dot(P1, np.dot(Sb, P1.T))
        U2, _, _ = np.linalg.svd(Sbt)
        P2 = U2[:, 0 : self.m]
        self.W = np.dot(P2.T, P1).T
        return self.W

    def solve_eigh(self) -> np.ndarray:
        Sb = self._compute_SB()
        Sw = self._compute_SW()
        _, U = scipy.linalg.eigh(Sb, Sw)
        self.W = U[:, 0 : self.m]
        return self.W

    def get_projected_matrix(self):
        if hasattr(self, "W") is False:
            raise ValueError("W is not set. Run fit method first")

        self.projected_matrix: np.ndarray = self.W.T @ self.DTrain
        return self.projected_matrix

    def calculate_threshold(self) -> float:
        if hasattr(self, "projected_matrix") is False:
            raise ValueError("Projected matrix is not set. Run fit method first")
        
        self.threshold: float = (
            self.projected_matrix[0, self.LTrain == 1].mean()
            + self.projected_matrix[0, self.LTrain == 2].mean()
        ) / 2.0
        return self.threshold

    def predict(self, DVal: np.ndarray) -> np.ndarray:
        if hasattr(self, "W") is False:
            raise ValueError("W is not set. Run fit method first")
        if hasattr(self, "threshold") is False:
            self.calculate_threshold()

        projected_values = self.W.T @ DVal
        print(projected_values.shape)
        print(self.threshold)
        predicted_values = np.zeros(shape=projected_values.shape, dtype=np.int32)
        predicted_values[projected_values >= self.threshold] = 2
        predicted_values[projected_values < self.threshold] = 1
        return predicted_values
