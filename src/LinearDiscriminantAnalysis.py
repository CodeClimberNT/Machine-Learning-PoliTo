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
        self.labels = list(np.unique(LTrain))
        # print(self.labels)
        self.num_classes = len(np.unique(LTrain))

    def fit(self, x, y):
        self.set_train_data(x, y)
        self.solve()
        self.get_projected_matrix()
        self.calculate_threshold()

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
        U1, s, _ = np.linalg.svd(Sw)
        P1 = np.dot(U1 * mh.v_row(1.0 / (s**0.5)), U1.T)

        Sb = self._compute_SB()
        Sbt = np.dot(P1, np.dot(Sb, P1.T))
        self.U, _, _ = np.linalg.svd(Sbt)
        P2 = self.U[:, 0 : self.m]
        self.W = np.dot(P2.T, P1).T
        return self.W

    def solve_eigh(self) -> np.ndarray:
        Sb = self._compute_SB()
        Sw = self._compute_SW()
        _, self.U = scipy.linalg.eigh(Sb, Sw)
        self.W = self.U[:, 0 : self.m]
        return self.W

    def get_projected_matrix(self):
        if hasattr(self, "W") is False:
            raise ValueError("W is not set. Run fit method first")

        self.projected_matrix: np.ndarray = self.W.T @ self.DTrain
        if (
            self.projected_matrix[0, self.LTrain == self.labels[0]].mean()
            > self.projected_matrix[0, self.LTrain == self.labels[1]].mean()
        ):
            self.W = -self.W
            self.projected_matrix = self.W.T @ self.DTrain

        return self.projected_matrix

    def calculate_threshold(self) -> float:
        if hasattr(self, "projected_matrix") is False or self.projected_matrix is None:
            raise ValueError("Projected matrix is not set. Run fit method first")

        self.threshold: float = (
            self.projected_matrix[0, self.LTrain == self.labels[0]].mean()
            + self.projected_matrix[0, self.LTrain == self.labels[1]].mean()
        ) / 2.0

        return self.threshold

    def predict(
        self,
        DVal: np.ndarray,
        *,
        show_error_rate: bool = False,
        LVal: np.ndarray = None,
    ) -> np.ndarray:
        if hasattr(self, "W") is False:
            raise ValueError("W is not set. Run fit method first")
        if hasattr(self, "threshold") is False or not self.threshold:
            self.calculate_threshold()

        projected = self.W.T @ DVal

        PVal = np.zeros(shape=projected.shape, dtype=np.int32)
        PVal[projected >= self.threshold] = self.labels[1]
        PVal[projected < self.threshold] = self.labels[0]

        if show_error_rate and LVal is not None:
            print(f"Error rate: {(PVal != LVal).sum()}/{(LVal.size)} => {((PVal != LVal).sum() / float(LVal.size) * 100):.1f}%")

        return PVal

    def take_n_components(self, n: int) -> np.ndarray:
        return self.U[:, n]

    def predict_custom_dir(
        self, *, U: np.ndarray = None, D: np.ndarray = None
    ) -> np.ndarray:
        if U is None or D is None:
            raise ValueError("Direction or Data not set")
        return np.dot(U.T, D)
