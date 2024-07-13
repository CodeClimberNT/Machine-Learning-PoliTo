from typing import Any
import numpy as np


class DataHandler:
    @staticmethod
    def split_db_2to1(
        x: np.ndarray, y: np.ndarray, seed=0
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        n_train = int(x.shape[1] * 2.0 / 3.0)
        np.random.seed(seed)

        idx = np.random.permutation(x.shape[1])

        idx_train = idx[0:n_train]
        idx_test = idx[n_train:]

        x_train = x[:, idx_train]
        x_val = x[:, idx_test]

        y_train = y[idx_train]
        y_val = y[idx_test]

        return (x_train, y_train), (x_val, y_val)

    @staticmethod
    def split_data(lines, n) -> tuple[list[Any], list[Any]]:
        n_lines: int = len(lines)
        x_train, x_eval = [], []
        for i in range(n_lines):
            if i % n == 0:
                x_eval.append(lines[i])
            else:
                x_train.append(lines[i])

        return x_train, x_eval

    @staticmethod
    def remove_features(x: np.ndarray, features: list[int]) -> np.ndarray:
        return np.delete(x, features, axis=0)
