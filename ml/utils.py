import numpy as np
from matplotlib import pyplot as plt


def load_file(file_path: str, features_cols: tuple) -> tuple[np.ndarray, np.ndarray]:
    # Load the array from the text file
    x = np.loadtxt(file_path, delimiter=",", usecols=features_cols, dtype=float)
    y = np.loadtxt(file_path, delimiter=",", usecols=features_cols[-1] + 1, dtype=str)
    # Return the array
    return x, y


def v_col(x):
    return x.reshape((x.size, 1))


def v_row(x):
    return x.reshape((1, x.size))
