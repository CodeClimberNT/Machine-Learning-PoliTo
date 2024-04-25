import numpy as np

# from matplotlib import pyplot as plt


def load_file(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    # Load the array from the text file
    data = np.loadtxt(file_path, delimiter=",", dtype=str)

    # Extract features (first 4 columns)
    x = data[:, :-1].astype(float)

    # Extract labels (last column)
    y = data[:, -1]

    return x, y


def v_col(x):
    return x.reshape((x.size, 1))


def v_row(x):
    return x.reshape((1, x.size))
