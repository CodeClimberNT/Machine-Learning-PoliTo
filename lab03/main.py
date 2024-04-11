import numpy as np

from ml.utils import *
from matplotlib import pyplot as plt

def check_solution() -> None:
    raise NotImplementedError


def compute_matrix_centered(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=1).reshape((matrix.shape[0], 1))


def covariance_matrix(matrix: np.ndarray) -> float:
    # mu = matrix.mean(axis=1)
    #
    # matrix_centered: np.ndarray = matrix - mu.reshape((mu.size, 1))

    matrix_centered: np.ndarray = compute_matrix_centered(matrix)

    return np.dot(matrix_centered, matrix_centered.T) / float(matrix.shape[1])


def compute_projection_matrix(matrix: np.ndarray, m: int) -> np.ndarray:

    u, s, vh = np.linalg.svd(covariance_matrix(matrix))
    print("eigenvalues ", u.shape)
    print("s ", s.shape)
    print("vh ", vh.shape)
    p = u[:, 0:m]
    print("column eig ", p.shape)

    return np.dot(p.T, matrix)


def plot_pca(x: np.ndarray, y: np.ndarray, features: list[str]) -> None:
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], alpha=0.5)
    plt.legend()
    plt.show()


def main() -> None:
    [x, y] = load_file("../datasets/iris.csv", (0, 1, 2, 3))
    features_name: str = ('Sepal length', 'Sepal width', 'Petal length', 'Petal width')
    print("matrix shape", x.shape)

    m2 = compute_projection_matrix(x, 2)
    print(m2.shape)
    plot_pca(m2, y, features_name)


if __name__ == '__main__':
    main()
