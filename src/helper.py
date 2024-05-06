import numpy as np
from matplotlib import pyplot as plt
import logging


def load_iris():
    import sklearn.datasets

    return (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )


def load_txt(
    file_path: str,
    delimiter: str = ",",
    features_type: type = float,
    return_labels_in_data_matrix: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    # Load the array from the text file and convert all but last column to float
    data = np.loadtxt(file_path, delimiter=delimiter, dtype=str)

    if return_labels_in_data_matrix:
        x = data
    else:
        # Extract features (all but last column)
        x = data[:, :-1].astype(features_type)

    # Extract labels (last column)
    y = data[:, -1]

    return x, y


def plot_matrix(
    matrix: np.ndarray,
    labels: np.ndarray,
    labels_name: dict  = None,
    *,
    title: str,
    x_label: str,
    y_label: str,
    grid: bool = True,
    invert_yaxis: bool = False,
) -> None:

    plt.figure()

    unique_labels = np.unique(labels)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    if labels_name and len(labels_name) == len(unique_labels):
        unique_labels = labels_name

    for label, color in zip(unique_labels, colors):
        indices = np.where(labels == label)

        plt.scatter(matrix[0, indices], matrix[1, indices], color=color, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend()

    plt.grid(grid)

    if invert_yaxis:
        plt.gca().invert_yaxis()

    plt.show()


def v_col(x: np.ndarray) -> np.ndarray:
    return x.reshape((x.size, 1))


def v_row(x: np.ndarray) -> np.ndarray:
    return x.reshape((1, x.size))


def center_matrix(matrix: np.ndarray) -> np.ndarray:
    # compute mean over the rows
    mu = matrix.mean(1).reshape(-1, 1)

    logging.debug(f"Mean: \n{mu}")

    return matrix - mu


def mean_vector(matrix: np.ndarray) -> np.ndarray:
    # compute mean over the rows
    return matrix.mean(0)


def check_solution() -> None:
    raise NotImplementedError


def cv_matrix(matrix: np.ndarray) -> np.ndarray:
    centered_matrix: np.ndarray = center_matrix(matrix)

    return np.dot(centered_matrix, centered_matrix.T) / float(matrix.shape[0])


def main():
    raise NotImplementedError


if __name__ == "__main__":
    # Configure logging level
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    main()
