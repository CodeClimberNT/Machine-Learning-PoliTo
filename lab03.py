import numpy as np


from matplotlib import pyplot as plt


# from Helpers.helper import load_file


def check_solution() -> None:
    raise NotImplementedError


def load_file(
    file_path: str, return_labels_in_data_matrix: bool = False
) -> tuple[np.ndarray, np.ndarray]:

    # Load the array from the text file and convert all but last column to float

    data = np.loadtxt(file_path, delimiter=",", dtype=str)

    if return_labels_in_data_matrix:
        return data

    # Extract features (first 4 columns)

    x = data[:, :-1].astype(float)

    # Extract labels (last column)

    y = data[:, -1]

    return x, y


def center_matrix(matrix: np.ndarray) -> np.ndarray:

    # compute mean over the rows

    mu = matrix.mean(0)

    print(f"Mean: \n{mu}")

    return matrix - mu


def covariance_matrix(matrix: np.ndarray) -> np.ndarray:

    centered_matrix: np.ndarray = center_matrix(matrix)

    return np.dot(centered_matrix.T, centered_matrix) / float(matrix.shape[0])


def projection_matrix(matrix: np.ndarray, m: int) -> np.ndarray:

    cm = covariance_matrix(matrix)

    print(f"Covariance matrix shape: {cm.shape}")

    print(f"Covariance matrix:\n {cm}")

    # Compute PCA using Singular Value Decomposition

    U, _, _ = np.linalg.svd(cm)

    # Select the leading m eigenvectors

    P = U[:, :m]

    return np.dot(matrix, P)


def between_class_covariance(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:

    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        swc = center_matrix(matrix[indices, :])
        
    return matrix


def plot_pca(projection: np.ndarray, labels: np.ndarray) -> None:

    plt.figure()

    unique_labels = np.unique(labels)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):

        indices = np.where(labels == label)
        plt.scatter(
            projection[indices, 0], projection[indices, 1], color=color, label=label
        )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("PCA Projection of Iris Dataset")

    plt.legend()

    plt.grid(True)

    plt.gca().invert_yaxis()

    plt.show()


def pca() -> None:
    x, y = load_file("datasets/iris.csv")

    print(f"matrix shape {x.shape}")

    m2 = projection_matrix(x, 2)

    # print(f"Projected matrix:\n {m2[:5]}")
    # print(f"Projected matrix shape {m2.shape}")

    plot_pca(m2, y)


def lda() -> None:

    x, y = load_file("datasets/iris.csv")

    # labels_map: dict[str, int] = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    xs = separate_data_by_label(x, y)
    print(f"shape: {xs[1]}")


def main() -> None:
    lda()


if __name__ == "__main__":
    main()
