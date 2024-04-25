import numpy as np


from helper import load_txt, center_matrix, cv_matrix, plot_matrix


def projection_matrix(matrix: np.ndarray, m: int) -> np.ndarray:

    cm = cv_matrix(matrix)

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


def pca() -> None:

    x, y = load_txt("datasets/iris.csv")

    print(f"matrix shape {x.shape}")

    m2 = projection_matrix(x, 2)

    # print(f"Projected matrix:\n {m2[:5]}")

    # print(f"Projected matrix shape {m2.shape}")

    plot_matrix(
        m2,
        y,
        title="PCA Projection of Iris Dataset",
        xlabel="Principal Component 1",
        ylabel="Principal Component 2",
        invert_yaxis=True,
    )


def lda() -> None:

    x, y = load_txt("datasets/iris.csv")


def main() -> None:
    pca()


if __name__ == "__main__":
    main()
