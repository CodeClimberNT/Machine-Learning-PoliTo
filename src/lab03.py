import logging

import numpy as np

from helper import load_iris, cv_matrix, plot_matrix, v_col


def projection_matrix(D: np.ndarray, m: int) -> np.ndarray:
    C = cv_matrix(D)

    logging.debug(f"Covariance matrix shape: {C.shape}")
    logging.debug(f"Covariance matrix:\n {C}")

    # Compute PCA using Singular Value Decomposition
    U, _, _ = np.linalg.svd(C)

    # Select the leading m eigenvectors
    P = U[:, 0:m]

    return np.dot(P.T, D)


def pca() -> None:
    D, L = load_iris()

    logging.debug(f"matrix shape {D.shape}")

    m2 = projection_matrix(D, 2)

    logging.info(f"Projected matrix:\n {m2[:5]}")

    logging.info(f"Projected matrix shape {m2.shape}")

    plot_matrix(
        m2,
        L,
        labels_list=("Setosa", "Versicolour", "Virginica"),
        title="PCA Projection of Iris Dataset",
        x_label="Principal Component 1",
        y_label="Principal Component 2",
        invert_yaxis=True,
    )


# Function for the LDA projection matrix


def compute_Sb(D, L):
    Sb = 0

    mu = v_col(np.mean(D, axis=1))

    for c in np.unique(L):
        Dc = D[:, L == c]
        mu_c = v_col(np.mean(Dc, axis=1))
        Sb += np.dot((mu_c - mu), (mu_c - mu).T) * Dc.shape[1]

    return Sb / D.shape[1]


def compute_Sw(D, L):
    Sw = 0

    for c in np.unique(L):
        Dc = D[:, L == c]
        mu_c = v_col(np.mean(Dc, axis=1))
        Sw += np.dot((Dc - mu_c), (Dc - mu_c).T)

    return Sw / D.shape[1]


def lda() -> None:
    D, L = load_iris()
    logging.debug(f"matrix shape {D.shape}")

    logging.debug(f"Labels: {L}")

    Sb = compute_Sb(D, L)
    Sw = compute_Sw(D, L)

    logging.info(f"Between-class covariance matrix:\n {Sb}")
    logging.info(f"Within-class covariance matrix:\n {Sw}")


def main() -> None:
    lda()


if __name__ == "__main__":
    # Configure logging level
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    main()
