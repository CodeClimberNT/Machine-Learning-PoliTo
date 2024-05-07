import logging

import numpy as np
import scipy
from helper import MathHelper as mh
from helper import Data as d


import LinearDiscriminantAnalysis as LDA
import Visualizer as vis


def projection_matrix(D: np.ndarray, m: int) -> np.ndarray:

    C = mh.cv_matrix(D)

    # Compute PCA using Singular Value Decomposition

    U, _, _ = np.linalg.svd(C)

    # Select the leading m eigenvectors

    P = U[:, 0:m]

    return np.dot(P.T, D)


"""
def pca() -> None:
    D, L = d.load_iris()

    m2 = mh.projection_matrix(D, 2)
"""


def compute_pca(D: np.ndarray, *, m: int) -> np.ndarray:

    mu = mh.v_col(np.mean(D, axis=1))

    C = np.dot((D - mu), (D - mu).T) / float(D.shape[1])
    U, _, _ = np.linalg.svd(C)
    P = U[:, 0:m]

    return P


def apply_pca(P: np.ndarray, D: np.ndarray) -> np.ndarray:

    return np.dot(P.T, D)


# Function for the LDA projection matrix


def compute_Sb(D, L):

    Sb = 0

    mu = mh.v_col(np.mean(D, axis=1))

    for c in np.unique(L):

        Dc = D[:, L == c]

        mu_c = mh.v_col(np.mean(Dc, axis=1))

        Sb += np.dot((mu_c - mu), (mu_c - mu).T) * Dc.shape[1]

    return Sb / D.shape[1]


def compute_Sw(D, L):
    Sw = 0

    for c in np.unique(L):
        Dc = D[:, L == c]
        mu_c = mh.v_col(np.mean(Dc, axis=1))
        Sw += np.dot((Dc - mu_c), (Dc - mu_c).T)
    return Sw / D.shape[1]


def lda_eigh(D, L, *, m):

    Sb = compute_Sb(D, L)

    Sw = compute_Sw(D, L)

    s, U = scipy.linalg.eigh(Sb, Sw)

    return U[:, ::-1][:, 0:m]


def lda_joint_diagonalization(D, L, m):

    Sw = compute_Sw(D, L)

    U, s, _ = np.linalg.svd(Sw)

    P1 = np.dot(U * mh.v_row(1.0 / (s**0.5)), U.T)

    Sb = compute_Sb(D, L)

    Sbt = np.dot(P1, np.dot(Sb, P1.T))

    U2, s2, _ = np.linalg.svd(Sbt)

    P2 = U2[:, 0:m]

    return np.dot(P2.T, P1).T


def lda() -> np.ndarray:
    D, L = d.load_iris()

    logging.debug(f"matrix shape {D.shape}")

    U = lda_eigh(D, L, m=2)

    print(U)


def apply_lda(U: np.ndarray, D: np.ndarray) -> np.ndarray:

    return U.T @ D


# CLASSIFICATION


def split_db_2to1(D: np.ndarray, L: np.ndarray, seed=0) -> tuple:

    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)

    idx = np.random.permutation(D.shape[1])

    idxTrain = idx[0:nTrain]

    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]

    DVAL = D[:, idxTest]

    LTR = L[idxTrain]

    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


def classification_no_preprocess() -> None:

    DIris, LIris = d.load_iris()

    D = DIris[:, LIris != 0]

    L = LIris[LIris != 0]

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    ULDA = lda_joint_diagonalization(DTR, LTR, m=1)

    DTR_lda = apply_lda(ULDA, DTR)

    if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 2].mean():

        ULDA = -ULDA

        DTR_lda = apply_lda(ULDA, DTR)

    DVAL_lda = apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2.0

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)

    PVAL[DVAL_lda[0] >= threshold] = 2

    PVAL[DVAL_lda[0] < threshold] = 1

    print("Labels:     ", LVAL)

    print("Predictions:", PVAL)
    print(
        "Number of errors:", (PVAL != LVAL).sum(), "(out of %d samples)" % (LVAL.size)
    )

    print("Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))


def classification() -> None:

    DIris, LIris = d.load_iris()

    D = DIris[:, LIris != 0]

    L = LIris[LIris != 0]

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    UPCA = compute_pca(DTR, m=2)

    DTR_pca = apply_pca(UPCA, DTR)

    DVAL_pca = apply_pca(UPCA, DVAL)

    ULDA = lda_joint_diagonalization(DTR_pca, LTR, m=1)

    DTR_lda = apply_lda(ULDA, DTR_pca)

    if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 2].mean():

        ULDA = -ULDA

        DTR_lda = apply_lda(ULDA, DTR_pca)

    DVAL_lda = apply_lda(ULDA, DVAL_pca)

    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2.0

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)

    PVAL[DVAL_lda[0] >= threshold] = 2

    PVAL[DVAL_lda[0] < threshold] = 1

    print("Labels:     ", LVAL)

    print("Predictions:", PVAL)
    print(
        "Number of errors:", (PVAL != LVAL).sum(), "(out of %d samples)" % (LVAL.size)
    )

    print("Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

    labels_name = {1: "Versicolor", 2: "Virginica"}

    vis.Visualizer.plot_multiple_hist(
        DVAL_lda,
        LVAL,
        labels_name,
        title="LDA Projection of Iris Dataset",
        x_label="LDA Projection",
        y_label="Frequency",
    )


def test_lda():
    D, L = d.load_iris()
    lda = LDA.LDA(solver="svd", m=2)
    lda.set_train_data(D, L)
    lda.fit()
    y = lda.get_projected_matrix()
    print(y)
    vis.Visualizer.plot_scatter_matrix(
        y,
        L,
        labels_name={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
        title="LDA Projection of Iris Dataset",
        x_label="LDA Projection 1",
        y_label="LDA Projection 2",
        invert_y_axis=True,
        invert_x_axis=True,
    )


def main() -> None:
    test_lda()
    # test_pca()
    # classification_no_preprocess()

    # classification()


if __name__ == "__main__":

    # Configure logging level

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    main()
