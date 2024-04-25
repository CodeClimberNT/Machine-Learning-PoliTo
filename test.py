import numpy as np
import matplotlib.pyplot as plt
from helper import load_txt


# Function to compute PCA projection matrix
def compute_pca_projection_matrix(data, m):
    # Step 1: Compute the covariance matrix
    C = np.cov(data.T)

    # Step 2: Compute the eigenvectors and eigenvalues
    _, U = np.linalg.eigh(C)

    # Step 3: Select the top m eigenvectors
    P = U[:, ::-1][:, :m]

    return P


# Function to compute LDA transformation matrix using generalized eigenvalue problem
def compute_lda_transformation_matrix_gen(SB, SW, m):
    # Solve the generalized eigenvalue problem
    s, U = np.linalg.eigh(SB, SW)

    # Select the top m eigenvectors
    W = U[:, ::-1][:, :m]

    return W


# Function to compute LDA transformation matrix using joint diagonalization
def compute_lda_transformation_matrix_joint(SB, SW, m):
    # Whitening the within-class covariance matrix
    U, s, _ = np.linalg.svd(SW)
    s_inv_sqrt = 1.0 / np.sqrt(s)
    P1 = np.dot(U * s_inv_sqrt, U.T)

    # Transforming between-class covariance matrix
    SBT = np.dot(np.dot(P1, SB), P1.T)

    # Computing the matrix of eigenvectors of SBT
    _, V = np.linalg.eigh(SBT)

    # Selecting the top m eigenvectors
    P2 = V[:, ::-1][:, :m]

    # LDA transformation matrix
    W = np.dot(P1, P2)

    return W


# Function to plot the results
def plot_results(data, labels, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    label_set = np.unique(labels)
    colors = ["r", "g", "b"][: len(label_set)]
    for idx, label in enumerate(label_set):
        plt.scatter(
            data[labels == label, 0],
            data[labels == label, 1],
            label=label,
            color=colors[idx],
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


# Here, you need to compute SB and SW using the provided formulas
# Function to compute the mean vector for each class
def compute_class_means(data, labels):
    unique_labels = np.unique(labels)
    class_means = {}
    for label in unique_labels:
        class_data = data[labels == label]
        class_mean = np.mean(class_data, axis=0)
        class_means[label] = class_mean
    return class_means


# Function to compute the within-class scatter matrix (SW)
def compute_within_class_scatter_matrix(data, labels):
    class_means = compute_class_means(data, labels)
    unique_labels = np.unique(labels)
    SW = np.zeros((data.shape[1], data.shape[1]))
    for label in unique_labels:
        class_data = data[labels == label]
        class_mean = class_means[label]
        diff = class_data - class_mean
        SW += np.dot(diff.T, diff)
    return SW


# Function to compute the between-class scatter matrix (SB)
def compute_between_class_scatter_matrix(data, labels):
    class_means = compute_class_means(data, labels)
    overall_mean = np.mean(data, axis=0)
    unique_labels = np.unique(labels)
    SB = np.zeros((data.shape[1], data.shape[1]))
    for label in unique_labels:
        class_mean = class_means[label]
        n = np.sum(labels == label)
        diff = (class_mean - overall_mean).reshape(-1, 1)
        SB += n * np.dot(diff, diff.T)
    return SB


# Load the Iris dataset
file_path = "datasets/iris.csv"
data, labels = load_txt("datasets/iris.csv")

# Compute the within-class and between-class scatter matrices
SW = compute_within_class_scatter_matrix(data, labels)
SB = compute_between_class_scatter_matrix(data, labels)

# PCA projection matrix
P_pca = compute_pca_projection_matrix(data, 2)

# LDA transformation matrix using generalized eigenvalue problem
_, U_SB = np.linalg.eigh(SB)
_, U_SW = np.linalg.eigh(SW)
W_lda_gen = np.dot(U_SW.T, U_SB)[:, ::-1][:, :2]

# LDA transformation matrix using joint diagonalization
W_lda_joint = compute_lda_transformation_matrix_joint(SB, SW, 2)

# Projecting data onto PCA space
data_pca = np.dot(data, P_pca)

# Projecting data onto LDA space (generalized eigenvalue problem)
data_lda_gen = np.dot(data, W_lda_gen)

# Projecting data onto LDA space (joint diagonalization)
data_lda_joint = np.dot(data, W_lda_joint)

# Plotting results
plot_results(
    data_pca, labels, "PCA Projection", "Principal Component 1", "Principal Component 2"
)
plot_results(
    data_lda_gen,
    labels,
    "LDA Projection (Generalized Eigenvalue Problem)",
    "Linear Discriminant 1",
    "Linear Discriminant 2",
)
plot_results(
    data_lda_joint,
    labels,
    "LDA Projection (Joint Diagonalization)",
    "Linear Discriminant 1",
    "Linear Discriminant 2",
)
