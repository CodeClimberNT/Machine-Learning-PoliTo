import numpy as np
import os

from src.helpers import DatasetImporterHelper as dih, DataHandler as dh
from src.models.gaussian_models import (
    MultivariateGaussianModel as MVG,
    NaiveBayesBaseGaussianModel as NBG,
    TiedCovarianceBaseGaussianModel as TCG,
)

current_file_path: str = os.path.abspath(__file__)
project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
SOLUTION_FOLDER: str = os.path.join(project_root, "labs", "lab05", "solution")


def analyze_iris():
    # Load iris dataset
    x, y = dih.load_iris()

    # Split the dataset into training and validation sets
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(x, y)

    # Analyze each class using the MVG class
    for i in range(3):
        mvg_model = MVG().fit(x_train[:, y_train == i])
        print(f"Class {i} mu:\n{mvg_model.mu_}")
        print(f"Class {i} cv:\n{mvg_model.sigma_}")


def display_and_compare_SJoint():
    # Load iris dataset
    x, y = dih.load_iris()

    # Split the dataset into training and validation sets
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(x, y)

    # Fit a Gaussian Model on the entire training data
    mvg: MVG = MVG().fit(x_train, y_train)

    # Compute SJoint for the validation set
    post_prob_computed: np.ndarray = mvg.compute_log_posteriors(x_val)

    # Load the precomputed SJoint_MVG.npy
    solution_file_path = os.path.join(SOLUTION_FOLDER, "logPosterior_MVG.npy")
    post_prob_loaded: np.ndarray = np.load(solution_file_path)

    # Compare the computed SJoint with the loaded SJoint
    max_abs_error = np.abs(post_prob_computed - post_prob_loaded).max()

    # Display comparison results
    print(f"Max absolute error w.r.t. solution logPosterior_MVG.npy: {max_abs_error}")

    predict_labels = post_prob_computed.argmax(0)
    # print(predict_labels)
    print(f"Error rate: {(np.sum(predict_labels != y_val) / y_val.size) * 100}")


def naive_analysis():
    # Load iris dataset
    x, y = dih.load_iris()

    # Split the dataset into training and validation sets
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(x, y)

    # Analyze each class using the NaiveBayesGaussianModel class
    nbg = NBG().fit(x_train, y_train)

    # for c in nbg.classes:
    #     print(f"Naive Bayes Gaussian - Class {c}")
    #     print(f"mu:\n{nbg.h_params[c]['mean_']}")
    #     print(f"cv:\n{nbg.h_params[c]['sigma_']}")
    #     print()

    post_prob_computed = nbg.compute_log_posteriors(x_val)

    # Load the precomputed SJoint_MVG.npy
    solution_file_path = os.path.join(SOLUTION_FOLDER, "logPosterior_NaiveBayes.npy")
    post_prob_loaded: np.ndarray = np.load(solution_file_path)

    # Compare the computed SJoint with the loaded SJoint
    max_abs_error = np.abs(post_prob_computed - post_prob_loaded).max()

    # Display comparison results
    print(
        f"Max absolute error w.r.t. solution logPosterior_NaiveBayes.npy: {max_abs_error}"
    )

    predict_labels = post_prob_computed.argmax(0)
    print(
        f"Naive Bayes Gaussian - Error rate: {(np.sum(predict_labels != y_val) / y_val.size) * 100}"
    )


def tied_analysis():
    # Load iris dataset
    x, y = dih.load_iris()

    # Split the dataset into training and validation sets
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(x, y)

    # Analyze each class using the NaiveBayesGaussianModel class
    tied_model = TCG().fit(x_train, y_train)

    # for c in tied_model.classes:
    #     print(f"Naive Bayes Gaussian - Class {c}")
    #     print(f"mu:\n{tied_model.h_params[c]['mean_']}")
    #     print(f"cv:\n{tied_model.h_params[c]['sigma_']}")
    #     print()

    post_prob_computed = tied_model.compute_log_posteriors(x_val)

    # Load the precomputed SJoint_MVG.npy
    solution_file_path = os.path.join(SOLUTION_FOLDER, "logPosterior_TiedMVG.npy")
    post_prob_loaded: np.ndarray = np.load(solution_file_path)

    # Compare the computed SJoint with the loaded SJoint
    max_abs_error = np.abs(post_prob_computed - post_prob_loaded).max()

    # Display comparison results
    print(
        f"Max absolute error w.r.t. solution logPosterior_TiedMVG.npy: {max_abs_error}"
    )

    predict_labels = post_prob_computed.argmax(0)
    print(
        f"Naive Bayes Gaussian - Error rate: {(np.sum(predict_labels != y_val) / y_val.size) * 100}"
    )


def binary_task():
    x, y = dih.load_iris()
    x = x[:, y != 0]
    y = y[y != 0]
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(x, y)

    # Initialize MVG models for each class
    mvg_model = MVG().fit(x_train, y_train)

    # Compute the log likelihood ratio
    llr = mvg_model.log_likelihood_ratio(x_val)

    predicted_val = np.zeros(x_val.shape[1], dtype=np.int32)
    threshold = 0
    predicted_val[llr >= threshold] = 2
    predicted_val[llr < threshold] = 1
    print(f"MVG error rate: {(np.sum(predicted_val != y_val) / y_val.size) * 100}%")
    # Load solution and compare
    solution_file_path = os.path.join(SOLUTION_FOLDER, "llr_MVG.npy")
    solution_llrs: np.ndarray = np.load(solution_file_path)
    difference = np.abs(llr - solution_llrs).max()
    print(f"LLRs differs from the solution: {difference}")


def main():
    print(SOLUTION_FOLDER)
    analyze_iris()
    display_and_compare_SJoint()
    naive_analysis()
    tied_analysis()
    binary_task()


if __name__ == "__main__":
    main()
