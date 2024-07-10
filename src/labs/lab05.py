import numpy as np

from src.helpers import DatasetImporterHelper as dih, DataHandler as dh
from src.models.gaussian_models import GaussianModel as MVG, NaiveBayesGaussianModel as NBG, \
    TiedCovarianceGaussianModel as TCG


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
    post_prob_computed: np.ndarray = mvg.compute_log_posterior(x_val)

    # Load the precomputed SJoint_MVG.npy
    post_prob_loaded: np.ndarray = np.load('../../labs/lab05/solution/logPosterior_MVG.npy')

    # Compare the computed SJoint with the loaded SJoint
    max_abs_error = np.abs(post_prob_computed - post_prob_loaded).max()

    # Display comparison results
    print(f"Max absolute error w.r.t. solution logPosterior_MVG.npy: {max_abs_error}")


def analyze_class_posterior_probabilities():
    # Load iris dataset
    x, y = dih.load_iris()

    # Split the dataset into training and validation sets
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(x, y)

    # Fit a Naive Bayes Gaussian Model on the training data
    nbg_model = NBG(x_train, y_train).fit(x_train, y_train)

    # Predict class posterior probabilities for the validation set
    # Note: The Naive Bayes model in the provided context does not directly support posterior probability prediction.
    # We will simulate this by using the predict_proba_one method from the GaussianClassifier base class,
    # which requires modification to return probabilities for all classes.
    # This step assumes such a method exists or has been implemented.
    posterior_probs = np.array([nbg_model.predict_proba_one(x_val[i], c)
                                for i in range(x_val.shape[0])
                                for c in range(3)]).reshape(x_val.shape[0], 3)

    # Display the posterior probabilities for each class in the validation set
    for i in range(posterior_probs.shape[0]):
        print(f"Sample {i + 1} posterior probabilities:")
        for c in range(3):
            print(f"Class {c}: {posterior_probs[i, c]:.4f}")
        print("----------")


def main():
    # analyze_iris()
    display_and_compare_SJoint()


if __name__ == "__main__":
    main()
