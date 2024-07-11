import numpy as np
import scipy

from src.models.gaussian_models.base_gaussian_model import BaseGaussianModel
from src.helpers import MathHelper as mh


class NaiveBayesBaseGaussianModel(BaseGaussianModel):
    def __init__(self):
        super().__init__()
        self.h_params = {}
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'NaiveBayesBaseGaussianModel':
        super().fit(X, y)  # Utilize the fit method from BaseModel to set classes
        self.X = X
        self.y = y
        for c in self.classes:
            # Filter data for each class and fit a MultivariateGaussianModel
            class_data = X[:, y == c]
            class_labels = y[y == c]
            mu_, cov_ = self.utils.compute_mu_and_sigma(class_data)
            self.h_params[c] = {"mean_": mu_, "sigma_": cov_ * np.eye(len(mu_))}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        print(X.shape)
        for x in X.T:  # Iterate over each sample
            probabilities = []
            for c in self.classes:  # Iterate over each class
                prob = self.predict_proba_one(x, c)
                probabilities.append(prob)
            predicted_class = self.classes[np.argmax(probabilities)]  # Class with the highest probability
            predictions.append(predicted_class)
        return np.array(predictions)

    def compute_log_posterior(self, X: np.ndarray) -> np.ndarray:
        log_likelihood = self.compute_log_likelihood(X)
        S_joint = super().compute_SJoint(log_likelihood, np.ones(self.num_classes) / float(self.num_classes))
        S_marginal = mh.v_row(scipy.special.logsumexp(S_joint, axis=0))

        return S_joint - S_marginal

    def compute_log_likelihood(self, X_val: np.ndarray) -> np.ndarray:
        log_likelihood = np.zeros((self.num_classes, X_val.shape[1]))
        for c in self.classes:
            log_likelihood[c, :] = self.utils.calculate_probability_distribution(X_val, self.h_params[c]['mean_'],
                                                                                 self.h_params[c]['sigma_'])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Calculate the accuracy of the model
        return np.mean(self.predict(X) == y)
