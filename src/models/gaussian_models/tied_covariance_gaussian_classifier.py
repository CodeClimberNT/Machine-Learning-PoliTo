import numpy as np
import scipy

from src.helpers import MathHelper as mh
from src.models.gaussian_models.base_gaussian_model import BaseGaussianModel


class TiedCovarianceBaseGaussianModel(BaseGaussianModel):
    def __init__(self):
        super().__init__()
        self.class_models: dict = {}
        self.shared_sigma_ = 0
        self.h_params = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)

        for c in self.classes:
            class_data = X[:, y == c]
            class_labels = y[y == c]
            mu_, cov_class = self.utils.compute_mu_and_sigma(class_data)
            self.shared_sigma_ += cov_class * class_data.shape[1]
            self.h_params[c] = {"mean_": mu_, "sigma_": cov_class}

        self.shared_sigma_ /= X.shape[1]
        for c in self.classes:
            self.h_params[c]["sigma_"] = self.shared_sigma_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            probs = [self.predict_proba_one(x, c) for c in self.classes]
            predictions.append(self.classes[np.argmax(probs)])
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
