from typing import Optional
import numpy as np

from src.models.gaussian_models.base_gaussian_model import BaseGaussianModel


class TiedCovarianceBaseGaussianModel(BaseGaussianModel):
    def __init__(self):
        super().__init__()
        self.shared_sigma_ = 0
        self.h_params = {}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]):
        super().fit(X, y)

        for c in self.classes:
            class_data = X[:, y == c]
            mu_, cov_class = self.utils.compute_mu_and_sigma(class_data)
            self.shared_sigma_ += cov_class * class_data.shape[1]
            self.h_params[c] = {"mean_": mu_, "sigma_": cov_class}

        self.shared_sigma_ /= X.shape[1]
        for c in self.classes:
            self.h_params[c]["sigma_"] = self.shared_sigma_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().compute_class_posteriors(X, self.h_params)
