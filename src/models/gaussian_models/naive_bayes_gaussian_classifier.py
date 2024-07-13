from typing import Optional
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

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "NaiveBayesBaseGaussianModel":
        super().fit(X, y)  # Utilize the fit method from BaseModel to set classes
        self.X = X
        self.y = y
        for c in self.classes:
            # Filter data for each class and fit a MultivariateGaussianModel
            class_data = X[:, y == c]
            mu_, cov_ = self.utils.compute_mu_and_sigma(class_data)
            self.h_params[c] = {"mean_": mu_, "sigma_": cov_ * np.eye(len(mu_))}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().compute_class_posteriors(X, self.h_params)
