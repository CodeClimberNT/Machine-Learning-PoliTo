import numpy as np
from src.models.base_model import BaseModel
from src.models.gaussian_models.multivariate_gaussian_model import MultivariateGaussianModel
from src.models.gaussian_models.gaussian_utils import GaussianUtils


class BaseGaussianModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = MultivariateGaussianModel()
        self.utils = GaussianUtils()
        self.h_params = {}
        self.X = None
        self.y = None

    def fit(self, X, y):
        super().fit(X, y)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba_one(self, x: np.ndarray, c: int) -> float:
        if c in self.h_params:
            mvg = self.h_params[c]
            log_prob = mvg.logpdf_GAU_ND(x.reshape(1, -1))
            prior_log_prob = np.log(len(self.y[self.y == c]) / len(self.y))
            return np.exp(log_prob + prior_log_prob)
        else:
            print(f"Class {c} not found in class_models")
            return 0.0

    def compute_log_posterior(self, X: np.ndarray) -> np.ndarray:
        return self.model.compute_log_posterior(X)

    @staticmethod
    def compute_SJoint(log_likelihood, prior_prob) -> np.ndarray:
        return MultivariateGaussianModel.compute_SJoint(log_likelihood, prior_prob)
