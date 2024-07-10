import numpy as np
from src.models.base_model import BaseModel
from src.models.gaussian_models.gaussian_model import GaussianModel


class GaussianClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GaussianModel()
        self.class_models = {}
        self.X = None
        self.y = None

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba_one(self, x: np.ndarray, c: int) -> float:
        if c in self.class_models:
            mvg = self.class_models[c]
            log_prob = mvg.logpdf_GAU_ND(x.reshape(1, -1))
            prior_log_prob = np.log(len(self.y[self.y == c]) / len(self.y))
            return np.exp(log_prob + prior_log_prob)
        else:
            print(f"Class {c} not found in class_models")
            return 0.0
