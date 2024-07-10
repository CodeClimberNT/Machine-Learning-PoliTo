import numpy as np

from src.models.gaussian_models.gaussian_model import GaussianModel
from src.models.gaussian_models.gaussian_classifier import GaussianClassifier


class TiedCovarianceGaussianModel(GaussianClassifier):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X, y)
        self.class_models: dict = {}
        self.shared_sigma_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        self.shared_sigma_ = self.model.sigma_

        for c in self.classes:
            class_data = X[y == c]
            class_labels = y[y == c]
            mvg = GaussianModel()
            mvg.fit(class_data, class_labels)
            mvg.sigma_ = self.shared_sigma_  # Set the shared covariance matrix
            mvg.inv_sigma_ = np.linalg.inv(self.shared_sigma_)  # Update the inverse covariance matrix
            self.class_models[c] = mvg
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            probs = [self.predict_proba_one(x, c) for c in self.classes]
            predictions.append(self.classes[np.argmax(probs)])
        return np.array(predictions)

    def predict_proba_one(self, x: np.ndarray, c: int) -> float:
        return super().predict_proba_one(x, c)
