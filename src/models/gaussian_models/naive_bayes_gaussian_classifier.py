import numpy as np

from src.models.gaussian_models import GaussianModel
from src.models.gaussian_models.gaussian_classifier import GaussianClassifier


class NaiveBayesGaussianModel(GaussianClassifier):
    def __init__(self):
        super().__init__()
        self.class_models = {}
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'NaiveBayesGaussianModel':
        super().fit(X, y)  # Utilize the fit method from BaseModel to set classes
        self.X = X
        self.y = y
        for c in self.classes:
            # Filter data for each class and fit a MultivariateGaussianModel
            class_data = X[:, y == c]
            class_labels = y[y == c]
            mvg = GaussianModel()
            mvg.fit(class_data, class_labels)
            self.class_models[c] = mvg
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

    def predict_vectorized(self, X: np.ndarray) -> np.ndarray:
        # Ensure X is 2D
        X = X.reshape(X.shape[0], -1).T  # Transpose to have samples as rows if not already

        # Initialize an array to hold probabilities for each class for each sample
        probabilities = np.zeros((X.shape[0], len(self.classes)))

        # Compute probabilities for each class in a vectorized manner
        for idx, c in enumerate(self.classes):
            class_model = self.class_models[c]
            # Assuming GaussianModel or a similar class has a method to compute log probabilities for multiple samples
            log_probs = class_model.logpdf_GAU_ND(X)  # This needs to be implemented in GaussianModel
            prior_log_prob = np.log(len(self.y[self.y == c]) / len(self.y))
            probabilities[:, idx] = np.exp(log_probs + prior_log_prob)

        # Find the class with the highest probability for each sample
        predicted_classes_indices = np.argmax(probabilities, axis=1)
        predicted_classes = np.array(self.classes)[predicted_classes_indices]

        return predicted_classes

    def predict_proba_one(self, x: np.ndarray, c: int) -> float:
        return super().predict_proba_one(x, c)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Calculate the accuracy of the model
        return np.mean(self.predict(X) == y)
