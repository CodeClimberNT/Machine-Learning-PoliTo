from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, TypeVar, Generic

T = TypeVar("T", bound="BaseModel")


class BaseModel(ABC, Generic[T]):
    """
    An abstract base class for models, providing a template for fitting models to data,
    making predictions, and calculating accuracy.
    """

    def __init__(self):
        """
        Initialize the BaseModel with an empty list of classes.
        """
        self.classes: np.ndarray = np.array([])
        self.num_classes = 0
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> "BaseModel[T]":
        """
        Fit the model to the data. This method calculates and stores the unique classes
        from the `y` parameter, if provided.

        :param X: Feature dataset as a NumPy array.
        :param y: Labels as a NumPy array. Optional, default is None.
        """
        self.X = X
        if y is not None:
            self.y = y
            self.classes = np.unique(y)
            self.num_classes: int = len(self.classes)
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for a given dataset. This method should be implemented by subclasses
        to provide the logic for making predictions based on the model.

        :param X: Feature dataset as a NumPy array.
        :return: Predicted labels as a NumPy array.
        """
        pass

    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy of the model on a given dataset. This method uses the predict method
        to generate predictions and then compares them to the true labels to calculate accuracy.

        :param X: Feature dataset as a NumPy array.
        :param y: True labels as a NumPy array.
        :return: Accuracy score as a float.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
