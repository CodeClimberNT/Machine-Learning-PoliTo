from typing import Optional
import numpy as np
from src.models.base_model import BaseModel
from src.models.gaussian_models.gaussian_utils import GaussianUtils


class BaseGaussianModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.utils = GaussianUtils()
        self.h_params = {}
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]):
        super().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Method predict not implemented")

    def compute_log_likelihood(self, X_val: np.ndarray) -> np.ndarray:
        return self.utils.compute_log_likelihood(X_val, self.h_params)

    def compute_log_posteriors(self, X: np.ndarray, log_prior=None) -> np.ndarray:
        """
        Compute the matrix of joint densities SJoint for all test samples and classes.

        Parameters:
        ----------
        X_test : np.ndarray
            Test dataset.

        Returns:
        -------
        np.ndarray
            The matrix of joint densities SJoint.
        """
        return self.utils.compute_log_posteriors(X, self.h_params, log_prior)

    def compute_class_posteriors(self, X_val: np.ndarray, log_prior=None) -> np.ndarray:
        return self.utils.compute_class_posteriors(X_val, self.h_params, log_prior)

    def compute_SJoint(
        self, log_likelihood, prior_prob, is_prior_log: bool = False
    ) -> np.ndarray:
        return self.utils.compute_SJoint(
            log_likelihood, prior_prob, prob_is_log=is_prior_log
        )

    def compute_log_marginal(
        self, s_joint: np.ndarray, *, axis=0
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.utils.compute_log_marginal(s_joint, axis=axis)

    def log_likelihood_ratio(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log likelihood ratio for the given data.

        Parameters:
        ----------
        X : np.ndarray
            Data for which to compute the log likelihood ratio.

        Returns:
        -------
        np.ndarray
            The log likelihood ratio.
        """
        return self.utils.log_likelihood_ratio(X, self.classes, self.h_params)

    def compute_error_rate(
        self, llr: np.ndarray, y_val: np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        return self.utils.compute_error_rate(llr, y_val, threshold)

    def compute_accuracy(self, y_p, y_true) -> float:
        return self.utils.compute_accuracy(y_p, y_true)
