import numpy as np
import scipy.special

from src.helpers import MathHelper as mh


class GaussianUtils:
    @staticmethod
    def compute_mu_and_sigma(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return mh.compute_mu_and_sigma(X)

    @staticmethod
    def inv_matrix(matrix: np.ndarray | None) -> np.ndarray | None:
        if matrix is None:
            return None
        return mh.inv_matrix(matrix)

    @staticmethod
    def log_det_matrix(matrix: np.ndarray | None) -> float | None:
        if matrix is None:
            return None
        return mh.log_det_matrix(matrix)

    @staticmethod
    def calculate_probability_distribution(
        x: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        inv_cov = mh.inv_matrix(sigma)
        log_det_cov = mh.log_det_matrix(sigma)
        return (
            -0.5 * x.shape[0] * np.log(np.pi * 2)
            - 0.5 * log_det_cov
            - 0.5 * ((x - mu) * (inv_cov @ (x - mu))).sum(0)
        )

    @staticmethod
    def log_likelihood_ratio(
        X: np.ndarray, classes: np.ndarray, h_params: dict
    ) -> np.ndarray:
        for c in classes:
            h_params[c]["probability_"] = (
                GaussianUtils.calculate_probability_distribution(
                    X, h_params[c]["mean_"], h_params[c]["sigma_"]
                )
            )
        return (
            h_params[classes[1]]["probability_"] - h_params[classes[0]]["probability_"]
        )

    @staticmethod
    def compute_SJoint(
        log_likelihood, prior_prob, is_prior_log: bool = False
    ) -> np.ndarray:
        if not is_prior_log:
            return log_likelihood + mh.v_col(np.log(prior_prob))
        return log_likelihood + mh.v_col(prior_prob)

    @staticmethod
    def compute_log_marginal(
        SJoint: np.ndarray, *, axis=0
    ) -> tuple[np.ndarray, np.ndarray]:
        return scipy.special.logsumexp(SJoint, axis=axis)

    @staticmethod
    def compute_prior(num_classes: int) -> np.ndarray:
        return np.ones(num_classes) / float(num_classes)

    @staticmethod
    def compute_log_prior(num_classes: int) -> np.ndarray:
        return np.log(GaussianUtils.compute_prior(num_classes))

    @staticmethod
    def compute_log_likelihood(X_val: np.ndarray, h_params: dict) -> np.ndarray:
        S = np.zeros((len(h_params), X_val.shape[1]))
        for c in h_params:
            S[c, :] = GaussianUtils.calculate_probability_distribution(
                X_val, h_params[c]["mean_"], h_params[c]["sigma_"]
            )
        return S

    @staticmethod
    def compute_log_posteriors(X_val, h_params: dict, log_prior=None) -> np.ndarray:
        if log_prior is None:
            log_prior = GaussianUtils.compute_log_prior(len(h_params))

        log_likelihood = GaussianUtils.compute_log_likelihood(X_val, h_params)
        joint = GaussianUtils.compute_SJoint(
            log_likelihood, log_prior, is_prior_log=True
        )
        marginal = GaussianUtils.compute_log_marginal(joint)
        return joint - marginal

    @staticmethod
    def compute_class_posteriors(X_val, h_params: dict, log_prior=None) -> np.ndarray:
        return np.exp(GaussianUtils.compute_log_posteriors(X_val, h_params, log_prior))

    @staticmethod
    def compute_error_rate(
        llr: np.ndarray, y_val: np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        predicted_val = np.zeros(y_val.size, dtype=np.int32)
        if threshold is None:
            threshold = 0
        predicted_val[llr >= threshold] = 1
        predicted_val[llr < threshold] = 0
        return (np.sum(predicted_val != y_val) / y_val.size) * 100

    @staticmethod
    def compute_accuracy(y_p: np.ndarray, y_true: np.ndarray) -> float:
        predicted_label = np.argmax(y_p, axis=0)
        num_correct = np.sum(predicted_label.ravel() == y_true.ravel())
        num_total = y_true.size
        return float(num_correct) / float(num_total)
