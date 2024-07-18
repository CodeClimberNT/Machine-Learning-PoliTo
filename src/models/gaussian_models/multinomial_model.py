import numpy as np
from typing import Optional, Self

from src.models.gaussian_models.base_gaussian_model import BaseGaussianModel


class MultinomialModel(BaseGaussianModel):
    def __init__(self, *, eps: float = 1.0) -> None:
        super().__init__()
        self.h_params = {}
        self.eps: float = eps
        self.common_words: set[str] = set([])
        self.h_cls_log_prob: dict[str, dict[str, float]] = {}

    def fit(self, X: dict[str, list[str]], y: Optional[np.ndarray] = None) -> Self:  # type: ignore

        self.common_words = self._common_words_set(X)
        self.h_cls_log_prob = self._estimate_counts(X)
        return self

    def _common_words_set(self, X: dict[str, list[str]]) -> set[str]:
        common_words = set([])
        for c in X:
            tercets_c = X[c]
            dict_c = self._all_words_set(tercets_c)
            common_words = common_words.union(dict_c)
        return common_words

    def _all_words_set(self, X: list[str]) -> set[str]:
        freq_dict = set([])
        for s in X:
            words = s.split()
            for word in words:
                freq_dict.add(word)
        return freq_dict

    def _estimate_counts(self, X: dict[str, list[str]]) -> dict[str, dict[str, float]]:
        counts = self._initialize_word_counts(X)
        for c in X:
            for tercet in X[c]:
                words = tercet.split()
                for word in words:
                    counts[c][word] += 1
        return counts

    def _initialize_word_counts(
        self, X: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        counts = {}
        for c in X:
            counts[c] = {w: self.eps for w in self.common_words}
        return counts

    def compute_frequencies(
        self, X: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        if len(self.h_cls_log_prob) == 0:
            self.h_cls_log_prob = self._estimate_counts(X)

        for c in X:
            num_words_c = sum(self.h_cls_log_prob[c].values())
            for w in self.h_cls_log_prob:
                self.h_cls_log_prob[c][w] = np.log(self.h_params[c][w]) - np.log(
                    num_words_c
                )
        return self.h_cls_log_prob

    def compute_log_likelihood(self, X_val: str) -> dict[str, float]:  # type: ignore
        log_likelihood_cls: dict[str, float] = {c: 0 for c in self.h_cls_log_prob}
        for c in self.h_cls_log_prob:
            for word in X_val.split():
                if word in self.h_cls_log_prob[c]:
                    log_likelihood_cls[c] += self.h_cls_log_prob[c][word]

        return log_likelihood_cls

    def compute_log_likelihood_matrix(
        self, X_val: list[str], h_class_to_idx: Optional[dict[str, int]] = None
    ) -> np.ndarray:
        if h_class_to_idx is None:
            h_class_to_idx = {c: i for i, c in enumerate(sorted(self.h_cls_log_prob))}

        log_likelihood = np.zeros((len(self.h_cls_log_prob), len(X_val)))
        for i, tercet in enumerate(X_val):
            scores: dict[str, float] = self.compute_log_likelihood(tercet)
            for c in self.h_cls_log_prob:
                log_likelihood[h_class_to_idx[c], i] = scores[c]

        return log_likelihood
