"""Baselines the real model has to beat.

Two of them, for two different reasons.

``MajorityClassBaseline`` is the model the legacy ``log_regression.ipynb``
accidentally trained: it predicts the base rate for everyone and never identifies
a single case. It is included so that the headline metric can be seen scoring it
at chance. Under accuracy it would score 95.13%.

``age_only_pipeline`` is the harder and more interesting bar. Age is by far the
strongest single predictor in this dataset, and a one-feature logistic regression
on it is genuinely competitive. Any multi-feature model that cannot clearly beat
it has not earned its complexity, and reporting that comparison is the difference
between a result and a claim.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


class MajorityClassBaseline(ClassifierMixin, BaseEstimator):
    """Always predicts the negative class; probability is the training base rate.

    A constant predictor has no ability to rank, so its average precision equals
    the prevalence and its ROC-AUC is exactly 0.5 — which is the point.
    """

    def fit(self, X, y):  # noqa: ARG002 - X is required by the scikit-learn API
        y = np.asarray(y, dtype=int)
        self.base_rate_ = float(y.mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, self.base_rate_, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def age_only_pipeline(seed: int = SEED) -> Pipeline:
    """Logistic regression on ``age`` alone."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            ["age"],
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=5000, random_state=seed)),
        ]
    )


def majority_pipeline(seed: int = SEED) -> Pipeline:  # noqa: ARG001 - uniform signature
    """Wrap the constant predictor so it has the same interface as the others."""
    return Pipeline(steps=[("model", MajorityClassBaseline())])
