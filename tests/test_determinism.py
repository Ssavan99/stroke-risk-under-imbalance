"""Same seed, same numbers.

The legacy code set no ``random_state`` anywhere, so its reported figures could
not be reproduced even by re-running the notebook that produced them. Anything
quoted in the README has to survive a second run.
"""

from __future__ import annotations

import numpy as np
import pytest

from stroke.data import load_raw, split
from stroke.evaluate import discrimination
from stroke.pipeline import build_pipeline


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_split_is_reproducible(raw):
    a = split(raw, seed=42)
    b = split(raw, seed=42)
    assert list(a[0].index) == list(b[0].index)
    assert list(a[1].index) == list(b[1].index)


def test_different_seeds_give_different_splits(raw):
    a = split(raw, seed=42)
    b = split(raw, seed=1)
    assert list(a[1].index) != list(b[1].index)


@pytest.mark.parametrize(
    ("name", "strategy"),
    [
        ("logistic_regression", "none"),
        ("logistic_regression", "smote"),
        ("random_forest", "none"),
    ],
)
def test_fitted_model_gives_identical_probabilities(raw, name, strategy):
    X_train, X_test, y_train, _ = split(raw, seed=42)

    first = build_pipeline(name, strategy, seed=42).fit(X_train, y_train)
    second = build_pipeline(name, strategy, seed=42).fit(X_train, y_train)

    np.testing.assert_array_equal(
        first.predict_proba(X_test)[:, 1],
        second.predict_proba(X_test)[:, 1],
    )


def test_reported_metrics_are_identical_across_runs(raw):
    X_train, X_test, y_train, y_test = split(raw, seed=42)
    scores = []
    for _ in range(2):
        pipe = build_pipeline("logistic_regression", "none", seed=42).fit(X_train, y_train)
        scores.append(discrimination(y_test, pipe.predict_proba(X_test)[:, 1]))
    assert scores[0] == scores[1]
