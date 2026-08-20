"""A missing categorical value must impute, not crash.

The categorical columns have no nulls in the shipped CSV, so the categorical
imputer never actually runs. That made a latent failure invisible: with pandas
StringDtype the missing marker is pd.NA, and SimpleImputer's `X != X` detection
returns pd.NA rather than True, raising "boolean value of NA is ambiguous".

It would have surfaced the first time anyone re-downloaded the data with a blank
cell, mapped "Unknown" to null, or submitted a partially filled form.
"""

from __future__ import annotations

import numpy as np
import pytest

from stroke.config import CATEGORICAL_FEATURES
from stroke.data import load_raw, split
from stroke.pipeline import build_pipeline


@pytest.mark.parametrize("column", CATEGORICAL_FEATURES)
def test_a_missing_category_is_imputed_rather_than_raising(column):
    raw = load_raw()
    X_train, X_test, y_train, _ = split(raw, seed=42)

    holed = X_test.copy()
    holed.loc[holed.index[:5], column] = np.nan

    pipe = build_pipeline("logistic_regression", "none", seed=42).fit(X_train, y_train)
    probs = pipe.predict_proba(holed)[:, 1]

    assert probs.shape == (len(holed),)
    assert np.all(np.isfinite(probs))


def test_training_with_missing_categories_also_works():
    raw = load_raw()
    X_train, X_test, y_train, _ = split(raw, seed=42)

    holed = X_train.copy()
    holed.loc[holed.index[:50], "smoking_status"] = np.nan

    pipe = build_pipeline("logistic_regression", "none", seed=42).fit(holed, y_train)
    assert np.all(np.isfinite(pipe.predict_proba(X_test)[:, 1]))
