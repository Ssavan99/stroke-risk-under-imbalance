"""What the CSV must contain for any reported number to mean anything.

Every figure in the README is stated against this exact shape. If the data file
is ever replaced or re-cleaned, these fail rather than letting the README quote
numbers from a different dataset.
"""

from __future__ import annotations

import pytest

from stroke.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from stroke.data import load_raw, prevalence


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_shape(raw):
    assert len(raw) == 5110


def test_id_is_dropped(raw):
    assert "id" not in raw.columns


def test_all_declared_features_exist(raw):
    missing = [c for c in [*FEATURES, TARGET] if c not in raw.columns]
    assert not missing, f"missing columns: {missing}"


def test_prevalence(raw):
    assert int(raw[TARGET].sum()) == 249
    assert prevalence(raw[TARGET]) == pytest.approx(0.0487, abs=0.0005)


def test_accuracy_of_a_useless_model_is_embarrassingly_high(raw):
    """The number that makes plain accuracy unusable here, asserted explicitly."""
    always_negative_accuracy = 1.0 - prevalence(raw[TARGET])
    assert always_negative_accuracy == pytest.approx(0.9513, abs=0.0005)


def test_bmi_missingness_is_informative(raw):
    """The finding that justifies imputing rather than dropping.

    If this ever stops holding, the imputation rationale in the data card and the
    README is no longer supported by the data and must be rewritten.
    """
    missing = raw[raw["bmi"].isna()]
    present = raw[raw["bmi"].notna()]

    assert len(missing) == 201
    assert prevalence(missing[TARGET]) == pytest.approx(0.199, abs=0.005)
    assert prevalence(present[TARGET]) == pytest.approx(0.0426, abs=0.005)
    # Dropping them would discard 40 of 249 positives.
    assert int(missing[TARGET].sum()) == 40
    assert prevalence(missing[TARGET]) > 4 * prevalence(present[TARGET])


def test_numeric_features_are_numeric(raw):
    for col in NUMERIC_FEATURES:
        assert raw[col].dtype.kind in "if", f"{col} should be numeric, got {raw[col].dtype}"


def test_categorical_features_are_strings(raw):
    for col in CATEGORICAL_FEATURES:
        assert raw[col].dtype == "string", f"{col} should be string, got {raw[col].dtype}"


def test_only_bmi_has_missing_values(raw):
    counts = raw.isna().sum()
    assert counts["bmi"] == 201
    assert counts.drop("bmi").sum() == 0


def test_rare_category_is_retained(raw):
    """gender == 'Other' has one row and must survive loading."""
    assert (raw["gender"] == "Other").sum() == 1
