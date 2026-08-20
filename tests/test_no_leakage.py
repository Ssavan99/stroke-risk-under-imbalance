"""Guards against the defect this repository exists to correct.

The legacy pipeline resampled before splitting, which produced a balanced test
set and inflated every reported score. These tests fail if that ordering ever
comes back — structurally, and by measuring the inflation directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from stroke.data import load_raw, prevalence, split
from stroke.pipeline import ESTIMATORS, STRATEGIES, build_pipeline


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_held_out_prevalence_matches_population(raw):
    """The test set must look like the world, not like a balanced sample.

    This is the single check that would have caught the original bug. The legacy
    code produced a test set of 958 negatives to 922 positives from a population
    that is 4.87% positive.
    """
    _, _, y_train, y_test = split(raw)
    population = prevalence(raw["stroke"])

    assert prevalence(y_test) == pytest.approx(population, abs=0.005)
    assert prevalence(y_train) == pytest.approx(population, abs=0.005)
    # And explicitly: nowhere near balanced.
    assert prevalence(y_test) < 0.10


def test_no_row_appears_in_both_splits(raw):
    X_train, X_test, _, _ = split(raw)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) + len(X_test) == len(raw)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_resampler_is_inside_the_pipeline(name, strategy):
    """Structural guard: SMOTE may exist only as a pipeline step.

    A resampler inside an imblearn Pipeline is applied on fit and skipped on
    predict. Called outside one — as ``data_utils.fit_resample`` does — it
    rewrites the dataset before the split and contaminates everything after.
    """
    pipe = build_pipeline(name, strategy)
    assert isinstance(pipe, ImbPipeline)

    resamplers = [s for _, s in pipe.steps if isinstance(s, SMOTE)]
    if strategy == "smote":
        assert len(resamplers) == 1, "smote strategy must contribute exactly one resampler"
        assert pipe.steps[-1][0] == "model", "the estimator must remain the final step"
    else:
        assert not resamplers, f"strategy {strategy!r} must not introduce a resampler"


def test_pipeline_does_not_resample_at_predict_time(raw):
    """Predicting must return one row per input row, resampler or not."""
    X_train, X_test, y_train, _ = split(raw)
    pipe = build_pipeline("logistic_regression", "smote")
    pipe.fit(X_train, y_train)
    assert pipe.predict_proba(X_test).shape[0] == len(X_test)


def test_leakage_inflates_the_score_and_we_can_measure_it(raw):
    """Reproduce the legacy ordering and confirm it reports a far better model.

    Same estimator, same seed, same data. The only difference is whether SMOTE
    runs before or inside the split. If this test ever stops showing a large gap,
    either the correct path has regressed or the leaky path has been fixed by
    accident — both are worth failing over.
    """
    from stroke.config import CATEGORICAL_FEATURES, FEATURES, SEED, TARGET

    df = raw.copy()
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string")
    X = X.fillna({"bmi": X["bmi"].median()})
    X = X.join(
        __import__("pandas").get_dummies(X[CATEGORICAL_FEATURES], dtype=float)
    ).drop(columns=CATEGORICAL_FEATURES)
    y = df[TARGET].astype(int)

    # --- the legacy ordering: resample everything, then split -----------------
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X, y)
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
        X_res, y_res, test_size=0.2, random_state=SEED
    )
    leaky = LogisticRegression(max_iter=5000, random_state=SEED).fit(Xl_tr, yl_tr)
    leaky_ap = average_precision_score(yl_te, leaky.predict_proba(Xl_te)[:, 1])

    # --- the corrected ordering: split, then resample inside the pipeline -----
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    honest = ImbPipeline(
        steps=[
            ("resample", SMOTE(random_state=SEED)),
            ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
        ]
    ).fit(Xc_tr, yc_tr)
    honest_ap = average_precision_score(yc_te, honest.predict_proba(Xc_te)[:, 1])

    # The leaked test set is balanced; the honest one is not.
    assert np.mean(yl_te) > 0.4, "legacy ordering should produce a ~balanced test set"
    assert np.mean(yc_te) < 0.10, "correct ordering must preserve the population rate"

    # And the leak buys an enormous, entirely fictitious improvement.
    assert leaky_ap > 0.80, f"expected the leaked score to look great, got {leaky_ap:.3f}"
    assert honest_ap < 0.40, f"expected the honest score to be modest, got {honest_ap:.3f}"
    assert leaky_ap > honest_ap * 2
