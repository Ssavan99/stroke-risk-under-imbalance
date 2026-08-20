"""Metric behaviour, checked against cases with known answers.

The operating-point search and the baselines are the two places where a silent
error would corrupt the headline claim, so both are pinned to hand-computable
results rather than to whatever the code currently returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from stroke.baselines import MajorityClassBaseline
from stroke.evaluate import (
    apply_threshold,
    bootstrap_ci,
    calibration_bins,
    discrimination,
    operating_point,
    threshold_sweep,
)


def test_majority_baseline_scores_at_chance():
    """A constant predictor must score exactly prevalence on PR-AUC and 0.5 on ROC.

    This is the test that would have caught the legacy 95.7% result: the model in
    log_regression.ipynb was, functionally, this class.
    """
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)

    model = MajorityClassBaseline().fit(None, y)
    prob = model.predict_proba(np.zeros((y.size, 1)))[:, 1]

    scores = discrimination(y, prob)
    assert scores["roc_auc"] == pytest.approx(0.5, abs=1e-9)
    assert scores["pr_auc"] == pytest.approx(y.mean(), abs=0.01)
    assert scores["pr_auc"] == pytest.approx(scores["pr_auc_no_skill"], abs=0.01)
    # It never predicts a positive, exactly like the legacy notebook.
    assert model.predict(np.zeros((y.size, 1))).sum() == 0


def test_perfect_ranking_scores_one():
    y = np.array([0, 0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
    scores = discrimination(y, prob)
    assert scores["pr_auc"] == pytest.approx(1.0)
    assert scores["roc_auc"] == pytest.approx(1.0)


def test_operating_point_reaches_the_requested_sensitivity():
    """The chosen threshold must actually achieve at least the target recall."""
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(int)
    prob = np.clip(0.05 + 0.35 * y + rng.normal(0, 0.15, y.size), 0, 1)

    op = operating_point(y, prob, target_sensitivity=0.80)
    assert op.sensitivity >= 0.80
    assert op.tp + op.fn == y.sum()
    assert op.tp + op.fp + op.fn + op.tn == y.size


def test_operating_point_picks_the_highest_qualifying_threshold():
    """Hand-computable case: the best precision available at the target recall.

    Four positives; catching three of them is 0.75 sensitivity. Raising the
    threshold any further drops to 0.50, so 0.6 is the highest threshold that
    still qualifies at a 0.75 target.
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.5, 0.3, 0.1])

    op = operating_point(y, prob, target_sensitivity=0.75)
    assert op.sensitivity == pytest.approx(0.75)
    assert op.threshold == pytest.approx(0.6)
    assert (op.tp, op.fp, op.fn, op.tn) == (3, 1, 1, 3)
    assert op.precision == pytest.approx(0.75)
    assert op.number_needed_to_screen == pytest.approx(1 / 0.75)


def test_apply_threshold_does_not_move_the_threshold():
    y = np.array([1, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.3, 0.4, 0.2, 0.1, 0.05])

    op = apply_threshold(y, prob, 0.35)
    assert op.threshold == pytest.approx(0.35)
    # 0.9 and 0.4 clear the bar: one true positive, one false positive.
    assert (op.tp, op.fp, op.fn, op.tn) == (1, 1, 1, 3)
    assert op.sensitivity == pytest.approx(0.5)
    assert op.precision == pytest.approx(0.5)


def test_apply_threshold_flagging_nobody_is_handled():
    y = np.array([1, 0, 0, 0])
    prob = np.array([0.2, 0.1, 0.05, 0.01])
    op = apply_threshold(y, prob, 0.99)
    assert (op.tp, op.fp) == (0, 0)
    assert op.precision == 0.0
    assert op.number_needed_to_screen == float("inf")


def test_threshold_sweep_shows_accuracy_peaking_where_sensitivity_dies():
    """The argument the README makes, asserted on the real model and real data.

    Maximising accuracy on this dataset drives the model toward answering "no
    stroke" for everyone. The threshold that scores best on accuracy should
    therefore be one that catches almost nobody — which is precisely why the
    README never quotes accuracy.
    """
    from stroke.data import load_raw, split
    from stroke.pipeline import build_pipeline

    X_train, X_test, y_train, y_test = split(load_raw(), seed=42)
    pipe = build_pipeline("logistic_regression", "none", seed=42).fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]

    rows = threshold_sweep(y_test, prob)
    best_accuracy = max(rows, key=lambda r: r["accuracy"])

    assert best_accuracy["accuracy"] > 0.94
    assert best_accuracy["sensitivity"] < 0.10, (
        "the accuracy-optimal threshold should catch almost no cases; "
        f"got sensitivity {best_accuracy['sensitivity']:.3f}"
    )
    # And it is barely better than refusing to predict anyone as positive.
    always_negative_accuracy = 1.0 - float(np.mean(y_test))
    assert best_accuracy["accuracy"] - always_negative_accuracy < 0.02


def test_calibration_bins_are_ordered_and_cover_everything():
    rng = np.random.default_rng(3)
    prob = rng.random(1000)
    y = (rng.random(1000) < prob).astype(int)

    bins = calibration_bins(y, prob, n_bins=10)
    assert bins
    assert sum(b["count"] for b in bins) == y.size
    means = [b["mean_predicted"] for b in bins]
    assert means == sorted(means)
    # A well-calibrated generator should track the diagonal closely.
    for b in bins:
        assert abs(b["mean_predicted"] - b["observed_rate"]) < 0.20


def test_calibration_bins_survive_a_constant_predictor():
    y = np.array([0, 1, 0, 0])
    prob = np.full(4, 0.25)
    assert calibration_bins(y, prob) == []


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(4)
    y = (rng.random(1500) < 0.05).astype(int)
    prob = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.2, y.size), 0, 1)

    point = discrimination(y, prob)["pr_auc"]
    lo, hi = bootstrap_ci(y, prob, metric="pr_auc", n_resamples=300, seed=7)
    assert lo < point < hi

    again = bootstrap_ci(y, prob, metric="pr_auc", n_resamples=300, seed=7)
    assert (lo, hi) == again
