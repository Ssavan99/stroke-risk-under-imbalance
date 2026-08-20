"""Metrics for a 4.87%-positive problem.

Plain accuracy is deliberately absent from every headline in this module. On this
dataset a model that answers "no stroke" for everyone scores 95.13%, and the
legacy code in ``notebooks/legacy/`` contains exactly such a model reporting
exactly that number. Accuracy is computed in one place only — the threshold sweep
— where it sits beside sensitivity and precision so its uselessness is visible
rather than flattering.

The metrics that are reported:

* **Average precision (PR-AUC)** is the headline. It is threshold-free, and its
  no-skill baseline equals the prevalence, so a degenerate model scores ~0.049
  and cannot hide behind the majority class.
* **ROC-AUC** is reported for comparability with published work on this dataset,
  flagged as optimistic: it is insensitive to the false-positive burden that
  dominates a rare-outcome problem.
* **Brier score** and a reliability curve, because a predicted risk that is not
  calibrated is not a risk.
* **An operating point** — everything above is threshold-free, and nobody deploys
  a threshold-free model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import N_BOOTSTRAP, SEED, TARGET_SENSITIVITY


@dataclass(frozen=True)
class OperatingPoint:
    """What the model actually costs to use at one chosen threshold."""

    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    false_positive_rate: float
    #: How many people must be flagged to find one true case (1 / precision).
    number_needed_to_screen: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict:
        return asdict(self)


def discrimination(y_true, y_prob) -> dict[str, float]:
    """Threshold-free scores, each paired with what a useless model would get."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    base = float(y_true.mean())
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "pr_auc_no_skill": base,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "roc_auc_no_skill": 0.5,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "brier_always_base_rate": float(np.mean((y_true - base) ** 2)),
        "prevalence": base,
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
    }


def operating_point(
    y_true,
    y_prob,
    target_sensitivity: float = TARGET_SENSITIVITY,
) -> OperatingPoint:
    """Pick the highest threshold that still reaches ``target_sensitivity``.

    Choosing the threshold by sensitivity rather than by maximising F1 or Youden's
    J is a deliberate framing: for a rare, serious outcome the question a reader
    actually has is "if it catches 4 in 5 cases, how many people does it flag?"
    Taking the *highest* qualifying threshold gives the best precision available
    at that recall.

    Call this on training out-of-fold predictions, then pass the result to
    :func:`apply_threshold` for the held-out set. Calling it directly on the
    held-out set fits the decision rule to the data being reported.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # roc_curve gives tpr at every distinct threshold, which is what we search.
    # drop_intermediate=False because the default (True) discards collinear
    # points for plotting, which can leave the highest qualifying threshold out
    # of the array entirely when scores are tied. It makes no measurable
    # difference on this dataset, but it makes the docstring above exactly true
    # rather than true-in-practice.
    _fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    qualifying = np.flatnonzero(tpr >= target_sensitivity)
    # thresholds is descending, so the first qualifying index is the highest
    # threshold that still reaches the target recall. Nothing qualifying is only
    # possible when every score is tied, in which case flag everyone.
    threshold = (
        float(thresholds[qualifying[0]]) if qualifying.size else float(np.min(y_prob))
    )

    predicted = y_prob >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    tn = int(np.sum(~predicted & (y_true == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return OperatingPoint(
        threshold=threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        npv=npv,
        false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
        number_needed_to_screen=(1.0 / precision) if precision else float("inf"),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def apply_threshold(y_true, y_prob, threshold: float) -> OperatingPoint:
    """Score a *pre-chosen* threshold, without letting the data move it.

    The threshold this project reports is selected on cross-validated
    out-of-fold predictions over the training portion, then applied here
    unchanged. Picking it on the held-out set instead would tune a decision rule
    on the data used to report it — a smaller version of the leak this repository
    exists to correct — and would make the achieved sensitivity look exact when it
    is really an estimate.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    predicted = y_prob >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    tn = int(np.sum(~predicted & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return OperatingPoint(
        threshold=float(threshold),
        sensitivity=tp / (tp + fn) if (tp + fn) else 0.0,
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        precision=precision,
        npv=tn / (tn + fn) if (tn + fn) else 0.0,
        false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
        number_needed_to_screen=(1.0 / precision) if precision else float("inf"),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def threshold_sweep(y_true, y_prob, n_points: int = 40) -> list[dict[str, float]]:
    """Sensitivity/precision/accuracy across the whole threshold range.

    This is the one place accuracy appears, and it appears here on purpose: the
    row where accuracy peaks is the row where sensitivity collapses, which makes
    the argument better than any amount of prose.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    lo, hi = float(np.min(y_prob)), float(np.max(y_prob))
    grid = np.unique(np.clip(np.linspace(lo, hi, n_points), 0.0, 1.0))

    rows = []
    for t in grid:
        predicted = y_prob >= t
        tp = int(np.sum(predicted & (y_true == 1)))
        fp = int(np.sum(predicted & (y_true == 0)))
        fn = int(np.sum(~predicted & (y_true == 1)))
        tn = int(np.sum(~predicted & (y_true == 0)))
        rows.append(
            {
                "threshold": float(t),
                "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "accuracy": (tp + tn) / y_true.size,
                "flagged": int(np.sum(predicted)),
                "flagged_fraction": float(np.mean(predicted)),
            }
        )
    return rows


def bootstrap_ci(
    y_true,
    y_prob,
    metric: str = "pr_auc",
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval, stratified to keep the positive count fixed.

    Roughly 50 positive cases land in the held-out set, so a point estimate on its
    own overstates what this dataset can support. Resampling within each class
    avoids draws that contain no positives at all, which would make the metric
    undefined.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score

    scores = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        scores[i] = fn(y_true[idx], y_prob[idx])

    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def proportion_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Sensitivity and precision at the reported threshold are proportions over 50
    and 309 cases respectively, so quoting them as bare point estimates implies
    a precision the data does not have. Wilson rather than normal-approximation
    because it stays inside [0, 1] and behaves at small counts, which is the
    regime this dataset is permanently in.
    """
    if trials <= 0:
        return (0.0, 0.0)

    from math import sqrt

    # 1.959964 is the 97.5th percentile of the standard normal, i.e. alpha=0.05.
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z_for(alpha)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    halfwidth = z * sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def _z_for(alpha: float) -> float:  # pragma: no cover - only for non-default alpha
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - alpha / 2)


def operating_point_ci(op: OperatingPoint) -> dict[str, list[float]]:
    """Wilson intervals for the rates quoted at the reported operating point."""
    return {
        "sensitivity_ci95": list(proportion_ci(op.tp, op.tp + op.fn)),
        "precision_ci95": list(proportion_ci(op.tp, op.tp + op.fp)),
        "specificity_ci95": list(proportion_ci(op.tn, op.tn + op.fp)),
    }


def calibration_bins(y_true, y_prob, n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability curve over quantile bins.

    Quantile bins rather than equal-width, because predicted risks on a 4.87%
    problem cluster near zero and equal-width bins would leave most of them empty.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:  # pragma: no cover - degenerate constant predictor
        return []

    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


def pr_curve_points(y_true, y_prob) -> dict[str, list[float]]:
    """Precision/recall curve, thinned for plotting and JSON."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    step = max(1, precision.size // 300)
    return {
        "precision": [float(v) for v in precision[::step]],
        "recall": [float(v) for v in recall[::step]],
    }
