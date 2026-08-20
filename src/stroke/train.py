"""Run the full comparison and write the reported numbers to disk.

    python -m stroke.train              # full run
    python -m stroke.train --fast       # single CV repeat, fewer bootstraps

The protocol, in order:

1. Split once, stratified, before anything is fitted.
2. Cross-validate every estimator x imbalance strategy on the training portion
   only, with all preprocessing and resampling inside the pipeline.
3. Select one configuration on cross-validated PR-AUC.
4. Choose the decision threshold on cross-validated out-of-fold predictions over
   the training portion, so the held-out set never moves it.
5. Evaluate that configuration and that threshold on the held-out set, and stop.

Every other configuration is also scored on the held-out set, so the comparison
table is like-for-like — but each gets its own threshold fixed on its own
training folds, and *selection is never revised after seeing any of it*. That
last clause is the whole discipline. Reporting a comparison is not the same as
choosing from one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from . import __version__
from .baselines import age_only_pipeline, majority_pipeline
from .config import (
    CV_REPEATS,
    CV_SPLITS,
    FIGURE_DIR,
    N_BOOTSTRAP,
    N_JOBS,
    OUTPUT_DIR,
    SEED,
    TARGET_SENSITIVITY,
)
from .data import load_raw, prevalence, split
from .evaluate import (
    apply_threshold,
    bootstrap_ci,
    calibration_bins,
    discrimination,
    operating_point,
    operating_point_ci,
    pr_curve_points,
    threshold_sweep,
)
from .pipeline import ESTIMATORS, STRATEGIES, build_pipeline
from .plots import render_all

#: The configuration exported to the browser demo. A plain, unweighted logistic
#: regression is used there rather than whichever model wins on PR-AUC, for two
#: reasons: its probabilities are calibrated by construction, and it reduces to a
#: dot product that runs client-side with no runtime dependency. The README
#: reports its metrics next to the selected model's so the tradeoff is visible.
WEB_CONFIG = ("logistic_regression", "none")


@dataclass(frozen=True)
class CVResult:
    estimator: str
    strategy: str
    pr_auc_mean: float
    pr_auc_std: float
    roc_auc_mean: float
    roc_auc_std: float

    def as_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "strategy": self.strategy,
            "pr_auc_mean": self.pr_auc_mean,
            "pr_auc_std": self.pr_auc_std,
            "roc_auc_mean": self.roc_auc_mean,
            "roc_auc_std": self.roc_auc_std,
        }


def _cv(pipe, X, y, *, splits: int, repeats: int, seed: int) -> tuple[float, float, float, float]:
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=("average_precision", "roc_auc"),
        n_jobs=N_JOBS,
        error_score="raise",
    )
    return (
        float(np.mean(scores["test_average_precision"])),
        float(np.std(scores["test_average_precision"])),
        float(np.mean(scores["test_roc_auc"])),
        float(np.std(scores["test_roc_auc"])),
    )


def run_cv_comparison(X, y, *, splits: int, repeats: int, seed: int) -> list[CVResult]:
    """Every estimator against every imbalance strategy, plus both baselines."""
    results: list[CVResult] = []

    for name in ESTIMATORS:
        for strategy in STRATEGIES:
            pr_m, pr_s, roc_m, roc_s = _cv(
                build_pipeline(name, strategy, seed),
                X,
                y,
                splits=splits,
                repeats=repeats,
                seed=seed,
            )
            results.append(CVResult(name, strategy, pr_m, pr_s, roc_m, roc_s))
            print(
                f"  {name:20s} {strategy:13s} "
                f"PR-AUC {pr_m:.4f} +/- {pr_s:.4f}   ROC-AUC {roc_m:.4f}"
            )

    for label, pipe in (
        ("baseline_majority", majority_pipeline(seed)),
        ("baseline_age_only", age_only_pipeline(seed)),
    ):
        pr_m, pr_s, roc_m, roc_s = _cv(
            pipe, X, y, splits=splits, repeats=repeats, seed=seed
        )
        results.append(CVResult(label, "n/a", pr_m, pr_s, roc_m, roc_s))
        print(
            f"  {label:20s} {'n/a':13s} "
            f"PR-AUC {pr_m:.4f} +/- {pr_s:.4f}   ROC-AUC {roc_m:.4f}"
        )

    return results


def choose_threshold(pipe, X_train, y_train, *, splits: int, seed: int) -> tuple[float, dict]:
    """Fix the operating threshold using the training portion only.

    ``cross_val_predict`` refits the whole pipeline per fold, so each training row
    is scored by a model that never saw it. The threshold reaching the target
    sensitivity on those out-of-fold scores is then frozen and carried to the
    held-out set unchanged.
    """
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    oof = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=N_JOBS
    )[:, 1]
    op = operating_point(y_train, oof, TARGET_SENSITIVITY)
    return op.threshold, op.as_dict()


def evaluate_on_test(
    pipe,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    n_bootstrap: int,
    seed: int,
    threshold: float | None = None,
) -> dict:
    """Fit on the training portion, score once on the held-out portion."""
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = discrimination(y_test, y_prob)
    pr_lo, pr_hi = bootstrap_ci(
        y_test, y_prob, metric="pr_auc", n_resamples=n_bootstrap, seed=seed
    )
    roc_lo, roc_hi = bootstrap_ci(
        y_test, y_prob, metric="roc_auc", n_resamples=n_bootstrap, seed=seed
    )
    metrics["pr_auc_ci95"] = [pr_lo, pr_hi]
    metrics["roc_auc_ci95"] = [roc_lo, roc_hi]
    metrics["mean_predicted_risk"] = float(np.mean(y_prob))

    # `operating_point` is only ever populated from a threshold fixed on training
    # data. A caller that supplies no threshold gets no `operating_point` key at
    # all, rather than a test-tuned one under an innocent-looking name — every
    # config in this file is scored the same way, so a table templated from these
    # values cannot accidentally publish oracle numbers.
    if threshold is not None:
        op = apply_threshold(y_test, y_prob, threshold)
        entry = op.as_dict()
        entry["threshold_source"] = "training out-of-fold"
        entry.update(operating_point_ci(op))
        metrics["operating_point"] = entry

    # Kept for reference under an unambiguous name: what the threshold would have
    # been if tuned on the held-out set. Optimistic by construction. Its presence
    # is what lets the README quantify how much that tuning would have bought.
    oracle = operating_point(y_test, y_prob, TARGET_SENSITIVITY).as_dict()
    oracle["threshold_source"] = "held-out (oracle, NOT reported)"
    metrics["operating_point_oracle"] = oracle
    metrics["calibration"] = calibration_bins(y_test, y_prob)
    metrics["threshold_sweep"] = threshold_sweep(y_test, y_prob)
    metrics["pr_curve"] = pr_curve_points(y_test, y_prob)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="one CV repeat and 200 bootstrap resamples; for CI and smoke tests",
    )
    args = parser.parse_args(argv)

    repeats = 1 if args.fast else CV_REPEATS
    n_bootstrap = 200 if args.fast else N_BOOTSTRAP
    seed = args.seed

    df = load_raw()
    X_train, X_test, y_train, y_test = split(df, seed=seed)

    print(
        f"rows {len(df)}  prevalence {prevalence(df['stroke']):.4f}\n"
        f"train {len(X_train)} ({int(y_train.sum())} positive)  "
        f"test {len(X_test)} ({int(y_test.sum())} positive)\n"
    )

    print(f"cross-validation ({CV_SPLITS} folds x {repeats} repeats, training portion only):")
    cv_results = run_cv_comparison(
        X_train, y_train, splits=CV_SPLITS, repeats=repeats, seed=seed
    )

    model_rows = [r for r in cv_results if r.strategy != "n/a"]
    best = max(model_rows, key=lambda r: r.pr_auc_mean)
    print(f"\nselected on CV PR-AUC: {best.estimator} / {best.strategy}")

    threshold, oof_op = choose_threshold(
        build_pipeline(best.estimator, best.strategy, seed),
        X_train,
        y_train,
        splits=CV_SPLITS,
        seed=seed,
    )
    print(
        f"threshold fixed on training out-of-fold predictions: {threshold:.4f} "
        f"(OOF sensitivity {oof_op['sensitivity']:.3f}, precision {oof_op['precision']:.3f})"
    )

    print("\nheld-out evaluation (touched once):")
    selected = evaluate_on_test(
        build_pipeline(best.estimator, best.strategy, seed),
        X_train,
        y_train,
        X_test,
        y_test,
        n_bootstrap=n_bootstrap,
        seed=seed,
        threshold=threshold,
    )
    selected["threshold_selection"] = {"oof_operating_point": oof_op}
    op = selected["operating_point"]
    print(
        f"  PR-AUC {selected['pr_auc']:.4f} "
        f"[{selected['pr_auc_ci95'][0]:.4f}, {selected['pr_auc_ci95'][1]:.4f}]  "
        f"(no-skill {selected['pr_auc_no_skill']:.4f})\n"
        f"  ROC-AUC {selected['roc_auc']:.4f}  Brier {selected['brier']:.4f}\n"
        f"  at the pre-chosen threshold: sensitivity {op['sensitivity']:.3f}, "
        f"precision {op['precision']:.4f}, specificity {op['specificity']:.4f}\n"
        f"  {op['tp']} caught / {op['fn']} missed / {op['fp']} false alarms "
        f"({op['tp'] + op['fp']} flagged of {selected['n']})"
    )

    # Every configuration is also scored on the held-out set so the comparison
    # table is like-for-like. Each one gets its *own* threshold fixed on its own
    # training out-of-fold predictions, so no row in the table is test-tuned.
    # Selection still used cross-validation only.
    print("\nscoring every configuration on the held-out set:")
    per_config: dict[str, dict] = {}
    candidates: list[tuple[str, object]] = [
        (f"{name}__{strategy}", build_pipeline(name, strategy, seed))
        for name in ESTIMATORS
        for strategy in STRATEGIES
    ]
    candidates += [
        ("baseline_majority", majority_pipeline(seed)),
        ("baseline_age_only", age_only_pipeline(seed)),
    ]

    for label, pipe in candidates:
        if label == "baseline_majority":
            # A constant predictor has no threshold that separates anything; a
            # sensitivity target is meaningless for it. Recorded without one.
            own_threshold = None
        else:
            own_threshold, _ = choose_threshold(
                clone(pipe), X_train, y_train, splits=CV_SPLITS, seed=seed
            )
        per_config[label] = evaluate_on_test(
            clone(pipe),
            X_train,
            y_train,
            X_test,
            y_test,
            n_bootstrap=n_bootstrap,
            seed=seed,
            threshold=own_threshold,
        )
        print(f"  {label}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "stroke_version": __version__,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "seed": seed,
            "cv_splits": CV_SPLITS,
            "cv_repeats": repeats,
            "n_bootstrap": n_bootstrap,
            "fast": args.fast,
        },
        "dataset": {
            "rows": int(len(df)),
            "prevalence": prevalence(df["stroke"]),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_test_positive": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()),
        },
        "target_sensitivity": TARGET_SENSITIVITY,
        "reported_threshold": threshold,
        "cv": [r.as_dict() for r in cv_results],
        "selected": {
            "estimator": best.estimator,
            "strategy": best.strategy,
            "test": selected,
        },
        "per_config_test": per_config,
        "web_config": {"estimator": WEB_CONFIG[0], "strategy": WEB_CONFIG[1]},
    }

    out = OUTPUT_DIR / "metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(OUTPUT_DIR.parent)}")

    for path in render_all(payload, FIGURE_DIR):
        print(f"wrote {path.relative_to(FIGURE_DIR.parents[1])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
