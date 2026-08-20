"""Export the demo model as JSON the browser can evaluate directly.

Logistic regression reduces to a dot product, so the whole model is a list of
coefficients and the preprocessing constants needed to reach them. That keeps
the demo page fully static: no server, no runtime dependency, no hosting cost.

Two decisions worth stating, because both are about honesty rather than
convenience:

* The exported model is the **unweighted** logistic regression, not whichever
  configuration scores best. Class weighting and SMOTE leave discrimination
  roughly unchanged while multiplying every predicted probability several-fold —
  on the held-out set the weighted model's mean predicted risk is ~0.31 against a
  true rate of ~0.049. A page whose entire purpose is to show a number cannot
  show a number that is wrong by 6x.
* The exported payload carries the operating point and its precision with it, so
  the page cannot render a risk figure without also having the context that makes
  it interpretable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    SEED,
    TARGET_SENSITIVITY,
)
from .data import load_raw, split
from .evaluate import apply_threshold, discrimination, threshold_sweep
from .pipeline import build_pipeline

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


# Fitted floating-point values differ in the last one or two units in the last
# place between BLAS builds, so an exported payload built on one machine never
# matches one built on another byte for byte. Quantising every float on the way
# out makes the committed artefact reproducible anywhere, which is the only way
# a "regenerate and diff" staleness check can mean anything. Twelve significant
# digits is far finer than the demo displays and far finer than ``verify``'s
# 1e-9 tolerance, so it costs nothing real.
EXPORT_SIG_DIGITS = 12


def _quantize(obj, sig: int = EXPORT_SIG_DIGITS):
    """Recursively round every float so the serialised bytes are platform-stable."""
    if isinstance(obj, float):
        if not np.isfinite(obj) or obj == 0.0:
            return obj
        return float(f"%.{sig}g" % obj)
    if isinstance(obj, dict):
        return {k: _quantize(v, sig) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_quantize(v, sig) for v in obj]
    return obj


def _preprocessor_spec(pipe) -> dict:
    """Pull the fitted imputation/scaling/encoding constants out of the pipeline."""
    pre = pipe.named_steps["preprocess"]

    num_pipe = pre.named_transformers_["num"]
    imputer = num_pipe.named_steps["impute"]
    scaler = num_pipe.named_steps["scale"]

    cat_pipe = pre.named_transformers_["cat"]
    encoder = cat_pipe.named_steps["encode"]
    cat_imputer = cat_pipe.named_steps["impute"]

    # SimpleImputer(add_indicator=True) appends indicator columns for exactly the
    # features that had missing values at fit time, in feature order.
    indicator_features = [
        NUMERIC_FEATURES[i] for i in (imputer.indicator_.features_ or [])
    ]

    return {
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_fill": {
            f: float(v) for f, v in zip(NUMERIC_FEATURES, imputer.statistics_, strict=False)
        },
        "indicator_features": indicator_features,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categorical_fill": {
            f: str(v) for f, v in zip(CATEGORICAL_FEATURES, cat_imputer.statistics_, strict=False)
        },
        "categories": {
            f: [str(c) for c in cats]
            for f, cats in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=False)
        },
    }


def build_payload(seed: int = SEED) -> dict:
    """Fit the demo model and package everything the page needs."""
    df = load_raw()
    X_train, X_test, y_train, y_test = split(df, seed=seed)

    pipe = build_pipeline("logistic_regression", "none", seed)
    pipe.fit(X_train, y_train)

    prob_test = pipe.predict_proba(X_test)[:, 1]
    scores = discrimination(y_test, prob_test)

    # Threshold fixed on training out-of-fold predictions, consistent with the
    # rest of the project — never tuned on the held-out set.
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from .config import CV_SPLITS, N_JOBS
    from .evaluate import operating_point

    oof = cross_val_predict(
        build_pipeline("logistic_regression", "none", seed),
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed),
        method="predict_proba",
        n_jobs=N_JOBS,
    )[:, 1]
    threshold = operating_point(y_train, oof, TARGET_SENSITIVITY).threshold
    op = apply_threshold(y_test, prob_test, threshold)

    model = pipe.named_steps["model"]
    spec = _preprocessor_spec(pipe)

    return {
        "schema": 1,
        "model": "logistic_regression (unweighted)",
        "preprocess": spec,
        "coefficients": [float(v) for v in np.ravel(model.coef_)],
        "intercept": float(np.ravel(model.intercept_)[0]),
        "feature_names": [str(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()],
        "operating_point": op.as_dict(),
        # The page's threshold slider reports real held-out consequences rather
        # than an illustration, so the sweep travels with the model.
        "sweep": threshold_sweep(y_test, prob_test, n_points=60),
        "performance": {
            "pr_auc": scores["pr_auc"],
            "pr_auc_no_skill": scores["pr_auc_no_skill"],
            "roc_auc": scores["roc_auc"],
            "brier": scores["brier"],
            "prevalence": scores["prevalence"],
            "n_test": scores["n"],
            "n_test_positive": scores["n_positive"],
            "mean_predicted_risk": float(np.mean(prob_test)),
        },
        "dataset": {
            "rows": int(len(df)),
            "positives": int(df["stroke"].sum()),
            "prevalence": float(df["stroke"].mean()),
        },
    }


def verify(payload: dict, seed: int = SEED, n: int = 200, tol: float = 1e-9) -> float:
    """Re-score the held-out set with pure arithmetic and compare to scikit-learn.

    The page reimplements the model in JavaScript. This reimplements it in plain
    NumPy from the same exported payload, so a mismatch means the export itself
    is wrong rather than the JavaScript. The page has its own parity check
    against the vectors written by :func:`write_all`.
    """
    df = load_raw()
    X_train, X_test, y_train, y_test = split(df, seed=seed)
    pipe = build_pipeline("logistic_regression", "none", seed).fit(X_train, y_train)

    expected = pipe.predict_proba(X_test)[:, 1][:n]
    got = np.array([score_row(payload, row) for _, row in X_test.head(n).iterrows()])

    worst = float(np.max(np.abs(expected - got)))
    if worst > tol:
        raise AssertionError(f"export mismatch: max |delta| = {worst:.3e}")
    return worst


def score_row(payload: dict, row) -> float:
    """Reference implementation of exactly what the page's JavaScript does."""
    spec = payload["preprocess"]
    values: list[float] = []

    for i, f in enumerate(spec["numeric_features"]):
        v = row.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = spec["numeric_fill"][f]
        values.append((float(v) - spec["scaler_mean"][i]) / spec["scaler_scale"][i])

    # Indicator columns follow the numeric block, before scaling is re-applied to
    # them by the same StandardScaler, so they are scaled with the tail entries.
    offset = len(spec["numeric_features"])
    for j, f in enumerate(spec["indicator_features"]):
        v = row.get(f)
        missing = 1.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else 0.0
        idx = offset + j
        values.append((missing - spec["scaler_mean"][idx]) / spec["scaler_scale"][idx])

    for f in spec["categorical_features"]:
        v = row.get(f)
        v = spec["categorical_fill"][f] if v is None else str(v)
        for cat in spec["categories"][f]:
            values.append(1.0 if v == cat else 0.0)

    z = payload["intercept"] + float(np.dot(payload["coefficients"], values))
    return 1.0 / (1.0 + np.exp(-z))


def write_all(seed: int = SEED) -> list[Path]:
    """Write the model payload and a parity fixture the page checks itself against."""
    payload = build_payload(seed)
    worst = verify(payload, seed)

    df = load_raw()
    _, X_test, _, _ = split(df, seed=seed)
    pipe = build_pipeline("logistic_regression", "none", seed)
    X_train, _, y_train, _ = split(df, seed=seed)
    pipe.fit(X_train, y_train)

    sample = X_test.head(25)
    fixture = {
        "max_abs_error_python": worst,
        "cases": [
            {
                "input": {
                    k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                    for k, v in row.items()
                },
                "expected": float(p),
            }
            for (_, row), p in zip(
                sample.iterrows(), pipe.predict_proba(sample)[:, 1], strict=False
            )
        ],
    }

    payload = _quantize(payload)
    fixture = _quantize(fixture)

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for path in (WEB_DIR / "model.json", OUTPUT_DIR / "model.json"):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(path)

    fixture_path = WEB_DIR / "parity-fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    written.append(fixture_path)

    # The page is also loadable straight off the filesystem, where fetch() of a
    # local .json is blocked as a cross-origin request. Plain <script> tags are
    # not, so the same payloads are emitted as globals too.
    for path, name, obj in (
        (WEB_DIR / "model.js", "STROKE_MODEL", payload),
        (WEB_DIR / "parity-fixture.js", "STROKE_PARITY", fixture),
    ):
        path.write_text(
            f"window.{name} = {json.dumps(obj, indent=2)};\n", encoding="utf-8"
        )
        written.append(path)

    return written


def _compare(a, b, path: str, rtol: float, atol: float, out: list[str]) -> None:
    """Walk two payloads together, allowing floats to differ within tolerance."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            out.append(f"{path}: {a!r} vs {b!r}")
            return
        if not np.isclose(fa, fb, rtol=rtol, atol=atol, equal_nan=True):
            out.append(f"{path}: {fa!r} != {fb!r} (delta {abs(fa - fb):.3e})")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing from rebuilt payload")
            elif k not in b:
                out.append(f"{path}.{k}: missing from committed payload")
            else:
                _compare(a[k], b[k], f"{path}.{k}", rtol, atol, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _compare(x, y, f"{path}[{i}]", rtol, atol, out)
        return
    if a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def check_all(seed: int = SEED, rtol: float = 1e-6, atol: float = 1e-12) -> list[str]:
    """Rebuild the payloads and report how the committed ones differ, if at all.

    Returns a list of human-readable differences; empty means ``web/`` is current.
    A numeric comparison rather than a byte comparison, so that a rounding tie at
    the quantisation boundary reports as "identical within tolerance" instead of
    failing a build over the twelfth decimal place.
    """
    payload = _quantize(build_payload(seed))

    diffs: list[str] = []
    for name, rebuilt in (("model.json", payload),):
        path = WEB_DIR / name
        if not path.exists():
            diffs.append(f"{name}: not committed")
            continue
        committed = json.loads(path.read_text(encoding="utf-8"))
        _compare(rebuilt, committed, name, rtol, atol, diffs)
    return diffs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed web/ payload matches a fresh rebuild, without writing",
    )
    args = ap.parse_args()

    if args.check:
        differences = check_all()
        if differences:
            print("web/ is stale; differences beyond tolerance:")
            for d in differences[:20]:
                print(f"  {d}")
            if len(differences) > 20:
                print(f"  ... and {len(differences) - 20} more")
            raise SystemExit(1)
        print("web/ is current (within tolerance)")
    else:
        for p in write_all():
            print(f"wrote {p}")
