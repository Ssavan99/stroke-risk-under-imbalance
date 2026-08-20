"""Model pipelines.

Every transformation that learns anything from the data — the imputer's medians,
the scaler's mean and variance, the encoder's category list, and any resampler —
is a step inside the pipeline. Cross-validation therefore refits all of them on
each training fold, and none of them can see the validation fold.

This is the single structural difference between this code and the legacy code,
and it is the reason the numbers here are ~5x lower.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED

#: Imbalance strategies compared head to head. The legacy project assumed
#: oversampling was necessary; with the leak closed, that assumption is testable.
STRATEGIES = ("none", "class_weight", "smote")


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale and encode — all fitted per fold.

    ``add_indicator=True`` appends a binary column marking imputed values. For
    this dataset that is a real feature, not bookkeeping: missing BMI carries a
    19.9% stroke rate against 4.26% elsewhere, so the fact of the value being
    absent is informative and the model is allowed to use it.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" keeps inference working when a rare
            # category (gender == "Other", n=1) is absent from a training fold.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_estimator(name: str, strategy: str, seed: int = SEED):
    """Instantiate a classifier, applying class weighting if that is the strategy."""
    balanced = strategy == "class_weight"

    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample" if balanced else None,
            random_state=seed,
            # Parallelism is applied at the cross-validation level instead.
            n_jobs=1,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        )
    raise ValueError(f"unknown estimator: {name!r}")


ESTIMATORS = ("logistic_regression", "random_forest", "gradient_boosting")


def build_pipeline(name: str, strategy: str, seed: int = SEED) -> ImbPipeline:
    """Preprocessing + optional resampler + classifier, as one fittable object.

    An ``imblearn`` pipeline is used rather than a scikit-learn one because it is
    resampler-aware: SMOTE runs on the training fold only and is skipped entirely
    at predict time. Calling ``fit_resample`` outside a pipeline — as the legacy
    ``data_utils.py`` does — is what leaks the test set.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")

    steps: list[tuple[str, object]] = [("preprocess", build_preprocessor())]
    if strategy == "smote":
        steps.append(("resample", SMOTE(random_state=seed)))
    steps.append(("model", build_estimator(name, strategy, seed)))
    return ImbPipeline(steps=steps)
