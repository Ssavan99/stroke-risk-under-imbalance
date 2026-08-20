"""Loading and splitting.

The one rule this module exists to enforce: the split happens before anything is
fitted. No imputation, scaling, encoding or resampling occurs here — all of that
lives inside the pipeline so it can only ever see a training fold.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DROP_COLUMNS,
    FEATURES,
    SEED,
    TARGET,
    TEST_SIZE,
)


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV with no cleaning beyond typing.

    ``bmi`` is stored as the string ``N/A`` for missing values, which pandas
    already treats as null. Nothing is dropped: the rows with a missing BMI have
    a stroke rate roughly 4.7x the rest of the dataset, so discarding them —
    as the legacy code does — removes 16% of all positive cases and biases the
    sample. See ``docs/DATA_CARD.md``.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string")
    return df


def split(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split, performed before any fitting.

    Stratification keeps the held-out positive rate equal to the population rate.
    That is the property the legacy pipeline destroyed: it resampled to balance
    first and split second, producing a test set that was ~50% positive and
    therefore meaningless.
    """
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def prevalence(y) -> float:
    """Positive rate. Also the no-skill baseline for average precision."""
    return float(pd.Series(y).mean())
