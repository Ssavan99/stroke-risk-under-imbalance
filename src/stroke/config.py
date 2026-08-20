"""Project-wide constants.

Everything that could silently change a reported number lives here, so that a
result can be traced back to the settings that produced it.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = ROOT / "outputs"

# --- reproducibility -------------------------------------------------------

SEED = 42

# --- data ------------------------------------------------------------------

TARGET = "stroke"
DROP_COLUMNS = ["id"]

NUMERIC_FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Fraction held out for the single final evaluation. Split before anything is
#: fitted, and touched exactly once.
TEST_SIZE = 0.2

# --- evaluation ------------------------------------------------------------

#: Cross-validation on the training portion only.
CV_SPLITS = 5
CV_REPEATS = 5

#: The operating point the project reports against.
#:
#: A screening-style tool is chosen by how many true cases it catches, not by how
#: often it is right, so the threshold is set by fixing sensitivity and then
#: reporting what that costs in precision and false alarms. 0.80 is a
#: deliberately ordinary choice: high enough to be a plausible screening target,
#: low enough that the resulting precision is not absurd.
TARGET_SENSITIVITY = 0.80

#: Bootstrap resamples used for confidence intervals on test-set metrics.
N_BOOTSTRAP = 2000

# --- parallelism -----------------------------------------------------------

#: Worker processes for cross-validation.
#:
#: Deliberately capped rather than -1. Under loky, -1 spawns one process per
#: core, each holding its own copy of a resampled fold; on a many-core machine
#: that has been observed to exhaust memory mid-run and, with
#: ``error_score="raise"``, take the whole job down after several minutes of
#: work. The entry point of this repository is the first thing a reader runs, so
#: it favours finishing over finishing fastest.
N_JOBS = min(4, (os.cpu_count() or 1))
