# Legacy notebooks

These are the original notebooks from the project's first iteration, preserved
here unchanged except for a banner cell at the top of each one. They are kept
deliberately: the corrected analysis in
[`../01_why_the_old_numbers_were_wrong.ipynb`](../01_why_the_old_numbers_were_wrong.ipynb)
reproduces the numbers below and shows exactly where they come from, which is
only checkable if the originals are still here.

**Nothing in this folder is a current result.** The supported code path is
`src/stroke/`.

| Notebook | Reported | What is actually wrong |
|---|---|---|
| `log_regression.ipynb` | 95.7% accuracy | Predicts "no stroke" for **every** row — confusion matrix `[[4652, 0], [208, 0]]`, so the positive class is never predicted once. Also trains on 1% of the data (`test_size=0.99`). Cannot run at all on scikit-learn ≥ 1.2, which removed `plot_confusion_matrix`. |
| `model_runs.ipynb` | ~0.98 across 7 models | Resampling happens **before** the split, so the test set is leaked. Visible in the outputs: 958 negatives vs 922 positives, from a population that is 4.87% positive. |
| `Kmeans.ipynb` | 0.977 accuracy | Same resample-before-split leak. Also contains **no K-means** — it fits KNN and an SVM. Two cells raise live `InvalidIndexError`, and the decision-boundary plot is copied from an unrelated tutorial (its y-axis says "Estimated Salary"). |
| `dec_trees.ipynb` | 0.92 / 0.94 accuracy | Does not run. `X, y = get_smote_data()` unpacks a 4-tuple into two names, one cell holds a live `TypeError`, and it uses undefined names. Its stored outputs came from an older `data_utils.py`. |
| `EDA.ipynb` | — | Descriptive only, and still broadly sound. Makes no predictive-performance claims. Reads the CSV from the repository root, which has since moved to `data/`. |

## Supporting modules

`data_utils.py`, `data_cleaning.py` and `model_utils.py` are kept alongside the
notebooks so the imports above still resolve. They contain the defects the
notebooks inherit:

- `data_utils.py` — `fit_resample` on the full dataset, then `train_test_split`
  afterwards, in all three of `get_smote_data`, `get_adasyn_data` and
  `get_ros_data`. `get_clean_data` also fits a `StandardScaler` on the full
  dataset before any split, and calls `dropna()`.
- `data_cleaning.py` — same `dropna()`, plus label-encodes categorical columns by
  first-seen order, which invents an ordinal relationship between unordered
  categories such as `work_type`.
- `model_utils.py` — `print(model.best_params_)` on line 16 should be
  `grid.best_params_` and raises `AttributeError`; the path is dead because
  `model_params` is never passed. It also writes confusion-matrix PNGs into the
  working directory with no file extension.

## Why `dropna()` matters more than it looks

Both cleaning modules drop the 201 rows with a missing BMI. Those rows are not
missing at random:

| Subset | Rows | Strokes | Stroke rate |
|---|---|---|---|
| BMI present | 4,909 | 209 | 4.26% |
| BMI missing | 201 | 40 | **19.90%** |

Dropping them discards 40 of the dataset's 249 stroke cases — 16% of all
positives — and biases the sample toward healthier patients. The current pipeline
imputes BMI and adds an explicit missingness indicator instead.
