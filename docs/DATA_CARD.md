# Data card — stroke prediction dataset

## Source and attribution

`data/healthcare-dataset-stroke-data.csv` is the **Stroke Prediction Dataset**
published by **fedesoriano** on Kaggle. It has circulated widely since 2021 and is
the basis for a large number of public stroke-prediction projects.

- Publisher: fedesoriano, Kaggle
- Redistributed here as committed in the repository's first commit (2022-11-25)
- The Kaggle listing gives no explicit open licence; the source is credited and
  the file is unmodified. **Treat it as attribution-required and do not assume a
  permissive licence.**

The dataset's own listing states that the source is confidential and intended for
educational use. Its collection protocol, country, time period and inclusion
criteria are **not published**. That is a real limitation, not a footnote: without
knowing the sampling frame, the base rate below cannot be assumed to match any
actual population, and no result from this data transfers to a clinical setting.

## Shape

5,110 rows × 12 columns. One row per patient. No identifier is reused.

| Column | Type | Notes |
|---|---|---|
| `id` | int | Row identifier. Dropped before modelling — it carries no signal. |
| `gender` | categorical | `Female` 2,994 / `Male` 2,115 / `Other` **1** |
| `age` | float | 0.08 – 82. Includes infants and children. |
| `hypertension` | binary | 0 / 1 |
| `heart_disease` | binary | 0 / 1 |
| `ever_married` | categorical | `Yes` / `No` |
| `work_type` | categorical | `Private` 2,925 / `Self-employed` 819 / `children` 687 / `Govt_job` 657 / `Never_worked` 22 |
| `Residence_type` | categorical | `Rural` / `Urban` |
| `avg_glucose_level` | float | mg/dL |
| `bmi` | float | **201 missing**, encoded as the string `N/A` |
| `smoking_status` | categorical | `never smoked` 1,892 / **`Unknown` 1,544** / `formerly smoked` 885 / `smokes` 789 |
| `stroke` | binary | Target. 249 positive (**4.87%**) |

## Class balance

**249 of 5,110 rows are positive — a prevalence of 4.87%.**

This single number drives every methodological decision in this repository. A
classifier that always answers "no stroke" achieves **95.13% accuracy** while
being completely useless, so plain accuracy is never reported as a headline here.
See the README for the metrics used instead.

## Known issues and biases

**Missing BMI is informative.** The 201 rows with a missing BMI have a stroke rate
of **19.90%**, against **4.26%** among rows where BMI is present — roughly 4.7x.
Whatever caused the value to be missing is correlated with the outcome. Dropping
those rows, as the legacy code does, removes 40 of 249 positive cases and biases
the remaining sample toward healthier patients. The current pipeline imputes BMI
using the training fold's median and adds a `bmi_missing` indicator column so the
model can use the missingness itself as a feature.

**`smoking_status` is 30% `Unknown`.** This is treated as its own category rather
than imputed, because "we don't know" is genuinely different information from any
of the three known values, and its distribution is not random.

**`gender` has a single `Other` row.** It is retained. One-hot encoding is fitted
inside the pipeline with `handle_unknown="ignore"`, so a category absent from a
given training fold does not break inference.

**Children are included.** `work_type` contains 687 `children` and `age` starts at
0.08. Paediatric stroke has entirely different risk factors from adult stroke, so
the dataset mixes two populations under one label.

**Age dominates.** Age alone is a strong predictor of the target. The README
reports an age-only logistic regression as an explicit baseline for exactly this
reason — a multi-feature model that does not clearly beat it has not demonstrated
much.

**No external validation is possible.** There is one dataset, from one unknown
source. Every number in this repository is an internal estimate on a held-out
split of that single dataset, and the confidence intervals reported alongside them
are wide because ~50 positive cases land in any held-out test set.

## Use restrictions

This dataset supports a portfolio demonstration of evaluation methodology under
severe class imbalance. It does not support clinical claims, and no model trained
on it should be used to inform a decision about a real person.
