# Phase 2B — development baseline results

Completed (UTC): 2026-10-30T19:41:52.662136+00:00

Split lock published before fitting (UTC): 2026-10-30T19:41:07.721053+00:00

Split payload SHA-256: `76ba816f2c149c0eecdea931aaabab72605ca7f44fd98e1c1c91bec23a3abee8`.

The approved protocol and model parameters were locked before the first fit. Exactly nine fits were completed: three prespecified models × three development folds. No tuning, model-specific splits or post-result protocol changes were made.

The reserved final test was used only for membership/integrity checks. No test features, predictions or metrics were computed. It was exposed during prior EDA, so it is not an untouched exploratory holdout.

## Reproduce

```text
uv sync --locked --group baseline --python 3.11
uv run --locked --group baseline python scripts/lock_splits.py
uv run --locked --group baseline python scripts/run_baselines.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
```

The locking command verifies an existing matching lock and refuses to overwrite a different/corrupted lock. Detailed membership, input checksums, feature order and parameters are in `artifacts/splits/phase2b-v1.json` and `config/baseline-v1.json`. Machine-readable results are in [baseline-results.json](baseline-results.json). Development OOF predictions are local in `data/processed/baseline-oof.csv`.

## Fixed design and parameters

Groups are `(cooler_condition_pct, valve_condition_pct, accumulator_pressure_bar)`. Outer SGKF: five folds, shuffle=True, seed=2026; fold 2 (one-based) is the 339-cycle test (133/103/103). A 20-original-cycle-ID gap leaves 1,328 development cycles (846/246/236) and 538 excluded cycles. Inner SGKF uses three folds, the same seed and an additional inclusive gap 20 against each validation set. Before every fit, count, class, coverage, group, feature and original-ID gap checks passed. Inner training/validation sizes are 607/442, 576/457 and 607/429; minimum separation is 21. IDs are never compressed for gap calculations.

Features: 14 physical sensors × mean, population std (ddof=0), RMS, min, max = 70 float64 columns in a fixed sensor-major order. CE/CP/SE are excluded. Cycle IDs are a separate array, and target, condition labels, stability and groups never enter X. There are no fitted EDA values. Only development cycles were summarized.

```json
{
  "schema_version": 1,
  "feature_contract": "physical14-statistics5-v1",
  "models": {
    "dummy": {
      "strategy": "most_frequent"
    },
    "logistic_regression": {
      "scaler": {
        "with_mean": true,
        "with_std": true
      },
      "classifier": {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "tol": 0.0001,
        "fit_intercept": true,
        "class_weight": null,
        "random_state": 2026
      }
    },
    "random_forest": {
      "n_estimators": 200,
      "criterion": "gini",
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "max_features": "sqrt",
      "bootstrap": true,
      "class_weight": null,
      "random_state": 2026,
      "n_jobs": 1
    }
  },
  "thread_limit": 1,
  "primary_metric": "macro_f1",
  "class_order": [
    0,
    1,
    2
  ],
  "final_test_prediction_allowed": false
}
```

Logistic regression uses L2 regularization (`l1_ratio=0.0` in the locked scikit-learn 1.9 API) and a StandardScaler fitted inside each training-only Pipeline. No class weights were applied. RF uses 200 trees and one worker. All fits/predictions use a single numerical thread; effective library versions are `{"numpy": "2.4.6", "pandas": "2.3.3", "python": "3.11.15", "scikit_learn": "1.9.1"}`.

## Development CV and pooled OOF

Macro-F1 is the primary metric. Fold mean/std are unweighted (std ddof=0). Pooled metrics are recomputed from exactly one OOF prediction per development cycle; they are not averages of fold metrics.

| Model | Evaluation | Support (mean row: sum) | True classes 0/1/2 | Macro-F1 | Balanced accuracy | Class 2 recall |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | fold 1 | 442 | 290/81/71 | 0.264117 | 0.333333 | 0.000000 |
| dummy | fold 2 | 457 | 293/82/82 | 0.260444 | 0.333333 | 0.000000 |
| dummy | fold 3 | 429 | 263/83/83 | 0.253372 | 0.333333 | 0.000000 |
| dummy | fold mean ± std | 1328 | see folds | 0.259311 ± 0.004459 | 0.333333 ± 0.000000 | 0.000000 ± 0.000000 |
| dummy | pooled OOF | 1328 | 846/246/236 | 0.259430 | 0.333333 | 0.000000 |
| logistic_regression | fold 1 | 442 | 290/81/71 | 0.738355 | 0.776146 | 1.000000 |
| logistic_regression | fold 2 | 457 | 293/82/82 | 0.786529 | 0.784400 | 0.670732 |
| logistic_regression | fold 3 | 429 | 263/83/83 | 0.834559 | 0.834288 | 0.614458 |
| logistic_regression | fold mean ± std | 1328 | see folds | 0.786481 ± 0.039275 | 0.798278 ± 0.025685 | 0.761730 ± 0.170042 |
| logistic_regression | pooled OOF | 1328 | 846/246/236 | 0.793653 | 0.794917 | 0.750000 |
| random_forest | fold 1 | 442 | 290/81/71 | 0.719587 | 0.647598 | 0.436620 |
| random_forest | fold 2 | 457 | 293/82/82 | 0.813257 | 0.819029 | 0.573171 |
| random_forest | fold 3 | 429 | 263/83/83 | 0.812506 | 0.813994 | 0.951807 |
| random_forest | fold mean ± std | 1328 | see folds | 0.781784 ± 0.043980 | 0.760207 ± 0.079653 | 0.653866 ± 0.217927 |
| random_forest | pooled OOF | 1328 | 846/246/236 | 0.791460 | 0.763882 | 0.665254 |

## Per-class metrics

Undefined precision means no predictions for that class; its supported-class F1 is still zero when false negatives exist. Class order is always 0/1/2.

| Model | Evaluation | Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | fold 1 | 0 | 290 | 0.656109 | 1.000000 | 0.792350 |
| dummy | fold 1 | 1 | 81 | undefined | 0.000000 | 0.000000 |
| dummy | fold 1 | 2 | 71 | undefined | 0.000000 | 0.000000 |
| dummy | fold 2 | 0 | 293 | 0.641138 | 1.000000 | 0.781333 |
| dummy | fold 2 | 1 | 82 | undefined | 0.000000 | 0.000000 |
| dummy | fold 2 | 2 | 82 | undefined | 0.000000 | 0.000000 |
| dummy | fold 3 | 0 | 263 | 0.613054 | 1.000000 | 0.760116 |
| dummy | fold 3 | 1 | 83 | undefined | 0.000000 | 0.000000 |
| dummy | fold 3 | 2 | 83 | undefined | 0.000000 | 0.000000 |
| dummy | pooled OOF | 0 | 846 | 0.637048 | 1.000000 | 0.778289 |
| dummy | pooled OOF | 1 | 246 | undefined | 0.000000 | 0.000000 |
| dummy | pooled OOF | 2 | 236 | undefined | 0.000000 | 0.000000 |
| dummy | fold mean | 0 | 282.000 | 0.636767 | 1.000000 | 0.777933 |
| dummy | fold mean | 1 | 82.000 | undefined | 0.000000 | 0.000000 |
| dummy | fold mean | 2 | 78.667 | undefined | 0.000000 | 0.000000 |
| logistic_regression | fold 1 | 0 | 290 | 1.000000 | 0.982759 | 0.991304 |
| logistic_regression | fold 1 | 1 | 81 | 1.000000 | 0.345679 | 0.513761 |
| logistic_regression | fold 1 | 2 | 71 | 0.550388 | 1.000000 | 0.710000 |
| logistic_regression | fold 2 | 0 | 293 | 0.938567 | 0.938567 | 0.938567 |
| logistic_regression | fold 2 | 1 | 82 | 0.622449 | 0.743902 | 0.677778 |
| logistic_regression | fold 2 | 2 | 82 | 0.833333 | 0.670732 | 0.743243 |
| logistic_regression | fold 3 | 0 | 263 | 0.973684 | 0.984791 | 0.979206 |
| logistic_regression | fold 3 | 1 | 83 | 0.694444 | 0.903614 | 0.785340 |
| logistic_regression | fold 3 | 2 | 83 | 0.927273 | 0.614458 | 0.739130 |
| logistic_regression | pooled OOF | 0 | 846 | 0.970379 | 0.968085 | 0.969231 |
| logistic_regression | pooled OOF | 1 | 246 | 0.700855 | 0.666667 | 0.683333 |
| logistic_regression | pooled OOF | 2 | 236 | 0.708000 | 0.750000 | 0.728395 |
| logistic_regression | fold mean | 0 | 282.000 | 0.970750 | 0.968705 | 0.969692 |
| logistic_regression | fold mean | 1 | 82.000 | 0.772298 | 0.664399 | 0.658960 |
| logistic_regression | fold mean | 2 | 78.667 | 0.770331 | 0.761730 | 0.730791 |
| random_forest | fold 1 | 0 | 290 | 0.783784 | 1.000000 | 0.878788 |
| random_forest | fold 1 | 1 | 81 | 1.000000 | 0.506173 | 0.672131 |
| random_forest | fold 1 | 2 | 71 | 1.000000 | 0.436620 | 0.607843 |
| random_forest | fold 2 | 0 | 293 | 0.982699 | 0.969283 | 0.975945 |
| random_forest | fold 2 | 1 | 82 | 0.630252 | 0.914634 | 0.746269 |
| random_forest | fold 2 | 2 | 82 | 0.959184 | 0.573171 | 0.717557 |
| random_forest | fold 3 | 0 | 263 | 0.952727 | 0.996198 | 0.973978 |
| random_forest | fold 3 | 1 | 83 | 0.911111 | 0.493976 | 0.640625 |
| random_forest | fold 3 | 2 | 83 | 0.724771 | 0.951807 | 0.822917 |
| random_forest | pooled OOF | 0 | 846 | 0.895075 | 0.988180 | 0.939326 |
| random_forest | pooled OOF | 1 | 246 | 0.765854 | 0.638211 | 0.696231 |
| random_forest | pooled OOF | 2 | 236 | 0.830688 | 0.665254 | 0.738824 |
| random_forest | fold mean | 0 | 282.000 | 0.906403 | 0.988494 | 0.942904 |
| random_forest | fold mean | 1 | 82.000 | 0.847121 | 0.638261 | 0.686342 |
| random_forest | fold mean | 2 | 78.667 | 0.894651 | 0.653866 | 0.716106 |

## Confusion matrices

Rows = true class, columns = predicted class, order `[0, 1, 2]`. Pooled counts equal the sum of fold matrices.

### dummy — fold 1

```text
 290    0    0
  81    0    0
  71    0    0
```

### dummy — fold 2

```text
 293    0    0
  82    0    0
  82    0    0
```

### dummy — fold 3

```text
 263    0    0
  83    0    0
  83    0    0
```

### dummy — pooled OOF

```text
 846    0    0
 246    0    0
 236    0    0
```

### logistic_regression — fold 1

```text
 285    0    5
   0   28   53
   0    0   71
```

### logistic_regression — fold 2

```text
 275   10    8
  18   61    3
   0   27   55
```

### logistic_regression — fold 3

```text
 259    3    1
   5   75    3
   2   30   51
```

### logistic_regression — pooled OOF

```text
 819   13   14
  23  164   59
   2   57  177
```

### random_forest — fold 1

```text
 290    0    0
  40   41    0
  40    0   31
```

### random_forest — fold 2

```text
 284    9    0
   5   75    2
   0   35   47
```

### random_forest — fold 3

```text
 262    0    1
  13   41   29
   0    4   79
```

### random_forest — pooled OOF

```text
 836    9    1
  58  157   31
  40   39  157
```

## Stability subgroups

Flag 0 = stable; flag 1 = static conditions may not have been reached. Every validation case is retained. The same OOF predictions are stratified afterward; no subgroup-specific model is fitted. A fixed-three-class macro-F1/balanced accuracy is marked undefined if an actual class is absent. Per-class recall is undefined at zero true support, precision at zero predicted support, and F1 when both are zero. Missing values are JSON null, never silently replaced with zero. An unweighted mean is undefined if any constituent fold metric is undefined.

| Flag | Model | Evaluation | Support (mean row: sum) | Classes 0/1/2 | Macro-F1 | Balanced accuracy | Class 2 recall | Undefined metrics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dummy | fold 1 | 240 | 89/80/71 | 0.180344 | 0.333333 | 0.000000 | class_1.precision, class_2.precision |
| 0 | dummy | fold 2 | 240 | 80/80/80 | 0.166667 | 0.333333 | 0.000000 | class_1.precision, class_2.precision |
| 0 | dummy | fold 3 | 220 | 60/80/80 | 0.142857 | 0.333333 | 0.000000 | class_1.precision, class_2.precision |
| 0 | dummy | pooled OOF | 700 | 229/240/231 | 0.164334 | 0.333333 | 0.000000 | class_1.precision, class_2.precision |
| 0 | dummy | fold mean | 700 | see folds | 0.163289 | 0.333333 | 0.000000 | any undefined fold makes its mean undefined |
| 1 | dummy | fold 1 | 202 | 201/1/0 | undefined | undefined | undefined | class_1.precision, class_2.precision, class_2.recall, class_2.f1, macro_f1, balanced_accuracy, recall_class_2 |
| 1 | dummy | fold 2 | 217 | 213/2/2 | 0.330233 | 0.333333 | 0.000000 | class_1.precision, class_2.precision |
| 1 | dummy | fold 3 | 209 | 203/3/3 | 0.328479 | 0.333333 | 0.000000 | class_1.precision, class_2.precision |
| 1 | dummy | pooled OOF | 628 | 617/6/5 | 0.330388 | 0.333333 | 0.000000 | class_1.precision, class_2.precision |
| 1 | dummy | fold mean | 628 | see folds | undefined | undefined | undefined | any undefined fold makes its mean undefined |
| 0 | logistic_regression | fold 1 | 240 | 89/80/71 | 0.728590 | 0.760440 | 1.000000 | none |
| 0 | logistic_regression | fold 2 | 240 | 80/80/80 | 0.783164 | 0.783333 | 0.675000 | none |
| 0 | logistic_regression | fold 3 | 220 | 60/80/80 | 0.832693 | 0.840278 | 0.625000 | none |
| 0 | logistic_regression | pooled OOF | 700 | 229/240/231 | 0.787727 | 0.790680 | 0.757576 | none |
| 0 | logistic_regression | fold mean | 700 | see folds | 0.781482 | 0.794684 | 0.766667 | any undefined fold makes its mean undefined |
| 1 | logistic_regression | fold 1 | 202 | 201/1/0 | undefined | undefined | undefined | class_2.precision, class_2.recall, class_2.f1, macro_f1, balanced_accuracy, recall_class_2 |
| 1 | logistic_regression | fold 2 | 217 | 213/2/2 | 0.516780 | 0.812989 | 0.500000 | none |
| 1 | logistic_regression | fold 3 | 209 | 203/3/3 | 0.611514 | 0.661741 | 0.333333 | none |
| 1 | logistic_regression | pooled OOF | 628 | 617/6/5 | 0.570748 | 0.735800 | 0.400000 | none |
| 1 | logistic_regression | fold mean | 628 | see folds | undefined | undefined | undefined | any undefined fold makes its mean undefined |
| 0 | random_forest | fold 1 | 240 | 89/80/71 | 0.654811 | 0.645540 | 0.436620 | none |
| 0 | random_forest | fold 2 | 240 | 80/80/80 | 0.790252 | 0.791667 | 0.562500 | none |
| 0 | random_forest | fold 3 | 220 | 60/80/80 | 0.789182 | 0.811111 | 0.950000 | none |
| 0 | random_forest | pooled OOF | 700 | 229/240/231 | 0.745205 | 0.752069 | 0.658009 | none |
| 0 | random_forest | fold mean | 700 | see folds | 0.744748 | 0.749439 | 0.649707 | any undefined fold makes its mean undefined |
| 1 | random_forest | fold 1 | 202 | 201/1/0 | undefined | undefined | undefined | class_2.precision, class_2.recall, class_2.f1, macro_f1, balanced_accuracy, recall_class_2 |
| 1 | random_forest | fold 2 | 217 | 213/2/2 | 0.932549 | 0.998435 | 1.000000 | none |
| 1 | random_forest | fold 3 | 209 | 203/3/3 | 0.784895 | 0.777778 | 1.000000 | none |
| 1 | random_forest | pooled OOF | 628 | 617/6/5 | 0.878248 | 0.888349 | 1.000000 | none |
| 1 | random_forest | fold mean | 628 | see folds | undefined | undefined | undefined | any undefined fold makes its mean undefined |

Pooled subgroup per-class details are below; every fold's subgroup class metrics and confusion matrix are also retained in the companion JSON.

| Flag | Model | Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | dummy | 0 | 229 | 0.327143 | 1.000000 | 0.493003 |
| 0 | dummy | 1 | 240 | undefined | 0.000000 | 0.000000 |
| 0 | dummy | 2 | 231 | undefined | 0.000000 | 0.000000 |
| 1 | dummy | 0 | 617 | 0.982484 | 1.000000 | 0.991165 |
| 1 | dummy | 1 | 6 | undefined | 0.000000 | 0.000000 |
| 1 | dummy | 2 | 5 | undefined | 0.000000 | 0.000000 |
| 0 | logistic_regression | 0 | 229 | 0.900826 | 0.951965 | 0.925690 |
| 0 | logistic_regression | 1 | 240 | 0.726027 | 0.662500 | 0.692810 |
| 0 | logistic_regression | 2 | 231 | 0.732218 | 0.757576 | 0.744681 |
| 1 | logistic_regression | 0 | 617 | 0.998339 | 0.974068 | 0.986054 |
| 1 | logistic_regression | 1 | 6 | 0.333333 | 0.833333 | 0.476190 |
| 1 | logistic_regression | 2 | 5 | 0.181818 | 0.400000 | 0.250000 |
| 0 | random_forest | 0 | 229 | 0.694006 | 0.960699 | 0.805861 |
| 0 | random_forest | 1 | 240 | 0.765000 | 0.637500 | 0.695455 |
| 0 | random_forest | 2 | 231 | 0.830601 | 0.658009 | 0.734300 |
| 1 | random_forest | 0 | 617 | 0.998379 | 0.998379 | 0.998379 |
| 1 | random_forest | 1 | 6 | 0.800000 | 0.666667 | 0.727273 |
| 1 | random_forest | 2 | 5 | 0.833333 | 1.000000 | 0.909091 |

## Interpretation and limitations

Observed pooled development macro-F1: logistic regression 0.793653, random forest 0.791460. Their class 2 recall values are 0.750000 and 0.665254; class 2 precision values are 0.708000 and 0.830688. These expose a recall/precision tradeoff that a single macro-F1 value does not describe. Close development macro-F1 values do not establish a statistically reliable winning model.

The pooled flag-1 true class support is [617, 6, 5] in class order 0/1/2. Fold 1 has no flagged class-2 validation cases, so its corresponding recall and fixed-three-class aggregate metrics, and the aggregate fold means, remain undefined. The pooled flagged scores use only a handful of fault cases; they cannot establish robust performance on unstable faults.

These are first prespecified development baselines, not tuned or final-test results. Macro-F1, balanced accuracy and class 2 recall answer different questions; read the class supports and confusion matrices before preferring a model. No winning-model refit on all development data or final-test scoring was performed.

The test was exposed in Phase 2A EDA. Configuration groups are not verified independent experiments; gap 20 is not a validated decorrelation time. Purging changes training sizes/class mixtures and can remove rare unstable fault cases. Stability subgroup support is small and does not justify precise reliability claims. Fold standard deviation is descriptive, not an independent-sample confidence interval. No new-rig/general industrial performance claim is supported.

## Execution integrity

All expected split/count/group/gap/class checks passed before every fit. Raw-data checksums and original metadata were unchanged at completion. The feature matrix contained exactly 70 allowed sensor columns. See [validation-plan.md](validation-plan.md) for the frozen protocol and previous EDA exposure. Test predictions and metrics remain prohibited in Phase 2B.

## Completed validation

Verification (UTC): 2026-10-30T19:51:06.078759+00:00. Offline tests: 202 passed, 0 failures, 0 errors, 0 skipped. Ruff lint, Ruff format and uv lock checks passed. Independent scikit-learn recomputation of pooled metrics agreed with the saved OOF results. Fold/stability scores, original-ID membership/gaps, raw checksums, unchanged split-lock bytes and exclusion of final-test/outer-excluded OOF IDs were verified.
