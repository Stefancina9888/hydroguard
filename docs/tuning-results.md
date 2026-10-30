# Phase 3A — bounded development tuning

**These development results were used for candidate selection. They are not an unbiased generalization estimate or final-test results.**

Status: completed; completed (UTC): 2026-10-30T20:12:07.137007+00:00.

Experiment preregistered (UTC): 2026-10-30T20:11:03.587964+00:00; candidate configuration created (UTC): 2026-10-30T20:11:03.572623+00:00.

Manifest payload SHA-256: `d2e44ed64cbb74a0da2e24ffc8b6cd0e95304ac7773c732405b0428616f1c584`.

The experiment references the existing split payload `76ba816f2c149c0eecdea931aaabab72605ca7f44fd98e1c1c91bec23a3abee8`. Neither the split nor the baseline configuration/results were rewritten. No new CV split was generated, no selected model was refitted on full development data, and no final-test or outer-excluded features/predictions/metrics were computed.

## Reproduce and audit

```text
uv run --locked --group baseline python scripts/run_tuning.py --prepare-only
uv run --locked --group baseline python scripts/run_tuning.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
```

Preparation publishes/verifies `config/tuning-v1.json` and `artifacts/experiments/phase3a-v1.json` without replacing an existing different lock. The training command requires both preregistrations. It reuses saved memberships in `artifacts/splits/phase2b-v1.json` and never calls a splitter or baseline membership generator. Source/config/input byte fingerprints are verified. No dependency or baseline files are updated.

The complete [JSON results](tuning-results.json) retain every candidate's full parameters, fold metrics, warnings, errors and timings. Local/Git-ignored `data/processed/tuning-runs/<run_id>/` retains this run's progress/results and development-only OOF predictions; `data/processed/tuning-oof.csv` is the latest OOF table for all candidates. A rerun gets a new journal rather than erasing the earlier local execution history.

## Prespecified candidates and selection rule

LR uses the unchanged baseline StandardScaler Pipeline and constructor parameters except the Cartesian product C=[0.01, 0.1, 1, 10, 100] and class_weight=[null, balanced]. Its C=1/unweighted candidate is also the baseline reference. No full-dataset scaler or EDA-fitted value is used. Balanced weights are learned by the estimator from training labels only.

RF keeps baseline parameters including 200 trees, one worker and seed 2026. The specified four-dimensional space has 96 configurations. Its exact baseline is excluded from a 95-item sampling pool and added separately; Python `random.Random(2026).sample` selects 12 distinct pool positions without replacement. Pool enumeration follows max_depth, min_samples_leaf, max_features, class_weight in the declared list order. Concrete sampled positions: `[15, 40, 64, 65, 82, 13, 28, 76, 79, 71, 53, 73]`. Thus RF has 13 evaluated configurations including the extra baseline reference, and LR has 10. Reference candidates remain eligible for selection.

The rule below was frozen before the first tuning fit. A lower simplicity key is preferred after both scores tie exactly. Simplicity is an explicit operational tie-break, not a proved ordering of predictive or compute complexity.

```json
{
  "primary": "Maximum unweighted mean of three fold macro-F1 scores",
  "secondary": "Maximum unweighted mean of three fold class-2 recalls",
  "ties": "Exact full-precision float equality; no rounding or tolerance",
  "lr_simplicity": "Lower C, then class_weight=None before balanced",
  "rf_simplicity": "Lower bounded max_depth (None last), larger min_samples_leaf, max_features sqrt before 0.5 before 1.0, then unweighted before balanced",
  "final_tie": "Lexicographically smaller candidate_id",
  "reference_candidates_eligible": true,
  "eligibility": "All three fits and predictions successful, finite selection metrics, and no ConvergenceWarning; other warnings are retained",
  "failures": "Structural/integrity violations stop the experiment. Model exceptions are recorded per fold and make the candidate ineligible; no parameter changes.",
  "baseline_reproduction": "Reference candidate OOF predictions must equal the saved baseline predictions when present, and reference metrics must agree within 1e-12"
}
```

| ID | Family | C | Class weight | Max depth | Min leaf | Max features | Baseline reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_01 | logistic_regression | 0.01 | None | — | — | — | False |
| lr_02 | logistic_regression | 0.01 | balanced | — | — | — | False |
| lr_03 | logistic_regression | 0.1 | None | — | — | — | False |
| lr_04 | logistic_regression | 0.1 | balanced | — | — | — | False |
| lr_05 | logistic_regression | 1 | None | — | — | — | True |
| lr_06 | logistic_regression | 1 | balanced | — | — | — | False |
| lr_07 | logistic_regression | 10 | None | — | — | — | False |
| lr_08 | logistic_regression | 10 | balanced | — | — | — | False |
| lr_09 | logistic_regression | 100 | None | — | — | — | False |
| lr_10 | logistic_regression | 100 | balanced | — | — | — | False |
| rf_baseline | random_forest | — | None | None | 1 | sqrt | True |
| rf_01 | random_forest | — | None | None | 4 | 1.0 | False |
| rf_02 | random_forest | — | balanced | 6 | 4 | 1.0 | False |
| rf_03 | random_forest | — | balanced | 12 | 4 | 1.0 | False |
| rf_04 | random_forest | — | None | 12 | 8 | sqrt | False |
| rf_05 | random_forest | — | balanced | 20 | 2 | 1.0 | False |
| rf_06 | random_forest | — | None | None | 4 | 0.5 | False |
| rf_07 | random_forest | — | balanced | 6 | 1 | 1.0 | False |
| rf_08 | random_forest | — | balanced | 20 | 1 | 1.0 | False |
| rf_09 | random_forest | — | None | 20 | 2 | 0.5 | False |
| rf_10 | random_forest | — | None | 20 | 1 | sqrt | False |
| rf_11 | random_forest | — | None | 12 | 2 | sqrt | False |
| rf_12 | random_forest | — | None | 20 | 1 | 0.5 | False |

The remaining parameters are copied from the unchanged baseline config and the entire resolved parameter dictionaries are in the locked tuning config. All candidates use the same 70-feature extractor and the same three original-ID gap-20 inner folds (607/442, 576/457, 607/429 training/validation cycles). All count, class, group and gap checks run before each fit. Numerical thread count is one. No estimator is retained or fitted after selection.

## Candidate ranking and execution

Classifier fits attempted/completed: 69/69 (planned 69). Recorded warnings: 0. Run wall time, including verification/extraction: 27.723 seconds. Fit/predict times use a monotonic clock; candidate elapsed times also include preflight and journal overhead. Classifier fit count excludes internal trees and training-only scaler fits.

| Candidate | Eligibility | Mean fold macro-F1 | Mean fold class-2 recall | Warnings | Elapsed seconds | Simplicity key |
| --- | --- | --- | --- | --- | --- | --- |
| lr_01 | eligible | 0.599789 | 0.461322 | 0 | 0.128 | [0.01, 0] |
| lr_02 | eligible | 0.674683 | 0.676609 | 0 | 0.108 | [0.01, 1] |
| lr_03 | eligible | 0.714988 | 0.680723 | 0 | 0.111 | [0.1, 0] |
| lr_04 | eligible | 0.749575 | 0.721324 | 0 | 0.123 | [0.1, 1] |
| lr_05 | eligible | 0.786481 | 0.761730 | 0 | 0.131 | [1, 0] |
| lr_06 | eligible | 0.800643 | 0.765942 | 0 | 0.142 | [1, 1] |
| lr_07 | eligible | 0.815738 | 0.782153 | 0 | 0.138 | [10, 0] |
| lr_08 | eligible | 0.829028 | 0.806396 | 0 | 0.148 | [10, 1] |
| lr_09 | eligible | 0.837084 | 0.822803 | 0 | 0.140 | [100, 0] |
| lr_10 | eligible | 0.829488 | 0.838868 | 0 | 0.142 | [100, 1] |
| rf_baseline | eligible | 0.781784 | 0.653866 | 0 | 0.655 | [2147483647, -1, 0, 0] |
| rf_01 | eligible | 0.777088 | 0.759547 | 0 | 1.372 | [2147483647, -4, 2, 0] |
| rf_02 | eligible | 0.780022 | 0.781614 | 0 | 1.386 | [6, -4, 2, 1] |
| rf_03 | eligible | 0.780022 | 0.781614 | 0 | 1.414 | [12, -4, 2, 1] |
| rf_04 | eligible | 0.772968 | 0.653964 | 0 | 0.680 | [12, -8, 0, 0] |
| rf_05 | eligible | 0.780022 | 0.781614 | 0 | 1.436 | [20, -2, 2, 1] |
| rf_06 | eligible | 0.781389 | 0.735401 | 0 | 0.981 | [2147483647, -4, 1, 0] |
| rf_07 | eligible | 0.784254 | 0.785630 | 0 | 1.447 | [6, -1, 2, 1] |
| rf_08 | eligible | 0.784254 | 0.785630 | 0 | 1.453 | [20, -1, 2, 1] |
| rf_09 | eligible | 0.783190 | 0.739417 | 0 | 1.016 | [20, -2, 1, 0] |
| rf_10 | eligible | 0.781784 | 0.653866 | 0 | 0.694 | [20, -1, 0, 0] |
| rf_11 | eligible | 0.783940 | 0.657931 | 0 | 0.694 | [12, -2, 0, 0] |
| rf_12 | eligible | 0.785987 | 0.743433 | 0 | 1.028 | [20, -1, 1, 0] |

## Every candidate and fold

| Candidate | Fold | Macro-F1 | Balanced accuracy | Class-2 recall | Fit seconds | Predict seconds | Warnings | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_01 | 1 | 0.695544 | 0.737950 | 0.971831 | 0.0218 | 0.0007 | none | none |
| lr_01 | 2 | 0.566294 | 0.555454 | 0.207317 | 0.0060 | 0.0007 | none | none |
| lr_01 | 3 | 0.537530 | 0.516805 | 0.204819 | 0.0058 | 0.0007 | none | none |
| lr_02 | 1 | 0.581043 | 0.653753 | 1.000000 | 0.0085 | 0.0007 | none | none |
| lr_02 | 2 | 0.698148 | 0.751741 | 0.475610 | 0.0065 | 0.0007 | none | none |
| lr_02 | 3 | 0.744858 | 0.758853 | 0.554217 | 0.0071 | 0.0007 | none | none |
| lr_03 | 1 | 0.714842 | 0.759685 | 1.000000 | 0.0078 | 0.0007 | none | none |
| lr_03 | 2 | 0.709130 | 0.704889 | 0.500000 | 0.0076 | 0.0007 | none | none |
| lr_03 | 3 | 0.720992 | 0.693512 | 0.542169 | 0.0083 | 0.0007 | none | none |
| lr_04 | 1 | 0.686788 | 0.742444 | 1.000000 | 0.0103 | 0.0006 | none | none |
| lr_04 | 2 | 0.791252 | 0.817087 | 0.609756 | 0.0097 | 0.0007 | none | none |
| lr_04 | 3 | 0.770684 | 0.787805 | 0.554217 | 0.0102 | 0.0006 | none | none |
| lr_05 | 1 | 0.738355 | 0.776146 | 1.000000 | 0.0091 | 0.0007 | none | none |
| lr_05 | 2 | 0.786529 | 0.784400 | 0.670732 | 0.0120 | 0.0006 | none | none |
| lr_05 | 3 | 0.834559 | 0.834288 | 0.614458 | 0.0128 | 0.0007 | none | none |
| lr_06 | 1 | 0.755229 | 0.788492 | 1.000000 | 0.0119 | 0.0007 | none | none |
| lr_06 | 2 | 0.825366 | 0.839035 | 0.719512 | 0.0142 | 0.0007 | none | none |
| lr_06 | 3 | 0.821335 | 0.830699 | 0.578313 | 0.0146 | 0.0007 | none | none |
| lr_07 | 1 | 0.802594 | 0.825529 | 1.000000 | 0.0065 | 0.0007 | none | none |
| lr_07 | 2 | 0.813078 | 0.820500 | 0.756098 | 0.0131 | 0.0006 | none | none |
| lr_07 | 3 | 0.831541 | 0.831539 | 0.590361 | 0.0131 | 0.0007 | none | none |
| lr_08 | 1 | 0.807598 | 0.829644 | 1.000000 | 0.0076 | 0.0007 | none | none |
| lr_08 | 2 | 0.834612 | 0.850745 | 0.792683 | 0.0129 | 0.0007 | none | none |
| lr_08 | 3 | 0.844874 | 0.850566 | 0.626506 | 0.0169 | 0.0007 | none | none |
| lr_09 | 1 | 0.807598 | 0.829644 | 1.000000 | 0.0052 | 0.0007 | none | none |
| lr_09 | 2 | 0.876335 | 0.898388 | 0.878049 | 0.0077 | 0.0007 | none | none |
| lr_09 | 3 | 0.827320 | 0.822026 | 0.590361 | 0.0092 | 0.0007 | none | none |
| lr_10 | 1 | 0.774530 | 0.803803 | 1.000000 | 0.0051 | 0.0007 | none | none |
| lr_10 | 2 | 0.859832 | 0.889287 | 0.878049 | 0.0084 | 0.0006 | none | none |
| lr_10 | 3 | 0.854103 | 0.852887 | 0.638554 | 0.0109 | 0.0008 | none | none |
| rf_baseline | 1 | 0.719587 | 0.647598 | 0.436620 | 0.1563 | 0.0062 | none | none |
| rf_baseline | 2 | 0.813257 | 0.819029 | 0.573171 | 0.1748 | 0.0064 | none | none |
| rf_baseline | 3 | 0.812506 | 0.813994 | 0.951807 | 0.1822 | 0.0063 | none | none |
| rf_01 | 1 | 0.741330 | 0.779927 | 0.957746 | 0.3639 | 0.0059 | none | none |
| rf_01 | 2 | 0.783184 | 0.778379 | 0.634146 | 0.4058 | 0.0061 | none | none |
| rf_01 | 3 | 0.806751 | 0.789897 | 0.686747 | 0.4583 | 0.0060 | none | none |
| rf_02 | 1 | 0.725428 | 0.782517 | 1.000000 | 0.3741 | 0.0060 | none | none |
| rf_02 | 2 | 0.782767 | 0.782444 | 0.621951 | 0.4179 | 0.0064 | none | none |
| rf_02 | 3 | 0.831870 | 0.818010 | 0.722892 | 0.4458 | 0.0061 | none | none |
| rf_03 | 1 | 0.725428 | 0.782517 | 1.000000 | 0.3732 | 0.0058 | none | none |
| rf_03 | 2 | 0.782767 | 0.782444 | 0.621951 | 0.4162 | 0.0060 | none | none |
| rf_03 | 3 | 0.831870 | 0.818010 | 0.722892 | 0.4689 | 0.0063 | none | none |
| rf_04 | 1 | 0.719587 | 0.647598 | 0.436620 | 0.1590 | 0.0063 | none | none |
| rf_04 | 2 | 0.813582 | 0.814964 | 0.597561 | 0.1746 | 0.0064 | none | none |
| rf_04 | 3 | 0.785734 | 0.781865 | 0.927711 | 0.1809 | 0.0069 | none | none |
| rf_05 | 1 | 0.725428 | 0.782517 | 1.000000 | 0.3844 | 0.0059 | none | none |
| rf_05 | 2 | 0.782767 | 0.782444 | 0.621951 | 0.4275 | 0.0061 | none | none |
| rf_05 | 3 | 0.831870 | 0.818010 | 0.722892 | 0.4631 | 0.0063 | none | none |
| rf_06 | 1 | 0.743156 | 0.781077 | 0.957746 | 0.2453 | 0.0060 | none | none |
| rf_06 | 2 | 0.786070 | 0.782444 | 0.621951 | 0.2774 | 0.0060 | none | none |
| rf_06 | 3 | 0.814943 | 0.801945 | 0.626506 | 0.2931 | 0.0061 | none | none |
| rf_07 | 1 | 0.725428 | 0.782517 | 1.000000 | 0.3796 | 0.0059 | none | none |
| rf_07 | 2 | 0.782767 | 0.782444 | 0.621951 | 0.4318 | 0.0061 | none | none |
| rf_07 | 3 | 0.844568 | 0.830058 | 0.734940 | 0.4662 | 0.0062 | none | none |
| rf_08 | 1 | 0.725428 | 0.782517 | 1.000000 | 0.3844 | 0.0059 | none | none |
| rf_08 | 2 | 0.782767 | 0.782444 | 0.621951 | 0.4306 | 0.0060 | none | none |
| rf_08 | 3 | 0.844568 | 0.830058 | 0.734940 | 0.4631 | 0.0061 | none | none |
| rf_09 | 1 | 0.752489 | 0.786824 | 0.957746 | 0.2479 | 0.0059 | none | none |
| rf_09 | 2 | 0.781980 | 0.778379 | 0.621951 | 0.2853 | 0.0061 | none | none |
| rf_09 | 3 | 0.815102 | 0.801945 | 0.638554 | 0.3034 | 0.0062 | none | none |
| rf_10 | 1 | 0.719587 | 0.647598 | 0.436620 | 0.1567 | 0.0061 | none | none |
| rf_10 | 2 | 0.813257 | 0.819029 | 0.573171 | 0.1738 | 0.0062 | none | none |
| rf_10 | 3 | 0.812506 | 0.813994 | 0.951807 | 0.1816 | 0.0062 | none | none |
| rf_11 | 1 | 0.719587 | 0.647598 | 0.436620 | 0.1554 | 0.0061 | none | none |
| rf_11 | 2 | 0.817173 | 0.823094 | 0.585366 | 0.1737 | 0.0062 | none | none |
| rf_11 | 3 | 0.815060 | 0.813994 | 0.951807 | 0.1794 | 0.0062 | none | none |
| rf_12 | 1 | 0.752489 | 0.786824 | 0.957746 | 0.2475 | 0.0060 | none | none |
| rf_12 | 2 | 0.786070 | 0.782444 | 0.621951 | 0.2878 | 0.0061 | none | none |
| rf_12 | 3 | 0.819402 | 0.805961 | 0.650602 | 0.3046 | 0.0061 | none | none |

A ConvergenceWarning or failed fit makes a candidate ineligible by the preregistered rule; any structural/integrity violation aborts the run. No search-space expansion or warning-driven parameter adjustment is allowed.

## Selected family configurations and baseline comparison

Selection uses unweighted fold means, not pooled OOF macro-F1. Pooled metrics below are recomputed from cached development predictions without additional fits. All 1,328 development cycles have one prediction per candidate. Baseline reference candidates reproduced the saved baseline metrics and, when available, exact saved baseline OOF labels.

### logistic_regression: lr_09

```json
{
  "scaler": {
    "with_mean": true,
    "with_std": true
  },
  "classifier": {
    "C": 100,
    "l1_ratio": 0.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "tol": 0.0001,
    "fit_intercept": true,
    "class_weight": null,
    "random_state": 2026
  }
}
```

Change from baseline mean fold macro-F1: +0.050603; change in pooled OOF macro-F1: +0.047368.

### random_forest: rf_12

```json
{
  "n_estimators": 200,
  "criterion": "gini",
  "max_depth": 20,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": 0.5,
  "bootstrap": true,
  "class_weight": null,
  "random_state": 2026,
  "n_jobs": 1
}
```

Change from baseline mean fold macro-F1: +0.004203; change in pooled OOF macro-F1: -0.003006.

| Family | Configuration | Mean fold macro-F1 | Mean fold class-2 recall | OOF macro-F1 | OOF balanced accuracy | OOF class-2 recall |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | Baseline | 0.786481 | 0.761730 | 0.793653 | 0.794917 | 0.750000 |
| logistic_regression | Selected | 0.837084 | 0.822803 | 0.841021 | 0.847048 | 0.813559 |
| random_forest | Baseline | 0.781784 | 0.653866 | 0.791460 | 0.763882 | 0.665254 |
| random_forest | Selected | 0.785987 | 0.743433 | 0.788454 | 0.788210 | 0.733051 |

## Selected OOF confusion matrices and class metrics

Confusion rows are true classes and columns predictions, fixed order [0,1,2]. OOF support is 846/246/236. Fold spread is descriptive, not an IID confidence interval.

### logistic_regression

```text
 815   21   10
  12  188   46
   2   42  192
```

### random_forest

```text
 806   34    6
  34  167   45
   3   60  173
```

| Family | Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0 | 846 | 0.983112 | 0.963357 | 0.973134 |
| logistic_regression | 1 | 246 | 0.749004 | 0.764228 | 0.756539 |
| logistic_regression | 2 | 236 | 0.774194 | 0.813559 | 0.793388 |
| random_forest | 0 | 846 | 0.956109 | 0.952719 | 0.954411 |
| random_forest | 1 | 246 | 0.639847 | 0.678862 | 0.658777 |
| random_forest | 2 | 236 | 0.772321 | 0.733051 | 0.752174 |

## Selected stability subgroups

The same selected OOF predictions are stratified by flag, without subgroup fitting or cycle filtering. Flag 0 means stable; flag 1 means static conditions might not have been reached. Undefined precision/recall/F1 and fixed-three-class subgroup macro scores follow the unchanged Phase 2B metric policy. Fold 1 has no flagged class-2 validation cases; affected fold metrics and their means are undefined, not silently omitted. JSON retains all fold subgroup class metrics and confusion matrices.

| Family | Flag | Evaluation | Support (mean: sum) | Classes 0/1/2 | Macro-F1 | Balanced accuracy | Class-2 recall | Undefined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0 | fold 1 | 240 | 89/80/71 | 0.799041 | 0.814607 | 1.000000 | none |
| logistic_regression | 0 | fold 2 | 240 | 80/80/80 | 0.889108 | 0.887500 | 0.875000 | none |
| logistic_regression | 0 | fold 3 | 220 | 60/80/80 | 0.804728 | 0.815278 | 0.587500 | none |
| logistic_regression | 0 | pooled OOF | 700 | 229/240/231 | 0.835485 | 0.835561 | 0.813853 | none |
| logistic_regression | 0 | fold mean | 700 | see folds | 0.830959 | 0.839128 | 0.820833 | undefined if any fold is undefined |
| logistic_regression | 1 | fold 1 | 202 | 201/1/0 | undefined | undefined | undefined | class_2.precision, class_2.recall, class_2.f1, macro_f1, balanced_accuracy, recall_class_2 |
| logistic_regression | 1 | fold 2 | 217 | 213/2/2 | 0.570325 | 0.974961 | 1.000000 | none |
| logistic_regression | 1 | fold 3 | 209 | 203/3/3 | 0.932514 | 0.888889 | 0.666667 | none |
| logistic_regression | 1 | pooled OOF | 628 | 617/6/5 | 0.700480 | 0.924689 | 0.800000 | none |
| logistic_regression | 1 | fold mean | 628 | see folds | undefined | undefined | undefined | undefined if any fold is undefined |
| random_forest | 0 | fold 1 | 240 | 89/80/71 | 0.776521 | 0.793032 | 0.957746 | none |
| random_forest | 0 | fold 2 | 240 | 80/80/80 | 0.759006 | 0.758333 | 0.625000 | none |
| random_forest | 0 | fold 3 | 220 | 60/80/80 | 0.787455 | 0.798611 | 0.637500 | none |
| random_forest | 0 | pooled OOF | 700 | 229/240/231 | 0.776186 | 0.778911 | 0.731602 | none |
| random_forest | 0 | fold mean | 700 | see folds | 0.774327 | 0.783325 | 0.740082 | undefined if any fold is undefined |
| random_forest | 1 | fold 1 | 202 | 201/1/0 | undefined | undefined | undefined | class_2.precision, class_2.recall, class_2.f1, macro_f1, balanced_accuracy, recall_class_2 |
| random_forest | 1 | fold 2 | 217 | 213/2/2 | 0.776993 | 0.831768 | 0.500000 | none |
| random_forest | 1 | fold 3 | 209 | 203/3/3 | 0.932514 | 0.888889 | 1.000000 | none |
| random_forest | 1 | pooled OOF | 628 | 617/6/5 | 0.715341 | 0.864812 | 0.800000 | none |
| random_forest | 1 | fold mean | 628 | see folds | undefined | undefined | undefined | undefined if any fold is undefined |

| Family | Flag | Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0 | 0 | 229 | 0.942731 | 0.934498 | 0.938596 |
| logistic_regression | 0 | 1 | 240 | 0.781116 | 0.758333 | 0.769556 |
| logistic_regression | 0 | 2 | 231 | 0.783333 | 0.813853 | 0.798301 |
| logistic_regression | 1 | 0 | 617 | 0.998339 | 0.974068 | 0.986054 |
| logistic_regression | 1 | 1 | 6 | 0.333333 | 1.000000 | 0.500000 |
| logistic_regression | 1 | 2 | 5 | 0.500000 | 0.800000 | 0.615385 |
| random_forest | 0 | 0 | 229 | 0.855422 | 0.930131 | 0.891213 |
| random_forest | 0 | 1 | 240 | 0.701299 | 0.675000 | 0.687898 |
| random_forest | 0 | 2 | 231 | 0.768182 | 0.731602 | 0.749446 |
| random_forest | 1 | 0 | 617 | 0.998316 | 0.961102 | 0.979356 |
| random_forest | 1 | 1 | 6 | 0.166667 | 0.833333 | 0.277778 |
| random_forest | 1 | 2 | 5 | 1.000000 | 0.800000 | 0.888889 |

Baseline pooled subgroup metrics are also retained in the JSON baseline comparison; the original [baseline report](baseline-results.md) remains unchanged. Flagged OOF support is only 617/6/5 for classes 0/1/2; rare-stratum scores are not precise reliability estimates.

## Limitations and review boundary

The same development folds were used to select configurations, so their selected scores are selection-affected and may be optimistic. They are not nested-CV or independent external validation estimates. No significance or deployment claim follows from a small development-score difference. The reserved final test was previously exposed in Phase 2A EDA, but remains unused for Phase 3A features, predictions and metrics. Group separation and gap 20 do not establish independent experiments or a physical decorrelation time. Shared rig/thermal trajectory and purging-induced class mixtures remain limitations.

All raw inputs, original IDs/order, baseline configuration/results, split lock, uv.lock and the 70-feature extractor were verified unchanged. No full-development winner refit, new feature, XGBoost, API, Dockerfile, commit or push is part of this phase. Stop here for review.

## Executed checks

220 offline tests passed; 0 failures, 0 errors and 0 skipped. Ruff lint/format and uv lock checks passed. Candidate selection, every fold metric, selected OOF/stability results, 69-fit accounting, no test/excluded OOF IDs and protected-file checksums were independently verified.

## Observed mean-versus-pooled distinction

The selected RF changes mean fold macro-F1 by +0.004203, while pooled OOF macro-F1 changes by -0.003006. The positive selection-criterion change and negative pooled-score change are both reported. The preregistered mean-fold rule was followed; the rule was not changed to optimize pooled performance after seeing this result.

Method references: official [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) documentation.
