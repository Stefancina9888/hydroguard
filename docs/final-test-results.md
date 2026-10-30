# Phase 3B — fixed Logistic Regression final-test evaluation

**Reserved-test results for the single model fixed before test prediction. Development selection results are reported separately and are not pooled with the test.**

Completed (UTC): 2026-10-30T20:49:03.274370+00:00. Evaluation elapsed: 5.184 seconds.

Phase 3B preregistration (UTC): 2026-10-30T20:44:50.985301+00:00; test-access receipt (UTC): 2026-10-30T20:48:58.086626+00:00.

Manifest payload SHA-256: `47b31e727a99fefa6945d326cce568b809b8d47fadca28c8911c4128852431f4`.

## Fixed model and chronological procedure

The project owner selected Logistic Regression `lr_09` (C=100, class_weight=None) from the locked Phase 3A candidate configuration before this evaluation. No RF or alternative model was evaluated on the reserved test. All other constructor parameters and the StandardScaler Pipeline remained unchanged:

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

A separate immutable `artifacts/experiments/phase3b-v1.json` was published before fitting and before any test sensor feature/prediction access. It references the existing split, tuning configuration/manifest/results and 70-feature contract. One fresh complete Pipeline was fitted on exactly the 1,328 locked development cycles (846/246/236). StandardScaler observed only these training rows. The 538 outer-excluded cycles were neither fitted nor evaluated. Original IDs, row order, group definitions and gap rules were retained.

Fit completed (UTC): 2026-10-30T20:48:42.194949+00:00; fit duration: 0.055515 seconds; scaler training samples: 1328; LR iterations: [114].

After training and bundle verification, only this saved model received the 339 reserved cycles (133/103/103). No parameter, feature, class decision, threshold, split or protocol was changed after observing the final-test results. No training on all 2,205 cycles, further tuning, calibration, API or Docker was performed.

## Final reserved-test results

| Cases | True classes 0/1/2 | Macro-F1 | Balanced accuracy | Class-2 recall |
| --- | --- | --- | --- | --- |
| 339 | 133/103/103 | 0.675047 | 0.724919 | 1.000000 |

Macro-F1 is primary. Undefined values are shown explicitly rather than silently dropped. Precision uses predicted support and recall uses true support.

| Class | True support | Predicted support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| 0 | 133 | 205 | 0.648780 | 1.000000 | 0.786982 |
| 1 | 103 | 18 | 1.000000 | 0.174757 | 0.297521 |
| 2 | 103 | 116 | 0.887931 | 1.000000 | 0.940639 |

Confusion matrix: rows are true classes, columns predictions, order `[0,1,2]`.

```text
 133    0    0
  72   18   13
   0    0  103
```

## Test stability subgroups

Flag 0 means stable; flag 1 means static conditions might not have been reached. The same test predictions are stratified afterward: no subgroup fitting/filtering or alternative model evaluation. Class denominators and undefined metrics are explicit.

### Flag 0 confusion matrix

```text
 100    0    0
  70   17   13
   0    0  100
```

### Flag 1 confusion matrix

```text
  33    0    0
   2    1    0
   0    0    3
```

| Flag | Cases | Classes 0/1/2 | Macro-F1 | Balanced accuracy | Class-2 recall | Undefined |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 300 | 100/100/100 | 0.656769 | 0.723333 | 1.000000 | none |
| 1 | 39 | 33/3/3 | 0.823529 | 0.777778 | 1.000000 | none |

| Flag | Class | True support | Predicted support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 100 | 170 | 0.588235 | 1.000000 | 0.740741 |
| 0 | 1 | 100 | 17 | 1.000000 | 0.170000 | 0.290598 |
| 0 | 2 | 100 | 113 | 0.884956 | 1.000000 | 0.938967 |
| 1 | 0 | 33 | 35 | 0.942857 | 1.000000 | 0.970588 |
| 1 | 1 | 3 | 1 | 1.000000 | 0.333333 | 0.500000 |
| 1 | 2 | 3 | 3 | 1.000000 | 1.000000 | 1.000000 |

Zero predicted support makes precision undefined; zero true support makes recall undefined. F1 is undefined if both supports are zero. A fixed-three-class subgroup macro-F1/balanced accuracy is undefined if any actual class is absent. This is the same explicit denominator policy as Phases 2B/3A. Small flagged fault counts do not support precise reliability claims.

## Earlier development selection context — not final-test results

| Development summary | Macro-F1 | Balanced accuracy | Class-2 recall |
| --- | --- | --- | --- |
| Unweighted three-fold mean (used for selection) | 0.837084 | 0.850019 | 0.822803 |
| Pooled development OOF (also inspected in selection) | 0.841021 | 0.847048 | 0.813559 |

Those scores came from smaller, purged inner training folds and were used to select C=100. The final Pipeline was fitted once on all retained development rows, with a different test class mix and held-out nuisance configurations. Differences are descriptive; no post-test model reselection was performed.

## Saved model, metadata and inference

Model bundle: `artifacts/models/phase3b-v1/pipeline.pkl` and `metadata.json`. Model SHA-256: `d6d74074e6d08080c3b2d331b21b0b912017b82b74954504b8da05565d24cb26`.

The pickle contains the complete fitted scaler and classifier. Metadata contain the selected parameters, class mapping, confirmed classes_ order, all 70 feature names, 14-sensor rate/length contract, library versions, all training IDs, input/source references and the model checksum. No raw training samples or test-derived fitted preprocessing is included.

Before/after-loading equality was verified on all 1,328 development rows:

```json
{
  "scope": "all development rows only",
  "rows": 1328,
  "predict_equal": true,
  "predict_proba_equal": true,
  "max_probability_difference": 0.0,
  "predictions_sha256_before": "8a9edab17890371cbd80d8c717c5b1803cdf0755bae784fec2b95acf7ff63c1d",
  "predictions_sha256_after": "8a9edab17890371cbd80d8c717c5b1803cdf0755bae784fec2b95acf7ff63c1d",
  "probabilities_sha256_before": "6b2e194942e38070dd024e739b4ce4b6c1a16887c27c187146742fed22d7889b",
  "probabilities_sha256_after": "6b2e194942e38070dd024e739b4ce4b6c1a16887c27c187146742fed22d7889b"
}
```

Confirmed probability column order: `[0, 1, 2]`. The single-cycle function uses the existing extractor and requires exactly 14 physical-sensor 1D finite real vectors at sampling_rate × 60 samples. Missing/extra sensors, metadata keys, wrong lengths/dimensions and nonnumeric/nonfinite values are rejected. It returns class, English label, classes and aligned probabilities. These are raw LR probabilities, not a separately calibrated reliability guarantee.

```text
uv run --locked --group baseline python scripts/train_final.py --prepare-only
uv run --locked --group baseline python scripts/train_final.py
uv run --locked --group baseline python scripts/evaluate_final.py
uv run --locked --group baseline python scripts/predict_cycle.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
```

Training/evaluation attempts have separate immutable receipts. Completed reruns verify the existing model or cached predictions/results without refitting or reopening test features. Incomplete attempts stop for investigation instead of silently retrying. The inference example is restricted to a development ID.

## Exposure, limitations and review boundary

All cycles, including this reserved test, were inspected in Phase 2A EDA. This is the first authorized Phase 3B test feature/prediction evaluation, but the test is not wholly untouched by prior exploration. Groups are not verified independent experiments; gap 20 is not a validated physical decorrelation time. The same controlled rig, thermal trajectory, fixed loads and scarce unstable fault cases limit external generalization. No industrial reliability or calibrated uncertainty claim follows from these results.

History and uv.lock remain unchanged. No further tuning or other test-model comparison was performed. No API, Docker, commit or push was created. Stop here for review.

The bundle is pickle protocol 5: load only trusted, locally generated artifacts. Checksums detect corruption, not malicious replacement of both model and metadata. The loader requires matching library versions. See the official [scikit-learn model-persistence documentation](https://scikit-learn.org/stable/model_persistence.html).

## Completed validation

297 tests passed; 0 failures, 0 errors, 0 skipped. Ruff lint/format and uv lock checks passed. Independent saved-prediction metric/probability checks, training/test membership, persistence equality, development-only inference example and historical checksums passed.
