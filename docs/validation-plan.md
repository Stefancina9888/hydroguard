# Phase 2B — locked evaluation protocol

Status: approved for Phase 2B and fixed before any dataset model fit. This
supersedes the provisional Phase 2A protocol. Outer validation fold 2 (one-based)
was selected by the project owner, not by comparing model scores. Historical
feasibility evidence remains in [the notebook](../notebooks/01_eda.ipynb) and
[EDA findings](eda-findings.md).

## Estimand and metadata contract

The target is transfer to held-out cooler/valve/accumulator combinations on the
same rig, not new-rig, new-pump, new-load or strictly prospective generalization.
Configurations are not certified independent experiment/session IDs. Gap 20 is a
cycle-proximity guard, not an established physical decorrelation time.

Preserve original cycle IDs and source order. Target, cycle ID, condition labels,
stability and grouping IDs are metadata only, never X columns. Do not silently
filter unstable cycles. EDA summaries/scales are not fitted evaluation features.

## Frozen outer partition

- Group key: `(cooler_condition_pct, valve_condition_pct, accumulator_pressure_bar)`.
  Exclude target/stability. Pool every occurrence, including nonconsecutive returns.
- Use sorted tuple factorization as in Phase 2A, then
  `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`.
- Reserve validation fold **2, numbered from 1**, as the final test.
- From its development candidates remove IDs at absolute original `cycle_id`
  distance **at most 20** from any test ID. Keep only distances strictly above 20.

Required counts, checked before publication and before every development fit:

| Role | Cycles | Class 0 | Class 1 | Class 2 |
| --- | ---: | ---: | ---: | ---: |
| Development | 1328 | 846 | 246 | 236 |
| Final test | 339 | 133 | 103 | 103 |
| Excluded by outer gap | 538 | 242 | 143 | 153 |

Roles are disjoint and cover IDs 1–2205 exactly once. Development/test have no
shared nuisance groups and minimum original-ID distance 21. Excluded cycles remain
in raw data but cannot enter fitting, validation or additional scoring in Phase 2B.

**Final test:** membership, required class/count coverage and integrity checks only.
No test sensor summaries for X, predictions, probabilities or classification
metrics. The existing loader may read complete raw matrices for integrity checks;
only development rows are summarized by the extractor.

## Frozen inner development CV

Within development, retain original order, recompute sorted nuisance tuple codes,
and use `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026)`.
For each validation fold, exclude training candidates within 20 original cycle IDs
of any validation case, on either side. Never calculate gaps on compressed row
positions. Require all three classes, group/ID disjointness and separation >20
before every fit. No adaptive fallback is allowed.

Expected memberships/counts were computed before model fitting:

| Fold | Train | Validation | Inner gap exclusion | Train classes 0/1/2 | Validation classes 0/1/2 |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 607 | 442 | 279 | 453/82/72 | 290/81/71 |
| 2 | 576 | 457 | 295 | 422/82/72 | 293/82/82 |
| 3 | 607 | 429 | 292 | 463/82/62 | 263/83/83 |

Each fold's train/validation/excluded roles partition development only. Every
development cycle receives exactly one inner OOF prediction per model. A case
excluded by one inner gap can be eligible in another fold. All models use the
same memberships; any count, class, coverage, group, checksum or gap mismatch stops
execution before fitting.

## Durable split lock

Run `uv run --locked --group baseline python scripts/lock_splits.py` first.
The version-control-intended `artifacts/splits/phase2b-v1.json` stores all outer and
inner cycle-ID members, diagnostics, protocol parameters, fixed feature schema,
model presets, environment versions and creation time. Checksums cover the official
archive, all 20 extracted files, existing acquisition manifest, `uv.lock` and
`config/baseline-v1.json`. A canonical SHA-256 protects the payload.

Publication flushes/syncs a temporary file and uses a same-filesystem hard link
for atomic publication without replacement. A matching existing lock is verified
without rewriting bytes; a different, partial or corrupted lock is rejected.
There is no overwrite/reset flag. Filesystems without hard-link support fail closed.
A future change requires explicit review and a separately versioned protocol.
Git attributes preserve LF bytes for the checksummed environment/configuration
text files across checkouts; they do not modify or normalize any raw data.

The baseline command refuses a missing lock. It recomputes source/environment/config
fingerprints and expected membership before feature extraction, revalidates all
partition requirements before every fit, and checks that the lock is unchanged.
Raw checksums are checked after extraction and after CV. Lock time/fingerprint are
included in the result report. This is integrity checking, not publisher-authenticated
provenance; the original locally observed archive pin is unchanged.

## Fixed deterministic feature contract

Sensor order: `PS1, PS2, PS3, PS4, PS5, PS6, EPS1, FS1, FS2, TS1, TS2, TS3, TS4, VS1`.
CE, CP and SE are excluded. Within each sensor, use exactly:

1. Mean of its within-cycle samples.
2. Population standard deviation, `ddof=0`.
3. RMS: square root of the mean of squared samples.
4. Minimum sample.
5. Maximum sample.

Names are `<sensor>__<statistic>`, with statistics ordered
`mean, std, rms, min, max`: exactly 70 float64 columns from `PS1__mean` to `VS1__max`.
A strict ordered-column allowlist rejects metadata, extra/missing/reordered columns,
nonfinite values and named metadata indices. Cycle IDs are a separate array.
Statistics depend only on their own cycle; there is no fitted extractor state,
across-cycle normalization, imputation or reuse of EDA-fitted values.

## Prespecified first baselines — no tuning

Complete constructor parameters are declared in
[config/baseline-v1.json](../config/baseline-v1.json) and copied into the split lock
before any fit. Only these three models are authorized:

| Model | Fixed choices |
| --- | --- |
| Dummy | strategy=most_frequent |
| Logistic regression | Pipeline: StandardScaler (with_mean=True, with_std=True), then LogisticRegression (C=1, l1_ratio=0, solver=lbfgs, max_iter=2000, tol=1e-4, fit_intercept=True, class_weight=None, random_state=2026) |
| Random forest | n_estimators=200, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, bootstrap=True, class_weight=None, random_state=2026, n_jobs=1 |

With the locked scikit-learn 1.9 API, l1_ratio=0 gives L2 regularization. Numerical
fits/predictions use one thread. Create a fresh estimator/Pipeline per fold; fit
the scaler only on that fold's training rows and verify its sample count/means.
A convergence warning is an error; no result-driven parameter change is permitted.
No searches, XGBoost, model-driven feature selection, full-development refit or
final-test scoring occurs in this phase.

## Metrics, OOF and stability

Primary: macro-F1. Also report balanced accuracy, class 2 recall, per-class
precision/recall/F1/support and confusion matrices. Class order is `[0,1,2]`;
confusion rows are truth and columns predictions. Show every fold, unweighted
fold means and population standard deviations, and metrics recomputed from pooled
OOF predictions. Fold confusion counts must sum to the pooled matrix. A mean of
fold scores and a score of pooled predictions are different quantities.

Stratify the same OOF predictions by stability 0/1, with support, fold and pooled
results. No subgroup-specific model is fitted. Mark undefined quantities as JSON
null and Markdown `undefined`:

- Precision: undefined when predicted class support is zero.
- Recall: undefined when true class support is zero.
- F1: undefined when both supports are zero; zero for a supported class with only
  false negatives, even if precision is undefined.
- Fixed-three-class subgroup macro-F1/balanced accuracy: undefined if any actual
  class is absent. Never silently substitute a present-class-only macro average.
- Fold means: undefined if any corresponding fold value is undefined; report the
  number of defined folds instead of silently dropping a fold.

Generate [baseline-results.md](baseline-results.md) and the JSON companion from
actual execution. Local/Git-ignored `data/processed/baseline-oof.csv` must contain
only development IDs and no final-test/outer-excluded predictions.

## Prior exposure and limitations

All 2205 cycles, including the now-reserved test, were examined during Phase 2A
EDA. This partition is locked before training but is not untouched by exploration.
Thermal trajectory, shared rig and repeated controlled conditions may induce
remaining dependence after grouping and gap purging. Fold spread is descriptive,
not an IID confidence interval or variability across independent experiments.

Purging changes training sizes and class mixtures. Unstable weak/severe cases are
scarce; inspect denominators before interpreting subgroup scores. Development
baselines do not establish industrial reliability or external generalization.
Do not change the protocol based on model results. Stop after the report for
review; tuning, final-test scoring, API, Docker, commits and pushes are excluded.

## References

Official [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
describes group-disjoint approximate stratification.
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
documents the locked regularization API. The explicit missing-class subgroup
policy above is a project reporting choice, not an assertion that all library
metrics use that same convention.
