# Phase 2A — focused EDA findings

Evidence generated (UTC): 2026-10-30T19:09:36.180540+00:00. All measurements below are from
the executed [notebook](../notebooks/01_eda.ipynb), not estimates copied from other
projects. Runtime: Python 3.11.15, NumPy 2.4.6,
pandas 2.3.3, Matplotlib 3.11.2,
scikit-learn 1.9.1. Sampling/split seed: 2026.

## Scope and integrity

The notebook uses the existing strict `load_matrix`, profile/sensor validators,
`cycle_metadata` and read-only `audit_dataset`. It validates all 2,205 cycles and
17 sensor matrices against the unchanged pinned ZIP. Original cycle order and
one-based IDs are asserted before and after EDA; all raw-file hashes, archive hash
and original `download.json` bytes are unchanged. Phase 1A reports and CSV exports
are left untouched. The existing full-cycle duplicate audit still finds zero groups.

Targets, IDs, condition labels and stability remain metadata. Sensor cycle means,
standard deviations and distances are descriptive only. No prediction feature
table, estimator, tuning, final split assignment file, service or deployment was
created. The Git-ignored `data/processed/eda-summary.json` contains aggregate
measurements and fold diagnostics, with no per-cycle sensor features or membership
lists. All candidate subsets have been inspected during EDA and are not pristine
final test sets.

## 1. Target and condition associations

| Target | Meaning | Cycles | Share | Flag 1 within class |
| --- | --- | --- | --- | --- |
| 0 | No leakage | 1,221 | 55.37% | 59.95% |
| 1 | Weak leakage | 492 | 22.31% | 2.44% |
| 2 | Severe leakage | 492 | 22.31% | 2.44% |

The full notebook reports counts and conditional leakage percentages for cooler,
valve, accumulator and stability. Unadjusted descriptive Cramér's V is:

| Condition | Descriptive Cramér's V |
| --- | --- |
| cooler_condition_pct | 0.0037 |
| valve_condition_pct | 0.3072 |
| accumulator_pressure_bar | 0.1919 |
| stability_flag | 0.6023 |

Measured contrasts:

- Flag 0: 1,449 cycles, with class counts 489/480/480. Flag 1: 756 cycles,
  with counts 732/12/12. Thus 96.83% of flagged cycles have no leakage, versus
  33.75% of stable cycles. Only 12 flagged examples exist for each leakage class
  1 and 2; any later per-stratum scores will have very limited support.
- At valve 100%, leakage counts are 861/132/132 (76.53% class 0). Each of valve
  73%, 80% and 90% has 120/120/120. This is co-occurrence, not a valve-to-leakage
  causal relationship.
- Accumulator counts for classes 0/1/2 are 562/123/123 at 90 bar, 153/123/123
  at both 100 and 115 bar, and 353/123/123 at 130 bar.
- Cooler class counts are 404/164/164 at both 3% and 20%, and 413/164/164 at
  100%. Nearly balanced target proportions across cooler states do not prevent
  substantial sensor distribution shifts across those states.

Interpretation: condition/stability co-occurrence creates plausible shortcut risks
even when those labels are excluded as inputs, because sensors may carry their
physical effects. Cramér's V is not feature importance. No independent-cycle
p-values, causal estimates or predictive scores are claimed.

## 2. Ordered signals and representative waveforms

Cooler states occupy exactly three consecutive source-order regimes:

| Cooler condition (%) | First cycle_id | Last cycle_id | Cycles |
| --- | --- | --- | --- |
| 3 | 1 | 732 | 732 |
| 20 | 733 | 1464 | 732 |
| 100 | 1465 | 2205 | 741 |

The six illustrated sensors were chosen by physical quantity before inspecting
predictive results: pressure PS1, motor power EPS1, flows FS1/FS2, temperature TS1
and vibration VS1. The notebook includes full within-cycle traces and cycle means
against original IDs, plus descriptions for all 17 sensors. Median cycle means
within the three observed cooler regimes are:

| Sensor | Unit | Cooler 3% | Cooler 20% | Cooler 100% |
| --- | --- | --- | --- | --- |
| PS1 | bar | 157.0863 | 158.4537 | 160.9231 |
| EPS1 | W | 2407.9366 | 2465.4933 | 2546.5453 |
| FS1 | l/min | 6.5416 | 6.6248 | 6.6671 |
| FS2 | l/min | 9.0609 | 9.6894 | 10.1667 |
| TS1 | °C | 55.6925 | 44.9063 | 35.8881 |
| VS1 | mm/s | 0.6631 | 0.6110 | 0.5485 |

Measured: TS1 temperature and FS2 flow change strongly across cooler/order regimes.
The first regime also has wider PS1, EPS1 and FS1 excursions in the plotted cycle
means. For example, FS1's cycle-mean interquartile range is 3.3275–6.5996 l/min at
cooler 3%, versus 6.4227–6.6439 at 20% and 6.4732–6.6922 at 100%.
Pooled lag-one cycle-mean correlations are 0.999917 for TS1, 0.999850 for FS2,
0.953371 for VS1 and 0.715877 for PS1. These correlations include regime changes
and do not isolate within-condition temporal dependence.

Raw waveform examples use the first target-free nuisance configuration with stable
support in every leakage class, then the middle source-order cycle of each class:

| cycle_id | Leakage | Cooler (%) | Valve (%) | Accumulator (bar) | Stability |
| --- | --- | --- | --- | --- | --- |
| 616 | 2 | 3 | 73 | 90 | 0 |
| 657 | 1 | 3 | 73 | 90 | 0 |
| 698 | 0 | 3 | 73 | 90 | 0 |

Measured visual observations: these examples share the repeated load-cycle timing;
PS1 traces are close for much of the cycle, while EPS1 and FS1 differ over some
load intervals. TS1 differs even under the matched annotations. These three
examples are illustrative, not evidence that a sensor can classify leakage in
general. Their different cycle IDs, temperatures and unobserved conditions remain
potential confounders. No physical cause for the early-regime excursions is
established here, and no cycles are removed.

## 3. Neighbor similarity and repeated configurations

The sensor distance is waveform RMSE divided by that sensor's standard deviation
over all raw values. The aggregate is the root mean square of the 17 normalized
sensor distances, weighting sensors equally despite different sampling rates.
Scales are descriptive for this full-dataset comparison, not preprocessing fitted
for evaluation. Positive/negative zero compare equally in the separate exact
duplicate audit; these distances do not use rounding or a near-duplicate threshold.

| Pair set | Pairs | Median distance | Q25–Q75 | Median cycle lag |
| --- | --- | --- | --- | --- |
| adjacent | 2,204 | 0.06816 | 0.05761–0.10391 | 1 |
| random_nonadjacent | 2,204 | 1.03980 | 0.48495–1.30888 | 638.5 |
| same_full_configuration_nonadjacent | 2,204 | 0.14898 | 0.08787–0.31526 | 72 |
| adjacent_same_full_configuration | 2,011 | 0.06589 | 0.05690–0.09184 | 1 |
| adjacent_configuration_change | 193 | 0.14290 | 0.10649–0.20627 | 1 |

Random comparator pairs are a fixed sample of distinct, unordered, nonadjacent
cycle pairs. The same-configuration comparator is sampled without replacement
from nonadjacent pairs with identical first four condition columns, including
target, independent of sensor values. It does not match stability or cycle lag.
Pair observations overlap and are dependent; quartiles are not confidence intervals.

Measured: adjacent cycles are much closer by this metric than the random sample;
the difference remains versus the same-configuration nonadjacent comparator.
Interpretation: the result is consistent with slow change and repeated operating
states, but does not prove causality, independent experiments, a particular
decorrelation gap, or a numerical amount of model-score optimism. Differing lag,
condition and stability mixtures can contribute. Zero exact duplicates does not
establish that random cycle partitioning is safe.

There are 194 observed consecutive blocks, 144 distinct full condition tuples and
36 tuples recurring in separate blocks. Grouping only by observed block ID can
therefore put the same tuple in different partitions. Excluding leakage and
stability leaves 48 nuisance configurations, all containing all three classes.
Each has 10–11 weak and 10–11 severe cycles; no-leakage counts range from 10 to 230.
These unequal group sizes explain why group-aware stratification is approximate.
Neither a tuple nor a consecutive block is a verified independent experiment.

## 4. Evaluation feasibility highlights

- Random stratified five-fold validation has all classes, but mean immediate
  training-neighbor exposure among validation cycles is 95.24%.
  Every validation cycle's full configuration also occurs in training.
- Grouping the 194 observed blocks avoids sharing block IDs but still shares up
  to 19 full configurations per fold. Grouping full four-column tuples removes
  that overlap, yet the grouping definition includes the target and allows up to
  33 target-free nuisance configurations to cross partitions.
- Target-free nuisance grouping with a 20-cycle training gap is feasible: every
  fold has all classes, zero shared nuisance/full configurations, zero adjacent
  cross-partition pairs and a minimum cycle-position distance of 21. Training sizes
  are 774–1,328; the smallest training class has 118 cycles. Validation sizes are
  339–526; the smallest validation class has 92 cycles. All marginal cooler, valve
  and accumulator values seen in validation still occur in training.
- Five forward-only folds with gap 20 also contain all classes (minimum train
  and validation class support 41). However, training size and cooler exposure
  change by fold; repeated configurations can still cross the temporal boundary.
- Contiguous blocked folds with gap 20 retain all classes but can share observed
  blocks and configurations and, except the last fold, train on later cycles too.

The [validation plan](validation-plan.md) contains every candidate summary and
the primary/supplemental proposal. These are feasibility diagnostics, not scored
models or committed split files.

## 5. Verification and open questions

The notebook executed all 9 code cells successfully in a fresh Python kernel and
contains five rendered figures. The 142 existing offline tests passed. Ruff lint,
Ruff format checks (including the notebook), and `uv lock --check` passed. Raw and
metadata equality assertions passed. The optional `eda` dependencies were locked;
the Python version, loader, target mapping and archive pin were unchanged.

Open questions for review:

1. Is the desired estimand new condition combinations on this rig, later cycles,
   a new pump/rig, or deployment at new operating loads? The available data do not
   establish the latter forms of generalization.
2. Can experiment/session identifiers, timestamps, load-control details, thermal
   settling information or explanations for early-regime excursions be obtained?
3. What cycle gap has a defensible physical basis? The examined 5/20/60-cycle gaps
   are sensitivity probes; no one of them establishes independence.
4. How should the 12 flagged weak and 12 flagged severe examples be interpreted?
   All remain included; the present evidence cannot support precise rare-stratum
   performance claims.
5. The entire dataset has informed EDA. A future frozen evaluation can constrain
   model development, but cannot be described as untouched by these analyses.

Dataset source and CC BY 4.0 attribution remain in [dataset.md](dataset.md).
