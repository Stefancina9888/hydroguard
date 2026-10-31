# HydroGuard: project explanation and design decisions

## What problem does it address?

Classify internal hydraulic pump leakage as no, weak or severe from one 60-second
sensor cycle. The intended audience is maintenance engineers and ML peer reviewers.
The deliverable demonstrates a complete ML workflow, not an operational maintenance
decision system. In particular, weak leakage is poorly detected on the reserved test.

## Why this dataset and these features?

UCI 447 supplies 2,205 ordered cycles from one controlled hydraulic rig. Fourteen
physical sensors supply mean, population standard deviation, RMS, minimum and
maximum: 70 deterministic attributes. These summarize level and variability at a
small computational cost. CE/CP/SE are excluded from X; labels, cycle IDs,
configuration and stability flags remain metadata. These decisions simplify the
demonstrator; they do not prove that the summaries retain all diagnostic information.
The loader rejects malformed/nonfinite inputs rather than silently imputing them.

## Why not use a random row split?

Neighboring cycles and operating configurations are strongly related. A random split
would mix close neighbors across training and validation. The saved protocol holds
out cooler/valve/accumulator configurations and purges nearby original cycle IDs.
It retains 1,328 development, 339 test and 538 excluded cycles, with three inner CV
folds. The groups are not proven independent experiments, gap 20 is not a validated
decorrelation time, and the test was exposed during earlier EDA. The protocol limits
some obvious overlap but does not establish external generalization.

## What models were compared and how was the winner chosen?

Dummy, scaled Logistic Regression and Random Forest baselines used the same saved
development folds. A prespecified search evaluated 23 LR/RF candidates in 69 fold
fits. Selection used mean fold macro-F1, then the recorded tie-breakers. Macro-F1
weights each leakage class equally. LR lr_09 (C=100, class_weight=None) was selected
before test prediction. Scaling belongs inside the Pipeline and learns only from
each training fold. Complex algorithms were not required to justify the selected
development result. See the executed modeling notebook and tuning report.

## What did the final test show?

Historical development selection macro-F1 was 0.837084; reserved-test macro-F1 was
0.675047. Weak-leakage recall was 18/103: 72 weak cases were predicted as no leakage,
13 as severe. Severe recall was 103/103 in this sample. That sample result does not
establish perfect future performance. The development/test difference is consistent
with limited generalization but does not isolate a single cause. No post-test tuning,
threshold change, feature change or alternative test-model comparison was performed.
Future improvements require a new evaluation design and preferably independent rig
data; the existing test must not become an optimization target.

## What is reproduced now?

The EDA notebook repeats descriptive exploration. The modeling notebook displays
historical baseline/tuning/test evidence and executes one new fit of the fixed
candidate using only the original development IDs. The standalone training script
calls the same reproduction function. New bundles go to separate ignored directories.
Data/versions/contracts are checked, actual acquisition provenance is retained,
and original model/report bytes stay fixed. No final-test metrics are recomputed.

## How does the service use the model?

FastAPI loads the official fitted Pipeline once. POST /predict validates exactly
14 sensor vectors and their lengths, extracts the same statistics, then applies
the saved scaler and classifier. It returns a class, label, ordered probabilities
and model version. Probabilities are uncalibrated. Missing sensors, metadata,
boolean mixtures, wrong dimensions and nonfinite values are rejected. Docker
contains runtime dependencies and the model, so serving needs no training dataset.

## What should the owner be able to explain before submission?

Run the README Docker example and explain the input, output and class mapping.
Locate the notebook comparison that led to lr_09. Explain why scaling is fitted
inside folds, why temporal/configuration overlap matters, and what 18/103 weak
recall means. Be able to distinguish historical test evidence from development
reproduction. AI assistance is disclosed; the owner remains responsible for the
implementation, explanation and course rules.

## Why add Kubernetes?

The same immutable API image is loaded into a local kind cluster. A Deployment
maintains one Pod, health probes determine startup/readiness/liveness, and a
ClusterIP Service selects the ready Pod. A loopback-only port-forward makes
the Service reachable for review. The optional local browser client sends the
existing development example through that Service and shows its actual HTTP
response. Kubernetes changes deployment, not model quality. This single-node,
single-replica demonstration does not establish high availability or industrial
suitability. See the deployment guide for repeatable startup and cleanup.
