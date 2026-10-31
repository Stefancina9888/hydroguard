# Historical phase walkthrough

This is the original phase-by-phase command reference, retained for provenance. For a fresh checkout, use the [current README](../README.md) instead. Historical completion statements describe the time of each phase. The project has since been published; see the [submission guide](submission-guide.md).


The sections below document the original phase commands and artifacts. They are
historical procedures, not the fresh-acquisition reproduction path above. Do not
rerun training, tuning or reserved-test evaluation for peer review. Phases 4/5 do
not authorize another test evaluation.

## Run Phase 1A

Install [uv](https://docs.astral.sh/uv/), then run these commands from this directory
(PowerShell, Bash or another terminal; no environment activation is needed):

```text
uv sync --locked --python 3.11
uv run --locked python scripts/download_data.py
uv run --locked python scripts/inspect_data.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
```

`.python-version` selects Python 3.11; `uv.lock` pins resolved dependencies.
The first download requires internet access and about 74 MB of transfer. Allow
at least 1 GB of free disk space and several hundred MB of RAM. Parsing processes
one sensor matrix at a time. Default paths resolve from the project root even if
the scripts are launched from another working directory.

The downloader uses the official UCI ZIP and the unchanged source-code SHA-256 pin.
It derives a fresh per-file manifest directly from the verified ZIP content and
compares both `download.json` and the extracted files against it. Editing a raw file
and its local manifest together cannot bypass this check. Cached runs require no
network access. Existing raw files are never silently replaced.

Publication uses a verified archive, a staged extraction and an atomic manifest
write (temporary file, flush, `fsync`, replace). After an interruption, rerun the
same command: a missing extraction can be recreated from the pinned archive; a
missing manifest can be reconstructed only after every existing extracted file
matches the archive. Recovery records its timestamp and leaves the original
download time unknown if it cannot be recovered. Malformed or mismatched existing
manifests cause an explicit error and are not silently rewritten. Interrupted
temporary files are never trusted. Use `--raw-dir PATH` for an isolated acquisition;
the same pin still applies. No credentials are required.

The inspector exits nonzero if any required file, shape, value, profile domain or
provenance checksum fails. Empty and whitespace-only physical rows are rejected
with the filename and one-based line number, including leading/trailing blank rows.
It writes failure details to both reports. On successful
validation it also writes `data/processed/cycle_metadata.csv`, containing the target,
all original condition columns and one-based cycle IDs. The previous CSV is removed
before every audit; a replacement is published atomically only after a successful
audit and report publication. A failed or interrupted run therefore leaves no stale
CSV available at that path. Failed reports do not certify cycle completeness.

The audit groups cycles with exactly equal parsed values across **all 17 sensors**,
independently of target and other profile columns. SHA-256 narrows candidates and
exact numeric row comparisons confirm equality; no rounding or tolerance is used.
Reports list every duplicate `cycle_id` group and its leakage targets, including
conflicts. Duplicates/conflicts are warnings, not grounds for automatic deletion or
relabeling. No cycles are filtered. JSON also records every observed target run and
condition block. Malformed fields are rejected rather than imputed.

## Run Phase 2A

After acquiring the raw dataset using the Phase 1A commands:

```text
uv sync --locked --group eda --python 3.11
uv run --locked --group eda python scripts/run_eda.py
uv run --locked --group eda pytest
uv run --locked --group eda ruff check .
uv run --locked --group eda ruff format --check .
uv lock --check
```

The runner executes [01_eda.ipynb](../notebooks/01_eda.ipynb) from a fresh kernel using
the active uv Python and saves the executed notebook with figures and tables.
Alternatively, select that environment in VS Code's notebook interface and Run All.
Allow up to 300 seconds per cell (configurable with `--timeout`); the notebook reads
all raw sensors through the existing strict loader and verifies checksums before
and after analysis. It leaves raw files and Phase 1A reports/exports unchanged.

The optional `eda` dependency group contains plotting, notebook execution and
splitter tools. Scikit-learn is used only to inspect temporary split feasibility;
no estimator is fitted. Cycle summaries are descriptive and stay in memory. The
only extra local export is `data/processed/eda-summary.json`, an aggregate results
record containing no per-cycle sensor feature rows or split membership lists.
Read [EDA findings](eda-findings.md) and [the current validation protocol](validation-plan.md).
The notebook retains its historical Phase 2A feasibility analysis; Phase 2B locks
the reviewed partition described below. Every candidate subset was inspected in EDA.

## Run Phase 2B

```text
uv sync --locked --group baseline --python 3.11
uv run --locked --group baseline python scripts/lock_splits.py
uv run --locked --group baseline python scripts/run_baselines.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
```

The default development group also installs baseline dependencies for normal tests.
The split command publishes `artifacts/splits/phase2b-v1.json` once, or verifies
the exact existing lock without rewriting it. Different inputs, environment,
parameters or membership cause a stop; there is no automatic overwrite/reset flag.
The baseline command requires this lock and never creates a replacement itself.

The final test is outer SGKF fold 2: 339 cycles with class counts 133/103/103.
The outer original-cycle gap leaves 1,328 development and 538 excluded cycles.
Inner three-fold group CV applies an additional original-ID gap 20. Counts,
classes, groups, gaps and feature schema are checked before every fit.

The shared extractor computes exactly 70 deterministic attributes from 14 physical
sensors; no metadata or CE/CP/SE enters X. Only development sensor rows are summarized.
Three prespecified models (most-frequent dummy, scaled logistic regression and
random forest) are evaluated on identical inner folds, without tuning or a final
development refit. Parameters are in `config/baseline-v1.json` and copied into the
split lock before fitting. Test data are used only for membership/integrity checks.

Actual metrics, class reports, confusion matrices and stability denominators are
generated in [baseline-results.md](baseline-results.md) and its JSON companion.
Development-only OOF predictions are local and Git-ignored in
`data/processed/baseline-oof.csv`. No fitted estimator or final-test feature table
is exported. See [validation-plan.md](validation-plan.md) for prior EDA exposure,
undefined-metric conventions and the limitations of the gap/group assumptions.

## Run Phase 3A

Use the existing locked environment and reproduced Phase 2B artifacts:

```text
uv run --locked --group baseline python scripts/run_tuning.py --prepare-only
uv run --locked --group baseline python scripts/run_tuning.py
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
```

Preparation permanently records all concrete candidates and the selection rule in
the separate immutable `config/tuning-v1.json` and experiment manifest
`artifacts/experiments/phase3a-v1.json`. Existing different preregistrations are
rejected, not overwritten. The manifest references the original split hash;
neither the Phase 2B split lock nor its baseline parameters/results are changed.

The script evaluates 10 Logistic Regression configurations and 12 seeded distinct
Random Forest configurations plus the RF baseline reference: 23 candidates and
69 classifier fits. It reads the same three saved development folds and checks
their original-ID gaps without regenerating CV. The unchanged 70-feature extractor
summarizes development rows only. Scalers and balanced weights use training rows
only. Selection maximizes unweighted mean fold macro-F1, then mean class-2 recall,
then the preregistered simplicity ordering and candidate ID, using exact score ties.

Selected OOF metrics use cached development predictions with no additional fit.
No winner is refitted on full development data, and test/outer-excluded cycles
receive no features, predictions or metrics. Read [tuning-results.md](tuning-results.md)
and [JSON results](tuning-results.json), explicitly development results used
for selection. Each execution retains a separate local progress/results journal
and OOF table under `data/processed/tuning-runs/`; `tuning-oof.csv` is the latest
development-only candidate prediction table. Baseline output paths stay untouched.

## Run Phase 3B

Use the unchanged locked environment and reviewed Phase 2B/3A artifacts:

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

The first command permanently records the selected `lr_09` candidate, exact
parameters, 70-feature/14-sensor contracts and checksum references in
`artifacts/experiments/phase3b-v1.json` before fitting or test feature access.
The training command fits one new complete StandardScaler/LogisticRegression
Pipeline on exactly the locked 1,328 development cycles. C=100, class_weight=None
and every other selected parameter remains fixed. Outer-excluded cycles stay out.

The saved bundle is `artifacts/models/phase3b-v1/pipeline.pkl` plus `metadata.json`.
Metadata include the model checksum, all feature names, class mapping/classes_
order, library versions and every training cycle ID. Predictions and predict_proba
are checked for exact equality before/after loading on all development rows.
The checksum is verified before unpickling; load only trusted local artifacts and
use the recorded library versions. Pickle checksums are not source authentication.

Evaluation requires the completed bundle and writes a separate test-access receipt
before extracting the 339 reserved test cycles. Only the fixed chosen model is
evaluated. Reports are [final-test-results.md](final-test-results.md) and its
JSON companion; predictions/probabilities are local in
`data/processed/final-test-predictions.csv`. Development selection metrics remain
separate. Earlier EDA exposure of the test and gap/group limitations are explicit.

Completed training reruns verify the bundle without a new fit; completed evaluation
reruns verify cached results without further test features/predictions. An incomplete
attempt stops for investigation. Existing historical locks/configurations/results
and uv.lock are never overwritten.

The default inference example uses the first development training cycle. Optional
`--cycle-id N` accepts development training IDs only. For raw in-memory inference:

```python
from pathlib import Path
from hydroguard.inference import load_model_bundle, predict_cycle

bundle = load_model_bundle(Path("artifacts/models/phase3b-v1"))
# raw_cycle maps exactly the 14 physical sensor names to one numeric vector each.
result = predict_cycle(raw_cycle, bundle)
# result: class, English label, classes [0, 1, 2], probabilities in that order.
```

Each vector must be one-dimensional, finite and real numeric, with exactly rate ×
60 samples: 6,000 at 100 Hz, 600 at 10 Hz, 60 at 1 Hz. Metadata, missing/extra
sensors, wrong lengths/dimensions, strings, complex and nonfinite values
are rejected. The function uses the existing deterministic extractor and needs no
dataset or split files when provided raw cycle vectors and the loaded bundle.
Probabilities are uncalibrated model outputs, not reliability guarantees.

**Historical Python-helper caveat:** pure boolean vectors are rejected, but a
mixed float/boolean list can be coerced to numeric values by NumPy before the
legacy helper checks its dtype. For current integrations, use the validated
HTTP API or `hydroguard_serving.runtime.Runtime.predict` in the serving environment.
That boundary checks each original scalar before conversion and rejects mixed
booleans as well. The frozen helper and official model remain historical artifacts;
see the current [serving validation](../serving/src/hydroguard_serving/validation.py).

## Run Phase 4 — local API and Docker

Run commands from the repository root. Do not rerun historical training/evaluation
commands to start the API. The serving project has its own `pyproject.toml`,
`.python-version`, `.venv` and `uv.lock`; root dependencies/locks and historical
checksummed source files remain unchanged. Python **3.11.15**, NumPy **2.4.6**,
pandas **2.3.3**, scikit-learn **1.9.1**, SciPy **1.17.1** and threadpoolctl **3.7.0**
match the model metadata exactly.

```text
uv sync --project serving --locked --no-editable --python 3.11.15
uv run --project serving --locked --no-editable uvicorn hydroguard_serving.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Leave that terminal running; stop it with Ctrl+C. The default bundle path is
`artifacts/models/phase3b-v1`, relative to the server's working directory. For a
different directory, set `HYDROGUARD_MODEL_DIR` to an absolute path containing only
the **existing** `pipeline.pkl` and `metadata.json`. Their checksums are pinned by
`serving/src/hydroguard_serving/model-contract.json`; another model is refused.
The loader verifies library versions, bundle contents and the historical inference
source hashes once at startup. A mismatch fails startup. No raw data, split files,
training reports or root project install is required at runtime. Non-editable
packaging includes only four unchanged historical modules needed by inference.

The checked-in request `serving/examples/development-cycle.json` contains only
the 14 raw sensors from **development cycle 1**. Separate `provenance.json` records
membership, source attribution and SHA-256. These files are excluded from the
Docker image. Optional regeneration requires the original raw data and the root
environment: `uv run --locked python serving/scripts/make_example.py`.

In a second terminal, use curl (on Windows use **curl.exe**, not the PowerShell alias):

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -H "Content-Type: application/json" --data-binary "@serving/examples/development-cycle.json" http://127.0.0.1:8000/predict
```

PowerShell equivalent:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -InFile '.\serving\examples\development-cycle.json' | ConvertTo-Json -Depth 4
```

`GET /health` returns `ready` and `model_version` (503 when no model is ready).
`POST /predict` accepts `{"sensors": {...}}` and returns `class`, English `label`,
`classes: [0,1,2]`, aligned `probabilities`, and `model_version`. The full model
SHA-256 is part of the version. API schema is at `/docs` and `/openapi.json`.
Every vector must be a JSON array of finite real numbers: 6,000 samples for
PS1–PS6/EPS1, 600 for FS1/FS2, and 60 for TS1–TS4/VS1. Boolean values are checked
**before NumPy conversion**, including mixed float/boolean arrays. Missing/extra
sensors, metadata, extra root fields, null, strings, wrong lengths/dimensions and
nonfinite samples return **HTTP 422**. Errors report locations and reasons without
echoing sensor arrays. Models cannot be uploaded through HTTP. Inference uses the
unchanged 70-feature extractor and complete fitted scaler/classifier Pipeline.

For Docker, start your installed Docker daemon in Linux-container mode; no Docker
installation is performed by the project. Build from this directory:

```text
docker build -t hydroguard:phase4 .
docker run --rm --detach --name hydroguard-phase4 --read-only --cap-drop=ALL --security-opt=no-new-privileges --publish 127.0.0.1:8001:8000 hydroguard:phase4
docker inspect --format "{{.State.Health.Status}}" hydroguard-phase4
```

The image uses Python 3.11.15, locked runtime dependencies, an unprivileged user and
only the selected model bundle. The default-deny `.dockerignore` excludes raw data,
historical reports/splits, environments, examples, tests and temporary files.
No volumes or dataset mounts are needed. Allow startup to reach `healthy`, then:

```bash
curl --fail http://127.0.0.1:8001/health
curl --fail -H "Content-Type: application/json" --data-binary "@serving/examples/development-cycle.json" http://127.0.0.1:8001/predict
```

PowerShell uses the same `Invoke-RestMethod` commands above with port **8001**.
Stop and remove the `--rm` container with `docker stop hydroguard-phase4`.
The built image stays local; optionally remove it with `docker image rm hydroguard:phase4`.

Validation commands (no training or test-set model access):

```text
uv run --locked pytest --junitxml=tmp/phase4/historical-tests.xml
uv run --project serving --locked --no-editable pytest -c serving/pyproject.toml serving/tests --junitxml=tmp/phase4/api-tests.xml
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --project serving --locked --no-editable ruff check serving
uv run --project serving --locked --no-editable ruff format --check serving
uv lock --check
uv lock --project serving --check
```

Actual direct/local HTTP/container agreement, image inspection, commands and
limitations are recorded in [serving-verification.md](serving-verification.md)
and its JSON companion. The verification runner is
`uv run --project serving --locked --no-editable python serving/scripts/verify_serving.py`.
It uses temporary localhost ports, starts/stops its own Uvicorn process and Docker
smoke container, and records unavailable Docker checks explicitly. See the report
for the prerequisites and check records. This local demonstration has no cloud
deployment, authentication or production traffic controls.

## Repository layout

```text
data/README.md               Local data policy and generated file layout
docs/dataset.md             Dataset specification, attribution and limitations
docs/dataset-audit.md       Human-readable results from actual execution
docs/dataset-audit.json     Machine-readable results from the same execution
src/hydroguard/data.py      Acquisition, parsing, validation and audit functions
scripts/download_data.py   Official UCI download command
scripts/inspect_data.py    Validation and report command
tests/test_data.py         Small synthetic fixtures; no network/full dataset needed
docs/phase-1a-review.md    Review fixes, executed checks and remaining limitations
notebooks/01_eda.ipynb      Executed focused EDA and split-feasibility notebook
docs/eda-findings.md        Observations, interpretations and open questions
docs/validation-plan.md     Proposed protocols, estimands and limitations
scripts/run_eda.py          Fresh-kernel notebook execution using uv Python
src/hydroguard/features.py  Fixed 14-sensor/70-column extractor and X allowlist
src/hydroguard/splitting.py Locked group membership and original-cycle gap checks
src/hydroguard/metrics.py   Metrics with explicit undefined values
src/hydroguard/baselines.py Development-only fitting and report generation
scripts/lock_splits.py      Publish/verify the split lock before any dataset fit
scripts/run_baselines.py    Run the three fixed development baselines
config/baseline-v1.json     Prespecified, untuned model parameters
artifacts/splits/phase2b-v1.json  Version-control-intended membership/checksum lock
docs/baseline-results.md    Executed development results and limitations
src/hydroguard/tuning.py    Separate preregistration and bounded development tuning
src/hydroguard/tuning_report.py  Rendering of actual tuning results
scripts/run_tuning.py       Prepare/run the fixed Phase 3A experiment
config/tuning-v1.json       Locked concrete candidates and selection policy
artifacts/experiments/phase3a-v1.json  Manifest referencing the existing split hash
docs/tuning-results.md      Development selection results and limitations
src/hydroguard/inference.py Validated raw-cycle inference and complete bundle loading
src/hydroguard/final_model.py Phase 3B manifest, single fit and fixed-model test evaluation
src/hydroguard/final_report.py Final-test reporting separated from development selection
scripts/train_final.py      Prepare manifest / train the final selected Pipeline
scripts/evaluate_final.py   Evaluate or verify cached reserved-test results
scripts/predict_cycle.py    Development-only raw-cycle inference example
artifacts/experiments/phase3b-v1.json  Pre-test selected-model manifest
artifacts/models/phase3b-v1/ Complete fitted Pipeline and inference metadata
docs/final-test-results.md  Actual reserved-test evaluation and limitations
serving/pyproject.toml     Independent API project; root environment remains unchanged
serving/uv.lock            Locked API runtime and test dependencies
serving/src/hydroguard_serving/ Strict raw input, pinned local runtime and FastAPI app
serving/tests/test_api.py  Validation, startup, isolation and HTTP regressions
serving/examples/          One attributed development-cycle request and provenance
serving/scripts/           Example exporter and actual HTTP/Docker verification
Dockerfile                Local runtime image with selected model only
.dockerignore             Default-deny build context
docs/serving-verification.md Actual Phase 4 checks and limitations
pyproject.toml             Python package and development tools
uv.lock                    Locked dependencies
.python-version            Default Python version
```

Downloaded data, derived data, environments and caches are ignored by Git.
The small audit reports are intended for version control. Tests run entirely on
synthetic matrices and an in-memory HTTP response; the real-data audit is a separate
explicit command. See [the dataset specification](dataset.md) and
[the executed audit](dataset-audit.md) for source and measured results.
Keep `uv.lock` and `.python-version` in Git along with the source files.
