# Phase 4 — serving verification

Actual execution started: 2026-10-30T23:24:42.149379+00:00; completed: 2026-10-30T23:24:54.029259+00:00. UTC.

This is a local academic demonstrator for the unchanged Phase 3B Logistic Regression lr_09 Pipeline (C=100, class_weight=None). No training, tuning, threshold change, new feature, final-test prediction or final-test evaluation was performed in Phase 4.

## Environment and locked history

Model version: `phase3b-v1:lr_09:sha256:d6d74074e6d08080c3b2d331b21b0b912017b82b74954504b8da05565d24cb26`.

Numerical environment matches the saved metadata exactly:

```json
{
  "python": "3.11.15",
  "numpy": "2.4.6",
  "pandas": "2.3.3",
  "scikit_learn": "1.9.1",
  "scipy": "1.17.1",
  "threadpoolctl": "3.7.0"
}
```

Serving versions: `{'fastapi': '0.162.0', 'uvicorn': '0.54.0', 'pydantic': '2.14.2', 'starlette': '1.10.0'}`. The independent `serving/pyproject.toml`, `.python-version` and `uv.lock` preserve the root environment. The wheel packages byte-identical copies of only four historical modules: __init__, data, features and inference. Runtime pins both bundle file hashes and those source hashes, then uses the historical loader to verify metadata, fitted Pipeline, 70-column contract, classes and versions. A mismatch fails startup. The bundle is loaded once per application lifespan, never from HTTP.

Historical integrity: `{'historical_files': 54, 'all_unchanged': True, 'raw_archive_and_files_unchanged': True}`.

## Raw input boundary and endpoint checks

POST /predict accepts only {"sensors": {...}} with exactly 14 named physical sensors. Each raw sample must have Python type int or float before NumPy conversion; bool (including mixed float/bool lists), strings, null, complex values and nonfinite numbers are rejected. Vectors must be JSON lists of exactly rate × 60 numeric scalars. Metadata, additional root/sensor fields, wrong dimensions and wrong lengths return 422. The historical inference.py remains unchanged. Overflowing derived statistics also produce 422. Error responses omit input/context and cap displayed errors at eight; they never echo sensor arrays. /health returns readiness and the pinned model version; an absent runtime returns 503.

## Development example and actual HTTP agreement

Only original development cycle 1 is used. The payload has no metadata; its separate provenance includes source attribution, development membership and request checksum. The API needs no raw dataset, splits or training reports. The local Uvicorn smoke server was launched in an empty temporary working directory with an explicit local model path. Direct, HTTP and container probability agreement was prespecified at atol=rtol=1e-12 for platform numerical variation; class, label, classes and model version must match exactly.

Request SHA-256: `1dec343d49adabd0e21dfd23b9f5b08433ea5045356b3eb8e8f706b2ebd9ab63`.

Direct prediction:

```json
{
  "class": 0,
  "label": "no leakage",
  "classes": [
    0,
    1,
    2
  ],
  "probabilities": [
    1.0,
    1.8077945507920338e-34,
    1.277992518658375e-41
  ],
  "model_version": "phase3b-v1:lr_09:sha256:d6d74074e6d08080c3b2d331b21b0b912017b82b74954504b8da05565d24cb26"
}
```

Local HTTP: health=200, predict=200; agreement=True; exact probabilities=True; maximum absolute probability difference=0.0. Mixed boolean, additional metadata and missing-sensor requests each returned 422.

Docker HTTP: health=200, predict=200; agreement=True; exact probabilities=True; maximum absolute probability difference=0.0. Mixed boolean, additional metadata and missing-sensor requests each returned 422.

## Actual Docker build and run

Docker status: **passed**.

Engine: `29.9.2`; image ID: `sha256:607337719ad9290785cba9b0c95b2d36e10dbea4e39c4ae308ebcc7f3fc1d8d4`.

`docker build --progress=plain -t hydroguard:phase4 .`: exit 0, 1.445 seconds. Full local build log: `tmp/phase4/docker-build.log`.

The smoke container started without bind mounts, with a read-only filesystem, UID 10001, dropped capabilities and localhost-only port publication. Its Docker HEALTHCHECK reached healthy. Runtime filesystem inspection confirmed only the selected model bundle under /app/model and the serving environment; no dataset, splits, historical reports, training modules, tests, pytest, Ruff or httpx. The image has a default-deny context and locked non-editable runtime install. The smoke container was stopped and removed after verification. The image remains local.

Container inspection: `{'uid': 10001, 'app_entries': ['model', 'serving'], 'model_entries': ['metadata.json', 'pipeline.pkl'], 'hydroguard_modules': ['__init__.py', 'data.py', 'features.py', 'inference.py'], 'development_packages_present': []}`.

## Checks and limitations

Earlier interrupted verification attempts (if any) are preserved below. The initial Windows smoke runner encountered a temporary-directory cleanup sharing violation after its successful local HTTP checks, before Docker checks. Bounded cleanup retries now handle that transient OS condition.

```json
[
  {
    "started_at_utc": "2026-10-30T23:23:56.089967+00:00",
    "completed_at_utc": "2026-10-30T23:23:59.404998+00:00",
    "failure": {
      "type": "PermissionError",
      "message": "[WinError 32] The process cannot access the file because it is being used by another process: 'C:\\\\Users\\\\iTC\\\\AppData\\\\Local\\\\Temp\\\\hydroguard-serving-c3b4yw7t'"
    }
  }
]
```

```json
{
  "historical_tests": {
    "tests": 297,
    "failures": 0,
    "errors": 0,
    "skipped": 0
  },
  "api_tests": {
    "tests": 117,
    "failures": 0,
    "errors": 0,
    "skipped": 0
  },
  "quality_commands": [
    {
      "command": [
        "uv",
        "run",
        "--locked",
        "ruff",
        "check",
        "."
      ],
      "exit_code": 0,
      "seconds": 0.04530050000175834,
      "log": "tmp/phase4/quality-1.log",
      "stdout": "All checks passed!",
      "stderr": ""
    },
    {
      "command": [
        "uv",
        "run",
        "--locked",
        "ruff",
        "format",
        "--check",
        "."
      ],
      "exit_code": 0,
      "seconds": 0.05671529998653568,
      "log": "tmp/phase4/quality-2.log",
      "stdout": "47 files already formatted",
      "stderr": ""
    },
    {
      "command": [
        "uv",
        "run",
        "--project",
        "serving",
        "--locked",
        "--no-editable",
        "ruff",
        "check",
        "serving"
      ],
      "exit_code": 0,
      "seconds": 0.04315880002104677,
      "log": "tmp/phase4/quality-3.log",
      "stdout": "All checks passed!",
      "stderr": ""
    },
    {
      "command": [
        "uv",
        "run",
        "--project",
        "serving",
        "--locked",
        "--no-editable",
        "ruff",
        "format",
        "--check",
        "serving"
      ],
      "exit_code": 0,
      "seconds": 0.043241700012004,
      "log": "tmp/phase4/quality-4.log",
      "stdout": "7 files already formatted",
      "stderr": ""
    },
    {
      "command": [
        "uv",
        "lock",
        "--check"
      ],
      "exit_code": 0,
      "seconds": 0.018477999983588234,
      "log": "tmp/phase4/quality-5.log",
      "stdout": "",
      "stderr": "Resolved 66 packages in 0.59ms"
    },
    {
      "command": [
        "uv",
        "lock",
        "--project",
        "serving",
        "--check"
      ],
      "exit_code": 0,
      "seconds": 0.01680589999887161,
      "log": "tmp/phase4/quality-6.log",
      "stdout": "",
      "stderr": "Resolved 37 packages in 0.51ms"
    }
  ]
}
```

The API tests use the existing saved bundle and synthetic requests; no estimator is fitted. A Starlette TestClient deprecation warning reports that its httpx adapter should eventually migrate to httpx2; current tests and actual Uvicorn requests work. No warning is hidden. Docker base tags are versioned; the actual image ID above records this build, rather than claiming identical future tag contents.

Historical final-test macro-F1 is **0.675047** and weak-leakage recall is **18/103 (0.174757)**. These are the existing Phase 3B results, not a Phase 4 reevaluation. All test cycles were previously exposed in EDA. Group independence and gap-20 physical decorrelation are unverified; this single controlled rig and small unstable fault subgroups limit generalization. Model probabilities are uncalibrated. This service is an academic demonstrator, not an industrial diagnostic or safety system. Pickle loading is limited to the trusted pinned local bundle; checksums are integrity checks, not malicious-source authentication. The service is intended for localhost use; no cloud deployment, API authentication or production traffic controls are included.

Reproduction: follow the Phase 4 commands in README.md. To rerun this smoke report, after the listed tests and checks: `uv run --project serving --locked --no-editable python serving/scripts/verify_serving.py`.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), [FastAPI validation handlers](https://fastapi.tiangolo.com/tutorial/handling-errors/), [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).

## README client commands verified

The documented `docker run --rm --detach --name hydroguard-phase4` command on port 8001 was executed. Both `curl.exe` and PowerShell `Invoke-RestMethod` successfully called `/health` and `/predict` with the saved development example. Both parsed prediction responses were exactly equal to the direct prediction, including probabilities and model version. `docker stop hydroguard-phase4` succeeded and automatic removal was verified. The JSON companion contains the actual responses and container ID.
