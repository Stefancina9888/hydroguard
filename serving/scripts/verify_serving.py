"""Run direct/local HTTP/Docker checks on a saved development example, never on test data."""

import copy
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from hydroguard.data import sha256_file

from hydroguard_serving.runtime import Runtime

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/serving-verification.json"
LOGS = ROOT / "tmp/phase4"
# Declared before observing cross-platform predictions; no model decision is changed.
PROBABILITY_ATOL = 1e-12
PROBABILITY_RTOL = 1e-12


def now():
    return datetime.now(UTC).isoformat()


def command(arguments, log_name, timeout=60):
    started = time.perf_counter()
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"},
    )
    log = LOGS / log_name
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return {
        "command": arguments,
        "exit_code": completed.returncode,
        "seconds": time.perf_counter() - started,
        "log": log.relative_to(ROOT).as_posix(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def http_json(url, payload=None):
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_ready(url):
    deadline = time.monotonic() + 45
    last = None
    while time.monotonic() < deadline:
        try:
            status, response = http_json(url + "/health")
            if status == 200 and response.get("ready") is True:
                return response
        except (OSError, ValueError) as error:
            last = type(error).__name__
        time.sleep(0.25)
    raise RuntimeError(f"Service did not become ready: {last}")


@contextmanager
def isolated_working_directory():
    temporary = tempfile.TemporaryDirectory(prefix="isolated-server-", dir=LOGS)
    # Verify the managed recursive-cleanup target belongs to this task's log directory.
    assert Path(temporary.name).resolve().parent == LOGS.resolve()
    try:
        yield temporary.name
    finally:
        # Windows can retain the exited Python launcher's cwd handle briefly.
        for attempt in range(20):
            try:
                temporary.cleanup()
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)


def probe(url, payload, direct):
    health = wait_ready(url)
    assert health == {"ready": True, "model_version": direct["model_version"]}
    status, response = http_json(url + "/predict", payload)
    assert status == 200
    assert set(response) == set(direct)
    for name in ("class", "label", "classes", "model_version"):
        assert response[name] == direct[name]
    observed, expected = np.asarray(response["probabilities"]), np.asarray(direct["probabilities"])
    np.testing.assert_allclose(observed, expected, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL)
    invalid_results = {}
    for name in ("mixed_boolean", "extra_metadata", "missing_sensor"):
        invalid = copy.deepcopy(payload)
        if name == "mixed_boolean":
            invalid["sensors"]["PS1"][5] = True
        elif name == "extra_metadata":
            invalid["cycle_id"] = 1
        else:
            del invalid["sensors"]["VS1"]
        invalid_status, detail = http_json(url + "/predict", invalid)
        assert invalid_status == 422
        assert len(json.dumps(detail)) < 6000
        assert all("input" not in item and "ctx" not in item for item in detail["detail"])
        invalid_results[name] = {"http_status": invalid_status, "response": detail}
    return {
        "url": url,
        "health_http_status": 200,
        "health": health,
        "predict_http_status": status,
        "prediction": response,
        "probabilities_exactly_equal": bool(np.array_equal(observed, expected)),
        "max_probability_difference": float(np.max(np.abs(observed - expected))),
        "agreement_with_direct": True,
        "invalid_requests": invalid_results,
    }


def historical_integrity():
    snapshot = json.loads((ROOT / "serving/historical-checksums.json").read_text())
    changed = [name for name, expected in snapshot.items() if sha256_file(ROOT / name) != expected]
    assert not changed, changed
    metadata = json.loads((ROOT / "artifacts/models/phase3b-v1/metadata.json").read_text())
    for name, expected in metadata["input_checksums"]["extracted_sha256"].items():
        assert sha256_file(ROOT / "data/raw/uci447" / name) == expected
    assert (
        sha256_file(ROOT / "data/raw/uci447.zip") == metadata["input_checksums"]["archive_sha256"]
    )
    return {
        "historical_files": len(snapshot),
        "all_unchanged": True,
        "raw_archive_and_files_unchanged": True,
    }


def test_counts(path):
    suite = ET.parse(path).getroot().find("testsuite")
    counts = {key: int(suite.attrib[key]) for key in ["tests", "failures", "errors", "skipped"]}
    assert counts["tests"] > 0
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
    return counts


def quality_commands():
    commands = [
        ["uv", "run", "--locked", "ruff", "check", "."],
        ["uv", "run", "--locked", "ruff", "format", "--check", "."],
        [
            "uv",
            "run",
            "--project",
            "serving",
            "--locked",
            "--no-editable",
            "ruff",
            "check",
            "serving",
        ],
        [
            "uv",
            "run",
            "--project",
            "serving",
            "--locked",
            "--no-editable",
            "ruff",
            "format",
            "--check",
            "serving",
        ],
        ["uv", "lock", "--check"],
        ["uv", "lock", "--project", "serving", "--check"],
    ]
    results = [command(args, f"quality-{i + 1}.log") for i, args in enumerate(commands)]
    assert all(item["exit_code"] == 0 for item in results), "Quality checks failed"
    return results


def write_report(result):
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    docker = result.get("docker", {})
    lines = [
        "# Phase 4 — serving verification",
        "",
        f"Actual execution started: {result['started_at_utc']}; "
        f"completed: {result.get('completed_at_utc', 'incomplete')}. UTC.",
        "",
        "This is a local academic demonstrator for the unchanged Phase 3B Logistic Regression "
        "lr_09 Pipeline (C=100, class_weight=None). No training, tuning, threshold change, "
        "new feature, final-test prediction or final-test evaluation was performed in Phase 4.",
        "",
        "## Environment and locked history",
        "",
        f"Model version: `{result['model_version']}`.",
        "",
        "Numerical environment matches the saved metadata exactly:",
        "",
        "```json",
        json.dumps(result["versions"], indent=2),
        "```",
        "",
        f"Serving versions: `{result['serving_versions']}`. "
        "The independent `serving/pyproject.toml`, `.python-version` and `uv.lock` preserve "
        "the root environment. The wheel packages byte-identical copies of only four "
        "historical modules: __init__, data, features and inference. Runtime pins both bundle "
        "file hashes and those source hashes, then uses the historical loader to verify "
        "metadata, fitted Pipeline, 70-column contract, classes and versions. A mismatch "
        "fails startup. The bundle is loaded once per application lifespan, never from HTTP.",
        "",
        f"Historical integrity: `{result.get('integrity')}`.",
        "",
        "## Raw input boundary and endpoint checks",
        "",
        'POST /predict accepts only {"sensors": {...}} with exactly 14 named physical '
        "sensors. Each raw sample must have Python type int or float before NumPy conversion; "
        "bool (including mixed float/bool lists), strings, null, complex values and nonfinite "
        "numbers are rejected. Vectors must be JSON lists of exactly rate × 60 numeric scalars. "
        "Metadata, additional root/sensor fields, wrong dimensions and wrong lengths return 422. "
        "The historical inference.py remains unchanged. Overflowing derived statistics also "
        "produce 422. Error responses omit input/context and cap displayed errors at eight; "
        "they never echo sensor arrays. /health returns readiness and the pinned model version; "
        "an absent runtime returns 503.",
        "",
        "## Development example and actual HTTP agreement",
        "",
        "Only original development cycle 1 is used. The payload has no metadata; its separate "
        "provenance includes source attribution, development membership and request checksum. "
        "The API needs no raw dataset, splits or training reports. The local Uvicorn smoke "
        "server was launched in an empty temporary working directory with an explicit local "
        "model path. Direct, HTTP and container probability agreement was prespecified at "
        "atol=rtol=1e-12 for platform numerical variation; class, label, classes and model "
        "version must match exactly.",
        "",
        f"Request SHA-256: `{result['request_sha256']}`.",
        "",
        "Direct prediction:",
        "",
        "```json",
        json.dumps(result["direct_prediction"], indent=2),
        "```",
        "",
    ]
    for name, details in [
        ("Local HTTP", result.get("local_http")),
        ("Docker HTTP", docker.get("http")),
    ]:
        if details:
            lines.extend(
                [
                    f"{name}: health={details['health_http_status']}, "
                    f"predict={details['predict_http_status']}; "
                    f"agreement={details['agreement_with_direct']}; "
                    f"exact probabilities={details['probabilities_exactly_equal']}; "
                    "maximum absolute probability difference="
                    f"{details['max_probability_difference']}. "
                    "Mixed boolean, additional metadata and missing-sensor requests "
                    "each returned 422.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Actual Docker build and run",
            "",
            f"Docker status: **{docker.get('status', 'not executed')}**.",
            "",
        ]
    )
    if docker.get("status") == "passed":
        lines.extend(
            [
                f"Engine: `{docker['engine']['stdout']}`; image ID: `{docker['image_id']}`.",
                "",
                "`docker build --progress=plain -t hydroguard:phase4 .`: "
                f"exit {docker['build']['exit_code']}, {docker['build']['seconds']:.3f} seconds. "
                f"Full local build log: `{docker['build']['log']}`.",
                "",
                "The smoke container started without bind mounts, with a read-only filesystem, "
                "UID 10001, dropped capabilities and localhost-only port publication. Its Docker "
                "HEALTHCHECK reached healthy. Runtime filesystem inspection confirmed only the "
                "selected model bundle under /app/model and the serving environment; no dataset, "
                "splits, historical reports, training modules, tests, pytest, Ruff or httpx. "
                "The image has a default-deny context and locked non-editable runtime install. "
                "The smoke container was stopped and removed after verification. "
                "The image remains local.",
                "",
                f"Container inspection: `{docker['runtime_inspection']}`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Docker build/run/health/predict have not all succeeded. See the JSON record "
                "for the exact failure or unavailable daemon; no container-success claim is made.",
                "",
            ]
        )
    lines.extend(
        [
            "## Checks and limitations",
            "",
            "Earlier interrupted verification attempts (if any) are preserved below. "
            "The initial Windows smoke runner encountered a temporary-directory cleanup "
            "sharing violation after its successful local HTTP checks, before Docker "
            "checks. Bounded cleanup retries now handle that transient OS condition.",
            "",
            "```json",
            json.dumps(result.get("prior_attempts", []), indent=2),
            "```",
            "",
            "```json",
            json.dumps(result.get("checks", {}), indent=2),
            "```",
            "",
            "The API tests use the existing saved bundle and synthetic requests; no estimator "
            "is fitted. A Starlette TestClient deprecation warning reports that its httpx adapter "
            "should eventually migrate to httpx2; current tests and actual Uvicorn requests work. "
            "No warning is hidden. Docker base tags are versioned; the actual image ID above "
            "records this build, rather than claiming identical future tag contents.",
            "",
            "Historical final-test macro-F1 is **0.675047** and weak-leakage recall is "
            "**18/103 (0.174757)**. These are the existing Phase 3B results, not a Phase 4 "
            "reevaluation. All test cycles were previously exposed in EDA. Group independence "
            "and gap-20 physical decorrelation are unverified; this single controlled rig "
            "and small unstable fault subgroups limit generalization. Model probabilities "
            "are uncalibrated. This service is an academic demonstrator, not an industrial "
            "diagnostic or safety system. Pickle loading is limited to the trusted pinned local "
            "bundle; checksums are integrity checks, not malicious-source authentication. "
            "The service is intended for localhost use; no cloud deployment, API authentication "
            "or production traffic controls are included.",
            "",
            "Reproduction: follow the Phase 4 commands in README.md. To rerun this smoke report, "
            "after the listed tests and checks: "
            "`uv run --project serving --locked --no-editable python "
            "serving/scripts/verify_serving.py`.",
            "",
            "Implementation references: "
            "[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), "
            "[FastAPI validation handlers]"
            "(https://fastapi.tiangolo.com/tutorial/handling-errors/), "
            "[uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).",
            "",
        ]
    )
    (ROOT / "docs/serving-verification.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    payload_path = ROOT / "serving/examples/development-cycle.json"
    payload = json.loads(payload_path.read_text())
    provenance = json.loads((payload_path.parent / "provenance.json").read_text())
    assert sha256_file(payload_path) == provenance["request_sha256"]
    bundle_directory = ROOT / "artifacts/models/phase3b-v1"
    runtime = Runtime(bundle_directory)
    assert provenance["cycle_id"] in runtime.bundle.metadata["training"]["cycle_ids"]
    direct = runtime.predict(payload)
    assert direct == {
        **provenance["direct_historical_prediction"],
        "model_version": runtime.version,
    }
    prior_attempts = []
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text())
        prior_attempts = previous.get("prior_attempts", [])
        if "failure" in previous:
            prior_attempts.append(
                {key: previous[key] for key in ["started_at_utc", "completed_at_utc", "failure"]}
            )
    result = {
        "prior_attempts": prior_attempts,
        "started_at_utc": now(),
        "model_version": runtime.version,
        "versions": runtime.bundle.metadata["versions"],
        "serving_versions": {
            name: importlib.metadata.version(name)
            for name in ["fastapi", "uvicorn", "pydantic", "starlette"]
        },
        "request_sha256": sha256_file(payload_path),
        "direct_prediction": direct,
        "probability_comparison": {"atol": PROBABILITY_ATOL, "rtol": PROBABILITY_RTOL},
        "integrity": historical_integrity(),
        "checks": {
            "historical_tests": test_counts(LOGS / "historical-tests.xml"),
            "api_tests": test_counts(LOGS / "api-tests.xml"),
            "quality_commands": quality_commands(),
        },
        "docker": {"status": "not executed"},
    }
    # Save the running record before any subprocess operation.
    write_report(result)
    container_name = "hydroguard-phase4-smoke-" + uuid.uuid4().hex[:10]
    container_started = False
    try:
        with isolated_working_directory() as working:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            environment = {**os.environ, "HYDROGUARD_MODEL_DIR": str(bundle_directory)}
            with (LOGS / "local-uvicorn.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "hydroguard_serving.app:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--workers",
                        "1",
                    ],
                    cwd=working,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                try:
                    result["local_http"] = probe(f"http://127.0.0.1:{port}", payload, direct)
                finally:
                    process.terminate()
                    process.wait(timeout=15)
        if not shutil.which("docker"):
            result["docker"] = {
                "status": "unavailable",
                "reason": "Docker CLI not installed; build/run/health/predict not executed",
            }
            return
        engine = command(["docker", "info", "--format", "{{.ServerVersion}}"], "docker-info.log")
        result["docker"]["engine"] = engine
        if engine["exit_code"]:
            result["docker"].update(
                status="unavailable",
                reason="Docker daemon unavailable; build/run/health/predict not executed",
            )
            return
        build = command(
            ["docker", "build", "--progress=plain", "-t", "hydroguard:phase4", "."],
            "docker-build.log",
            timeout=600,
        )
        result["docker"]["build"] = {
            key: value for key, value in build.items() if key not in ["stdout", "stderr"]
        }
        assert build["exit_code"] == 0, "Docker build failed; inspect log"
        result["docker"]["image_id"] = command(
            ["docker", "image", "inspect", "--format", "{{.Id}}", "hydroguard:phase4"],
            "docker-image.log",
        )["stdout"]
        run = command(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                container_name,
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--publish",
                "127.0.0.1::8000",
                "hydroguard:phase4",
            ],
            "docker-run.log",
        )
        assert run["exit_code"] == 0, "Docker run failed"
        container_started = True
        result["docker"]["run"] = run
        port_result = command(["docker", "port", container_name, "8000/tcp"], "docker-port.log")
        url = "http://" + port_result["stdout"]
        result["docker"]["http"] = probe(url, payload, direct)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            health = command(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
                "docker-health.log",
            )
            if health["stdout"] == "healthy":
                break
            time.sleep(0.5)
        assert health["stdout"] == "healthy"
        result["docker"]["healthcheck"] = health
        inspection_code = (
            "import json, pathlib, importlib.metadata as m, os; "
            "import hydroguard; "
            "p=pathlib.Path(hydroguard.__file__).parent; "
            "r={'uid':os.getuid(),"
            "'app_entries':sorted(x.name for x in pathlib.Path('/app').iterdir()),"
            "'model_entries':sorted(x.name for x in pathlib.Path('/app/model').iterdir()),"
            "'hydroguard_modules':sorted(x.name for x in p.glob('*.py')),"
            "'development_packages_present':sorted(set(['pytest','ruff','httpx']) & "
            "{d.metadata['Name'].lower() for d in m.distributions()})}; "
            "print(json.dumps(r)); assert r['uid']==10001; "
            "assert r['app_entries']==['model','serving']; "
            "assert r['model_entries']==['metadata.json','pipeline.pkl']; "
            "assert r['hydroguard_modules']=="
            "['__init__.py','data.py','features.py','inference.py']; "
            "assert not r['development_packages_present']"
        )
        inspected = command(
            ["docker", "exec", container_name, "python", "-c", inspection_code],
            "docker-runtime-inspection.log",
        )
        assert inspected["exit_code"] == 0, "Runtime isolation inspection failed"
        result["docker"]["runtime_inspection"] = json.loads(inspected["stdout"])
        result["docker"]["status"] = "passed"
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        if result["docker"]["status"] != "passed":
            result["docker"]["status"] = "failed"
        raise
    finally:
        if container_started:
            result["docker"]["logs"] = command(
                ["docker", "logs", container_name], "docker-server.log"
            )
            removed = command(["docker", "rm", "--force", container_name], "docker-stop.log")
            result["docker"]["container_removed"] = removed["exit_code"] == 0
            if removed["exit_code"]:
                result["docker"]["status"] = "failed"
        result["integrity"] = historical_integrity()
        result["completed_at_utc"] = now()
        write_report(result)
        print(
            json.dumps(
                {
                    "local_http": result.get("local_http", {}).get("agreement_with_direct"),
                    "docker": result["docker"]["status"],
                    "integrity": result["integrity"],
                    "report": str(OUTPUT),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
