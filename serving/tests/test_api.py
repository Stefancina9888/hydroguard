"""Serving regressions without fitting any estimator or reading the raw dataset."""

import copy
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hydroguard.data import DataValidationError
from hydroguard.inference import predict_cycle
from pydantic import ValidationError

from hydroguard_serving.app import create_app
from hydroguard_serving.runtime import Runtime, model_contract
from hydroguard_serving.validation import PredictionRequest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/models/phase3b-v1"
CONTRACT = model_contract()


@pytest.fixture
def payload():
    return {
        "sensors": {
            name: [1.25] * description["samples"]
            for name, description in json.loads((BUNDLE / "metadata.json").read_text())[
                "sensor_contract"
            ]["sensors"].items()
        }
    }


@pytest.fixture(scope="module")
def runtime():
    return Runtime(BUNDLE)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(BUNDLE)) as client:
        yield client


def assert_safe_error(response):
    assert response.status_code == 422
    assert len(response.content) < 6000
    details = response.json()["detail"]
    if isinstance(details, list):
        assert len(details) <= 8
        assert all(set(item) == {"loc", "msg", "type"} for item in details)


def test_health_and_direct_http_prediction_agree(client, runtime, payload):
    assert client.get("/health").json() == {"ready": True, "model_version": runtime.version}
    direct = runtime.predict(payload)
    historical = predict_cycle(payload["sensors"], runtime.bundle)
    assert direct == {**historical, "model_version": runtime.version}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == direct
    assert direct["classes"] == [0, 1, 2]
    assert len(direct["probabilities"]) == 3
    assert sum(direct["probabilities"]) == pytest.approx(1)


@pytest.mark.parametrize("value", [True, False, "1.25", None, [1.25], {"value": 1.25}])
def test_mixed_non_numbers_rejected_before_historical_numpy(client, runtime, payload, value):
    payload["sensors"]["PS1"][15] = value
    with patch("hydroguard_serving.runtime.predict_cycle") as inference:
        with pytest.raises(ValidationError):
            runtime.predict(payload)
        assert_safe_error(client.post("/predict", json=payload))
    inference.assert_not_called()


@pytest.mark.parametrize("value", [1 + 2j, float("nan"), float("inf"), -float("inf"), 10**400])
def test_python_public_boundary_rejects_complex_nonfinite_and_overflow(runtime, payload, value):
    payload["sensors"]["PS1"][3] = value
    with patch("hydroguard_serving.runtime.predict_cycle") as inference:
        with pytest.raises(ValidationError):
            runtime.predict(payload)
    inference.assert_not_called()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_http_nonfinite_numbers_are_safe_422(client, payload, token):
    raw = json.dumps(payload).replace("1.25", token, 1)
    assert_safe_error(
        client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    )


def test_pure_boolean_arrays_rejected_without_echoing_array(client, payload):
    payload["sensors"]["PS1"] = [True] * 6000
    response = client.post("/predict", json=payload)
    assert_safe_error(response)
    assert response.json()["error_count"] == 6000
    assert "input" not in response.json()["detail"][0]


@pytest.mark.parametrize("sensor", CONTRACT["historical_source_sha256"].keys())
def test_historical_source_pins_are_present(sensor):
    from hydroguard.data import sha256_file

    assert (
        sha256_file(ROOT / "src/hydroguard" / sensor)
        == (CONTRACT["historical_source_sha256"][sensor])
    )


@pytest.mark.parametrize(
    "sensor",
    [
        "PS1",
        "PS2",
        "PS3",
        "PS4",
        "PS5",
        "PS6",
        "EPS1",
        "FS1",
        "FS2",
        "TS1",
        "TS2",
        "TS3",
        "TS4",
        "VS1",
    ],
)
@pytest.mark.parametrize("change", ["missing", "short", "long", "nested"])
def test_all_sensor_memberships_lengths_and_dimensions(client, payload, sensor, change):
    if change == "missing":
        del payload["sensors"][sensor]
    elif change == "short":
        payload["sensors"][sensor].pop()
    elif change == "long":
        payload["sensors"][sensor].append(1.25)
    else:
        payload["sensors"][sensor] = [payload["sensors"][sensor]]
    assert_safe_error(client.post("/predict", json=payload))


@pytest.mark.parametrize(
    "field",
    [
        "cycle_id",
        "target",
        "pump_leakage",
        "stability_flag",
        "group_id",
        "CE",
        "CP",
        "SE",
        "model",
        "metadata",
    ],
)
@pytest.mark.parametrize("level", ["root", "sensors"])
def test_extra_fields_metadata_and_model_uploads_rejected(client, payload, field, level):
    target = payload if level == "root" else payload["sensors"]
    target[field] = [424242] * 100
    response = client.post("/predict", json=payload)
    assert_safe_error(response)
    assert "424242" not in response.text


@pytest.mark.parametrize(
    "body", ["null", "[]", '{"sensors": null}', '{"sensors": []}', '{"sensors":', '"secret text"']
)
def test_invalid_envelopes_and_json_do_not_echo_input(client, body):
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert_safe_error(response)
    assert "secret text" not in response.text


def test_finite_numbers_with_overflowing_statistics_are_422(client, payload):
    payload["sensors"]["PS1"][0] = 1e308
    assert_safe_error(client.post("/predict", json=payload))


def test_integer_samples_allowed_but_coercible_strings_are_not(payload):
    payload["sensors"]["VS1"] = list(range(60))
    parsed = PredictionRequest.model_validate(payload)
    assert parsed.sensors.VS1 == [float(i) for i in range(60)]


def test_startup_loads_once_and_requests_do_not_reload(payload):
    import hydroguard_serving.runtime as module

    with patch.object(module, "load_model_bundle", wraps=module.load_model_bundle) as load:
        with TestClient(create_app(BUNDLE)) as running:
            assert running.get("/health").status_code == 200
            assert running.post("/predict", json=payload).status_code == 200
            assert running.post("/predict", json=payload).status_code == 200
        assert load.call_count == 1


@pytest.mark.parametrize("problem", ["missing", "model", "metadata", "both", "versions", "source"])
def test_startup_refuses_bad_or_incompatible_bundle_before_unpickling(tmp_path, problem):
    shutil.copyfile(BUNDLE / "pipeline.pkl", tmp_path / "pipeline.pkl")
    shutil.copyfile(BUNDLE / "metadata.json", tmp_path / "metadata.json")
    if problem in ["model", "both"]:
        (tmp_path / "pipeline.pkl").write_bytes(b"changed model")
    if problem in ["metadata", "both", "versions"]:
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        metadata["model_sha256"] = "0" * 64
        metadata["versions"]["numpy"] = "0.0.0"
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    if problem == "missing":
        (tmp_path / "pipeline.pkl").unlink()
    modified = copy.deepcopy(CONTRACT)
    if problem == "versions":
        # Even a revised local bundle pin cannot bypass historical version validation.
        import hashlib

        modified["bundle_sha256"]["metadata.json"] = hashlib.sha256(
            (tmp_path / "metadata.json").read_bytes()
        ).hexdigest()
    if problem == "source":
        modified["historical_source_sha256"]["inference.py"] = "0" * 64
    with patch("hydroguard_serving.runtime.model_contract", return_value=modified):
        with patch("hydroguard.inference.pickle.loads") as unpickle:
            with pytest.raises((DataValidationError, OSError)):
                with TestClient(create_app(tmp_path)):
                    pass
    unpickle.assert_not_called()


def test_service_works_in_directory_with_only_model_bundle(tmp_path, payload, monkeypatch):
    directory = tmp_path / "model"
    directory.mkdir()
    for name in ["pipeline.pkl", "metadata.json"]:
        shutil.copyfile(BUNDLE / name, directory / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HYDROGUARD_MODEL_DIR", str(directory))
    with TestClient(create_app()) as isolated:
        assert isolated.get("/health").json()["ready"]
        assert isolated.post("/predict", json=payload).status_code == 200
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model"]


def test_not_ready_is_503_without_lifespan():
    with patch("hydroguard_serving.app.Runtime"):
        application = create_app(BUNDLE)
        client = TestClient(application)
        assert client.get("/health").status_code == 503


def test_openapi_describes_required_lengths_and_forbids_extra_fields(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert schemas["PredictionRequest"]["additionalProperties"] is False
    sensors = schemas["Sensors"]
    assert sensors["additionalProperties"] is False
    assert len(sensors["required"]) == 14
    for name, size in [("PS1", 6000), ("FS1", 600), ("TS1", 60)]:
        assert sensors["properties"][name]["minItems"] == size
        assert sensors["properties"][name]["maxItems"] == size


def test_runtime_never_mutates_raw_payload(runtime, payload):
    before = copy.deepcopy(payload)
    runtime.predict(payload)
    assert payload == before


def test_selected_development_example_matches_recorded_direct_prediction(client):
    path = ROOT / "serving/examples/development-cycle.json"
    payload = json.loads(path.read_text())
    provenance = json.loads((path.parent / "provenance.json").read_text())
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    expected = provenance["direct_historical_prediction"]
    observed = response.json()
    assert observed == {**expected, "model_version": CONTRACT["model_version"]}
    np.testing.assert_array_equal(observed["probabilities"], expected["probabilities"])
