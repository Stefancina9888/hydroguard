"""Offline raw-cycle validation and complete-Pipeline persistence with synthetic data."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hydroguard.baselines import make_model
from hydroguard.data import SENSOR_RATES, TARGET_LABELS, DataValidationError
from hydroguard.features import FEATURE_CONTRACT, FEATURE_NAMES, PHYSICAL_SENSORS, extract_features
from hydroguard.final_model import save_pipeline_bundle
from hydroguard.inference import (
    AUTHORIZED_PARAMETERS,
    SENSOR_CONTRACT,
    cycle_features,
    library_versions,
    load_model_bundle,
    predict_cycle,
    predict_feature_rows,
)


@pytest.fixture
def raw_cycle():
    return {
        sensor: np.full(SENSOR_RATES[sensor] * 60, 2.0, dtype=np.float64)
        for sensor in PHYSICAL_SENSORS
    }


@pytest.fixture(scope="module")
def synthetic_pipeline():
    signals = {
        sensor: np.repeat(np.arange(1.0, 10.0)[:, None], 3, axis=1) for sensor in PHYSICAL_SENSORS
    }
    batch = extract_features(signals, np.arange(1, 10))
    model = make_model(
        "logistic_regression", {"models": {"logistic_regression": AUTHORIZED_PARAMETERS}}
    )
    # This is a nine-row synthetic fixture, never a fit on HydroGuard raw data.
    model.fit(batch.X, np.repeat([0, 1, 2], 3))
    metadata = {
        "schema_version": 1,
        "phase": "3B",
        "candidate_id": "lr_09",
        "parameters": copy.deepcopy(AUTHORIZED_PARAMETERS),
        "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(FEATURE_NAMES),
        "sensor_contract": SENSOR_CONTRACT,
        "classes_": [0, 1, 2],
        "class_mapping": {str(k): value for k, value in TARGET_LABELS.items()},
        "versions": library_versions(),
        "serialization": {"format": "python-pickle", "protocol": 5},
        "training": {"cycle_ids": list(range(1, 10)), "cycles": 9, "pipeline_fits": 1},
    }
    return model, batch.X, metadata


@pytest.fixture
def bundle(tmp_path, synthetic_pipeline):
    model, features, metadata = synthetic_pipeline
    directory = tmp_path / "model"
    saved = save_pipeline_bundle(model, directory, metadata, features)
    return directory, load_model_bundle(directory), saved


def test_raw_cycle_features_use_shared_order_and_correct_population_summaries(raw_cycle):
    frame = cycle_features(raw_cycle)
    assert frame.shape == (1, 70) and tuple(frame.columns) == FEATURE_NAMES
    assert frame.iloc[0, :5].tolist() == [2.0, 0.0, 2.0, 2.0, 2.0]
    pd.testing.assert_frame_equal(frame, cycle_features(dict(reversed(list(raw_cycle.items())))))
    assert SENSOR_CONTRACT["sensors"]["PS1"]["samples"] == 6000
    assert SENSOR_CONTRACT["sensors"]["FS1"]["samples"] == 600
    assert SENSOR_CONTRACT["sensors"]["TS1"]["samples"] == 60


@pytest.mark.parametrize("sensor", PHYSICAL_SENSORS)
def test_wrong_rate_based_length_rejected(raw_cycle, sensor):
    raw_cycle[sensor] = raw_cycle[sensor][:-1]
    with pytest.raises(DataValidationError, match=f"{sensor}.*samples"):
        cycle_features(raw_cycle)


@pytest.mark.parametrize("sensor", PHYSICAL_SENSORS)
def test_missing_sensor_rejected(raw_cycle, sensor):
    del raw_cycle[sensor]
    with pytest.raises(DataValidationError, match="exactly 14"):
        cycle_features(raw_cycle)


@pytest.mark.parametrize(
    "extra",
    [
        "CE",
        "CP",
        "SE",
        "cycle_id",
        "target",
        "pump_leakage",
        "stability_flag",
        "cooler_condition_pct",
        "valve_condition_pct",
        "accumulator_pressure_bar",
        "group_id",
    ],
)
def test_extra_sensors_and_metadata_rejected(raw_cycle, extra):
    raw_cycle[extra] = np.ones(60)
    with pytest.raises(DataValidationError, match="extra="):
        cycle_features(raw_cycle)


@pytest.mark.parametrize("bad", [np.zeros((1, 6000)), np.zeros((6000, 1)), 1.0, [], None])
def test_wrong_dimensions_rejected(raw_cycle, bad):
    raw_cycle["PS1"] = bad
    with pytest.raises(DataValidationError, match="PS1.*1D"):
        cycle_features(raw_cycle)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_sample_rejected(raw_cycle, value):
    raw_cycle["PS1"][123] = value
    with pytest.raises(DataValidationError, match="finite real numeric"):
        cycle_features(raw_cycle)


@pytest.mark.parametrize("dtype", [bool, complex, str, object])
def test_non_real_numeric_dtypes_are_not_silently_coerced(raw_cycle, dtype):
    raw_cycle["PS1"] = np.ones(6000).astype(dtype)
    with pytest.raises(DataValidationError, match="finite real numeric"):
        cycle_features(raw_cycle)


def test_model_predictions_and_probabilities_exactly_match_after_load(bundle, synthetic_pipeline):
    directory, loaded, metadata = bundle
    original, features, _ = synthetic_pipeline
    predicted, probability = predict_feature_rows(loaded, features)
    np.testing.assert_array_equal(predicted, original.predict(features))
    np.testing.assert_array_equal(probability, original.predict_proba(features))
    assert metadata["serialization_check"]["predict_equal"]
    assert metadata["serialization_check"]["predict_proba_equal"]
    assert metadata["serialization_check"]["max_probability_difference"] == 0
    with pytest.raises(DataValidationError, match="replacement is forbidden"):
        save_pipeline_bundle(original, directory, metadata, features)


def test_single_cycle_prediction_is_portable_and_probabilities_follow_classes(bundle, raw_cycle):
    _, loaded, _ = bundle
    with patch(
        "hydroguard.data.load_matrix",
        side_effect=AssertionError("Inference must not read a dataset"),
    ):
        result = predict_cycle(raw_cycle, loaded)
    assert result["classes"] == loaded.pipeline.classes_.tolist() == [0, 1, 2]
    assert result["label"] == TARGET_LABELS[result["class"]]
    np.testing.assert_allclose(sum(result["probabilities"]), 1.0, atol=1e-12)
    np.testing.assert_array_equal(
        result["probabilities"], loaded.pipeline.predict_proba(cycle_features(raw_cycle))[0]
    )


def test_corrupt_model_is_rejected_before_unpickling(bundle):
    directory, _, _ = bundle
    with (directory / "pipeline.pkl").open("ab") as stream:
        stream.write(b"corrupt")
    with patch("hydroguard.inference.pickle.loads") as loads:
        with pytest.raises(DataValidationError, match="checksum mismatch"):
            load_model_bundle(directory)
    loads.assert_not_called()


@pytest.mark.parametrize(
    "problem", ["versions", "feature_order", "classes", "mapping", "params", "training_count"]
)
def test_invalid_metadata_rejected_before_unpickling(bundle, problem):
    directory, _, _ = bundle
    path = directory / "metadata.json"
    meta = json.loads(path.read_text())
    if problem == "versions":
        meta["versions"]["scikit_learn"] = "0.0"
    elif problem == "feature_order":
        meta["feature_names"] = meta["feature_names"][::-1]
    elif problem == "classes":
        meta["classes_"] = [2, 1, 0]
    elif problem == "mapping":
        meta["class_mapping"]["2"] = "wrong label"
    elif problem == "params":
        meta["parameters"]["classifier"]["C"] = 10
    else:
        meta["training"]["cycles"] = 8
    path.write_text(json.dumps(meta))
    with patch("hydroguard.inference.pickle.loads") as loads:
        with pytest.raises(DataValidationError):
            load_model_bundle(directory)
    loads.assert_not_called()
