"""Analytical fixtures for the exact, metadata-free physical feature contract."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hydroguard.data import SENSOR_RATES, DataValidationError
from hydroguard.features import (
    FEATURE_NAMES,
    PHYSICAL_SENSORS,
    extract_features,
    extract_from_directory,
    rows_for_cycles,
    sensor_statistics,
    validate_cycle_ids,
    validate_X,
)


@pytest.fixture
def signals():
    return {sensor: np.array([[-2.0, 0.0, 2.0], [1.0, 2.0, 3.0]]) for sensor in PHYSICAL_SENSORS}


def test_analytical_statistics_and_population_std():
    result = sensor_statistics(np.array([[-2, 0, 2], [1, 2, 3]], dtype=float))
    np.testing.assert_allclose(
        result,
        [[0, np.sqrt(8 / 3), np.sqrt(8 / 3), -2, 2], [2, np.sqrt(2 / 3), np.sqrt(14 / 3), 1, 3]],
    )
    assert sensor_statistics(np.array([[7.0]]))[0].tolist() == [7.0, 0.0, 7.0, 7.0, 7.0]


def test_feature_order_and_repeatability_are_independent_of_mapping_order(signals):
    ids = np.array([1, 7])
    batch = extract_features(signals, ids)
    reversed_batch = extract_features(dict(reversed(list(signals.items()))), ids)
    pd.testing.assert_frame_equal(batch.X, reversed_batch.X)
    assert batch.X.shape == (2, 70)
    assert tuple(batch.X.columns) == FEATURE_NAMES
    assert FEATURE_NAMES[:5] == ("PS1__mean", "PS1__std", "PS1__rms", "PS1__min", "PS1__max")
    assert FEATURE_NAMES[-1] == "VS1__max"
    assert batch.X.index.name is None
    assert not any(name.startswith(("CE__", "CP__", "SE__")) for name in FEATURE_NAMES)
    ids[0] = 99
    assert batch.cycle_ids.tolist() == [1, 7]


def test_other_cycles_do_not_fit_or_change_existing_statistics(signals):
    original = extract_features(signals, [1, 7])
    extended = {sensor: np.vstack((values, [1e6, -1e6, 0])) for sensor, values in signals.items()}
    batch = extract_features(extended, [1, 7, 100])
    pd.testing.assert_frame_equal(original.X, batch.X.iloc[:2])


@pytest.mark.parametrize(
    "extra", ["CE", "CP", "SE", "cycle_id", "pump_leakage", "stability_flag", "group_id"]
)
def test_extractor_rejects_virtual_signals_and_metadata(signals, extra):
    signals[extra] = np.ones((2, 3))
    with pytest.raises(DataValidationError, match="exactly the 14"):
        extract_features(signals, [1, 7])


@pytest.mark.parametrize(
    "column",
    [
        "cycle_id",
        "pump_leakage",
        "stability_flag",
        "group_id",
        "cooler_condition_pct",
        "valve_condition_pct",
        "accumulator_pressure_bar",
        "CE__mean",
    ],
)
def test_X_allowlist_rejects_all_metadata_and_virtual_columns(signals, column):
    frame = extract_features(signals, [1, 7]).X.assign(**{column: 0.0})
    with pytest.raises(DataValidationError, match="fixed 70"):
        validate_X(frame)


def test_X_rejects_wrong_order_named_index_nonfinite_and_wrong_dtype(signals):
    frame = extract_features(signals, [1, 7]).X
    for bad in [
        frame.iloc[:, ::-1],
        frame.rename_axis("cycle_id"),
        frame.astype(int),
        frame.assign(PS1__mean=np.inf),
        frame.iloc[:0],
    ]:
        with pytest.raises(DataValidationError):
            validate_X(bad)


@pytest.mark.parametrize(
    "values", [np.ones(3), np.empty((0, 3)), np.array([[np.nan]]), np.array([[np.inf]]), [["bad"]]]
)
def test_invalid_sensor_statistics_fail(values):
    with pytest.raises(DataValidationError):
        sensor_statistics(values)


@pytest.mark.parametrize(
    "ids",
    [
        [],
        [0, 1],
        [1, 1],
        [2, 1],
        [1.0, 2.0],
        [True, False],
        [-1, 2],
        np.array([2, 1], dtype=np.uint64),
    ],
)
def test_invalid_original_cycle_ids(ids):
    with pytest.raises(DataValidationError):
        validate_cycle_ids(ids)


def test_directory_extraction_only_summarizes_requested_physical_cycles(tmp_path):
    calls = []

    def loader(path):
        sensor = path.stem
        calls.append(sensor)
        width = SENSOR_RATES[sensor] * 60
        return np.repeat(np.array([[1.0], [10000.0], [3.0]]), width, axis=1)

    with patch("hydroguard.features.load_matrix", side_effect=loader):
        batch = extract_from_directory(tmp_path, [1, 3], expected_cycles=3)
    assert calls == list(PHYSICAL_SENSORS)
    assert batch.cycle_ids.tolist() == [1, 3]
    np.testing.assert_allclose(batch.X.PS1__mean, [1, 3])
    np.testing.assert_allclose(batch.X.PS1__std, [0, 0])
    with pytest.raises(DataValidationError, match="development-only"):
        rows_for_cycles(batch, [2])


def test_missing_sensor_and_row_misalignment(signals):
    signals["PS1"] = np.ones((1, 3))
    with pytest.raises(DataValidationError, match="rows do not align"):
        extract_features(signals, [1, 7])
    del signals["PS1"]
    with pytest.raises(DataValidationError, match="exactly the 14"):
        extract_features(signals, [1, 7])
