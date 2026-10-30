"""Deterministic physical-sensor statistics with identifiers kept outside X."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hydroguard.data import (
    EXPECTED_CYCLES,
    SENSOR_RATES,
    DataValidationError,
    load_matrix,
    validate_sensor,
)

PHYSICAL_SENSORS = (
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
)
STATISTICS = ("mean", "std", "rms", "min", "max")
FEATURE_NAMES = tuple(f"{sensor}__{stat}" for sensor in PHYSICAL_SENSORS for stat in STATISTICS)
FEATURE_CONTRACT = {
    "name": "physical14-statistics5-v1",
    "sensors": list(PHYSICAL_SENSORS),
    "statistics": list(STATISTICS),
    "std_ddof": 0,
    "dtype": "float64",
    "names": list(FEATURE_NAMES),
    "excluded_signals": ["CE", "CP", "SE"],
    "fitted_values": False,
}


@dataclass(frozen=True)
class FeatureBatch:
    """Positional feature rows and a separate original-cycle identifier array."""

    cycle_ids: np.ndarray
    X: pd.DataFrame


def validate_cycle_ids(values, *, maximum: int | None = None) -> np.ndarray:
    ids = np.asarray(values)
    if (
        ids.ndim != 1
        or ids.size == 0
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 1)
        or np.any(ids[1:] <= ids[:-1])
        or np.any(ids > np.iinfo(np.int64).max)
        or (maximum is not None and np.any(ids > maximum))
    ):
        raise DataValidationError(
            "cycle_ids must be positive, strictly increasing original integers"
        )
    return ids.astype(np.int64, copy=True)


def sensor_statistics(matrix: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("Sensor statistics require numeric values") from error
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise DataValidationError("Sensor statistics require a nonempty finite 2D numeric matrix")
    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.column_stack(
                (
                    values.mean(axis=1),
                    values.std(axis=1, ddof=0),
                    np.sqrt(np.mean(values**2, axis=1)),
                    values.min(axis=1),
                    values.max(axis=1),
                )
            )
    except FloatingPointError as error:
        raise DataValidationError(f"Invalid sensor statistic: {error}") from error


def validate_X(frame: pd.DataFrame, *, expected_rows: int | None = None) -> None:
    """An exact allowlist rejects every metadata/virtual-sensor/extra column."""
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != FEATURE_NAMES:
        raise DataValidationError(
            "X must contain exactly the fixed 70 physical-sensor feature columns"
        )
    if frame.index.name is not None:
        raise DataValidationError("X must not carry a named metadata index; cycle_ids are separate")
    if (
        len(frame) == 0
        or (expected_rows is not None and len(frame) != expected_rows)
        or not all(dtype == np.dtype("float64") for dtype in frame.dtypes)
        or not np.isfinite(frame.to_numpy()).all()
    ):
        raise DataValidationError("X must contain the expected number of finite float64 rows")


def extract_features(signals: Mapping[str, np.ndarray], cycle_ids) -> FeatureBatch:
    """Extract from aligned physical-sensor matrices, without any fitted state."""
    if set(signals) != set(PHYSICAL_SENSORS):
        raise DataValidationError(
            "Provide exactly the 14 physical sensors, without metadata or virtual signals"
        )
    ids = validate_cycle_ids(cycle_ids)
    pieces = []
    for sensor in PHYSICAL_SENSORS:
        piece = sensor_statistics(signals[sensor])
        if len(piece) != len(ids):
            raise DataValidationError(f"{sensor}: rows do not align with cycle_ids")
        pieces.append(piece)
    frame = pd.DataFrame(np.column_stack(pieces), columns=FEATURE_NAMES, dtype=np.float64)
    validate_X(frame, expected_rows=len(ids))
    return FeatureBatch(ids, frame)


def extract_from_directory(
    dataset_dir: Path, cycle_ids, *, expected_cycles=EXPECTED_CYCLES
) -> FeatureBatch:
    """Read one matrix at a time; compute statistics only for the requested cycles."""
    ids = validate_cycle_ids(cycle_ids, maximum=expected_cycles)
    pieces = []
    for sensor in PHYSICAL_SENSORS:
        matrix = load_matrix(dataset_dir / f"{sensor}.txt")
        validate_sensor(matrix, sensor, expected_cycles)
        pieces.append(sensor_statistics(matrix[ids - 1]))
        del matrix
    frame = pd.DataFrame(np.column_stack(pieces), columns=FEATURE_NAMES, dtype=np.float64)
    validate_X(frame, expected_rows=len(ids))
    assert len(PHYSICAL_SENSORS) == 14 and len(FEATURE_NAMES) == 70
    assert set(PHYSICAL_SENSORS) <= set(SENSOR_RATES)
    return FeatureBatch(ids, frame)


def rows_for_cycles(batch: FeatureBatch, cycle_ids) -> pd.DataFrame:
    validate_cycle_ids(batch.cycle_ids)
    ids = validate_cycle_ids(cycle_ids)
    positions = pd.Index(batch.cycle_ids).get_indexer(ids)
    if np.any(positions < 0):
        raise DataValidationError(
            "Requested cycles are absent from the development-only feature batch"
        )
    frame = batch.X.iloc[positions].reset_index(drop=True)
    validate_X(frame, expected_rows=len(ids))
    return frame
