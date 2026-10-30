"""Validated single-cycle inference using the saved complete scikit-learn Pipeline."""

import hashlib
import importlib.metadata
import json
import pickle
import platform
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from hydroguard.data import CYCLE_SECONDS, SENSOR_RATES, TARGET_LABELS, DataValidationError
from hydroguard.features import (
    FEATURE_CONTRACT,
    FEATURE_NAMES,
    PHYSICAL_SENSORS,
    extract_features,
    validate_X,
)

SENSOR_CONTRACT = {
    "cycle_seconds": CYCLE_SECONDS,
    "sensor_order": list(PHYSICAL_SENSORS),
    "sensors": {
        sensor: {
            "sampling_rate_hz": SENSOR_RATES[sensor],
            "samples": CYCLE_SECONDS * SENSOR_RATES[sensor],
            "ndim": 1,
        }
        for sensor in PHYSICAL_SENSORS
    },
    "values": "Finite real numeric samples; no strings, booleans, complex values or metadata",
}
AUTHORIZED_PARAMETERS = {
    "scaler": {"with_mean": True, "with_std": True},
    "classifier": {
        "C": 100,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "tol": 0.0001,
        "fit_intercept": True,
        "class_weight": None,
        "random_state": 2026,
    },
}


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
    }


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    metadata: dict


def cycle_features(raw_cycle: Mapping) -> pd.DataFrame:
    """Require exactly one 1D vector per physical sensor, at its documented length."""
    if not isinstance(raw_cycle, Mapping):
        raise DataValidationError("A raw cycle must be a sensor-to-vector mapping")
    if set(raw_cycle) != set(PHYSICAL_SENSORS):
        missing = sorted(set(PHYSICAL_SENSORS) - set(raw_cycle))
        extra = sorted(map(str, set(raw_cycle) - set(PHYSICAL_SENSORS)))
        raise DataValidationError(
            f"Expected exactly 14 physical sensors; missing={missing}, extra={extra}"
        )
    signals = {}
    for sensor in PHYSICAL_SENSORS:
        try:
            values = np.asarray(raw_cycle[sensor])
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"{sensor}: cannot read a numeric sample vector") from error
        expected = SENSOR_CONTRACT["sensors"][sensor]["samples"]
        if values.ndim != 1 or values.shape != (expected,):
            raise DataValidationError(
                f"{sensor}: expected one 1D vector of {expected} samples; got {values.shape}"
            )
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise DataValidationError(f"{sensor}: samples must be finite real numeric values")
        try:
            with np.errstate(over="raise", invalid="raise"):
                signals[sensor] = values.astype(np.float64)[None, :]
        except FloatingPointError as error:
            raise DataValidationError(
                f"{sensor}: samples cannot be represented as float64"
            ) from error
    # The extractor's one-row token is separate from X and is not a source cycle ID.
    frame = extract_features(signals, [1]).X
    validate_X(frame, expected_rows=1)
    return frame


def validate_pipeline(pipeline, metadata: dict) -> None:
    if (
        not isinstance(pipeline, Pipeline)
        or list(pipeline.named_steps) != ["scaler", "classifier"]
        or not isinstance(pipeline.named_steps["scaler"], StandardScaler)
        or not isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    ):
        raise DataValidationError(
            "Expected the complete StandardScaler/LogisticRegression Pipeline"
        )
    scaler, classifier = pipeline.named_steps["scaler"], pipeline.named_steps["classifier"]
    try:
        if (
            list(pipeline.feature_names_in_) != list(FEATURE_NAMES)
            or pipeline.n_features_in_ != 70
            or classifier.n_features_in_ != 70
            or pipeline.classes_.tolist() != metadata["classes_"]
            or pipeline.classes_.tolist() != [0, 1, 2]
            or scaler.n_samples_seen_ != metadata["training"]["cycles"]
        ):
            raise DataValidationError(
                "Fitted feature names/classes/training sample count differ from metadata"
            )
        for step, instance in (("scaler", scaler), ("classifier", classifier)):
            actual = instance.get_params(deep=False)
            if any(actual.get(key) != value for key, value in metadata["parameters"][step].items()):
                raise DataValidationError(
                    f"Fitted {step} parameters differ from the selected candidate"
                )
        arrays = [scaler.mean_, scaler.var_, scaler.scale_, classifier.coef_, classifier.intercept_]
        if not all(np.isfinite(values).all() for values in arrays):
            raise DataValidationError("Fitted Pipeline contains non-finite values")
    except (AttributeError, KeyError, TypeError) as error:
        raise DataValidationError(f"Pipeline is unfitted or inconsistent: {error}") from error


def validate_model_metadata(metadata: dict) -> None:
    try:
        if (
            not isinstance(metadata, dict)
            or metadata["schema_version"] != 1
            or metadata["phase"] != "3B"
            or metadata["candidate_id"] != "lr_09"
            or metadata["feature_contract"] != FEATURE_CONTRACT
            or metadata["feature_names"] != list(FEATURE_NAMES)
            or metadata["sensor_contract"] != SENSOR_CONTRACT
            or metadata["classes_"] != [0, 1, 2]
            or metadata["class_mapping"] != {str(k): value for k, value in TARGET_LABELS.items()}
            or metadata["versions"] != library_versions()
            or metadata["serialization"] != {"format": "python-pickle", "protocol": 5}
            or not re.fullmatch(r"[0-9a-f]{64}", metadata["model_sha256"])
        ):
            raise DataValidationError(
                "Model metadata contract/classes/versions do not match this runtime"
            )
        ids = np.asarray(metadata["training"]["cycle_ids"])
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or not len(ids)
            or np.any(ids < 1)
            or np.any(ids[1:] <= ids[:-1])
            or type(metadata["training"]["cycles"]) is not int
            or metadata["training"]["cycles"] != len(ids)
        ):
            raise DataValidationError("Invalid model training membership metadata")
        params = metadata["parameters"]
        if params != AUTHORIZED_PARAMETERS:
            raise DataValidationError("Expected the authorized lr_09 parameters")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, DataValidationError):
            raise
        raise DataValidationError(f"Invalid model metadata: {error}") from error


def load_model_bundle(directory: Path) -> LoadedModel:
    """Load only a trusted locally generated bundle; a checksum is not authentication."""
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        validate_model_metadata(metadata)
        model_bytes = (directory / "pipeline.pkl").read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != metadata["model_sha256"]:
            raise DataValidationError("Saved model checksum mismatch; deserialization refused")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = pickle.loads(model_bytes)  # Trusted local artifacts only.
        validate_pipeline(pipeline, metadata)
        return LoadedModel(pipeline, metadata)
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        AttributeError,
        ImportError,
        pickle.UnpicklingError,
        InconsistentVersionWarning,
    ) as error:
        if isinstance(error, DataValidationError):
            raise
        raise DataValidationError(f"Cannot load model bundle: {error}") from error


def predict_feature_rows(bundle: LoadedModel, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    validate_X(frame)
    validate_pipeline(bundle.pipeline, bundle.metadata)
    with threadpool_limits(limits=1):
        predicted = bundle.pipeline.predict(frame)
        probabilities = bundle.pipeline.predict_proba(frame)
    classes = np.asarray(bundle.metadata["classes_"])
    if (
        probabilities.shape != (len(frame), len(classes))
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        or not np.array_equal(predicted, classes[np.argmax(probabilities, axis=1)])
    ):
        raise DataValidationError("Invalid predictions/probabilities or inconsistent class order")
    return predicted, probabilities


def predict_cycle(raw_cycle: Mapping, bundle: LoadedModel) -> dict:
    """Predict one raw cycle without reading any dataset, label or split metadata."""
    predicted, probabilities = predict_feature_rows(bundle, cycle_features(raw_cycle))
    label = int(predicted[0])
    return {
        "class": label,
        "label": bundle.metadata["class_mapping"][str(label)],
        "classes": list(bundle.metadata["classes_"]),
        "probabilities": probabilities[0].tolist(),
    }
