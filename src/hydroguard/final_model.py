"""One authorized development fit, persisted inference bundle and reserved-test evaluation."""

import copy
import hashlib
import json
import os
import pickle
import tempfile
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from hydroguard.baselines import check_inputs_unchanged, metrics_with_stability
from hydroguard.data import (
    TARGET_LABELS,
    DataValidationError,
    atomic_text_writer,
    load_matrix,
    sha256_file,
    utc_now,
    validate_sensor,
    write_json_atomic,
)
from hydroguard.features import (
    FEATURE_CONTRACT,
    FEATURE_NAMES,
    PHYSICAL_SENSORS,
    extract_from_directory,
    validate_X,
)
from hydroguard.inference import (
    AUTHORIZED_PARAMETERS,
    SENSOR_CONTRACT,
    LoadedModel,
    library_versions,
    load_model_bundle,
    predict_cycle,
    predict_feature_rows,
    validate_pipeline,
)
from hydroguard.splitting import canonical_sha256, class_counts, load_split_lock, lock_or_verify
from hydroguard.tuning import (
    CONFIG_PATH as TUNING_CONFIG_PATH,
)
from hydroguard.tuning import (
    IMPLEMENTATION_PATHS as TUNING_IMPLEMENTATION_PATHS,
)
from hydroguard.tuning import (
    MANIFEST_PATH as TUNING_MANIFEST_PATH,
)
from hydroguard.tuning import (
    PROTECTED_PATHS as BASE_PROTECTED_PATHS,
)
from hydroguard.tuning import (
    load_experiment,
    make_candidate_model,
)

MANIFEST_PATH = Path("artifacts/experiments/phase3b-v1.json")
MODEL_DIRECTORY = Path("artifacts/models/phase3b-v1")
TRAINING_CLAIM = Path("artifacts/experiments/phase3b-training-attempt-v1.json")
TEST_CLAIM = Path("artifacts/experiments/phase3b-test-access-v1.json")
TEST_JSON = Path("docs/final-test-results.json")
TEST_CSV = Path("data/processed/final-test-predictions.csv")
IMPLEMENTATION_PATHS = (
    "src/hydroguard/inference.py",
    "src/hydroguard/final_model.py",
    "src/hydroguard/final_report.py",
    "scripts/train_final.py",
    "scripts/evaluate_final.py",
    "scripts/predict_cycle.py",
)
HISTORY_PATHS = (
    *BASE_PROTECTED_PATHS,
    *TUNING_IMPLEMENTATION_PATHS,
    TUNING_CONFIG_PATH.as_posix(),
    TUNING_MANIFEST_PATH.as_posix(),
    "docs/tuning-results.json",
    "docs/tuning-results.md",
)


def history_hashes(root: Path) -> dict:
    paths = list(HISTORY_PATHS)
    for name in ("data/processed/baseline-oof.csv", "data/processed/tuning-oof.csv"):
        if (root / name).exists():
            paths.append(name)
    return {name: sha256_file(root / name) for name in paths}


def authorized_candidate(configuration: dict, results: dict) -> dict:
    matches = [
        candidate
        for candidate in configuration["candidates"]
        if candidate["candidate_id"] == "lr_09"
    ]
    if len(matches) != 1:
        raise DataValidationError("Expected exactly the locked lr_09 candidate")
    candidate = matches[0]
    selected = results["selected"]["logistic_regression"]
    if (
        candidate["family"] != "logistic_regression"
        or candidate["parameters"] != AUTHORIZED_PARAMETERS
        or results["status"] != "completed"
        or selected["candidate_id"] != "lr_09"
        or selected["parameters"] != candidate["parameters"]
    ):
        raise DataValidationError(
            "Phase 3B authorizes only unchanged lr_09 with C=100 and no class weights"
        )
    return copy.deepcopy(candidate)


def selected_context(root: Path) -> dict:
    config, tuning_manifest, split, metadata, baseline, _ = load_experiment(root)
    results = json.loads((root / "docs/tuning-results.json").read_text(encoding="utf-8"))
    if results["manifest_payload_sha256"] != tuning_manifest["payload_sha256"]:
        raise DataValidationError("Tuning results do not match the locked tuning experiment")
    return {
        "config": config,
        "tuning_manifest": tuning_manifest,
        "split": split,
        "metadata": metadata,
        "baseline": baseline,
        "tuning_results": results,
        "candidate": authorized_candidate(config["payload"], results),
    }


def final_manifest_payload(root: Path, context: dict) -> dict:
    split = context["split"]
    outer = split["payload"]["memberships"]["outer"]
    return {
        "schema_version": 1,
        "phase": "3B",
        "candidate": context["candidate"],
        "selection_authority": (
            "User fixed Logistic Regression lr_09 before final-test model access"
        ),
        "feature_contract": FEATURE_CONTRACT,
        "sensor_contract": SENSOR_CONTRACT,
        "classes": [0, 1, 2],
        "class_mapping": {str(k): value for k, value in TARGET_LABELS.items()},
        "versions": library_versions(),
        "protected_history": history_hashes(root),
        "split_reference": {
            "path": "artifacts/splits/phase2b-v1.json",
            "file_sha256": sha256_file(root / "artifacts/splits/phase2b-v1.json"),
            "payload_sha256": split["payload_sha256"],
            "memberships_sha256": canonical_sha256(split["payload"]["memberships"]),
        },
        "tuning_reference": {
            "config_sha256": sha256_file(root / TUNING_CONFIG_PATH),
            "config_payload_sha256": context["config"]["payload_sha256"],
            "manifest_sha256": sha256_file(root / TUNING_MANIFEST_PATH),
            "manifest_payload_sha256": context["tuning_manifest"]["payload_sha256"],
            "results_sha256": sha256_file(root / "docs/tuning-results.json"),
            "results_markdown_sha256": sha256_file(root / "docs/tuning-results.md"),
        },
        "input_checksums": split["payload"]["input_checksums"],
        "training": {
            "cycle_ids": outer["development_cycle_ids"],
            "cycles": 1328,
            "class_counts": [846, 246, 236],
            "pipeline_fits": 1,
            "scaler_fit_scope": "development only",
        },
        "test": {
            "cycle_ids": outer["test_cycle_ids"],
            "cycles": 339,
            "class_counts": [133, 103, 103],
            "model_evaluation_after_training_only": True,
        },
        "outer_excluded": {
            "cycle_ids": outer["excluded_cycle_ids"],
            "cycles": 538,
            "fit_or_evaluation_allowed": False,
        },
        "decision_rule": "Unmodified Pipeline.predict; no threshold changes or calibration",
        "model_directory": MODEL_DIRECTORY.as_posix(),
        "thread_limit": 1,
        "prior_exposure": "All cycles including this test were inspected in Phase 2A EDA",
        "no_result_driven_model_or_protocol_changes": True,
        "implementation_sha256": {name: sha256_file(root / name) for name in IMPLEMENTATION_PATHS},
    }


def prepare_final_manifest(root: Path) -> dict:
    context = selected_context(root)
    return lock_or_verify(root / MANIFEST_PATH, final_manifest_payload(root, context))


def load_final_context(root: Path) -> tuple[dict, dict]:
    manifest = load_split_lock(root / MANIFEST_PATH)  # Must exist before any feature access.
    context = selected_context(root)
    if manifest["payload"] != final_manifest_payload(root, context):
        raise DataValidationError(
            "Phase 3B manifest/source/history changed; no automatic overwrite"
        )
    return manifest, context


def check_history_unchanged(root: Path, manifest: dict) -> None:
    if history_hashes(root) != manifest["payload"]["protected_history"]:
        raise DataValidationError(
            "Historical split/config/results/dependency/extractor files changed"
        )
    if load_split_lock(root / MANIFEST_PATH) != manifest:
        raise DataValidationError("Phase 3B manifest changed during execution")
    check_inputs_unchanged(root, manifest["payload"]["input_checksums"])


def validate_cohort(batch, metadata, payload: dict, role: str) -> None:
    if role not in ("training", "test"):
        raise DataValidationError("Outer-excluded cycles cannot be fitted or evaluated")
    expected = payload[role]
    if batch.cycle_ids.tolist() != expected["cycle_ids"]:
        raise DataValidationError(
            f"{role}: feature membership/order differs from the locked cohort"
        )
    validate_X(batch.X, expected_rows=expected["cycles"])
    if class_counts(metadata, batch.cycle_ids) != expected["class_counts"]:
        raise DataValidationError(f"{role}: unexpected class support")
    if set(batch.cycle_ids) & set(payload["outer_excluded"]["cycle_ids"]):
        raise DataValidationError("Outer-excluded cycles entered a feature batch")
    opposite = "test" if role == "training" else "training"
    if set(batch.cycle_ids) & set(payload[opposite]["cycle_ids"]):
        raise DataValidationError("Training and test feature memberships overlap")


def _array_sha256(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def save_pipeline_bundle(
    pipeline, directory: Path, metadata: dict, reference_X: pd.DataFrame
) -> dict:
    """Publish a fresh complete bundle only after exact development round-trip checks."""
    if directory.exists():
        raise DataValidationError("Model bundle already exists; automatic replacement is forbidden")
    directory.parent.mkdir(parents=True, exist_ok=True)
    validate_pipeline(pipeline, metadata)
    before_class, before_probability = predict_feature_rows(
        LoadedModel(pipeline, metadata), reference_X
    )
    with tempfile.TemporaryDirectory(prefix=".phase3b-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        with (staging / "pipeline.pkl").open("xb") as stream:
            pickle.dump(pipeline, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = copy.deepcopy(metadata)
        metadata["model_sha256"] = sha256_file(staging / "pipeline.pkl")
        write_json_atomic(staging / "metadata.json", metadata)
        loaded = load_model_bundle(staging)
        after_class, after_probability = predict_feature_rows(loaded, reference_X)
        if not np.array_equal(before_class, after_class) or not np.array_equal(
            before_probability, after_probability
        ):
            raise DataValidationError(
                "Saved Pipeline predictions or probabilities changed after loading"
            )
        metadata["serialization_check"] = {
            "scope": "all development rows only",
            "rows": len(reference_X),
            "predict_equal": True,
            "predict_proba_equal": True,
            "max_probability_difference": float(
                np.max(np.abs(before_probability - after_probability))
            ),
            "predictions_sha256_before": _array_sha256(before_class),
            "predictions_sha256_after": _array_sha256(after_class),
            "probabilities_sha256_before": _array_sha256(before_probability),
            "probabilities_sha256_after": _array_sha256(after_probability),
        }
        metadata["published_at_utc"] = utc_now()
        write_json_atomic(staging / "metadata.json", metadata)
        staging.rename(directory)
    return metadata


def validate_published_bundle(root: Path, manifest: dict) -> LoadedModel:
    bundle = load_model_bundle(root / MODEL_DIRECTORY)
    metadata, payload = bundle.metadata, manifest["payload"]
    attempt = load_split_lock(root / TRAINING_CLAIM)
    if (
        metadata["parameters"] != payload["candidate"]["parameters"]
        or metadata["training"]["cycle_ids"] != payload["training"]["cycle_ids"]
        or metadata["training"]["cycles"] != 1328
        or metadata["training"]["pipeline_fits"] != 1
        or metadata["phase3b_reference"]["payload_sha256"] != manifest["payload_sha256"]
        or metadata["phase3b_reference"]["file_sha256"] != sha256_file(root / MANIFEST_PATH)
        or metadata["training"]["attempt_payload_sha256"] != attempt["payload_sha256"]
        or attempt["payload"]["training_cycle_ids"] != payload["training"]["cycle_ids"]
        or attempt["payload"]["phase3b_payload_sha256"] != manifest["payload_sha256"]
        or not metadata["serialization_check"]["predict_equal"]
        or not metadata["serialization_check"]["predict_proba_equal"]
    ):
        raise DataValidationError("Saved Pipeline does not match the authorized Phase 3B training")
    return bundle


def train_final(root: Path) -> dict:
    manifest, context = load_final_context(root)
    if (root / MODEL_DIRECTORY).exists():
        metadata = validate_published_bundle(root, manifest).metadata
        check_history_unchanged(root, manifest)
        return {"status": "existing_model_verified", "fits_this_call": 0, "metadata": metadata}
    if (root / TRAINING_CLAIM).exists():
        raise DataValidationError(
            "A prior training attempt exists without a complete model; investigate before refitting"
        )
    payload = manifest["payload"]
    batch = extract_from_directory(root / "data/raw/uci447", payload["training"]["cycle_ids"])
    validate_cohort(batch, context["metadata"], payload, "training")
    check_history_unchanged(root, manifest)
    attempt = lock_or_verify(
        root / TRAINING_CLAIM,
        {
            "attempt_id": uuid.uuid4().hex,
            "phase3b_payload_sha256": manifest["payload_sha256"],
            "candidate_id": "lr_09",
            "training_cycle_ids": batch.cycle_ids.tolist(),
            "authorized_pipeline_fits": 1,
        },
    )
    pipeline = make_candidate_model(payload["candidate"], context["baseline"])
    y = context["metadata"].loc[batch.cycle_ids, "pump_leakage"].to_numpy()
    fit_started_utc, started = utc_now(), time.perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(batch.X, y)  # The sole actual-dataset Pipeline fit in Phase 3B.
    fit_seconds = time.perf_counter() - started
    scaler = pipeline.named_steps["scaler"]
    if (
        scaler.n_samples_seen_ != 1328
        or not np.allclose(scaler.mean_, batch.X.mean().to_numpy())
        or not np.allclose(scaler.var_, batch.X.var(ddof=0).to_numpy())
    ):
        raise DataValidationError("StandardScaler was not fitted on all and only development rows")
    model_metadata = {
        "schema_version": 1,
        "phase": "3B",
        "candidate_id": "lr_09",
        "parameters": copy.deepcopy(payload["candidate"]["parameters"]),
        "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(FEATURE_NAMES),
        "sensor_contract": SENSOR_CONTRACT,
        "classes_": pipeline.classes_.tolist(),
        "class_mapping": payload["class_mapping"],
        "versions": library_versions(),
        "serialization": {"format": "python-pickle", "protocol": 5},
        "phase3b_reference": {
            "path": MANIFEST_PATH.as_posix(),
            "payload_sha256": manifest["payload_sha256"],
            "file_sha256": sha256_file(root / MANIFEST_PATH),
        },
        "input_checksums": payload["input_checksums"],
        "split_reference": payload["split_reference"],
        "tuning_reference": payload["tuning_reference"],
        "training": {
            **copy.deepcopy(payload["training"]),
            "started_at_utc": fit_started_utc,
            "completed_at_utc": utc_now(),
            "fit_seconds": fit_seconds,
            "scaler_training_samples": int(scaler.n_samples_seen_),
            "scaler_training_only_verified": True,
            "classifier_iterations": pipeline.named_steps["classifier"].n_iter_.tolist(),
            "attempt_payload_sha256": attempt["payload_sha256"],
            "warnings": [
                {"category": w.category.__name__, "message": str(w.message)} for w in captured
            ],
        },
    }
    metadata = save_pipeline_bundle(pipeline, root / MODEL_DIRECTORY, model_metadata, batch.X)
    validate_published_bundle(root, manifest)
    check_history_unchanged(root, manifest)
    return {"status": "trained_and_published", "fits_this_call": 1, "metadata": metadata}


def _cached_test_result(root: Path, manifest: dict, bundle: LoadedModel, metadata) -> dict:
    result = json.loads((root / TEST_JSON).read_text(encoding="utf-8"))
    access = load_split_lock(root / TEST_CLAIM)
    if (
        result["phase3b_payload_sha256"] != manifest["payload_sha256"]
        or result["model_sha256"] != bundle.metadata["model_sha256"]
        or result["predictions_sha256"] != sha256_file(root / TEST_CSV)
        or access["payload_sha256"] != result["test_access_payload_sha256"]
    ):
        raise DataValidationError(
            "Cached test results do not match the fixed model/manifest/predictions"
        )
    rows = pd.read_csv(root / TEST_CSV)
    if rows.cycle_id.tolist() != manifest["payload"]["test"]["cycle_ids"]:
        raise DataValidationError("Cached test prediction membership changed")
    if (
        rows.pump_leakage.tolist() != metadata.loc[rows.cycle_id, "pump_leakage"].tolist()
        or rows.stability_flag.tolist() != metadata.loc[rows.cycle_id, "stability_flag"].tolist()
    ):
        raise DataValidationError("Cached test labels/stability differ from the original profile")
    expected = metrics_with_stability(
        rows.pump_leakage.to_numpy(),
        rows.predicted_class.to_numpy(),
        rows.stability_flag.to_numpy(),
    )
    if expected != result["test_results"]:
        raise DataValidationError("Cached test metrics do not match the saved predictions")
    return result


def evaluate_final(root: Path) -> tuple[dict, bool]:
    manifest, context = load_final_context(root)
    bundle = validate_published_bundle(root, manifest)  # Before any test feature extraction.
    if (root / TEST_JSON).exists():
        result = _cached_test_result(root, manifest, bundle, context["metadata"])
        check_history_unchanged(root, manifest)
        return result, True
    if (root / TEST_CLAIM).exists():
        raise DataValidationError(
            "Prior test-access receipt exists without complete results; no automatic repeat access"
        )
    access = lock_or_verify(
        root / TEST_CLAIM,
        {
            "access_id": uuid.uuid4().hex,
            "phase3b_payload_sha256": manifest["payload_sha256"],
            "model_sha256": bundle.metadata["model_sha256"],
            "candidate_id": "lr_09",
            "test_cycle_ids": manifest["payload"]["test"]["cycle_ids"],
            "authorized_model": "Only the completed development-trained lr_09 Pipeline",
        },
    )
    started_utc, started = utc_now(), time.perf_counter()
    batch = extract_from_directory(
        root / "data/raw/uci447", manifest["payload"]["test"]["cycle_ids"]
    )
    validate_cohort(batch, context["metadata"], manifest["payload"], "test")
    check_history_unchanged(root, manifest)
    predicted, probabilities = predict_feature_rows(bundle, batch.X)
    actual = context["metadata"].loc[batch.cycle_ids, "pump_leakage"].to_numpy()
    stability = context["metadata"].loc[batch.cycle_ids, "stability_flag"].to_numpy()
    rows = pd.DataFrame(
        {
            "cycle_id": batch.cycle_ids,
            "pump_leakage": actual,
            "stability_flag": stability,
            "predicted_class": predicted,
            "predicted_label": [TARGET_LABELS[int(label)] for label in predicted],
        }
    )
    for index, label in enumerate(bundle.metadata["classes_"]):
        rows[f"probability_class_{label}"] = probabilities[:, index]
    with atomic_text_writer(root / TEST_CSV) as stream:
        rows.to_csv(stream, index=False)
    result = {
        "schema_version": 1,
        "phase": "3B",
        "status": "completed",
        "started_at_utc": started_utc,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "phase3b_payload_sha256": manifest["payload_sha256"],
        "phase3b_created_at_utc": manifest["created_at_utc"],
        "test_access_created_at_utc": access["created_at_utc"],
        "test_access_payload_sha256": access["payload_sha256"],
        "candidate_id": "lr_09",
        "parameters": bundle.metadata["parameters"],
        "model_directory": MODEL_DIRECTORY.as_posix(),
        "model_sha256": bundle.metadata["model_sha256"],
        "model_metadata_sha256": sha256_file(root / MODEL_DIRECTORY / "metadata.json"),
        "model_training": bundle.metadata["training"],
        "serialization_check": bundle.metadata["serialization_check"],
        "versions": bundle.metadata["versions"],
        "classes_": bundle.metadata["classes_"],
        "feature_contract": FEATURE_CONTRACT,
        "sensor_contract": SENSOR_CONTRACT,
        "test_results": metrics_with_stability(actual, predicted, stability),
        "test_cycle_ids": batch.cycle_ids.tolist(),
        "predictions_path": TEST_CSV.as_posix(),
        "predictions_sha256": sha256_file(root / TEST_CSV),
        "development_selection": copy.deepcopy(
            context["tuning_results"]["selected"]["logistic_regression"]
        ),
        "development_results_role": (
            "Previously used for model selection; not the final-test estimate"
        ),
        "outer_excluded": {
            "cycles": 538,
            "features_extracted": False,
            "fitted": False,
            "evaluated": False,
        },
        "historical_files_unchanged": True,
        "raw_data_unchanged": True,
        "prior_test_eda_exposure": True,
        "model_or_protocol_changed_after_test": False,
    }
    check_history_unchanged(root, manifest)
    if sha256_file(root / MODEL_DIRECTORY / "pipeline.pkl") != bundle.metadata["model_sha256"]:
        raise DataValidationError("Model changed during evaluation")
    write_json_atomic(root / TEST_JSON, result)
    return result, False


def predict_development_example(root: Path, cycle_id: int | None = None) -> dict:
    bundle = load_model_bundle(root / MODEL_DIRECTORY)
    training_ids = bundle.metadata["training"]["cycle_ids"]
    selected_id = training_ids[0] if cycle_id is None else cycle_id
    if selected_id not in training_ids:
        raise DataValidationError("The inference example must use a development training cycle")
    check_inputs_unchanged(root, bundle.metadata["input_checksums"])
    raw_cycle = {}
    for sensor in PHYSICAL_SENSORS:
        matrix = load_matrix(root / "data/raw/uci447" / f"{sensor}.txt")
        validate_sensor(matrix, sensor)
        raw_cycle[sensor] = matrix[selected_id - 1].copy()
        del matrix
    return {
        "example_cycle_id": selected_id,
        "source_partition": "development",
        **predict_cycle(raw_cycle, bundle),
    }
