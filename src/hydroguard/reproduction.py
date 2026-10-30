"""Fresh-acquisition development reproduction, separate from immutable historical runs."""

import json
import time
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from hydroguard.baselines import check_inputs_unchanged, verified_context
from hydroguard.data import (
    TARGET_LABELS,
    DataValidationError,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from hydroguard.features import FEATURE_CONTRACT, FEATURE_NAMES, extract_from_directory
from hydroguard.final_model import authorized_candidate, save_pipeline_bundle, validate_cohort
from hydroguard.inference import SENSOR_CONTRACT, library_versions
from hydroguard.splitting import load_split_lock, lock_or_verify
from hydroguard.tuning import make_candidate_model, validate_frozen_memberships


def verify_review_history(root: Path) -> dict:
    record = json.loads((root / "config/review-integrity-v1.json").read_text(encoding="utf-8"))
    changed = [
        name for name, digest in record["files"].items() if sha256_file(root / name) != digest
    ]
    if changed:
        raise DataValidationError(f"Historical review inputs changed: {changed}")
    return record


def verify_content_identity(historical: dict, current: dict) -> None:
    """Only acquisition-record identity may differ; data and environment remain pinned."""
    ignored = {"download_manifest_sha256"}
    old = {k: v for k, v in historical.items() if k not in ignored}
    new = {k: v for k, v in current.items() if k not in ignored}
    if old != new:
        raise DataValidationError(
            "Fresh acquisition content/environment differs from the historical pin"
        )


def reproduction_context(root: Path) -> dict:
    verify_review_history(root)
    split = load_split_lock(root / "artifacts/splits/phase2b-v1.json")
    configuration = load_split_lock(root / "config/tuning-v1.json")
    results = json.loads((root / "docs/tuning-results.json").read_text(encoding="utf-8"))
    official = json.loads(
        (root / "artifacts/models/phase3b-v1/metadata.json").read_text(encoding="utf-8")
    )
    if library_versions() != official["versions"]:
        raise DataValidationError(
            "Reproduction requires the official model's exact numerical versions"
        )
    # This unchanged loader verifies the pinned ZIP, every extracted file, provenance,
    # shapes, finite values and original profile. It does not fit or predict anything.
    metadata, fingerprints, baseline = verified_context(root)
    verify_content_identity(split["payload"]["input_checksums"], fingerprints)
    validate_frozen_memberships(metadata, split["payload"])  # Saved IDs; no new splitter.
    candidate = authorized_candidate(configuration["payload"], results)
    return {
        "split": split,
        "candidate": candidate,
        "metadata": metadata,
        "fingerprints": fingerprints,
        "baseline": baseline,
        "official": official,
    }


def reproduce_training(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    allowed = root / "data/processed/reproductions"
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise DataValidationError(
            "Use a new directory below data/processed/reproductions; replacement is forbidden"
        )
    context = reproduction_context(root)
    split, official = context["split"], context["official"]
    development = split["payload"]["memberships"]["outer"]["development_cycle_ids"]
    if development != official["training"]["cycle_ids"] or len(development) != 1328:
        raise DataValidationError(
            "Reproduction must use exactly the official development membership"
        )
    fresh = context["fingerprints"]
    output.mkdir(parents=True, exist_ok=False)  # Reserve this run before the first fit.
    provenance = (root / "data/raw/download.json").read_bytes()
    (output / "download.json").write_bytes(provenance)
    if sha256_file(output / "download.json") != fresh["download_manifest_sha256"]:
        raise DataValidationError("Acquisition record changed before preregistration")
    manifest = lock_or_verify(
        output / "manifest.json",
        {
            "phase": "5",
            "role": "Independent development-only training reproduction",
            "candidate": context["candidate"],
            "feature_contract": FEATURE_CONTRACT,
            "sensor_contract": SENSOR_CONTRACT,
            "versions": library_versions(),
            "training_cycle_ids": development,
            "training_cycles": 1328,
            "pipeline_fits": 1,
            "test_predictions_or_metrics_allowed": False,
            "outer_excluded_used": False,
            "input_checksums": fresh,
            "historical_download_manifest_sha256": split["payload"]["input_checksums"][
                "download_manifest_sha256"
            ],
            "provenance_policy": "Only provenance may differ; archive and raw content stay pinned",
            "split_file_sha256": sha256_file(root / "artifacts/splits/phase2b-v1.json"),
            "split_payload_sha256": split["payload_sha256"],
            "official_model_sha256": official["model_sha256"],
            "review_history_sha256": sha256_file(root / "config/review-integrity-v1.json"),
            "implementation_sha256": sha256_file(Path(__file__)),
        },
    )
    started_utc, started = utc_now(), time.perf_counter()
    batch = extract_from_directory(root / "data/raw/uci447", development)
    # Reuse the official cohort guard without calling its acquisition-bound entry points.
    payload = {
        "training": official["training"],
        "test": {"cycle_ids": split["payload"]["memberships"]["outer"]["test_cycle_ids"]},
        "outer_excluded": {
            "cycle_ids": split["payload"]["memberships"]["outer"]["excluded_cycle_ids"]
        },
    }
    validate_cohort(batch, context["metadata"], payload, "training")
    check_inputs_unchanged(root, fresh)
    pipeline = make_candidate_model(context["candidate"], context["baseline"])
    target = context["metadata"].loc[development, "pump_leakage"].to_numpy()
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(batch.X, target)
    fit_seconds = time.perf_counter() - fit_started
    scaler = pipeline.named_steps["scaler"]
    if scaler.n_samples_seen_ != 1328:
        raise DataValidationError("Scaler did not observe exactly the development cohort")
    metadata = {
        # Historical portable bundle schema; the separate role/manifest identify this new run.
        "schema_version": 1,
        "phase": "3B",
        "reproduction_phase": "5",
        "artifact_role": "Non-official development reproduction; not for test evaluation",
        "candidate_id": "lr_09",
        "parameters": context["candidate"]["parameters"],
        "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(FEATURE_NAMES),
        "sensor_contract": SENSOR_CONTRACT,
        "classes_": pipeline.classes_.tolist(),
        "class_mapping": {str(k): v for k, v in TARGET_LABELS.items()},
        "versions": library_versions(),
        "serialization": {"format": "python-pickle", "protocol": 5},
        "reproduction_manifest_payload_sha256": manifest["payload_sha256"],
        "training": {
            "cycle_ids": development,
            "cycles": 1328,
            "class_counts": [846, 246, 236],
            "pipeline_fits": 1,
            "scaler_training_samples": int(scaler.n_samples_seen_),
            "fit_seconds": fit_seconds,
        },
    }
    saved = save_pipeline_bundle(pipeline, output / "model", metadata, batch.X)
    check_inputs_unchanged(root, fresh)
    verify_review_history(root)
    record = {
        "status": "completed",
        "started_at_utc": started_utc,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "fit_seconds": fit_seconds,
        "training_cycles": 1328,
        "class_counts": [846, 246, 236],
        "pipeline_fits": 1,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "new_provenance_sha256": fresh["download_manifest_sha256"],
        "historical_provenance_sha256": split["payload"]["input_checksums"][
            "download_manifest_sha256"
        ],
        "model_sha256": saved["model_sha256"],
        "official_model_sha256": official["model_sha256"],
        "model_bytes_equal_official": saved["model_sha256"] == official["model_sha256"],
        "serialization_check": saved["serialization_check"],
        "warnings": [
            {"category": w.category.__name__, "message": str(w.message)} for w in captured
        ],
        "test_features_extracted": False,
        "test_predictions": False,
        "test_metrics": False,
        "official_artifacts_unchanged": True,
    }
    write_json_atomic(output / "results.json", record)
    return record
