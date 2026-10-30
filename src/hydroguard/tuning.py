"""Preregistered, bounded development tuning on existing split memberships only."""

import copy
import itertools
import json
import platform
import random
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from hydroguard.baselines import (
    LOCK_RELATIVE_PATH,
    check_inputs_unchanged,
    make_model,
    metrics_with_stability,
    verified_context,
)
from hydroguard.data import DataValidationError, sha256_file, utc_now, write_json_atomic
from hydroguard.features import (
    FEATURE_CONTRACT,
    extract_from_directory,
    rows_for_cycles,
    validate_cycle_ids,
    validate_X,
)
from hydroguard.metrics import summarize_folds
from hydroguard.splitting import (
    EXPECTED_INNER,
    EXPECTED_OUTER,
    PROTOCOL,
    canonical_sha256,
    check_partition,
    class_counts,
    load_split_lock,
    lock_or_verify,
    nearest_cycle_distance,
    validate_metadata,
)

CONFIG_PATH = Path("config/tuning-v1.json")
MANIFEST_PATH = Path("artifacts/experiments/phase3a-v1.json")
PROTECTED_PATHS = (
    "artifacts/splits/phase2b-v1.json",
    "config/baseline-v1.json",
    "docs/baseline-results.json",
    "docs/baseline-results.md",
    "uv.lock",
    "pyproject.toml",
    ".python-version",
    "src/hydroguard/features.py",
    "src/hydroguard/splitting.py",
    "src/hydroguard/baselines.py",
    "src/hydroguard/data.py",
    "src/hydroguard/metrics.py",
)
IMPLEMENTATION_PATHS = (
    "src/hydroguard/tuning.py",
    "src/hydroguard/tuning_report.py",
    "scripts/run_tuning.py",
)
SELECTION_RULE = {
    "primary": "Maximum unweighted mean of three fold macro-F1 scores",
    "secondary": "Maximum unweighted mean of three fold class-2 recalls",
    "ties": "Exact full-precision float equality; no rounding or tolerance",
    "lr_simplicity": "Lower C, then class_weight=None before balanced",
    "rf_simplicity": "Lower bounded max_depth (None last), larger min_samples_leaf, "
    "max_features sqrt before 0.5 before 1.0, then unweighted before balanced",
    "final_tie": "Lexicographically smaller candidate_id",
    "reference_candidates_eligible": True,
    "eligibility": "All three fits and predictions successful, finite selection metrics, "
    "and no ConvergenceWarning; other warnings are retained",
    "failures": "Structural/integrity violations stop the experiment. Model exceptions are "
    "recorded per fold and make the candidate ineligible; no parameter changes.",
    "baseline_reproduction": "Reference candidate OOF predictions must equal the saved baseline "
    "predictions when present, and reference metrics must agree within 1e-12",
}


def simplicity_key(candidate: dict) -> tuple:
    params = candidate["parameters"]
    if candidate["family"] == "logistic_regression":
        params = params["classifier"]
        return params["C"], int(params["class_weight"] is not None)
    if candidate["family"] == "random_forest":
        feature_rank = {"sqrt": 0, 0.5: 1, 1.0: 2}[params["max_features"]]
        return (
            params["max_depth"] if params["max_depth"] is not None else 2147483647,
            -params["min_samples_leaf"],
            feature_rank,
            int(params["class_weight"] is not None),
        )
    raise DataValidationError("Only Logistic Regression and Random Forest are authorized")


def candidate_configuration(baseline: dict) -> dict:
    candidates = []
    for index, (regularization, weight) in enumerate(
        itertools.product([0.01, 0.1, 1, 10, 100], [None, "balanced"]), start=1
    ):
        params = copy.deepcopy(baseline["models"]["logistic_regression"])
        params["classifier"].update(C=regularization, class_weight=weight)
        candidates.append(
            {
                "candidate_id": f"lr_{index:02d}",
                "family": "logistic_regression",
                "parameters": params,
                "baseline_reference": regularization == 1 and weight is None,
            }
        )
    base_rf = copy.deepcopy(baseline["models"]["random_forest"])
    candidates.append(
        {
            "candidate_id": "rf_baseline",
            "family": "random_forest",
            "parameters": base_rf,
            "baseline_reference": True,
        }
    )
    pool = []
    for depth, leaf, features, weight in itertools.product(
        [None, 6, 12, 20], [1, 2, 4, 8], ["sqrt", 0.5, 1.0], [None, "balanced"]
    ):
        params = {
            **base_rf,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": features,
            "class_weight": weight,
        }
        if params != base_rf:
            pool.append(params)
    indices = random.Random(2026).sample(range(len(pool)), 12)
    for index, position in enumerate(indices, start=1):
        candidates.append(
            {
                "candidate_id": f"rf_{index:02d}",
                "family": "random_forest",
                "parameters": pool[position],
                "baseline_reference": False,
            }
        )
    for candidate in candidates:
        candidate["simplicity_key"] = list(simplicity_key(candidate))
    return {
        "schema_version": 1,
        "phase": "3A",
        "candidates": candidates,
        "selection_rule": SELECTION_RULE,
        "rf_sampling": {
            "seed": 2026,
            "algorithm": "Python random.Random.sample; without replacement",
            "enumeration_order": ["max_depth", "min_samples_leaf", "max_features", "class_weight"],
            "max_depth": [None, 6, 12, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.5, 1.0],
            "class_weight": [None, "balanced"],
            "space_size": 96,
            "pool_size_excluding_reference": len(pool),
            "sampled_pool_indices_0_based": indices,
            "baseline_excluded_from_sample_and_added_separately": True,
        },
        "expected_candidates": 23,
        "expected_classifier_fits": 69,
        "folds_per_candidate": 3,
        "thread_limit": 1,
        "refit_selected": False,
        "new_cv_generation": False,
        "test_predictions_allowed": False,
        "outer_excluded_predictions_allowed": False,
    }


def validate_member_roles(metadata, memberships, expected_outer, expected_inner) -> None:
    """Validate saved roles directly. No splitter or membership generator is called."""
    validate_metadata(metadata)
    outer = memberships["outer"]

    def check_roles(roles, expectations, universe):
        coverage = []
        for role, (size, counts) in expectations.items():
            values = roles[f"{role}_cycle_ids"]
            ids = validate_cycle_ids(values) if len(values) else np.array([], dtype=np.int64)
            if not np.isin(ids, metadata.index).all():
                raise DataValidationError("Saved role contains unknown original cycle IDs")
            if len(ids) != size or class_counts(metadata, ids) != list(counts):
                raise DataValidationError(f"Locked {role} count/class support changed")
            coverage.extend(ids.tolist())
        if sorted(coverage) != list(universe):
            raise DataValidationError("Saved roles are not a disjoint complete partition")

    check_roles(outer, expected_outer, metadata.index)
    development, test = outer["development_cycle_ids"], outer["test_cycle_ids"]
    if check_partition(metadata, development, test) != outer["diagnostics"]:
        raise DataValidationError("Saved outer diagnostics do not match metadata")
    if np.any(nearest_cycle_distance(outer["excluded_cycle_ids"], test) > 20):
        raise DataValidationError("Outer exclusion is outside the original-ID test gap")
    if len(memberships["inner"]) != len(expected_inner):
        raise DataValidationError("Wrong number of saved inner folds")
    validations = []
    for number, (fold, expected) in enumerate(
        zip(memberships["inner"], expected_inner, strict=True), 1
    ):
        if fold["fold"] != number:
            raise DataValidationError("Saved inner fold numbering changed")
        check_roles(fold, expected, development)
        if (
            check_partition(metadata, fold["train_cycle_ids"], fold["validation_cycle_ids"])
            != fold["diagnostics"]
        ):
            raise DataValidationError("Saved inner diagnostics do not match metadata")
        if np.any(
            nearest_cycle_distance(fold["excluded_cycle_ids"], fold["validation_cycle_ids"]) > 20
        ):
            raise DataValidationError("Inner exclusion is outside the original-ID validation gap")
        validations.extend(fold["validation_cycle_ids"])
    if sorted(validations) != development:
        raise DataValidationError("Development OOF coverage is not exactly once")


def validate_frozen_memberships(metadata, payload) -> None:
    if payload["protocol"] != PROTOCOL or payload["feature_contract"] != FEATURE_CONTRACT:
        raise DataValidationError("Existing protocol or 70-feature contract changed")
    if metadata.index.tolist() != list(range(1, 2206)):
        raise DataValidationError("Expected original cycle IDs 1 through 2205")
    validate_member_roles(metadata, payload["memberships"], EXPECTED_OUTER, EXPECTED_INNER)


def protected_hashes(root: Path) -> dict:
    paths = list(PROTECTED_PATHS)
    if (root / "data/processed/baseline-oof.csv").exists():
        paths.append("data/processed/baseline-oof.csv")
    return {name: sha256_file(root / name) for name in paths}


def verify_protected(root: Path, expected: dict) -> None:
    if protected_hashes(root) != expected:
        raise DataValidationError("Protected split, baseline, dependency or feature files changed")


def read_frozen_context(root: Path):
    wrapper = load_split_lock(root / LOCK_RELATIVE_PATH)
    metadata, fingerprints, baseline = verified_context(root)
    payload = wrapper["payload"]
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if (
        payload["input_checksums"] != fingerprints
        or payload["environment"] != versions
        or payload["baseline_config"] != baseline
    ):
        raise DataValidationError(
            "Current input/environment/baseline differs from the existing split lock"
        )
    validate_frozen_memberships(metadata, payload)
    baseline_results = json.loads((root / "docs/baseline-results.json").read_text(encoding="utf-8"))
    if (
        baseline_results["split_payload_sha256"] != wrapper["payload_sha256"]
        or baseline_results["feature_contract"] != FEATURE_CONTRACT
        or baseline_results["baseline_config"] != baseline
        or baseline_results["input_checksums"] != fingerprints
    ):
        raise DataValidationError("Saved baseline results refer to different inputs or protocol")
    return wrapper, metadata, baseline, baseline_results


def experiment_payload(root: Path, wrapper: dict, configuration: dict) -> dict:
    return {
        "schema_version": 1,
        "phase": "3A",
        "config_file": CONFIG_PATH.as_posix(),
        "config_file_sha256": sha256_file(root / CONFIG_PATH),
        "config_payload_sha256": canonical_sha256(configuration),
        "split_reference": {
            "path": LOCK_RELATIVE_PATH.as_posix(),
            "file_sha256": sha256_file(root / LOCK_RELATIVE_PATH),
            "payload_sha256": wrapper["payload_sha256"],
            "memberships_sha256": canonical_sha256(wrapper["payload"]["memberships"]),
        },
        "protected_files": protected_hashes(root),
        "implementation_sha256": {name: sha256_file(root / name) for name in IMPLEMENTATION_PATHS},
        "input_checksums": wrapper["payload"]["input_checksums"],
        "environment": wrapper["payload"]["environment"],
        "feature_contract": FEATURE_CONTRACT,
        "expected_classifier_fits": 69,
        "new_memberships_generated": False,
        "no_full_development_refit": True,
        "final_test_and_outer_excluded": "Integrity/membership only",
    }


def prepare_experiment(root: Path) -> tuple[dict, dict]:
    wrapper, _, baseline, _ = read_frozen_context(root)
    configuration = candidate_configuration(baseline)
    locked_config = lock_or_verify(root / CONFIG_PATH, configuration)
    manifest = lock_or_verify(
        root / MANIFEST_PATH, experiment_payload(root, wrapper, configuration)
    )
    return locked_config, manifest


def load_experiment(root: Path):
    # Require preregistration before any feature extraction or classifier construction.
    config = load_split_lock(root / CONFIG_PATH)
    manifest = load_split_lock(root / MANIFEST_PATH)
    wrapper, metadata, baseline, baseline_results = read_frozen_context(root)
    expected_config = candidate_configuration(baseline)
    if config["payload"] != expected_config:
        raise DataValidationError(
            "Tuning candidates or selection rule differ from the frozen specification"
        )
    if manifest["payload"] != experiment_payload(root, wrapper, expected_config):
        raise DataValidationError(
            "Experiment manifest/reference/implementation changed; no automatic overwrite"
        )
    return config, manifest, wrapper, metadata, baseline, baseline_results


def guard_development_request(payload: dict, batch, fold: dict, validation_ids) -> None:
    development = payload["memberships"]["outer"]["development_cycle_ids"]
    validate_cycle_ids(batch.cycle_ids)
    if batch.cycle_ids.tolist() != development:
        raise DataValidationError(
            "Feature batch must contain development cycles only, in original order"
        )
    validate_X(batch.X, expected_rows=len(development))
    if (
        fold not in payload["memberships"]["inner"]
        or list(validation_ids) != fold["validation_cycle_ids"]
    ):
        raise DataValidationError("Only the saved inner validation IDs can receive predictions")
    if not set(validation_ids) <= set(development):
        raise DataValidationError("Final-test/outer-excluded cycles cannot receive predictions")
    rows_for_cycles(batch, fold["train_cycle_ids"])
    rows_for_cycles(batch, validation_ids)


def make_candidate_model(candidate: dict, baseline: dict):
    if candidate["family"] not in ("logistic_regression", "random_forest"):
        raise DataValidationError("Unauthorized tuning model family")
    config = copy.deepcopy(baseline)
    config["models"][candidate["family"]] = copy.deepcopy(candidate["parameters"])
    return make_model(candidate["family"], config)


def fit_candidate_fold(
    candidate, baseline, X_train, y_train, X_validation
) -> tuple[dict, np.ndarray | None]:
    """Capture every model warning/error; integrity checks happen outside this scope."""
    model = make_candidate_model(candidate, baseline)
    started = time.perf_counter()
    record = {
        "fit_attempted": True,
        "fit_completed": False,
        "prediction_completed": False,
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "warnings": [],
        "error": None,
    }
    predicted = None
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        try:
            fit_started = time.perf_counter()
            try:
                model.fit(X_train, y_train)
            finally:
                record["fit_seconds"] = time.perf_counter() - fit_started
            record["fit_completed"] = True
            if candidate["family"] == "logistic_regression":
                scaler = model.named_steps["scaler"]
                if scaler.n_samples_seen_ != len(y_train) or not np.allclose(
                    scaler.mean_, X_train.mean()
                ):
                    raise DataValidationError(
                        "Scaler was not fitted on the fold's training rows only"
                    )
                record["scaler_training_samples"] = int(scaler.n_samples_seen_)
                record["iterations"] = model.named_steps["classifier"].n_iter_.tolist()
            prediction_started = time.perf_counter()
            try:
                predicted = model.predict(X_validation)
            finally:
                record["predict_seconds"] = time.perf_counter() - prediction_started
            record["prediction_completed"] = True
        except DataValidationError:
            raise
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["warnings"] = [
        {"category": warning.category.__name__, "message": str(warning.message)}
        for warning in captured
    ]
    record["convergence_warning"] = any(
        issubclass(warning.category, ConvergenceWarning) for warning in captured
    )
    record["elapsed_seconds"] = time.perf_counter() - started
    return record, predicted


def select_candidate(records: list[dict], family: str) -> dict:
    eligible = [record for record in records if record["family"] == family and record["eligible"]]
    if not eligible:
        raise DataValidationError(f"No eligible candidate in family {family}")
    return min(
        eligible,
        key=lambda record: (
            -record["mean_macro_f1"],
            -record["mean_recall_class_2"],
            tuple(record["simplicity_key"]),
            record["candidate_id"],
        ),
    )


def verify_baseline_reference(
    record: dict, previous: dict, prediction, saved_oof: pd.DataFrame | None
):
    if not record["eligible"]:
        raise DataValidationError("Baseline reference did not complete as an eligible candidate")
    family = record["family"]
    old = previous["models"][family]
    for fresh, original in zip(record["folds"], old["folds"], strict=True):
        for metric in ("macro_f1", "balanced_accuracy", "recall_class_2"):
            if not np.isclose(
                fresh["metrics"]["overall"][metric], original["overall"][metric], rtol=0, atol=1e-12
            ):
                raise DataValidationError(
                    "Baseline reference metric changed on the same saved fold"
                )
    if saved_oof is not None and not np.array_equal(
        prediction, saved_oof[f"{family}_prediction"].to_numpy()
    ):
        raise DataValidationError(
            "Baseline reference OOF predictions differ from the saved baseline"
        )


def run_tuning(root: Path) -> tuple[dict, pd.DataFrame]:
    wall_started = time.perf_counter()
    config_lock, manifest, wrapper, metadata, baseline, previous = load_experiment(root)
    config, payload = config_lock["payload"], wrapper["payload"]
    development = payload["memberships"]["outer"]["development_cycle_ids"]
    protected = manifest["payload"]["protected_files"]
    run_id = utc_now().replace(":", "-") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / "data/processed/tuning-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "phase": "3A",
        "status": "running",
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "config_created_at_utc": config_lock["created_at_utc"],
        "manifest_created_at_utc": manifest["created_at_utc"],
        "manifest_payload_sha256": manifest["payload_sha256"],
        "manifest_file_sha256": sha256_file(root / MANIFEST_PATH),
        "experiment_manifest": manifest["payload"],
        "configuration": config,
        "candidates": [],
        "fit_attempts": 0,
        "fit_completions": 0,
        "baseline_comparison": {},
        "selected": {},
        "final_test": {
            "membership_integrity_only": True,
            "features_extracted": False,
            "predictions_computed": False,
            "metrics_computed": False,
        },
        "outer_excluded": {
            "cycle_count": 538,
            "features_extracted": False,
            "predictions_computed": False,
            "metrics_computed": False,
        },
        "new_cv_generated": False,
        "full_development_refits": 0,
    }
    progress_path = run_dir / "progress.json"
    write_json_atomic(progress_path, report)
    try:
        batch = extract_from_directory(root / "data/raw/uci447", development)
        check_inputs_unchanged(root, payload["input_checksums"])
        oof = metadata.loc[development, ["pump_leakage", "stability_flag"]].copy()
        oof["inner_fold"] = -1
        for fold in payload["memberships"]["inner"]:
            oof.loc[fold["validation_cycle_ids"], "inner_fold"] = fold["fold"]
        baseline_oof_path = root / "data/processed/baseline-oof.csv"
        old_oof = (
            pd.read_csv(baseline_oof_path).set_index("cycle_id")
            if baseline_oof_path.exists()
            else None
        )
        if old_oof is not None and old_oof.index.tolist() != development:
            raise DataValidationError(
                "Saved baseline OOF membership/order differs from development"
            )
        for candidate in config["candidates"]:
            candidate_started = time.perf_counter()
            record = {
                **copy.deepcopy(candidate),
                "folds": [],
                "eligible": False,
                "mean_macro_f1": None,
                "mean_recall_class_2": None,
            }
            report["candidates"].append(record)
            predictions = pd.Series(pd.NA, index=oof.index, dtype="Int64")
            for fold in payload["memberships"]["inner"]:
                verify_protected(root, protected)
                if (
                    load_split_lock(root / MANIFEST_PATH) != manifest
                    or load_split_lock(root / CONFIG_PATH) != config_lock
                ):
                    raise DataValidationError("Tuning preregistration changed during execution")
                validate_frozen_memberships(metadata, payload)
                validation = fold["validation_cycle_ids"]
                guard_development_request(payload, batch, fold, validation)
                train = fold["train_cycle_ids"]
                X_train, X_validation = (
                    rows_for_cycles(batch, train),
                    rows_for_cycles(batch, validation),
                )
                y_train = metadata.loc[train, "pump_leakage"].to_numpy()
                report["fit_attempts"] += 1
                report["current_fit"] = {
                    "candidate_id": candidate["candidate_id"],
                    "fold": fold["fold"],
                }
                write_json_atomic(progress_path, report)
                fit, predicted = fit_candidate_fold(
                    candidate, baseline, X_train, y_train, X_validation
                )
                report["fit_completions"] += int(fit["fit_completed"])
                result = {
                    "fold": fold["fold"],
                    "train_cycles": len(train),
                    "validation_cycles": len(validation),
                    "preflight_passed": True,
                    **fit,
                }
                if predicted is not None:
                    result["metrics"] = metrics_with_stability(
                        metadata.loc[validation, "pump_leakage"].to_numpy(),
                        predicted,
                        metadata.loc[validation, "stability_flag"].to_numpy(),
                    )
                    predictions.loc[validation] = predicted
                record["folds"].append(result)
                print(
                    f"{candidate['candidate_id']} fold {fold['fold']}: "
                    f"macro-F1={result.get('metrics', {}).get('overall', {}).get('macro_f1')}; "
                    f"warnings={len(fit['warnings'])}; fit={fit['fit_seconds']:.3f}s",
                    flush=True,
                )
                write_json_atomic(progress_path, report)
            complete = not predictions.isna().any() and all(
                "metrics" in fold for fold in record["folds"]
            )
            if complete:
                record["fold_summary"] = summarize_folds(
                    [fold["metrics"]["overall"] for fold in record["folds"]]
                )
                record["mean_macro_f1"] = record["fold_summary"]["metric_means"]["macro_f1"]
                record["mean_recall_class_2"] = record["fold_summary"]["metric_means"][
                    "recall_class_2"
                ]
                record["eligible"] = all(
                    np.isfinite(record[key]) for key in ("mean_macro_f1", "mean_recall_class_2")
                ) and not any(
                    fold["convergence_warning"] or fold["error"] for fold in record["folds"]
                )
            record["elapsed_seconds"] = time.perf_counter() - candidate_started
            record["status"] = "eligible" if record["eligible"] else "ineligible"
            oof[f"{candidate['candidate_id']}_prediction"] = predictions
            if candidate["baseline_reference"]:
                verify_baseline_reference(
                    record, previous, predictions.to_numpy(dtype=np.int64), old_oof
                )
                record["baseline_reproduced"] = True
        for family in ("logistic_regression", "random_forest"):
            winner = select_candidate(report["candidates"], family)
            predicted = oof[f"{winner['candidate_id']}_prediction"].to_numpy(dtype=np.int64)
            selected_folds = [fold["metrics"] for fold in winner["folds"]]
            pooled = metrics_with_stability(
                oof.pump_leakage.to_numpy(), predicted, oof.stability_flag.to_numpy()
            )
            report["selected"][family] = {
                "candidate_id": winner["candidate_id"],
                "parameters": winner["parameters"],
                "mean_macro_f1": winner["mean_macro_f1"],
                "mean_recall_class_2": winner["mean_recall_class_2"],
                "fold_summary": {
                    "overall": winner["fold_summary"],
                    "stability": {
                        str(flag): summarize_folds(
                            [fold["stability"][str(flag)] for fold in selected_folds]
                        )
                        for flag in [0, 1]
                    },
                },
                "pooled_oof": pooled,
            }
            old = previous["models"][family]
            report["baseline_comparison"][family] = {
                "baseline_fold_mean": old["fold_summary"]["overall"]["metric_means"],
                "baseline_pooled_oof": old["pooled_oof"],
                "mean_macro_f1_difference": winner["mean_macro_f1"]
                - old["fold_summary"]["overall"]["metric_means"]["macro_f1"],
                "pooled_macro_f1_difference": pooled["overall"]["macro_f1"]
                - old["pooled_oof"]["overall"]["macro_f1"],
            }
        if report["fit_attempts"] != 69 or report["fit_completions"] != 69:
            raise DataValidationError("Expected 69 attempted and completed classifier fits")
        verify_protected(root, protected)
        check_inputs_unchanged(root, payload["input_checksums"])
        if oof.index.tolist() != development or not (oof.inner_fold > 0).all():
            raise DataValidationError("OOF order/coverage changed")
        report.update(
            status="completed",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.perf_counter() - wall_started,
            warning_count=sum(
                len(fold["warnings"]) for record in report["candidates"] for fold in record["folds"]
            ),
            protected_files_unchanged=True,
            raw_data_unchanged=True,
            protected_files_after=protected_hashes(root),
        )
        report.pop("current_fit", None)
        write_json_atomic(run_dir / "results.json", report)
        write_json_atomic(progress_path, report)
        return report, oof.reset_index()
    except BaseException as error:
        report.update(
            status="failed",
            failed_at_utc=utc_now(),
            elapsed_seconds=time.perf_counter() - wall_started,
            failure={"type": type(error).__name__, "message": str(error)},
        )
        write_json_atomic(progress_path, report)
        raise
