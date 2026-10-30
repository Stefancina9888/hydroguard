"""Prespecified development-only baselines; the final test is never extracted or scored."""

import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from hydroguard.data import (
    ARCHIVE_SHA256,
    DataValidationError,
    audit_dataset,
    cycle_metadata,
    extraction_inventory,
    load_matrix,
    sha256_file,
    utc_now,
    validate_profile,
)
from hydroguard.features import (
    FEATURE_CONTRACT,
    FeatureBatch,
    extract_from_directory,
    rows_for_cycles,
    validate_X,
)
from hydroguard.metrics import MAIN_METRICS, evaluate_predictions, summarize_folds
from hydroguard.splitting import (
    PROTOCOL,
    assert_expected_memberships,
    load_split_lock,
    make_memberships,
)

MODEL_NAMES = ("dummy", "logistic_regression", "random_forest")
LOCK_RELATIVE_PATH = Path("artifacts/splits/phase2b-v1.json")


def load_baseline_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("feature_contract") != FEATURE_CONTRACT["name"]
        or tuple(config.get("models", {})) != MODEL_NAMES
        or config.get("thread_limit") != 1
        or config.get("class_order") != [0, 1, 2]
        or config.get("primary_metric") != "macro_f1"
        or config.get("final_test_prediction_allowed") is not False
    ):
        raise DataValidationError("Unexpected Phase 2B baseline configuration")
    return config


def verified_context(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    raw = root / "data/raw"
    audit, rows = audit_dataset(raw / "uci447", provenance_path=raw / "download.json")
    if not audit["validation_passed"]:
        raise DataValidationError(f"Input audit failed: {audit['errors']}")
    profile = validate_profile(load_matrix(raw / "uci447/profile.txt"))
    metadata = pd.DataFrame(cycle_metadata(profile)).set_index("cycle_id", verify_integrity=True)
    if metadata.reset_index().to_dict("records") != rows:
        raise DataValidationError("Audit metadata differ from the original profile")
    fingerprints = {
        "dataset_id": 447,
        "archive_sha256": ARCHIVE_SHA256,
        "extracted_sha256": {item["file"]: item["sha256"] for item in audit["inventory"]},
        "download_manifest_sha256": sha256_file(raw / "download.json"),
        "uv_lock_sha256": sha256_file(root / "uv.lock"),
        "baseline_config_sha256": sha256_file(root / "config/baseline-v1.json"),
    }
    return metadata, fingerprints, load_baseline_config(root / "config/baseline-v1.json")


def build_payload(metadata: pd.DataFrame, fingerprints: dict, config: dict) -> dict:
    memberships = make_memberships(metadata)
    assert_expected_memberships(metadata, memberships)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "input_checksums": fingerprints,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "feature_contract": FEATURE_CONTRACT,
        "baseline_config": config,
        "memberships": memberships,
    }


def check_inputs_unchanged(root: Path, fingerprints: dict) -> None:
    raw = root / "data/raw"
    if (
        sha256_file(raw / "uci447.zip") != fingerprints["archive_sha256"]
        or sha256_file(raw / "download.json") != fingerprints["download_manifest_sha256"]
        or extraction_inventory(raw / "uci447") != fingerprints["extracted_sha256"]
        or sha256_file(root / "uv.lock") != fingerprints["uv_lock_sha256"]
        or sha256_file(root / "config/baseline-v1.json") != fingerprints["baseline_config_sha256"]
    ):
        raise DataValidationError("Input data, environment lock or prespecified config changed")


def make_model(name: str, config: dict):
    parameters = config["models"][name]
    if name == "dummy":
        return DummyClassifier(**parameters)
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler(**parameters["scaler"])),
                ("classifier", LogisticRegression(**parameters["classifier"])),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(**parameters)
    raise DataValidationError(f"Model not authorized in Phase 2B: {name}")


def preflight_fit(payload: dict, metadata: pd.DataFrame, batch: FeatureBatch, fold: dict) -> None:
    """Run the full approved count/group/gap/class checks immediately before each fit."""
    if (
        payload["protocol"] != PROTOCOL
        or payload["feature_contract"] != FEATURE_CONTRACT
        or fold not in payload["memberships"]["inner"]
    ):
        raise DataValidationError("Protocol, features or requested inner fold changed")
    assert_expected_memberships(metadata, payload["memberships"])
    development = payload["memberships"]["outer"]["development_cycle_ids"]
    if batch.cycle_ids.tolist() != development:
        raise DataValidationError(
            "Feature rows must match development only in original cycle order"
        )
    validate_X(batch.X, expected_rows=len(development))
    rows_for_cycles(batch, fold["train_cycle_ids"])
    rows_for_cycles(batch, fold["validation_cycle_ids"])


def metrics_with_stability(truth, predictions, stability) -> dict:
    result = {"overall": evaluate_predictions(truth, predictions), "stability": {}}
    for flag in [0, 1]:
        mask = np.asarray(stability) == flag
        result["stability"][str(flag)] = evaluate_predictions(
            np.asarray(truth)[mask], predictions[mask]
        )
    return result


def run_development_cv(root: Path) -> tuple[dict, pd.DataFrame]:
    lock_path = root / LOCK_RELATIVE_PATH
    wrapper = load_split_lock(
        lock_path
    )  # Missing/invalid lock stops before feature extraction/fitting.
    metadata, fingerprints, config = verified_context(root)
    payload = build_payload(metadata, fingerprints, config)
    if wrapper["payload"] != payload:
        raise DataValidationError(
            "Current protocol/inputs/environment differ from the existing lock; stop"
        )
    started = utc_now()
    development = payload["memberships"]["outer"]["development_cycle_ids"]
    # Test/excluded sensor cycles are parsed only by the integrity loader, never summarized for X.
    batch = extract_from_directory(root / "data/raw/uci447", development)
    check_inputs_unchanged(root, fingerprints)
    metadata_original = metadata.copy(deep=True)
    oof = metadata.loc[development, ["pump_leakage", "stability_flag"]].copy()
    oof["inner_fold"] = -1
    for fold in payload["memberships"]["inner"]:
        oof.loc[fold["validation_cycle_ids"], "inner_fold"] = fold["fold"]
    report = {
        "schema_version": 1,
        "phase": "2B",
        "started_at_utc": started,
        "split_lock_created_at_utc": wrapper["created_at_utc"],
        "split_payload_sha256": wrapper["payload_sha256"],
        "split_file_sha256": sha256_file(lock_path),
        "environment": payload["environment"],
        "input_checksums": fingerprints,
        "feature_contract": FEATURE_CONTRACT,
        "baseline_config": config,
        "models": {},
        "final_test": {
            "membership_verified": True,
            "cycle_count": len(payload["memberships"]["outer"]["test_cycle_ids"]),
            "predictions_computed": False,
            "metrics_computed": False,
            "features_extracted": False,
        },
        "prior_eda_exposure": "All cycles, including the reserved test, were inspected in Phase 2A",
    }
    for name in MODEL_NAMES:
        fold_results = []
        predictions = pd.Series(index=oof.index, dtype=float)
        for fold in payload["memberships"]["inner"]:
            if load_split_lock(lock_path) != wrapper:
                raise DataValidationError("Split lock changed during execution; stop")
            if (
                sha256_file(root / "config/baseline-v1.json")
                != fingerprints["baseline_config_sha256"]
            ):
                raise DataValidationError(
                    "Prespecified model config changed during execution; stop"
                )
            preflight_fit(payload, metadata, batch, fold)
            train, validation = fold["train_cycle_ids"], fold["validation_cycle_ids"]
            X_train, X_validation = (
                rows_for_cycles(batch, train),
                rows_for_cycles(batch, validation),
            )
            y_train = metadata.loc[train, "pump_leakage"].to_numpy()
            y_validation = metadata.loc[validation, "pump_leakage"].to_numpy()
            model = make_model(name, config)
            with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X_train, y_train)
                predicted = model.predict(X_validation)
            fit_info = {
                "preflight_passed": True,
                "fit_cycles": len(train),
                "prediction_cycles": len(validation),
                "feature_count": X_train.shape[1],
                "warnings": [str(w.message) for w in captured],
            }
            if name == "logistic_regression":
                scaler = model.named_steps["scaler"]
                if scaler.n_samples_seen_ != len(train) or not np.allclose(
                    scaler.mean_, X_train.mean()
                ):
                    raise DataValidationError("Scaler was not fitted on inner training rows only")
                fit_info["scaler_training_samples"] = int(scaler.n_samples_seen_)
                fit_info["iterations"] = model.named_steps["classifier"].n_iter_.tolist()
            predictions.loc[validation] = predicted
            evaluations = metrics_with_stability(
                y_validation, predicted, metadata.loc[validation, "stability_flag"].to_numpy()
            )
            fold_results.append({"fold": fold["fold"], "fit_info": fit_info, **evaluations})
            print(
                f"{name}, fold {fold['fold']}: macro-F1={evaluations['overall']['macro_f1']:.6f}",
                flush=True,
            )
        if predictions.isna().any() or not (oof.inner_fold > 0).all():
            raise DataValidationError("Incomplete development out-of-fold predictions")
        predicted = predictions.astype(np.int64).to_numpy()
        oof[f"{name}_prediction"] = predicted
        pooled = metrics_with_stability(
            oof.pump_leakage.to_numpy(), predicted, oof.stability_flag.to_numpy()
        )
        mean = {
            "overall": summarize_folds([row["overall"] for row in fold_results]),
            "stability": {
                str(flag): summarize_folds([row["stability"][str(flag)] for row in fold_results])
                for flag in [0, 1]
            },
        }
        if (
            np.sum([row["overall"]["confusion_matrix"] for row in fold_results], axis=0).tolist()
            != pooled["overall"]["confusion_matrix"]
        ):
            raise DataValidationError("Pooled confusion matrix does not match fold totals")
        report["models"][name] = {"folds": fold_results, "fold_summary": mean, "pooled_oof": pooled}
    check_inputs_unchanged(root, fingerprints)
    if not metadata.equals(metadata_original) or oof.index.tolist() != development:
        raise DataValidationError("Original metadata order changed")
    report.update(
        completed_at_utc=utc_now(),
        raw_data_unchanged=True,
        metadata_unchanged=True,
        development_cycles=len(development),
        fits_completed=9,
    )
    return report, oof.reset_index()


def render_baseline_report(report: dict) -> str:
    def value(number):
        return "undefined" if number is None else f"{number:.6f}"

    def table(headers, rows):
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(map(str, row)) + " |" for row in rows],
            ]
        )

    def metrics_row(model, section, metrics):
        return [
            model,
            section,
            metrics["support"],
            "/".join(map(str, metrics["class_support"])),
            *[value(metrics[key]) for key in MAIN_METRICS],
        ]

    lines = [
        "# Phase 2B — development baseline results",
        "",
        f"Completed (UTC): {report['completed_at_utc']}",
        "",
        f"Split lock published before fitting (UTC): {report['split_lock_created_at_utc']}",
        "",
        f"Split payload SHA-256: `{report['split_payload_sha256']}`.",
        "",
        "The approved protocol and model parameters were locked before the first fit. "
        "Exactly nine fits were completed: three prespecified models × three development folds. "
        "No tuning, model-specific splits or post-result protocol changes were made.",
        "",
        "The reserved final test was used only for membership/integrity checks. No test "
        "features, predictions or metrics were computed. It was exposed during prior EDA, "
        "so it is not an untouched exploratory holdout.",
        "",
        "## Reproduce",
        "",
        "```text",
        "uv sync --locked --group baseline --python 3.11",
        "uv run --locked --group baseline python scripts/lock_splits.py",
        "uv run --locked --group baseline python scripts/run_baselines.py",
        "uv run --locked pytest",
        "uv run --locked ruff check .",
        "uv run --locked ruff format --check .",
        "uv lock --check",
        "```",
        "",
        "The locking command verifies an existing matching lock and refuses to overwrite a "
        "different/corrupted lock. Detailed membership, input checksums, feature order and "
        "parameters are in `artifacts/splits/phase2b-v1.json` and `config/baseline-v1.json`. "
        "Machine-readable results are in [baseline-results.json](baseline-results.json). "
        "Development OOF predictions are local in `data/processed/baseline-oof.csv`.",
        "",
        "## Fixed design and parameters",
        "",
        "Groups are `(cooler_condition_pct, valve_condition_pct, accumulator_pressure_bar)`. "
        "Outer SGKF: five folds, shuffle=True, seed=2026; fold 2 (one-based) is the "
        "339-cycle test (133/103/103). A 20-original-cycle-ID gap leaves 1,328 development "
        "cycles (846/246/236) and 538 excluded cycles. Inner SGKF uses three folds, the "
        "same seed and an additional inclusive gap 20 against each validation set. "
        "Before every fit, count, class, coverage, group, feature and original-ID gap "
        "checks passed. Inner training/validation sizes are 607/442, 576/457 and 607/429; "
        "minimum separation is 21. IDs are never compressed for gap calculations.",
        "",
        "Features: 14 physical sensors × mean, population std (ddof=0), RMS, min, max = "
        "70 float64 columns in a fixed sensor-major order. CE/CP/SE are excluded. Cycle IDs "
        "are a separate array, and target, condition labels, stability and groups never "
        "enter X. There are no fitted EDA values. Only development cycles were summarized.",
        "",
        "```json",
        json.dumps(report["baseline_config"], indent=2),
        "```",
        "",
        "Logistic regression uses L2 regularization (`l1_ratio=0.0` in the locked "
        "scikit-learn 1.9 API) and a StandardScaler fitted inside each training-only "
        "Pipeline. No class weights were applied. RF uses 200 trees and one worker. "
        "All fits/predictions use a single numerical thread; effective library versions "
        f"are `{json.dumps(report['environment'], sort_keys=True)}`.",
        "",
        "## Development CV and pooled OOF",
        "",
        "Macro-F1 is the primary metric. Fold mean/std are unweighted (std ddof=0). "
        "Pooled metrics are recomputed from exactly one OOF prediction per development "
        "cycle; they are not averages of fold metrics.",
        "",
    ]
    rows = []
    for name, model in report["models"].items():
        rows.extend(
            metrics_row(name, f"fold {fold['fold']}", fold["overall"]) for fold in model["folds"]
        )
        mean = model["fold_summary"]["overall"]
        rows.append(
            [
                name,
                "fold mean ± std",
                mean["support_sum"],
                "see folds",
                *[
                    f"{value(mean['metric_means'][key])} ± {value(mean['metric_std_ddof0'][key])}"
                    for key in MAIN_METRICS
                ],
            ]
        )
        rows.append(metrics_row(name, "pooled OOF", model["pooled_oof"]["overall"]))
    lines += [
        table(
            [
                "Model",
                "Evaluation",
                "Support (mean row: sum)",
                "True classes 0/1/2",
                "Macro-F1",
                "Balanced accuracy",
                "Class 2 recall",
            ],
            rows,
        ),
        "",
        "## Per-class metrics",
        "",
        "Undefined precision means no predictions for that class; its supported-class F1 "
        "is still zero when false negatives exist. Class order is always 0/1/2.",
        "",
    ]
    rows = []
    for name, model in report["models"].items():
        for label, metrics in [(f"fold {f['fold']}", f["overall"]) for f in model["folds"]] + [
            ("pooled OOF", model["pooled_oof"]["overall"])
        ]:
            for row in metrics["per_class"]:
                rows.append(
                    [
                        name,
                        label,
                        row["label"],
                        row["support"],
                        value(row["precision"]),
                        value(row["recall"]),
                        value(row["f1"]),
                    ]
                )
        for row in model["fold_summary"]["overall"]["per_class_means"]:
            rows.append(
                [
                    name,
                    "fold mean",
                    row["label"],
                    f"{row['support_mean']:.3f}",
                    value(row["precision"]),
                    value(row["recall"]),
                    value(row["f1"]),
                ]
            )
    lines += [
        table(["Model", "Evaluation", "Class", "Support", "Precision", "Recall", "F1"], rows),
        "",
        "## Confusion matrices",
        "",
        "Rows = true class, columns = predicted class, order `[0, 1, 2]`. "
        "Pooled counts equal the sum of fold matrices.",
        "",
    ]
    for name, model in report["models"].items():
        for label, metrics in [(f"fold {f['fold']}", f["overall"]) for f in model["folds"]] + [
            ("pooled OOF", model["pooled_oof"]["overall"])
        ]:
            lines += [
                f"### {name} — {label}",
                "",
                "```text",
                *[" ".join(f"{count:4d}" for count in row) for row in metrics["confusion_matrix"]],
                "```",
                "",
            ]
    lines += [
        "## Stability subgroups",
        "",
        "Flag 0 = stable; flag 1 = static conditions may not have been reached. Every "
        "validation case is retained. The same OOF predictions are stratified afterward; "
        "no subgroup-specific model is fitted. A fixed-three-class macro-F1/balanced "
        "accuracy is marked undefined if an actual class is absent. Per-class recall is "
        "undefined at zero true support, precision at zero predicted support, and F1 when "
        "both are zero. Missing values are JSON null, never silently replaced with zero. "
        "An unweighted mean is undefined if any constituent fold metric is undefined.",
        "",
    ]
    rows = []
    for name, model in report["models"].items():
        for flag in ["0", "1"]:
            for label, metrics in [
                (f"fold {f['fold']}", f["stability"][flag]) for f in model["folds"]
            ] + [("pooled OOF", model["pooled_oof"]["stability"][flag])]:
                rows.append(
                    [
                        flag,
                        *metrics_row(name, label, metrics),
                        ", ".join(metrics["undefined_metrics"]) or "none",
                    ]
                )
            mean = model["fold_summary"]["stability"][flag]
            rows.append(
                [
                    flag,
                    name,
                    "fold mean",
                    mean["support_sum"],
                    "see folds",
                    *[value(mean["metric_means"][key]) for key in MAIN_METRICS],
                    "any undefined fold makes its mean undefined",
                ]
            )
    lines += [
        table(
            [
                "Flag",
                "Model",
                "Evaluation",
                "Support (mean row: sum)",
                "Classes 0/1/2",
                "Macro-F1",
                "Balanced accuracy",
                "Class 2 recall",
                "Undefined metrics",
            ],
            rows,
        ),
        "",
        "Pooled subgroup per-class details are below; every fold's subgroup class metrics "
        "and confusion matrix are also retained in the companion JSON.",
        "",
    ]
    rows = []
    for name, model in report["models"].items():
        for flag in ["0", "1"]:
            for row in model["pooled_oof"]["stability"][flag]["per_class"]:
                rows.append(
                    [
                        flag,
                        name,
                        row["label"],
                        row["support"],
                        value(row["precision"]),
                        value(row["recall"]),
                        value(row["f1"]),
                    ]
                )
    logistic_oof = report["models"]["logistic_regression"]["pooled_oof"]["overall"]
    forest_oof = report["models"]["random_forest"]["pooled_oof"]["overall"]
    unstable_support = report["models"]["dummy"]["pooled_oof"]["stability"]["1"]["class_support"]
    lines += [
        table(["Flag", "Model", "Class", "Support", "Precision", "Recall", "F1"], rows),
        "",
        "## Interpretation and limitations",
        "",
        f"Observed pooled development macro-F1: logistic regression "
        f"{value(logistic_oof['macro_f1'])}, random forest {value(forest_oof['macro_f1'])}. "
        f"Their class 2 recall values are {value(logistic_oof['recall_class_2'])} and "
        f"{value(forest_oof['recall_class_2'])}; class 2 precision values are "
        f"{value(logistic_oof['per_class'][2]['precision'])} and "
        f"{value(forest_oof['per_class'][2]['precision'])}. These expose a recall/precision "
        "tradeoff that a single macro-F1 value does not describe. Close development "
        "macro-F1 values do not establish a statistically reliable winning model.",
        "",
        f"The pooled flag-1 true class support is {unstable_support} in class order "
        "0/1/2. Fold 1 has no flagged class-2 validation cases, so its corresponding "
        "recall and fixed-three-class aggregate metrics, and the aggregate fold means, "
        "remain undefined. The pooled flagged scores use only a handful of fault cases; "
        "they cannot establish robust performance on unstable faults.",
        "",
        "These are first prespecified development baselines, not tuned or final-test "
        "results. Macro-F1, balanced accuracy and class 2 recall answer different questions; "
        "read the class supports and confusion matrices before preferring a model. "
        "No winning-model refit on all development data or final-test scoring was performed.",
        "",
        "The test was exposed in Phase 2A EDA. Configuration groups are not verified "
        "independent experiments; gap 20 is not a validated decorrelation time. Purging "
        "changes training sizes/class mixtures and can remove rare unstable fault cases. "
        "Stability subgroup support is small and does not justify precise reliability "
        "claims. Fold standard deviation is descriptive, not an independent-sample "
        "confidence interval. No new-rig/general industrial performance claim is supported.",
        "",
        "## Execution integrity",
        "",
        "All expected split/count/group/gap/class checks "
        "passed before every fit. Raw-data checksums and original metadata were unchanged "
        "at completion. The feature matrix contained exactly 70 allowed sensor columns. "
        "See [validation-plan.md](validation-plan.md) for the frozen protocol and previous "
        "EDA exposure. Test predictions and metrics remain prohibited in Phase 2B.",
        "",
    ]
    if "verification" in report:
        verification = report["verification"]
        lines += [
            "## Completed validation",
            "",
            f"Verification (UTC): {verification['completed_at_utc']}. "
            f"Offline tests: {verification['tests']} passed, "
            f"{verification['failures']} failures, {verification['errors']} errors, "
            f"{verification['skipped']} skipped. Ruff lint, Ruff format and uv lock "
            "checks passed. Independent scikit-learn recomputation of pooled metrics "
            "agreed with the saved OOF results. Fold/stability scores, original-ID "
            "membership/gaps, raw checksums, unchanged split-lock bytes and exclusion "
            "of final-test/outer-excluded OOF IDs were verified.",
            "",
        ]
    return "\n".join(lines)
