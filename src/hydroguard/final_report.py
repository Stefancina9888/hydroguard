"""Render the selected-model reserved-test evaluation separately from development selection."""

import json

from hydroguard.metrics import MAIN_METRICS


def render_final_report(result: dict) -> str:
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

    overall = result["test_results"]["overall"]
    development = result["development_selection"]
    lines = [
        "# Phase 3B — fixed Logistic Regression final-test evaluation",
        "",
        "**Reserved-test results for the single model fixed before test prediction. "
        "Development selection results are reported separately and are not pooled with the test.**",
        "",
        f"Completed (UTC): {result['completed_at_utc']}. "
        f"Evaluation elapsed: {result['elapsed_seconds']:.3f} seconds.",
        "",
        f"Phase 3B preregistration (UTC): {result['phase3b_created_at_utc']}; "
        f"test-access receipt (UTC): {result['test_access_created_at_utc']}.",
        "",
        f"Manifest payload SHA-256: `{result['phase3b_payload_sha256']}`.",
        "",
        "## Fixed model and chronological procedure",
        "",
        "The project owner selected Logistic Regression `lr_09` (C=100, class_weight=None) "
        "from the locked Phase 3A candidate configuration before this evaluation. No RF or "
        "alternative model was evaluated on the reserved test. All other constructor "
        "parameters and the StandardScaler Pipeline remained unchanged:",
        "",
        "```json",
        json.dumps(result["parameters"], indent=2),
        "```",
        "",
        "A separate immutable `artifacts/experiments/phase3b-v1.json` was published before "
        "fitting and before any test sensor feature/prediction access. It references the "
        "existing split, tuning configuration/manifest/results and 70-feature contract. "
        "One fresh complete Pipeline was fitted on exactly the 1,328 locked development "
        "cycles (846/246/236). StandardScaler observed only these training rows. "
        "The 538 outer-excluded cycles were neither fitted nor evaluated. Original "
        "IDs, row order, group definitions and gap rules were retained.",
        "",
        f"Fit completed (UTC): {result['model_training']['completed_at_utc']}; "
        f"fit duration: {result['model_training']['fit_seconds']:.6f} seconds; "
        f"scaler training samples: {result['model_training']['scaler_training_samples']}; "
        f"LR iterations: {result['model_training']['classifier_iterations']}.",
        "",
        "After training and bundle verification, only this saved model received the 339 "
        "reserved cycles (133/103/103). No parameter, feature, class decision, threshold, "
        "split or protocol was changed after observing the final-test results. "
        "No training on all 2,205 cycles, further tuning, calibration, API or Docker "
        "was performed.",
        "",
        "## Final reserved-test results",
        "",
        table(
            ["Cases", "True classes 0/1/2", "Macro-F1", "Balanced accuracy", "Class-2 recall"],
            [
                [
                    overall["support"],
                    "/".join(map(str, overall["class_support"])),
                    *[value(overall[key]) for key in MAIN_METRICS],
                ]
            ],
        ),
        "",
        "Macro-F1 is primary. Undefined values are shown explicitly rather than "
        "silently dropped. Precision uses predicted support and recall uses true support.",
        "",
    ]
    rows = []
    for row in overall["per_class"]:
        rows.append(
            [
                row["label"],
                row["support"],
                row["predicted_count"],
                value(row["precision"]),
                value(row["recall"]),
                value(row["f1"]),
            ]
        )
    lines += [
        table(["Class", "True support", "Predicted support", "Precision", "Recall", "F1"], rows),
        "",
        "Confusion matrix: rows are true classes, columns predictions, order `[0,1,2]`.",
        "",
        "```text",
        *[" ".join(f"{count:4d}" for count in row) for row in overall["confusion_matrix"]],
        "```",
        "",
        "## Test stability subgroups",
        "",
        "Flag 0 means stable; flag 1 means static conditions might not have been reached. "
        "The same test predictions are stratified afterward: no subgroup fitting/filtering "
        "or alternative model evaluation. Class denominators and undefined metrics are explicit.",
        "",
    ]
    rows, class_rows = [], []
    for flag in ["0", "1"]:
        metrics = result["test_results"]["stability"][flag]
        rows.append(
            [
                flag,
                metrics["support"],
                "/".join(map(str, metrics["class_support"])),
                *[value(metrics[key]) for key in MAIN_METRICS],
                ", ".join(metrics["undefined_metrics"]) or "none",
            ]
        )
        for row in metrics["per_class"]:
            class_rows.append(
                [
                    flag,
                    row["label"],
                    row["support"],
                    row["predicted_count"],
                    value(row["precision"]),
                    value(row["recall"]),
                    value(row["f1"]),
                ]
            )
        lines += [
            f"### Flag {flag} confusion matrix",
            "",
            "```text",
            *[" ".join(f"{count:4d}" for count in row) for row in metrics["confusion_matrix"]],
            "```",
            "",
        ]
    lines += [
        table(
            [
                "Flag",
                "Cases",
                "Classes 0/1/2",
                "Macro-F1",
                "Balanced accuracy",
                "Class-2 recall",
                "Undefined",
            ],
            rows,
        ),
        "",
        table(
            ["Flag", "Class", "True support", "Predicted support", "Precision", "Recall", "F1"],
            class_rows,
        ),
        "",
        "Zero predicted support makes precision undefined; zero true support makes recall "
        "undefined. F1 is undefined if both supports are zero. A fixed-three-class subgroup "
        "macro-F1/balanced accuracy is undefined if any actual class is absent. This is the "
        "same explicit denominator policy as Phases 2B/3A. Small flagged fault counts do not "
        "support precise reliability claims.",
        "",
        "## Earlier development selection context — not final-test results",
        "",
        table(
            ["Development summary", "Macro-F1", "Balanced accuracy", "Class-2 recall"],
            [
                [
                    "Unweighted three-fold mean (used for selection)",
                    *[
                        value(development["fold_summary"]["overall"]["metric_means"][key])
                        for key in MAIN_METRICS
                    ],
                ],
                [
                    "Pooled development OOF (also inspected in selection)",
                    *[value(development["pooled_oof"]["overall"][key]) for key in MAIN_METRICS],
                ],
            ],
        ),
        "",
        "Those scores came from smaller, purged inner training folds and were used to "
        "select C=100. The final Pipeline was fitted once on all retained development "
        "rows, with a different test class mix and held-out nuisance configurations. "
        "Differences are descriptive; no post-test model reselection was performed.",
        "",
        "## Saved model, metadata and inference",
        "",
        f"Model bundle: `{result['model_directory']}/pipeline.pkl` and `metadata.json`. "
        f"Model SHA-256: `{result['model_sha256']}`.",
        "",
        "The pickle contains the complete fitted scaler and classifier. Metadata contain "
        "the selected parameters, class mapping, confirmed classes_ order, all 70 feature "
        "names, 14-sensor rate/length contract, library versions, all training IDs, "
        "input/source references and the model checksum. No raw training samples or "
        "test-derived fitted preprocessing is included.",
        "",
        "Before/after-loading equality was verified on all 1,328 development rows:",
        "",
        "```json",
        json.dumps(result["serialization_check"], indent=2),
        "```",
        "",
        f"Confirmed probability column order: `{result['classes_']}`. "
        "The single-cycle function uses the existing extractor and requires exactly 14 "
        "physical-sensor 1D finite real vectors at sampling_rate × 60 samples. Missing/extra "
        "sensors, metadata keys, wrong lengths/dimensions and nonnumeric/nonfinite values "
        "are rejected. It returns class, English label, classes and aligned probabilities. "
        "These are raw LR probabilities, not a separately calibrated reliability guarantee.",
        "",
        "```text",
        "uv run --locked --group baseline python scripts/train_final.py --prepare-only",
        "uv run --locked --group baseline python scripts/train_final.py",
        "uv run --locked --group baseline python scripts/evaluate_final.py",
        "uv run --locked --group baseline python scripts/predict_cycle.py",
        "uv run --locked pytest",
        "uv run --locked ruff check .",
        "uv run --locked ruff format --check .",
        "uv lock --check",
        "```",
        "",
        "Training/evaluation attempts have separate immutable receipts. Completed reruns "
        "verify the existing model or cached predictions/results without refitting or "
        "reopening test features. Incomplete attempts stop for investigation instead of "
        "silently retrying. The inference example is restricted to a development ID.",
        "",
        "## Exposure, limitations and review boundary",
        "",
        "All cycles, including this reserved test, were inspected in Phase 2A EDA. This "
        "is the first authorized Phase 3B test feature/prediction evaluation, but the test "
        "is not wholly untouched by prior exploration. Groups are not verified independent "
        "experiments; gap 20 is not a validated physical decorrelation time. The same "
        "controlled rig, thermal trajectory, fixed loads and scarce unstable fault cases "
        "limit external generalization. No industrial reliability or calibrated uncertainty "
        "claim follows from these results.",
        "",
        "History and uv.lock remain unchanged. No further tuning or other test-model "
        "comparison was performed. No API, Docker, commit or push was created. Stop here "
        "for review.",
        "",
        "The bundle is pickle protocol 5: load only trusted, locally generated artifacts. "
        "Checksums detect corruption, not malicious replacement of both model and metadata. "
        "The loader requires matching library versions. See the official "
        "[scikit-learn model-persistence documentation](https://scikit-learn.org/stable/model_persistence.html).",
        "",
    ]
    if "verification" in result:
        checks = result["verification"]
        lines += [
            "## Completed validation",
            "",
            f"{checks['tests']} tests passed; {checks['failures']} failures, "
            f"{checks['errors']} errors, {checks['skipped']} skipped. Ruff lint/format and "
            "uv lock checks passed. Independent saved-prediction metric/probability checks, "
            "training/test membership, persistence equality, development-only inference "
            "example and historical checksums passed.",
            "",
        ]
    return "\n".join(lines)
