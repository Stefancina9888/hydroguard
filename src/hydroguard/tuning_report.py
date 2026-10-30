"""Render actual Phase 3A selection results without modifying baseline reports."""

import json

from hydroguard.metrics import MAIN_METRICS


def render_tuning_report(report: dict) -> str:
    def number(value):
        return "undefined" if value is None else f"{value:.6f}"

    def table(headers, rows):
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(map(str, row)) + " |" for row in rows],
            ]
        )

    config = report["configuration"]
    lines = [
        "# Phase 3A — bounded development tuning",
        "",
        "**These development results were used for candidate selection. They are not "
        "an unbiased generalization estimate or final-test results.**",
        "",
        f"Status: {report['status']}; completed (UTC): {report['completed_at_utc']}.",
        "",
        f"Experiment preregistered (UTC): {report['manifest_created_at_utc']}; "
        f"candidate configuration created (UTC): {report['config_created_at_utc']}.",
        "",
        f"Manifest payload SHA-256: `{report['manifest_payload_sha256']}`.",
        "",
        "The experiment references the existing split payload "
        f"`{report['experiment_manifest']['split_reference']['payload_sha256']}`. "
        "Neither the split nor the baseline configuration/results were rewritten. "
        "No new CV split was generated, no selected model was refitted on full "
        "development data, and no final-test or outer-excluded features/predictions/metrics "
        "were computed.",
        "",
        "## Reproduce and audit",
        "",
        "```text",
        "uv run --locked --group baseline python scripts/run_tuning.py --prepare-only",
        "uv run --locked --group baseline python scripts/run_tuning.py",
        "uv run --locked pytest",
        "uv run --locked ruff check .",
        "uv run --locked ruff format --check .",
        "uv lock --check",
        "```",
        "",
        "Preparation publishes/verifies `config/tuning-v1.json` and "
        "`artifacts/experiments/phase3a-v1.json` without replacing an existing different lock. "
        "The training command requires both preregistrations. It reuses saved memberships "
        "in `artifacts/splits/phase2b-v1.json` and never calls a splitter or baseline "
        "membership generator. Source/config/input byte fingerprints are verified. "
        "No dependency or baseline files are updated.",
        "",
        "The complete [JSON results](tuning-results.json) retain every candidate's full "
        "parameters, fold metrics, warnings, errors and timings. Local/Git-ignored "
        "`data/processed/tuning-runs/<run_id>/` retains this run's progress/results and "
        "development-only OOF predictions; `data/processed/tuning-oof.csv` is the latest "
        "OOF table for all candidates. A rerun gets a new journal rather than erasing "
        "the earlier local execution history.",
        "",
        "## Prespecified candidates and selection rule",
        "",
        "LR uses the unchanged baseline StandardScaler Pipeline and constructor "
        "parameters except the Cartesian product C=[0.01, 0.1, 1, 10, 100] and "
        "class_weight=[null, balanced]. Its C=1/unweighted candidate is also the "
        "baseline reference. No full-dataset scaler or EDA-fitted value is used. "
        "Balanced weights are learned by the estimator from training labels only.",
        "",
        "RF keeps baseline parameters including 200 trees, one worker and seed 2026. "
        "The specified four-dimensional space has 96 configurations. Its exact baseline "
        "is excluded from a 95-item sampling pool and added separately; "
        "Python `random.Random(2026).sample` selects 12 distinct pool positions without "
        "replacement. Pool enumeration follows max_depth, min_samples_leaf, max_features, "
        "class_weight in the declared list order. Concrete sampled positions: "
        f"`{config['rf_sampling']['sampled_pool_indices_0_based']}`. "
        "Thus RF has 13 evaluated configurations including the extra baseline reference, "
        "and LR has 10. Reference candidates remain eligible for selection.",
        "",
        "The rule below was frozen before the first tuning fit. A lower simplicity "
        "key is preferred after both scores tie exactly. Simplicity is an explicit "
        "operational tie-break, not a proved ordering of predictive or compute complexity.",
        "",
        "```json",
        json.dumps(config["selection_rule"], indent=2),
        "```",
        "",
    ]
    rows = []
    for candidate in config["candidates"]:
        params = candidate["parameters"]
        classifier = params.get("classifier", params)
        rows.append(
            [
                candidate["candidate_id"],
                candidate["family"],
                classifier.get("C", "—"),
                str(classifier["class_weight"]),
                str(classifier.get("max_depth", "—")),
                classifier.get("min_samples_leaf", "—"),
                classifier.get("max_features", "—"),
                candidate["baseline_reference"],
            ]
        )
    lines += [
        table(
            [
                "ID",
                "Family",
                "C",
                "Class weight",
                "Max depth",
                "Min leaf",
                "Max features",
                "Baseline reference",
            ],
            rows,
        ),
        "",
        "The remaining parameters are copied from the unchanged baseline config and "
        "the entire resolved parameter dictionaries are in the locked tuning config. "
        "All candidates use the same 70-feature extractor and the same three original-ID "
        "gap-20 inner folds (607/442, 576/457, 607/429 training/validation cycles). "
        "All count, class, group and gap checks run before each fit. Numerical thread "
        "count is one. No estimator is retained or fitted after selection.",
        "",
        "## Candidate ranking and execution",
        "",
        f"Classifier fits attempted/completed: "
        f"{report['fit_attempts']}/{report['fit_completions']} "
        f"(planned 69). Recorded warnings: {report['warning_count']}. "
        "Run wall time, including verification/extraction: "
        f"{report['elapsed_seconds']:.3f} seconds. "
        "Fit/predict times use a monotonic clock; candidate elapsed times also include "
        "preflight and journal overhead. Classifier fit count excludes internal trees "
        "and training-only scaler fits.",
        "",
    ]
    rows = []
    for candidate in report["candidates"]:
        rows.append(
            [
                candidate["candidate_id"],
                candidate["status"],
                number(candidate["mean_macro_f1"]),
                number(candidate["mean_recall_class_2"]),
                sum(len(fold["warnings"]) for fold in candidate["folds"]),
                f"{candidate['elapsed_seconds']:.3f}",
                candidate["simplicity_key"],
            ]
        )
    lines += [
        table(
            [
                "Candidate",
                "Eligibility",
                "Mean fold macro-F1",
                "Mean fold class-2 recall",
                "Warnings",
                "Elapsed seconds",
                "Simplicity key",
            ],
            rows,
        ),
        "",
        "## Every candidate and fold",
        "",
    ]
    rows = []
    for candidate in report["candidates"]:
        for fold in candidate["folds"]:
            metrics = fold.get("metrics", {}).get("overall", {})
            rows.append(
                [
                    candidate["candidate_id"],
                    fold["fold"],
                    *[number(metrics.get(key)) for key in MAIN_METRICS],
                    f"{fold['fit_seconds']:.4f}",
                    f"{fold['predict_seconds']:.4f}",
                    "; ".join(f"{w['category']}: {w['message']}" for w in fold["warnings"])
                    or "none",
                    str(fold["error"]) if fold["error"] else "none",
                ]
            )
    lines += [
        table(
            [
                "Candidate",
                "Fold",
                "Macro-F1",
                "Balanced accuracy",
                "Class-2 recall",
                "Fit seconds",
                "Predict seconds",
                "Warnings",
                "Error",
            ],
            rows,
        ),
        "",
        "A ConvergenceWarning or failed fit makes a candidate ineligible by the "
        "preregistered rule; any structural/integrity violation aborts the run. "
        "No search-space expansion or warning-driven parameter adjustment is allowed.",
        "",
        "## Selected family configurations and baseline comparison",
        "",
        "Selection uses unweighted fold means, not pooled OOF macro-F1. Pooled "
        "metrics below are recomputed from cached development predictions without "
        "additional fits. All 1,328 development cycles have one prediction per candidate. "
        "Baseline reference candidates reproduced the saved baseline metrics and, when "
        "available, exact saved baseline OOF labels.",
        "",
    ]
    rows = []
    for family, selected in report["selected"].items():
        comparison = report["baseline_comparison"][family]
        lines += [
            f"### {family}: {selected['candidate_id']}",
            "",
            "```json",
            json.dumps(selected["parameters"], indent=2),
            "```",
            "",
        ]
        for label, means, pooled in [
            (
                "Baseline",
                comparison["baseline_fold_mean"],
                comparison["baseline_pooled_oof"]["overall"],
            ),
            (
                "Selected",
                selected["fold_summary"]["overall"]["metric_means"],
                selected["pooled_oof"]["overall"],
            ),
        ]:
            rows.append(
                [
                    family,
                    label,
                    number(means["macro_f1"]),
                    number(means["recall_class_2"]),
                    number(pooled["macro_f1"]),
                    number(pooled["balanced_accuracy"]),
                    number(pooled["recall_class_2"]),
                ]
            )
        lines += [
            "Change from baseline mean fold macro-F1: "
            f"{comparison['mean_macro_f1_difference']:+.6f}; "
            f"change in pooled OOF macro-F1: {comparison['pooled_macro_f1_difference']:+.6f}.",
            "",
        ]
    lines += [
        table(
            [
                "Family",
                "Configuration",
                "Mean fold macro-F1",
                "Mean fold class-2 recall",
                "OOF macro-F1",
                "OOF balanced accuracy",
                "OOF class-2 recall",
            ],
            rows,
        ),
        "",
        "## Selected OOF confusion matrices and class metrics",
        "",
        "Confusion rows are true classes and columns predictions, fixed order [0,1,2]. "
        "OOF support is 846/246/236. Fold spread is descriptive, not an IID confidence interval.",
        "",
    ]
    class_rows = []
    for family, selected in report["selected"].items():
        overall = selected["pooled_oof"]["overall"]
        lines += [
            f"### {family}",
            "",
            "```text",
            *[" ".join(f"{value:4d}" for value in row) for row in overall["confusion_matrix"]],
            "```",
            "",
        ]
        for row in overall["per_class"]:
            class_rows.append(
                [
                    family,
                    row["label"],
                    row["support"],
                    number(row["precision"]),
                    number(row["recall"]),
                    number(row["f1"]),
                ]
            )
    lines += [
        table(["Family", "Class", "Support", "Precision", "Recall", "F1"], class_rows),
        "",
        "## Selected stability subgroups",
        "",
        "The same selected OOF predictions are stratified by flag, without subgroup "
        "fitting or cycle filtering. Flag 0 means stable; flag 1 means static conditions "
        "might not have been reached. Undefined precision/recall/F1 and fixed-three-class "
        "subgroup macro scores follow the unchanged Phase 2B metric policy. Fold 1 has "
        "no flagged class-2 validation cases; affected fold metrics and their means are "
        "undefined, not silently omitted. JSON retains all fold subgroup class metrics "
        "and confusion matrices.",
        "",
    ]
    rows, class_rows = [], []
    for family, selected in report["selected"].items():
        candidate = next(
            row for row in report["candidates"] if row["candidate_id"] == selected["candidate_id"]
        )
        for flag in ["0", "1"]:
            for label, metrics in [
                (f"fold {fold['fold']}", fold["metrics"]["stability"][flag])
                for fold in candidate["folds"]
            ] + [("pooled OOF", selected["pooled_oof"]["stability"][flag])]:
                rows.append(
                    [
                        family,
                        flag,
                        label,
                        metrics["support"],
                        "/".join(map(str, metrics["class_support"])),
                        *[number(metrics[key]) for key in MAIN_METRICS],
                        ", ".join(metrics["undefined_metrics"]) or "none",
                    ]
                )
            for row in selected["pooled_oof"]["stability"][flag]["per_class"]:
                class_rows.append(
                    [
                        family,
                        flag,
                        row["label"],
                        row["support"],
                        number(row["precision"]),
                        number(row["recall"]),
                        number(row["f1"]),
                    ]
                )
            means = selected["fold_summary"]["stability"][flag]
            rows.append(
                [
                    family,
                    flag,
                    "fold mean",
                    means["support_sum"],
                    "see folds",
                    *[number(means["metric_means"][key]) for key in MAIN_METRICS],
                    "undefined if any fold is undefined",
                ]
            )
    lines += [
        table(
            [
                "Family",
                "Flag",
                "Evaluation",
                "Support (mean: sum)",
                "Classes 0/1/2",
                "Macro-F1",
                "Balanced accuracy",
                "Class-2 recall",
                "Undefined",
            ],
            rows,
        ),
        "",
        table(["Family", "Flag", "Class", "Support", "Precision", "Recall", "F1"], class_rows),
        "",
        "Baseline pooled subgroup metrics are also retained in the JSON baseline comparison; "
        "the original [baseline report](baseline-results.md) remains unchanged. Flagged OOF "
        "support is only 617/6/5 for classes 0/1/2; rare-stratum scores are not precise "
        "reliability estimates.",
        "",
        "## Limitations and review boundary",
        "",
        "The same development folds were used to select configurations, so their selected "
        "scores are selection-affected and may be optimistic. They are not nested-CV or "
        "independent external validation estimates. No significance or deployment claim "
        "follows from a small development-score difference. The reserved final test was "
        "previously exposed in Phase 2A EDA, but remains unused for Phase 3A features, "
        "predictions and metrics. Group separation and gap 20 do not establish independent "
        "experiments or a physical decorrelation time. Shared rig/thermal trajectory and "
        "purging-induced class mixtures remain limitations.",
        "",
        "All raw inputs, original IDs/order, baseline configuration/results, split lock, "
        "uv.lock and the 70-feature extractor were verified unchanged. No full-development "
        "winner refit, new feature, XGBoost, API, Dockerfile, commit or push is part of this "
        "phase. Stop here for review.",
        "",
    ]
    if "verification" in report:
        checked = report["verification"]
        lines += [
            "## Executed checks",
            "",
            f"{checked['tests']} offline tests passed; {checked['failures']} failures, "
            f"{checked['errors']} errors and {checked['skipped']} skipped. Ruff lint/format "
            "and uv lock checks passed. Candidate selection, every fold metric, selected "
            "OOF/stability results, 69-fit accounting, no test/excluded OOF IDs and "
            "protected-file checksums were independently verified.",
            "",
        ]
    return "\n".join(lines)
