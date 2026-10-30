"""Fixed-class metrics with explicit nulls and denominators for undefined quantities."""

import numpy as np

from hydroguard.data import DataValidationError

LABELS = (0, 1, 2)
MAIN_METRICS = ("macro_f1", "balanced_accuracy", "recall_class_2")


def evaluate_predictions(y_true, y_pred) -> dict:
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if (
        truth.ndim != 1
        or predicted.shape != truth.shape
        or not np.isin(truth, LABELS).all()
        or not np.isin(predicted, LABELS).all()
    ):
        raise DataValidationError("Metrics require aligned vectors with class codes 0, 1, 2")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth.astype(int), predicted.astype(int)), 1)
    support, predicted_support = matrix.sum(axis=1), matrix.sum(axis=0)
    per_class, undefined = [], []
    for label in LABELS:
        tp = int(matrix[label, label])
        actual, guesses = int(support[label]), int(predicted_support[label])
        precision = tp / guesses if guesses else None
        recall = tp / actual if actual else None
        f1 = 2 * tp / (actual + guesses) if actual + guesses else None
        values = {
            "label": label,
            "support": actual,
            "predicted_count": guesses,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        per_class.append(values)
        undefined.extend(
            f"class_{label}.{key}" for key in ("precision", "recall", "f1") if values[key] is None
        )
    # A three-class subgroup macro score is not claimed when an actual class is absent.
    all_classes_present = bool(np.all(support > 0))
    macro = float(np.mean([row["f1"] for row in per_class])) if all_classes_present else None
    balanced = float(np.mean([row["recall"] for row in per_class])) if all_classes_present else None
    result = {
        "support": len(truth),
        "class_support": support.tolist(),
        "macro_f1": macro,
        "balanced_accuracy": balanced,
        "recall_class_2": per_class[2]["recall"],
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
    undefined.extend(key for key in MAIN_METRICS if result[key] is None)
    result["undefined_metrics"] = undefined
    return result


def summarize_folds(evaluations: list[dict]) -> dict:
    """Unweighted fold means/std; never silently discard an undefined fold metric."""
    if not evaluations:
        raise DataValidationError("Cannot summarize zero folds")

    def aggregate(values):
        defined = sum(value is not None for value in values)
        if defined != len(values):
            return None, None, defined
        return float(np.mean(values)), float(np.std(values, ddof=0)), defined

    means, deviations, defined_counts = {}, {}, {}
    for key in MAIN_METRICS:
        means[key], deviations[key], defined_counts[key] = aggregate(
            [row[key] for row in evaluations]
        )
    class_means = []
    for label in LABELS:
        row = {
            "label": label,
            "support_mean": float(np.mean([e["per_class"][label]["support"] for e in evaluations])),
        }
        for key in ("precision", "recall", "f1"):
            row[key], _, row[f"{key}_defined_folds"] = aggregate(
                [e["per_class"][label][key] for e in evaluations]
            )
        class_means.append(row)
    return {
        "folds": len(evaluations),
        "support_sum": sum(e["support"] for e in evaluations),
        "metric_means": means,
        "metric_std_ddof0": deviations,
        "defined_fold_counts": defined_counts,
        "per_class_means": class_means,
    }
