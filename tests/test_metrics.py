"""Metric denominators, undefined quantities and pooled-vs-mean behavior."""

import json

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from hydroguard.data import DataValidationError
from hydroguard.metrics import evaluate_predictions, summarize_folds


def test_known_confusion_matrix_and_sklearn_agreement():
    truth, predicted = [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0]
    result = evaluate_predictions(truth, predicted)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert result["macro_f1"] == f1_score(truth, predicted, average="macro") == 0.5
    assert result["balanced_accuracy"] == balanced_accuracy_score(truth, predicted) == 0.5
    assert result["recall_class_2"] == 0.5


def test_most_frequent_predictions_mark_precision_undefined_but_F1_zero():
    result = evaluate_predictions([0, 1, 2], [0, 0, 0])
    assert result["per_class"][1]["precision"] is None
    assert result["per_class"][1]["recall"] == result["per_class"][1]["f1"] == 0
    assert result["macro_f1"] == pytest.approx(1 / 6)
    assert result["balanced_accuracy"] == pytest.approx(1 / 3)


def test_missing_true_class_does_not_silently_become_a_two_class_macro_score():
    result = evaluate_predictions([0, 1], [0, 1])
    assert result["macro_f1"] is result["balanced_accuracy"] is result["recall_class_2"] is None
    assert result["class_support"] == [1, 1, 0]
    assert result["per_class"][2]["f1"] is None
    assert "class_2.recall" in result["undefined_metrics"]
    json.dumps(result, allow_nan=False)


def test_empty_subgroup_and_false_positives_for_absent_class():
    assert evaluate_predictions([], [])["confusion_matrix"] == [[0] * 3 for _ in range(3)]
    result = evaluate_predictions([0, 0], [2, 2])
    assert result["per_class"][2]["f1"] == result["per_class"][2]["precision"] == 0
    assert result["per_class"][2]["recall"] is None


def test_fold_average_never_discards_undefined_subgroup_metrics():
    good = evaluate_predictions([0, 1, 2], [0, 1, 2])
    absent = evaluate_predictions([0, 1], [0, 1])
    result = summarize_folds([good, absent])
    assert result["metric_means"]["macro_f1"] is None
    assert result["defined_fold_counts"]["macro_f1"] == 1


def test_pooled_metrics_are_not_an_unweighted_fold_average():
    a = evaluate_predictions([0, 1, 2], [0, 1, 2])
    b = evaluate_predictions([0] * 4 + [1] * 4 + [2] * 4, [0] * 12)
    pooled = evaluate_predictions([0, 1, 2] + [0] * 4 + [1] * 4 + [2] * 4, [0, 1, 2] + [0] * 12)
    assert pooled["macro_f1"] != summarize_folds([a, b])["metric_means"]["macro_f1"]
    np.testing.assert_array_equal(
        pooled["confusion_matrix"], np.array(a["confusion_matrix"]) + b["confusion_matrix"]
    )


@pytest.mark.parametrize(
    "truth,predicted", [([0], [0, 1]), ([3], [0]), ([0], [0.5]), ([[0]], [[0]])]
)
def test_invalid_metric_inputs(truth, predicted):
    with pytest.raises(DataValidationError):
        evaluate_predictions(truth, predicted)
