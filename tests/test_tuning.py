"""Offline tests for preregistration, fixed memberships, selection and holdout fences."""

import copy
import json
import shutil
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from hydroguard.baselines import load_baseline_config
from hydroguard.data import PROFILE_COLUMNS, DataValidationError
from hydroguard.features import FEATURE_NAMES, FeatureBatch
from hydroguard.splitting import canonical_sha256, check_partition, load_split_lock
from hydroguard.tuning import (
    CONFIG_PATH,
    IMPLEMENTATION_PATHS,
    MANIFEST_PATH,
    PROTECTED_PATHS,
    candidate_configuration,
    fit_candidate_fold,
    guard_development_request,
    load_experiment,
    make_candidate_model,
    prepare_experiment,
    run_tuning,
    select_candidate,
    simplicity_key,
    validate_member_roles,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def baseline():
    return load_baseline_config(ROOT / "config/baseline-v1.json")


def test_exact_candidate_budget_grid_and_seeded_RF_sample(baseline):
    original = copy.deepcopy(baseline)
    config = candidate_configuration(baseline)
    assert config == candidate_configuration(baseline)
    assert baseline == original
    lr = [c for c in config["candidates"] if c["family"] == "logistic_regression"]
    rf = [c for c in config["candidates"] if c["family"] == "random_forest"]
    assert len(lr) == 10 and len(rf) == 13
    assert config["expected_classifier_fits"] == 69
    assert {
        (c["parameters"]["classifier"]["C"], c["parameters"]["classifier"]["class_weight"])
        for c in lr
    } == {(C, weight) for C in [0.01, 0.1, 1, 10, 100] for weight in [None, "balanced"]}
    sampled = [c for c in rf if not c["baseline_reference"]]
    assert len({canonical_sha256(c["parameters"]) for c in sampled}) == 12
    assert all(c["parameters"] != baseline["models"]["random_forest"] for c in sampled)
    assert config["rf_sampling"]["sampled_pool_indices_0_based"] == [
        15,
        40,
        64,
        65,
        82,
        13,
        28,
        76,
        79,
        71,
        53,
        73,
    ]
    assert sum(c["baseline_reference"] for c in config["candidates"]) == 2


def test_only_requested_parameters_change_and_models_are_fresh(baseline):
    for candidate in candidate_configuration(baseline)["candidates"]:
        model = make_candidate_model(candidate, baseline)
        assert model is not make_candidate_model(candidate, baseline)
        if candidate["family"] == "logistic_regression":
            fixed = {
                key: value
                for key, value in candidate["parameters"]["classifier"].items()
                if key not in ("C", "class_weight")
            }
            assert fixed == {
                key: value
                for key, value in baseline["models"]["logistic_regression"]["classifier"].items()
                if key not in ("C", "class_weight")
            }
            assert list(model.named_steps) == ["scaler", "classifier"]
            assert (
                candidate["parameters"]["scaler"]
                == baseline["models"]["logistic_regression"]["scaler"]
            )
        else:
            assert model.n_estimators == 200 and model.n_jobs == 1 and model.random_state == 2026


def ranked(candidate, macro=0.8, recall=0.7, eligible=True):
    return {
        **copy.deepcopy(candidate),
        "eligible": eligible,
        "mean_macro_f1": macro,
        "mean_recall_class_2": recall,
    }


def test_selection_uses_unweighted_mean_not_pooled_score(baseline):
    candidates = candidate_configuration(baseline)["candidates"][:2]
    a, b = ranked(candidates[0], macro=0.8), ranked(candidates[1], macro=0.81)
    a["pooled_macro_f1"], b["pooled_macro_f1"] = 0.99, 0.70
    assert select_candidate([a, b], "logistic_regression")["candidate_id"] == b["candidate_id"]


def test_selection_secondary_recall_and_exact_not_rounded_ties(baseline):
    candidates = candidate_configuration(baseline)["candidates"][:2]
    a, b = ranked(candidates[0], recall=0.7), ranked(candidates[1], recall=0.8)
    assert select_candidate([a, b], "logistic_regression")["candidate_id"] == b["candidate_id"]
    a["mean_macro_f1"] = np.nextafter(b["mean_macro_f1"], 1.0)
    assert select_candidate([a, b], "logistic_regression")["candidate_id"] == a["candidate_id"]


def test_simplicity_and_candidate_ID_complete_tie_break(baseline):
    config = candidate_configuration(baseline)["candidates"]
    assert (
        select_candidate([ranked(config[3]), ranked(config[0])], "logistic_regression")[
            "candidate_id"
        ]
        == "lr_01"
    )
    rf = [c for c in config if c["family"] == "random_forest"]
    simpler = min(rf, key=simplicity_key)
    assert (
        select_candidate([ranked(c) for c in rf], "random_forest")["candidate_id"]
        == simpler["candidate_id"]
    )
    a = ranked(config[0])
    b = {**copy.deepcopy(a), "candidate_id": "z_duplicate_score"}
    assert select_candidate([b, a], "logistic_regression")["candidate_id"] == a["candidate_id"]
    a["eligible"] = False
    assert select_candidate([a, b], "logistic_regression")["candidate_id"] == b["candidate_id"]
    with pytest.raises(DataValidationError, match="No eligible"):
        select_candidate([a], "logistic_regression")


@pytest.fixture
def toy_roles():
    groups = [(3, 100, 130), (20, 100, 130), (100, 100, 130), (3, 90, 130)]
    rows = [
        [cooler, valve, target, acc, target % 2]
        for cooler, valve, acc in groups
        for target in [0, 1, 2]
    ]
    ids = np.arange(12) * 100 + 1
    meta = pd.DataFrame(rows, columns=PROFILE_COLUMNS, index=pd.Index(ids, name="cycle_id"))
    dev, test = ids[:9].tolist(), ids[9:].tolist()
    inner = []
    for index in range(3):
        val = dev[index * 3 : (index + 1) * 3]
        train = [cycle for cycle in dev if cycle not in val]
        inner.append(
            {
                "fold": index + 1,
                "train_cycle_ids": train,
                "validation_cycle_ids": val,
                "excluded_cycle_ids": [],
                "diagnostics": check_partition(meta, train, val),
            }
        )
    members = {
        "outer": {
            "development_cycle_ids": dev,
            "test_cycle_ids": test,
            "excluded_cycle_ids": [],
            "diagnostics": check_partition(meta, dev, test),
        },
        "inner": inner,
    }
    expected_outer = {
        "development": (9, [3, 3, 3]),
        "test": (3, [1, 1, 1]),
        "excluded": (0, [0, 0, 0]),
    }
    expected_inner = [
        {"train": (6, [2, 2, 2]), "validation": (3, [1, 1, 1]), "excluded": (0, [0, 0, 0])}
        for _ in range(3)
    ]
    return meta, members, expected_outer, expected_inner


def test_saved_membership_validation_never_generates_CV_and_never_mutates(toy_roles):
    meta, members, outer, inner = toy_roles
    before = copy.deepcopy(members)
    with patch(
        "hydroguard.splitting.make_memberships", side_effect=AssertionError("New split forbidden")
    ):
        with patch(
            "sklearn.model_selection.StratifiedGroupKFold.split",
            side_effect=AssertionError("New CV forbidden"),
        ):
            validate_member_roles(meta, members, outer, inner)
    assert members == before


@pytest.mark.parametrize("problem", ["counts", "test_in_train", "duplicate_validation", "gap"])
def test_saved_roles_reject_membership_and_gap_corruption(toy_roles, problem):
    meta, members, outer, inner = toy_roles
    if problem == "counts":
        members["inner"][0]["train_cycle_ids"].pop()
    elif problem == "test_in_train":
        members["inner"][0]["train_cycle_ids"][-1] = members["outer"]["test_cycle_ids"][0]
    elif problem == "duplicate_validation":
        members["inner"][1] = copy.deepcopy(members["inner"][0])
        members["inner"][1]["fold"] = 2
    else:
        meta = meta.rename(index={901: 821})
        members["outer"]["test_cycle_ids"][0] = 821
    with pytest.raises(DataValidationError):
        validate_member_roles(meta, members, outer, inner)


@pytest.fixture
def real_member_batch():
    payload = load_split_lock(ROOT / "artifacts/splits/phase2b-v1.json")["payload"]
    ids = np.array(payload["memberships"]["outer"]["development_cycle_ids"])
    batch = FeatureBatch(ids, pd.DataFrame(np.zeros((len(ids), 70)), columns=FEATURE_NAMES))
    return payload, batch


@pytest.mark.parametrize("role", ["test", "excluded"])
def test_test_and_outer_excluded_cannot_receive_predictions(real_member_batch, role):
    payload, batch = real_member_batch
    fold = payload["memberships"]["inner"][0]
    with pytest.raises(DataValidationError, match="saved inner"):
        guard_development_request(
            payload, batch, fold, payload["memberships"]["outer"][f"{role}_cycle_ids"]
        )
    # Even a mistakenly forged fold cannot authorize non-development predictions.
    forged = copy.deepcopy(payload)
    forged["memberships"]["inner"][0]["validation_cycle_ids"] = payload["memberships"]["outer"][
        f"{role}_cycle_ids"
    ]
    with pytest.raises(DataValidationError, match="cannot receive"):
        guard_development_request(
            forged,
            batch,
            forged["memberships"]["inner"][0],
            forged["memberships"]["inner"][0]["validation_cycle_ids"],
        )


def test_development_features_cannot_add_metadata_or_include_test_rows(real_member_batch):
    payload, batch = real_member_batch
    fold = payload["memberships"]["inner"][0]
    contaminated = FeatureBatch(batch.cycle_ids, batch.X.assign(cycle_id=batch.cycle_ids))
    with pytest.raises(DataValidationError, match="fixed 70"):
        guard_development_request(payload, contaminated, fold, fold["validation_cycle_ids"])
    contaminated = FeatureBatch(
        np.array(sorted(payload["memberships"]["outer"]["test_cycle_ids"])), batch.X
    )
    with pytest.raises(DataValidationError, match="development cycles only"):
        guard_development_request(payload, contaminated, fold, fold["validation_cycle_ids"])


@pytest.fixture
def registration_root(tmp_path, baseline):
    for name in (*PROTECTED_PATHS, *IMPLEMENTATION_PATHS):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    wrapper = load_split_lock(tmp_path / "artifacts/splits/phase2b-v1.json")
    with patch(
        "hydroguard.tuning.read_frozen_context", return_value=(wrapper, None, baseline, None)
    ):
        yield tmp_path


def test_preregistration_is_idempotent_and_preserves_original_split(registration_root):
    root = registration_root
    split_before = (root / "artifacts/splits/phase2b-v1.json").read_bytes()
    with patch(
        "hydroguard.tuning.make_model",
        side_effect=AssertionError("Fit preparation cannot create classifiers"),
    ):
        config, manifest = prepare_experiment(root)
        first = {(root / path): (root / path).read_bytes() for path in [CONFIG_PATH, MANIFEST_PATH]}
        again = prepare_experiment(root)
    assert (config, manifest) == again
    assert len(config["payload"]["candidates"]) == 23
    assert (
        manifest["payload"]["split_reference"]["payload_sha256"]
        == load_split_lock(root / "artifacts/splits/phase2b-v1.json")["payload_sha256"]
    )
    assert (root / "artifacts/splits/phase2b-v1.json").read_bytes() == split_before
    assert all(path.read_bytes() == content for path, content in first.items())


def test_changed_candidate_rule_is_not_automatically_overwritten(registration_root):
    root = registration_root
    prepare_experiment(root)
    config = load_split_lock(root / CONFIG_PATH)
    config["payload"]["selection_rule"]["primary"] = "changed"
    config["payload_sha256"] = canonical_sha256(config["payload"])
    (root / CONFIG_PATH).write_text(json.dumps(config))
    before = (root / CONFIG_PATH).read_bytes()
    with pytest.raises(DataValidationError, match="overwrite is forbidden"):
        prepare_experiment(root)
    assert (root / CONFIG_PATH).read_bytes() == before


def test_changed_baseline_report_aborts_before_extraction_or_fit(registration_root):
    root = registration_root
    prepare_experiment(root)
    (root / "docs/baseline-results.md").write_text("changed")
    with patch("hydroguard.tuning.extract_from_directory") as extractor:
        with patch("hydroguard.tuning.make_model") as model:
            with pytest.raises(DataValidationError, match="manifest/reference"):
                load_experiment(root)
    extractor.assert_not_called()
    model.assert_not_called()


def test_missing_preregistration_stops_before_dataset_loading_or_fit(tmp_path):
    with patch("hydroguard.tuning.read_frozen_context") as context:
        with patch("hydroguard.tuning.extract_from_directory") as extractor:
            with pytest.raises(DataValidationError):
                run_tuning(tmp_path)
    context.assert_not_called()
    extractor.assert_not_called()


def test_model_warning_and_failure_are_retained_without_parameter_changes(baseline):
    candidate = next(
        c
        for c in candidate_configuration(baseline)["candidates"]
        if c["candidate_id"] == "rf_baseline"
    )
    original = copy.deepcopy(candidate)

    class WarningModel:
        def fit(self, X, y):
            warnings.warn("Synthetic non-convergence", ConvergenceWarning, stacklevel=2)

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = pd.DataFrame(np.ones((3, 70)), columns=FEATURE_NAMES)
    with patch("hydroguard.tuning.make_candidate_model", return_value=WarningModel()):
        record, predicted = fit_candidate_fold(candidate, baseline, X, np.array([0, 1, 2]), X)
    assert record["fit_completed"] and record["convergence_warning"]
    assert record["warnings"][0]["category"] == "ConvergenceWarning"
    assert len(predicted) == 3

    class BrokenModel:
        def fit(self, X, y):
            raise ValueError("Synthetic model failure")

    with patch("hydroguard.tuning.make_candidate_model", return_value=BrokenModel()):
        record, predicted = fit_candidate_fold(candidate, baseline, X, np.array([0, 1, 2]), X)
    assert not record["fit_completed"] and predicted is None
    assert record["error"] == {"type": "ValueError", "message": "Synthetic model failure"}
    assert record["fit_seconds"] >= 0
    assert candidate == original
