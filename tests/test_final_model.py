"""Manifest, cohort, single-fit and pre-test authorization fences on synthetic fixtures."""

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hydroguard.data import PROFILE_COLUMNS, DataValidationError
from hydroguard.features import FEATURE_NAMES, FeatureBatch
from hydroguard.final_model import (
    HISTORY_PATHS,
    IMPLEMENTATION_PATHS,
    MANIFEST_PATH,
    MODEL_DIRECTORY,
    TEST_CLAIM,
    TEST_JSON,
    TRAINING_CLAIM,
    authorized_candidate,
    evaluate_final,
    load_final_context,
    predict_development_example,
    prepare_final_manifest,
    train_final,
    validate_cohort,
)
from hydroguard.inference import AUTHORIZED_PARAMETERS
from hydroguard.splitting import load_split_lock

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def selected_history():
    candidate = {
        "candidate_id": "lr_09",
        "family": "logistic_regression",
        "parameters": copy.deepcopy(AUTHORIZED_PARAMETERS),
    }
    config = {"candidates": [candidate]}
    results = {
        "status": "completed",
        "selected": {
            "logistic_regression": {
                "candidate_id": "lr_09",
                "parameters": copy.deepcopy(AUTHORIZED_PARAMETERS),
            }
        },
    }
    return config, results


@pytest.mark.parametrize("problem", ["C", "class_weight", "max_iter", "family", "selected_id"])
def test_only_exact_selected_candidate_is_authorized(selected_history, problem):
    config, results = selected_history
    assert authorized_candidate(config, results)["candidate_id"] == "lr_09"
    if problem == "family":
        config["candidates"][0]["family"] = "random_forest"
    elif problem == "selected_id":
        results["selected"]["logistic_regression"]["candidate_id"] = "lr_08"
    else:
        config["candidates"][0]["parameters"]["classifier"][problem] = {
            "C": 10,
            "class_weight": "balanced",
            "max_iter": 1999,
        }[problem]
    with pytest.raises(DataValidationError, match="only unchanged lr_09"):
        authorized_candidate(config, results)


@pytest.fixture
def cohorts():
    ids = np.arange(9) * 100 + 1
    rows = [[3, 100, label, 130, label % 2] for _ in range(3) for label in [0, 1, 2]]
    meta = pd.DataFrame(rows, columns=PROFILE_COLUMNS, index=pd.Index(ids, name="cycle_id"))
    payload = {
        role: {"cycle_ids": ids[start : start + 3].tolist(), "cycles": 3, "class_counts": [1, 1, 1]}
        for role, start in [("training", 0), ("test", 3), ("outer_excluded", 6)]
    }
    return meta, payload


def test_exact_cohort_membership_order_and_metadata_free_X(cohorts):
    meta, payload = cohorts
    for role in ["training", "test"]:
        batch = FeatureBatch(
            np.array(payload[role]["cycle_ids"]),
            pd.DataFrame(np.zeros((3, 70)), columns=FEATURE_NAMES),
        )
        validate_cohort(batch, meta, payload, role)
        with pytest.raises(DataValidationError):
            validate_cohort(FeatureBatch(batch.cycle_ids[::-1], batch.X), meta, payload, role)
        with pytest.raises(DataValidationError, match="fixed 70"):
            validate_cohort(
                FeatureBatch(batch.cycle_ids, batch.X.assign(pump_leakage=0.0)), meta, payload, role
            )


def test_excluded_cycles_and_wrong_classes_are_rejected(cohorts):
    meta, payload = cohorts
    batch = FeatureBatch(
        np.array(payload["outer_excluded"]["cycle_ids"]),
        pd.DataFrame(np.zeros((3, 70)), columns=FEATURE_NAMES),
    )
    with pytest.raises(DataValidationError, match="cannot be fitted or evaluated"):
        validate_cohort(batch, meta, payload, "outer_excluded")
    with pytest.raises(DataValidationError, match="membership/order"):
        validate_cohort(batch, meta, payload, "training")
    batch = FeatureBatch(np.array(payload["test"]["cycle_ids"]), batch.X)
    meta.loc[batch.cycle_ids[-1], "pump_leakage"] = 1
    with pytest.raises(DataValidationError, match="class support"):
        validate_cohort(batch, meta, payload, "test")


@pytest.mark.parametrize("command", [train_final, evaluate_final])
def test_missing_final_manifest_stops_before_any_context_feature_or_model_access(tmp_path, command):
    with patch("hydroguard.final_model.selected_context") as context:
        with patch("hydroguard.final_model.extract_from_directory") as features:
            with patch("hydroguard.final_model.make_candidate_model") as model:
                with pytest.raises(DataValidationError, match="split lock"):
                    command(tmp_path)
    context.assert_not_called()
    features.assert_not_called()
    model.assert_not_called()


def test_evaluation_requires_trained_published_bundle_before_test_access(tmp_path):
    with patch("hydroguard.final_model.load_final_context", return_value=({}, {})):
        with patch(
            "hydroguard.final_model.validate_published_bundle",
            side_effect=DataValidationError("Model missing"),
        ):
            with patch("hydroguard.final_model.extract_from_directory") as features:
                with pytest.raises(DataValidationError, match="Model missing"):
                    evaluate_final(tmp_path)
    features.assert_not_called()
    assert not (tmp_path / TEST_CLAIM).exists()


def test_existing_training_bundle_is_verified_without_another_fit(tmp_path):
    (tmp_path / MODEL_DIRECTORY).mkdir(parents=True)
    saved = {"training": {"cycles": 1328}}
    with patch("hydroguard.final_model.load_final_context", return_value=({}, {})):
        with patch(
            "hydroguard.final_model.validate_published_bundle",
            return_value=SimpleNamespace(metadata=saved),
        ):
            with patch("hydroguard.final_model.check_history_unchanged"):
                with patch("hydroguard.final_model.extract_from_directory") as features:
                    with patch("hydroguard.final_model.make_candidate_model") as model:
                        result = train_final(tmp_path)
    assert result["fits_this_call"] == 0
    model.assert_not_called()
    features.assert_not_called()


def test_incomplete_training_or_test_attempt_is_not_silently_repeated(tmp_path):
    (tmp_path / TRAINING_CLAIM).parent.mkdir(parents=True)
    (tmp_path / TRAINING_CLAIM).write_text("attempt")
    (tmp_path / TEST_CLAIM).write_text("attempt")
    with patch("hydroguard.final_model.load_final_context", return_value=({}, {})):
        with patch("hydroguard.final_model.extract_from_directory") as features:
            with pytest.raises(DataValidationError, match="prior training attempt"):
                train_final(tmp_path)
            with patch("hydroguard.final_model.validate_published_bundle", return_value=object()):
                with pytest.raises(DataValidationError, match="Prior test-access"):
                    evaluate_final(tmp_path)
    features.assert_not_called()


def test_completed_evaluation_verifies_cache_without_test_features_or_prediction(tmp_path):
    (tmp_path / TEST_JSON).parent.mkdir(parents=True)
    (tmp_path / TEST_JSON).write_text("cached")
    with patch("hydroguard.final_model.load_final_context", return_value=({}, {"metadata": None})):
        with patch("hydroguard.final_model.validate_published_bundle", return_value=object()):
            with patch(
                "hydroguard.final_model._cached_test_result", return_value={"status": "completed"}
            ):
                with patch("hydroguard.final_model.check_history_unchanged"):
                    with patch("hydroguard.final_model.extract_from_directory") as features:
                        with patch("hydroguard.final_model.predict_feature_rows") as predict:
                            result, cached = evaluate_final(tmp_path)
    assert cached and result["status"] == "completed"
    features.assert_not_called()
    predict.assert_not_called()


def test_inference_example_rejects_non_development_ID_before_sensor_reads(tmp_path):
    bundle = SimpleNamespace(metadata={"training": {"cycle_ids": [1, 101, 201]}})
    with patch("hydroguard.final_model.load_model_bundle", return_value=bundle):
        with patch("hydroguard.final_model.load_matrix") as loader:
            with pytest.raises(DataValidationError, match="development training cycle"):
                predict_development_example(tmp_path, 999)
    loader.assert_not_called()


@pytest.fixture
def manifest_root(tmp_path):
    for name in (*HISTORY_PATHS, *IMPLEMENTATION_PATHS):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    config = load_split_lock(tmp_path / "config/tuning-v1.json")
    split = load_split_lock(tmp_path / "artifacts/splits/phase2b-v1.json")
    tuning = load_split_lock(tmp_path / "artifacts/experiments/phase3a-v1.json")
    context = {
        "candidate": next(
            c for c in config["payload"]["candidates"] if c["candidate_id"] == "lr_09"
        ),
        "split": split,
        "config": config,
        "tuning_manifest": tuning,
    }
    with patch("hydroguard.final_model.selected_context", return_value=context):
        yield tmp_path


def test_separate_manifest_is_saved_before_fit_and_preserves_history(manifest_root):
    root = manifest_root
    before = {name: (root / name).read_bytes() for name in HISTORY_PATHS}
    with patch("hydroguard.final_model.extract_from_directory") as features:
        with patch("hydroguard.final_model.make_candidate_model") as model:
            first = prepare_final_manifest(root)
            contents = (root / MANIFEST_PATH).read_bytes()
            assert prepare_final_manifest(root) == first
    assert (root / MANIFEST_PATH).read_bytes() == contents
    assert first["payload"]["candidate"]["candidate_id"] == "lr_09"
    assert first["payload"]["training"]["cycles"] == 1328
    assert first["payload"]["test"]["cycles"] == 339
    assert all((root / name).read_bytes() == content for name, content in before.items())
    features.assert_not_called()
    model.assert_not_called()


def test_history_change_after_manifest_prevents_test_or_train(manifest_root):
    root = manifest_root
    prepare_final_manifest(root)
    (root / "docs/tuning-results.md").write_text("changed")
    with patch("hydroguard.final_model.extract_from_directory") as features:
        with pytest.raises(DataValidationError, match="manifest/source/history changed"):
            load_final_context(root)
    features.assert_not_called()
