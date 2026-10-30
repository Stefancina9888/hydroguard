"""Baseline presets and fail-closed fences before any real-data fit."""

from pathlib import Path
from unittest.mock import patch

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from hydroguard.baselines import load_baseline_config, make_model, run_development_cv
from hydroguard.data import DataValidationError
from hydroguard.splitting import lock_or_verify

ROOT = Path(__file__).resolve().parents[1]


def test_prespecified_model_types_and_training_only_scaler_pipeline():
    config = load_baseline_config(ROOT / "config/baseline-v1.json")
    dummy = make_model("dummy", config)
    logistic = make_model("logistic_regression", config)
    forest = make_model("random_forest", config)
    assert isinstance(dummy, DummyClassifier) and dummy.strategy == "most_frequent"
    assert isinstance(logistic, Pipeline)
    assert list(logistic.named_steps) == ["scaler", "classifier"]
    assert logistic.named_steps["classifier"].C == 1
    assert logistic.named_steps["classifier"].max_iter == 2000
    assert isinstance(forest, RandomForestClassifier)
    assert forest.random_state == 2026 and forest.n_jobs == 1 and forest.n_estimators == 200


def test_missing_lock_stops_before_loading_features_or_constructing_models(tmp_path):
    with patch("hydroguard.baselines.verified_context") as context:
        with patch("hydroguard.baselines.make_model") as model:
            with pytest.raises(DataValidationError, match="split lock"):
                run_development_cv(tmp_path)
    context.assert_not_called()
    model.assert_not_called()


def test_changed_inputs_or_protocol_stop_before_feature_extraction(tmp_path):
    lock_or_verify(tmp_path / "artifacts/splits/phase2b-v1.json", {"source": "original"})
    with patch("hydroguard.baselines.verified_context", return_value=(None, {}, {})):
        with patch("hydroguard.baselines.build_payload", return_value={"source": "changed"}):
            with patch("hydroguard.baselines.extract_from_directory") as extract:
                with patch("hydroguard.baselines.make_model") as model:
                    with pytest.raises(DataValidationError, match="differ from the existing lock"):
                        run_development_cv(tmp_path)
    extract.assert_not_called()
    model.assert_not_called()
