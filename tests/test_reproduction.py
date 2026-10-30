"""Fresh provenance may change; immutable content, cohort and output isolation may not."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hydroguard.data import DataValidationError
from hydroguard.features import FEATURE_NAMES, FeatureBatch
from hydroguard.reproduction import (
    reproduce_training,
    verify_content_identity,
    verify_review_history,
)
from hydroguard.splitting import load_split_lock

ROOT = Path(__file__).resolve().parents[1]


def test_only_new_acquisition_record_may_change():
    old = load_split_lock(ROOT / "artifacts/splits/phase2b-v1.json")["payload"]["input_checksums"]
    fresh = copy.deepcopy(old)
    fresh["download_manifest_sha256"] = "f" * 64
    verify_content_identity(old, fresh)
    for key in old.keys() - {"download_manifest_sha256"}:
        changed = copy.deepcopy(fresh)
        changed[key] = "changed"
        with pytest.raises(DataValidationError, match="content/environment"):
            verify_content_identity(old, changed)
    fresh["unrecognized"] = True
    with pytest.raises(DataValidationError):
        verify_content_identity(old, fresh)


@pytest.mark.parametrize(
    "destination", ["artifacts/models/new", "docs/new", "data/processed/reproductions"]
)
def test_unsafe_output_rejected_before_loading_or_fitting(tmp_path, destination):
    with patch("hydroguard.reproduction.reproduction_context") as context:
        with pytest.raises(DataValidationError, match="new directory"):
            reproduce_training(tmp_path, tmp_path / destination)
    context.assert_not_called()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "data/processed/reproductions/existing"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text("original")
    with patch("hydroguard.reproduction.reproduction_context") as context:
        with pytest.raises(DataValidationError):
            reproduce_training(tmp_path, output)
    context.assert_not_called()
    assert (output / "manifest.json").read_text() == "original"


def test_review_history_detects_any_modified_file(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "input.txt").write_text("changed")
    (tmp_path / "config/review-integrity-v1.json").write_text(
        json.dumps({"files": {"input.txt": "0" * 64}})
    )
    with pytest.raises(DataValidationError, match="Historical review inputs"):
        verify_review_history(tmp_path)


def test_reproduction_fits_exact_development_rows_and_preregisters_first(tmp_path):
    from hydroguard.data import sha256_file

    split = load_split_lock(ROOT / "artifacts/splits/phase2b-v1.json")
    ids = split["payload"]["memberships"]["outer"]["development_cycle_ids"]
    official = json.loads((ROOT / "artifacts/models/phase3b-v1/metadata.json").read_text())
    config = load_split_lock(ROOT / "config/tuning-v1.json")
    candidate = next(c for c in config["payload"]["candidates"] if c["candidate_id"] == "lr_09")
    for name in ["artifacts/splits/phase2b-v1.json", "config/review-integrity-v1.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "download.json").write_text('{"fresh": true}')
    fingerprints = {"download_manifest_sha256": sha256_file(raw / "download.json")}
    meta = pd.DataFrame({"pump_leakage": np.repeat([0, 1, 2], [846, 246, 236])}, index=ids)
    batch = FeatureBatch(np.asarray(ids), pd.DataFrame(np.zeros((1328, 70)), columns=FEATURE_NAMES))
    model = MagicMock()
    model.named_steps = {"scaler": SimpleNamespace(n_samples_seen_=1328)}
    model.classes_ = np.asarray([0, 1, 2])
    context = {
        "split": split,
        "official": official,
        "fingerprints": fingerprints,
        "metadata": meta,
        "baseline": {},
        "candidate": candidate,
    }
    output = tmp_path / "data/processed/reproductions/fixture"

    def on_fit(X, y):
        assert (output / "manifest.json").is_file()
        assert X is batch.X and len(y) == 1328
        assert list(X.columns) == list(FEATURE_NAMES)

    model.fit.side_effect = on_fit
    with patch("hydroguard.reproduction.reproduction_context", return_value=context):
        with patch("hydroguard.reproduction.extract_from_directory", return_value=batch) as extract:
            with patch("hydroguard.reproduction.validate_cohort") as cohort:
                with patch("hydroguard.reproduction.check_inputs_unchanged"):
                    with patch("hydroguard.reproduction.verify_review_history"):
                        with patch(
                            "hydroguard.reproduction.make_candidate_model", return_value=model
                        ):
                            with patch(
                                "hydroguard.reproduction.save_pipeline_bundle",
                                return_value={
                                    "model_sha256": "a" * 64,
                                    "serialization_check": {"rows": 1328},
                                },
                            ) as save:
                                result = reproduce_training(tmp_path, output)
    assert extract.call_args.args[1] == ids
    assert model.fit.call_count == 1
    model.predict.assert_not_called()
    assert cohort.call_args.args[3] == "training"
    assert save.call_args.args[1] == output / "model"
    assert result["test_predictions"] is result["test_metrics"] is False
    assert (output / "download.json").read_bytes() == (raw / "download.json").read_bytes()
