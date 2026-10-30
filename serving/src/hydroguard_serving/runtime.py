"""Load the single server-owned, pinned bundle and reuse the historical inference code."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from threading import Lock

from hydroguard.data import DataValidationError
from hydroguard.inference import load_model_bundle, predict_cycle

from hydroguard_serving.validation import PredictionRequest


def model_contract() -> dict:
    return json.loads(files("hydroguard_serving").joinpath("model-contract.json").read_text())


class Runtime:
    def __init__(self, directory: Path):
        contract = model_contract()
        for name, expected in contract["bundle_sha256"].items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected:
                raise DataValidationError(f"Pinned serving bundle checksum mismatch: {name}")
        for name, expected in contract["historical_source_sha256"].items():
            actual = hashlib.sha256(files("hydroguard").joinpath(name).read_bytes()).hexdigest()
            if actual != expected:
                raise DataValidationError(f"Historical inference source checksum mismatch: {name}")
        self.bundle = load_model_bundle(directory)
        if self.bundle.metadata["versions"] != contract["versions"]:
            raise DataValidationError("Serving environment differs from the selected model")
        self.version = contract["model_version"]
        # threadpoolctl limits are process-wide; serialize inference within this worker.
        self._prediction_lock = Lock()

    def predict(self, raw_request: dict) -> dict:
        """The public Python boundary also validates every raw scalar before NumPy."""
        request = PredictionRequest.model_validate(raw_request)
        with self._prediction_lock:
            prediction = predict_cycle(request.sensors.model_dump(), self.bundle)
        return {**prediction, "model_version": self.version}
