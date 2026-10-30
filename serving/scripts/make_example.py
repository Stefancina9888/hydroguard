"""Export one locked development cycle; run with the unchanged root environment."""

import argparse
import json
from pathlib import Path

from hydroguard.baselines import check_inputs_unchanged
from hydroguard.data import load_matrix, sha256_file, validate_sensor, write_json_atomic
from hydroguard.features import PHYSICAL_SENSORS
from hydroguard.inference import load_model_bundle, predict_cycle
from hydroguard.splitting import load_split_lock

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", type=int, default=1)
    args = parser.parse_args()
    lock = load_split_lock(ROOT / "artifacts/splits/phase2b-v1.json")
    development = lock["payload"]["memberships"]["outer"]["development_cycle_ids"]
    if args.cycle_id not in development:
        parser.error("Example cycle must belong to the locked development set")
    bundle = load_model_bundle(ROOT / "artifacts/models/phase3b-v1")
    if args.cycle_id not in bundle.metadata["training"]["cycle_ids"]:
        parser.error("Example cycle is absent from the model's development training membership")
    check_inputs_unchanged(ROOT, bundle.metadata["input_checksums"])
    sensors = {}
    for sensor in PHYSICAL_SENSORS:
        matrix = load_matrix(ROOT / "data/raw/uci447" / f"{sensor}.txt")
        validate_sensor(matrix, sensor)
        sensors[sensor] = matrix[args.cycle_id - 1].tolist()
        del matrix
    path = ROOT / "serving/examples/development-cycle.json"
    write_json_atomic(path, {"sensors": sensors})
    provenance = {
        "cycle_id": args.cycle_id,
        "partition": "development",
        "split_payload_sha256": lock["payload_sha256"],
        "model_sha256": bundle.metadata["model_sha256"],
        "request_sha256": sha256_file(path),
        "direct_historical_prediction": predict_cycle(sensors, bundle),
        "source": "UCI dataset 447, Helwig et al. (2015), CC BY 4.0",
        "source_url": "https://doi.org/10.24432/C5CW21",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "modifications": "One development cycle exported as JSON; no sample values changed",
    }
    write_json_atomic(ROOT / "serving/examples/provenance.json", provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
