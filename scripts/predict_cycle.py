"""Run a saved-model single-cycle inference example from development only."""

import argparse
import json
from pathlib import Path

from hydroguard.data import write_json_atomic
from hydroguard.final_model import predict_development_example

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycle-id",
        type=int,
        help="Original development cycle ID; defaults to the first training ID",
    )
    args = parser.parse_args()
    result = predict_development_example(ROOT, args.cycle_id)
    write_json_atomic(ROOT / "data/processed/inference-example.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
