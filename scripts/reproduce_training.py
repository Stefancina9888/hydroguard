"""Train lr_09 on saved development IDs using fresh verified acquisition provenance."""

import argparse
import json
from pathlib import Path

from hydroguard.reproduction import reproduce_training

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reproduce_training(ROOT, args.output), indent=2))


if __name__ == "__main__":
    main()
