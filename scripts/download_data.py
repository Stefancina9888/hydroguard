"""Download and checksum the official UCI 447 raw files."""

import argparse
import json
from pathlib import Path

from hydroguard.data import download_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    result = download_dataset(args.raw_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
