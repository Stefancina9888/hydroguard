"""Validate all raw matrices and write measured audit artifacts."""

import argparse
from pathlib import Path

from hydroguard.data import inspect_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    report = inspect_dataset(args.raw_dir, args.report_dir, args.metadata_dir)
    print(f"Validation: {'PASS' if report['validation_passed'] else 'FAIL'}")
    print(
        f"Sensor signals: {len(report['sensors'])}; "
        f"metadata published: {report['validation_passed']}"
    )
    duplicates = report["duplicate_sensor_cycles"]
    print(
        f"Complete sensor duplicate audit: {duplicates['status']}; "
        f"groups: {duplicates.get('group_count', 'unknown')}; "
        f"target conflicts: {duplicates.get('conflicting_target_group_count', 'unknown')}"
    )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Reports: {args.report_dir / 'dataset-audit.md'} and dataset-audit.json")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    raise SystemExit(0 if report["validation_passed"] else 1)


if __name__ == "__main__":
    main()
