"""Preregister or execute Phase 3A tuning; reuse the immutable Phase 2B folds."""

import argparse
from pathlib import Path

from hydroguard.data import atomic_text_writer, write_json_atomic
from hydroguard.tuning import CONFIG_PATH, MANIFEST_PATH, prepare_experiment, run_tuning
from hydroguard.tuning_report import render_tuning_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Publish/verify preregistration; no fits"
    )
    args = parser.parse_args()
    if args.prepare_only:
        config, manifest = prepare_experiment(ROOT)
        print(f"Locked {len(config['payload']['candidates'])} candidates: {ROOT / CONFIG_PATH}")
        print(f"Experiment manifest: {ROOT / MANIFEST_PATH}")
        print(f"Manifest payload SHA-256: {manifest['payload_sha256']}")
        print("No classifiers fitted. Planned fits: 69.")
        return
    report, oof = run_tuning(ROOT)
    run_dir = ROOT / "data/processed/tuning-runs" / report["run_id"]
    with atomic_text_writer(run_dir / "oof.csv") as stream:
        oof.to_csv(stream, index=False)
    with atomic_text_writer(ROOT / "data/processed/tuning-oof.csv") as stream:
        oof.to_csv(stream, index=False)
    write_json_atomic(ROOT / "docs/tuning-results.json", report)
    with atomic_text_writer(ROOT / "docs/tuning-results.md") as stream:
        stream.write(render_tuning_report(report))
    print(
        f"PASS: {report['fit_completions']} classifier fits; "
        "no full-development refit or test scoring"
    )
    for family, selected in report["selected"].items():
        print(
            f"Selected {family}: {selected['candidate_id']}; "
            f"mean macro-F1={selected['mean_macro_f1']:.6f}"
        )


if __name__ == "__main__":
    main()
