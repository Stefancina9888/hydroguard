"""Run only the frozen development CV and publish actual baseline reports."""

from pathlib import Path

from hydroguard.baselines import render_baseline_report, run_development_cv
from hydroguard.data import atomic_text_writer, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report, oof = run_development_cv(ROOT)
    with atomic_text_writer(ROOT / "data/processed/baseline-oof.csv") as stream:
        oof.to_csv(stream, index=False)
    write_json_atomic(ROOT / "docs/baseline-results.json", report)
    with atomic_text_writer(ROOT / "docs/baseline-results.md") as stream:
        stream.write(render_baseline_report(report))
    print(
        "PASS: 9 development fits; 1328 OOF rows/model; no final-test features/predictions/metrics"
    )
    print(f"Report: {ROOT / 'docs/baseline-results.md'}")


if __name__ == "__main__":
    main()
