"""Evaluate only the fixed saved lr_09 Pipeline on the reserved 339-cycle test."""

from pathlib import Path

from hydroguard.data import atomic_text_writer
from hydroguard.final_model import evaluate_final
from hydroguard.final_report import render_final_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result, cached = evaluate_final(ROOT)
    with atomic_text_writer(ROOT / "docs/final-test-results.md") as stream:
        stream.write(render_final_report(result))
    metrics = result["test_results"]["overall"]
    print(
        f"{'Cached results verified' if cached else 'Reserved test evaluated'}; "
        f"cases: {metrics['support']}"
    )
    print(
        f"macro-F1={metrics['macro_f1']:.6f}; "
        f"balanced accuracy={metrics['balanced_accuracy']:.6f}; "
        f"class-2 recall={metrics['recall_class_2']:.6f}"
    )
    print(f"Report: {ROOT / 'docs/final-test-results.md'}")


if __name__ == "__main__":
    main()
