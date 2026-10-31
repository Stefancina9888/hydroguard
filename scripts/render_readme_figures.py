"""Export existing EDA figures and visualize saved test counts; never fit or predict."""

import base64
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs/images"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    history = json.loads((ROOT / "config/review-integrity-v1.json").read_text("utf-8"))
    notebook_path = "notebooks/01_eda.ipynb"
    results_path = "docs/final-test-results.json"
    for source in [notebook_path, results_path]:
        if digest(ROOT / source) != history["files"][source]:
            raise RuntimeError(f"Historical figure source changed: {source}")
    notebook = json.loads((ROOT / notebook_path).read_text("utf-8"))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    images = {}
    for cell, name in [(5, "eda-target-distribution.png"), (10, "eda-sensor-waveforms.png")]:
        outputs = [
            output["data"]["image/png"]
            for output in notebook["cells"][cell]["outputs"]
            if "image/png" in output.get("data", {})
        ]
        data = outputs[0]
        encoded = data if isinstance(data, str) else "".join(data)
        (DESTINATION / name).write_bytes(base64.b64decode(encoded))
        images[name] = {
            "source": notebook_path,
            "source_sha256": digest(ROOT / notebook_path),
            "zero_based_cell_index": cell,
            "png_output_index_in_cell": 0,
            "operation": "Byte-for-byte export of an existing executed notebook PNG",
            "image_sha256": digest(DESTINATION / name),
        }

    saved = json.loads((ROOT / results_path).read_text("utf-8"))
    overall = saved["test_results"]["overall"]
    matrix = overall["confusion_matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert [sum(row) for row in matrix] == overall["class_support"]
    assert sum(map(sum, matrix)) == overall["support"] == 339
    labels = ["0: no leakage", "1: weak leakage", "2: severe leakage"]
    with plt.rc_context({"font.size": 11, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(8, 6.3))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(map(max, matrix)))
        ax.set_xticks(range(3), labels, rotation=12, ha="right")
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted class", labelpad=12)
        ax.set_ylabel("True class", labelpad=12)
        ax.set_title("Reserved-test predictions: 339 cycles", pad=20, weight="bold")
        for row in range(3):
            for column in range(3):
                count = matrix[row][column]
                percentage = 100 * count / sum(matrix[row])
                ax.text(
                    column,
                    row,
                    f"{count}\n{percentage:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if count > 65 else "#183448",
                    fontsize=14,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.05, label="Cycles")
        fig.text(
            0.5,
            0.025,
            "Saved fixed-model results. Percentages are within each true class.\n"
            "Weak-leakage recall: 18/103. No new model evaluation was performed.",
            ha="center",
            color="#46515c",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.085, 1, 1))
        output = DESTINATION / "final-test-confusion-matrix.png"
        fig.savefig(
            output,
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "HydroGuard README figure renderer"},
        )
        plt.close(fig)
    images[output.name] = {
        "source": results_path,
        "source_sha256": digest(ROOT / results_path),
        "json_field": "test_results.overall.confusion_matrix",
        "confusion_matrix": matrix,
        "operation": "Render existing counts and within-row percentages; no predictions",
        "image_sha256": digest(output),
    }
    record = {
        "purpose": "README illustrations from existing executed evidence",
        "renderer": "scripts/render_readme_figures.py",
        "matplotlib_version": matplotlib.__version__,
        "new_training": False,
        "new_test_predictions": False,
        "historical_inputs_unchanged": True,
        "images": images,
    }
    (DESTINATION / "readme-figures.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("Exported two existing EDA figures and rendered saved test confusion counts")


if __name__ == "__main__":
    main()
