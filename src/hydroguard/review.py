"""Read-only notebook tables from saved experiments and standardized LR coefficients."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydroguard.inference import load_model_bundle
from hydroguard.reproduction import verify_review_history


def saved_review(root: Path) -> dict:
    verify_review_history(root)
    return {
        name: json.loads((root / f"docs/{name}-results.json").read_text(encoding="utf-8"))
        for name in ["baseline", "tuning", "final-test"]
    }


def baseline_table(saved: dict) -> pd.DataFrame:
    rows = []
    for name, model in saved["baseline"]["models"].items():
        rows.append(
            {
                "model": name,
                "role": "development CV baseline",
                **model["fold_summary"]["overall"]["metric_means"],
            }
        )
    return pd.DataFrame(rows).set_index("model")


def candidate_table(saved: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": c["candidate_id"],
                "family": c["family"],
                "parameters": c["parameters"],
                "eligible": c["eligible"],
                "reference": c["baseline_reference"],
                "mean_macro_f1": c["mean_macro_f1"],
                "mean_recall_class_2": c["mean_recall_class_2"],
            }
            for c in saved["tuning"]["candidates"]
        ]
    ).set_index("candidate")


def standardized_coefficients(root: Path) -> pd.DataFrame:
    verify_review_history(root)
    bundle = load_model_bundle(root / "artifacts/models/phase3b-v1")
    classifier = bundle.pipeline.named_steps["classifier"]
    rows = []
    for label, coefficients in zip(classifier.classes_, classifier.coef_, strict=True):
        for name, coefficient in zip(bundle.metadata["feature_names"], coefficients, strict=True):
            rows.append(
                {
                    "class": int(label),
                    "label": bundle.metadata["class_mapping"][str(label)],
                    "feature": name,
                    "coefficient": float(coefficient),
                    "absolute_coefficient": float(np.abs(coefficient)),
                }
            )
    return pd.DataFrame(rows)
