"""Preregister Phase 3B or train its single final development-only Pipeline."""

import argparse
from pathlib import Path

from hydroguard.final_model import (
    MANIFEST_PATH,
    MODEL_DIRECTORY,
    prepare_final_manifest,
    train_final,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save/verify Phase 3B manifest; no fit or test features",
    )
    args = parser.parse_args()
    if args.prepare_only:
        manifest = prepare_final_manifest(ROOT)
        print(f"Manifest: {ROOT / MANIFEST_PATH}; payload SHA-256: {manifest['payload_sha256']}")
        print("No model fitted; no test features or predictions accessed.")
        return
    result = train_final(ROOT)
    print(f"{result['status']}; Pipeline fits this call: {result['fits_this_call']}")
    print(f"Saved complete Pipeline and metadata: {ROOT / MODEL_DIRECTORY}")
    print(f"Model SHA-256: {result['metadata']['model_sha256']}")
    print(f"Development serialization check: {result['metadata']['serialization_check']}")


if __name__ == "__main__":
    main()
