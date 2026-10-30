"""Export Git-eligible review files through a temporary index without changing the real index."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() or destination.is_relative_to(ROOT):
        parser.error("Destination must be a new directory outside the working repository")
    with tempfile.TemporaryDirectory(prefix="review-index-") as temporary:
        environment = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        subprocess.run(
            ["git", "-c", "core.autocrlf=true", "add", "--all"],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        destination.mkdir(parents=True, exist_ok=False)
        subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=true",
                "checkout-index",
                "--all",
                "--prefix=" + destination.as_posix() + "/",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    for forbidden in [".venv", "serving/.venv", "data/raw", "data/processed", "tmp", ".git"]:
        if (destination / forbidden).exists():
            raise RuntimeError(f"Forbidden clean-copy path: {forbidden}")
    print(f"Exported clean Git-eligible files: {destination}; real Git index unchanged")


if __name__ == "__main__":
    main()
