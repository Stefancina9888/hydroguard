"""Execute the Phase 2A notebook in a fresh kernel using the active uv Python."""

import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from hydroguard.data import atomic_text_writer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per cell")
    args = parser.parse_args()
    path = ROOT / "notebooks" / "01_eda.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=args.timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    with atomic_text_writer(path) as stream:
        nbformat.write(notebook, stream)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    print(f"PASS: {len(cells)} code cells executed in a fresh {sys.version.split()[0]} kernel")
    print(f"Executed notebook: {path}")


if __name__ == "__main__":
    main()
