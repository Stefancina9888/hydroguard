"""Execute review notebooks in fresh kernels and save separate outputs, preserving sources."""

import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from hydroguard.data import atomic_text_writer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/executed-notebooks"
    )
    args = parser.parse_args()
    for name in ["01_eda.ipynb", "02_modeling.ipynb"]:
        source, output = ROOT / "notebooks" / name, args.output_dir.resolve() / name
        if output == source.resolve():
            parser.error("Executed outputs must not replace canonical notebooks")
        notebook = nbformat.read(source, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
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
            timeout=300,
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        with atomic_text_writer(output) as stream:
            nbformat.write(notebook, stream)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in cells)
        print(f"PASS: {name}: {len(cells)} executed cells; separate output: {output}")


if __name__ == "__main__":
    main()
