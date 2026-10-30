# Phase 1A code review follow-up

Executed on 2026-10-30. Scope remains repository bootstrap, acquisition, structural
validation and dataset integrity audit. No EDA, prediction features, data split,
training, API, container or deployment was added. No commit, push or remote
repository was created.

## Implemented changes

- Complete-cycle duplicate detection compares all 17 sensor matrices independently
  of every profile column. Exact numeric equality is confirmed after hashing;
  cycle-ID groups and target conflicts are reported without deletion or relabeling.
- The parser rejects empty files and every blank or whitespace-only physical row,
  including leading, interior and trailing rows. Errors identify the filename and
  one-based line number. Failed reports no longer certify preservation/completeness
  of all cycles. Incomplete sensor validation prevents complete-cycle duplicate
  claims; invalid profiles leave duplicate targets/conflicts unknown.
- Inspection removes the previous `cycle_metadata.csv` before reading inputs and
  publishes a replacement atomically only after a successful audit and reports.
  Failed audits and unexpected/report-writing exceptions leave no stale CSV at the
  standard metadata path.
- Per-file checksums are freshly derived from the contents of the independently
  pinned ZIP. Both local manifest and extracted files are checked against this
  source, detecting coordinated file-plus-manifest modifications. The existing
  archive pin and target mapping were preserved.
- Download publication uses a staged extraction and atomic, flushed, synced JSON
  manifest writes. Reruns can recover a missing extraction or missing manifest
  from verified content. An unknown original acquisition timestamp remains null
  and recovery gets its own timestamp. Malformed existing manifests are rejected
  rather than silently repaired. The downloader validates source identity, field
  types, timestamps, relative paths, digests and duplicate JSON keys.
- Offline regression tests cover all findings, including every sensor in duplicate
  matching, hash collisions, nonconsecutive duplicate groups, conflicting labels,
  blank rows, successful-then-failed inspection, stale metadata after exceptions,
  file-plus-manifest tampering, interrupted HTTP transfer, directory publication,
  partial manifest writing and interrupted atomic manifest replacement.

## Executed validation

Environment: Windows, Python 3.11.15, NumPy 2.4.6, pytest 9.2.0, Ruff 0.17.3,
uv 0.13.4. The lockfile and Python selection were not changed.

| Check | Result |
| --- | --- |
| `uv run --locked pytest` | 142 passed; 14.11 seconds in the original workspace |
| `uv run --locked ruff check .` | Passed |
| `uv run --locked ruff format --check .` | Passed; 9 Python files formatted |
| `uv lock --check` | Passed |
| `uv run --locked python scripts/download_data.py` | Cached pinned archive, local manifest and all extracted files verified; existing manifest bytes unchanged |
| `uv run --locked python scripts/inspect_data.py` | Passed on original data; zero errors and zero warnings |
| Independent duplicate cross-check | SHA-256 over every complete concatenated numeric cycle, without candidate pruning or targets: 2,205 unique cycles, zero duplicate groups |
| Git index | `.python-version` and `uv.lock` tracked/staged; no commit exists |
| Git ignore checks | Raw data, derived CSV and `.venv` remain excluded |

All seven commands in the README were also executed in a separate temporary copy
with no virtual environment and no raw dataset at the start. This included a
fresh download from UCI, full parsing/audit and all 142 tests (13.69 seconds).
Every command exited with status 0. The temporary project and its separate data
were cleaned up afterward. The actual command outputs and fresh download time
are retained in [the clean-environment execution record](phase-1a-clean-check.json).

The first development test run had three failures caused by prior expectations:
an empty-file NumPy warning, old error capitalization, and a synthetic provenance
fixture that bypassed a real ZIP/pin. The fixtures now use verified synthetic ZIPs
and assert the stricter behavior. Ruff formatting issues were also corrected.
No failing check remains.

## Audit statistics and changes

The regenerated [Markdown audit](dataset-audit.md) and [JSON audit](dataset-audit.json)
come from the original, unchanged raw dataset. The JSON report schema is now
version 2, adding duplicate results, warnings and archive-content binding status.

| Measurement | Before review fixes | After review fixes |
| --- | --- | --- |
| Validated cycles | 2,205 | 2,205 |
| Validated sensor signals | 17 | 17 |
| Sensor shapes | 7 × (2205, 6000), 2 × (2205, 600), 8 × (2205, 60) | Unchanged |
| Profile shape | (2205, 5) | Unchanged |
| Missing/non-finite values and parsing errors | 0 | 0 |
| Leakage classes 0 / 1 / 2 | 1,221 / 492 / 492 | Unchanged |
| Stability flags 0 / 1 | 1,449 / 756 | Unchanged; no filtering |
| Consecutive condition blocks / distinct configurations | 194 / 144 | Unchanged |
| Adjacent equal-target pairs / total adjacent pairs | 2,168 / 2,204 | Unchanged |
| Leakage transitions | 36 | Unchanged |
| Identical complete sensor-cycle groups | Not previously measured | 0 |
| Cycles in duplicate groups / redundant cycles | Not previously measured | 0 / 0 |
| Conflicting-target duplicate groups / cycles | Not previously measured | 0 / 0 |
| Archive-content binding of local manifest | Not previously verified independently | Verified |
| Offline test cases | 60 | 142 |

Other component distributions and all cross-tabulations are unchanged. The original
archive remains 76,601,704 bytes; its 20 extracted files total 556,266,030 bytes.
The unchanged source-code/archive SHA-256 is:

```text
24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2
```

## Remaining limitations

- Duplicate matching is exact after float64 parsing; no near-duplicate, tolerance
  or physical similarity analysis is performed. Textual numeric representations
  that parse identically count as equal; signed zeros compare equally.
- Atomic replace protects individual artifacts from partial writes, but acquisition
  is not a multi-file transaction. Power loss, filesystem/hardware failures and
  concurrent downloader processes are not covered by these guarantees. Hard-kill
  orphan temporary files may require manual cleanup and are never trusted.
- Missing provenance can be reconstructed only after archive and extraction
  verification. A malformed existing canonical manifest is rejected and requires
  investigation or a separate acquisition directory. Original download time may
  be unknown after recovery.
- The pin is locally observed rather than independently authenticated by the
  publisher. Changing trusted source code or the pin is outside the integrity
  checks' threat model; neither was done here.
- Sensor plausibility, calibration, timing, independent experimental identities
  and generalization beyond the controlled rig remain unverified. Original order
  and all stability-flag values are retained; no data split is selected.

## Files changed for this review

- `src/hydroguard/data.py`: parsing, duplicate audit, trusted acquisition verification,
  recovery and atomic publication helpers.
- `scripts/inspect_data.py`: safe inspection/export lifecycle and duplicate summary.
- `tests/test_data.py`: adapted and expanded offline regression suite.
- `README.md`, `data/README.md`, `docs/dataset.md`: behavior and recovery documentation.
- `docs/dataset-audit.md`, `docs/dataset-audit.json`: regenerated measured reports.
- `docs/phase-1a-clean-check.json`: new clean-environment execution evidence.
- `docs/phase-1a-review.md`: this review summary.

`uv.lock` and `.python-version` were added to the Git index with unchanged contents;
they are staged for later review, not committed. Other project files remain in the
same uncommitted repository state. Raw data and the existing download manifest
were not modified.
