# Dataset specification and audit policy

## Source and license

The only acquisition source is the [official UCI dataset 447 page](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems).
The direct raw archive is
[the official UCI ZIP](https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip).
No other students' repositories or implementations are used.

Attribution: Helwig, N., Pignanelli, E., & Schütze, A. (2015).
*Condition monitoring of hydraulic systems* [Dataset]. UCI Machine Learning
Repository. [https://doi.org/10.24432/C5CW21](https://doi.org/10.24432/C5CW21).
The official page specifies [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Reuse and adaptation require appropriate credit, a license link and an indication
of changes. Raw files are unmodified. Generated audit metadata add descriptive
column names and one-based row identifiers; reports summarize the source data.

The downloader records the requested and resolved URL, UTC acquisition time, HTTP
Last-Modified/ETag if provided, archive size and SHA-256, and per-file SHA-256.
The source-code digest pin guards against future changes. It was observed during
this project's first acquisition, rather than obtained from an independent signed
publisher checksum. It supports reproducibility, not independent authenticity.
Every audit and cached acquisition re-derives extracted-file digests from the
contents of that pinned, verified ZIP. The local manifest is validated for field
types, source identity, aware timestamps, safe paths and SHA-256 formats, then
compared with those independent archive-content digests. A coordinated change to
an extracted file and `download.json` is detected. The original raw files and
source-code pin are never changed to make a verification pass.

## Expected structure

The test rig repeats 60-second constant-load cycles. Each tab-delimited sensor file
contains one cycle per row and within-cycle measurements in columns. Every sensor
must have exactly 2,205 rows. There are 17 sensor signals:

| Sensors | Physical quantity | Unit | Rate | Columns per cycle |
| --- | --- | --- | ---: | ---: |
| PS1–PS6 | Pressure | bar | 100 Hz | 6,000 |
| EPS1 | Motor power | W | 100 Hz | 6,000 |
| FS1–FS2 | Volume flow | l/min | 10 Hz | 600 |
| TS1–TS4 | Temperature | °C | 1 Hz | 60 |
| VS1 | Vibration | mm/s | 1 Hz | 60 |
| CE | Cooling efficiency (virtual) | % | 1 Hz | 60 |
| CP | Cooling power (virtual) | kW | 1 Hz | 60 |
| SE | Efficiency factor | % | 1 Hz | 60 |

The expected combined width is 43,680 samples per cycle across the 17 files; this
is a structural count, not an engineered feature set. Duration is checked through
the documented rates and matrix widths; timestamps are not present for independent
elapsed-time verification.

`profile.txt` must have exactly 2,205 rows and these five integer-coded columns:

| One-based column | Audit field | Allowed values and meaning |
| ---: | --- | --- |
| 1 | `cooler_condition_pct` | 3 near total failure; 20 reduced efficiency; 100 full efficiency |
| 2 | `valve_condition_pct` | 100 optimal; 90 small lag; 80 severe lag; 73 near total failure |
| 3 | `pump_leakage` | **Target:** 0 no leakage; 1 weak leakage; 2 severe leakage |
| 4 | `accumulator_pressure_bar` | 130 optimal; 115 slightly reduced; 100 severely reduced; 90 near total failure |
| 5 | `stability_flag` | 0 stable; 1 static conditions might not have been reached |

Cycle IDs use the original one-based row number in every file. Alignment relies on
the UCI row-order contract; equal shapes cannot detect an independently permuted
sensor file. The archive and extraction checksums detect changes from acquisition.
The target is extracted only after full profile validation. Cycle IDs and the other
four condition columns are audit metadata; they must not be prediction features.

## Validation and outputs

The audit reads all 17 matrices individually and checks exact shape, numeric
parsing, NaN and infinity. It validates profile shape, finite values and exact
allowed integer codes without rounding. Missing files, empty files, empty fields,
blank/whitespace-only physical rows, invalid tokens and inconsistent row widths
are explicit errors. Blank-row errors include the filename and one-based physical
line number so row-based cycle IDs cannot silently shift. Physical sensor range
thresholds are not supplied by the source, so finite numbers are not asserted to
be physically plausible. No samples are filled, truncated or discarded.

The generated [Markdown report](dataset-audit.md) and [JSON metadata](dataset-audit.json)
include file inventory/sizes/checksums, shapes, validity checks, all profile
distributions, leakage-versus-condition cross-tabulations, and observed ordering
patterns. The JSON records every consecutive target run and every consecutive
component-condition block with source cycle boundaries. A block means identical
first-four-column values in adjacent rows; it is not an independently verified
experiment/session identifier. Stability changes are counted separately.

Complete-cycle duplicates are detected using all 43,680 parsed sensor values,
without including targets, IDs or any other profile column in grouping. Candidates
are refined one sensor at a time, retaining a single matrix in memory. SHA-256
accelerates matching and `array_equal` confirms candidate row equality, including
when hashes collide. Equality is exact in parsed float64 values; positive and
negative zero compare equally. This is a data integrity audit, not a prediction
feature or an approximate similarity search.

Both reports list duplicate cycle-ID groups, their leakage targets and conflicting
labels. If any sensor fails validation, complete-cycle duplicate detection is
reported as not completed. If only the profile fails, sensor groups can still be
reported but their targets/conflicts are unknown. Duplicates and conflicting labels
are audit warnings and do not trigger deletion, relabeling or a model decision.

`data/processed/cycle_metadata.csv` preserves every target and condition value,
without filtering stability flag 1. Normal unit tests use tiny fixtures, mocked
downloads and synthetic ZIP files. Running the inspection command is the full
dataset integration check and is intentionally separate from pytest.
The inspection command invalidates a previous `cycle_metadata.csv` before starting
any audit, including runs that later fail unexpectedly. Reports and successful CSV
outputs use atomic file replacement. Only a successful complete audit republishes
the CSV; failed reports do not claim that all cycles were preserved.

## Interrupted acquisition and recovery

The downloader publishes the verified ZIP, then a staged verified extraction,
then `download.json` using a flushed and synced temporary file with atomic replace.
No existing raw extraction is overwritten. If interrupted before extraction
publication, a rerun can stage a new extraction from the same verified archive.
If interrupted after extraction publication but before manifest publication, a
rerun compares all files with the archive before reconstructing a missing manifest.
If the original acquisition time is unknown, recovery records `recovered_at_utc`
and a null `downloaded_at_utc` rather than inventing a download time. Legacy complete
manifests from the initial Phase 1A run remain valid and are not rewritten.

A malformed existing manifest or mismatched extraction is rejected explicitly;
recovery does not repair changed raw data or silently replace suspect provenance.
Incomplete `.part`, `.tmp` and staging directories from a hard process termination
are ignored and never used as authoritative data. A rerun can safely proceed from
the verified canonical archive; orphan temporary files may require manual cleanup.

## Limitations and future decisions

- The data come from a controlled hydraulic test rig with repeated fixed loads;
  representativeness for industrial pumps and new operating regimes is unknown.
- Multiple component faults are varied together. Cross-tabulations describe
  co-occurrence and cannot establish causation or isolate pump-only effects.
- Consecutive cycles and repeated condition blocks may be dependent. Acquisition
  order is preserved and must not be assumed random. An evaluation split policy
  requires later work; no train/test split is made here.
- The stability indicator is meaningful metadata. Whether future analyses filter,
  stratify or separately evaluate unstable cycles requires an explicit decision.
- Physical plausibility, sensor calibration, between-sensor timing and independent
  experimental-session identities are outside what this structure audit verifies.
- Exact duplicates do not establish why repetition occurred or which conflicting
  target, if any, is correct. Near duplicates are not evaluated in Phase 1A.
- Atomic file replacement covers ordinary interruption and prevents partial
  manifests. It is not a multi-file database transaction or a guarantee against
  hardware failure, filesystem failure or concurrent downloader processes.
- CE and CP are described as virtual sensors; future feature selection must assess
  their suitability. No sensor selection or feature engineering occurs here.

Phase 1A ends after verified acquisition, parsing, audit and tests. No model,
feature table, EDA notebook, API, Dockerfile, remote repository, commit or push is
part of this phase.
