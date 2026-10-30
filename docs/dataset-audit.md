# HydroGuard dataset audit

Generated (UTC): 2026-10-30T18:44:36.541941+00:00

Structural validation: **PASS**.

Source: [UCI dataset 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems).

All cycles retained in original order.

Pinned archive checksum verified: True. Download manifest bound to archive content: True.

Archive: https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip

Downloaded (UTC): 2026-10-30T18:18:16.876059+00:00

Archive size: 76601704 bytes.

SHA-256: `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`.

Digest is locally observed, not a publisher-supplied authenticity guarantee.

## File inventory

| File | Bytes |
| --- | ---: |
| CE.txt | 911,516 |
| CP.txt | 779,343 |
| description.txt | 4,044 |
| documentation.txt | 4,719 |
| EPS1.txt | 87,411,594 |
| FS1.txt | 7,581,926 |
| FS2.txt | 8,271,923 |
| profile.txt | 31,196 |
| PS1.txt | 91,391,892 |
| PS2.txt | 82,373,611 |
| PS3.txt | 69,916,801 |
| PS4.txt | 46,698,180 |
| PS5.txt | 77,776,429 |
| PS6.txt | 77,875,407 |
| SE.txt | 824,838 |
| TS1.txt | 904,586 |
| TS2.txt | 910,594 |
| TS3.txt | 908,805 |
| TS4.txt | 910,127 |
| VS1.txt | 778,499 |

Per-file SHA-256 digests are recorded in the companion JSON report.

## Sensor validation

Cycles are documented as 60 seconds; expected widths are sampling rate × 60. These files have no timestamp column to verify elapsed time.

| Signal | Hz | Shape | NaN | Infinity | Valid |
| --- | ---: | --- | ---: | ---: | --- |
| PS1 | 100 | [2205, 6000] | 0 | 0 | True |
| PS2 | 100 | [2205, 6000] | 0 | 0 | True |
| PS3 | 100 | [2205, 6000] | 0 | 0 | True |
| PS4 | 100 | [2205, 6000] | 0 | 0 | True |
| PS5 | 100 | [2205, 6000] | 0 | 0 | True |
| PS6 | 100 | [2205, 6000] | 0 | 0 | True |
| EPS1 | 100 | [2205, 6000] | 0 | 0 | True |
| FS1 | 10 | [2205, 600] | 0 | 0 | True |
| FS2 | 10 | [2205, 600] | 0 | 0 | True |
| TS1 | 1 | [2205, 60] | 0 | 0 | True |
| TS2 | 1 | [2205, 60] | 0 | 0 | True |
| TS3 | 1 | [2205, 60] | 0 | 0 | True |
| TS4 | 1 | [2205, 60] | 0 | 0 | True |
| VS1 | 1 | [2205, 60] | 0 | 0 | True |
| CE | 1 | [2205, 60] | 0 | 0 | True |
| CP | 1 | [2205, 60] | 0 | 0 | True |
| SE | 1 | [2205, 60] | 0 | 0 | True |

Numeric parsing, shape, NaN/infinity and profile code domains. Physical sensor range validity is not established by this audit.

Empty fields, ragged rows and nonnumeric tokens are parsing failures; they are never filled or discarded. NaN and infinity counts apply to successfully parsed matrices.

## Profile, target and conditions

Profile matrix: `[2205, 5]`; NaN: 0; infinity: 0.

The third column is the target: 0 = no leakage, 1 = weak leakage, 2 = severe leakage. Cycle IDs are one-based original row numbers. Other condition columns and cycle IDs are audit metadata, never prediction features.

| Profile column | Value | Cycles | Share |
| --- | ---: | ---: | ---: |
| cooler_condition_pct | 3 | 732 | 33.20% |
| cooler_condition_pct | 20 | 732 | 33.20% |
| cooler_condition_pct | 100 | 741 | 33.61% |
| valve_condition_pct | 73 | 360 | 16.33% |
| valve_condition_pct | 80 | 360 | 16.33% |
| valve_condition_pct | 90 | 360 | 16.33% |
| valve_condition_pct | 100 | 1125 | 51.02% |
| pump_leakage | 0 | 1221 | 55.37% |
| pump_leakage | 1 | 492 | 22.31% |
| pump_leakage | 2 | 492 | 22.31% |
| accumulator_pressure_bar | 90 | 808 | 36.64% |
| accumulator_pressure_bar | 100 | 399 | 18.10% |
| accumulator_pressure_bar | 115 | 399 | 18.10% |
| accumulator_pressure_bar | 130 | 599 | 27.17% |
| stability_flag | 0 | 1449 | 65.71% |
| stability_flag | 1 | 756 | 34.29% |

Stability flag 0 means stable; 1 means static conditions might not have been reached.

## Cross-tabulations

Cells are cycle counts; rows are pump leakage classes.

### cooler_condition_pct

| Pump leakage | 3 | 20 | 100 |
| --- | ---: | ---: | ---: |
| 0 | 404 | 404 | 413 |
| 1 | 164 | 164 | 164 |
| 2 | 164 | 164 | 164 |

### valve_condition_pct

| Pump leakage | 73 | 80 | 90 | 100 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 120 | 120 | 120 | 861 |
| 1 | 120 | 120 | 120 | 132 |
| 2 | 120 | 120 | 120 | 132 |

### accumulator_pressure_bar

| Pump leakage | 90 | 100 | 115 | 130 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 562 | 153 | 153 | 353 |
| 1 | 123 | 123 | 123 | 123 |
| 2 | 123 | 123 | 123 | 123 |

### stability_flag

| Pump leakage | 0 | 1 |
| --- | ---: | ---: |
| 0 | 489 | 732 |
| 1 | 480 | 12 |
| 2 | 480 | 12 |

## Consecutive cycles and condition blocks

Adjacent cycles with the same target: 2168 / 2204; target transitions: 36.

| Target | Consecutive runs | Shortest run | Longest run |
| --- | ---: | ---: | ---: |
| 0 | 13 | 50 | 251 |
| 1 | 12 | 41 | 41 |
| 2 | 12 | 41 | 41 |

Observed component-condition blocks: 194; unique configurations: 144. Block lengths (min / median / max): 1 / 10 / 210 cycles. Configurations recurring in separate blocks: 36.

A block is a maximal consecutive run with identical first four profile columns; the stability flag does not define block boundaries. These observed blocks do not establish independent experimental sessions or the acquisition protocol.

| Condition | Adjacent transitions |
| --- | ---: |
| cooler_condition_pct | 2 |
| valve_condition_pct | 144 |
| pump_leakage | 36 |
| accumulator_pressure_bar | 11 |
| stability_flag | 71 |

The companion JSON contains every target run and component block with original cycle boundaries. Observed sequential persistence means experimental order must not be assumed random. A later evaluation design must consider temporal dependence, repeated configurations and component confounding.

## Identical complete sensor cycles

Exact numeric equality of all within-cycle values in all 17 sensor matrices, after float64 parsing; target, cycle ID and other profile columns are excluded. SHA-256 narrows comparisons; candidate rows are also compared with array_equal.

Duplicate groups: 0; cycles in groups: 0; redundant cycles beyond one per group: 0. Conflicting-target groups: 0; cycles in conflicting groups: 0.

| Cycle IDs | Leakage targets in matching order | Target conflict |
| --- | --- | --- |

Report only; no cycles are automatically removed or relabeled.

Conflicting targets are data quality findings, not structural parse failures. They remain unresolved in Phase 1A; the audit does not choose a correct label.

## Limitations

One controlled hydraulic test rig and a fixed load cycle cannot establish generalization to industrial equipment. Several component faults are superimposed; condition labels can confound leakage diagnosis. Physical sensor plausibility and independent experiment identifiers are not established. This phase performs no EDA, feature engineering, split selection, model training or deployment.

## Audit warnings

None.

## Failed checks

None.
