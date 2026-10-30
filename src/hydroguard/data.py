"""Acquire, validate and audit UCI 447 without extracting prediction features."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SOURCE_URL = "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems"
ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/447/"
    "condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
)
# Pinned after first acquisition; UCI does not publish a reference digest.
ARCHIVE_SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
EXPECTED_CYCLES = 2205
CYCLE_SECONDS = 60
SENSOR_RATES = {
    **dict.fromkeys(["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"], 100),
    **dict.fromkeys(["FS1", "FS2"], 10),
    **dict.fromkeys(["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"], 1),
}
PROFILE_COLUMNS = (
    "cooler_condition_pct",
    "valve_condition_pct",
    "pump_leakage",
    "accumulator_pressure_bar",
    "stability_flag",
)
PROFILE_DOMAINS = ({3, 20, 100}, {73, 80, 90, 100}, {0, 1, 2}, {90, 100, 115, 130}, {0, 1})
TARGET_LABELS = {0: "no leakage", 1: "weak leakage", 2: "severe leakage"}


class DataValidationError(ValueError):
    """The dataset does not satisfy the documented UCI structure."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or str(path) != name
        or name == "."
    )


def checked_zip_members(bundle: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Validate every member before extracting or deriving trusted content hashes."""
    names = set()
    members = bundle.infolist()
    for member in members:
        name = member.filename.rstrip("/") if member.is_dir() else member.filename
        if (
            not safe_relative_name(name)
            or stat.S_ISLNK(member.external_attr >> 16)
            or name.casefold() in names
        ):
            raise DataValidationError(f"Unsafe or duplicate ZIP entry: {member.filename}")
        names.add(name.casefold())
    if sum(member.file_size for member in members) > 2_000_000_000:
        raise DataValidationError("Archive exceeds the 2 GB extraction limit")
    return members


def safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        checked_zip_members(bundle)
        bundle.extractall(destination)


def archive_content_hashes(archive: Path, expected_sha256: str) -> dict[str, str]:
    """Derive file hashes from ZIP content only after verifying its independent pin."""
    try:
        if sha256_file(archive) != expected_sha256:
            raise DataValidationError("Archive checksum mismatch with the source-code pin")
        with zipfile.ZipFile(archive) as bundle:
            hashes = {}
            for member in checked_zip_members(bundle):
                if member.is_dir():
                    continue
                digest = hashlib.sha256()
                with bundle.open(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[member.filename] = digest.hexdigest()
        missing = {f"{sensor}.txt" for sensor in SENSOR_RATES} | {"profile.txt"}
        missing -= hashes.keys()
        if missing:
            raise DataValidationError(f"Archive is missing required files: {sorted(missing)}")
        return hashes
    except (OSError, zipfile.BadZipFile) as error:
        raise DataValidationError(f"Cannot verify archive: {error}") from error


def validate_download_manifest(value: Any) -> dict:
    """Validate the complete acquisition schema, including legacy Phase 1A manifests."""
    if not isinstance(value, dict):
        raise DataValidationError("download.json: expected an object")
    for key, expected in (
        ("dataset_id", 447),
        ("source_url", SOURCE_URL),
        ("archive_url", ARCHIVE_URL),
    ):
        if value.get(key) != expected or (key == "dataset_id" and type(value[key]) is not int):
            raise DataValidationError(f"download.json: invalid {key}")
    if "schema_version" in value and (
        type(value["schema_version"]) is not int or value["schema_version"] != 1
    ):
        raise DataValidationError("download.json: unsupported schema_version")
    if type(value.get("archive_bytes")) is not int or value["archive_bytes"] <= 0:
        raise DataValidationError("download.json: invalid archive_bytes")
    if not isinstance(value.get("archive_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["archive_sha256"]
    ):
        raise DataValidationError("download.json: invalid archive_sha256")
    for key in ("resolved_url", "http_last_modified", "http_etag"):
        if key not in value or (value[key] is not None and not isinstance(value[key], str)):
            raise DataValidationError(f"download.json: invalid {key}")
    if not isinstance(value.get("checksum_basis"), str) or not value["checksum_basis"]:
        raise DataValidationError("download.json: invalid checksum_basis")
    for key in ("downloaded_at_utc", "recovered_at_utc"):
        if key == "recovered_at_utc" and key not in value:
            continue
        timestamp = value.get(key)
        if key == "downloaded_at_utc" and timestamp is None and value.get("recovered_at_utc"):
            continue
        try:
            parsed = datetime.fromisoformat(timestamp)
            if parsed.utcoffset() is None:
                raise ValueError("timezone required")
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"download.json: invalid {key}") from error
    hashes = value.get("extracted_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise DataValidationError("download.json: invalid extracted_sha256 mapping")
    names = set()
    for name, digest in hashes.items():
        if (
            not isinstance(name, str)
            or not safe_relative_name(name)
            or name.casefold() in names
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise DataValidationError("download.json: invalid extracted file name or digest")
        names.add(name.casefold())
    return value


def read_download_manifest(path: Path) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DataValidationError(f"download.json: duplicate key {key}")
            result[key] = value
        return result

    try:
        return validate_download_manifest(
            json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        )
    except (OSError, ValueError) as error:
        raise DataValidationError(f"Cannot read download provenance: {error}") from error


@contextmanager
def atomic_text_writer(path: Path):
    """Flush a same-directory temporary file, then atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".tmp",
        prefix=f".{path.name}-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: dict) -> None:
    with atomic_text_writer(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_manifest_content(provenance: dict, archive: Path, hashes: dict, digest: str) -> None:
    if (
        provenance["archive_sha256"] != digest
        or provenance["archive_bytes"] != archive.stat().st_size
    ):
        raise DataValidationError("Download manifest archive checksum or size mismatch")
    if provenance["extracted_sha256"] != hashes:
        raise DataValidationError("Download manifest hashes differ from verified archive content")


def extraction_inventory(dataset_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_symlink():
            raise DataValidationError(f"Unexpected link in extraction: {path.name}")
        if path.is_file():
            hashes[path.relative_to(dataset_dir).as_posix()] = sha256_file(path)
    return hashes


def verify_extraction(dataset_dir: Path, expected: dict[str, str]) -> None:
    actual = extraction_inventory(dataset_dir)
    if actual.keys() != expected.keys():
        raise DataValidationError("Extracted file inventory mismatch with verified archive")
    for name, digest in expected.items():
        if actual[name] != digest:
            raise DataValidationError(
                f"Extracted file checksum mismatch with verified archive: {name}"
            )


def download_dataset(raw_dir: Path, *, expected_sha256: str = ARCHIVE_SHA256) -> dict:
    """Resume from verified publication boundaries without overwriting existing raw files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "uci447.zip"
    provenance_path = raw_dir / "download.json"
    dataset_dir = raw_dir / "uci447"
    provenance = read_download_manifest(provenance_path) if provenance_path.exists() else None
    response_info = {"resolved_url": None, "http_last_modified": None, "http_etag": None}
    downloaded_at = None
    if not archive.exists():
        with tempfile.NamedTemporaryFile(
            dir=raw_dir, prefix=".uci447-", suffix=".part", delete=False
        ) as stream:
            part = Path(stream.name)
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "HydroGuard/0.1"})
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                part.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
                output.flush()
                os.fsync(output.fileno())
                response_info = {
                    "resolved_url": response.geturl(),
                    "http_last_modified": response.headers.get("Last-Modified"),
                    "http_etag": response.headers.get("ETag"),
                }
            if sha256_file(part) != expected_sha256:
                raise DataValidationError(
                    "Downloaded archive checksum mismatch with source-code pin"
                )
            part.rename(archive)
            downloaded_at = utc_now()
        finally:
            part.unlink(missing_ok=True)
    hashes = archive_content_hashes(archive, expected_sha256)
    if provenance is not None:
        verify_manifest_content(provenance, archive, hashes, expected_sha256)
    if dataset_dir.exists():
        verify_extraction(dataset_dir, hashes)
    else:
        with tempfile.TemporaryDirectory(prefix="extract-", dir=raw_dir) as staging:
            staging_path = Path(staging)
            safe_extract(archive, staging_path)
            verify_extraction(staging_path, hashes)
            staging_path.rename(dataset_dir)
    if provenance is not None:
        return provenance
    provenance = {
        "schema_version": 1,
        "dataset_id": 447,
        "source_url": SOURCE_URL,
        "archive_url": ARCHIVE_URL,
        **response_info,
        "downloaded_at_utc": downloaded_at,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": expected_sha256,
        "checksum_basis": "Locally observed SHA-256; not a publisher-supplied digest",
        "extracted_sha256": hashes,
    }
    if downloaded_at is None:
        provenance["recovered_at_utc"] = utc_now()
    validate_download_manifest(provenance)
    write_json_atomic(provenance_path, provenance)
    return provenance


def load_matrix(path: Path) -> np.ndarray:
    """Read strict tab-delimited numeric data; malformed or missing fields fail visibly."""

    def checked_rows(stream):
        line_number = 0
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise DataValidationError(
                    f"{path.name}: blank or whitespace-only row at line {line_number}"
                )
            yield line
        if line_number == 0:
            raise DataValidationError(f"{path.name}: empty file (expected row at line 1)")

    try:
        with path.open(encoding="utf-8") as stream:
            matrix = np.loadtxt(
                checked_rows(stream), delimiter="\t", dtype=np.float64, ndmin=2, comments=None
            )
    except (OSError, ValueError) as error:
        raise DataValidationError(f"Cannot parse {path.name}: {error}") from error
    if matrix.size == 0:
        raise DataValidationError(f"Empty matrix: {path.name}")
    return matrix


def validate_sensor(
    matrix: np.ndarray, sensor: str, expected_cycles: int = EXPECTED_CYCLES
) -> None:
    expected = (expected_cycles, SENSOR_RATES[sensor] * CYCLE_SECONDS)
    if matrix.shape != expected:
        raise DataValidationError(f"{sensor}: shape {matrix.shape}; expected {expected}")
    if not np.isfinite(matrix).all():
        raise DataValidationError(f"{sensor}: contains missing or non-finite values")


def validate_profile(profile: np.ndarray, expected_cycles: int = EXPECTED_CYCLES) -> np.ndarray:
    if profile.shape != (expected_cycles, 5):
        raise DataValidationError(
            f"profile.txt: shape {profile.shape}; expected {(expected_cycles, 5)}"
        )
    if not np.isfinite(profile).all():
        raise DataValidationError("profile.txt: contains missing or non-finite values")
    for index, (column, domain) in enumerate(zip(PROFILE_COLUMNS, PROFILE_DOMAINS, strict=True)):
        invalid = set(np.unique(profile[:, index]).tolist()) - domain
        if invalid:
            raise DataValidationError(f"profile.txt: invalid {column} values: {sorted(invalid)}")
    return profile.astype(np.int64)


def cycle_metadata(profile: np.ndarray, expected_cycles: int = EXPECTED_CYCLES) -> list[dict]:
    """One-based cycle IDs, target and condition metadata; never prediction features."""
    validated = validate_profile(profile, expected_cycles)
    return [
        {"cycle_id": index + 1, **dict(zip(PROFILE_COLUMNS, map(int, row), strict=True))}
        for index, row in enumerate(validated)
    ]


def counts(values: np.ndarray) -> dict[str, int]:
    unique, frequencies = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, frequencies, strict=True)}


def consecutive_runs(values: np.ndarray) -> list[dict]:
    if len(values) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return [
        {
            "value": int(values[start]),
            "start_cycle_id": int(start + 1),
            "end_cycle_id": int(end),
            "cycles": int(end - start),
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def profile_audit(profile: np.ndarray) -> dict:
    target = profile[:, 2]
    distributions = {name: counts(profile[:, index]) for index, name in enumerate(PROFILE_COLUMNS)}
    cross_tabs = {}
    for index in (0, 1, 3, 4):
        cross_tabs[PROFILE_COLUMNS[index]] = {
            str(leakage): {
                str(value): int(np.sum((target == leakage) & (profile[:, index] == value)))
                for value in sorted(PROFILE_DOMAINS[index])
            }
            for leakage in sorted(TARGET_LABELS)
        }
    runs = consecutive_runs(target)
    components = profile[:, :4]
    starts = np.r_[0, np.flatnonzero(np.any(components[1:] != components[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(profile)]
    blocks = [
        {
            "start_cycle_id": int(start + 1),
            "end_cycle_id": int(end),
            "cycles": int(end - start),
            "conditions": dict(zip(PROFILE_COLUMNS[:4], map(int, components[start]), strict=True)),
            "stability_flag_1_cycles": int(np.sum(profile[start:end, 4] == 1)),
        }
        for start, end in zip(starts, ends, strict=True)
    ]
    configurations = Counter(tuple(row) for row in components.tolist())
    block_configurations = Counter(tuple(components[start]) for start in starts)
    lengths = [block["cycles"] for block in blocks]
    return {
        "target_column_1_based": 3,
        "target_labels": TARGET_LABELS,
        "distributions": distributions,
        "pump_leakage_cross_tabs": cross_tabs,
        "ordering": {
            "cycle_id_basis": "One-based original file row number; no shuffling or filtering",
            "adjacent_pairs": len(target) - 1,
            "adjacent_equal_target_pairs": int(np.sum(target[1:] == target[:-1])),
            "target_transitions": len(runs) - 1,
            "target_runs": runs,
            "unique_component_configurations": len(configurations),
            "component_block_definition": (
                "Maximal consecutive run of identical first four profile columns"
            ),
            "component_blocks": blocks,
            "component_block_count": len(blocks),
            "block_length_min": min(lengths),
            "block_length_median": float(np.median(lengths)),
            "block_length_max": max(lengths),
            "configurations_in_multiple_blocks": sum(n > 1 for n in block_configurations.values()),
            "condition_transitions": {
                name: int(np.sum(profile[1:, i] != profile[:-1, i]))
                for i, name in enumerate(PROFILE_COLUMNS)
            },
        },
    }


def refine_duplicate_groups(groups: list[list[int]], matrix: np.ndarray) -> list[list[int]]:
    """Refine candidate groups using sensor values only, with exact equality after hashing."""
    refined = []
    for group in groups:
        buckets: dict[bytes, list[list[int]]] = {}
        for index in group:
            row = np.array(matrix[index], dtype="<f8", copy=True)
            row[row == 0] = 0.0  # Numeric equality treats positive and negative zero equally.
            digest = hashlib.sha256(row.tobytes()).digest()
            candidates = buckets.setdefault(digest, [])
            for candidate in candidates:
                if np.array_equal(matrix[candidate[0]], matrix[index]):
                    candidate.append(index)
                    break
            else:
                candidates.append([index])
        refined.extend(
            candidate for bucket in buckets.values() for candidate in bucket if len(candidate) > 1
        )
    return sorted(refined, key=lambda group: group[0])


def duplicate_cycle_audit(groups: list[list[int]], profile: np.ndarray | None) -> dict:
    """Annotate complete sensor-equality groups with targets only after grouping is finished."""
    findings = []
    for group in groups:
        targets = [int(profile[index, 2]) for index in group] if profile is not None else None
        findings.append(
            {
                "cycle_ids": [index + 1 for index in group],
                "pump_leakage_targets": targets,
                "target_values": sorted(set(targets)) if targets is not None else None,
                "conflicting_targets": len(set(targets)) > 1 if targets is not None else None,
            }
        )
    return {
        "status": "completed",
        "definition": (
            "Exact numeric equality of all within-cycle values in all 17 sensor matrices, "
            "after float64 parsing; target, cycle ID and other profile columns are excluded. "
            "SHA-256 narrows comparisons; candidate rows are also compared with array_equal."
        ),
        "group_count": len(findings),
        "cycles_in_duplicate_groups": sum(len(group) for group in groups),
        "redundant_cycles": sum(len(group) - 1 for group in groups),
        "conflicting_target_group_count": (
            sum(group["conflicting_targets"] for group in findings) if profile is not None else None
        ),
        "conflicting_target_cycle_count": (
            sum(len(group["cycle_ids"]) for group in findings if group["conflicting_targets"])
            if profile is not None
            else None
        ),
        "groups": findings,
        "action": "Report only; no cycles are automatically removed or relabeled",
    }


def audit_dataset(
    dataset_dir: Path,
    *,
    expected_cycles: int = EXPECTED_CYCLES,
    provenance_path: Path | None = None,
) -> tuple[dict, list[dict]]:
    """Scan one sensor at a time; collect failures instead of hiding malformed data."""
    report: dict[str, Any] = {
        "schema_version": 2,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "generated_at_utc": utc_now(),
        "source_url": SOURCE_URL,
        "dataset_id": 447,
        "expected_cycles": expected_cycles,
        "cycle_duration_seconds": CYCLE_SECONDS,
        "expected_sensor_signals": len(SENSOR_RATES),
        "inventory": [],
        "sensors": {},
        "errors": [],
        "warnings": [],
        "duplicate_sensor_cycles": {
            "status": "not_completed",
            "reason": "Sensor validation pending",
        },
        "value_validation_scope": (
            "Numeric parsing, shape, NaN/infinity and profile code domains. "
            "Physical sensor range validity is not established by this audit."
        ),
    }
    trusted_hashes = None
    if provenance_path is not None:
        report["archive_checksum_verified"] = False
        report["manifest_bound_to_archive"] = False
        archive = provenance_path.parent / "uci447.zip"
        try:
            trusted_hashes = archive_content_hashes(archive, ARCHIVE_SHA256)
            report["archive_checksum_verified"] = True
        except DataValidationError as error:
            report["errors"].append(str(error))
        try:
            provenance = read_download_manifest(provenance_path)
            report["download"] = provenance
            if trusted_hashes is not None:
                verify_manifest_content(provenance, archive, trusted_hashes, ARCHIVE_SHA256)
                report["manifest_bound_to_archive"] = True
        except (OSError, DataValidationError) as error:
            report["errors"].append(f"Cannot verify download provenance: {error}")
    try:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_symlink():
                raise DataValidationError(f"Unexpected link in extraction: {path.name}")
            if path.is_file():
                name = path.relative_to(dataset_dir).as_posix()
                digest = sha256_file(path)
                report["inventory"].append(
                    {"file": name, "bytes": path.stat().st_size, "sha256": digest}
                )
                if trusted_hashes is not None and digest != trusted_hashes.get(name):
                    report["errors"].append(f"File checksum differs from verified archive: {name}")
    except (OSError, DataValidationError) as error:
        report["errors"].append(f"Cannot inventory extraction: {error}")
    if trusted_hashes is not None:
        missing = trusted_hashes.keys() - {item["file"] for item in report["inventory"]}
        if missing:
            report["errors"].append(f"Files missing from extraction: {sorted(missing)}")
    duplicate_groups = [list(range(expected_cycles))]
    for sensor, rate in SENSOR_RATES.items():
        details: dict[str, Any] = {
            "sampling_rate_hz": rate,
            "expected_shape": [expected_cycles, rate * CYCLE_SECONDS],
        }
        report["sensors"][sensor] = details
        try:
            matrix = load_matrix(dataset_dir / f"{sensor}.txt")
            details.update(
                shape=list(matrix.shape),
                missing_nan=int(np.isnan(matrix).sum()),
                invalid_infinite=int(np.isinf(matrix).sum()),
            )
            validate_sensor(matrix, sensor, expected_cycles)
            duplicate_groups = refine_duplicate_groups(duplicate_groups, matrix)
            details["valid"] = True
            del matrix
        except DataValidationError as error:
            details.update(valid=False, error=str(error))
            report["errors"].append(str(error))
    metadata = []
    validated_profile = None
    try:
        profile = load_matrix(dataset_dir / "profile.txt")
        report["profile_matrix"] = {
            "shape": list(profile.shape),
            "missing_nan": int(np.isnan(profile).sum()),
            "invalid_infinite": int(np.isinf(profile).sum()),
        }
        profile = validate_profile(profile, expected_cycles)
        validated_profile = profile
        metadata = cycle_metadata(profile, expected_cycles)
        report["profile"] = profile_audit(profile)
    except DataValidationError as error:
        report["errors"].append(str(error))
    if all(sensor["valid"] for sensor in report["sensors"].values()):
        report["duplicate_sensor_cycles"] = duplicate_cycle_audit(
            duplicate_groups, validated_profile
        )
        if report["duplicate_sensor_cycles"]["group_count"]:
            report["warnings"].append(
                "Identical complete sensor cycles found; inspect duplicate groups"
            )
        if report["duplicate_sensor_cycles"]["conflicting_target_group_count"]:
            report["warnings"].append(
                "Identical complete sensor cycles have conflicting leakage targets"
            )
    else:
        report["duplicate_sensor_cycles"]["reason"] = "At least one sensor failed validation"
    report["validation_passed"] = not report["errors"]
    if report["validation_passed"]:
        report["all_cycles_retained"] = True
    return report, metadata


def write_cycle_metadata(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=("cycle_id", *PROFILE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def inspect_dataset(
    raw_dir: Path, report_dir: Path, metadata_dir: Path, *, expected_cycles: int = EXPECTED_CYCLES
) -> dict:
    """Invalidate old metadata first; publish a replacement only after a successful audit."""
    metadata_path = metadata_dir / "cycle_metadata.csv"
    metadata_path.unlink(missing_ok=True)
    report, rows = audit_dataset(
        raw_dir / "uci447",
        expected_cycles=expected_cycles,
        provenance_path=raw_dir / "download.json",
    )
    write_json_atomic(report_dir / "dataset-audit.json", report)
    with atomic_text_writer(report_dir / "dataset-audit.md") as stream:
        stream.write(render_audit(report))
    if report["validation_passed"]:
        write_cycle_metadata(rows, metadata_path)
    return report


def render_audit(report: dict) -> str:
    """Render measurements from an executed audit, including any failed checks."""
    lines = [
        "# HydroGuard dataset audit",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        f"Structural validation: **{'PASS' if report['validation_passed'] else 'FAIL'}**.",
        "",
        f"Source: [UCI dataset 447]({SOURCE_URL}).",
        "",
    ]
    lines += (
        ["All cycles retained in original order.", ""]
        if report.get("all_cycles_retained")
        else ["This failed audit does not certify cycle completeness or row alignment.", ""]
    )
    if "archive_checksum_verified" in report:
        lines += [
            f"Pinned archive checksum verified: {report['archive_checksum_verified']}. "
            f"Download manifest bound to archive content: {report['manifest_bound_to_archive']}.",
            "",
        ]
    if "download" in report:
        download = report["download"]
        lines += [
            f"Archive: {download.get('archive_url', 'not recorded')}",
            "",
            f"Downloaded (UTC): {download.get('downloaded_at_utc', 'not recorded')}",
            "",
            f"Archive size: {download.get('archive_bytes', 'not recorded')} bytes.",
            "",
            f"SHA-256: `{download['archive_sha256']}`.",
            "",
            "Digest is locally observed, not a publisher-supplied authenticity guarantee.",
            "",
        ]
    lines += ["## File inventory", "", "| File | Bytes |", "| --- | ---: |"]
    lines += [f"| {item['file']} | {item['bytes']:,} |" for item in report["inventory"]]
    lines += [
        "",
        "Per-file SHA-256 digests are recorded in the companion JSON report.",
        "",
        "## Sensor validation",
        "",
        "Cycles are documented as 60 seconds; expected widths "
        "are sampling rate × 60. These files have no timestamp column to verify elapsed time.",
        "",
        "| Signal | Hz | Shape | NaN | Infinity | Valid |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for sensor, values in report["sensors"].items():
        lines.append(
            f"| {sensor} | {values['sampling_rate_hz']} | {values.get('shape', 'unparsed')} "
            f"| {values.get('missing_nan', 'unknown')} "
            f"| {values.get('invalid_infinite', 'unknown')} | {values['valid']} |"
        )
    lines += [
        "",
        report["value_validation_scope"],
        "",
        "Empty fields, ragged rows and nonnumeric tokens are parsing failures; they are "
        "never filled or discarded. NaN and infinity counts apply to successfully parsed matrices.",
        "",
    ]
    if "profile" in report:
        profile = report["profile"]
        lines += [
            "## Profile, target and conditions",
            "",
            f"Profile matrix: `{report['profile_matrix']['shape']}`; "
            f"NaN: {report['profile_matrix']['missing_nan']}; "
            f"infinity: {report['profile_matrix']['invalid_infinite']}.",
            "",
            "The third column is the target: 0 = no leakage, 1 = weak leakage, "
            "2 = severe leakage. Cycle IDs are one-based original row numbers. Other "
            "condition columns and cycle IDs are audit metadata, never prediction features.",
            "",
            "| Profile column | Value | Cycles | Share |",
            "| --- | ---: | ---: | ---: |",
        ]
        for column, distribution in profile["distributions"].items():
            total = sum(distribution.values())
            for value, count in distribution.items():
                lines.append(f"| {column} | {value} | {count} | {count / total:.2%} |")
        lines += [
            "",
            "Stability flag 0 means stable; 1 means static conditions might not have been reached.",
            "",
            "## Cross-tabulations",
            "",
            "Cells are cycle counts; rows are pump leakage classes.",
            "",
        ]
        for name, table in profile["pump_leakage_cross_tabs"].items():
            columns = list(table["0"])
            lines += [
                f"### {name}",
                "",
                "| Pump leakage | " + " | ".join(columns) + " |",
                "| --- | " + " | ".join(["---:"] * len(columns)) + " |",
            ]
            lines += [
                f"| {label} | " + " | ".join(str(row[c]) for c in columns) + " |"
                for label, row in table.items()
            ]
            lines.append("")
        order = profile["ordering"]
        lines += [
            "## Consecutive cycles and condition blocks",
            "",
            f"Adjacent cycles with the same target: {order['adjacent_equal_target_pairs']} / "
            f"{order['adjacent_pairs']}; target transitions: {order['target_transitions']}.",
            "",
            "| Target | Consecutive runs | Shortest run | Longest run |",
            "| --- | ---: | ---: | ---: |",
        ]
        for label in TARGET_LABELS:
            lengths = [run["cycles"] for run in order["target_runs"] if run["value"] == label]
            lines.append(
                f"| {label} | {len(lengths)} | {min(lengths) if lengths else 0} "
                f"| {max(lengths) if lengths else 0} |"
            )
        lines += [
            "",
            f"Observed component-condition blocks: {order['component_block_count']}; "
            f"unique configurations: {order['unique_component_configurations']}. "
            f"Block lengths (min / median / max): {order['block_length_min']} / "
            f"{order['block_length_median']:g} / {order['block_length_max']} cycles. "
            f"Configurations recurring in separate blocks: "
            f"{order['configurations_in_multiple_blocks']}.",
            "",
            "A block is a maximal consecutive run with identical first four profile columns; "
            "the stability flag does not define block boundaries. These observed blocks "
            "do not establish independent experimental sessions or the acquisition protocol.",
            "",
            "| Condition | Adjacent transitions |",
            "| --- | ---: |",
        ]
        lines += [f"| {name} | {count} |" for name, count in order["condition_transitions"].items()]
        lines += [
            "",
            "The companion JSON contains every target run and component block with "
            "original cycle boundaries. Observed sequential persistence means experimental "
            "order must not be assumed random. A later evaluation design must consider "
            "temporal dependence, repeated configurations and component confounding.",
            "",
        ]
    duplicates = report["duplicate_sensor_cycles"]
    lines += ["## Identical complete sensor cycles", ""]
    if duplicates["status"] == "completed":
        lines += [
            duplicates["definition"],
            "",
            f"Duplicate groups: {duplicates['group_count']}; cycles in groups: "
            f"{duplicates['cycles_in_duplicate_groups']}; redundant cycles beyond one per group: "
            f"{duplicates['redundant_cycles']}. Conflicting-target groups: "
            f"{duplicates['conflicting_target_group_count']}; cycles in conflicting groups: "
            f"{duplicates['conflicting_target_cycle_count']}.",
            "",
            "| Cycle IDs | Leakage targets in matching order | Target conflict |",
            "| --- | --- | --- |",
        ]
        for group in duplicates["groups"]:
            lines.append(
                f"| {', '.join(map(str, group['cycle_ids']))} "
                f"| {group['pump_leakage_targets']} | {group['conflicting_targets']} |"
            )
        lines += [
            "",
            duplicates["action"] + ".",
            "",
            "Conflicting targets are data quality findings, not structural parse failures. "
            "They remain unresolved in Phase 1A; the audit does not choose a correct label.",
            "",
        ]
    else:
        lines += [
            f"Not completed: {duplicates['reason']}. No complete-cycle equality claim is made.",
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "One controlled hydraulic test rig and a fixed load cycle cannot establish "
        "generalization to industrial equipment. Several component faults are superimposed; "
        "condition labels can confound leakage diagnosis. Physical sensor plausibility and "
        "independent experiment identifiers are not established. This phase performs no "
        "EDA, feature engineering, split selection, model training or deployment.",
        "",
        "## Audit warnings",
        "",
        *([f"- {warning}" for warning in report["warnings"]] or ["None."]),
        "",
        "## Failed checks",
        "",
    ]
    lines += [f"- {error}" for error in report["errors"]] or ["None."]
    return "\n".join(lines) + "\n"
