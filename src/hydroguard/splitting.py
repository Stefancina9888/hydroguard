"""Frozen group-aware membership and original-cycle gap checks for Phase 2B."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hydroguard.data import PROFILE_COLUMNS, DataValidationError, utc_now, validate_profile
from hydroguard.features import validate_cycle_ids

GROUP_COLUMNS = ("cooler_condition_pct", "valve_condition_pct", "accumulator_pressure_bar")
PROTOCOL = {
    "name": "phase2b-nuisance-sgkf-v1",
    "group_columns": list(GROUP_COLUMNS),
    "outer_n_splits": 5,
    "test_fold_1_based": 2,
    "inner_n_splits": 3,
    "shuffle": True,
    "random_state": 2026,
    "outer_gap": 20,
    "inner_gap": 20,
    "distance_basis": "Absolute difference of original cycle_id; inclusive gap exclusion",
    "final_test_use": "Membership and integrity checks only; no predictions or metrics",
}
EXPECTED_OUTER = {
    "development": (1328, [846, 246, 236]),
    "test": (339, [133, 103, 103]),
    "excluded": (538, [242, 143, 153]),
}
EXPECTED_INNER = [
    {
        "train": (607, [453, 82, 72]),
        "validation": (442, [290, 81, 71]),
        "excluded": (279, [103, 83, 93]),
    },
    {
        "train": (576, [422, 82, 72]),
        "validation": (457, [293, 82, 82]),
        "excluded": (295, [131, 82, 82]),
    },
    {
        "train": (607, [463, 82, 62]),
        "validation": (429, [263, 83, 83]),
        "excluded": (292, [120, 81, 91]),
    },
]


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_metadata(metadata: pd.DataFrame) -> None:
    if metadata.index.name != "cycle_id" or tuple(metadata.columns) != PROFILE_COLUMNS:
        raise DataValidationError(
            "Metadata must keep cycle_id as its index and all five original profile columns"
        )
    validate_cycle_ids(metadata.index.to_numpy())
    validate_profile(metadata.to_numpy(), expected_cycles=len(metadata))


def group_codes(metadata: pd.DataFrame) -> np.ndarray:
    validate_metadata(metadata)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(metadata[list(GROUP_COLUMNS)]), sort=True)
    return codes


def nearest_cycle_distance(cycles, reference) -> np.ndarray:
    cycles, reference = np.asarray(cycles), np.sort(np.asarray(reference))
    if reference.size == 0:
        return np.full(cycles.size, np.inf)
    positions = np.searchsorted(reference, cycles)
    left = reference[np.maximum(positions - 1, 0)]
    right = reference[np.minimum(positions, len(reference) - 1)]
    return np.minimum(np.abs(cycles - left), np.abs(cycles - right))


def class_counts(metadata: pd.DataFrame, cycles) -> list[int]:
    values = metadata.loc[list(cycles), "pump_leakage"].to_numpy()
    return [int(np.sum(values == label)) for label in [0, 1, 2]]


def check_partition(metadata: pd.DataFrame, train, validation, *, gap: int = 20) -> dict:
    validate_metadata(metadata)
    train, validation = validate_cycle_ids(train), validate_cycle_ids(validation)
    if not np.isin(np.r_[train, validation], metadata.index).all():
        raise DataValidationError("Partition contains unknown cycle_ids")
    if np.intersect1d(train, validation).size:
        raise DataValidationError("Partition cycle_ids overlap")
    train_groups = set(map(tuple, metadata.loc[train, list(GROUP_COLUMNS)].to_numpy()))
    validation_groups = set(map(tuple, metadata.loc[validation, list(GROUP_COLUMNS)].to_numpy()))
    if train_groups & validation_groups:
        raise DataValidationError("Partition nuisance groups overlap")
    minimum = int(nearest_cycle_distance(train, validation).min())
    if minimum <= gap:
        raise DataValidationError(
            f"Original-cycle gap violated: minimum {minimum}, required > {gap}"
        )
    train_counts, validation_counts = (
        class_counts(metadata, train),
        class_counts(metadata, validation),
    )
    if not all(train_counts) or not all(validation_counts):
        raise DataValidationError("Both partitions must contain all three leakage classes")
    return {
        "train_cycles": len(train),
        "validation_cycles": len(validation),
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "minimum_original_cycle_distance": minimum,
        "shared_nuisance_groups": 0,
    }


def make_memberships(metadata: pd.DataFrame) -> dict:
    """Generate deterministic membership; original IDs survive development subsetting."""
    groups = group_codes(metadata)
    ids = metadata.index.to_numpy()
    outer = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026).split(
            np.zeros((len(ids), 1)), metadata.pump_leakage.to_numpy(), groups
        )
    )
    candidate_positions, test_positions = outer[1]
    test = ids[test_positions]
    candidate = ids[candidate_positions]
    keep = nearest_cycle_distance(candidate, test) > 20
    development, excluded = candidate[keep], candidate[~keep]
    outer_diagnostics = check_partition(metadata, development, test)
    development_meta = metadata.loc[development]
    inner = []
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026)
    for fold, (train_positions, val_positions) in enumerate(
        splitter.split(
            np.zeros((len(development), 1)),
            development_meta.pump_leakage.to_numpy(),
            group_codes(development_meta),
        ),
        start=1,
    ):
        candidate_train, validation = development[train_positions], development[val_positions]
        keep = nearest_cycle_distance(candidate_train, validation) > 20
        train, inner_excluded = candidate_train[keep], candidate_train[~keep]
        diagnostics = check_partition(metadata, train, validation)
        inner.append(
            {
                "fold": fold,
                "train_cycle_ids": train.tolist(),
                "validation_cycle_ids": validation.tolist(),
                "excluded_cycle_ids": inner_excluded.tolist(),
                "diagnostics": diagnostics,
            }
        )
    return {
        "outer": {
            "development_cycle_ids": development.tolist(),
            "test_cycle_ids": test.tolist(),
            "excluded_cycle_ids": excluded.tolist(),
            "diagnostics": outer_diagnostics,
        },
        "inner": inner,
    }


def assert_expected_memberships(metadata: pd.DataFrame, memberships: dict) -> None:
    """Fail closed on any count, coverage, group, class or original-ID gap deviation."""
    if len(metadata) != 2205 or metadata.index.tolist() != list(range(1, 2206)):
        raise DataValidationError("Expected original cycle_ids 1 through 2205")
    outer = memberships["outer"]
    coverage = []
    for role, (size, counts) in EXPECTED_OUTER.items():
        ids = validate_cycle_ids(outer[f"{role}_cycle_ids"])
        if len(ids) != size or class_counts(metadata, ids) != counts:
            raise DataValidationError(f"Unexpected outer {role} count/class support")
        coverage.extend(ids.tolist())
    if sorted(coverage) != metadata.index.tolist():
        raise DataValidationError("Outer roles must be a disjoint, complete partition")
    development, test = outer["development_cycle_ids"], outer["test_cycle_ids"]
    if check_partition(metadata, development, test) != outer["diagnostics"]:
        raise DataValidationError("Outer partition diagnostics changed")
    if np.any(nearest_cycle_distance(outer["excluded_cycle_ids"], test) > 20):
        raise DataValidationError("Outer excluded cycles are not inside the test gap")
    if len(memberships["inner"]) != 3:
        raise DataValidationError("Expected exactly three inner folds")
    validations = []
    for index, (fold, expected) in enumerate(
        zip(memberships["inner"], EXPECTED_INNER, strict=True), 1
    ):
        if fold["fold"] != index:
            raise DataValidationError("Inner fold numbering changed")
        all_ids = []
        for role, (size, counts) in expected.items():
            ids = validate_cycle_ids(fold[f"{role}_cycle_ids"])
            if len(ids) != size or class_counts(metadata, ids) != counts:
                raise DataValidationError(
                    f"Unexpected inner fold {index} {role} count/class support"
                )
            all_ids.extend(ids.tolist())
        if sorted(all_ids) != development:
            raise DataValidationError("Inner roles must partition development only")
        if (
            check_partition(metadata, fold["train_cycle_ids"], fold["validation_cycle_ids"])
            != fold["diagnostics"]
        ):
            raise DataValidationError("Inner partition diagnostics changed")
        if np.any(
            nearest_cycle_distance(fold["excluded_cycle_ids"], fold["validation_cycle_ids"]) > 20
        ):
            raise DataValidationError("Inner excluded cycles are not inside the validation gap")
        validations.extend(fold["validation_cycle_ids"])
    if sorted(validations) != development:
        raise DataValidationError(
            "Every development cycle must appear in inner validation exactly once"
        )
    if memberships != make_memberships(metadata):
        raise DataValidationError("Membership differs from the approved deterministic protocol")


def load_split_lock(path: Path) -> dict:
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        if set(wrapper) != {"created_at_utc", "payload", "payload_sha256"}:
            raise ValueError("Invalid split lock schema")
        if canonical_sha256(wrapper["payload"]) != wrapper["payload_sha256"]:
            raise ValueError("Split lock checksum mismatch")
        return wrapper
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise DataValidationError(f"Cannot use split lock: {error}") from error


def lock_or_verify(path: Path, payload: dict) -> dict:
    """Publish a complete lock without replacement; existing locks must match exactly."""
    if path.exists():
        existing = load_split_lock(path)
        if existing["payload"] != payload:
            raise DataValidationError(
                "Existing locked split differs; automatic overwrite is forbidden"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "created_at_utc": utc_now(),
        "payload": payload,
        "payload_sha256": canonical_sha256(payload),
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".split-lock-", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(wrapper, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)  # Atomic no-clobber publication on the same filesystem.
    except FileExistsError:
        return lock_or_verify(path, payload)
    finally:
        temporary.unlink(missing_ok=True)
    return wrapper
