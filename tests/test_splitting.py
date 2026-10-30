"""Synthetic original-ID boundaries, group isolation and immutable split publication."""

import itertools
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hydroguard.data import PROFILE_COLUMNS, DataValidationError
from hydroguard.splitting import (
    GROUP_COLUMNS,
    assert_expected_memberships,
    canonical_sha256,
    check_partition,
    group_codes,
    load_split_lock,
    lock_or_verify,
    make_memberships,
    nearest_cycle_distance,
)


@pytest.fixture
def metadata():
    rows = []
    groups = list(itertools.product([3, 20, 100], [73, 80, 90, 100], [90, 100, 115, 130]))[:10]
    for cooler, valve, accumulator in groups:
        for target in [0, 1, 2]:
            for repetition in range(3):
                rows.append([cooler, valve, target, accumulator, repetition % 2])
    return pd.DataFrame(
        rows,
        columns=PROFILE_COLUMNS,
        index=pd.Index(np.arange(len(rows)) * 100 + 1, name="cycle_id"),
    )


def test_membership_reproducibility_and_original_ID_gaps(metadata):
    first = make_memberships(metadata)
    assert make_memberships(metadata) == first
    assert first["outer"]["excluded_cycle_ids"] == []
    development = first["outer"]["development_cycle_ids"]
    assert all(fold["excluded_cycle_ids"] == [] for fold in first["inner"])
    assert sorted(c for fold in first["inner"] for c in fold["validation_cycle_ids"]) == development
    assert all(
        fold["diagnostics"]["minimum_original_cycle_distance"] >= 100 for fold in first["inner"]
    )
    # Compressing 72 development rows would incorrectly eliminate most/all candidates with gap 20.
    assert len(development) == 72 and len(first["outer"]["test_cycle_ids"]) == 18


def test_group_identity_excludes_target_and_stability(metadata):
    changed = metadata.copy()
    changed.pump_leakage = (changed.pump_leakage + 1) % 3
    changed.stability_flag = 1 - changed.stability_flag
    np.testing.assert_array_equal(group_codes(changed), group_codes(metadata))
    assert "pump_leakage" not in GROUP_COLUMNS and "stability_flag" not in GROUP_COLUMNS


def boundary_metadata(first_validation):
    rows = [[3, 100, target, 130, 0] for target in [0, 1, 2]]
    rows += [[20, 100, target, 130, 0] for target in [0, 1, 2]]
    return pd.DataFrame(
        rows,
        columns=PROFILE_COLUMNS,
        index=pd.Index(
            [1, 31, 61, first_validation, first_validation + 30, first_validation + 60],
            name="cycle_id",
        ),
    )


def test_gap_is_inclusive_at_20_and_allows_21():
    meta = boundary_metadata(81)
    with pytest.raises(DataValidationError, match="gap violated"):
        check_partition(meta, [1, 31, 61], [81, 111, 141])
    meta = boundary_metadata(82)
    result = check_partition(meta, [1, 31, 61], [82, 112, 142])
    assert result["minimum_original_cycle_distance"] == 21
    assert result["train_class_counts"] == result["validation_class_counts"] == [1, 1, 1]


@pytest.mark.parametrize("problem", ["overlap", "groups", "missing_class", "unknown_id"])
def test_partition_checks_fail_before_fitting(problem):
    meta = boundary_metadata(82)
    train, validation = [1, 31, 61], [82, 112, 142]
    if problem == "overlap":
        validation = [61, 82, 112]
    elif problem == "groups":
        meta.loc[82:, "cooler_condition_pct"] = 3
    elif problem == "missing_class":
        validation = [82, 112]
    else:
        validation = [82, 112, 999]
    with pytest.raises(DataValidationError):
        check_partition(meta, train, validation)


def test_nearest_distance_handles_edges_and_original_spacing():
    np.testing.assert_array_equal(
        nearest_cycle_distance([1, 81, 82, 1000], [61, 142]), [60, 20, 21, 858]
    )
    assert np.isinf(nearest_cycle_distance([1], [])).all()


def test_expected_production_counts_reject_small_synthetic_source(metadata):
    with pytest.raises(DataValidationError, match="1 through 2205"):
        assert_expected_memberships(metadata, make_memberships(metadata))


def test_expected_class_counts_reject_correct_length_but_wrong_support():
    meta = pd.DataFrame(
        [[3, 100, i % 3, 130, 0] for i in range(2205)],
        columns=PROFILE_COLUMNS,
        index=pd.Index(range(1, 2206), name="cycle_id"),
    )
    with pytest.raises(DataValidationError, match="outer development count/class"):
        assert_expected_memberships(
            meta, {"outer": {"development_cycle_ids": list(range(1, 1329))}}
        )


def test_split_lock_cannot_be_overwritten_and_reuse_keeps_exact_bytes(tmp_path):
    path = tmp_path / "split.json"
    payload = {"protocol": "fixed", "source_sha256": "a" * 64, "cycle_ids": [1, 31, 61]}
    first = lock_or_verify(path, payload)
    original = path.read_bytes()
    assert load_split_lock(path) == first
    assert lock_or_verify(path, payload) == first
    assert path.read_bytes() == original
    for changed in [{**payload, "source_sha256": "b" * 64}, {**payload, "cycle_ids": [1, 31, 62]}]:
        with pytest.raises(DataValidationError, match="overwrite is forbidden"):
            lock_or_verify(path, changed)
        assert path.read_bytes() == original


def test_corrupt_existing_lock_is_rejected_not_replaced(tmp_path):
    path = tmp_path / "split.json"
    lock_or_verify(path, {"source": "original"})
    wrapper = json.loads(path.read_text())
    wrapper["payload"]["source"] = "tampered"
    path.write_text(json.dumps(wrapper))
    before = path.read_bytes()
    with pytest.raises(DataValidationError, match="checksum mismatch"):
        lock_or_verify(path, {"source": "original"})
    assert path.read_bytes() == before


def test_interrupted_split_lock_write_publishes_nothing(tmp_path):
    path = tmp_path / "split.json"

    def interrupted(value, stream, **kwargs):
        stream.write('{"partial":')
        raise InterruptedError("lock write")

    with patch("hydroguard.splitting.json.dump", side_effect=interrupted):
        with pytest.raises(InterruptedError):
            lock_or_verify(path, {"source": "original"})
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    wrapper = lock_or_verify(path, {"source": "original"})
    assert wrapper["payload_sha256"] == canonical_sha256({"source": "original"})
