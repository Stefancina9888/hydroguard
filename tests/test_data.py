"""Small, offline fixtures for acquisition, structure, metadata and audit behavior."""

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hydroguard.data import (
    PROFILE_COLUMNS,
    SENSOR_RATES,
    DataValidationError,
    audit_dataset,
    consecutive_runs,
    cycle_metadata,
    download_dataset,
    inspect_dataset,
    load_matrix,
    profile_audit,
    read_download_manifest,
    refine_duplicate_groups,
    render_audit,
    safe_extract,
    sha256_file,
    validate_profile,
    validate_sensor,
    write_cycle_metadata,
    write_json_atomic,
)


@pytest.fixture
def profile():
    return np.array(
        [[3, 100, 0, 130, 0], [3, 100, 0, 130, 1], [20, 90, 1, 115, 1], [100, 73, 2, 90, 0]],
        dtype=float,
    )


@pytest.fixture
def dataset(tmp_path, profile):
    directory = tmp_path / "dataset"
    directory.mkdir()
    for sensor, rate in SENSOR_RATES.items():
        np.savetxt(directory / f"{sensor}.txt", np.ones((4, rate * 60)), delimiter="\t")
    np.savetxt(directory / "profile.txt", profile, delimiter="\t")
    return directory


@pytest.fixture
def acquisition(tmp_path, dataset, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    archive = raw_dir / "uci447.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in dataset.iterdir():
            bundle.write(path, path.name)
    digest = sha256_file(archive)
    monkeypatch.setattr("hydroguard.data.ARCHIVE_SHA256", digest)
    download_dataset(raw_dir, expected_sha256=digest)
    return raw_dir


def test_expected_sensor_contract():
    assert len(SENSOR_RATES) == 17
    assert sum(rate * 60 for rate in SENSOR_RATES.values()) == 43680
    assert list(SENSOR_RATES.values()).count(100) == 7
    assert list(SENSOR_RATES.values()).count(10) == 2
    assert list(SENSOR_RATES.values()).count(1) == 8


@pytest.mark.parametrize("sensor,rate", SENSOR_RATES.items())
def test_sensor_shapes(sensor, rate):
    validate_sensor(np.ones((2, rate * 60)), sensor, expected_cycles=2)
    with pytest.raises(DataValidationError, match="shape"):
        validate_sensor(np.ones((2, rate * 60 - 1)), sensor, expected_cycles=2)
    with pytest.raises(DataValidationError, match="shape"):
        validate_sensor(np.ones((1, rate * 60)), sensor, expected_cycles=2)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_sensor_nonfinite_rejected(value):
    matrix = np.ones((1, 60))
    matrix[0, 0] = value
    with pytest.raises(DataValidationError, match="non-finite"):
        validate_sensor(matrix, "TS1", expected_cycles=1)


def test_target_metadata_and_stability_retained(profile, tmp_path):
    rows = cycle_metadata(profile, expected_cycles=4)
    assert [row["cycle_id"] for row in rows] == [1, 2, 3, 4]
    assert [row["pump_leakage"] for row in rows] == [0, 0, 1, 2]
    assert [row["stability_flag"] for row in rows] == [0, 1, 1, 0]
    assert list(rows[0]) == ["cycle_id", *PROFILE_COLUMNS]
    path = tmp_path / "processed" / "cycles.csv"
    write_cycle_metadata(rows, path)
    assert len(path.read_text().splitlines()) == 5
    assert "1,3,100,0,130,0" in path.read_text()


@pytest.mark.parametrize(
    "column,value", [(0, 4), (1, 99), (2, 3), (2, 0.5), (3, 129), (4, 2), (0, np.nan), (2, np.inf)]
)
def test_invalid_profile_codes(profile, column, value):
    profile[0, column] = value
    with pytest.raises(DataValidationError):
        validate_profile(profile, expected_cycles=4)


@pytest.mark.parametrize("shape", [(3, 5), (4, 4), (4, 6)])
def test_profile_shape(shape):
    with pytest.raises(DataValidationError, match="shape"):
        validate_profile(np.ones(shape), expected_cycles=4)


@pytest.mark.parametrize("contents", ["1\t2\n3\n", "1\tbad\n", "1\t\t2\n", "1\t2 #bad\n", ""])
def test_bad_text_is_not_imputed(tmp_path, contents):
    path = tmp_path / "bad.txt"
    path.write_text(contents)
    with pytest.raises(DataValidationError):
        load_matrix(path)


def test_audit_and_ordering(dataset):
    report, rows = audit_dataset(dataset, expected_cycles=4)
    assert report["validation_passed"]
    assert len(rows) == 4
    assert len(report["inventory"]) == 18
    profile = report["profile"]
    assert profile["distributions"]["pump_leakage"] == {"0": 2, "1": 1, "2": 1}
    assert profile["distributions"]["stability_flag"] == {"0": 2, "1": 2}
    assert profile["pump_leakage_cross_tabs"]["stability_flag"]["0"] == {"0": 1, "1": 1}
    for table in profile["pump_leakage_cross_tabs"].values():
        assert sum(sum(row.values()) for row in table.values()) == 4
    order = profile["ordering"]
    assert order["target_transitions"] == 2
    assert order["adjacent_equal_target_pairs"] == 1
    assert order["component_block_count"] == 3
    assert order["component_blocks"][0]["cycles"] == 2
    assert order["component_blocks"][0]["stability_flag_1_cycles"] == 1
    assert order["target_runs"][0] == {
        "value": 0,
        "start_cycle_id": 1,
        "end_cycle_id": 2,
        "cycles": 2,
    }
    assert "**PASS**" in render_audit(report)
    json.dumps(report, allow_nan=False)


def test_recurrent_blocks_and_single_cycle(profile):
    repeated = np.vstack((profile, profile[0]))
    order = profile_audit(repeated)["ordering"]
    assert order["configurations_in_multiple_blocks"] == 1
    assert order["component_block_count"] == 4
    assert profile_audit(profile[:1])["ordering"]["adjacent_pairs"] == 0
    assert consecutive_runs(np.array([])) == []


@pytest.mark.parametrize("problem", ["missing", "ragged", "nan", "infinity", "shape", "profile"])
def test_audit_reports_errors(dataset, problem):
    path = dataset / "TS1.txt"
    if problem == "missing":
        path.unlink()
    elif problem == "profile":
        (dataset / "profile.txt").write_text("3\t100\t9\t130\t0\n" * 4)
    elif problem == "shape":
        np.savetxt(path, np.ones((4, 59)), delimiter="\t")
    elif problem == "ragged":
        path.write_text("1\t2\n3\n")
    else:
        matrix = np.ones((4, 60))
        matrix[0, 0] = np.nan if problem == "nan" else np.inf
        np.savetxt(path, matrix, delimiter="\t")
    report, _ = audit_dataset(dataset, expected_cycles=4)
    assert not report["validation_passed"]
    assert report["errors"]
    assert "**FAIL**" in render_audit(report)


@pytest.mark.parametrize(
    "name", ["../escape.txt", "/absolute.txt", "C:/evil.txt", "a\\..\\escape.txt"]
)
def test_unsafe_zip_paths(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, "bad")
    with pytest.raises(DataValidationError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extraction")
    assert not (tmp_path / "escape.txt").exists()


def test_zip_duplicate_and_symlink(tmp_path):
    for entries in ([("PS1.txt", "1"), ("ps1.txt", "2")], []):
        archive = tmp_path / "bad.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, content in entries:
                bundle.writestr(name, content)
            if not entries:
                link = zipfile.ZipInfo("link")
                link.external_attr = 0o120777 << 16
                bundle.writestr(link, "../escape")
        with pytest.raises(DataValidationError, match="Unsafe"):
            safe_extract(archive, tmp_path / "out")


def write_fixture_zip(path: Path, *, complete=True):
    with zipfile.ZipFile(path, "w") as bundle:
        for sensor in SENSOR_RATES if complete else ["PS1"]:
            bundle.writestr(f"{sensor}.txt", "1\n")
        bundle.writestr("profile.txt", "3\t100\t0\t130\t0\n")


def test_download_cache_checksums_and_missing_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    archive = raw_dir / "uci447.zip"
    write_fixture_zip(archive)
    digest = sha256_file(archive)
    with patch("urllib.request.urlopen", side_effect=AssertionError("Network called")):
        first = download_dataset(raw_dir, expected_sha256=digest)
        assert first["archive_sha256"] == digest
        assert download_dataset(raw_dir, expected_sha256=digest) == first
        sensor = raw_dir / "uci447" / "PS1.txt"
        sensor.write_text("changed")
        with pytest.raises(DataValidationError, match="checksum mismatch"):
            download_dataset(raw_dir, expected_sha256=digest)
        sensor.unlink()
        with pytest.raises(DataValidationError, match="inventory mismatch"):
            download_dataset(raw_dir, expected_sha256=digest)
        archive.write_bytes(b"corrupt")
        with pytest.raises(DataValidationError, match="Archive checksum mismatch"):
            download_dataset(raw_dir, expected_sha256=digest)


def test_incomplete_archive_and_wrong_digest(tmp_path):
    archive = tmp_path / "uci447.zip"
    write_fixture_zip(archive, complete=False)
    with pytest.raises(DataValidationError, match="checksum mismatch"):
        download_dataset(tmp_path, expected_sha256="0" * 64)
    with pytest.raises(DataValidationError, match="missing required files"):
        download_dataset(tmp_path, expected_sha256=sha256_file(archive))
    assert not (tmp_path / "uci447").exists()
    assert not list(tmp_path.glob("extract-*"))


def test_download_from_mock_http(tmp_path):
    archive = tmp_path / "fixture.zip"
    write_fixture_zip(archive)
    payload = archive.read_bytes()

    class Response(io.BytesIO):
        headers = {"ETag": '"synthetic"'}

        def geturl(self):
            return "https://archive.ics.uci.edu/synthetic-fixture.zip"

    with patch("urllib.request.urlopen", return_value=Response(payload)) as request:
        result = download_dataset(tmp_path / "raw", expected_sha256=sha256_file(archive))
    request.assert_called_once()
    assert result["http_etag"] == '"synthetic"'
    assert result["archive_bytes"] == len(payload)
    assert len(result["extracted_sha256"]) == 18
    assert not list((tmp_path / "raw").glob("*.part"))


def test_audit_provenance_tamper_detected(acquisition):
    dataset = acquisition / "uci447"
    manifest = acquisition / "download.json"
    good, _ = audit_dataset(dataset, expected_cycles=4, provenance_path=manifest)
    assert good["validation_passed"]
    assert good["manifest_bound_to_archive"]
    (dataset / "TS1.txt").write_text("changed")
    bad, _ = audit_dataset(dataset, expected_cycles=4, provenance_path=manifest)
    assert not bad["validation_passed"]
    assert any("File checksum" in error for error in bad["errors"])
    (acquisition / "uci447.zip").write_bytes(b"changed")
    bad, _ = audit_dataset(dataset, expected_cycles=4, provenance_path=manifest)
    assert not bad["archive_checksum_verified"]
    assert any("Archive checksum" in error for error in bad["errors"])


@pytest.mark.parametrize("contents", [None, "{broken", "[]", "{}", '{"extracted_sha256": {}}'])
def test_bad_provenance_still_produces_failure_report(dataset, tmp_path, contents):
    manifest = tmp_path / "download.json"
    if contents is not None:
        manifest.write_text(contents)
    report, _ = audit_dataset(dataset, expected_cycles=4, provenance_path=manifest)
    assert not report["validation_passed"]
    assert any("Cannot verify download provenance" in error for error in report["errors"])
    assert "**FAIL**" in render_audit(report)
    json.dumps(report, allow_nan=False)


def test_complete_cycle_duplicates_are_target_independent(dataset):
    report, rows = audit_dataset(dataset, expected_cycles=4)
    duplicates = report["duplicate_sensor_cycles"]
    assert report["validation_passed"]
    assert len(rows) == 4
    assert duplicates["group_count"] == 1
    assert duplicates["cycles_in_duplicate_groups"] == 4
    assert duplicates["redundant_cycles"] == 3
    assert duplicates["conflicting_target_group_count"] == 1
    assert duplicates["conflicting_target_cycle_count"] == 4
    assert duplicates["groups"] == [
        {
            "cycle_ids": [1, 2, 3, 4],
            "pump_leakage_targets": [0, 0, 1, 2],
            "target_values": [0, 1, 2],
            "conflicting_targets": True,
        }
    ]
    assert report["warnings"]
    assert "1, 2, 3, 4" in render_audit(report)
    profile = load_matrix(dataset / "profile.txt")
    profile[:, 2] = 0
    np.savetxt(dataset / "profile.txt", profile, delimiter="\t")
    changed, _ = audit_dataset(dataset, expected_cycles=4)
    group = changed["duplicate_sensor_cycles"]["groups"][0]
    assert group["cycle_ids"] == [1, 2, 3, 4]
    assert group["conflicting_targets"] is False


@pytest.mark.parametrize("sensor", SENSOR_RATES)
def test_duplicate_audit_compares_every_sensor(dataset, sensor):
    path = dataset / f"{sensor}.txt"
    matrix = load_matrix(path)
    matrix[-1, -1] = np.nextafter(matrix[-1, -1], np.inf)
    np.savetxt(path, matrix, delimiter="\t")
    report, _ = audit_dataset(dataset, expected_cycles=4)
    assert report["duplicate_sensor_cycles"]["groups"][0]["cycle_ids"] == [1, 2, 3]


def test_duplicate_hash_collisions_and_signed_zero():
    matrix = np.array([[0.0, 2.0], [-0.0, 2.0], [0.0, 3.0], [0.0, 3.0]])
    with patch("hydroguard.data.hashlib.sha256") as digest:
        digest.return_value.digest.return_value = b"collision"
        assert refine_duplicate_groups([[0, 1, 2, 3]], matrix) == [[0, 1], [2, 3]]


def test_distinct_duplicate_groups_use_original_cycle_ids(dataset):
    for sensor in SENSOR_RATES:
        path = dataset / f"{sensor}.txt"
        matrix = load_matrix(path)
        matrix[[1, 3], :] = 7
        np.savetxt(path, matrix, delimiter="\t")
    report, _ = audit_dataset(dataset, expected_cycles=4)
    duplicates = report["duplicate_sensor_cycles"]
    assert [group["cycle_ids"] for group in duplicates["groups"]] == [[1, 3], [2, 4]]
    assert [group["pump_leakage_targets"] for group in duplicates["groups"]] == [[0, 1], [0, 2]]
    assert duplicates["conflicting_target_group_count"] == 2


def test_failed_sensor_never_claims_complete_duplicate_audit(dataset):
    (dataset / "SE.txt").unlink()
    report, _ = audit_dataset(dataset, expected_cycles=4)
    assert report["duplicate_sensor_cycles"]["status"] == "not_completed"
    assert "groups" not in report["duplicate_sensor_cycles"]


def test_duplicate_audit_survives_invalid_target_without_guessing(dataset):
    (dataset / "profile.txt").write_text("3\t100\t9\t130\t0\n" * 4)
    report, _ = audit_dataset(dataset, expected_cycles=4)
    assert not report["validation_passed"]
    assert report["duplicate_sensor_cycles"]["groups"][0]["cycle_ids"] == [1, 2, 3, 4]
    assert report["duplicate_sensor_cycles"]["conflicting_target_group_count"] is None
    assert report["duplicate_sensor_cycles"]["groups"][0]["pump_leakage_targets"] is None


@pytest.mark.parametrize("filename", ["PS1.txt", "TS1.txt", "profile.txt"])
@pytest.mark.parametrize("blank", ["\n", " \t \n", "\r\n"])
@pytest.mark.parametrize("line_number", [1, 3, 5])
def test_blank_rows_rejected_with_physical_line_number(dataset, filename, blank, line_number):
    path = dataset / filename
    lines = path.read_text().splitlines(keepends=True)
    lines.insert(line_number - 1, blank)
    path.write_text("".join(lines))
    with pytest.raises(DataValidationError, match=f"{filename}.*line {line_number}"):
        load_matrix(path)
    report, _ = audit_dataset(dataset, expected_cycles=4)
    assert not report["validation_passed"]
    assert not report.get("all_cycles_retained")
    markdown = render_audit(report)
    assert "All cycles retained" not in markdown
    assert "No cycles were removed" not in markdown


def test_empty_file_reports_name_and_line(tmp_path):
    path = tmp_path / "empty.txt"
    path.touch()
    with pytest.raises(DataValidationError, match=r"empty.txt.*line 1"):
        load_matrix(path)


def test_success_then_failed_audit_removes_stale_metadata(acquisition, tmp_path):
    reports, metadata = tmp_path / "reports", tmp_path / "processed"
    first = inspect_dataset(acquisition, reports, metadata, expected_cycles=4)
    csv_path = metadata / "cycle_metadata.csv"
    assert first["validation_passed"]
    assert len(csv_path.read_text().splitlines()) == 5
    path = acquisition / "uci447" / "PS1.txt"
    path.write_text(path.read_text() + "\n")
    failed = inspect_dataset(acquisition, reports, metadata, expected_cycles=4)
    assert not failed["validation_passed"]
    assert not csv_path.exists()
    assert not json.loads((reports / "dataset-audit.json").read_text())["validation_passed"]
    assert "All cycles retained" not in (reports / "dataset-audit.md").read_text()


def test_unexpected_audit_exception_invalidates_metadata(acquisition, tmp_path):
    reports, metadata = tmp_path / "reports", tmp_path / "processed"
    inspect_dataset(acquisition, reports, metadata, expected_cycles=4)
    with patch("hydroguard.data.audit_dataset", side_effect=OSError("read interrupted")):
        with pytest.raises(OSError, match="read interrupted"):
            inspect_dataset(acquisition, reports, metadata, expected_cycles=4)
    assert not (metadata / "cycle_metadata.csv").exists()


def test_interrupted_report_write_does_not_republish_metadata(acquisition, tmp_path):
    reports, metadata = tmp_path / "reports", tmp_path / "processed"
    inspect_dataset(acquisition, reports, metadata, expected_cycles=4)
    with patch("hydroguard.data.write_json_atomic", side_effect=InterruptedError("report write")):
        with pytest.raises(InterruptedError):
            inspect_dataset(acquisition, reports, metadata, expected_cycles=4)
    assert not (metadata / "cycle_metadata.csv").exists()


def test_file_and_local_manifest_tampering_cannot_bypass_archive(acquisition):
    archive = acquisition / "uci447.zip"
    original_pin = sha256_file(archive)
    path = acquisition / "uci447" / "TS1.txt"
    matrix = load_matrix(path)
    matrix[0, 0] = 42
    np.savetxt(path, matrix, delimiter="\t")
    manifest_path = acquisition / "download.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extracted_sha256"]["TS1.txt"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="verified archive content"):
        download_dataset(acquisition, expected_sha256=original_pin)
    report, _ = audit_dataset(
        acquisition / "uci447", expected_cycles=4, provenance_path=manifest_path
    )
    assert not report["validation_passed"]
    assert report["archive_checksum_verified"]
    assert not report["manifest_bound_to_archive"]
    assert any(
        "File checksum differs from verified archive: TS1.txt" in e for e in report["errors"]
    )
    assert sha256_file(archive) == original_pin
    assert load_matrix(path)[0, 0] == 42  # Detection must not repair or overwrite raw files.


@pytest.mark.parametrize(
    "key,value",
    [
        ("dataset_id", 448),
        ("dataset_id", 447.0),
        ("schema_version", True),
        ("source_url", "https://example.org"),
        ("archive_url", None),
        ("archive_sha256", "bad"),
        ("archive_bytes", True),
        ("archive_bytes", -1),
        ("resolved_url", 1),
        ("http_etag", []),
        ("checksum_basis", ""),
        ("downloaded_at_utc", "bad"),
        ("downloaded_at_utc", "2026-01-01T12:00:00"),
        ("downloaded_at_utc", None),
        ("extracted_sha256", []),
        ("extracted_sha256", {"../outside.txt": "0" * 64}),
        ("extracted_sha256", {"PS1.txt": "bad"}),
    ],
)
def test_downloader_rejects_malformed_manifest(acquisition, key, value):
    manifest = acquisition / "download.json"
    contents = json.loads(manifest.read_text())
    # The synthetic acquisition starts from a cached archive and has a recovery timestamp.
    contents.pop("recovered_at_utc", None)
    contents["downloaded_at_utc"] = "2026-10-30T18:00:00+00:00"
    contents[key] = value
    manifest.write_text(json.dumps(contents))
    before = manifest.read_bytes()
    with patch("urllib.request.urlopen", side_effect=AssertionError("Network called")):
        with pytest.raises(DataValidationError, match="download.json"):
            download_dataset(acquisition, expected_sha256=sha256_file(acquisition / "uci447.zip"))
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("contents", ["{broken", "[]", "{}", '{"dataset_id":447,"dataset_id":447}'])
def test_downloader_rejects_invalid_json_and_duplicate_keys(acquisition, contents):
    (acquisition / "download.json").write_text(contents)
    with pytest.raises(DataValidationError, match="download provenance"):
        download_dataset(acquisition, expected_sha256=sha256_file(acquisition / "uci447.zip"))


def test_interrupted_manifest_write_recovers_verified_extraction(acquisition):
    manifest = acquisition / "download.json"
    manifest.unlink()
    before = {p.name: sha256_file(p) for p in (acquisition / "uci447").iterdir()}
    digest = sha256_file(acquisition / "uci447.zip")

    def interrupted_dump(value, stream, **kwargs):
        stream.write('{"interrupted":')
        raise InterruptedError("manifest write interrupted")

    with patch("hydroguard.data.json.dump", side_effect=interrupted_dump):
        with pytest.raises(InterruptedError, match="manifest write interrupted"):
            download_dataset(acquisition, expected_sha256=digest)
    assert not manifest.exists()
    assert not list(acquisition.glob("*.tmp"))
    with patch("urllib.request.urlopen", side_effect=AssertionError("Network called")):
        recovered = download_dataset(acquisition, expected_sha256=digest)
    assert recovered["downloaded_at_utc"] is None
    assert recovered["recovered_at_utc"]
    assert read_download_manifest(manifest) == recovered
    assert before == {p.name: sha256_file(p) for p in (acquisition / "uci447").iterdir()}


def test_atomic_manifest_failure_preserves_previous_manifest(acquisition):
    manifest = acquisition / "download.json"
    original = manifest.read_bytes()
    with patch("hydroguard.data.os.replace", side_effect=InterruptedError("publish interrupted")):
        with pytest.raises(InterruptedError):
            write_json_atomic(manifest, {"incomplete": True})
    assert manifest.read_bytes() == original
    assert not list(acquisition.glob("*.tmp"))


def test_interrupted_manifest_publication_after_first_extraction(tmp_path):
    archive = tmp_path / "uci447.zip"
    write_fixture_zip(archive)
    digest = sha256_file(archive)
    with patch(
        "hydroguard.data.os.replace", side_effect=InterruptedError("publication interrupted")
    ):
        with pytest.raises(InterruptedError):
            download_dataset(tmp_path, expected_sha256=digest)
    assert (tmp_path / "uci447" / "profile.txt").exists()
    assert not (tmp_path / "download.json").exists()
    assert download_dataset(tmp_path, expected_sha256=digest)["archive_sha256"] == digest


def test_interrupted_extraction_publication_can_resume(tmp_path):
    archive = tmp_path / "uci447.zip"
    write_fixture_zip(archive)
    digest = sha256_file(archive)
    with patch.object(Path, "rename", side_effect=InterruptedError("directory publish")):
        with pytest.raises(InterruptedError):
            download_dataset(tmp_path, expected_sha256=digest)
    assert not (tmp_path / "uci447").exists()
    assert not (tmp_path / "download.json").exists()
    assert download_dataset(tmp_path, expected_sha256=digest)["archive_sha256"] == digest


def test_recovery_refuses_modified_extraction_without_manifest(acquisition):
    (acquisition / "download.json").unlink()
    sensor = acquisition / "uci447" / "PS1.txt"
    sensor.write_text("changed")
    with pytest.raises(DataValidationError, match="checksum mismatch"):
        download_dataset(acquisition, expected_sha256=sha256_file(acquisition / "uci447.zip"))
    assert sensor.read_text() == "changed"
    assert not (acquisition / "download.json").exists()


def test_stale_temporary_publications_are_never_trusted(acquisition):
    (acquisition / "download.json").unlink()
    (acquisition / ".download.json-stale.tmp").write_text("bad manifest")
    (acquisition / ".uci447-stale.part").write_text("bad archive")
    staging = acquisition / "extract-stale"
    staging.mkdir()
    (staging / "profile.txt").write_text("bad profile")
    result = download_dataset(acquisition, expected_sha256=sha256_file(acquisition / "uci447.zip"))
    assert result["extracted_sha256"]["profile.txt"] == sha256_file(
        acquisition / "uci447" / "profile.txt"
    )


def test_interrupted_http_transfer_does_not_publish_partial_archive(tmp_path):
    fixture = tmp_path / "fixture.zip"
    write_fixture_zip(fixture)
    payload = fixture.read_bytes()
    raw_dir = tmp_path / "raw"

    class Response(io.BytesIO):
        headers = {}

        def geturl(self):
            return "https://archive.ics.uci.edu/synthetic-fixture.zip"

    def interrupted_copy(source, output):
        output.write(source.read(10))
        raise InterruptedError("HTTP transfer interrupted")

    with patch("urllib.request.urlopen", return_value=Response(payload)):
        with patch("hydroguard.data.shutil.copyfileobj", side_effect=interrupted_copy):
            with pytest.raises(InterruptedError, match="HTTP transfer interrupted"):
                download_dataset(raw_dir, expected_sha256=sha256_file(fixture))
    assert not (raw_dir / "uci447.zip").exists()
    assert not (raw_dir / "uci447").exists()
    assert not (raw_dir / "download.json").exists()
    assert not list(raw_dir.glob("*.part"))
    with patch("urllib.request.urlopen", return_value=Response(payload)):
        result = download_dataset(raw_dir, expected_sha256=sha256_file(fixture))
    assert result["archive_sha256"] == sha256_file(fixture)
    assert result["downloaded_at_utc"] is not None
