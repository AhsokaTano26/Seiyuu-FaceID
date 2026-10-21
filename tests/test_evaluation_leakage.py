from __future__ import annotations

from pathlib import Path

from seiyuu_faceid.dataset.scanner import scan_raw_dataset


def test_hash_overlap_identifies_an_enrollment_leak(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "kana"
    evaluation = tmp_path / "gallery" / "kana"
    raw.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    (raw / "source.jpg").write_bytes(b"same-image")
    (evaluation / "copy.jpg").write_bytes(b"same-image")

    raw_hashes = {source.sha256 for source in scan_raw_dataset(tmp_path / "raw")}
    evaluation_hashes = {source.sha256 for source in scan_raw_dataset(tmp_path / "gallery")}

    assert raw_hashes & evaluation_hashes
