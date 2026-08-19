from seiyuu_faceid.dataset.splitter import split_records


def _record(identity: str, source: str, face: int) -> dict[str, object]:
    return {
        "accepted": True,
        "identity": identity,
        "source_sha256": source,
        "aligned_face_path": f"{identity}/{source}_{face}.jpg",
    }


def test_splitter_is_deterministic_and_keeps_sources_together() -> None:
    records = [
        _record("a", "a1", 0),
        _record("a", "a1", 1),
        _record("a", "a2", 0),
        _record("a", "a3", 0),
        _record("b", "b1", 0),
        _record("b", "b2", 0),
        _record("b", "b3", 0),
    ]
    first = split_records(records, seed=7)
    second = split_records(records, seed=7)
    assert first == second
    source_splits: dict[str, set[str]] = {}
    for split, split_records_ in first.items():
        for record in split_records_:
            source_splits.setdefault(str(record["source_sha256"]), set()).add(split)
    assert all(len(splits) == 1 for splits in source_splits.values())
    assert sum(len(part) for part in first.values()) == len(records)
