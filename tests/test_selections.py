from __future__ import annotations

import pytest

from seiyuu_faceid.dataset.selections import load_face_selections, parse_review_face_indexes


def test_load_face_selections_reads_zero_based_face_indexes(tmp_path) -> None:
    selection_file = tmp_path / "face_selections.jsonl"
    selection_file.write_text('{"source_image":"kana/group.jpg","face_indexes":[1,2]}\n')

    assert load_face_selections(selection_file) == {"kana/group.jpg": [1, 2]}


def test_load_face_selections_accepts_legacy_single_face_index(tmp_path) -> None:
    selection_file = tmp_path / "face_selections.jsonl"
    selection_file.write_text('{"source_image":"kana/group.jpg","face_index":1}\n')

    assert load_face_selections(selection_file) == {"kana/group.jpg": [1]}


def test_load_face_selections_rejects_incomplete_review_templates(tmp_path) -> None:
    selection_file = tmp_path / "face_selections.jsonl"
    selection_file.write_text('{"source_image":"kana/group.jpg","face_indexes":null}\n')

    with pytest.raises(ValueError, match="face_indexes list"):
        load_face_selections(selection_file)


def test_parse_review_face_indexes_accepts_one_based_ids_and_all() -> None:
    assert parse_review_face_indexes("1, 3", 3) == [0, 2]
    assert parse_review_face_indexes("A", 3) == [0, 1, 2]
    assert parse_review_face_indexes("", 3) == []


def test_parse_review_face_indexes_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        parse_review_face_indexes("0", 3)
    with pytest.raises(ValueError, match="must not be repeated"):
        parse_review_face_indexes("1 1", 3)
