from __future__ import annotations

from app import query_image_paths


def test_query_image_paths_recurses_into_nested_directories(tmp_path) -> None:
    root = tmp_path / "query"
    nested = root / "event" / "day_01"
    nested.mkdir(parents=True)
    (root / "top.jpg").touch()
    (nested / "face.png").touch()
    (nested / "notes.txt").touch()

    results = query_image_paths(root)

    assert [path.relative_to(root).as_posix() for path in results] == [
        "event/day_01/face.png",
        "top.jpg",
    ]
