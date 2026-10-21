"""Pure and session-adjacent state helpers used by Streamlit pages."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from seiyuu_faceid.dataset.scanner import IMAGE_SUFFIXES


def parse_group_face_input(value: str, detected_face_count: int) -> list[int]:
    """Parse one-based face IDs and return unique zero-based indexes."""
    normalized = value.strip().casefold()
    if not normalized:
        return []
    if normalized in {"a", "all", "全部", "全选"}:
        return list(range(detected_face_count))
    tokens = [token for token in re.split(r"[\s,，]+", normalized) if token]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise ValueError("请使用如 1,3 的 Face ID，或输入 A 表示全部。")
    indexes = [int(token) - 1 for token in tokens]
    if any(index < 0 or index >= detected_face_count for index in indexes):
        raise ValueError(f"Face ID 必须在 1 到 {detected_face_count} 之间。")
    if len(set(indexes)) != len(indexes):
        raise ValueError("Face ID 不能重复。")
    return sorted(indexes)


def is_valid_identity(value: str) -> bool:
    """Allow one safe directory name, never a path supplied by the browser."""
    candidate = Path(value)
    return bool(value) and candidate.name == value and value not in {".", ".."}


def query_image_paths(directory: Path) -> list[Path]:
    """Return supported query images recursively without modifying the directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def consume_dialog_request(state: MutableMapping[str, Any], key: str) -> str | None:
    """Consume a one-shot dialog request so unrelated reruns never reopen it."""
    value = state.pop(key, None)
    return value if isinstance(value, str) else None


def result_rows_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Format serialized pipeline values for the compact Streamlit result table."""
    rows: list[dict[str, Any]] = []
    for number, face in enumerate(record.get("faces", []), start=1):
        if not isinstance(face, dict):
            continue
        candidates = face.get("candidates", [])
        top = " · ".join(
            f"{candidate['identity']} ({candidate['score']:.3f})"
            for candidate in candidates[:5]
            if isinstance(candidate, dict)
        )
        rows.append(
            {
                "face": number,
                "result": face.get("identity")
                or ("待复核" if face.get("status") == "review" else "Unknown"),
                "status": face.get("status"),
                "detector_score": round(float(face.get("detector_score", 0)), 3),
                "quality_score": round(float(face.get("quality_score", 0)), 3),
                "quality_flags": ", ".join(face.get("quality_reasons", [])) or "-",
                "top_candidates": top or "-",
            }
        )
    return rows
