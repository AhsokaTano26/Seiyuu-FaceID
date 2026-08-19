"""Append-only JSONL metadata, suitable for human review and later SQLite import."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {number} in {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Metadata line {number} in {path} is not an object.")
        records.append(value)
    return records
