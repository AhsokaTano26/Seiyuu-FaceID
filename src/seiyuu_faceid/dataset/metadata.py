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


def load_evaluation_identity_aliases(
    path: Path, evaluation_root: Path, repository_root: Path
) -> dict[str, str]:
    """Load folder or per-face identity corrections bound to one evaluation root."""
    if not path.is_file():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Evaluation identity aliases must be a JSON object.")
    declared_root = record.get("evaluation_root")
    aliases = record.get("aliases")
    if not isinstance(declared_root, str) or not isinstance(aliases, dict):
        raise ValueError("Identity aliases need evaluation_root and an aliases mapping.")
    if (repository_root / declared_root).resolve() != evaluation_root.resolve():
        return {}
    if any(
        not isinstance(source, str)
        or not isinstance(target, str)
        or not source.strip()
        or not target.strip()
        for source, target in aliases.items()
    ):
        raise ValueError("Identity aliases must map non-empty labels to non-empty labels.")
    return aliases
