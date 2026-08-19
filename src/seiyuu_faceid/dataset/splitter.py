"""Leakage-resistant, deterministic dataset splits for aligned-face metadata."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

SPLIT_NAMES = ("train", "validation", "test")


def split_records(
    records: Iterable[dict[str, object]],
    *,
    seed: int = 20260819,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict[str, list[dict[str, object]]]:
    """Split accepted records by identity and source image, never separating a source's faces."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between zero and one.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than one.")

    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if not record.get("accepted", False):
            continue
        identity = record.get("identity")
        source = record.get("source_sha256") or record.get("source_image")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Every accepted record needs a non-empty identity.")
        if not isinstance(source, str) or not source:
            raise ValueError("Every accepted record needs source_sha256 or source_image.")
        grouped[identity][source].append(record)
    if not grouped:
        raise ValueError("No accepted records were supplied.")

    output: dict[str, list[dict[str, object]]] = {name: [] for name in SPLIT_NAMES}
    for identity in sorted(grouped):
        source_groups = list(grouped[identity].values())
        random.Random(f"{seed}:{identity}").shuffle(source_groups)
        total = sum(len(group) for group in source_groups)
        targets = {
            "train": round(total * train_ratio),
            "validation": round(total * validation_ratio),
        }
        targets["test"] = total - targets["train"] - targets["validation"]
        assigned = dict.fromkeys(SPLIT_NAMES, 0)
        for group in source_groups:
            split = max(
                SPLIT_NAMES,
                key=lambda name: (targets[name] - assigned[name], -SPLIT_NAMES.index(name)),
            )
            output[split].extend({**record, "split": split} for record in group)
            assigned[split] += len(group)
    return output
