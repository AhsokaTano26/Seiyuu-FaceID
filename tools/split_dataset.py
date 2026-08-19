#!/usr/bin/env python3
"""Create private, source-grouped enrollment, calibration, and test manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import _bootstrap  # noqa: F401

from seiyuu_faceid.dataset.metadata import read_jsonl, write_jsonl
from seiyuu_faceid.dataset.splitter import split_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/aligned/metadata.jsonl")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    records = read_jsonl(_bootstrap.ROOT / args.metadata)
    splits = split_records(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    output_dir = _bootstrap.ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"seed": args.seed, "splits": {}}
    for name, split in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", split)
        identities = Counter(str(record["identity"]) for record in split)
        summary["splits"][name] = {
            "faces": len(split),
            "source_images": len({record.get("source_sha256") for record in split}),
            "identities": dict(sorted(identities.items())),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
