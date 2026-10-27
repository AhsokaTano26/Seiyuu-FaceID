#!/usr/bin/env python3
"""Select a separate portable classifier using grouped development-only cross-validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from seiyuu_faceid.experiments.classifier_search import run_search
from seiyuu_faceid.experiments.sample_size import load_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="A new private result directory."
    )
    parser.add_argument("--data", type=Path, default=_bootstrap.ROOT / "data/training")
    parser.add_argument(
        "--baseline", type=Path, default=_bootstrap.ROOT / "models/selected/classifier.npz"
    )
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative.")
    provenance = {}
    for path in (
        args.data / "features_and_splits.npz",
        args.data / "samples.jsonl",
        Path(__file__),
        _bootstrap.ROOT / "src/seiyuu_faceid/experiments/classifier_search.py",
    ):
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = run_search(
        load_data(args.data), args.baseline, args.output, seed=args.seed, provenance=provenance
    )
    holdout = dict(result["holdout"])
    for name in ("baseline", "winner"):
        holdout[name] = {key: value for key, value in holdout[name].items() if key != "per_class"}
    print(
        json.dumps(
            dict(
                output=str(args.output.resolve()),
                winner=result["winner"],
                holdout=holdout,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
