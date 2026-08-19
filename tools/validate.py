#!/usr/bin/env python3
"""Evaluate labeled embedding pairs from an NPZ archive and suggest thresholds."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from seiyuu_faceid.validation import evaluate_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_npz", help="NPZ with left, right (N×D) and same (N) arrays")
    args = parser.parse_args()
    import numpy as np

    with np.load(args.pairs_npz) as pairs:
        metrics = evaluate_pairs(pairs["left"], pairs["right"], pairs["same"])
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
