#!/usr/bin/env python3
"""Recognize every face in one image against the local gallery."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.runtime.engine import FaceIdEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(_bootstrap.ROOT / args.config)
    engine = FaceIdEngine(config)
    results = engine.recognize_image(_bootstrap.ROOT / args.image)
    if not results:
        print("No faces detected.")
        return 0
    for index, result in enumerate(results, start=1):
        print(f"Face {index}")
        print(f"Detection: {result.detection.score:.3f}; quality: {result.quality.score:.3f}")
        if result.quality.reasons:
            print("Quality flags: " + ", ".join(result.quality.reasons))
        print("Top candidates:")
        for rank, candidate in enumerate(result.decision.candidates, start=1):
            print(f"{rank}. {candidate.identity}  {candidate.score:.3f}")
        print(f"Result: {result.decision.identity or 'Unknown'}")
        print(f"Status: {result.decision.status}")
        if index != len(results):
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
