#!/usr/bin/env python3
"""Scan every raw image once and cache face detections for later review/building."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from seiyuu_faceid.dataset.detections import save_detection_records, scan_raw_detections
from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.vision.detector import SCRFDDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/runtime/detections.jsonl")
    args = parser.parse_args()

    config = load_config(_bootstrap.ROOT / args.config)
    detector = SCRFDDetector(
        config.detector_model,
        input_size=config.detector_input_size,
        score_threshold=config.detector_score_threshold,
        nms_threshold=config.detector_nms_threshold,
        max_long_edge=config.detector_max_long_edge,
        intra_op_threads=config.intra_op_threads,
        inter_op_threads=config.inter_op_threads,
    )

    def report(number: int, total: int, source) -> None:
        if number == 1 or number == total or number % 25 == 0:
            print(f"[{number}/{total}] {source.relative_path}")

    sources, records = scan_raw_detections(config.root / "data/raw", detector, progress=report)
    output_path = _bootstrap.ROOT / args.output
    save_detection_records(output_path, records)
    reviewable = sum(1 for record in records.values() if record.detections)
    print(
        f"Cached detections for {len(sources)} images; {reviewable} images need review: "
        f"{output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
