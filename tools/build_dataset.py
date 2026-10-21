#!/usr/bin/env python3
"""Prepare aligned, reviewed training faces without replacing the deployed classifier."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from seiyuu_faceid.dataset.cleaner import build_aligned_dataset
from seiyuu_faceid.dataset.detections import (
    current_detection_records,
    load_detection_records,
    save_detection_records,
    scan_raw_detections,
)
from seiyuu_faceid.dataset.scanner import scan_raw_dataset
from seiyuu_faceid.dataset.selections import load_face_selections
from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.vision.detector import SCRFDDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help=("Detect all raw images and write the reusable cache, without alignment or training."),
    )
    parser.add_argument(
        "--detection-cache",
        default="data/runtime/detections.jsonl",
        help="Reusable JSONL face-detection cache generated from data/raw/.",
    )
    parser.add_argument(
        "--face-selections",
        default="data/review/raw_faces/face_selections.jsonl",
        help=(
            "JSONL mapping every reviewed source_image to zero-based face_indexes. "
            "Any image without a selection is kept out of the training dataset."
        ),
    )
    args = parser.parse_args()
    config = load_config(_bootstrap.ROOT / args.config)
    selection_path = _bootstrap.ROOT / args.face_selections
    face_selections = load_face_selections(selection_path) if selection_path.is_file() else {}
    cache_path = _bootstrap.ROOT / args.detection_cache
    raw_root = config.root / "data/raw"
    sources = scan_raw_dataset(raw_root)
    cached_records = None
    if cache_path.is_file():
        cached_records = current_detection_records(sources, load_detection_records(cache_path))
    if cached_records is None:
        print("Detection cache is missing or stale; scanning all raw images once…")
        detector = SCRFDDetector(
            config.detector_model,
            input_size=config.detector_input_size,
            score_threshold=config.detector_score_threshold,
            nms_threshold=config.detector_nms_threshold,
            max_long_edge=config.detector_max_long_edge,
            intra_op_threads=config.intra_op_threads,
            inter_op_threads=config.inter_op_threads,
        )
        sources, cached_records = scan_raw_detections(raw_root, detector)
        save_detection_records(cache_path, cached_records)
    reviewable = sum(1 for record in cached_records.values() if record.detections)
    print(f"Detection cache ready: {len(sources)} images; {reviewable} images require review.")
    if args.scan_only:
        return 0
    accepted, rejected = build_aligned_dataset(
        config.root / "data/raw",
        config.root / "data/aligned",
        config.root / "data/rejected",
        detector=None,
        min_face_size=config.min_face_size,
        min_sharpness=config.min_sharpness,
        face_selections=face_selections,
        detection_records=cached_records,
    )
    print(f"Accepted faces: {accepted}; rejected records: {rejected}")
    print("The deployed identity classifier is unchanged; new samples require retraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
