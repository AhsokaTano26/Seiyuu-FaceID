#!/usr/bin/env python3
"""Generate numbered previews and a selection manifest for multi-face raw images."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from seiyuu_faceid.dataset.detections import (
    current_detection_records,
    load_detection_records,
)
from seiyuu_faceid.dataset.metadata import write_jsonl
from seiyuu_faceid.dataset.scanner import scan_raw_dataset


def annotate(image_bgr, detections):
    """Draw zero-based face indexes that match the generated manifest."""
    import cv2
    import numpy as np

    preview = image_bgr.copy()
    for face_index, detection in enumerate(detections):
        x1, y1, x2, y2 = np.round(detection.bbox).astype(int)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (48, 166, 74), 2)
        cv2.putText(
            preview,
            f"Face {face_index}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (48, 166, 74),
            2,
            cv2.LINE_AA,
        )
    return preview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--detection-cache", default="data/runtime/detections.jsonl")
    parser.add_argument("--output-dir", default="data/review/raw_faces")
    parser.add_argument(
        "--overwrite-selections",
        action="store_true",
        help="Replace an existing face_selections.jsonl template in the output directory.",
    )
    args = parser.parse_args()

    import cv2

    raw_root = _bootstrap.ROOT / args.raw_root
    output_dir = _bootstrap.ROOT / args.output_dir
    cache_path = _bootstrap.ROOT / args.detection_cache
    selection_path = output_dir / "face_selections.jsonl"
    if selection_path.exists() and not args.overwrite_selections:
        raise FileExistsError(
            f"Selection file already exists: {selection_path}. Review or move it first, "
            "or use --overwrite-selections to intentionally replace it."
        )

    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Detection cache not found: {cache_path}. Run tools/scan_raw_faces.py first."
        )
    sources = scan_raw_dataset(raw_root)
    detection_records = current_detection_records(sources, load_detection_records(cache_path))
    if detection_records is None:
        raise ValueError("Detection cache is stale. Run tools/scan_raw_faces.py again.")
    records: list[dict[str, object]] = []
    for source in sources:
        image = cv2.imread(str(source.path))
        if image is None:
            continue
        detections = list(detection_records[source.relative_path.as_posix()].detections)
        if len(detections) < 2:
            continue
        preview_path = output_dir / source.relative_path.with_suffix(".jpg")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_path), annotate(image, detections)):
            raise OSError(f"Could not write preview: {preview_path}")
        records.append(
            {
                "source_image": source.relative_path.as_posix(),
                "face_indexes": None,
                "detected_face_count": len(detections),
                "preview_image": str(preview_path.relative_to(_bootstrap.ROOT)),
            }
        )
    write_jsonl(selection_path, records)
    print(f"Wrote {len(records)} multi-face review records to {selection_path}")
    print("Open each preview, replace face_indexes: null with a list such as [0, 2], then build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
