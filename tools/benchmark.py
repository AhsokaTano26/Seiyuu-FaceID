#!/usr/bin/env python3
"""Emit machine-comparable local inference timing and RSS metrics as JSON."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time

import _bootstrap  # noqa: F401

from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.runtime.engine import FaceIdEngine
from seiyuu_faceid.vision.alignment import align_face
from seiyuu_faceid.vision.quality import assess_face_quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if args.runs < 1 or args.batch_size < 1:
        raise ValueError("--runs and --batch-size must be positive.")
    config = load_config(_bootstrap.ROOT / args.config)
    started = time.perf_counter()
    engine = FaceIdEngine(config)
    model_load_ms = (time.perf_counter() - started) * 1000
    image_path = _bootstrap.ROOT / args.image
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    engine.recognize_array(image)  # warmup
    detector_timings: list[float] = []
    recognizer_timings: list[float] = []
    end_to_end_timings: list[float] = []
    faces = 0
    for _ in range(args.runs):
        started = time.perf_counter()
        detections = engine.detector.detect(image)
        detector_timings.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        for detection in detections:
            quality = assess_face_quality(
                image,
                detection.bbox,
                min_face_size=config.min_face_size,
                min_sharpness=config.min_sharpness,
            )
            if quality.accepted:
                engine.classifier.embed(align_face(image, detection.landmarks))
        recognizer_timings.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        for _ in range(args.batch_size):
            results = engine.recognize_array(image)
        end_to_end_timings.append((time.perf_counter() - started) * 1000)
        faces = len(results)
    # Darwin reports ru_maxrss in bytes; Linux reports KiB.
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = max_rss / (1024 * 1024) if platform.system() == "Darwin" else max_rss / 1024
    detector_mean_ms = sum(detector_timings) / len(detector_timings)
    recognizer_mean_ms = sum(recognizer_timings) / len(recognizer_timings)
    end_to_end_mean_ms = sum(end_to_end_timings) / len(end_to_end_timings) / args.batch_size
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runs": args.runs,
        "batch_size": args.batch_size,
        "faces_in_image": faces,
        "model_load_ms": round(model_load_ms, 3),
        "detector_mean_ms": round(detector_mean_ms, 3),
        "recognition_mean_ms": round(recognizer_mean_ms, 3),
        "end_to_end_mean_ms": round(end_to_end_mean_ms, 3),
        "end_to_end_p95_ms": round(
            sorted(end_to_end_timings)[max(0, int(len(end_to_end_timings) * 0.95) - 1)]
            / args.batch_size,
            3,
        ),
        "batch_throughput_images_per_s": round(1000 / end_to_end_mean_ms, 3),
        "max_rss_mb": round(rss_mb, 3),
        "recognition_scope": "alignment + quality check + embedding only",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
