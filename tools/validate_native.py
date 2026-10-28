#!/usr/bin/env python3
"""Compare the C++ core with Python on explicitly selected local images."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
from pathlib import Path

import numpy as np

if __package__:
    from tools import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401


class NativeEngine:
    """Small validation-only ctypes adapter; production HTTP uses Go/cgo."""

    def __init__(self, library: Path, model: Path):
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        ptr = ctypes.c_void_p
        out = ctypes.POINTER(ptr)
        byte_ptr = ctypes.POINTER(ctypes.c_ubyte)
        self.library.faceid_create.argtypes = [ctypes.c_char_p, out, out]
        self.library.faceid_create.restype = ctypes.c_int
        self.library.faceid_destroy.argtypes = [ptr]
        self.library.faceid_destroy.restype = None
        self.library.faceid_free.argtypes = [ptr]
        self.library.faceid_free.restype = None
        self.library.faceid_model.argtypes = [ptr, out]
        self.library.faceid_model.restype = ctypes.c_int
        self.library.faceid_process_bgr.argtypes = [
            ptr, byte_ptr, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_double, out,
        ]
        self.library.faceid_process_bgr.restype = ctypes.c_int
        self.library.faceid_process_encoded.argtypes = [
            ptr, byte_ptr, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, out,
        ]
        self.library.faceid_process_encoded.restype = ctypes.c_int
        self.handle = ptr()
        error = ptr()
        status = self.library.faceid_create(
            os.fsencode(Path(model).resolve()), ctypes.byref(self.handle), ctypes.byref(error)
        )
        self._consume(status, error)
        if not self.handle.value:
            raise RuntimeError("Native engine returned an empty handle.")

    def _consume(self, status: int, result: ctypes.c_void_p):
        try:
            data = json.loads(ctypes.string_at(result)) if result.value else None
        finally:
            if result.value:
                self.library.faceid_free(result)
        if status:
            code = (data.get("error", {}).get("code", "native_error")
                    if isinstance(data, dict) else "native_error")
            raise RuntimeError(f"Native core failed: {code}")
        return data

    def _require_open(self):
        if not self.handle.value:
            raise RuntimeError("Native engine is closed.")

    def model(self) -> dict:
        self._require_open()
        output = ctypes.c_void_p()
        status = self.library.faceid_model(self.handle, ctypes.byref(output))
        return self._consume(status, output)

    def process_bgr(self, pixels: np.ndarray, *, mode=0, top_k=5, threshold=0.95) -> dict:
        self._require_open()
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("Expected uint8 H x W x 3 BGR pixels.")
        pixels = np.ascontiguousarray(pixels)
        height, width = pixels.shape[:2]
        output = ctypes.c_void_p()
        status = self.library.faceid_process_bgr(
            self.handle, pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), pixels.nbytes,
            width, height, mode, top_k, threshold, ctypes.byref(output),
        )
        return self._consume(status, output)

    def process_encoded(self, data: bytes, *, mode=0, top_k=5, threshold=0.95) -> dict:
        self._require_open()
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        output = ctypes.c_void_p()
        status = self.library.faceid_process_encoded(
            self.handle, buffer, len(data), mode, top_k, threshold, ctypes.byref(output)
        )
        return self._consume(status, output)

    def close(self):
        if self.handle.value:
            self.library.faceid_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def compare_crops(reference_features, native_features, reference_scores, native_scores,
                  threshold: float) -> dict:
    """Require matching decisions/ranks and declared numeric tolerances, without labels."""
    result = {"feature_atol": 2e-5, "probability_atol": 5e-6, "rtol": 0.0}
    for label, left, right, tolerance in (
        ("features", reference_features, native_features, 2e-5),
        ("probabilities", reference_scores, native_scores, 5e-6),
    ):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape or left.ndim != 2 or not left.size:
            raise ValueError("Parity arrays have incompatible shapes or are empty.")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("Parity arrays contain nonfinite values.")
        result[f"{label}_exact"] = bool(np.array_equal(left, right))
        result[f"{label}_allclose"] = bool(np.allclose(left, right, atol=tolerance, rtol=0))
        result[f"{label}_max_abs_difference"] = float(np.max(np.abs(left - right)))
    left, right = np.asarray(reference_scores), np.asarray(native_scores)
    left_order = np.argsort(-left, axis=1, kind="stable")
    right_order = np.argsort(-right, axis=1, kind="stable")
    samples = len(left)
    result.update({
        "samples": samples,
        "top1_matches": int(np.sum(left_order[:, 0] == right_order[:, 0])),
        "top5_matches": int(np.sum(np.all(left_order[:, :5] == right_order[:, :5], axis=1))),
        # Production Candidate converts float32 scores to Python float before
        # comparison; NumPy weak-scalar comparison would round the threshold.
        "threshold_matches": int(np.sum((left.astype(np.float64).max(1) >= threshold)
                                        == (right.astype(np.float64).max(1) >= threshold))),
        "probability_threshold": threshold,
    })
    result["passed"] = (
        result["features_allclose"] and result["probabilities_allclose"]
        and all(result[key] == samples for key in
                ("top1_matches", "top5_matches", "threshold_matches"))
    )
    return result


def compare_images(reference: list[list[dict]], native: list[list[dict]]) -> dict:
    """Aggregate image pipeline parity, omitting identity strings and image locations."""
    if len(reference) != len(native):
        raise ValueError("Image result counts differ.")
    result = {"images": len(reference), "faces": sum(map(len, reference)),
              "face_count_matches": 0, "decisions_match": 0, "candidate_order_matches": 0,
              "bbox_max_abs_difference": 0.0, "detector_max_abs_difference": 0.0,
              "quality_max_abs_difference": 0.0, "probability_max_abs_difference": 0.0,
              "finite": True, "bbox_atol": 0.01, "score_atol": 5e-5}
    for before, after in zip(reference, native, strict=True):
        if len(before) != len(after):
            continue
        result["face_count_matches"] += 1
        for left, right in zip(before, after, strict=True):
            result["decisions_match"] += int(all(left[key] == right[key] for key in (
                "status", "identity", "decision_reason", "quality_reasons",
                "probability_threshold", "score_type")))
            expected = [row["identity"] for row in left["candidates"]]
            actual = [row["identity"] for row in right["candidates"]]
            result["candidate_order_matches"] += int(expected == actual)
            fields = [
                ("bbox", left["bbox"], right["bbox"]),
                ("detector", [left["detector_score"]], [right["detector_score"]]),
                ("quality", [left["quality_score"]], [right["quality_score"]]),
            ]
            if len(left["candidates"]) == len(right["candidates"]):
                fields.append(("probability", [row["score"] for row in left["candidates"]],
                               [row["score"] for row in right["candidates"]]))
            for label, x, y in fields:
                x, y = np.asarray(x), np.asarray(y)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    result["finite"] = False
                if x.size:
                    key = f"{label}_max_abs_difference"
                    result[key] = max(result[key], float(np.max(np.abs(x - y))))
    result["passed"] = (
        result["finite"] and result["face_count_matches"] == result["images"]
        and result["decisions_match"] == result["faces"]
        and result["candidate_order_matches"] == result["faces"]
        and result["bbox_max_abs_difference"] <= result["bbox_atol"]
        and all(result[f"{key}_max_abs_difference"] <= result["score_atol"]
                for key in ("detector", "quality", "probability"))
    )
    return result


def _paths(directory: Path, samples: int) -> list[Path]:
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not paths:
        raise ValueError("An input directory has no supported image files.")
    return paths[:samples]


def check_model_metadata(metadata: dict, model_id: str, class_names: list[str]) -> None:
    """Probability indices are comparable only when their identity order also matches."""
    if metadata.get("model_id") != model_id:
        raise ValueError("Python and native model identities differ.")
    if metadata.get("class_names") != class_names:
        raise ValueError("Python and native class order differs.")


def validate(library: Path, model: Path, config_path: Path, *, crop_dir: Path | None,
             image_dir: Path | None, samples: int, encoded: bool) -> dict:
    import cv2

    from seiyuu_faceid.api.uploads import UploadLimits, decode_image
    from seiyuu_faceid.runtime.config import load_config
    from seiyuu_faceid.runtime.engine import FaceIdEngine
    from seiyuu_faceid.runtime.pipeline import serialize_result

    config = load_config(config_path)
    report = {"schema_version": 1, "platform": platform.platform(),
              "scope": "Local Python/C++ parity; reused models, not independent accuracy evidence.",
              "runtime": "CPUExecutionProvider", "opencv_python": cv2.__version__}
    engine = FaceIdEngine(config)
    try:
        with NativeEngine(library, model) as native:
            metadata = native.model()
            check_model_metadata(metadata, engine.classifier.model_id,
                                 engine.classifier.class_names)
            report["model_id"] = engine.classifier.model_id
            report["class_order_matches"] = True
            options = {"top_k": config.top_k, "threshold": config.probability_threshold}
            if crop_dir is not None:
                features, scores, native_features, native_scores = [], [], [], []
                for path in _paths(crop_dir, samples):
                    if path.stat().st_size > 16 * 1024 * 1024:
                        raise ValueError("Crop exceeds the 16 MiB file limit.")
                    pixels = decode_image(path.read_bytes(), "crop", UploadLimits()).pixels
                    if pixels.shape != (112, 112, 3):
                        raise ValueError("Every aligned crop must be exactly 112 x 112 pixels.")
                    feature = engine.classifier.embed(pixels)
                    features.append(feature)
                    scores.append(engine.classifier.head.predict_proba(feature[None, :])[0])
                    result = native.process_bgr(pixels, mode=2, **options)
                    native_features.append(result["features"])
                    native_scores.append(result["probabilities"])
                report["crops"] = compare_crops(features, native_features, scores, native_scores,
                                                 config.probability_threshold)
            if image_dir is not None:
                expected_images, raw_images, encoded_images = [], [], []
                dimensions_match = True
                for path in _paths(image_dir, samples):
                    if path.stat().st_size > UploadLimits().max_upload_bytes:
                        raise ValueError("Original image exceeds the API file limit.")
                    data = path.read_bytes()
                    pixels = decode_image(data, "image", UploadLimits()).pixels
                    expected_images.append([serialize_result(row)
                                            for row in engine.recognize_array(pixels)])
                    result = native.process_bgr(pixels, **options)
                    raw_images.append(result["faces"])
                    dimensions_match &= (result["height"], result["width"]) == pixels.shape[:2]
                    if encoded:
                        result = native.process_encoded(data, **options)
                        encoded_images.append(result["faces"])
                        dimensions_match &= (result["height"], result["width"]) == pixels.shape[:2]
                report["raw_bgr_images"] = compare_images(expected_images, raw_images)
                report["dimensions_match"] = dimensions_match
                if encoded:
                    report["encoded_images"] = compare_images(expected_images, encoded_images)
            report["passed"] = all(
                value["passed"] for value in report.values()
                if isinstance(value, dict) and "passed" in value
            ) and report.get("dimensions_match", True)
            return report
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/selected/native-model.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/n150.yaml"))
    parser.add_argument("--crop-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--encoded", action="store_true",
                        help="Also compare native image decoding.")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.crop_dir and not args.image_dir:
        parser.error("Provide --crop-dir and/or --image-dir explicitly.")
    if args.encoded and not args.image_dir:
        parser.error("--encoded requires --image-dir.")
    if not 1 <= args.samples <= 10000:
        parser.error("--samples must be 1..10000.")
    if args.output.exists():
        parser.error("Output already exists; choose a new filepath.")
    report = validate(args.library, args.model, args.config, crop_dir=args.crop_dir,
                      image_dir=args.image_dir, samples=args.samples, encoded=args.encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
