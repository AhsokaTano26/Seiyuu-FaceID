"""Model-independent face detector interface and SCRFD ONNX implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: np.ndarray  # xyxy in source-image coordinates
    score: float
    landmarks: np.ndarray  # five points, shape (5, 2), source-image coordinates


class FaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]: ...


class DetectorError(RuntimeError):
    """Raised when a detector does not provide the expected interface."""


class ImageAspectError(ValueError):
    """Resizing this image to the detector canvas would collapse one dimension."""


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= threshold]
    return np.asarray(kept, dtype=np.int64)


class SCRFDDetector:
    """SCRFD ONNX detector for the official three-stride five-landmark export."""

    _STRIDES = (8, 16, 32)

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_long_edge: int = 1920,
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
        allow_spinning: bool = True,
        enable_cpu_mem_arena: bool = True,
    ) -> None:
        # Must be set before importing ORT; see ONNX Runtime's privacy guidance.
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        import onnxruntime as ort

        if not model_path.is_file():
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        if score_threshold <= 0 or score_threshold >= 1:
            raise ValueError("score_threshold must be between 0 and 1.")
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_long_edge = max_long_edge
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        spinning = "1" if allow_spinning else "0"
        options.add_session_config_entry("session.intra_op.allow_spinning", spinning)
        options.add_session_config_entry("session.inter_op.allow_spinning", spinning)
        options.enable_cpu_mem_arena = enable_cpu_mem_arena
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise DetectorError(
                "Expected a five-landmark SCRFD ONNX model with 9 outputs; "
                f"found {len(self.output_names)}. Use a *_KPS SCRFD export."
            )

    def detect(
        self, image_bgr: np.ndarray, *, score_threshold: float | None = None
    ) -> list[FaceDetection]:
        """Detect faces after bounded resize and letterbox; map results to source pixels."""
        import cv2

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise DetectorError("Expected a BGR image with three channels.")
        original_height, original_width = image_bgr.shape[:2]
        working, resize_scale = self._bounded_resize(image_bgr)
        canvas, scale, pad_x, pad_y = self._letterbox(working)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, self.input_size, (127.5,) * 3, swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        boxes, scores, landmarks = self._decode(
            outputs, canvas.shape[1], canvas.shape[0], score_threshold=score_threshold
        )
        if not len(boxes):
            return []
        keep = _nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        points = landmarks[keep]
        boxes -= np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)
        points -= np.array([pad_x, pad_y], dtype=np.float32)
        boxes /= scale
        boxes /= resize_scale
        points /= scale
        points /= resize_scale
        np.clip(boxes[:, ::2], 0, original_width - 1, out=boxes[:, ::2])
        np.clip(boxes[:, 1::2], 0, original_height - 1, out=boxes[:, 1::2])
        return [
            FaceDetection(bbox, float(score), keypoints)
            for bbox, score, keypoints in zip(boxes, scores[keep], points, strict=True)
        ]

    def _bounded_resize(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        import cv2

        height, width = image_bgr.shape[:2]
        long_edge = max(height, width)
        if long_edge <= self.max_long_edge:
            return image_bgr, 1.0
        scale = self.max_long_edge / long_edge
        size = (round(width * scale), round(height * scale))
        if min(size) < 1:
            raise ImageAspectError("Image aspect ratio is unsupported.")
        resized = cv2.resize(image_bgr, size)
        return resized, scale

    def _letterbox(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        import cv2

        target_width, target_height = self.input_size
        height, width = image_bgr.shape[:2]
        scale = min(target_width / width, target_height / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        if min(resized_width, resized_height) < 1:
            raise ImageAspectError("Image aspect ratio is unsupported.")
        resized = cv2.resize(image_bgr, (resized_width, resized_height))
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        pad_x = (target_width - resized_width) // 2
        pad_y = (target_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    def _decode(
        self,
        outputs: list[np.ndarray],
        input_width: int,
        input_height: int,
        *,
        score_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_landmarks: list[np.ndarray] = []
        for level, stride in enumerate(self._STRIDES):
            scores = np.asarray(outputs[level]).reshape(-1)
            feature_width = input_width // stride
            feature_height = input_height // stride
            locations = feature_width * feature_height
            if not locations or scores.size % locations:
                raise DetectorError(
                    "SCRFD output shapes do not match the configured detector input size."
                )
            anchors = scores.size // locations
            threshold = (
                self.score_threshold if score_threshold is None else np.float64(score_threshold)
            )
            selected = np.flatnonzero(scores >= threshold)
            if not selected.size:
                continue
            cells = selected // anchors
            center = np.column_stack((cells % feature_width, cells // feature_width))
            center = center.astype(np.float32) * stride
            distance = np.asarray(outputs[level + 3]).reshape(-1, 4)[selected] * stride
            box = np.column_stack(
                (
                    center[:, 0] - distance[:, 0],
                    center[:, 1] - distance[:, 1],
                    center[:, 0] + distance[:, 2],
                    center[:, 1] + distance[:, 3],
                )
            )
            offsets = np.asarray(outputs[level + 6]).reshape(-1, 5, 2)[selected]
            points = offsets * stride + center[:, None, :]
            all_boxes.append(box)
            all_scores.append(scores[selected])
            all_landmarks.append(points)
        if not all_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
            )
        return np.vstack(all_boxes), np.concatenate(all_scores), np.vstack(all_landmarks)
