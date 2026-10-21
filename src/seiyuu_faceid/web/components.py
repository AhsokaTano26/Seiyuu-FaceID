"""Reusable visual components for the local Streamlit interface."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import streamlit as st

from seiyuu_faceid.vision.detector import FaceDetection

# OpenCV uses BGR. Selection and review intentionally remain distinct.
FACE_BOX_COLORS = {
    "detected": (145, 145, 145),
    "selected": (190, 75, 200),
    "review": (0, 220, 255),
    "recognized": (0, 210, 0),
    "unknown": (0, 0, 255),
}
FACE_BOX_LABELS = {
    "selected": "已选",
    "review": "复核",
    "recognized": "已知",
    "unknown": "未知",
}


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode an uploaded image in memory without persisting it."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def annotate_detections(
    image_bgr: np.ndarray,
    detections: list[FaceDetection],
    selected_indexes: set[int] | None,
    recognition_statuses: dict[int, str] | None = None,
    recognition_names: dict[int, str] | None = None,
    coordinate_scale: float = 1.0,
) -> np.ndarray:
    """Draw legible boxes and labels using the shared recognition colour rules."""
    annotated = image_bgr.copy()
    selected = selected_indexes or set()
    statuses = recognition_statuses or {}
    names = recognition_names or {}
    for index, detection in enumerate(detections):
        number = index + 1
        x1, y1, x2, y2 = np.round(detection.bbox * coordinate_scale).astype(int)
        status = statuses.get(index)
        state = (
            "selected" if index in selected else status if status in FACE_BOX_COLORS else "detected"
        )
        color = FACE_BOX_COLORS[state]
        labels = [f"Face {number}"]
        if name := names.get(index):
            labels.append(name)
        elif state_label := FACE_BOX_LABELS.get(state):
            labels.append(state_label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if state != "detected" else 2)
        metrics = [cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1) for label in labels]
        label_width = max(width for (width, _), _ in metrics)
        label_height = sum(height + baseline for (_, height), baseline in metrics) + 8
        label_top = max(0, y1 - label_height - 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + label_width + 8, label_top + label_height + 4),
            color,
            cv2.FILLED,
        )
        text_y = label_top + 4
        for label, ((_, height), baseline) in zip(labels, metrics, strict=True):
            text_y += height
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            text_y += baseline + 2
    return annotated


def resize_for_display(image_bgr: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Bound preview pixels while leaving inference on the original image."""
    height, width = image_bgr.shape[:2]
    if max(height, width) <= max_long_edge:
        return image_bgr
    scale = max_long_edge / max(height, width)
    return cv2.resize(
        image_bgr,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def resize_for_height(image_bgr: np.ndarray, max_height: int) -> np.ndarray:
    """Constrain display height without distorting the image."""
    height, width = image_bgr.shape[:2]
    if height <= max_height:
        return image_bgr
    scale = max_height / height
    return cv2.resize(image_bgr, (round(width * scale), max_height), interpolation=cv2.INTER_AREA)


def detection_preview(
    image_bgr: np.ndarray,
    detections: list[FaceDetection],
    selected_indexes: set[int] | None,
    *,
    max_long_edge: int = 640,
    recognition_statuses: dict[int, str] | None = None,
    recognition_names: dict[int, str] | None = None,
    max_height: int | None = None,
) -> np.ndarray:
    """Resize before drawing so face boxes remain readable at display size."""
    display = (
        resize_for_height(image_bgr, max_height)
        if max_height is not None
        else resize_for_display(image_bgr, max_long_edge)
    )
    return annotate_detections(
        display,
        detections,
        selected_indexes,
        recognition_statuses,
        recognition_names,
        coordinate_scale=display.shape[1] / image_bgr.shape[1],
    )


def render_group_review_image(image_bgr: np.ndarray) -> None:
    """Render a centred review image that fits the browser viewport."""
    success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        st.error("无法生成审核预览。")
        return
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
          .group-review-image {{
            align-items: center;
            display: flex;
            height: calc(100vh - 165px);
            justify-content: center;
            overflow: hidden;
          }}
          .group-review-image img {{
            height: auto;
            max-height: 100%;
            max-width: 100%;
            object-fit: contain;
            width: auto;
          }}
        </style>
        <div class="group-review-image">
          <img alt="群像审核预览" src="data:image/jpeg;base64,{payload}" />
        </div>
        """,
        unsafe_allow_html=True,
    )
