"""Long-lived, single-instance CPU runtime engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seiyuu_faceid.gallery.index import Gallery
from seiyuu_faceid.runtime.config import AppConfig
from seiyuu_faceid.runtime.pipeline import FaceRecognition, recognize_detection, recognize_faces
from seiyuu_faceid.vision.detector import FaceDetection, SCRFDDetector
from seiyuu_faceid.vision.recognizer import ONNXRecognizer


class FaceIdEngine:
    """Owns one detector, recognizer and gallery for repeated local inference."""

    def __init__(
        self,
        config: AppConfig,
        gallery: Gallery | None = None,
        *,
        allow_missing_gallery: bool = False,
    ) -> None:
        self.config = config
        self.detector = SCRFDDetector(
            config.detector_model,
            input_size=config.detector_input_size,
            score_threshold=config.detector_score_threshold,
            nms_threshold=config.detector_nms_threshold,
            max_long_edge=config.detector_max_long_edge,
            intra_op_threads=config.intra_op_threads,
            inter_op_threads=config.inter_op_threads,
        )
        self.recognizer = ONNXRecognizer(
            config.recognizer_model,
            intra_op_threads=config.intra_op_threads,
            inter_op_threads=config.inter_op_threads,
        )
        if gallery is not None:
            self.gallery: Gallery | None = gallery
        else:
            try:
                self.gallery = Gallery.load(
                    config.gallery_vectors_path, config.gallery_metadata_path
                )
            except FileNotFoundError:
                if not allow_missing_gallery:
                    raise
                self.gallery = None

    def recognize_image(self, image_path: Path) -> list[FaceRecognition]:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")
        return self.recognize_array(image)

    def recognize_array(self, image_bgr: np.ndarray) -> list[FaceRecognition]:
        return recognize_faces(
            image_bgr,
            detector=self.detector,
            recognizer=self.recognizer,
            gallery=self._require_gallery(),
            min_face_size=self.config.min_face_size,
            min_sharpness=self.config.min_sharpness,
            top_k=self.config.top_k,
            candidate_search_top_k=self.config.candidate_search_top_k,
            aggregation=self.config.aggregation,
            verification_top_k=self.config.verification_top_k,
            high_threshold=self.config.high_threshold,
            low_threshold=self.config.low_threshold,
        )

    def detect_array(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        """Detect faces for an interactive caller before it chooses one to recognize."""
        return self.detector.detect(image_bgr)

    def recognize_detection(
        self, image_bgr: np.ndarray, detection: FaceDetection
    ) -> FaceRecognition:
        """Recognize one face returned by :meth:`detect_array`."""
        return recognize_detection(
            image_bgr,
            detection=detection,
            recognizer=self.recognizer,
            gallery=self._require_gallery(),
            min_face_size=self.config.min_face_size,
            min_sharpness=self.config.min_sharpness,
            top_k=self.config.top_k,
            candidate_search_top_k=self.config.candidate_search_top_k,
            aggregation=self.config.aggregation,
            verification_top_k=self.config.verification_top_k,
            high_threshold=self.config.high_threshold,
            low_threshold=self.config.low_threshold,
        )

    def _require_gallery(self) -> Gallery:
        if self.gallery is None:
            raise RuntimeError("Gallery is not built yet. Build it before running recognition.")
        return self.gallery
