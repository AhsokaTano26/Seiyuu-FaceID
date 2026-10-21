"""Long-lived CPU detector and the selected supervised identity classifier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from seiyuu_faceid.runtime.config import AppConfig
from seiyuu_faceid.runtime.pipeline import FaceRecognition, recognize_detection, recognize_faces
from seiyuu_faceid.vision.classifier import IdentityClassifier
from seiyuu_faceid.vision.detector import FaceDetection, SCRFDDetector


class FaceIdEngine:
    """Load the selected model once for repeated local inference."""

    def __init__(self, config: AppConfig) -> None:
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
        self.classifier = IdentityClassifier(
            config.classifier_bundle,
            intra_op_threads=config.intra_op_threads,
            inter_op_threads=config.inter_op_threads,
        )
        self.recognition_id = hashlib.sha256(
            (
                self.classifier.model_id + json.dumps(asdict(config), sort_keys=True, default=str)
            ).encode()
        ).hexdigest()

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
            classifier=self.classifier,
            min_face_size=self.config.min_face_size,
            min_sharpness=self.config.min_sharpness,
            top_k=self.config.top_k,
            probability_threshold=self.config.probability_threshold,
        )

    def detect_array(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        return self.detector.detect(image_bgr)

    def recognize_detection(
        self, image_bgr: np.ndarray, detection: FaceDetection
    ) -> FaceRecognition:
        return recognize_detection(
            image_bgr,
            detection=detection,
            classifier=self.classifier,
            min_face_size=self.config.min_face_size,
            min_sharpness=self.config.min_sharpness,
            top_k=self.config.top_k,
            probability_threshold=self.config.probability_threshold,
        )
