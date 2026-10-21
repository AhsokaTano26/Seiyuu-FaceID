"""Typed runtime configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    root: Path
    detector_model: Path
    classifier_bundle: Path
    detector_input_size: tuple[int, int]
    detector_score_threshold: float
    detector_nms_threshold: float
    detector_max_long_edge: int
    min_face_size: int
    min_sharpness: float
    top_k: int
    probability_threshold: float
    intra_op_threads: int
    inter_op_threads: int


def load_config(path: Path) -> AppConfig:
    """Load default-style YAML and resolve all user paths from the repository root."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    try:
        root = path.resolve().parent.parent
        models = data["models"]
        detector = data["detector"]
        quality = data["quality"]
        recognition = data["recognition"]
        runtime = data["runtime"]
        input_size = tuple(int(value) for value in detector["input_size"])
        if len(input_size) != 2:
            raise ValueError("detector.input_size must have width and height.")
        if "high_threshold" in recognition or "low_threshold" in recognition:
            raise ValueError("Similarity thresholds are obsolete for the identity classifier.")
        config = AppConfig(
            root=root,
            detector_model=root / str(models["detector"]),
            classifier_bundle=root / str(models["classifier"]),
            detector_input_size=(input_size[0], input_size[1]),
            detector_score_threshold=float(detector["score_threshold"]),
            detector_nms_threshold=float(detector["nms_threshold"]),
            detector_max_long_edge=int(detector["max_long_edge"]),
            min_face_size=int(quality["min_face_size"]),
            min_sharpness=float(quality["min_sharpness"]),
            top_k=int(recognition["top_k"]),
            probability_threshold=float(recognition.get("probability_threshold", 0.95)),
            intra_op_threads=int(runtime["intra_op_threads"]),
            inter_op_threads=int(runtime["inter_op_threads"]),
        )
        _validate_config(config)
        return config
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid config {path}: {error}") from error


def _validate_config(config: AppConfig) -> None:
    """Reject invalid runtime settings before loading large ONNX models."""
    if any(value < 1 for value in config.detector_input_size):
        raise ValueError("detector.input_size values must be positive.")
    if config.detector_max_long_edge < 1:
        raise ValueError("detector.max_long_edge must be positive.")
    if not 0 < config.detector_score_threshold < 1:
        raise ValueError("detector.score_threshold must be between 0 and 1.")
    if not 0 < config.detector_nms_threshold <= 1:
        raise ValueError("detector.nms_threshold must be in (0, 1].")
    if config.min_face_size < 1 or config.min_sharpness < 0:
        raise ValueError("quality thresholds must be non-negative, with min_face_size positive.")
    if config.top_k < 1:
        raise ValueError("recognition.top_k must be positive.")
    if not 0 <= config.probability_threshold <= 1:
        raise ValueError("recognition.probability_threshold must be in [0, 1].")
    if config.intra_op_threads < 1 or config.inter_op_threads < 1:
        raise ValueError("runtime thread counts must be positive.")
