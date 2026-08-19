"""Typed runtime configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    root: Path
    detector_model: Path
    recognizer_model: Path
    detector_input_size: tuple[int, int]
    detector_score_threshold: float
    detector_nms_threshold: float
    detector_max_long_edge: int
    min_face_size: int
    min_sharpness: float
    gallery_vectors_path: Path
    gallery_metadata_path: Path
    top_k: int
    candidate_search_top_k: int
    aggregation: str
    verification_top_k: int
    high_threshold: float | None
    low_threshold: float | None
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
        gallery = data["gallery"]
        runtime = data["runtime"]
        input_size = tuple(int(value) for value in detector["input_size"])
        if len(input_size) != 2:
            raise ValueError("detector.input_size must have width and height.")
        high = recognition.get("high_threshold")
        low = recognition.get("low_threshold")
        return AppConfig(
            root=root,
            detector_model=root / str(models["detector"]),
            recognizer_model=root / str(models["recognizer"]),
            detector_input_size=(input_size[0], input_size[1]),
            detector_score_threshold=float(detector["score_threshold"]),
            detector_nms_threshold=float(detector["nms_threshold"]),
            detector_max_long_edge=int(detector["max_long_edge"]),
            min_face_size=int(quality["min_face_size"]),
            min_sharpness=float(quality["min_sharpness"]),
            gallery_vectors_path=root / str(gallery["path"]),
            gallery_metadata_path=root / str(gallery["metadata_path"]),
            top_k=int(recognition["top_k"]),
            candidate_search_top_k=int(recognition.get("candidate_search_top_k", 16)),
            aggregation=str(recognition["aggregation"]),
            verification_top_k=int(recognition["verification_top_k"]),
            high_threshold=None if high is None else float(high),
            low_threshold=None if low is None else float(low),
            intra_op_threads=int(runtime["intra_op_threads"]),
            inter_op_threads=int(runtime["inter_op_threads"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid config {path}: {error}") from error
