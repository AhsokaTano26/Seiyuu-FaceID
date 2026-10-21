"""ONNX ArcFace-style embedding inference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class RecognizerError(RuntimeError):
    """Raised for an incompatible or unavailable recognition model."""


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """Return a float32 unit vector, rejecting zero vectors early."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= 1e-12:
        raise RecognizerError("Recognition model returned a zero-norm embedding.")
    return vector / norm


class ONNXRecognizer:
    """512-D ONNX embedding inference with explicit model-specific channel order."""

    def __init__(
        self,
        model_path: Path,
        *,
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
        color_order: str = "rgb",
    ):
        if color_order not in {"rgb", "bgr"}:
            raise ValueError("color_order must be rgb or bgr.")
        self.color_order = color_order
        # Must be set before importing ORT; see ONNX Runtime's privacy guidance.
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        import onnxruntime as ort

        if not model_path.is_file():
            raise FileNotFoundError(f"Recognizer model not found: {model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_names = [output.name for output in self.session.get_outputs()]
        shape = input_meta.shape
        if len(shape) != 4 or shape[1] not in (3, "3"):
            raise RecognizerError(f"Expected NCHW three-channel recognizer input, got {shape}.")
        self.height = int(shape[2]) if isinstance(shape[2], int) else 112
        self.width = int(shape[3]) if isinstance(shape[3], int) else 112
        if len(self.output_names) != 1:
            raise RecognizerError("Recognizer must expose exactly one embedding output.")

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Embed and L2-normalize an aligned face with ArcFace preprocessing."""
        import cv2

        if aligned_bgr.ndim != 3 or aligned_bgr.shape[2] != 3:
            raise RecognizerError("Expected a BGR image with three channels.")
        resized = cv2.resize(aligned_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.color_order == "bgr":
            pixels = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        else:
            pixels = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        blob = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, ...])
        output = self.session.run(self.output_names, {self.input_name: blob})[0]
        embedding = l2_normalize(output[0])
        if embedding.size != 512:
            raise RecognizerError(f"Expected a 512-D embedding, got {embedding.size} dimensions.")
        return embedding
