"""The selected AdaFace + R50 supervised identity classifier."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

import numpy as np

from seiyuu_faceid.runtime.results import Candidate
from seiyuu_faceid.vision.recognizer import ONNXRecognizer


class LinearIdentityHead:
    """Portable StandardScaler and multinomial logistic regression parameters."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.mean = archive["mean"]
            self.scale = archive["scale"]
            self.coefficients = archive["coefficients"]
            self.intercept = archive["intercept"]
            self.class_names = archive["class_names"].tolist()
        dimensions = self.mean.size
        if (
            self.mean.shape != (dimensions,)
            or self.scale.shape != (dimensions,)
            or self.coefficients.shape != (len(self.class_names), dimensions)
            or self.intercept.shape != (len(self.class_names),)
            or len(self.class_names) < 2
            or len(set(self.class_names)) != len(self.class_names)
            or any(not isinstance(name, str) or not name for name in self.class_names)
            or not all(
                np.isfinite(x).all()
                for x in (self.mean, self.scale, self.coefficients, self.intercept)
            )
            or np.any(self.scale <= 0)
        ):
            raise ValueError("Invalid identity classifier parameters.")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Match sklearn's float32 scaling, then evaluate stable softmax scores."""
        values = np.array(features, dtype=np.float32, copy=True)
        if values.ndim != 2 or values.shape[1] != self.mean.size or not np.isfinite(values).all():
            raise ValueError(f"Expected finite N x {self.mean.size} features.")
        # Separate in-place operations preserve the fitted sklearn pipeline's rounding.
        values -= self.mean
        values /= self.scale
        logits = values @ self.coefficients.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        scores = np.exp(logits)
        return scores / scores.sum(axis=1, keepdims=True)

    def candidates(self, features: np.ndarray, *, top_k: int) -> list[Candidate]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        scores = self.predict_proba(np.asarray(features).reshape(1, -1))[0]
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Candidate(self.class_names[index], float(scores[index])) for index in order]


class IdentityClassifier:
    """Own the two frozen ONNX backbones and the locally trained identity head."""

    def __init__(
        self,
        bundle: Path,
        *,
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
        parallel_backbones: bool = False,
        allow_spinning: bool = True,
        enable_cpu_mem_arena: bool = True,
    ) -> None:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("feature_order") != ["adaface_ir101", "w600k_r50"]
            or manifest.get("feature_dimensions") != 1024
        ):
            raise ValueError("Unsupported identity model bundle.")
        for name in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz"):
            with (bundle / name).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != manifest["assets"][name]:
                raise ValueError(f"Model checksum mismatch: {name}")
        self.head = LinearIdentityHead(bundle / "classifier.npz")
        if self.head.mean.size != 1024 or len(self.head.class_names) != manifest["classes"]:
            raise ValueError("Classifier dimensions do not match the model bundle.")
        self.class_names = self.head.class_names
        self.model_id = hashlib.sha256(
            json.dumps(manifest["assets"], sort_keys=True).encode()
        ).hexdigest()
        options = {
            "intra_op_threads": intra_op_threads,
            "inter_op_threads": inter_op_threads,
            "allow_spinning": allow_spinning,
            "enable_cpu_mem_arena": enable_cpu_mem_arena,
        }
        self.adaface = ONNXRecognizer(bundle / "adaface_ir101.onnx", color_order="bgr", **options)
        self.r50 = ONNXRecognizer(bundle / "w600k_r50.onnx", **options)
        self._lifecycle_lock = Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=2) if parallel_backbones else None

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Concatenate unit embeddings in the exact feature order used for training."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Identity classifier is closed.")
            futures = (
                [
                    self._executor.submit(recognizer.embed, aligned_bgr)
                    for recognizer in (self.adaface, self.r50)
                ]
                if self._executor is not None
                else None
            )
        if futures is not None:
            # Drain both tasks even when one fails, before releasing the caller's image.
            wait(futures)
            embeddings = [future.result() for future in futures]
        else:
            embeddings = [self.adaface.embed(aligned_bgr), self.r50.embed(aligned_bgr)]
        return np.concatenate(embeddings) / np.float32(np.sqrt(2))

    def close(self) -> None:
        """Stop worker threads once and reject subsequent recognition calls."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True)

    def classify(self, aligned_bgr: np.ndarray, *, top_k: int) -> list[Candidate]:
        return self.head.candidates(self.embed(aligned_bgr), top_k=top_k)
