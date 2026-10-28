#!/usr/bin/env python3
"""Prepare the selected model for the C++ runtime; Python is not needed to serve it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

if __package__:
    from tools import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.vision.classifier import LinearIdentityHead

ASSETS = ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")


def _check_digest(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Model checksum mismatch: {path.name}")


def export_native_model(
    config_path: Path, output: Path | None = None, *, flush_denormals: bool = False,
) -> Path:
    """Export exact model parameters beside the source bundle, without overwriting."""
    config = load_config(Path(config_path))
    bundle = config.classifier_bundle.resolve()
    output = (Path(output) if output is not None else bundle / "native-model.json").resolve()
    checksum_output = output.with_name(output.name + ".sha256")
    if output.parent != bundle:
        raise ValueError("The native model JSON must be beside the source classifier bundle.")
    if output.exists():
        raise FileExistsError(output)
    if checksum_output.exists():
        raise FileExistsError(checksum_output)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("feature_order") != ["adaface_ir101", "w600k_r50"]
        or manifest.get("feature_dimensions") != 1024
        or set(manifest.get("assets", {})) != set(ASSETS)
    ):
        raise ValueError("Unsupported source model manifest.")
    assets = manifest["assets"]
    for name in ASSETS:
        _check_digest(bundle / name, assets[name])
    detector_digest = manifest.get("detector", {}).get("sha256")
    _check_digest(config.detector_model, detector_digest)
    head = LinearIdentityHead(bundle / "classifier.npz")
    if (
        head.mean.shape != (1024,)
        or len(head.class_names) != manifest.get("classes")
        or config.top_k > len(head.class_names)
    ):
        raise ValueError("Invalid native head dimensions or requested top_k.")
    for name, expected in (("mean", np.float64), ("scale", np.float64),
                           ("coefficients", np.float32), ("intercept", np.float32)):
        if getattr(head, name).dtype != expected:
            raise ValueError(f"Native head {name} dtype must be {np.dtype(expected)}.")
    if not math.isfinite(config.min_sharpness):
        raise ValueError("Native quality threshold must be finite.")
    document = {
        "schema_version": 1,
        "model_id": hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(),
        "source_assets": assets,
        "detector_sha256": detector_digest,
        "models": {
            "detector": Path(os.path.relpath(config.detector_model.resolve(), bundle)).as_posix(),
            "adaface": "adaface_ir101.onnx",
            "r50": "w600k_r50.onnx",
        },
        "class_names": head.class_names,
        "head": {name: getattr(head, name).tolist()
                 for name in ("mean", "scale", "coefficients", "intercept")},
        "runtime": {
            "intra_op_threads": config.intra_op_threads,
            "detector_intra_op_threads": config.effective_detector_threads,
            "inter_op_threads": config.inter_op_threads,
            "parallel_backbones": config.parallel_backbones,
            "allow_spinning": config.allow_spinning,
            "enable_cpu_mem_arena": config.enable_cpu_mem_arena,
        },
        "detector": {
            "input_size": list(config.detector_input_size),
            "max_long_edge": config.detector_max_long_edge,
            "score_threshold": config.detector_score_threshold,
            "nms_threshold": config.detector_nms_threshold,
        },
        "quality": {"min_face_size": config.min_face_size,
                    "min_sharpness": config.min_sharpness},
        "recognition": {"top_k": config.top_k,
                        "probability_threshold": config.probability_threshold},
    }
    if flush_denormals:
        document["runtime"]["flush_denormals"] = True
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    try:
        with checksum_output.open("x", encoding="utf-8") as handle:
            handle.write(f"{checksum}  {output.name}\n")
    except OSError:
        output.unlink()
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/n150.yaml"))
    parser.add_argument("--output", type=Path,
                        help="Output beside classifier.npz; defaults to native-model.json.")
    parser.add_argument("--flush-denormals", action="store_true",
                        help="Enable validated FTZ/DAZ inference on Linux x86_64 (e.g. N150).")
    args = parser.parse_args()
    try:
        output = export_native_model(args.config, args.output, flush_denormals=args.flush_denormals)
    except (ValueError, OSError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Exported native model metadata and head: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
