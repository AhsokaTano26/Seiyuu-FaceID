#!/usr/bin/env python3
"""Compare target CPU resources in fresh processes, without private training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import _bootstrap  # noqa: F401

PROFILES = [
    ("baseline2seqspinOn", 2, False, True, True),
    ("current2parspinOn", 2, True, True, True),
    ("efficient1parOff", 1, True, False, True),
    ("parallel2Off", 2, True, False, True),
    ("sequential2Off", 2, False, False, True),
    ("sequential4Off", 4, False, False, True),
    ("efficient1parOffNoArena", 1, True, False, False),
    ("parallel2OffNoArena", 2, True, False, False),
]


def summarize(wall_ms, cpu_ms):
    def metrics(values):
        ordered = sorted(values)
        rank = (len(ordered) - 1) * 0.95
        low, high = math.floor(rank), math.ceil(rank)
        return {"mean": statistics.mean(values), "median": statistics.median(values),
                "p95": ordered[low] + (ordered[high] - ordered[low]) * (rank - low)}

    return {"observations": len(wall_ms), "wall_ms": metrics(wall_ms),
            "cpu_ms": metrics(cpu_ms), "cpu_core_equivalents": sum(cpu_ms) / sum(wall_ms)}


def rss_mib(value, system):
    return value / (1024 * 1024 if system == "Darwin" else 1024)


def compare_parity(reference, candidate, threshold):
    import numpy as np

    result = {"rtol": 1e-5, "atol": 1e-6}
    for key, label in (("embeddings", "embedding"), ("probabilities", "probability")):
        left, right = np.asarray(reference[key]), np.asarray(candidate[key])
        if left.shape != right.shape or not np.isfinite(right).all():
            raise ValueError("Parity arrays have incompatible shapes or nonfinite values.")
        result.update({f"{label}_exact": bool(np.array_equal(left, right)),
                       f"{label}_allclose": bool(np.allclose(left, right, rtol=1e-5, atol=1e-6)),
                       f"{label}_max_abs_difference": float(np.max(np.abs(left - right)))})
    left, right = (np.asarray(values["probabilities"]) for values in (reference, candidate))
    left_order, right_order = (np.argsort(-x, axis=1, kind="stable") for x in (left, right))
    result.update({"predictions": len(left),
                   "top1_matches": int(np.sum(left_order[:, 0] == right_order[:, 0])),
                   "top5_matches": int(np.sum(np.all(left_order[:, :5] == right_order[:, :5],
                                                      axis=1))),
                   "threshold_matches": int(np.sum((left.max(1) >= threshold)
                                                    == (right.max(1) >= threshold)))})
    return result


def load_inputs(spec):
    # OpenCV checks dimensions before allocating decoded pixels; enforce a crop-sized limit.
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(112 * 112)
    import cv2
    import numpy as np

    if spec["synthetic"]:
        rng = np.random.default_rng(spec["seed"])
        images = [rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
                  for _ in range(spec["samples"])]
    else:
        paths = sorted(path for path in Path(spec["crop_dir"]).iterdir()
                       if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        images = []
        for path in paths[:spec["samples"]]:
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Crop file exceeds the 16 MiB input limit.")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None or image.shape != (112, 112, 3):
                raise ValueError("Every crop must decode to exactly 112 x 112 BGR pixels.")
            images.append(image)
        if not images:
            raise ValueError("Crop directory contains no PNG/JPEG images.")
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.tobytes())
    return images, digest.hexdigest()


def run_worker(spec):
    images, corpus_hash = load_inputs(spec)
    from seiyuu_faceid.runtime.config import load_config
    from seiyuu_faceid.runtime.engine import FaceIdEngine

    profile = spec["profile"]
    config = replace(load_config(Path(spec["config"])),
                     **{key: value for key, value in profile.items() if key != "name"})
    profile = {**profile, "detector_intra_op_threads": config.effective_detector_threads}
    started, cpu_started = time.perf_counter_ns(), time.process_time_ns()
    engine = FaceIdEngine(config)
    load_wall = (time.perf_counter_ns() - started) / 1e6
    load_cpu = (time.process_time_ns() - cpu_started) / 1e6
    try:
        classifier = engine.classifier

        def request(image):
            embedding = classifier.embed(image)
            return classifier.head.predict_proba(embedding.reshape(1, -1))

        for _ in range(spec["warmups"]):
            request(images[0])
        started, cpu_started = time.perf_counter_ns(), time.process_time_ns()
        time.sleep(spec["idle_seconds"])
        idle_wall = (time.perf_counter_ns() - started) / 1e6
        idle_cpu = (time.process_time_ns() - cpu_started) / 1e6
        wall_ms, cpu_ms = [], []
        for _ in range(spec["runs"]):
            for image in images:
                started, cpu_started = time.perf_counter_ns(), time.process_time_ns()
                request(image)
                cpu_ms.append((time.process_time_ns() - cpu_started) / 1e6)
                wall_ms.append((time.perf_counter_ns() - started) / 1e6)
        # Capture before allocating retained parity arrays or serializing the worker response.
        peak = rss_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, platform.system())
        embeddings = [classifier.embed(image) for image in images]
        probabilities = [classifier.head.predict_proba(value.reshape(1, -1))[0]
                         for value in embeddings]
        cpu_info = Path("/proc/cpuinfo")
        cpu_model = platform.processor() or platform.machine()
        if cpu_info.exists():
            cpu_model = next((line.split(":", 1)[1].strip()
                              for line in cpu_info.read_text().splitlines()
                              if line.startswith("model name")), cpu_model)
        return {"profile": profile, "repeat": spec["repeat"], "pid": os.getpid(),
                "model_id": classifier.model_id, "corpus_sha256": corpus_hash,
                "samples": len(images), "classes": len(classifier.class_names),
                "probability_threshold": config.probability_threshold,
                "platform": platform.platform(), "cpu_model": cpu_model,
                "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                "versions": {name: version(name) for name in ("onnxruntime", "numpy")},
                "model_load_wall_ms": load_wall, "model_load_cpu_ms": load_cpu,
                "idle_wall_ms": idle_wall, "idle_cpu_ms": idle_cpu,
                "idle_cpu_core_equivalents": idle_cpu / idle_wall,
                "peak_rss_mib": peak, **summarize(wall_ms, cpu_ms),
                "embeddings": [value.tolist() for value in embeddings],
                "probabilities": [value.tolist() for value in probabilities]}
    finally:
        engine.close()


def run_children(spec):
    results = []
    for repeat in range(spec["repeats"]):
        offset = repeat % len(PROFILES)
        for name, threads, parallel, spin, arena in PROFILES[offset:] + PROFILES[:offset]:
            profile = {"name": name, "intra_op_threads": threads, "inter_op_threads": 1,
                       "parallel_backbones": parallel, "allow_spinning": spin,
                       "enable_cpu_mem_arena": arena}
            try:
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                       input=json.dumps({**spec, "profile": profile,
                                                         "repeat": repeat}),
                                       capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or "").strip() or "No worker error details."
                message = f"Worker {name}, repeat {repeat} failed: {detail.splitlines()[-1][:1000]}"
                raise RuntimeError(message) from None
            results.append(json.loads(child.stdout))
    return results


def build_report(spec, results):
    reference = next(row for row in results if row["profile"]["name"] == PROFILES[0][0])
    metadata_keys = ("model_id", "corpus_sha256", "samples", "classes", "probability_threshold",
                     "platform", "cpu_model", "logical_cpus", "python", "versions")
    metadata = {key: reference[key] for key in metadata_keys}
    for result in results:
        if any(result[key] != reference[key] for key in metadata_keys):
            raise ValueError("Worker models, inputs, or environment changed during benchmarking.")
        result["parity"] = compare_parity(reference, result, reference["probability_threshold"])
    # Compare every result before dropping the baseline arrays.
    for result in results:
        result.pop("embeddings")
        result.pop("probabilities")
        for key in metadata_keys:
            result.pop(key)
    summaries = []
    for profile in PROFILES:
        rows = [row for row in results if row["profile"]["name"] == profile[0]]
        summaries.append({"profile": profile[0], "median_wall_mean_ms": statistics.median(
            row["wall_ms"]["mean"] for row in rows), "median_cpu_mean_ms": statistics.median(
            row["cpu_ms"]["mean"] for row in rows), "median_peak_rss_mib": statistics.median(
            row["peak_rss_mib"] for row in rows)})
    return {**metadata, "schema_version": 1, "parameters": {key: spec[key] for key in
            ("samples", "runs", "warmups", "repeats", "idle_seconds", "seed")},
            "input_kind": "synthetic" if spec["synthetic"] else "local_aligned_crops",
            "scope": "112x112 aligned crop embedding + classifier head; detector loaded, not run",
            "rss_scope": "Fresh process startup through timing, before retained parity outputs",
            "cpu_scope": "Process CPU time across all threads; 1 core equivalent = 100% of one CPU",
            "limitations": ("Crop parity is not accuracy evaluation or full-image latency. "
                            "Timers add overhead; idle is immediately after warmup."),
            "order": "Rotate profiles each repeat; fresh child per profile and repeat",
            "summaries": summaries, "results": results}


def main():
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(run_worker(json.load(sys.stdin)), allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--crop-dir")
    for name, default in (("samples", 8), ("runs", 5), ("warmups", 2), ("repeats", 2),
                          ("seed", 20261027)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--idle-seconds", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 64 or min(args.runs, args.warmups, args.repeats) < 1:
        parser.error("samples must be 1..64; runs, warmups, repeats must be positive.")
    if not math.isfinite(args.idle_seconds) or not 0 <= args.idle_seconds <= 5 or args.seed < 0:
        parser.error("idle-seconds must be finite and 0..5; seed must be nonnegative.")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output already exists; choose a new filepath.")
    spec = vars(args).copy()
    spec["config"] = str(Path(args.config).resolve())
    if args.crop_dir:
        spec["crop_dir"] = str(Path(args.crop_dir).resolve())
    try:
        results = run_children(spec)
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    report = {**build_report(spec, results), "provider": "CPUExecutionProvider"}
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
