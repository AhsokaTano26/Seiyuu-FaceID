#!/usr/bin/env python3
"""Compare sequential and concurrent frozen backbones without changing the runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import _bootstrap  # noqa: F401
import numpy as np

from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.vision.classifier import IdentityClassifier


def select_indices(rows, development, *, count, seed):
    groups = {}
    for index in sorted(development):
        groups.setdefault(rows[index]["source_group"], int(index))
    if count > len(groups):
        raise ValueError("Requested more samples than development source groups.")
    keys = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(keys))[:count]
    return [groups[keys[position]] for position in order]


def parallel_embed(classifier, image, executor):
    adaface = executor.submit(classifier.adaface.embed, image)
    r50 = executor.submit(classifier.r50.embed, image)
    return np.concatenate([adaface.result(), r50.result()]) / np.float32(np.sqrt(2))


def benchmark_modes(classifier, images, *, warmups, runs):
    timings = {"sequential": [], "parallel": []}
    features = {mode: [] for mode in timings}
    with ThreadPoolExecutor(max_workers=2) as executor:
        methods = {
            "sequential": classifier.embed,
            "parallel": lambda image: parallel_embed(classifier, image, executor),
        }
        for _ in range(warmups):
            for method in methods.values():
                method(images[0])
        for run in range(runs):
            for position, image in enumerate(images):
                modes = list(methods) if (run + position) % 2 == 0 else list(reversed(methods))
                for mode in modes:
                    started = time.perf_counter()
                    result = methods[mode](image)
                    timings[mode].append((time.perf_counter() - started) * 1000)
                    features[mode].append(result)
    sequential, parallel = (np.asarray(features[mode]) for mode in timings)
    scores = [classifier.head.predict_proba(values) for values in (sequential, parallel)]
    report = {
        mode: {
            "observations": len(values),
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
        for mode, values in timings.items()
    }
    report["speedup_mean"] = report["sequential"]["mean_ms"] / report["parallel"]["mean_ms"]
    report["parity"] = {
        "embedding_exact": bool(np.array_equal(sequential, parallel)),
        "embedding_max_abs_difference": float(np.max(np.abs(sequential - parallel))),
        "embedding_allclose": bool(np.allclose(sequential, parallel, rtol=1e-5, atol=1e-6)),
        "probability_exact": bool(np.array_equal(*scores)),
        "probability_max_abs_difference": float(np.max(np.abs(scores[0] - scores[1]))),
        "probability_allclose": bool(np.allclose(*scores, rtol=1e-5, atol=1e-6)),
        "top1_matches": int((scores[0].argmax(1) == scores[1].argmax(1)).sum()),
        "predictions": len(sequential),
        "rtol": 1e-5,
        "atol": 1e-6,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.threads + [args.samples, args.runs]) < 1 or args.warmups < 1:
        parser.error("Threads, samples, runs and warmups must be positive.")
    config = load_config(_bootstrap.ROOT / args.config)
    output = (config.root / args.output).resolve()
    if not output.is_relative_to((config.root / "data/experiments").resolve()):
        parser.error("Output must be under data/experiments/.")
    if output.exists():
        raise FileExistsError("Choose a new benchmark report path.")
    training = config.root / "data/training"
    rows = [json.loads(line) for line in (training / "samples.jsonl").read_text().splitlines()]
    with np.load(training / "features_and_splits.npz", allow_pickle=False) as archive:
        development = np.r_[archive["train_indices"], archive["validation_indices"]]
        if set(development) & set(archive["test_indices"]):
            raise ValueError("Development and historical test indices overlap.")
        indices = select_indices(rows, development, count=args.samples, seed=args.seed)
    import cv2

    images = []
    for index in indices:
        path = config.root / rows[index]["crop_image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != rows[index]["crop_sha256"]:
            raise ValueError("Benchmark crop checksum mismatch.")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Could not decode benchmark crop.")
        images.append(image)
    report = {
        "schema_version": 1,
        "scope": "Aligned crop embedding only; head parity checked outside timing.",
        "limitations": "Local sampled crops; no throughput, detection or new accuracy estimate.",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": version("onnxruntime"),
        "samples": args.samples,
        "runs": args.runs,
        "warmups_per_mode": args.warmups,
        "seed": args.seed,
        "subset_sha256": hashlib.sha256(
            "\n".join(rows[i]["sample_id"] for i in indices).encode()
        ).hexdigest(),
        "order": "Alternate sequential/parallel first by run plus crop position.",
        "executor_workers": 2,
        "inter_op_threads": config.inter_op_threads,
        "results": [],
    }
    for threads in dict.fromkeys(args.threads):
        classifier = IdentityClassifier(
            config.classifier_bundle, intra_op_threads=threads,
            inter_op_threads=config.inter_op_threads,
        )
        report["model_id"] = classifier.model_id
        result = benchmark_modes(classifier, images, warmups=args.warmups, runs=args.runs)
        report["results"].append({"intra_op_threads": threads, **result})
        del classifier
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2)
    with output.open("x") as handle:
        handle.write(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
