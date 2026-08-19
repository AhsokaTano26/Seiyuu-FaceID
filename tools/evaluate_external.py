#!/usr/bin/env python3
"""Evaluate an existing Gallery against independent labelled image folders.

The default input is ``data/gallery/<identity>/...``.  It is intentionally
read-only: all embeddings, per-image records and reports are written below
``data/runtime/evaluation``.  Images with zero or multiple faces are excluded
from this automatic baseline to avoid treating an unreviewed group face as the
folder identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from seiyuu_faceid.dataset.metadata import write_jsonl
from seiyuu_faceid.dataset.scanner import scan_raw_dataset
from seiyuu_faceid.gallery.index import Gallery
from seiyuu_faceid.gallery.scoring import classify_candidates, search_gallery
from seiyuu_faceid.runtime.config import AppConfig, load_config
from seiyuu_faceid.runtime.engine import FaceIdEngine
from seiyuu_faceid.validation import evaluate_identification
from seiyuu_faceid.vision.alignment import AlignmentError, align_face
from seiyuu_faceid.vision.quality import assess_face_quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--evaluation-root",
        default="data/gallery",
        help="Independent labelled image root; it is never written.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/runtime/evaluation",
        help="Directory for generated evaluation artifacts.",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse external_queries.npz in output-dir rather than detecting again.",
    )
    args = parser.parse_args()
    config = load_config(_bootstrap.ROOT / args.config)
    evaluation_root = (_bootstrap.ROOT / args.evaluation_root).resolve()
    output_dir = (_bootstrap.ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "external_queries.npz"

    engine = FaceIdEngine(config)
    if args.reuse_embeddings:
        if not embeddings_path.is_file():
            parser.error(f"No reusable embeddings found: {embeddings_path}")
        embeddings, expected, sources = _load_embeddings(embeddings_path)
        print(f"Reused {len(expected)} accepted external-face embeddings.")
    else:
        embeddings, expected, sources = _extract_embeddings(
            engine, evaluation_root, output_dir / "external_records.jsonl"
        )
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            expected=np.asarray(expected, dtype=str),
            sources=np.asarray(sources, dtype=str),
        )

    if not len(expected):
        raise RuntimeError("No eligible single-face evaluation images were found.")
    assert engine.gallery is not None
    rankings: list[list[str]] = []
    scores: list[float] = []
    predictions: list[dict[str, object]] = []
    for embedding, identity, source in zip(embeddings, expected, sources, strict=True):
        candidates = search_gallery(
            engine.gallery,
            embedding,
            top_k=config.top_k,
            candidate_search_top_k=config.candidate_search_top_k,
            aggregation=config.aggregation,
            verification_top_k=config.verification_top_k,
        )
        rankings.append([candidate.identity for candidate in candidates])
        scores.append(candidates[0].score)
        decision = classify_candidates(
            candidates, config.high_threshold, config.low_threshold
        )
        predictions.append(
            {
                "source_image": source,
                "expected_identity": identity,
                "top1_correct": candidates[0].identity == identity,
                "status": decision.status,
                "identity": decision.identity,
                "candidates": [
                    {
                        "identity": candidate.identity,
                        "score": candidate.score,
                        "prototype_score": candidate.prototype_score,
                        "reference_score": candidate.reference_score,
                    }
                    for candidate in candidates
                ],
            }
        )
    write_jsonl(output_dir / "external_predictions.jsonl", predictions)
    report = {
        "evaluation_root": str(evaluation_root),
        "gallery_vectors": str(config.gallery_vectors_path),
        "scoring": {
            "aggregation": config.aggregation,
            "verification_top_k": config.verification_top_k,
            "candidate_top_k": config.top_k,
            "candidate_search_top_k": config.candidate_search_top_k,
        },
        "identification": evaluate_identification(expected, rankings, np.asarray(scores)),
        "open_set_proxy": _evaluate_open_set_proxy(engine.gallery, embeddings, expected, config),
    }
    report_path = output_dir / "external_baseline.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = report["identification"]
    print(f"External baseline: {metrics['samples']} eligible faces")
    print(f"Top-1: {metrics['top1_accuracy']:.2%}; Top-5: {metrics['top5_accuracy']:.2%}")
    print(json.dumps(metrics["threshold_advice"], ensure_ascii=False))
    print(f"Report: {report_path}")
    return 0


def _extract_embeddings(
    engine: FaceIdEngine, evaluation_root: Path, records_path: Path
) -> tuple[np.ndarray, list[str], list[str]]:
    import cv2

    vectors: list[np.ndarray] = []
    expected: list[str] = []
    sources: list[str] = []
    records: list[dict[str, object]] = []
    images = scan_raw_dataset(evaluation_root)
    for index, source in enumerate(images, start=1):
        image = cv2.imread(str(source.path))
        record: dict[str, object] = {
            "source_image": source.relative_path.as_posix(),
            "expected_identity": source.identity,
        }
        if image is None:
            record["reason"] = "decode_failed"
        else:
            detections = engine.detect_array(image)
            record["detected_faces"] = len(detections)
            if len(detections) != 1:
                record["reason"] = (
                    "selection_required" if len(detections) > 1 else "no_face_detected"
                )
            else:
                quality = assess_face_quality(
                    image,
                    detections[0].bbox,
                    min_face_size=engine.config.min_face_size,
                    min_sharpness=engine.config.min_sharpness,
                )
                record["quality_score"] = quality.score
                if not quality.accepted:
                    record["reason"] = "quality_rejected"
                    record["quality_reasons"] = list(quality.reasons)
                else:
                    try:
                        aligned = align_face(image, detections[0].landmarks)
                    except AlignmentError:
                        record["reason"] = "alignment_failed"
                    else:
                        vectors.append(engine.recognizer.embed(aligned))
                        expected.append(source.identity)
                        sources.append(source.relative_path.as_posix())
                        record["accepted"] = True
        records.append(record)
        if index % 100 == 0 or index == len(images):
            print(f"External evaluation extraction: {index}/{len(images)} images")
    write_jsonl(records_path, records)
    dimensions = engine.gallery.embedding_dim if engine.gallery is not None else 512
    return np.asarray(vectors, dtype=np.float32).reshape(-1, dimensions), expected, sources


def _load_embeddings(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with np.load(path) as values:
        return (
            np.asarray(values["embeddings"], dtype=np.float32),
            [str(value) for value in values["expected"]],
            [str(value) for value in values["sources"]],
        )


def _evaluate_open_set_proxy(
    gallery: Gallery, embeddings: np.ndarray, expected: list[str], config: AppConfig
) -> dict[str, object]:
    """Measure unknown rejection by hiding each query's correct identity.

    This is a proxy rather than a replacement for real unknown-person images,
    but it is materially better than calibrating only on closed-set Top-1.
    """
    excluded: dict[str, Gallery] = {}
    for identity in sorted(set(expected)):
        keep = np.asarray([item != identity for item in gallery.reference_identities], dtype=bool)
        identities = [
            item
            for item, is_kept in zip(gallery.reference_identities, keep, strict=True)
            if is_kept
        ]
        metadata = [
            item for item, is_kept in zip(gallery.reference_metadata, keep, strict=True) if is_kept
        ]
        excluded[identity] = Gallery.from_references(gallery.references[keep], identities, metadata)
    scores = np.asarray(
        [
            search_gallery(
                excluded[identity],
                embedding,
                top_k=config.top_k,
                candidate_search_top_k=config.candidate_search_top_k,
                aggregation=config.aggregation,
                verification_top_k=config.verification_top_k,
            )[0].score
            for embedding, identity in zip(embeddings, expected, strict=True)
        ],
        dtype=np.float32,
    )
    return {
        "definition": "Each query is scored after removing its true identity from the Gallery.",
        "samples": int(len(scores)),
        "score_quantiles": {
            "p50": float(np.quantile(scores, 0.50)),
            "p80": float(np.quantile(scores, 0.80)),
            "p95": float(np.quantile(scores, 0.95)),
            "p99": float(np.quantile(scores, 0.99)),
            "p99_5": float(np.quantile(scores, 0.995)),
        },
        "at_configured_thresholds": {
            "false_recognized_rate": float((scores >= config.high_threshold).mean()),
            "review_rate": float(
                ((scores >= config.low_threshold) & (scores < config.high_threshold)).mean()
            ),
            "unknown_rate": float((scores < config.low_threshold).mean()),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
