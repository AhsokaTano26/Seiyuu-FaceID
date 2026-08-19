#!/usr/bin/env python3
"""Extract reference embeddings and create a portable local gallery."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from seiyuu_faceid.dataset.metadata import read_jsonl
from seiyuu_faceid.gallery.builder import build_gallery
from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.vision.recognizer import ONNXRecognizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(_bootstrap.ROOT / args.config)
    recognizer = ONNXRecognizer(
        config.recognizer_model,
        intra_op_threads=config.intra_op_threads,
        inter_op_threads=config.inter_op_threads,
    )
    records = read_jsonl(config.root / "data/aligned/metadata.jsonl")
    gallery = build_gallery(records, config.root / "data/aligned", recognizer)
    gallery.save(config.gallery_vectors_path, config.gallery_metadata_path)
    print(
        f"Saved {len(gallery.references)} references "
        f"for {len(gallery.prototype_identities)} identities."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
