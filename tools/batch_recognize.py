#!/usr/bin/env python3
"""Recognize images recursively and write one local JSON record per image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from seiyuu_faceid.dataset.scanner import IMAGE_SUFFIXES
from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.runtime.engine import FaceIdEngine
from seiyuu_faceid.runtime.pipeline import serialize_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("recognition_results.jsonl"))
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    directory = (_bootstrap.ROOT / args.directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")
    config = load_config(_bootstrap.ROOT / args.config)
    engine = FaceIdEngine(config)
    output = (_bootstrap.ROOT / args.output).resolve()
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for image_path in sorted(directory.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                results = engine.recognize_image(image_path)
                value: dict[str, object] = {
                    "image": str(image_path),
                    "faces": [serialize_result(result) for result in results],
                }
            except Exception as error:  # Preserve the batch and make every failure inspectable.
                value = {"image": str(image_path), "error": str(error)}
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} image records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
