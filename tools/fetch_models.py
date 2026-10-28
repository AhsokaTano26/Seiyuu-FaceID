#!/usr/bin/env python3
"""Download and install the exact public model allowlist; uses only Python's stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_FILES = frozenset(
    {
        "models/det_500m.onnx",
        "models/selected/adaface_ir101.onnx",
        "models/selected/w600k_r50.onnx",
        "models/selected/classifier.npz",
        "models/selected/manifest.json",
        "models/selected/MODEL_CARD.md",
        "models/selected/ADAFACE_LICENSE",
        "models/MODEL_LICENSES.md",
        "models/TRAINING_SAMPLES.csv",
        "models/SHA256SUMS",
    }
)
OPTIONAL_FILES = frozenset({"models/TRAINING_PROVENANCE.json"})


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_source(path: Path) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    url = urlparse(source["url"])
    files = source["files"]
    if (
        source.get("schema_version") != 1
        or url.scheme != "https"
        or url.netloc != "github.com"
        or "/releases/download/" not in url.path
        or not REQUIRED_FILES <= files.keys()
        or not files.keys() <= REQUIRED_FILES | OPTIONAL_FILES
    ):
        raise ValueError("Unsupported or unsafe model source.")
    for entry in [source, *files.values()]:
        value = entry["sha256"]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid model source checksum.")
    if any(type(v["size"]) is not int or not 0 <= v["size"] <= 1024**3 for v in files.values()):
        raise ValueError("Invalid model member size.")
    return source


def verify_files(root: Path, source: dict) -> None:
    for name, expected in source["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected["sha256"]:
            raise ValueError(f"Model member checksum mismatch: {name}")


def fetch_models(root: Path, source_path: Path, *, archive: Path | None = None) -> None:
    """Validate a complete archive before copying; never replace conflicting local files."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = load_source(source_path)
    with tempfile.TemporaryDirectory(prefix=".model-download-", dir=root) as temporary:
        stage = Path(temporary)
        if archive is None:
            archive = stage / "download.tar.gz"
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "seiyuu-faceid-models"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                archive.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
        if sha256(archive) != source["sha256"]:
            raise ValueError("Model archive checksum mismatch.")
        seen = set()
        allowed_dirs = {"models", "models/selected"}
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.isdir() and member.name in allowed_dirs:
                    continue
                if (
                    not member.isfile()
                    or member.name not in source["files"]
                    or member.name in seen
                    or member.size != source["files"][member.name]["size"]
                ):
                    raise ValueError(f"Unsafe or unexpected model archive member: {member.name}")
                seen.add(member.name)
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as handle, target.open("wb") as output:
                    shutil.copyfileobj(handle, output)
        if seen != set(source["files"]):
            raise ValueError("Model archive is missing required files.")
        verify_files(stage, source)
        # Preflight every destination before installing any member.
        for name, expected in source["files"].items():
            path = root / name
            for parent in [path, *path.parents]:
                if parent == root:
                    break
                if parent.is_symlink():
                    raise ValueError(f"Existing symlink in model destination: {name}")
            if path.exists() and (not path.is_file() or sha256(path) != expected["sha256"]):
                raise ValueError(f"Existing model differs; choose a clean destination: {name}")
        for name in source["files"]:
            path = root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / name, path)
                path.chmod(0o644)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded archive offline")
    args = parser.parse_args()
    try:
        fetch_models(
            args.root, args.source or args.root / "release/model-source.json", archive=args.archive
        )
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, f"Could not install models: {error}\n")
    print("Verified model files are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
