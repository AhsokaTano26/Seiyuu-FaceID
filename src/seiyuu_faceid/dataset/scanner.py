"""Scan a user-owned raw dataset without changing any source image."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class RawImage:
    identity: str
    path: Path
    relative_path: Path
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_raw_dataset(raw_root: Path) -> list[RawImage]:
    """Return sorted image records where the immediate directory is the identity key."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")
    images: list[RawImage] = []
    for identity_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for path in sorted(identity_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(
                    RawImage(
                        identity=identity_dir.name,
                        path=path,
                        relative_path=path.relative_to(raw_root),
                        sha256=file_sha256(path),
                    )
                )
    return images
