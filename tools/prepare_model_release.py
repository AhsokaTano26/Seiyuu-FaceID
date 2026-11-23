#!/usr/bin/env python3
"""Verify a published model Release and pin its inputs for deployment builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

if __package__:
    from tools.fetch_models import OPTIONAL_FILES, REQUIRED_FILES, fetch_models, sha256
    from tools.package_models import ASSET_NAMES, validate_training
    from tools.publish_models import GitHubReleases
else:
    from fetch_models import OPTIONAL_FILES, REQUIRED_FILES, fetch_models, sha256
    from package_models import ASSET_NAMES, validate_training
    from publish_models import GitHubReleases


def release_assets(release: dict, repository: str) -> dict:
    tag = release["tag_name"]
    assets = release["assets"]
    if (
        not re.fullmatch(r"models-[0-9a-f]{16}", tag)
        or release["draft"]
        or len(assets) != len(ASSET_NAMES)
        or {a["name"] for a in assets} != ASSET_NAMES
    ):
        raise ValueError("Expected a published, content-addressed, model-only Release.")
    for asset in assets:
        limit = 2 * 1024**3 if asset["name"].endswith(".tar.gz") else 1024**2
        if (
            asset.get("state") != "uploaded"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", asset.get("digest") or "")
            or type(asset.get("size")) is not int
            or not 0 < asset["size"] <= limit
            or asset.get("browser_download_url")
            != f"https://github.com/{repository}/releases/download/{tag}/{asset['name']}"
        ):
            raise ValueError("Invalid model Release asset metadata.")
    return {a["name"]: a for a in assets}


def prepare_release(root: Path, directory: Path, release: dict, *, repository: str) -> None:
    assets = release_assets(release, repository)
    for name, asset in assets.items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != asset["size"]
            or "sha256:" + sha256(path) != asset["digest"]
        ):
            raise ValueError(f"Release asset checksum mismatch: {name}")
    checksums = (directory / "SHA256SUMS").read_text("utf-8").splitlines()
    expected = {f"{a['digest'][7:]}  {n}" for n, a in assets.items() if n != "SHA256SUMS"}
    if len(checksums) != len(expected) or set(checksums) != expected:
        raise ValueError("Release SHA256SUMS does not match the model-only assets.")
    archive = directory / "seiyuu-faceid-models.tar.gz"
    archive_hash = sha256(archive)
    if release["tag_name"] != "models-" + archive_hash[:16]:
        raise ValueError("Model tag does not match the archive content digest.")
    files = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isdir() and member.name in {"models", "models/selected"}:
                continue
            if (
                not member.isfile()
                or member.name not in REQUIRED_FILES | OPTIONAL_FILES
                or member.name in files
                or not 0 <= member.size <= 1024**3
            ):
                raise ValueError(f"Unsafe or unexpected model archive member: {member.name}")
            with tar.extractfile(member) as handle:
                files[member.name] = {
                    "size": member.size,
                    "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
                }
    if set(files) != REQUIRED_FILES | OPTIONAL_FILES:
        raise ValueError("Model Release must include the model allowlist and training provenance.")
    source = {
        "schema_version": 1,
        "url": assets[archive.name]["browser_download_url"],
        "sha256": archive_hash,
        "files": files,
    }
    # Validate in isolation before replacing either deployment input in the checkout.
    with tempfile.TemporaryDirectory(prefix="model-release-verify-") as temporary:
        stage = Path(temporary)
        inputs = stage / "release"
        inputs.mkdir()
        (inputs / "model-source.json").write_text(json.dumps(source, indent=2) + "\n")
        shutil.copyfile(directory / "TRAINING_PROVENANCE.json", inputs / "training.json")
        fetch_models(stage, inputs / "model-source.json", archive=archive)
        for name in ("TRAINING_SAMPLES.csv", "TRAINING_PROVENANCE.json", "MODEL_LICENSES.md"):
            if (stage / "models" / name).read_bytes() != (directory / name).read_bytes():
                raise ValueError(f"Outer model metadata differs from archive: {name}")
        validate_training(stage)
        target = root / "release"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("model-source.json", "training.json"):
            shutil.copyfile(inputs / name, target / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"models-[0-9a-f]{16}", args.tag):
        parser.error("Expected models- followed by 16 lowercase hexadecimal characters.")
    service = GitHubReleases(
        os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GH_TOKEN", "")
    )
    release = service.get(args.tag)
    if release is None or release["tag_name"] != args.tag:
        parser.error("Published model Release not found.")
    assets = release_assets(release, service.repository)
    with tempfile.TemporaryDirectory(prefix="model-release-download-") as temporary:
        directory = Path(temporary)
        for name, asset in assets.items():
            request = urllib.request.Request(
                asset["browser_download_url"], headers={"User-Agent": "seiyuu-faceid-models"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with (directory / name).open("wb") as output:
                    remaining = asset["size"]
                    while chunk := response.read(min(1024**2, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ValueError("Model download exceeds its declared size.")
                        output.write(chunk)
        prepare_release(args.root, directory, release, repository=service.repository)
    print(f"Verified deployment model inputs: {args.tag}")


if __name__ == "__main__":
    main()
