"""Package local inference assets, optionally including a Docker image for offline deployment."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def create_bundle(root: Path, image_tar: Path | None, output: Path) -> None:
    """Copy an explicit allowlist, verify weights, and publish the bundle atomically."""
    root, output = root.resolve(), output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite deployment directory: {output}")
    files = {
        "compose.yaml": root / "compose.yaml",
        ".env.example": root / "deploy/.env.example",
        "README.md": root / "deploy/README.md",
        "docs/API.md": root / "docs/API.md",
        "docs/DEPLOY_LINUX.md": root / "docs/DEPLOY_LINUX.md",
        "LICENSE": root / "LICENSE",
        "configs/default.yaml": root / "configs/default.yaml",
        "models/det_500m.onnx": root / "models/det_500m.onnx",
        "models/selected/manifest.json": root / "models/selected/manifest.json",
    }
    if image_tar is not None:
        files["image.tar"] = image_tar.resolve()
    for name in (*MODEL_FILES, "MODEL_CARD.md", "ADAFACE_LICENSE"):
        files[f"models/selected/{name}"] = root / "models/selected" / name
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required deployment file is missing: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".faceid-package-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o755)
        for name, source in files.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o644)
        manifest = json.loads((stage / "models/selected/manifest.json").read_text("utf-8"))
        expected = {f"models/selected/{name}": manifest["assets"][name] for name in MODEL_FILES}
        expected["models/det_500m.onnx"] = manifest["detector"]["sha256"]
        digests = {name: sha256(stage / name) for name in sorted(files)}
        for name, digest in expected.items():
            if digests[name] != digest:
                raise ValueError(f"Model checksum mismatch: {name}")
        (stage / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in digests.items()), encoding="utf-8"
        )
        # Bind-mounted models must be readable by the container's unprivileged user,
        # even when the machine creating this bundle has a restrictive umask.
        for path in stage.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        stage.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-tar", type=Path, help="Optional image from docker image save for offline startup"
    )
    parser.add_argument("--output", type=Path, required=True, help="New deployment directory")
    args = parser.parse_args()
    try:
        create_bundle(ROOT, args.image_tar, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Could not package deployment: {error}\n")
    print(f"Deployment bundle: {args.output.resolve()}")


if __name__ == "__main__":
    main()
