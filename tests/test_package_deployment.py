"""Deployment bundles must carry local models without leaking the training workspace."""

import hashlib
import json

import pytest

from tools.package_deployment import create_bundle


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "source"
    assets = {
        "models/det_500m.onnx": b"detector",
        "models/selected/adaface_ir101.onnx": b"adaface",
        "models/selected/w600k_r50.onnx": b"r50",
        "models/selected/classifier.npz": b"classifier",
        "models/selected/MODEL_CARD.md": b"model provenance",
        "models/selected/ADAFACE_LICENSE": b"upstream license",
        "configs/default.yaml": b"models:\n  detector: models/det_500m.onnx\n",
        "compose.yaml": b"services: {}\n",
        "deploy/.env.example": b"SEIYUU_FACEID_API_KEY=\n",
        "deploy/README.md": b"Offline deployment instructions\n",
        "docs/API.md": b"API instructions\n",
        "docs/DEPLOY_LINUX.md": b"Linux deployment instructions\n",
        "LICENSE": b"project license",
        "images/private.jpg": b"private photo",
        "data/training/features.npz": b"private features",
        "models/selected/holdout_scores.npz": b"private evaluation",
        ".env": b"SEIYUU_FACEID_API_KEY=private-secret",
    }
    for name, content in assets.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {
        "assets": {
            name: hashlib.sha256(assets[f"models/selected/{name}"]).hexdigest()
            for name in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")
        },
        "detector": {"sha256": hashlib.sha256(b"detector").hexdigest()},
    }
    (root / "models/selected/manifest.json").write_text(json.dumps(manifest))
    image = tmp_path / "linux-image.tar"
    image.write_bytes(b"prebuilt Linux image")
    return root, image, tmp_path / "bundle"


def test_bundle_is_complete_and_excludes_private_workspace(workspace):
    root, image, output = workspace
    create_bundle(root, image, output)
    names = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert names == {
        "image.tar", "compose.yaml", ".env.example", "README.md", "LICENSE", "SHA256SUMS",
        "docs/API.md", "docs/DEPLOY_LINUX.md", "configs/default.yaml", "models/det_500m.onnx",
        "models/selected/manifest.json", "models/selected/classifier.npz",
        "models/selected/adaface_ir101.onnx", "models/selected/w600k_r50.onnx",
        "models/selected/ADAFACE_LICENSE", "models/selected/MODEL_CARD.md",
    }
    assert (output / "image.tar").read_bytes() == b"prebuilt Linux image"
    assert (output / "models/det_500m.onnx").read_bytes() == b"detector"
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert len((output / "SHA256SUMS").read_text().splitlines()) == len(names) - 1


@pytest.mark.parametrize("asset", ["det_500m.onnx", "selected/classifier.npz"])
def test_rejects_corrupted_weights_before_creating_bundle(workspace, asset):
    root, image, output = workspace
    (root / "models" / asset).write_bytes(b"corrupted model")
    with pytest.raises(ValueError, match="checksum"):
        create_bundle(root, image, output)
    assert not output.exists()


def test_refuses_to_overwrite_existing_bundle(workspace):
    root, image, output = workspace
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing deployment")
    with pytest.raises(FileExistsError):
        create_bundle(root, image, output)
    assert sentinel.read_text() == "existing deployment"


def test_missing_image_does_not_leave_a_partial_bundle(workspace):
    root, image, output = workspace
    image.unlink()
    with pytest.raises(FileNotFoundError):
        create_bundle(root, image, output)
    assert not output.exists()


def test_can_package_models_for_a_registry_image_without_copying_an_image_tar(workspace):
    root, _, output = workspace
    create_bundle(root, None, output)
    assert (output / "models/selected/classifier.npz").read_bytes() == b"classifier"
    assert (output / "compose.yaml").is_file()
    assert not (output / "image.tar").exists()
    assert "image.tar" not in (output / "SHA256SUMS").read_text()
