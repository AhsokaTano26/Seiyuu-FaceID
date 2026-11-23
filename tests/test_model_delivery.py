"""Release downloads and artifacts bind exact model bytes without private data."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def delivery_source(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    head = io.BytesIO()
    np.savez(head, class_names=np.array(["甲", "乙"]))
    files = {
        "models/det_500m.onnx": b"detector",
        "models/selected/adaface_ir101.onnx": b"adaface",
        "models/selected/w600k_r50.onnx": b"r50",
        "models/selected/classifier.npz": head.getvalue(),
        "models/selected/ADAFACE_LICENSE": b"MIT notice",
        "models/selected/MODEL_CARD.md": b"model scope",
        "models/MODEL_LICENSES.md": b"upstream model terms",
        "models/TRAINING_SAMPLES.csv": "identity,training_samples\n甲,2\n乙,1\n".encode(),
    }
    manifest = {
        "schema_version": 1,
        "classes": 2,
        "training_faces": 3,
        "assets": {
            Path(p).name: digest(v)
            for p, v in files.items()
            if p.endswith(("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz"))
        },
        "detector": {"sha256": digest(b"detector")},
    }
    files["models/selected/manifest.json"] = json.dumps(manifest).encode()
    files["models/SHA256SUMS"] = b"legacy checksum list\n"
    archive = tmp_path / "source.tar.gz"
    write_tar(archive, files)
    source = {
        "schema_version": 1,
        "url": "https://github.com/example/project/releases/download/v1/models.tar.gz",
        "sha256": digest(archive.read_bytes()),
        "files": {p: {"sha256": digest(v), "size": len(v)} for p, v in files.items()},
    }
    (root / "release").mkdir()
    (root / "release/model-source.json").write_text(json.dumps(source))
    training = {
        "schema_version": 1,
        "classifier_sha256": digest(head.getvalue()),
        "features_and_splits_sha256": "a" * 64,
        "fit_splits": ["train", "validation"],
        "split_sizes": {"train": 2, "validation": 1, "test_excluded": 1},
        "count_unit": "aligned_face_crop",
        "identities": [
            {"name": "甲", "training_samples": 2},
            {"name": "乙", "training_samples": 1},
        ],
    }
    (root / "release/training.json").write_text(json.dumps(training))
    return root, archive, source, files


def write_tar(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_fetch_only_installs_verified_models_and_preserves_native_config(delivery_source):
    from tools.fetch_models import fetch_models

    root, archive, _, files = delivery_source
    native = root / "models/selected/native-model.json"
    native.parent.mkdir(parents=True)
    native.write_text("keep native configuration")
    fetch_models(root, root / "release/model-source.json", archive=archive)
    for name, data in files.items():
        assert (root / name).read_bytes() == data
    assert native.read_text() == "keep native configuration"
    fetch_models(root, root / "release/model-source.json", archive=archive)


def test_fetch_rejects_corruption_without_installing_anything(delivery_source):
    from tools.fetch_models import fetch_models

    root, archive, _, _ = delivery_source
    archive.write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="checksum"):
        fetch_models(root, root / "release/model-source.json", archive=archive)
    assert not (root / "models").exists()


@pytest.mark.parametrize("attack", ["traversal", "extra_file", "symlink", "wrong_member_hash"])
def test_fetch_rejects_unsafe_or_unexpected_members(delivery_source, attack):
    from tools.fetch_models import fetch_models

    root, archive, source, files = delivery_source
    if attack == "traversal":
        files["../outside"] = b"attack"
    elif attack == "extra_file":
        files["models/selected/holdout_scores.npz"] = b"private"
    elif attack == "wrong_member_hash":
        files["models/det_500m.onnx"] = b"bad data"
    write_tar(archive, files)
    if attack == "symlink":
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("models/det_500m.onnx")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            tar.addfile(info)
    source["sha256"] = digest(archive.read_bytes())
    (root / "release/model-source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError):
        fetch_models(root, root / "release/model-source.json", archive=archive)
    assert not (root / "models").exists()
    assert not (root.parent / "outside").exists()


def test_fetch_refuses_to_overwrite_existing_different_weights(delivery_source):
    from tools.fetch_models import fetch_models

    root, archive, _, _ = delivery_source
    detector = root / "models/det_500m.onnx"
    detector.parent.mkdir()
    detector.write_bytes(b"existing model")
    with pytest.raises(ValueError, match="existing|Existing"):
        fetch_models(root, root / "release/model-source.json", archive=archive)
    assert detector.read_bytes() == b"existing model"
    assert not (root / "models/selected").exists()


def test_model_release_is_reproducible_model_only_and_lists_every_identity(delivery_source):
    from tools.fetch_models import fetch_models
    from tools.package_models import package_models

    root, archive, _, _ = delivery_source
    fetch_models(root, root / "release/model-source.json", archive=archive)
    (root / "models/selected/holdout_scores.npz").write_bytes(b"private evaluation")
    (root / "models/selected/native-model.json").write_bytes(b"deployment settings")
    first = package_models(root, root / "first")
    second = package_models(root, root / "second")
    assert first == second
    assert first["tag"].startswith("models-")
    names = {p.name for p in (root / "first").iterdir()}
    assert names == {
        "seiyuu-faceid-models.tar.gz",
        "SHA256SUMS",
        "TRAINING_SAMPLES.csv",
        "TRAINING_PROVENANCE.json",
        "MODEL_LICENSES.md",
    }
    assert (root / "first/seiyuu-faceid-models.tar.gz").read_bytes() == (
        root / "second/seiyuu-faceid-models.tar.gz"
    ).read_bytes()
    with tarfile.open(root / "first/seiyuu-faceid-models.tar.gz") as tar:
        members = tar.getnames()
        assert all(name.startswith("models/") for name in members)
        assert not any("holdout" in name or "native-model" in name for name in members)
    notes = first["notes"]
    assert (
        f"https://github.com/example/project/releases/download/{first['tag']}/"
        "seiyuu-faceid-models.tar.gz"
    ) in notes
    assert "甲 | 2" in notes and "乙 | 1" in notes
    assert "测试集" in notes and "3" in notes
    assert "Docker" not in notes and "systemd" not in notes and "API" not in notes
    for line in (root / "first/SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ")
        assert digest((root / "first" / filename).read_bytes()) == expected


@pytest.mark.parametrize("change", ["labels", "count", "classifier", "corrupt_weights"])
def test_model_release_rejects_mismatched_training_or_assets(delivery_source, change):
    from tools.fetch_models import fetch_models
    from tools.package_models import package_models

    root, archive, _, _ = delivery_source
    fetch_models(root, root / "release/model-source.json", archive=archive)
    path = root / "release/training.json"
    training = json.loads(path.read_text())
    if change == "labels":
        training["identities"].reverse()
    elif change == "count":
        training["identities"][0]["training_samples"] += 1
    elif change == "classifier":
        training["classifier_sha256"] = "0" * 64
    else:
        (root / "models/det_500m.onnx").write_bytes(b"corrupt")
    path.write_text(json.dumps(training))
    with pytest.raises(ValueError):
        package_models(root, root / "release-output")
    assert not (root / "release-output").exists()


def test_training_metadata_counts_only_fit_rows_and_binds_classifier(tmp_path):
    from tools.export_training_metadata import export_training_metadata

    source = tmp_path / "features.npz"
    np.savez(
        source,
        labels=np.array([0, 1, 0, 1]),
        class_names=np.array(["甲", "乙"]),
        train_indices=np.array([0, 1]),
        validation_indices=np.array([2]),
        test_indices=np.array([3]),
    )
    head = tmp_path / "classifier.npz"
    np.savez(head, class_names=np.array(["甲", "乙"]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "classes": 2,
                "training_faces": 3,
                "assets": {"classifier.npz": digest(head.read_bytes())},
            }
        )
    )
    result = export_training_metadata(source, head, manifest)
    assert result["identities"] == [
        {"name": "甲", "training_samples": 2},
        {"name": "乙", "training_samples": 1},
    ]
    assert result["split_sizes"] == {"train": 2, "validation": 1, "test_excluded": 1}
    assert result["classifier_sha256"] == digest(head.read_bytes())
    assert "features" not in result and "labels" not in result
    np.savez(
        source,
        labels=np.array([0, 1, 0, 1]),
        class_names=np.array(["甲", "乙"]),
        train_indices=np.array([0, 1]),
        validation_indices=np.array([2]),
        test_indices=np.array([2, 3]),
    )
    with pytest.raises(ValueError, match="split|overlap"):
        export_training_metadata(source, head, manifest)


@pytest.fixture
def published_models(delivery_source):
    from tools.fetch_models import fetch_models
    from tools.package_models import package_models

    root, archive, _, _ = delivery_source
    fetch_models(root, root / "release/model-source.json", archive=archive)
    directory = root / "published"
    metadata = package_models(root, directory)
    release = {
        "tag_name": metadata["tag"],
        "draft": False,
        "assets": [
            {
                "name": p.name,
                "state": "uploaded",
                "size": p.stat().st_size,
                "digest": "sha256:" + digest(p.read_bytes()),
                "browser_download_url": (
                    f"https://github.com/example/project/releases/download/{metadata['tag']}/{p.name}"
                ),
            }
            for p in directory.iterdir()
        ],
    }
    return directory, release


def test_release_event_pins_the_published_model_for_deployment(published_models, tmp_path):
    from tools.fetch_models import fetch_models, load_source
    from tools.prepare_model_release import prepare_release

    directory, release = published_models
    target = tmp_path / "checkout"
    prepare_release(target, directory, release, repository="example/project")
    source_path = target / "release/model-source.json"
    source = load_source(source_path)
    assert source["url"].endswith(f"/{release['tag_name']}/seiyuu-faceid-models.tar.gz")
    assert source["sha256"] == digest((directory / "seiyuu-faceid-models.tar.gz").read_bytes())
    assert not (target / "models").exists()
    fetch_models(target, source_path, archive=directory / "seiyuu-faceid-models.tar.gz")
    from tools.package_models import validate_training

    assert validate_training(target)["identities"] == [
        {"name": "甲", "training_samples": 2},
        {"name": "乙", "training_samples": 1},
    ]


@pytest.mark.parametrize(
    "change", ["digest", "tag", "draft", "extra", "unsafe_member", "training", "metadata"]
)
def test_bad_published_models_never_replace_deployment_inputs(published_models, tmp_path, change):
    from tools.prepare_model_release import prepare_release

    directory, release = published_models
    if change == "digest":
        (directory / "TRAINING_SAMPLES.csv").write_text("corrupt")
    elif change == "tag":
        release["tag_name"] = "models-" + "0" * 16
    elif change == "draft":
        release["draft"] = True
    elif change == "extra":
        release["assets"].append({"name": "deployment.tar.gz"})
    else:
        archive_path = directory / "seiyuu-faceid-models.tar.gz"
        with tarfile.open(archive_path) as tar:
            files = {member.name: tar.extractfile(member).read() for member in tar}
        if change == "unsafe_member":
            files["../outside"] = b"attack"
        elif change == "training":
            training = json.loads(files["models/TRAINING_PROVENANCE.json"])
            training["identities"][0]["training_samples"] += 1
            data = json.dumps(training).encode()
            files["models/TRAINING_PROVENANCE.json"] = data
            (directory / "TRAINING_PROVENANCE.json").write_bytes(data)
        else:
            (directory / "MODEL_LICENSES.md").write_text("different outer metadata")
        write_tar(archive_path, files)
        release["tag_name"] = "models-" + digest(archive_path.read_bytes())[:16]
        (directory / "SHA256SUMS").write_text(
            "".join(
                f"{digest(p.read_bytes())}  {p.name}\n"
                for p in sorted(directory.iterdir())
                if p.name != "SHA256SUMS"
            )
        )
        for asset in release["assets"]:
            path = directory / asset["name"]
            asset.update(size=path.stat().st_size, digest="sha256:" + digest(path.read_bytes()))
            asset["browser_download_url"] = (
                f"https://github.com/example/project/releases/download/{release['tag_name']}/{path.name}"
            )
    target = tmp_path / "checkout"
    (target / "release").mkdir(parents=True)
    (target / "release/model-source.json").write_text("original source")
    with pytest.raises(ValueError):
        prepare_release(target, directory, release, repository="example/project")
    assert (target / "release/model-source.json").read_text() == "original source"
    assert not (target / "models").exists()
    assert not (tmp_path / "outside").exists()
