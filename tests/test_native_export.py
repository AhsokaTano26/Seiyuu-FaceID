"""Native export preserves the selected head and rejects corrupt source models."""

import hashlib
import json

import numpy as np
import pytest
import yaml

from tools.export_native_model import export_native_model


@pytest.fixture
def native_source(tmp_path):
    root = tmp_path / "source"
    bundle = root / "models/selected"
    bundle.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "models/det_500m.onnx").write_bytes(b"detector")
    (bundle / "adaface_ir101.onnx").write_bytes(b"adaface")
    (bundle / "w600k_r50.onnx").write_bytes(b"r50")
    parameters = {
        "mean": np.linspace(-0.12345678901234, 0.2, 1024, dtype=np.float64),
        "scale": np.linspace(0.000123456789, 0.9, 1024, dtype=np.float64),
        "coefficients": np.linspace(-1, 1, 2048, dtype=np.float32).reshape(2, 1024),
        "intercept": np.array([0.125, -0.33333333333], dtype=np.float32),
        "class_names": np.array(["声优甲", "声优乙"]),
    }

    def write_head(values):
        np.savez(bundle / "classifier.npz", **values)
        assets = {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")
        }
        manifest = {
            "schema_version": 1,
            "feature_order": ["adaface_ir101", "w600k_r50"],
            "feature_dimensions": 1024,
            "classes": 2,
            "assets": assets,
            "detector": {"sha256": hashlib.sha256(b"detector").hexdigest()},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    manifest = write_head(parameters)
    settings = {
        "models": {"detector": "models/det_500m.onnx", "classifier": "models/selected"},
        "detector": {"input_size": [640, 640], "max_long_edge": 1920,
                     "score_threshold": 0.5, "nms_threshold": 0.4},
        "quality": {"min_face_size": 48, "min_sharpness": 20.0},
        "recognition": {"top_k": 2, "probability_threshold": 0.95},
        "runtime": {"intra_op_threads": 1, "detector_intra_op_threads": 2,
                    "inter_op_threads": 1, "parallel_backbones": True,
                    "allow_spinning": False, "enable_cpu_mem_arena": True},
    }
    config = root / "configs/n150.yaml"
    config.write_text(yaml.safe_dump(settings))
    return root, bundle, config, parameters, manifest, write_head


def test_export_roundtrips_all_parameters_and_model_identity(native_source):
    _, bundle, config, parameters, manifest, _ = native_source
    output = export_native_model(config)
    assert output == bundle / "native-model.json"
    exported = json.loads(output.read_text())
    checksum = output.with_name(output.name + ".sha256").read_text().split()
    assert checksum == [hashlib.sha256(output.read_bytes()).hexdigest(), output.name]
    assert exported["schema_version"] == 1
    assert exported["model_id"] == hashlib.sha256(
        json.dumps(manifest["assets"], sort_keys=True).encode()
    ).hexdigest()
    assert exported["source_assets"] == manifest["assets"]
    assert exported["detector_sha256"] == manifest["detector"]["sha256"]
    assert exported["models"] == {
        "detector": "../det_500m.onnx", "adaface": "adaface_ir101.onnx",
        "r50": "w600k_r50.onnx",
    }
    assert exported["class_names"] == parameters["class_names"].tolist()
    for name in ("mean", "scale", "coefficients", "intercept"):
        np.testing.assert_array_equal(
            np.asarray(exported["head"][name], dtype=parameters[name].dtype), parameters[name]
        )
    assert exported["runtime"]["detector_intra_op_threads"] == 2
    assert exported["runtime"]["allow_spinning"] is False
    assert not any(str(bundle) in value for value in exported["models"].values())


@pytest.mark.parametrize("asset", ["models/det_500m.onnx", "models/selected/adaface_ir101.onnx",
                                    "models/selected/classifier.npz"])
def test_export_rejects_corrupt_assets_without_partial_output(native_source, asset):
    root, bundle, config, *_ = native_source
    (root / asset).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        export_native_model(config)
    assert not (bundle / "native-model.json").exists()


@pytest.mark.parametrize("change", ["scale_zero", "nonfinite", "wrong_dtype", "wrong_shape",
                                     "duplicate_names"])
def test_export_rejects_invalid_head(native_source, change):
    _, bundle, config, parameters, _, write_head = native_source
    if change == "scale_zero":
        parameters["scale"][0] = 0
    elif change == "nonfinite":
        parameters["coefficients"][0, 0] = np.nan
    elif change == "wrong_dtype":
        parameters["coefficients"] = parameters["coefficients"].astype(np.float64)
    elif change == "wrong_shape":
        parameters["mean"] = parameters["mean"][:-1]
    else:
        parameters["class_names"][:] = "duplicate"
    write_head(parameters)
    with pytest.raises(ValueError, match="head|parameters|dtype"):
        export_native_model(config)
    assert not (bundle / "native-model.json").exists()


def test_export_defaults_resolve_detector_thread_inheritance(native_source):
    _, _, config, *_ = native_source
    settings = yaml.safe_load(config.read_text())
    settings["runtime"] = {"intra_op_threads": 3, "inter_op_threads": 1}
    config.write_text(yaml.safe_dump(settings))
    exported = json.loads(export_native_model(config).read_text())
    assert exported["runtime"] == {
        "intra_op_threads": 3, "detector_intra_op_threads": 3, "inter_op_threads": 1,
        "parallel_backbones": False, "allow_spinning": True, "enable_cpu_mem_arena": True,
    }


def test_export_explicit_flush_denormals_preserves_models_and_head(native_source):
    _, bundle, config, *_ = native_source
    baseline = json.loads(export_native_model(config).read_text())
    output = export_native_model(config, bundle / "native-ftz.json", flush_denormals=True)
    candidate = json.loads(output.read_text())
    assert candidate["runtime"].pop("flush_denormals") is True
    assert candidate == baseline
    assert output.with_name(output.name + ".sha256").read_text().split() == [
        hashlib.sha256(output.read_bytes()).hexdigest(), output.name,
    ]


def test_export_refuses_overwrite_and_unrelated_output_directory(native_source):
    root, bundle, config, *_ = native_source
    output = bundle / "native-model.json"
    output.write_text("existing model")
    with pytest.raises(FileExistsError):
        export_native_model(config)
    assert output.read_text() == "existing model"
    with pytest.raises(ValueError, match="beside"):
        export_native_model(config, root / "unrelated.json")


def test_export_refuses_existing_checksum_file(native_source):
    _, bundle, config, *_ = native_source
    checksum = bundle / "native-model.json.sha256"
    checksum.write_text("existing checksum")
    with pytest.raises(FileExistsError):
        export_native_model(config)
    assert checksum.read_text() == "existing checksum"
    assert not (bundle / "native-model.json").exists()
