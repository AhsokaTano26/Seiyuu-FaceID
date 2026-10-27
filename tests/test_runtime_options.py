from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seiyuu_faceid.runtime.config import load_config
from seiyuu_faceid.runtime.engine import FaceIdEngine
from seiyuu_faceid.vision.detector import SCRFDDetector
from seiyuu_faceid.vision.recognizer import ONNXRecognizer


@pytest.fixture
def stub_model_loading(monkeypatch):
    # Keep real ORT SessionOptions; replace only loading large binary models.
    class Session:
        def __init__(self, path, *, sess_options, providers):
            self.options = sess_options
            self.output_count = 9 if Path(path).name == "det.onnx" else 1

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 112, 112])]

        def get_outputs(self):
            return [SimpleNamespace(name=f"output{index}") for index in range(self.output_count)]

    monkeypatch.setattr("onnxruntime.InferenceSession", Session)


@pytest.mark.parametrize(
    "constructor, name", [(ONNXRecognizer, "face.onnx"), (SCRFDDetector, "det.onnx")]
)
@pytest.mark.parametrize(
    "spinning, arena", [(True, True), (True, False), (False, True), (False, False)]
)
def test_model_constructors_map_resource_options(
    tmp_path, stub_model_loading, constructor, name, spinning, arena
):
    path = tmp_path / name
    path.touch()
    model = constructor(
        path,
        intra_op_threads=3,
        inter_op_threads=2,
        allow_spinning=spinning,
        enable_cpu_mem_arena=arena,
    )
    options = model.session.options
    expected = "1" if spinning else "0"
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == expected
    assert options.get_session_config_entry("session.inter_op.allow_spinning") == expected
    assert options.enable_cpu_mem_arena is arena
    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 2


@pytest.mark.parametrize(
    "constructor, name", [(ONNXRecognizer, "face.onnx"), (SCRFDDetector, "det.onnx")]
)
def test_model_constructors_keep_default_resource_behavior(
    tmp_path, stub_model_loading, constructor, name
):
    path = tmp_path / name
    path.touch()
    options = constructor(path).session.options
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "1"
    assert options.get_session_config_entry("session.inter_op.allow_spinning") == "1"
    assert options.enable_cpu_mem_arena is True


@pytest.mark.parametrize("detector_threads, effective", [(None, 1), (2, 2)])
def test_engine_applies_resource_options_to_detector_and_both_backbones(
    tmp_path, stub_model_loading, detector_threads, effective
):
    bundle = tmp_path / "selected"
    bundle.mkdir()
    for name in ("adaface_ir101.onnx", "w600k_r50.onnx"):
        (bundle / name).write_bytes(name.encode())
    np.savez(
        bundle / "classifier.npz",
        mean=np.zeros(1024),
        scale=np.ones(1024),
        coefficients=np.zeros((2, 1024)),
        intercept=np.zeros(2),
        class_names=np.array(["alice", "bob"]),
    )
    assets = {
        name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        for name in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")
    }
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_order": ["adaface_ir101", "w600k_r50"],
                "feature_dimensions": 1024,
                "assets": assets,
                "classes": 2,
            }
        )
    )
    detector = tmp_path / "det.onnx"
    detector.touch()
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml"),
        detector_model=detector,
        classifier_bundle=bundle,
        allow_spinning=False,
        enable_cpu_mem_arena=False,
        intra_op_threads=1,
        detector_intra_op_threads=detector_threads,
    )
    engine = FaceIdEngine(config)
    try:
        for model in (engine.detector, engine.classifier.adaface, engine.classifier.r50):
            options = model.session.options
            assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
            assert options.get_session_config_entry("session.inter_op.allow_spinning") == "0"
            assert options.enable_cpu_mem_arena is False
        assert engine.detector.session.options.intra_op_num_threads == effective
        assert engine.classifier.adaface.session.options.intra_op_num_threads == 1
        assert engine.classifier.r50.session.options.intra_op_num_threads == 1
    finally:
        engine.close()
