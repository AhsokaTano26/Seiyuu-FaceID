from __future__ import annotations

import pytest

from seiyuu_faceid.runtime.config import load_config


def write_config(path, *, top_k=5, extra=""):
    path.parent.mkdir(parents=True)
    path.write_text(f"""models: {{detector: models/det.onnx, classifier: models/selected}}
detector: {{max_long_edge: 640, input_size: [640, 640], score_threshold: 0.5,
  nms_threshold: 0.4}}
quality: {{min_face_size: 48, min_sharpness: 20}}
recognition: {{top_k: {top_k}{extra}}}
runtime: {{intra_op_threads: 2, inter_op_threads: 1}}
""")


def test_config_resolves_model_bundle_without_a_gallery(tmp_path):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    config = load_config(path)
    assert config.classifier_bundle == tmp_path / "models/selected"


def test_config_rejects_invalid_candidate_count(tmp_path):
    path = tmp_path / "configs/default.yaml"
    write_config(path, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        load_config(path)


def test_config_rejects_obsolete_similarity_thresholds(tmp_path):
    path = tmp_path / "configs/default.yaml"
    write_config(path, extra=", high_threshold: 0.58, low_threshold: 0.35")
    with pytest.raises(ValueError, match="threshold"):
        load_config(path)


def test_config_loads_probability_acceptance_threshold(tmp_path):
    path = tmp_path / "configs/default.yaml"
    write_config(path, extra=", probability_threshold: 0.97")
    assert load_config(path).probability_threshold == 0.97


@pytest.mark.parametrize("threshold", ["-0.1", "1.1", ".nan"])
def test_config_rejects_invalid_probability_threshold(tmp_path, threshold):
    path = tmp_path / "configs/default.yaml"
    write_config(path, extra=f", probability_threshold: {threshold}")
    with pytest.raises(ValueError, match="probability_threshold"):
        load_config(path)


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_config_loads_parallel_backbones_bool(tmp_path, value, expected):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    path.write_text(
        path.read_text().replace(
            "inter_op_threads: 1", f"inter_op_threads: 1, parallel_backbones: {value}"
        )
    )
    assert load_config(path).parallel_backbones is expected


def test_config_omitted_parallel_backbones_is_false(tmp_path):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    assert load_config(path).parallel_backbones is False


@pytest.mark.parametrize("value", ["1", "0", "null", '"true"', "[]", "{}"])
def test_config_rejects_nonboolean_parallel_backbones(tmp_path, value):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    path.write_text(
        path.read_text().replace(
            "inter_op_threads: 1", f"inter_op_threads: 1, parallel_backbones: {value}"
        )
    )
    with pytest.raises(ValueError, match="parallel_backbones"):
        load_config(path)


@pytest.mark.parametrize("name", ["allow_spinning", "enable_cpu_mem_arena"])
def test_config_omitted_resource_options_preserve_ort_defaults(tmp_path, name):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    assert getattr(load_config(path), name) is True


@pytest.mark.parametrize("name", ["allow_spinning", "enable_cpu_mem_arena"])
@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_config_loads_resource_option_bools(tmp_path, name, value, expected):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    path.write_text(
        path.read_text().replace("inter_op_threads: 1", f"inter_op_threads: 1, {name}: {value}")
    )
    assert getattr(load_config(path), name) is expected


@pytest.mark.parametrize("name", ["allow_spinning", "enable_cpu_mem_arena"])
@pytest.mark.parametrize("value", ["1", "0", "null", '"true"', "[]", "{}"])
def test_config_rejects_nonboolean_resource_options(tmp_path, name, value):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    path.write_text(
        path.read_text().replace("inter_op_threads: 1", f"inter_op_threads: 1, {name}: {value}")
    )
    with pytest.raises(ValueError, match=name):
        load_config(path)


@pytest.mark.parametrize("setting", ["", ", detector_intra_op_threads: null"])
def test_detector_threads_inherit_shared_threads_when_omitted_or_null(tmp_path, setting):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    path.write_text(
        path.read_text().replace("inter_op_threads: 1", f"inter_op_threads: 1{setting}")
    )
    config = load_config(path)
    assert config.detector_intra_op_threads is None
    assert config.effective_detector_threads == 2


def test_detector_threads_can_override_shared_threads(tmp_path):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    path.write_text(
        path.read_text().replace(
            "intra_op_threads: 2", "intra_op_threads: 1, detector_intra_op_threads: 2"
        )
    )
    config = load_config(path)
    assert config.intra_op_threads == 1
    assert config.detector_intra_op_threads == 2
    assert config.effective_detector_threads == 2


@pytest.mark.parametrize("value", ["0", "-1", "true", "false", "1.0", '"2"', "[]", "{}"])
def test_config_rejects_nonpositive_or_noninteger_detector_threads(tmp_path, value):
    path = tmp_path / "configs/default.yaml"
    write_config(path)
    path.write_text(
        path.read_text().replace(
            "inter_op_threads: 1", f"inter_op_threads: 1, detector_intra_op_threads: {value}"
        )
    )
    with pytest.raises(ValueError, match="detector_intra_op_threads"):
        load_config(path)
