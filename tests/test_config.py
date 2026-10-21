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
