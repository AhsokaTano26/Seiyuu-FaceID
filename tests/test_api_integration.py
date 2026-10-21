"""Exercise HTTP decoding and serialization with real, local-only ONNX weights."""

import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from seiyuu_faceid.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models/selected"
REQUIRED = [ROOT / "models/det_500m.onnx"] + [
    BUNDLE / name
    for name in ("manifest.json", "classifier.npz", "adaface_ir101.onnx", "w600k_r50.onnx")
]
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED), reason="Private ONNX model bundle is local-only"
)


@pytest.fixture(scope="module")
def live_model_client():
    with TestClient(create_app(config_path=ROOT / "configs/default.yaml", api_key="")) as client:
        yield client


def test_real_model_http_handles_blank_image(live_model_client):
    stream = BytesIO()
    Image.new("RGB", (640, 480)).save(stream, format="PNG")
    response = live_model_client.post(
        "/v1/recognize", files={"file": ("blank.png", stream.getvalue())}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image"] == {"filename": "blank.png", "width": 640, "height": 480}
    assert data["face_count"] == 0
    assert data["faces"] == []
    assert len(data["model_id"]) == 64


def test_real_model_http_recognizes_reference_photo_and_detects_landmarks(live_model_client):
    samples = ROOT / "data/training/samples.jsonl"
    if not samples.is_file():
        pytest.skip("Private reference images are local-only")
    rows = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()]
    row = next(
        (
            row
            for row in rows
            if row["split"] == "test" and (ROOT / row["original_image"]).is_file()
        ),
        None,
    )
    if row is None:
        pytest.skip("No local test photo available")
    path = ROOT / row["original_image"]
    with path.open("rb") as handle:
        recognized = live_model_client.post("/v1/recognize", files={"file": (path.name, handle)})
    assert recognized.status_code == 200
    result = recognized.json()
    assert any(face["identity"] == row["identity"] for face in result["faces"])
    assert all(face["score_type"] == "class_probability" for face in result["faces"])
    with path.open("rb") as handle:
        detected = live_model_client.post("/v1/detect", files={"file": (path.name, handle)})
    assert detected.status_code == 200
    assert detected.json()["face_count"] == result["face_count"]
    assert all(len(face["landmarks"]) == 5 for face in detected.json()["faces"])
