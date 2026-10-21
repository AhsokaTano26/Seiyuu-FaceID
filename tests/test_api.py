"""HTTP contracts using the real pipeline with lightweight inference backbones."""

import asyncio
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette import formparsers

from seiyuu_faceid.api.app import create_app
from seiyuu_faceid.api.uploads import UploadLimits
from seiyuu_faceid.runtime.engine import FaceIdEngine
from seiyuu_faceid.runtime.results import Candidate
from seiyuu_faceid.vision.detector import FaceDetection


class FixtureDetector:
    def detect(self, image_bgr):
        if not image_bgr.any():
            return []
        return [
            FaceDetection(
                np.array([10, 10, 90, 90], dtype=np.float32),
                0.9,
                np.array([[30, 35], [70, 35], [50, 55], [35, 75], [65, 75]], dtype=np.float32),
            )
        ]


class FixtureClassifier:
    class_names = ["alice", "bob"]
    model_id = "fixture-model"

    def classify(self, aligned, *, top_k):
        return [Candidate("alice", 0.97), Candidate("bob", 0.03)][:top_k]


class FixtureEngine(FaceIdEngine):
    def __init__(self, config):
        self.config = config
        self.detector = FixtureDetector()
        self.classifier = FixtureClassifier()
        self.recognition_id = "fixture-recognition"


def png_bytes(*, blank=False, blurry=False):
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    if blurry:
        pixels[:] = 128
    elif not blank:
        pixels[::2, ::2] = 255
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SEIYUU_FACEID_API_KEY", raising=False)
    monkeypatch.delenv("SEIYUU_FACEID_CONFIG", raising=False)
    with TestClient(create_app(engine_factory=FixtureEngine)) as instance:
        yield instance


def upload(client, path="/v1/recognize", **kwargs):
    return client.post(path, files={"file": ("photo.png", png_bytes(), "text/plain")}, **kwargs)


def test_model_and_health_report_loaded_engine(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}
    model = client.get("/v1/model").json()
    assert model["model_id"] == "fixture-model"
    assert model["class_count"] == 2
    assert model["class_names"] == ["alice", "bob"]
    assert model["defaults"] == {"top_k": 5, "probability_threshold": 0.95}
    assert model["limits"]["max_batch_size"] == 8


def test_homepage_serves_test_ui_without_authentication(monkeypatch):
    monkeypatch.setenv("SEIYUU_FACEID_API_KEY", "test-secret")
    with TestClient(create_app(engine_factory=FixtureEngine)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/v1/model").status_code == 401


def test_api_only_mode_disables_page_and_preserves_inference(monkeypatch):
    monkeypatch.setenv("SEIYUU_FACEID_SERVE_UI", "0")
    with TestClient(create_app(engine_factory=FixtureEngine, api_key="")) as client:
        assert client.get("/").status_code == 404
        assert client.get("/ready").status_code == 200
        assert upload(client).status_code == 200


def test_recognition_returns_real_pipeline_results_with_json_scalars(client):
    response = upload(client)
    assert response.status_code == 200
    data = response.json()
    assert data["image"] == {"filename": "photo.png", "width": 100, "height": 100}
    assert data["face_count"] == 1
    assert data["processing_ms"] >= 0
    face = data["faces"][0]
    assert face["bbox"] == [10.0, 10.0, 90.0, 90.0]
    assert face["identity"] == "alice"
    assert face["status"] == "recognized"
    assert face["score_type"] == "class_probability"
    assert face["candidates"] == [
        {"identity": "alice", "score": 0.97},
        {"identity": "bob", "score": 0.03},
    ]


def test_per_request_options_do_not_mutate_engine_defaults(client):
    response = upload(client, params={"top_k": 1, "probability_threshold": 0.99})
    face = response.json()["faces"][0]
    assert face["status"] == "review"
    assert face["identity"] is None
    assert face["decision_reason"] == "low_probability"
    assert len(face["candidates"]) == 1
    next_face = upload(client).json()["faces"][0]
    assert next_face["status"] == "recognized"
    assert len(next_face["candidates"]) == 2
    assert next_face["probability_threshold"] == 0.95


def test_blank_image_is_a_successful_empty_result(client):
    response = client.post("/v1/recognize", files={"file": ("blank.png", png_bytes(blank=True))})
    assert response.status_code == 200
    assert response.json()["faces"] == []
    assert response.json()["face_count"] == 0


def test_blurry_face_is_review_without_candidates(client):
    response = client.post("/v1/recognize", files={"file": ("blur.png", png_bytes(blurry=True))})
    face = response.json()["faces"][0]
    assert face["status"] == "review"
    assert face["decision_reason"] == "quality_rejected"
    assert face["quality_reasons"] == ["too_blurry"]
    assert face["candidates"] == []


def test_detection_returns_landmarks_without_running_classifier(client):
    def forbidden_classify(*args, **kwargs):
        raise AssertionError("detection must not classify")

    client.app.state.engine.classifier.classify = forbidden_classify
    response = upload(client, "/v1/detect")
    assert response.status_code == 200
    assert response.json()["faces"] == [
        {
            "bbox": [10.0, 10.0, 90.0, 90.0],
            "detector_score": 0.9,
            "landmarks": [[30.0, 35.0], [70.0, 35.0], [50.0, 55.0], [35.0, 75.0], [65.0, 75.0]],
        }
    ]


def test_batch_preserves_order_same_filenames_and_item_errors(client):
    response = client.post(
        "/v1/recognize/batch",
        files=[
            ("files", ("same.png", png_bytes())),
            ("files", ("broken.jpg", b"not an image")),
            ("files", ("empty.jpg", b"")),
            ("files", ("same.png", png_bytes(blank=True))),
        ],
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [item["index"] for item in results] == [0, 1, 2, 3]
    assert results[0]["faces"][0]["identity"] == "alice"
    assert results[1]["error"]["code"] == "invalid_image"
    assert results[2]["error"]["code"] == "empty_file"
    assert results[3]["faces"] == []


@pytest.mark.parametrize(
    "params",
    [
        {"top_k": 0},
        {"top_k": 51},
        {"top_k": "x"},
        {"probability_threshold": -0.1},
        {"probability_threshold": 1.1},
        {"probability_threshold": "nan"},
        {"probability_threshold": "inf"},
    ],
)
def test_query_validation_uses_stable_error_envelope(client, params):
    response = upload(client, params=params)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["field"].startswith("query.")
    assert "input" not in error["details"][0]


def test_missing_file_and_excessive_batch_count_are_validation_errors(client):
    response = client.post("/v1/recognize")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    response = client.post(
        "/v1/recognize/batch", files=[("files", (f"{i}.png", png_bytes())) for i in range(9)]
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(("data", "code"), [(b"", "empty_file"), (b"bad", "invalid_image")])
def test_invalid_single_upload_has_client_error(client, data, code):
    response = client.post("/v1/recognize", files={"file": ("private.jpg", data)})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


def test_bearer_key_protects_every_v1_route_but_not_health_or_docs(monkeypatch):
    monkeypatch.setenv("SEIYUU_FACEID_API_KEY", "test-secret")
    with TestClient(create_app(engine_factory=FixtureEngine)) as secured:
        for path in ("/v1/model", "/v1/detect", "/v1/recognize", "/v1/recognize/batch"):
            method = "GET" if path == "/v1/model" else "POST"
            response = secured.request(method, path, files={"file": ("x.png", png_bytes())})
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.json()["error"]["code"] == "unauthorized"
        assert upload(secured, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert upload(secured, headers={"Authorization": "Bearer test-secret"}).status_code == 200
        assert secured.get("/health").status_code == 200
        assert secured.get("/docs").status_code == 200
        schema = secured.get("/openapi.json").json()
        assert "test-secret" not in str(schema)
        assert schema["paths"]["/v1/model"]["get"]["security"]


def test_not_started_app_is_alive_but_not_ready():
    app = create_app(engine_factory=FixtureEngine, api_key="")
    with TestClient(app) as started:
        started.app.state.engine = None
        assert started.get("/health").status_code == 200
        assert started.get("/ready").status_code == 503
        assert upload(started).status_code == 503


def test_model_loading_error_fails_startup():
    def missing_model(config):
        raise FileNotFoundError("missing model fixture")

    with pytest.raises(FileNotFoundError, match="missing model fixture"):
        with TestClient(create_app(engine_factory=missing_model)):
            pytest.fail("startup should fail")


def test_server_errors_do_not_leak_internal_paths(client):
    def broken_classify(*args, **kwargs):
        raise RuntimeError("secret path /private/model.onnx")

    client.app.state.engine.classifier.classify = broken_classify
    with TestClient(client.app, raise_server_exceptions=False) as errors:
        errors.app.state.engine.classifier.classify = broken_classify
        response = upload(errors)
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert "/private" not in response.text
        batch = errors.post("/v1/recognize/batch", files=[("files", ("x.png", png_bytes()))])
        assert batch.status_code == 500


def test_body_limit_is_json_even_when_multipart_parser_reads_it():
    limits = UploadLimits(max_request_bytes=100)
    with TestClient(create_app(engine_factory=FixtureEngine, limits=limits)) as small:
        response = upload(small)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"


def test_chunked_limit_closes_partially_parsed_uploads(monkeypatch):
    opened = []
    original = formparsers.SpooledTemporaryFile

    def track_tempfile(*args, **kwargs):
        temporary = original(*args, **kwargs)
        opened.append(temporary)
        return temporary

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_tempfile)
    first = (
        b'--stream\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n'
        b"Content-Type: image/png\r\n\r\n" + png_bytes()
    )

    async def chunks():
        yield first
        yield b"x" * 100

    async def exercise():
        app = create_app(
            engine_factory=FixtureEngine, limits=UploadLimits(max_request_bytes=len(first) + 10)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post(
                    "/v1/recognize",
                    content=chunks(),
                    headers={"Content-Type": "multipart/form-data; boundary=stream"},
                )

    response = asyncio.run(exercise())
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert opened and all(temporary.closed for temporary in opened)


def test_per_file_limit_and_pixel_limit_are_reported_over_http():
    for limits, code in [
        (UploadLimits(max_upload_bytes=10), "file_too_large"),
        (UploadLimits(max_image_pixels=99), "image_too_large"),
    ]:
        with TestClient(create_app(engine_factory=FixtureEngine, limits=limits)) as client:
            response = upload(client)
            assert response.status_code == 413
            assert response.json()["error"]["code"] == code


def test_malformed_multipart_has_json_error(client):
    response = client.post(
        "/v1/recognize", content=b"bad", headers={"Content-Type": "multipart/form-data"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_corrupt_exif_is_an_item_error_without_aborting_batch(client):
    valid = png_bytes()
    chunk_type = b"eXIf"
    payload = b"II*\x00"
    exif_chunk = (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload))
    )
    broken = valid[:33] + exif_chunk + valid[33:]
    with TestClient(client.app, raise_server_exceptions=False) as errors:
        response = errors.post("/v1/recognize", files={"file": ("broken.png", broken)})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_image"
        batch = errors.post(
            "/v1/recognize/batch",
            files=[
                ("files", ("first.png", valid)),
                ("files", ("broken.png", broken)),
                ("files", ("last.png", valid)),
            ],
        )
        assert batch.status_code == 200
        results = batch.json()["results"]
        assert results[0]["faces"][0]["identity"] == "alice"
        assert results[1]["error"]["code"] == "invalid_image"
        assert results[2]["faces"][0]["identity"] == "alice"


def test_http_404_and_405_use_same_error_envelope(client):
    assert client.get("/no-such-route").json()["error"]["code"] == "not_found"
    response = client.get("/v1/recognize")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_health_remains_responsive_and_inference_is_serialized(client):
    entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum = 0
    lock = threading.Lock()

    def slow_classify(aligned, *, top_k):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        entered.set()
        assert release.wait(timeout=5)
        with lock:
            active -= 1
        return [Candidate("alice", 0.97)]

    client.app.state.engine.classifier.classify = slow_classify
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(upload, client)
        assert entered.wait(timeout=3)
        second = pool.submit(upload, client)
        try:
            assert pool.submit(client.get, "/health").result(timeout=2).status_code == 200
        finally:
            release.set()
        assert first.result(timeout=3).status_code == 200
        assert second.result(timeout=3).status_code == 200
    assert maximum == 1


def test_openapi_documents_uploads_outputs_and_errors(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/v1/recognize"]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert "413" in operation["responses"]
    assert "415" in operation["responses"]
    assert "422" in operation["responses"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]
