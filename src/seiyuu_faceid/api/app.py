"""HTTP adapter for local inference: uvicorn seiyuu_faceid.api.app:create_app --factory."""

import asyncio
import logging
import os
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse

from seiyuu_faceid.api.errors import ApiError
from seiyuu_faceid.api.schemas import (
    BatchResponse,
    DetectionResponse,
    ErrorResponse,
    HealthResponse,
    ModelResponse,
    RecognitionResponse,
)
from seiyuu_faceid.api.uploads import (
    SUPPORTED_FORMATS,
    RequestSizeLimit,
    UploadLimits,
    decode_image,
    safe_filename,
)
from seiyuu_faceid.runtime.config import AppConfig, load_config
from seiyuu_faceid.runtime.engine import FaceIdEngine
from seiyuu_faceid.runtime.pipeline import recognize_faces, serialize_result

LOGGER = logging.getLogger(__name__)
TopK = Annotated[int | None, Query(ge=1, le=50, description="Number of identity candidates.")]
Threshold = Annotated[
    float | None,
    Query(ge=0, le=1, allow_inf_nan=False, description="Acceptance probability for this request."),
]


def _process_upload(
    engine: FaceIdEngine,
    upload: UploadFile,
    limits: UploadLimits,
    *,
    detection_only: bool = False,
    top_k: int | None = None,
    probability_threshold: float | None = None,
) -> dict[str, object]:
    started = perf_counter()
    decoded = decode_image(
        upload.file.read(limits.max_upload_bytes + 1), upload.filename or "image", limits
    )
    image = decoded.pixels
    if detection_only:
        faces = [
            {
                "bbox": [round(float(value), 2) for value in face.bbox],
                "detector_score": float(face.score),
                "landmarks": [
                    [round(float(value), 2) for value in point] for point in face.landmarks
                ],
            }
            for face in engine.detect_array(image)
        ]
    else:
        config = engine.config
        faces = [
            serialize_result(face)
            for face in recognize_faces(
                image,
                detector=engine.detector,
                classifier=engine.classifier,
                min_face_size=config.min_face_size,
                min_sharpness=config.min_sharpness,
                top_k=config.top_k if top_k is None else top_k,
                probability_threshold=(
                    config.probability_threshold
                    if probability_threshold is None
                    else probability_threshold
                ),
            )
        ]
    return {
        "model_id": engine.classifier.model_id,
        "image": {"filename": decoded.filename, "width": image.shape[1], "height": image.shape[0]},
        "face_count": len(faces),
        "processing_ms": round((perf_counter() - started) * 1000, 3),
        "faces": faces,
    }


def _process_batch(
    engine: FaceIdEngine,
    uploads: list[UploadFile],
    limits: UploadLimits,
    top_k: int | None,
    probability_threshold: float | None,
) -> dict[str, object]:
    results = []
    for index, upload in enumerate(uploads):
        try:
            result = _process_upload(
                engine, upload, limits, top_k=top_k, probability_threshold=probability_threshold
            )
        except ApiError as error:
            result = {
                "filename": safe_filename(upload.filename or "image"),
                "error": {"code": error.code, "message": error.detail},
            }
        results.append({"index": index, **result})
    return {"results": results}


def create_app(
    *,
    config_path: Path | str | None = None,
    engine_factory: Callable[[AppConfig], FaceIdEngine] = FaceIdEngine,
    limits: UploadLimits | None = None,
    api_key: str | None = None,
) -> FastAPI:
    """Construct the app without loading models until its lifespan starts."""
    limits = limits or UploadLimits()
    key = os.environ.get("SEIYUU_FACEID_API_KEY", "") if api_key is None else api_key
    path = (
        Path(
            config_path
            or os.environ.get("SEIYUU_FACEID_CONFIG")
            or Path(__file__).resolve().parents[3] / "configs/default.yaml"
        )
        .expanduser()
        .resolve()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = load_config(path)
        app.state.inference_lock = asyncio.Lock()
        app.state.engine = await run_in_threadpool(engine_factory, config)
        LOGGER.info("FaceID model loaded: %s", app.state.engine.classifier.model_id)
        try:
            yield
        finally:
            app.state.engine = None

    app = FastAPI(
        title="Seiyuu FaceID API",
        version="1.0.0",
        description="Local face detection and supervised identity classification.",
        lifespan=lifespan,
    )
    app.state.engine = None
    app.add_middleware(RequestSizeLimit, max_bytes=limits.max_request_bytes)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        if isinstance(error, ApiError):
            payload = error.payload()
        else:
            codes = {400: "bad_request", 404: "not_found", 405: "method_not_allowed"}
            payload = {
                "error": {
                    "code": codes.get(error.status_code, "http_error"),
                    "message": str(error.detail),
                }
            }
        headers = {"WWW-Authenticate": "Bearer"} if error.status_code == 401 else error.headers
        return JSONResponse(payload, status_code=error.status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        details = [
            {"field": ".".join(map(str, item["loc"])), "message": item["msg"]}
            for item in error.errors()
        ]
        return JSONResponse(
            {
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request parameters.",
                    "details": details,
                }
            },
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception):
        LOGGER.exception("FaceID request failed: %s", request.url.path, exc_info=error)
        return JSONResponse(
            {
                "error": {
                    "code": "internal_error",
                    "message": "An internal inference error occurred.",
                }
            },
            status_code=500,
        )

    def require_engine() -> FaceIdEngine:
        if app.state.engine is None:
            raise ApiError(503, "model_unavailable", "The model is not ready.")
        return app.state.engine

    bearer = HTTPBearer(auto_error=False)

    async def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if key and (
            credentials is None
            or not secrets.compare_digest(credentials.credentials.encode(), key.encode())
        ):
            raise ApiError(401, "unauthorized", "A valid Bearer API key is required.")

    error_responses = {
        status: {"model": ErrorResponse} for status in (400, 401, 413, 415, 422, 500, 503)
    }
    router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)], responses=error_responses)

    if os.environ.get("SEIYUU_FACEID_SERVE_UI", "1") != "0":

        @app.get("/", response_class=FileResponse, include_in_schema=False)
        async def test_page():
            return FileResponse(
                Path(__file__).with_name("test.html"), headers={"Cache-Control": "no-store"}
            )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        return {"status": "ok"}

    @app.get(
        "/ready",
        response_model=HealthResponse,
        tags=["Health"],
        responses={503: {"model": ErrorResponse}},
    )
    async def ready():
        require_engine()
        return {"status": "ready"}

    @router.get("/model", response_model=ModelResponse, tags=["Model"])
    async def model():
        engine = require_engine()
        return {
            "model_id": engine.classifier.model_id,
            "recognition_id": engine.recognition_id,
            "class_count": len(engine.classifier.class_names),
            "class_names": engine.classifier.class_names,
            "defaults": {
                "top_k": engine.config.top_k,
                "probability_threshold": engine.config.probability_threshold,
            },
            "limits": asdict(limits),
            "supported_formats": list(SUPPORTED_FORMATS),
        }

    @router.post("/detect", response_model=DetectionResponse, tags=["Inference"])
    async def detect(file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image.")]):
        engine = require_engine()
        async with app.state.inference_lock:
            return await run_in_threadpool(
                _process_upload, engine, file, limits, detection_only=True
            )

    @router.post("/recognize", response_model=RecognitionResponse, tags=["Inference"])
    async def recognize(
        file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image.")],
        top_k: TopK = None,
        probability_threshold: Threshold = None,
    ):
        engine = require_engine()
        async with app.state.inference_lock:
            return await run_in_threadpool(
                _process_upload,
                engine,
                file,
                limits,
                top_k=top_k,
                probability_threshold=probability_threshold,
            )

    @router.post("/recognize/batch", response_model=BatchResponse, tags=["Inference"])
    async def recognize_batch(
        files: Annotated[list[UploadFile], File(description="Repeat files for each image.")],
        top_k: TopK = None,
        probability_threshold: Threshold = None,
    ):
        if not 1 <= len(files) <= limits.max_batch_size:
            raise ApiError(422, "validation_error", f"Upload 1 to {limits.max_batch_size} images.")
        engine = require_engine()
        async with app.state.inference_lock:
            return await run_in_threadpool(
                _process_batch, engine, files, limits, top_k, probability_threshold
            )

    app.include_router(router)
    return app
