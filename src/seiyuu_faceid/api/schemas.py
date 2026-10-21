"""Public response schemas shared by validation and generated OpenAPI docs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]
Box = tuple[float, float, float, float]


class ResponseModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class ErrorDetail(ResponseModel):
    field: str
    message: str


class ErrorBody(ResponseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(ResponseModel):
    error: ErrorBody


class HealthResponse(ResponseModel):
    status: Literal["ok", "ready"]


class Defaults(ResponseModel):
    top_k: int
    probability_threshold: float


class LimitsResponse(ResponseModel):
    max_upload_bytes: int
    max_request_bytes: int
    max_image_pixels: int
    max_batch_size: int


class ModelResponse(ResponseModel):
    model_id: str
    recognition_id: str
    class_count: int
    class_names: list[str]
    defaults: Defaults
    limits: LimitsResponse
    supported_formats: list[str]


class ImageInfo(ResponseModel):
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CandidateResponse(ResponseModel):
    identity: str
    score: float


class RecognitionFace(ResponseModel):
    bbox: Box
    detector_score: float
    quality_score: float
    quality_reasons: list[str]
    status: Literal["recognized", "review"]
    identity: str | None
    probability_threshold: float
    decision_reason: Literal["low_probability", "quality_rejected"] | None
    score_type: Literal["class_probability"]
    candidates: list[CandidateResponse]


class DetectionFace(ResponseModel):
    bbox: Box
    detector_score: float
    landmarks: tuple[Point, Point, Point, Point, Point]


class ImageResponse(ResponseModel):
    model_id: str
    image: ImageInfo
    face_count: int = Field(ge=0)
    processing_ms: float = Field(ge=0)


class RecognitionResponse(ImageResponse):
    faces: list[RecognitionFace]


class DetectionResponse(ImageResponse):
    faces: list[DetectionFace]


class BatchSuccess(RecognitionResponse):
    index: int = Field(ge=0)


class BatchFailure(ResponseModel):
    index: int = Field(ge=0)
    filename: str
    error: ErrorBody


class BatchResponse(ResponseModel):
    results: list[BatchSuccess | BatchFailure]
