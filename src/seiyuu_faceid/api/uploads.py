"""Bound compressed uploads and image dimensions before expensive decoding."""

import struct
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from seiyuu_faceid.api.errors import ApiError

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")


@dataclass(frozen=True)
class UploadLimits:
    max_upload_bytes: int = 10 * 1024 * 1024
    max_request_bytes: int = 81 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    max_batch_size: int = 8


@dataclass(frozen=True)
class DecodedImage:
    filename: str
    pixels: np.ndarray


class _BodyLimitExceeded(ApiError, OSError):
    # Starlette closes partially parsed multipart files on OSError. Retaining
    # HTTPException semantics also preserves the 413 through FastAPI's parser.
    pass


def safe_filename(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(char for char in name if char.isprintable())[:255] or "image"


def decode_image(data: bytes, filename: str, limits: UploadLimits) -> DecodedImage:
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > limits.max_upload_bytes:
        raise ApiError(413, "file_too_large", "The uploaded file exceeds the byte limit.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in SUPPORTED_FORMATS or getattr(image, "is_animated", False):
                raise ApiError(415, "unsupported_image", "Use a non-animated JPEG, PNG or WebP.")
            if image.width * image.height > limits.max_image_pixels:
                raise ApiError(413, "image_too_large", "The image exceeds the pixel limit.")
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            pixels = np.ascontiguousarray(np.asarray(normalized)[:, :, ::-1])
    except Image.DecompressionBombError as error:
        raise ApiError(413, "image_too_large", "The image exceeds the pixel limit.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, struct.error) as error:
        raise ApiError(400, "invalid_image", "Could not decode image.") from error
    return DecodedImage(safe_filename(filename), pixels)


class RequestSizeLimit:
    """Count raw HTTP bytes, including multipart overhead and chunked bodies."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False
        count = 0

        async def limited_receive() -> Message:
            nonlocal count
            message = await receive()
            if message["type"] == "http.request":
                count += len(message.get("body", b""))
                if count > self.max_bytes:
                    raise _BodyLimitExceeded(
                        413, "request_too_large", "The request exceeds the byte limit."
                    )
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            lengths = [value for key, value in scope["headers"] if key.lower() == b"content-length"]
            if lengths:
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise ApiError(400, "bad_request", "Invalid Content-Length header.")
                if int(lengths[0]) > self.max_bytes:
                    raise ApiError(413, "request_too_large", "The request exceeds the byte limit.")
            await self.app(scope, limited_receive, tracked_send)
        except ApiError as error:
            if response_started:
                raise
            await JSONResponse(error.payload(), status_code=error.status_code)(scope, receive, send)
