"""Upload limits must run before allocating a decoded image or accepting large bodies."""

import asyncio
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from seiyuu_faceid.api.errors import ApiError
from seiyuu_faceid.api.uploads import RequestSizeLimit, UploadLimits, decode_image


def image_bytes(mode="RGB", size=(12, 8), image_format="PNG", **save_options):
    stream = BytesIO()
    Image.new(mode, size).save(stream, format=image_format, **save_options)
    return stream.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_decode_normalizes_to_bgr_without_using_extension(mode):
    decoded = decode_image(image_bytes(mode), "C:\\private\\photo.txt", UploadLimits())
    assert decoded.filename == "photo.txt"
    assert decoded.pixels.shape == (8, 12, 3)
    assert decoded.pixels.dtype == np.uint8
    assert decoded.pixels.flags.c_contiguous


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_supported_formats_preserve_dimensions(image_format):
    decoded = decode_image(image_bytes(image_format=image_format), "image", UploadLimits())
    assert decoded.pixels.shape == (8, 12, 3)


def test_decode_preserves_bgr_channels():
    stream = BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(stream, format="PNG")
    decoded = decode_image(stream.getvalue(), "red.png", UploadLimits())
    assert decoded.pixels[0, 0].tolist() == [30, 20, 10]


def test_exif_orientation_is_applied_before_coordinates_are_returned():
    exif = Image.Exif()
    exif[274] = 6
    decoded = decode_image(
        image_bytes(size=(12, 8), image_format="JPEG", exif=exif), "photo.jpg", UploadLimits()
    )
    assert decoded.pixels.shape == (12, 8, 3)


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 400, "empty_file"),
        (b"broken image", 400, "invalid_image"),
        (image_bytes(image_format="GIF"), 415, "unsupported_image"),
        (image_bytes(image_format="BMP"), 415, "unsupported_image"),
    ],
)
def test_decode_rejects_invalid_or_unsupported_input(data, status, code):
    with pytest.raises(ApiError) as caught:
        decode_image(data, "photo.jpg", UploadLimits())
    assert caught.value.status_code == status
    assert caught.value.code == code


def test_file_byte_limit_accepts_boundary_and_rejects_one_byte_over():
    data = image_bytes()
    limits = replace(UploadLimits(), max_upload_bytes=len(data))
    assert decode_image(data, "ok.png", limits).pixels.size > 0
    with pytest.raises(ApiError) as caught:
        decode_image(data + b"x", "large.png", limits)
    assert caught.value.code == "file_too_large"


def test_pixel_limit_rejects_before_loading_pixels(monkeypatch):
    data = image_bytes(size=(10, 10))
    limits = replace(UploadLimits(), max_image_pixels=100)
    assert decode_image(data, "ok.png", limits).pixels.shape == (10, 10, 3)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("oversized image was fully decoded")

    monkeypatch.setattr(Image.Image, "load", forbidden_load)
    with pytest.raises(ApiError) as caught:
        decode_image(data, "large.png", replace(limits, max_image_pixels=99))
    assert caught.value.code == "image_too_large"


def test_animated_webp_is_rejected():
    stream = BytesIO()
    Image.new("RGB", (8, 8), "red").save(
        stream,
        format="WEBP",
        save_all=True,
        append_images=[Image.new("RGB", (8, 8), "blue")],
        duration=100,
        loop=0,
    )
    with pytest.raises(ApiError) as caught:
        decode_image(stream.getvalue(), "animation.webp", UploadLimits())
    assert caught.value.code == "unsupported_image"


def run_body(chunks, headers=(), max_bytes=5):
    messages = []
    consumed = []

    async def app(scope, receive, send):
        while True:
            event = await receive()
            consumed.append(event)
            if not event.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        index = len(consumed)
        return {
            "type": "http.request",
            "body": chunks[index],
            "more_body": index < len(chunks) - 1,
        }

    async def send(message):
        messages.append(message)

    asyncio.run(
        RequestSizeLimit(app, max_bytes=max_bytes)(
            {"type": "http", "headers": list(headers)}, receive, send
        )
    )
    return messages, consumed


def test_chunked_body_limit_does_not_depend_on_content_length():
    messages, consumed = run_body([b"123", b"456"])
    assert messages[0]["status"] == 413
    assert b"request_too_large" in messages[1]["body"]
    assert len(consumed) == 1


def test_declared_oversize_is_rejected_before_reading():
    messages, consumed = run_body([b"123456"], [(b"content-length", b"6")])
    assert messages[0]["status"] == 413
    assert consumed == []


@pytest.mark.parametrize("length", [b"-1", b"invalid"])
def test_invalid_content_length_is_a_client_error(length):
    messages, consumed = run_body([b""], [(b"content-length", length)])
    assert messages[0]["status"] == 400
    assert b"bad_request" in messages[1]["body"]
    assert consumed == []


def test_request_body_exactly_at_limit_is_accepted():
    messages, consumed = run_body([b"12", b"345"])
    assert messages[0]["status"] == 204
    assert len(consumed) == 2
