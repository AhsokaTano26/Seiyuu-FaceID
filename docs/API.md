# Seiyuu FaceID HTTP API

[English](API.md) | [简体中文](API.zh-CN.md) | [日本語](API.ja.md)

The HTTP service is implemented. This document describes its interface contract and usage.
It reuses the local SCRFD detector and AdaFace IR101 + R50 classifier and makes no calls to external recognition services.

<a id="安装与启动"></a>

## Installation and startup

Run these commands from the project root.

```bash
uv sync --locked --extra api --extra web --extra dev
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

You need `models/det_500m.onnx` and the complete `models/selected/` model bundle.
Weights are excluded from Git, so cloning the repository alone does not provide runnable models.
Each process loads the models once at startup and validates the classifier files.
Missing configuration or models, or failed validation, causes startup to fail. The service does not report a false success.

- The base URL is `http://127.0.0.1:8000`.
- The minimal test page is `/`. Select an image to recognize identities or detect faces. Selecting multiple images automatically uses batch recognition. The key and parameters are under “Optional settings”.
- Interactive documentation is at `/docs`, ReDoc at `/redoc`, and OpenAPI at `/openapi.json`.
- The default configuration is the project's `configs/default.yaml`.
- `SEIYUU_FACEID_CONFIG` overrides the configuration file location. Relative paths resolve from the startup directory.
- `SEIYUU_FACEID_API_KEY` is an optional Bearer key. Authentication is disabled when it is unset.
- `SEIYUU_FACEID_SERVE_UI=0` disables the home test page and serves only the API and interface documentation.

Use the following command to disable the test page and allow other machines to connect through a reachable IP address of the server.

```bash
SEIYUU_FACEID_SERVE_UI=0 uv run --extra api uvicorn seiyuu_faceid.api.app:create_app \
  --factory --host 0.0.0.0 --port 8000
```

Clients use `http://<reachable-server-ip>:8000`, substituting a reachable server IP.
The wildcard listening address `0.0.0.0` cannot serve as the destination address.

```bash
export SEIYUU_FACEID_API_KEY='replace-with-your-key'
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

When a key is enabled, every `/v1/` endpoint and `/api/v1/detect` requires the header
`Authorization: Bearer replace-with-your-key`. Health checks and documentation do not require a key.
The documentation excludes the private identity list. Retrieve that list through the authenticated model endpoint.

<a id="接口一览"></a>

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process liveness |
| GET | `/ready` | Model readiness |
| GET | `/v1/model` | Model identifiers, classes, defaults, and upload limits |
| POST | `/api/v1/detect` | Face presence detection for services on the same host, returning count, bounding boxes, and detection confidence |
| POST | `/v1/detect` | Face locations and five landmarks only |
| POST | `/v1/recognize` | Detection and identity recognition for one image |
| POST | `/v1/recognize/batch` | Multi-image recognition, with one result per image in upload order |

<a id="图片与参数约定"></a>

## Images and parameters

- Upload using `multipart/form-data`. The single-image field is `file`. Repeat `files` for a batch.
- JPEG, PNG, and WebP are accepted according to actual file contents, independent of extensions or declared MIME types.
- Animated images are rejected. EXIF orientation is applied automatically. Grayscale and transparent images are converted to three channels.
- Each file is limited to **10 MiB (10,485,760 bytes)**. Dimensions are checked before decoding, with a limit of **20,000,000 pixels**. A batch accepts at most **8 images**, and the entire HTTP request is limited to **81 MiB**.
- Server-side paths, image URLs, and Base64 JSON are not accepted. Uploads are not added to the project dataset. The multipart parser may use temporary files, which are closed when the request ends.
- The coordinate origin is the upper-left corner after applying EXIF orientation. Coordinates use pixels. `bbox` is `[x1, y1, x2, y2]`. The five landmarks are ordered as left eye, right eye, nose tip, left mouth corner, and right mouth corner.
- Every recognition endpoint accepts query parameters `top_k` (integer, 1–50) and `probability_threshold` (finite number, 0–1). Omitted parameters use the YAML defaults, currently `5` and `0.95`. They affect only the current request and do not change other requests or loaded models.
- `processing_ms` includes image decoding and inference. It excludes upload and queueing time and provides no performance guarantee.

<a id="健康检查"></a>

## Health checks

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

`GET /health` returns HTTP 200 with this body.

```json
{"status": "ok"}
```

`GET /ready` returns HTTP 200 with this body when the models are ready.

```json
{"status": "ready"}
```

If the models are not yet available, `/ready` returns HTTP 503 with error code `model_unavailable`.
A startup failure exits the process, leaving health checks unreachable.

<a id="同机服务的人脸检测"></a>

## Face detection for services on the same host

Both the Python and C++/Go services support `POST /api/v1/detect`. Upload one image using field `file`.
The service reuses SCRFD loaded at startup and runs detection once per image.
The optional `score_threshold` must satisfy `0 < value ≤ 1` and applies only to this request.
If omitted, the configured `detector.score_threshold` is used. The supplied configuration sets it to `0.50`.

```bash
curl --fail-with-body 'http://127.0.0.1:8000/api/v1/detect?score_threshold=0.7' \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

```json
{
  "has_face": true,
  "face_count": 1,
  "faces": [{"bbox": [38.2, 27.1, 104.5, 119.7], "detector_score": 0.93}]
}
```

Each face occupies one entry after threshold filtering and deduplication of overlapping boxes.
Coordinates refer to pixels in the original image after EXIF orientation correction.
If no face qualifies, the response is HTTP 200 with `{"has_face":false,"face_count":0,"faces":[]}`.
Voice actor identity, sharpness, and recognition probability do not affect this decision.

| Setting | Default | Effect |
| --- | ---: | --- |
| `SEIYUU_FACEID_DETECT_MAX_PIXELS` | 20000000 | Maximum accepted width×height for the new endpoint. It can be lowered to 1. Dimensions are checked before decoding. The per-file limit remains 10 MiB |
| `detector.max_long_edge` | 1920 | The shared detector proportionally scales images whose long edge exceeds the limit. The native service uses the exported model configuration |
| `SEIYUU_FACEID_DETECT_TIMEOUT_MS` | 5000 | Positive integer millisecond budget shared by upload, queueing, decoding, and inference after the new endpoint receives request headers |
| `SEIYUU_FACEID_DETECT_CLIENT_CIDR` | Empty | IPv4/IPv6 loopback is allowed by default. One additional private address or subnet for a container on the same host can be authorized, such as `172.29.0.3/32` |

Processes on the same host call `127.0.0.1` with the existing Bearer key.
For Docker Compose, supply the environment variables above in `.env`.
When calling a published port from the host, set `FACEID_BIND_ADDRESS=127.0.0.1` and authorize the host bridge gateway visible to the container using a single-address CIDR.
For direct calls between containers on the same host, specify the calling container's address or subnet on the dedicated network.
For systemd, put the settings in the existing `EnvironmentFile` and restart the corresponding service after changes.
Exclude this path from public reverse proxies because the local proxy itself may have an authorized source address.

Deployment probes continue to use `/health` and `/ready`. Both endpoints remain responsive during inference.
A request timeout returns HTTP 503 with `detection_timeout`. Queued requests leave the queue.
Synchronous inference that has already started finishes before releasing the model execution slot.
Detection request status and duration appear in service logs. The Python logger is `seiyuu_faceid.api.detection` at INFO level.

<a id="模型信息"></a>

## Model information

`GET /v1/model` returns the following fields.

- `model_id` is the SHA-256 identifier of classifier model assets.
- `recognition_id` combines the model and default runtime configuration into one identifier.
- `class_count` and `class_names` provide the trained identity count and complete identity list.
- `defaults` contains `top_k` and `probability_threshold`.
- `limits` contains `max_upload_bytes`, `max_request_bytes`, `max_image_pixels`, and `max_batch_size`.
- `supported_formats` is `["JPEG", "PNG", "WEBP"]`.

```bash
curl --fail http://127.0.0.1:8000/v1/model \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY"
```

<a id="单图识别"></a>

## Single-image recognition

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/recognize \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" \
  -F 'file=@photo.jpg'

curl --fail-with-body \
  'http://127.0.0.1:8000/v1/recognize?top_k=3&probability_threshold=0.98' \
  -F 'file=@photo.jpg'
```

An HTTP 200 response may look like this. Values are illustrative.

```json
{
  "model_id": "model-sha256",
  "image": {"filename": "photo.jpg", "width": 1000, "height": 1200},
  "face_count": 1,
  "processing_ms": 120.5,
  "faces": [
    {
      "bbox": [100.0, 120.0, 300.0, 340.0],
      "detector_score": 0.89,
      "quality_score": 1.0,
      "quality_reasons": [],
      "status": "recognized",
      "identity": "上坂堇",
      "probability_threshold": 0.95,
      "decision_reason": null,
      "score_type": "class_probability",
      "candidates": [{"identity": "上坂堇", "score": 0.9999}]
    }
  ]
}
```

`candidates` is ordered by descending probability and contains at most `top_k` entries.
Other candidates are omitted from the example. Images with no detected faces still return HTTP 200 with `face_count: 0` and `faces: []`.

| Field/status | Meaning |
| --- | --- |
| `recognized` | Quality checks passed and the highest class probability meets this request's threshold. `identity` contains the recognized identity |
| `review` + `low_probability` | Highest class probability is insufficient. `identity: null`, with candidates retained |
| `review` + `quality_rejected` | Face is too small or too blurry. `identity: null`, with no candidates |
| `quality_reasons` | May contain `face_too_small` or `too_blurry` |
| `score_type` | `class_probability`, a probability within the fixed known classes |

The current model has 50 fixed identities. Class probability does not express certainty that the person belongs to a known identity.
Rejection of unknown people or character images has not been calibrated.
A high score cannot establish the real identity of an arbitrary person in an upload.

<a id="仅检测"></a>

## Detection only

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/detect -F 'file=@photo.jpg'
```

The top-level fields match the recognition response. Each face has the following shape.

```json
{
  "bbox": [100.0, 120.0, 300.0, 340.0],
  "detector_score": 0.89,
  "landmarks": [[150.0, 180.0], [250.0, 180.0], [200.0, 230.0],
                [160.0, 280.0], [240.0, 280.0]]
}
```

This endpoint performs neither identity classification nor face quality rejection.
Small faces are returned if detected.

<a id="批量识别"></a>

## Batch recognition

```bash
curl --fail-with-body 'http://127.0.0.1:8000/v1/recognize/batch?top_k=3' \
  -F 'files=@photo1.jpg' -F 'files=@photo2.png'
```

HTTP 200 returns `{"results": [...]}`. Each entry matches a single-image response and includes a zero-based `index`.
Images with identical filenames are not merged. An individual empty, unsupported, corrupt, or oversized image produces an error at its position while other images continue processing.

```json
{
  "results": [
    {
      "index": 0,
      "filename": "broken.jpg",
      "error": {"code": "invalid_image", "message": "Could not decode image."}
    }
  ]
}
```

Exceeding the batch count or request body limit, invalid parameters, missing fields, or failed authentication rejects the entire request before inference starts.
A model error during batch processing returns HTTP 500 for the whole request.

<a id="错误格式"></a>

## Error format

All HTTP errors use `{"error": {"code": "...", "message": "..."}}`.
Parameter validation also supplies `details`. Each item contains `field` and `message` and never echoes image contents, keys, or local paths.

| HTTP | code | Condition |
| --- | --- | --- |
| 400 | `empty_file` | Empty file |
| 400 | `invalid_image` | Corrupt or undecodable image |
| 400 | `bad_request` | Invalid multipart data or headers |
| 401 | `unauthorized` | No valid Bearer key when authentication is enabled |
| 403 | `internal_only` | Connection source is not authorized for the new detection endpoint |
| 404 | `not_found` | Route does not exist |
| 405 | `method_not_allowed` | Incorrect HTTP method |
| 413 | `file_too_large` | File exceeds 10 MiB |
| 413 | `request_too_large` | Body exceeds 81 MiB. Streaming bytes are counted even without Content-Length |
| 413 | `image_too_large` | Image exceeds the pixel limit for its endpoint |
| 415 | `unsupported_image` | Image is animated or is not JPEG/PNG/WebP |
| 422 | `validation_error` | Missing fields, invalid parameters, or batch size outside 1–8 |
| 500 | `internal_error` | Inference or unexpected service error. Detailed stack traces appear only in service logs |
| 503 | `model_unavailable` | Models are not ready |
| 503 | `detection_timeout` | New detection endpoint exceeds its request deadline |

<a id="python-http-客户端"></a>

## Python HTTP client

Install `httpx` separately on the client. If the service has a key, add Authorization to the headers.

```python
from pathlib import Path
import httpx

photo = Path("photo.jpg")
with photo.open("rb") as handle:
    response = httpx.post(
        "http://127.0.0.1:8000/v1/recognize",
        files={"file": (photo.name, handle, "application/octet-stream")},
        params={"top_k": 5},
        timeout=60,
    )
response.raise_for_status()
for face in response.json()["faces"]:
    print(face["bbox"], face["status"], face["identity"], face["candidates"])
```

<a id="部署与验证"></a>

## Deployment and validation

See [Linux deployment](DEPLOY_LINUX.md) for independent deployment on a standard Linux server and offline transfer of images and models.

The default listener is local only. For LAN access, explicitly use `--host 0.0.0.0` and configure a key.
Internet deployments use a reverse proxy for HTTPS, access control, and request limits. Cross-origin access is disabled by default.
Use one worker initially. Each worker separately loads approximately 415 MiB of weights and needs additional inference memory.
Within a process, inference runs serially in a thread pool. Health checks do not wait for inference to finish.

```bash
uv run pytest
uv run ruff check .
uv pip check
```

Automated API tests cover the HTTP contract and exceptional inputs.
Real-model integration tests run separately when private weights are available locally.
CI does not download or commit private photos or models.

Framework references include [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/) and [Uvicorn settings](https://uvicorn.dev/settings/).
