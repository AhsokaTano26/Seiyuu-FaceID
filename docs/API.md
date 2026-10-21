# Seiyuu FaceID HTTP API

HTTP 服务已实现，本文是接口约定与调用说明。服务复用本地 SCRFD 检测器和
AdaFace IR101 + R50 分类器，不调用外部识别服务。

## 安装与启动

在项目根目录执行：

```bash
uv sync --locked --extra api --extra web --extra dev
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

必须具备 `models/det_500m.onnx` 和完整的 `models/selected/` 模型包。
权重不在 Git 中，单独克隆仓库不包含可运行模型。每个进程在启动时加载一次模型并校验
分类器文件；配置、模型缺失或校验失败会使启动失败，不提供伪成功响应。

- 基础地址：`http://127.0.0.1:8000`
- 极简测试页面：`/`，选择图片即可识别或检测，多张图片自动批量识别；密钥和参数在「可选设置」。
- 交互文档：`/docs`，ReDoc：`/redoc`，OpenAPI：`/openapi.json`
- 默认配置：项目的 `configs/default.yaml`
- `SEIYUU_FACEID_CONFIG`：覆盖配置文件位置；相对路径从启动目录解析。
- `SEIYUU_FACEID_API_KEY`：可选的 Bearer 密钥，未设置时不要求鉴权。
- `SEIYUU_FACEID_SERVE_UI=0`：关闭首页测试页面，仅提供 API 和接口文档。

关闭测试页面，并允许其他机器通过本机可达 IP 访问：

```bash
SEIYUU_FACEID_SERVE_UI=0 uv run --extra api uvicorn seiyuu_faceid.api.app:create_app \
  --factory --host 0.0.0.0 --port 8000
```

调用方使用 `http://<服务机器的可达IP>:8000`，不使用 `0.0.0.0` 作为目标地址。

```bash
export SEIYUU_FACEID_API_KEY='replace-with-your-key'
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

开启密钥后，所有 `/v1/` 接口需要请求头
`Authorization: Bearer replace-with-your-key`。健康检查和文档无需密钥；
文档不包含私有身份列表，身份列表通过鉴权的模型接口获取。

## 接口一览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 进程存活 |
| GET | `/ready` | 模型就绪 |
| GET | `/v1/model` | 模型标识、类别、默认参数、上传限制 |
| POST | `/v1/detect` | 仅检测人脸位置和五点关键点 |
| POST | `/v1/recognize` | 单图检测与身份识别 |
| POST | `/v1/recognize/batch` | 多图识别，按上传顺序返回每张图的结果 |

## 图片与参数约定

- 上传方式：`multipart/form-data`；单图字段为 `file`，批量重复字段为 `files`。
- 按实际文件内容接受 JPEG、PNG、WebP，不依赖扩展名或 MIME 声明。
- 拒绝动图；自动应用 EXIF 方向，灰度与透明图片转换为三通道图片。
- 单文件最多 **10 MiB（10,485,760 字节）**，解码前检查图片尺寸，最多
  **20,000,000 像素**；批量最多 **8 张**，整个 HTTP 请求最多 **81 MiB**。
- 不接受服务端文件路径、图片 URL 或 Base64 JSON。上传文件不写入项目数据集；
  multipart 解析器可能使用临时文件，在请求结束时关闭。
- 坐标原点为已应用 EXIF 方向的图片左上角，单位为像素；`bbox` 是
  `[x1, y1, x2, y2]`。五点顺序：左眼、右眼、鼻尖、左嘴角、右嘴角。
- 所有识别接口可带查询参数 `top_k`（整数，1–50）和
  `probability_threshold`（有限数，0–1）。省略时使用 YAML 默认值，当前为 `5`、`0.95`。
  参数只影响本次请求，不修改其他请求或已加载模型。
- `processing_ms` 包含该图片的解码与推理时间，不包含上传与排队时间，不是性能保证。

## 健康检查

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

`GET /health` 返回 HTTP 200：

```json
{"status": "ok"}
```

`GET /ready` 在模型就绪时返回 HTTP 200：

```json
{"status": "ready"}
```

模型尚不可用时 `/ready` 返回 HTTP 503，错误码 `model_unavailable`。
启动失败时进程退出，无法访问健康检查。

## 模型信息

`GET /v1/model` 返回：

- `model_id`：分类模型资源的 SHA-256 标识。
- `recognition_id`：模型与默认运行配置的联合标识。
- `class_count` 与 `class_names`：已训练身份总数和完整身份列表。
- `defaults`：`top_k`、`probability_threshold`。
- `limits`：`max_upload_bytes`、`max_request_bytes`、`max_image_pixels`、`max_batch_size`。
- `supported_formats`：`["JPEG", "PNG", "WEBP"]`。

```bash
curl --fail http://127.0.0.1:8000/v1/model \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY"
```

## 单图识别

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/recognize \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" \
  -F 'file=@photo.jpg'

curl --fail-with-body \
  'http://127.0.0.1:8000/v1/recognize?top_k=3&probability_threshold=0.98' \
  -F 'file=@photo.jpg'
```

HTTP 200 示例（示意值）：

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

`candidates` 按概率从高到低返回，数量最多为 `top_k`；示例省略其他候选。
未检测到人脸时仍返回 HTTP 200，`face_count: 0`、`faces: []`。

| 字段/状态 | 含义 |
| --- | --- |
| `recognized` | 质量检查通过，最高类别概率达到本次阈值，`identity` 为识别身份 |
| `review` + `low_probability` | 最高类别概率不足，`identity: null`，保留候选 |
| `review` + `quality_rejected` | 人脸太小或太模糊，`identity: null`，候选为空 |
| `quality_reasons` | 可能含 `face_too_small`、`too_blurry` |
| `score_type` | `class_probability`，固定已知类别内的概率 |

当前固定 50 个身份；类别概率并不是“这个人一定属于已知身份”的概率。
尚未校准对未知人物或角色图的拒识，不能依靠高分判定任意上传者的真实身份。

## 仅检测

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/detect -F 'file=@photo.jpg'
```

返回与识别相同的顶层字段，各个人脸为：

```json
{
  "bbox": [100.0, 120.0, 300.0, 340.0],
  "detector_score": 0.89,
  "landmarks": [[150.0, 180.0], [250.0, 180.0], [200.0, 230.0],
                [160.0, 280.0], [240.0, 280.0]]
}
```

此接口不执行身份分类或人脸质量拒绝；检测到的小脸也会返回。

## 批量识别

```bash
curl --fail-with-body 'http://127.0.0.1:8000/v1/recognize/batch?top_k=3' \
  -F 'files=@photo1.jpg' -F 'files=@photo2.png'
```

HTTP 200 返回 `{"results": [...]}`，每项与单图响应一致，并包含从 0 开始的 `index`。
同名图片不会合并。单项空文件、非法格式、损坏或超限会在该位置返回错误，其他图片继续：

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

超过批量数量限制、请求体超限、参数非法、缺失字段或鉴权失败会使整个请求失败，
不开始推理。批量请求执行期间的模型错误返回整个请求 HTTP 500。

## 错误格式

HTTP 错误统一为 `{"error": {"code": "...", "message": "..."}}`。
参数校验额外提供 `details`，每项含 `field`、`message`，不回显图片、密钥或本地路径。

| HTTP | code | 情况 |
| --- | --- | --- |
| 400 | `empty_file` | 文件为空 |
| 400 | `invalid_image` | 图片损坏或无法解码 |
| 400 | `bad_request` | 非法 multipart 或请求头 |
| 401 | `unauthorized` | 开启密钥后未提供有效 Bearer 密钥 |
| 404 | `not_found` | 路由不存在 |
| 405 | `method_not_allowed` | HTTP 方法错误 |
| 413 | `file_too_large` | 单文件超过 10 MiB |
| 413 | `request_too_large` | 请求体超过 81 MiB，支持无 Content-Length 的流式计数 |
| 413 | `image_too_large` | 图片超过 20,000,000 像素 |
| 415 | `unsupported_image` | 非 JPEG/PNG/WebP 或为动图 |
| 422 | `validation_error` | 缺少字段、非法参数或批量数量不在 1–8 |
| 500 | `internal_error` | 推理或未预期的服务错误；详细堆栈仅写服务日志 |
| 503 | `model_unavailable` | 模型尚未就绪 |

## Python HTTP 客户端

客户端单独安装 `httpx`；如果服务设置密钥，在 headers 中加入 Authorization。

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

## 部署与验证

普通 Linux 服务器完整独立部署、镜像与模型离线迁移见
[Linux 部署文档](DEPLOY_LINUX.md)。

默认只监听本机；需要局域网访问时显式使用 `--host 0.0.0.0` 并配置密钥。
跨互联网部署通过反向代理提供 HTTPS、访问控制和请求限制。默认不开放跨域。
建议单 worker：每个 worker 都独立加载约 415 MiB 权重，并额外占用推理内存。
单进程内的推理串行在线程池执行，健康检查不等待推理完成。

```bash
uv run pytest
uv run ruff check .
uv pip check
```

API 自动化测试覆盖 HTTP 契约和异常输入；本机具备私有权重时另执行真实模型
集成测试。CI 不下载或提交私人照片与模型。

框架用法参考：[FastAPI 文件上传](https://fastapi.tiangolo.com/tutorial/request-files/)、
[Uvicorn 启动参数](https://uvicorn.dev/settings/)。
