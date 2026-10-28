# C++ core + Go HTTP experiment

User-authorized prototype for Intel N150 / 8 GB Linux. Preserve Python as the
reference. Runtime has no Python: one Go process calls a C++ shared library via
cgo. Three persistent ONNX Runtime CPU sessions and one model instance; serialize
requests while running the two recognition backbones concurrently. No push or
Release publication. The FTZ/DAZ implementation passed target-machine parity
and performance checks and was selected for the N150 service on 2026-10-28.

## Interfaces

`include/faceid.h` is the C ownership and error boundary. Go validates model
checksums before creating the engine. C++ validates dimensions, finite parameters
and configuration and performs all computer vision, inference and scoring.

`tools/export_native_model.py` converts existing NPZ parameters into
`models/selected/native-model.json`; Python is needed only for this preparation
step. ONNX and classifier source files are unchanged. JSON schema version 1:

- `model_id`: SHA256 of Python's sorted assets JSON, unchanged from Python.
- `source_assets`: original manifest assets mapping filename to SHA256;
  `detector_sha256`: detector SHA256.
- `models`: `detector`, `adaface`, `r50` paths relative to this JSON file.
- `class_names`: ordered string array.
- `head`: float64 `mean` / `scale`; float32 `coefficients` (rows) / `intercept`.
- `runtime`: `intra_op_threads`, `detector_intra_op_threads`, `inter_op_threads`,
  `parallel_backbones`, `allow_spinning`, `enable_cpu_mem_arena`, and optional
  `flush_denormals` (strict bool, absent means false). The latter enables FTZ/DAZ
  only on supported Linux x86_64, with complete caller MXCSR restoration around
  session creation and each inference call; it also configures ORT worker pools.
- `detector`: `input_size` [width,height], `max_long_edge`, `score_threshold`,
  `nms_threshold`.
- `quality`: `min_face_size`, `min_sharpness`.
- `recognition`: `top_k`, `probability_threshold`.

`faceid_model` returns `model_id`, a provisional `recognition_id` marker
(`native-v1:` + model ID), `class_count`, `class_names`, and `defaults`.
Go replaces `recognition_id` with SHA256 of `native-v1\n` + the raw model JSON
bytes, covering the full configuration and detector checksum, and adds upload
limits/formats. JSON whitespace changes also change the HTTP recognition ID.

## Scope and validation

Go implements health/readiness, authenticated model info, detect, recognize and
batch recognition. Multipart JPEG/PNG/WebP, 10 MiB/file, 81 MiB/request,
20 million pixels/image, 8 images/batch. Parameter validation, cancellation while
waiting, one inference at a time, and graceful shutdown. UI/OpenAPI are outside
this initial prototype. Preserve EXIF orientation and reject animation.

Check model corruption, malformed uploads, authentication, concurrent requests,
and shutdown ownership. Compare all 1,051 historical held-out crops' embeddings,
probabilities, top-1/top-5 and 0.95 decisions; also compare detection, alignment
and image decode on original photos. Do not treat unchanged weights as proof of
unchanged results. Benchmark Python and native in separate processes with fixed
inputs, warmups, thread settings and repeated runs. Local ARM timing does not
predict N150 timing. Keep private fixtures/results in ignored data directories.

## Work split

1. Core: C++ pipeline and C ABI, CMake library, small numerical/error tests.
2. HTTP: Go net/http service, cgo adapter, API behavior tests.
3. Integration: reproducible dependencies/build, model exporter, Python/native
   parity and performance measurements, Linux deployment instructions.
