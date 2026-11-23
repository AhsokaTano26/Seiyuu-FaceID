# C++ inference core + Go HTTP

[English](NATIVE_CPP_GO.md) | [简体中文](NATIVE_CPP_GO.zh-CN.md) | [日本語](NATIVE_CPP_GO.ja.md)

This runtime evaluates the benefits of native execution on Intel N150 / 8 GB. One Go process calls C++ directly through cgo. SCRFD, AdaFace IR101, and R50 load once. The two backbones run in parallel, while HTTP inference requests run serially. Models remain FP32, retaining the existing classification head and default 0.95 threshold.

See the [recognition architecture guide](TECHNICAL_ARCHITECTURE.md) for algorithms, training, evaluation, and module interaction.

N150 validation completed Linux builds and startup, parity on 1,051 crops and 12 original images, and the full three-way HTTP baseline. With matching settings, the native baseline showed no reliable speedup and higher peak RSS. The CPU-saving thread profile increased mean HTTP time by about 37.9% relative to the original-thread group. See the [N150 report](N150_BENCHMARK.md) for full scope and later independent experiments. On 2026-10-28 the FTZ/DAZ version took over the target's existing API. Long-run and stress acceptance testing remain outstanding.

A later independent FTZ/DAZ candidate completed full regression and HTTP measurements, with mean request time about 0.681 seconds and historical holdout Top-1 unchanged at 98.48%. Its implementation is now integrated into `native/`. Enable it explicitly with `--flush-denormals` during model export after building. It requires Linux x86_64 and CPU support for FTZ/DAZ. Mac ARM explicitly rejects enabling it. Omitting the flag keeps it off by default and preserves compatibility with existing JSON. See the [FTZ experiment](N150_BENCHMARK.md) for results, memory tradeoffs, and boundaries between experiment phases.

Deployment uses `bin/faceid-server`, an adjacent `lib/` directory of shared libraries, a model directory, and metadata. This is not a fully static single-file program. Serving requires no Python. Model-parameter export and parity validation use Python.

<a id="build"></a>

## Build

Native builds support **Linux x86_64** and macOS arm64. Mac build artifacts cannot be used as Linux executables. Build machines need a C++17 compiler, Make, CMake ≥ 3.20, Go ≥ 1.24, curl, tar, and a SHA-256 utility. Linux can install `build-essential cmake curl ca-certificates` and a Go distribution meeting the version requirement.

```bash
bash native/build.sh
```

The script pins ONNX Runtime 1.29.0, OpenCV 5.0.0, and nlohmann/json 3.12.0, verifies download SHA-256 values, builds the required OpenCV modules and C++ core, runs CTest and Go unit tests, and generates `dist/native-runtime/`. It defaults to 4 build jobs. Use `JOBS=2` when memory is tight. OpenCV may download additional platform-optimized dependencies, so builds require networking. Dependencies and build outputs stay in ignored directories. After dependencies are built, application-only changes can use `bash native/build.sh --app-only` to skip OpenCV configuration and compilation. Run a full build after changing dependency versions or build options.

```bash
uv run python tools/export_native_model.py --config configs/n150.yaml
```

The validated fast profile is recommended for N150. Existing output files are not overwritten.

```bash
uv run python tools/export_native_model.py --config configs/n150.yaml \
  --flush-denormals --output models/selected/native-n150.json
```

Point the later server's `-model` argument to the corresponding JSON. This flag configures both ORT session threads and FTZ/DAZ mode around every C++ call. Thread floating-point state is restored on completion or exception to accommodate Go thread scheduling. Weights, the classification head, and the 0.95 threshold remain unchanged. It changes subnormal semantics without converting to FP16 or INT8. Accuracy and decisions matched on 1,051 crops, which does not guarantee bitwise equality for all unknown inputs.

Preparation verifies the original model and classifier checksums, converts existing NPZ parameters losslessly into `models/selected/native-model.json`, and generates a `.sha256` sidecar. ONNX weights remain unchanged. Model paths in JSON are relative to its location. Keep this deployment layout.

```text
models/
  det_500m.onnx
  selected/
    adaface_ir101.onnx
    w600k_r50.onnx
    classifier.npz
    native-model.json
    native-model.json.sha256
```

`classifier.npz` verifies the original classifier source and is not read through Python at runtime. At startup, Go verifies the JSON sidecar, original model hashes, and model identifier, exiting immediately on failure. Checksums detect corruption but do not represent digital-signature authentication of the model package. Re-export JSON after configuration changes.

<a id="startup-and-api"></a>

## Startup and API

```bash
export SEIYUU_FACEID_API_KEY='replace-with-your-key'
dist/native-runtime/bin/faceid-server \
  -model models/selected/native-model.json \
  -addr 127.0.0.1:8000
```

The API key uses the existing service's environment variable. If unset, authentication is not required. The default listen address is local. See the [API documentation](API.md) for endpoints and upload fields. Routes are `/health`, `/ready`, `/v1/model`, `/v1/detect`, `/v1/recognize`, and `/v1/recognize/batch`. This prototype has no home-page UI, Swagger, ReDoc, or OpenAPI routes. Recognition fields follow the existing API, though error-detail wording can differ.

Limits are 10 MiB per file, 81 MiB per request, 20 million pixels per image, and 8 images per batch. Static JPEG, PNG, and WebP are supported with EXIF orientation applied. Up to 8 inference requests may be parsing uploads, waiting, or processing. Excess requests return `503 server_busy`, and queue cancellation returns `408 request_cancelled`. Started native inference finishes before releasing resources. Shutdown destroys the model only after the current call finishes. All APNG are rejected, including single-frame APNG.

Multipart file content exceeding the 32 KiB memory threshold spills to a temporary directory and is removed when the request ends, so deployment still needs temporary disk capacity. `processing_ms` includes file reading, validation, decoding, and inference, excluding multipart upload parsing and queueing. Measure client HTTP time separately for full requests. HTTP graceful shutdown waits for 90 seconds, but models cannot be destroyed until native calls finish, so the whole process is not guaranteed to exit within exactly 90 seconds.

`model_id` matches the original model. `recognition_id` incorporates the native implementation and full configuration to distinguish implementations and settings. Threshold and candidate count can be overridden per request.

```bash
curl --fail http://127.0.0.1:8000/ready
curl --fail-with-body http://127.0.0.1:8000/v1/recognize \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
```

HTTP latency measurement on the target does not require Python. Build outputs include `faceid-bench`.

```bash
dist/native-runtime/bin/faceid-bench -url http://127.0.0.1:8000 \
  -image ./photo.jpg -warmups 3 -runs 20
```

It reads the same API-key environment variable and reports mean / P50 / P95 HTTP latency and mean server processing time. It does not print identities, file paths, or response contents. Latency is separate from CPU and memory metrics. Start Python and native services separately for comparison, using identical images and parameters so two model instances do not compete for CPU.

<a id="parity-validation"></a>

## Parity validation

Unchanged models do not guarantee identical outputs. Check decoding, detection, five-point alignment, normalization, the linear head, candidate order, and acceptance thresholds. JPEG decoders and platform-specific numerical implementations can produce differences.

```bash
# Use .so on Linux and .dylib on macOS.
uv run --extra api python tools/validate_native.py \
  --library dist/native-runtime/lib/libfaceid_core.so \
  --model models/selected/native-model.json --config configs/n150.yaml \
  --crop-dir /path/to/aligned-112-crops \
  --image-dir /path/to/original-images --encoded --output /tmp/native-parity.json
```

The tool reads only explicitly supplied directories and does not scan training photos by default. Its report summarizes probability differences and decision parity without saving photos, identity lists, or features. It loads Python and C++ together, so its process memory does not represent native-service memory. Resource comparisons should start independent Python and Go services with identical inputs, threads, warmup, and request order, measuring latency and process RSS. The historical holdout was used in model selection and does not provide a new independent generalization guarantee. N150 has completed Linux builds, real startup, parity on all crops and fixed originals, and three HTTP baselines. Its results are listed separately in the [target-machine report](N150_BENCHMARK.md). The Mac records below retain their original platform and input scope.

Tied detection scores have a specific boundary. C++ NMS processes ties in reverse input-index order, while Python uses NumPy's default unstable sort. Overlapping boxes with exactly tied scores may select different landmarks. Sampled regression cannot establish bitwise equivalence for arbitrary images. Selecting a default runtime still requires acceptance testing on target hardware and actual business images.

<a id="local-regression"></a>

## Local regression record

On macOS ARM64, Python and C++ loaded their own ONNX Runtime 1.29.0 in separate processes to prevent dynamic-library reuse within one process from affecting validation. All 1,051 historical holdout crops had their features re-extracted.

- Top-1, ordered Top-5, and 0.95 acceptance decisions all matched, with no added misclassifications.
- Both achieved 1,035 / 1,051 = **98.48%** accuracy.
- Maximum absolute feature difference was `2.24e-8`, and probability difference was `3.58e-7`. Results were not bitwise identical.
- A separate test of 12 originals / 14 faces passed on both raw BGR and direct uploaded-file decoding paths. Boxes, detection scores, quality scores, candidate order, and acceptance decisions matched. Maximum probability difference was `2.09e-7`.

Private local details remain in ignored `data/experiments/native-cpp-go-20261027/` and are excluded from model Releases. The full Python suite of 206 tests, C++ CTest, Go unit and race tests, and real-model HTTP integration tests passed. Dynamic loading was checked after copying the Mac build directory elsewhere. These are Mac-phase records. See the [target-machine report](N150_BENCHMARK.md) for later Linux builds, N150 full parity, and HTTP baseline checks. That report lists the tests actually run on the target and does not automatically include the 206 tests or race tests.

<a id="local-http-resources"></a>

## Local HTTP resource comparison

On the same Mac, independent Python / Go services used the N150 profile. Another 12 originals were selected. Each service warmed up for one round and timed four rounds, totaling 48 requests. Each implementation ran in three fresh processes with alternating execution order. The table shows medians of the three process results. Both service instances were never resident simultaneously.

| Metric | Python HTTP | C++ + Go HTTP | Change |
| --- | ---: | ---: | ---: |
| Mean HTTP latency | 94.13 ms | 85.00 ms | 9.7% lower |
| HTTP P95 | 113.83 ms | 103.15 ms | 9.4% lower |
| Mean server processing time | 92.14 ms | 82.69 ms | 10.3% lower |
| Process peak RSS | 1160.89 MiB | 1026.94 MiB | 133.95 MiB / 11.5% lower |

Detection, quality decisions, candidate order, and acceptance matched for all three HTTP output rounds, with a maximum probability difference of `2.38e-7`. HTTP time includes local upload and response. P95 uses nearest-rank, and server processing time includes decoding and inference. Peak RSS comes from `wait4` after each service process exits and includes startup, model loading, and large-image decoding. It cannot be mixed with old benchmarks measuring only crop classification. These are gains from the complete runtime migration, including changes to decoding and data processing, and cannot all be attributed to language choice. They do not predict N150 latency, memory, or power consumption.

Build outputs and deployment materials remain in repository workflows and are excluded from GitHub model Release attachments.

<a id="standard-dataset-retest"></a>

## Retesting on public standard datasets

On 2026-10-28, three comparison rounds ran locally on M4 Pro / 24 GiB with fixed samples of 512 LFW images and 244 WIDER FACE validation images, totaling 9,072 timed HTTP requests. Mean full-recognition C++/Go latencies were 58.82 / 75.33 ms, respectively, 1.83% / 2.67% lower than Python. Detection-only latency fell about 7.9% / 14.7%, and median peak RSS fell 3.8%. All output-parity checks passed.

These Mac tests disabled FTZ/DAZ and retain a scope separate from N150. They do not report LFW verification accuracy or WIDER detection AP. See the [standard-dataset benchmark](STANDARD_DATASET_BENCHMARK.md) for inputs, workloads, statistics, and original evidence.

A later [full accuracy evaluation](STANDARD_DATASET_ACCURACY.md) found 99.13% fused ten-fold verification accuracy on 6,000 LFW pairs. WIDER full validation Easy / Medium / Hard AP (×100) was 85.49 / 79.48 / 48.84. The original voice-actor holdout remained at 98.48%. Python and C++ evaluation metrics and pair decisions matched.

<a id="linux-systemd-部署"></a>
<a id="linux-systemd"></a>

## Linux systemd deployment

The service template is [`native/deploy/seiyuu-faceid.service`](../native/deploy/seiyuu-faceid.service). It uses `DynamicUser`, a read-only file system, and a private temporary directory. Serving needs neither Python nor Docker. The template uses `0.0.0.0:8000`, which can be changed to a local address. A root-owned `/etc/seiyuu-faceid/api.env` supplies the API key as `SEIYUU_FACEID_API_KEY=existing-key`. Set permissions to `0600` and exclude it from the repository. Confirm that the port is free before deployment.

Use the layout below. Place models and runtime on the main disk. The service user must be able to read/traverse every directory and execute programs. Other files are read-only. The service user should not be able to write to programs or models.

```text
/opt/seiyuu-faceid/
  current -> releases/<version>
  releases/<version>/
    runtime/bin/faceid-server
    runtime/lib/...
    runtime/licenses/...
    models/det_500m.onnx
    models/selected/{adaface_ir101.onnx,w600k_r50.onnx,classifier.npz}
    models/selected/native-model.json
    models/selected/native-model.json.sha256
```

Here `native-model.json` comes from the fast profile exported with `--flush-denormals`. After renaming it, regenerate the SHA-256 sidecar for the new filename while preserving JSON bytes and all relative model paths. Once the layout, key file, and `current` symlink are ready, run the following.

```bash
sudo install -m 0644 native/deploy/seiyuu-faceid.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
sudo systemctl status seiyuu-faceid.service
```

The N150 switch reused the core library, Go executable, and configuration bytes already validated in full. Mac artifacts were not transferred to Linux. Validation first ran on a temporary port, followed by stopping the old container, disabling its automatic restart, and enabling the new service on the original port. The API key, other containers, and old model mounts were preserved, with the old container retained for rollback. After deployment, outputs were checked again on 12 originals / 14 faces. See the [deployment acceptance record](N150_BENCHMARK.md).

The native service exposes the API routes above. `/docs`, `/redoc`, and the upload home page are absent. Consult the repository [API documentation](API.md) for fields. A Swagger page in the previous deployment is not automatically retained.

When rolling back from this service to the previous Docker deployment, disable the systemd service before restoring the old container to avoid a port conflict.

```bash
sudo systemctl disable --now seiyuu-faceid.service
docker update --restart=unless-stopped <original-container-name>
docker start <original-container-name>
curl --fail http://127.0.0.1:8000/ready
```
