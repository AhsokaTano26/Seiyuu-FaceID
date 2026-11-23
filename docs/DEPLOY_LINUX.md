# Independent deployment on a Linux server

[English](DEPLOY_LINUX.md) | [简体中文](DEPLOY_LINUX.zh-CN.md) | [日本語](DEPLOY_LINUX.ja.md)

For installation and systemd configuration without Docker, see [native Linux deployment](DEPLOY_NATIVE_LINUX.md).

Continue with Python + FastAPI + ONNX Runtime and deploy the CPU version first.
Recognition runs entirely on the target server without calling the development computer or external model APIs. Retraining is unnecessary.
The runtime needs the following components.

- A Docker service image containing Python, locked API dependencies, project code, and native Linux libraries.
- Local models comprising the SCRFD detector, AdaFace IR101, R50, classification head, and model manifest.
- Configuration and Compose files defining listening ports, keys, health checks, logging, and restart policy.

Preinstall Docker Engine and Docker Compose v2 on the target host.
Start testing with 2 cores and 4 GiB of memory. This is a planning starting point, and the minimum configuration has not been verified.
Actual capacity depends on the target CPU, faces per image, and concurrency.
The supplied configuration uses 2 inference threads per model and 1 service worker, with the two recognition backbones running in parallel.
Together, the two models may use approximately 4 compute threads during recognition.
On a 2-core host or one with a smaller CPU quota, try `runtime.parallel_backbones: false` or reduce `runtime.intra_op_threads` to 1 and measure.
Local parallel benchmarks do not establish throughput on the target Linux host. See the [optimization report](CLASSIFIER_OPTIMIZATION.md).
The current implementation uses only the CPU. A server GPU does not automatically enable acceleration.

For **Intel N150 + 8 GB on Linux/Docker**, use the added `configs/n150.yaml` and `compose.n150.yaml`.
This preset reduces thread overhead and disables ORT spinning while retaining the original FP32 models and recognition rules.
See the [N150 guide](N150.md) for resource and latency tradeoffs and target-host benchmark commands.

<a id="1-通过-github-actions-发布到-docker-hub"></a>

## 1. Publish to Docker Hub through GitHub Actions

The workflow is `.github/workflows/ci.yml`, and the image repository is [`tano26/seiyuu-faceid`](https://hub.docker.com/r/tano26/seiyuu-faceid).
Add `DOCKERHUB_TOKEN` in GitHub [Actions Secrets](https://github.com/AhsokaTano26/Seiyuu-FaceID/settings/secrets/actions), using an access token permitted to push to that Docker repository.
The login username is fixed as `tano26`, so no additional username variable is needed.

Push the project code containing the workflow and Dockerfile to `main`.
Lint, tests, and compilation checks run before image build and push.
You can also run `CI and Docker Hub` manually in Actions.
PRs only run quality checks and do not log in to or publish on Docker Hub.

| Trigger | Image tags |
| --- | --- |
| Push to `main` or manually run on `main` | `latest`, `sha-<full-commit-sha>` using the full commit SHA |
| Push a version tag such as `v0.1.0` | `0.1.0`, `sha-<full-commit-sha>` using the full commit SHA |

Version tags do not overwrite `latest`. The workflow publishes `linux/amd64` and `linux/arm64` images, and the target server automatically selects its matching architecture when pulling.
Python dependencies are installed from `uv.lock`, with only the API enabled.
Training, Streamlit, and test dependencies are omitted. During each architecture's build, imports of native libraries and the API module are verified.
Image builds do not require training weights.

The Docker build context uses an allowlist excluding `models/`, photos, experimental data, `.env`, and local `.venv`.
These private models are also absent from GitHub Runners.
Models are distributed separately as described below, then mounted read-only on the target server.
The runtime never reads files from the development computer.

<a id="2-生成模型和部署配置包"></a>

## 2. Generate the model and deployment configuration bundle

Cloning Git alone is insufficient because `.gitignore` excludes the models.
Run packaging on a machine holding the complete current models. The required files are listed below.

```text
models/det_500m.onnx
models/selected/adaface_ir101.onnx
models/selected/w600k_r50.onnx
models/selected/classifier.npz
models/selected/manifest.json
models/selected/MODEL_CARD.md
models/selected/ADAFACE_LICENSE
```

Use the packaging tool, which requires only the Python standard library.

```bash
python3 tools/package_deployment.py \
  --output dist/seiyuu-faceid-deployment
tar -C dist -czf dist/seiyuu-faceid-deployment.tar.gz seiyuu-faceid-deployment
```

The tool verifies SHA-256 hashes for the four model files, creates `SHA256SUMS` for the full deployment bundle, and refuses to overwrite an existing output directory.
The output contains models, configuration, Compose, an example with an empty key, and instructions for independent operation.
It excludes training data, test photos, evaluation features, and local keys.
When changing image tags, also update `FACEID_IMAGE` in the bundle's `.env`.

Transfer the generated `.tar.gz` to the target server.
After pulling the service image from Docker Hub once, the server runs independently.
It needs no Python, uv, or training dependencies installed on the host and does not access PyPI.
Docker and Compose must already be installed.

If the target server cannot reach Docker Hub, pull and export the matching architecture's image on a connected machine, then include it with the packaging tool's `--image-tar` argument.

```bash
docker pull --platform linux/amd64 tano26/seiyuu-faceid:latest
mkdir -p dist
docker image save --output dist/image-linux-amd64.tar tano26/seiyuu-faceid:latest
python3 tools/package_deployment.py \
  --image-tar dist/image-linux-amd64.tar \
  --output dist/seiyuu-faceid-offline
tar -C dist -czf dist/seiyuu-faceid-offline.tar.gz seiyuu-faceid-offline
```

Check the server architecture with `uname -m`.
`x86_64` maps to `linux/amd64`, and `aarch64` maps to `linux/arm64`.
The exported offline image must match the target architecture.

<a id="3-在目标-linux-服务器启动"></a>

## 3. Start on the target Linux server

```bash
tar -xzf seiyuu-faceid-deployment.tar.gz
cd seiyuu-faceid-deployment
sha256sum --check SHA256SUMS
cp .env.example .env
chmod 600 .env
```

Generate a key with `openssl rand -hex 32` and place it after `SEIYUU_FACEID_API_KEY=` in `.env`.
The default listener uses port 8000 on all network interfaces.
Change `FACEID_PORT` for another port, or set `FACEID_BIND_ADDRESS` to `127.0.0.1` when using an existing HTTPS reverse proxy.
For production, `FACEID_IMAGE` can use a fixed version or SHA tag in place of `latest`.
After the image is successfully published, pull it and start the service.

```bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
docker compose ps
curl --fail http://127.0.0.1:8000/ready
```

The expected readiness response is `{"status":"ready"}`.
A successful `/ready` confirms model loading. A successful `/v1/recognize` response verifies the full image recognition path.
No outbound network connection is required at container runtime.
Compose disables automatic image pulls by default, allowing later starts and recreation offline.
Explicitly pull with `docker compose pull --policy always` during installation or updates.
Container UID 10001 must be able to read models and configuration.
The packaging tool grants read permissions to these files. If copying models manually, preserve file read and directory traversal permissions.

If an offline bundle contains `image.tar`, import it with `docker load --input image.tar`, then start with `docker compose up -d --pull never --wait --wait-timeout 180`.
Skip `docker compose pull`. Compose will not attempt to pull images online.

Call the API from another device with the following commands.

```bash
export FACEID_URL='http://服务器IP:8000'
export FACEID_API_KEY='与服务端一致的密钥'
curl --fail-with-body "$FACEID_URL/v1/model" \
  -H "Authorization: Bearer $FACEID_API_KEY"
curl --fail-with-body "$FACEID_URL/v1/recognize" \
  -H "Authorization: Bearer $FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

The server firewall or cloud security group must allow the caller to access the selected port.
Use HTTPS on public networks. See [API.md](API.md) for request formats and response statuses.
The model includes 50 trained identities.
Low confidence or inadequate quality returns `review` by default. Migrating the deployment does not change this rule.

<a id="4-检查和维护"></a>

## 4. Checks and maintenance

```bash
docker compose logs --tail 100 faceid
docker stats --no-stream
docker compose pull --policy always
docker compose up -d
docker compose restart faceid
docker compose down
```

Health checks request `/ready`. After process exit, the `unless-stopped` policy may restart the container.
An unhealthy status alone does not trigger a restart.
After changing `.env`, run `docker compose up -d --force-recreate`.
Restart the service after changing configuration or replacing models.
Back up the deployment directory and `.env` to preserve everything required for operation.

Target-device acceptance checks should at least cover `/ready`, authenticated `/v1/model`, `/v1/recognize` with a known identity photo, a 401 response without a key, and recovery after restart.
Then record memory and recognition latency to decide whether to tune threads or increase capacity.
Model files total approximately 417 MiB. Measure runtime memory separately.
Increasing workers usually increases model memory usage as well.

Docker references include [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/), [Actions image publishing](https://docs.docker.com/build/ci/github-actions/push-multi-registries/), [Compose environment variables](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/), and [Compose service configuration](https://docs.docker.com/reference/compose-file/services/).
