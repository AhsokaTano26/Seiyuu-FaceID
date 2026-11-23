# Independent Linux deployment bundle

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

This directory contains trained models and configuration.
GitHub Actions publishes the service image to `tano26/seiyuu-faceid`.
At runtime, detection, alignment, and identity recognition all take place locally on the target server.
Operation does not depend on the development computer, training photos, a host Python installation, PyPI, or external inference services.
Preinstall Docker Engine and Docker Compose v2 on the server. The image architecture must match the server.

<a id="启动"></a>

## Start

Work in this directory. Verify every file before importing the image.

```bash
sha256sum --check SHA256SUMS
cp .env.example .env
chmod 600 .env
```

Generate a key with `openssl rand -hex 32` and enter it after `SEIYUU_FACEID_API_KEY=` in `.env`.
The default listener uses port 8000 on all server interfaces. Change the bind address and port in `.env` as needed.
After the image is published, pull it and start the service.

```bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
curl --fail http://127.0.0.1:8000/ready
```

If initial model loading exceeds 180 seconds, inspect logs before increasing the wait timeout.
The service downloads no dependencies or models at runtime.
Compose disables automatic image pulls by default. Later starts and recreation require no Docker Hub connection.
Only installation or updates explicitly use the pull command above.
Both the host and container need read access to `models/` and `configs/default.yaml`.

If packaging also included `image.tar`, the target server can start fully offline.
Import the matching architecture's image with `docker load --input image.tar`, then run `docker compose up -d --pull never --wait --wait-timeout 180` without `docker compose pull`.
Docker and Compose themselves must be preinstalled.
Before updating, back up the image version in `.env`, then pull the new version and run `docker compose up -d`.

<a id="远程调用"></a>

## Remote calls

Set client variables on the calling device, using the same key as the server's `.env`.

```bash
export FACEID_URL='http://服务器IP:8000'
export FACEID_API_KEY='你填写的密钥'
curl --fail-with-body "$FACEID_URL/v1/model" \
  -H "Authorization: Bearer $FACEID_API_KEY"
curl --fail-with-body "$FACEID_URL/v1/recognize" \
  -H "Authorization: Bearer $FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

Allow callers to reach the selected port. Use an HTTPS reverse proxy for public services.
See [API.md](../docs/API.md) for the complete interface. Online documentation is at `/docs`.
The home test page is disabled by default.

<a id="运维"></a>

## Operations

```bash
docker compose ps
docker compose logs --tail 100 faceid
docker stats --no-stream
docker compose down
```

Manual startup after a normal shutdown still requires `docker compose up -d`.
A running service recovers from crashes or host reboots under the `unless-stopped` policy.
Health checks mark failures, but `unhealthy` alone does not restart a container.
After configuration changes, run `docker compose restart faceid`.
After `.env` changes, run `docker compose up -d --force-recreate`.

The service currently uses CPU inference and one worker.
A standard test server can start with 2 cores and 4 GiB of memory for capacity planning.
Minimum memory, latency, and throughput require measurement on the target server.
Weights total approximately 417 MiB, which does not equal runtime memory usage.
Images with multiple faces increase computation time.
Existing weights support 50 trained identities, and migration requires no retraining.

N150 + 8 GB Linux hosts can use the bundled CPU-saving configuration with a new image containing the corresponding code.

```bash
docker compose -f compose.yaml -f compose.n150.yaml up -d --wait --wait-timeout 180
```

This configuration keeps the original FP32 models and parallel dual backbones, with one thread per backbone and spinning disabled.
Measure actual latency, CPU, and memory on the target host.
See the [N150 guide](../docs/N150.md) for procedures.

Configuration and models reside in this directory, so backing up the entire directory supports migration.
The deployment contains local models. Store it within your intended access scope.
See `models/selected/MODEL_CARD.md` and `manifest.json` for model sources and permitted use.
