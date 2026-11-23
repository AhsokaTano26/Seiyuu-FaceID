# Native Linux deployment without Docker

[English](DEPLOY_NATIVE_LINUX.md) | [简体中文](DEPLOY_NATIVE_LINUX.zh-CN.md) | [日本語](DEPLOY_NATIVE_LINUX.ja.md)

The examples target Debian / Ubuntu amd64 hosts, install into `/opt/seiyuu-faceid`, and use systemd to manage one API worker.
For N150 + 8 GB, start with `configs/n150.yaml`. Models, FP32 precision, detection thresholds, and acceptance thresholds remain unchanged.
At the time these instructions were written, no target Linux host was accessible and this procedure had not been tested on that target.
See the scope notes in the [N150 guide](N150.md) for performance data.

Python handles the API and scheduling. ONNX Runtime's native code already performs the main model computations.
Deployment can omit Docker directly. Measure latency and memory on the same host to determine the effect.
At this stage there is no evidence supporting a promise that rewriting in another language improves end-to-end performance.

<a id="安装代码运行时和模型"></a>

## Install code, runtime, and models

Prepare a code checkout containing the native deployment files and resource configuration options introduced here.
Models are excluded from Git. Obtain authorized model files and their accompanying terms separately.
See the file list in the [Linux deployment guide](DEPLOY_LINUX.md#2-生成模型和部署配置包).
Do not copy training photos, training features, test data, or local keys.

First install uv following its [official instructions](https://docs.astral.sh/uv/getting-started/installation/) and place the executable at `/usr/local/bin/uv`.
The example assumes that path is available. A separate system Python 3.13 installation is unnecessary.

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates openssl libgl1 libglib2.0-0 libgomp1
sudo useradd --system --user-group --home-dir /opt/seiyuu-faceid --no-create-home --shell /usr/sbin/nologin faceid
sudo git clone https://github.com/AhsokaTano26/Seiyuu-FaceID.git /opt/seiyuu-faceid
cd /opt/seiyuu-faceid
```

If the `faceid` user or code directory already exists, reuse it and skip its creation command.
Check that the code version contains this guide and `deploy/seiyuu-faceid.service`.
Place the private models at the following locations, retaining accompanying terms such as `MODEL_CARD.md` and `ADAFACE_LICENSE`.

```text
/opt/seiyuu-faceid/models/det_500m.onnx
/opt/seiyuu-faceid/models/selected/adaface_ir101.onnx
/opt/seiyuu-faceid/models/selected/w600k_r50.onnx
/opt/seiyuu-faceid/models/selected/classifier.npz
/opt/seiyuu-faceid/models/selected/manifest.json
```

Install locked API dependencies from the repository directory.

```bash
sudo env UV_PYTHON_INSTALL_DIR=/opt/seiyuu-faceid/.python \
  UV_CACHE_DIR=/opt/seiyuu-faceid/.uv-cache UV_LINK_MODE=copy \
  UV_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=automatic \
  /usr/local/bin/uv sync --locked --no-dev --extra api --python 3.13
sudo chown -R faceid:faceid /opt/seiyuu-faceid
sudo -u faceid /opt/seiyuu-faceid/.venv/bin/python -c \
  'import cv2, numpy, onnxruntime; from seiyuu_faceid.api.app import create_app'
```

This step downloads Python and dependencies.
Managed Python is stored at `.python` to keep `.venv` from pointing to a personal cache accessible only by root.
`UV_LINK_MODE=copy` prevents installed packages from depending on links to the cache.
Finally, set directory ownership consistently so the service user can read models, configuration, and dependencies and execute Python.
At runtime, call `.venv` directly without invoking uv or automatically downloading a runtime.
See the [uv environment variable reference](https://docs.astral.sh/uv/reference/environment/) for these variables.

<a id="启动服务"></a>

## Start the service

```bash
sudo install -m 0600 deploy/faceid-native.env.example /etc/seiyuu-faceid.env
sudoedit /etc/seiyuu-faceid.env
```

Before starting, set `SEIYUU_FACEID_API_KEY` to your own nonempty random key, generated for example with `openssl rand -hex 32`.
An empty value disables API authentication.
The environment file is read by systemd, so `faceid` does not need read access to it.
Do not commit the completed file to the repository.

```bash
sudo install -m 0644 deploy/seiyuu-faceid.service /etc/systemd/system/seiyuu-faceid.service
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
sudo systemctl status seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
```

The service listens only on `127.0.0.1:8000` by default.
For remote calls, connect through an existing reverse proxy while retaining Bearer API key validation.
See the [API documentation](API.md) for the interface contract.
View logs with `sudo journalctl -u seiyuu-faceid.service -f`.
After changing configuration or the environment file, run `sudo systemctl restart seiyuu-faceid.service`.
Requests receive a 60-second graceful shutdown period. The systemd stop timeout is 90 seconds.

Stop the service before updating code and dependencies.
Run the preceding `uv sync` and `chown` commands again in the same directory, then start the service.
Models are still read locally, with no need for the development computer to remain online.

<a id="在同一主机比较资源"></a>

## Compare resources on the same host

Stop the API first to avoid CPU contention with the benchmark and loading two model copies simultaneously.

```bash
sudo systemctl stop seiyuu-faceid.service
cd /opt/seiyuu-faceid
sudo -u faceid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_target.py --config configs/n150.yaml \
  --synthetic --samples 8 --runs 5 --warmups 2 --repeats 2 \
  --output /tmp/n150-native-benchmark.json > /tmp/n150-native-stdout.json
sudo systemctl start seiyuu-faceid.service
```

The tool requires output paths that do not yet exist.
Use two new `/tmp` filenames for every rerun.
Standard output is complete JSON saved to `/tmp/n150-native-stdout.json`.
`--output` also saves a copy written by the service user.
This benchmark requires no API key and does not read training directories.
It measures feature extraction and classification of aligned crops.
The detector is loaded but does not run. These measurements do not represent accuracy on real people or full HTTP request duration.

To compare native deployment with Docker, use the same host, models, configuration, inputs, Python version, ONNX Runtime version, and thread environment variables.
Run the two modes sequentially to avoid contention.
Inspect versions, CPU time, wall time, and peak RSS in the report.
Also measure real API requests and idle service memory separately.
Total container memory and process RSS use different accounting scopes, so subtracting them does not measure Docker overhead.
Do not infer N150, Linux, or Docker performance differences from local macOS results.
