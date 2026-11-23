# Native Python service deployment bundle

[English](python.md) | [简体中文](python.zh-CN.md) | [日本語](python.ja.md)

This bundle contains the current commit's Python source, locked dependencies, configuration, and systemd template, excluding models and keys.
Model sources and per-file verification are pinned in release/model-source.json.
You need Linux, uv, and Python 3.13 for the API runtime.
The initial downloader requires Python 3.11+. You may download on another machine and copy models/.

First install the system runtime libraries needed by OpenCV. On Debian / Ubuntu, run the following commands.

~~~bash
sudo apt-get update
sudo apt-get install -y ca-certificates libgl1 libglib2.0-0 libgomp1
~~~

Other distributions must provide equivalent OpenGL, GLib, and OpenMP runtime libraries or importing cv2 will fail.

Run the following commands in the extracted directory.

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
uv sync --locked --no-dev --extra api --python 3.13
export SEIYUU_FACEID_API_KEY='填写自己的随机密钥'
SEIYUU_FACEID_CONFIG=configs/n150.yaml uv run --no-sync uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
~~~

HTTP API and /docs are provided by default.
The N150 configuration retains the original models and recognition thresholds and has no C++ FTZ/DAZ acceleration.
For production, place the directory at /opt/seiyuu-faceid, configure the faceid user following deploy/seiyuu-faceid.service, copy deploy/faceid-native.env.example to /etc/seiyuu-faceid.env, and enter a key.
Then install the template, run daemon-reload, and enable --now.
The service user needs read access to the code, models, and .venv.
Keep the environment file owned by root:root with mode 0600.
The service template disables the upload homepage by default. Enable it by changing the environment variable and restarting.

The initial uv sync installation needs network access.
For offline model retrieval, run `python3 tools/fetch_models.py --archive /path/to/downloaded-models.tar.gz` with the downloaded archive path substituted.
After extraction and file verification, operation needs no connection for model downloads.
See [API documentation](../../docs/API.md).
