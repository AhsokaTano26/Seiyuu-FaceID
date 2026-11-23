# C++ inference + Go HTTP deployment bundle for Linux amd64

[English](native.md) | [简体中文](native.zh-CN.md) | [日本語](native.ja.md)

This bundle contains a Linux x86_64 runtime, shared libraries, licenses, and JSON exports of the original model parameters.
The target host requires no compilation and no resident Python or Docker.
The build baseline is Ubuntu 22.04 with glibc 2.35+.
The target system also needs standard runtime libraries including libstdc++, libgcc, libm, and libpthread.
Verify actual compatibility with dynamic linking checks on the target host.
Do not use this binary on another architecture.

ONNX weights are downloaded from the fixed source in release/model-source.json and are not duplicated in every application bundle.
The initial downloader requires Python 3.11+.
You can also run it elsewhere and copy models/. Python is only needed for this preparation step.

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
export SEIYUU_FACEID_API_KEY='填写自己的随机密钥'
runtime/bin/faceid-server -model models/selected/native-model.json -addr 127.0.0.1:8000
~~~

native-model.json enables the FTZ/DAZ fast path validated on N150 by default, retaining FP32 weights, the classification head, and recognition thresholds.
For compatible floating-point mode, use models/selected/native-compatible.json and retain its corresponding .sha256 file.
Relative model paths in the JSON are already configured. Preserve the directory layout.

For systemd, place the extracted contents in `/opt/seiyuu-faceid/releases/<version>`, substituting the version number, and point /opt/seiyuu-faceid/current to it.
Create /etc/seiyuu-faceid/api.env owned by root:root with mode 0600, write `SEIYUU_FACEID_API_KEY=<your-key>` with your own key, then run the following commands.

~~~bash
sudo install -m 0644 deploy/seiyuu-faceid.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
~~~

The template listens on 0.0.0.0:8000. Avoid port conflicts with old containers or other services before starting.
The service provides an API without an upload homepage or /docs.
See [API documentation](../../docs/API.md) for interface fields.

For offline model retrieval, run `python3 tools/fetch_models.py --archive /path/to/models.tar.gz` with the archive path substituted.
Once models, runtime, and metadata are prepared, operation does not depend on the development computer or external inference services.
