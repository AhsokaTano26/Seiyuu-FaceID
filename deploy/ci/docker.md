# Docker Compose deployment bundle

[English](docker.md) | [简体中文](docker.zh-CN.md) | [日本語](docker.ja.md)

Preinstall Docker Engine and Compose v2. This bundle excludes models and keys.
.env.example pins the current commit's Docker Hub image tag to prevent accidental use of a later latest.
The image supports Linux amd64 / arm64 and runs the Python API.
For the fast native C++ / Go version, select the separate native-linux-amd64 artifact.

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
cp .env.example .env
chmod 600 .env
~~~

Initial model download requires Python 3.11+.
Alternatively, download on another machine and copy models/.
For an already downloaded archive, add --archive /path/to/models.tar.gz, replacing the path placeholder.
Generate your own key with openssl rand -hex 32, enter it as SEIYUU_FACEID_API_KEY in .env, then run the following commands.

~~~bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
curl --fail http://127.0.0.1:8000/ready
~~~

For the N150 CPU-saving thread configuration, run `docker compose -f compose.yaml -f compose.n150.yaml up -d --wait --wait-timeout 180`.
This does not enable native FTZ/DAZ.

For a fully offline installation, export an image of the matching architecture with docker save in advance, load it with docker load on the target host, then start.
See [API documentation](../../docs/API.md).
The upload homepage is disabled by default. Python interface documentation is available at /docs.
