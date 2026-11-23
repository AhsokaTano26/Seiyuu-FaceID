# Docker Compose 部署包

[English](docker.md) | [简体中文](docker.zh-CN.md) | [日本語](docker.ja.md)

需要预装 Docker Engine、Compose v2。模型、密钥不在此包。
.env.example 固定当前提交的 Docker Hub 镜像标签，防止误用后续 latest。
镜像支持 Linux amd64 / arm64，运行的是 Python API。
原生 C++ / Go 快速版本应选择单独的 native-linux-amd64 产物。

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
cp .env.example .env
chmod 600 .env
~~~

首次模型下载需要 Python 3.11+，也可以在其他机器下载后复制 models/。
使用预先下载的归档时加 --archive /路径/模型包.tar.gz。
用 openssl rand -hex 32 生成自己的密钥，填入 .env 的 SEIYUU_FACEID_API_KEY，再执行 —

~~~bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
curl --fail http://127.0.0.1:8000/ready
~~~

N150 省 CPU 线程配置（不会开启原生 FTZ/DAZ） —
docker compose -f compose.yaml -f compose.n150.yaml up -d --wait --wait-timeout 180。

完整离线安装可提前 docker save 导出同一架构镜像，目标机 docker load 后启动。
API 见 [API 文档](../../docs/API.zh-CN.md)。默认关闭上传首页，Python 接口文档在 /docs。
