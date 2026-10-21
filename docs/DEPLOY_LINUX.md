# 在 Linux 服务器完全独立部署

推荐继续使用 Python + FastAPI + ONNX Runtime，先部署 CPU 版本。识别完全在目标
服务器上执行，不调用开发电脑或外部模型 API，不需要重新训练。运行所需内容为：

- Docker 服务镜像：Python、锁定的 API 依赖、项目代码和 Linux 原生库。
- 本地模型：SCRFD 检测器、AdaFace IR101、R50、分类头和模型清单。
- 配置和 Compose：监听端口、密钥、健康检查、日志和重启策略。

目标主机预先安装 Docker Engine 和 Docker Compose v2。先按 2 核、4 GiB 内存准备
测试环境；这不是已验证的最低配置，实际容量取决于目标 CPU、图片中人脸数量和
并发量。程序默认使用 2 个推理线程、1 个 worker，当前实现只使用 CPU。
即使服务器有 GPU，也不会自动启用 GPU 加速。

## 1. 通过 GitHub Actions 发布到 Docker Hub

工作流为仓库中的 `.github/workflows/ci.yml`，镜像仓库为
[`tano26/seiyuu-faceid`](https://hub.docker.com/r/tano26/seiyuu-faceid)。
在 GitHub 的 [Actions Secrets](https://github.com/AhsokaTano26/Seiyuu-FaceID/settings/secrets/actions)
添加 `DOCKERHUB_TOKEN`，值为具备该 Docker 仓库推送权限的访问令牌。
登录用户名已固定为 `tano26`，无需额外配置用户名变量。

将包含工作流和 Dockerfile 的项目代码推送到 `main` 后，先执行 lint、测试和编译
检查，再构建并推送镜像；也可以在 Actions 中手动运行 `CI and Docker Hub`。
PR 只执行质量检查，不登录或发布到 Docker Hub。

| 触发方式 | 镜像标签 |
| --- | --- |
| 推送 `main` 或在 `main` 手动运行 | `latest`、`sha-完整提交SHA` |
| 推送版本标签，如 `v0.1.0` | `0.1.0`、`sha-完整提交SHA` |

版本标签不覆盖 `latest`。工作流发布 `linux/amd64` 和 `linux/arm64` 两种架构，
目标服务器拉取时自动选择匹配版本。Python 依赖按 `uv.lock` 安装，只启用 API，
不安装训练、Streamlit 或测试依赖；每种架构都会在构建时验证原生库和 API 模块
可导入。镜像构建不需要训练权重。

Docker 构建上下文采用允许列表，不包含 `models/`、照片、实验数据、`.env` 或本机
`.venv`。GitHub Runner 上也没有这些私有模型。模型按下一节单独分发，到达目标
服务器后只读挂载；运行时不从开发电脑读取文件。

## 2. 生成模型和部署配置包

仅克隆 Git 仓库不够：模型被 `.gitignore` 排除。打包必须在持有当前完整模型的
机器上执行，当前需要以下文件：

```text
models/det_500m.onnx
models/selected/adaface_ir101.onnx
models/selected/w600k_r50.onnx
models/selected/classifier.npz
models/selected/manifest.json
models/selected/MODEL_CARD.md
models/selected/ADAFACE_LICENSE
```

使用仅依赖 Python 标准库的打包工具：

```bash
python3 tools/package_deployment.py \
  --output dist/seiyuu-faceid-deployment
tar -C dist -czf dist/seiyuu-faceid-deployment.tar.gz seiyuu-faceid-deployment
```

打包工具核对四个模型文件的 SHA-256，生成整个部署包的 `SHA256SUMS`，拒绝覆盖
已有输出目录。输出包含模型、配置、Compose、空密钥示例和独立运行
说明；不包含训练数据、测试照片、评测特征或本机密钥。
更换镜像标签时，需同步修改部署包 `.env` 中的 `FACEID_IMAGE`。

将生成的 `.tar.gz` 文件传到目标服务器。服务镜像从 Docker Hub 拉取一次后
可独立运行；目标服务器不需要安装 Python、uv 或训练依赖，也不访问 PyPI。
Docker 与 Compose 自身应提前安装好。

如果目标服务器无法访问 Docker Hub，可以在能联网的机器上拉取对应架构的镜像
并导出，然后通过打包工具的 `--image-tar` 参数一并放入部署包：

```bash
docker pull --platform linux/amd64 tano26/seiyuu-faceid:latest
mkdir -p dist
docker image save --output dist/image-linux-amd64.tar tano26/seiyuu-faceid:latest
python3 tools/package_deployment.py \
  --image-tar dist/image-linux-amd64.tar \
  --output dist/seiyuu-faceid-offline
tar -C dist -czf dist/seiyuu-faceid-offline.tar.gz seiyuu-faceid-offline
```

在服务器用 `uname -m` 确认架构：`x86_64` 对应 `linux/amd64`，`aarch64` 对应
`linux/arm64`。离线导出的镜像需要与目标架构匹配。

## 3. 在目标 Linux 服务器启动

```bash
tar -xzf seiyuu-faceid-deployment.tar.gz
cd seiyuu-faceid-deployment
sha256sum --check SHA256SUMS
cp .env.example .env
chmod 600 .env
```

执行 `openssl rand -hex 32` 生成密钥，将密钥填写到 `.env` 的
`SEIYUU_FACEID_API_KEY=` 后。默认在所有网卡监听 8000 端口；需要其他端口时
修改 `FACEID_PORT`，已有 HTTPS 反向代理时可将 `FACEID_BIND_ADDRESS` 改成
`127.0.0.1`。生产部署可将 `FACEID_IMAGE` 从 `latest` 改为固定的版本或 SHA 标签。
镜像发布成功后，拉取并启动服务：

```bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
docker compose ps
curl --fail http://127.0.0.1:8000/ready
```

预期就绪响应为 `{"status":"ready"}`。`/ready` 成功表示模型已加载；
`/v1/recognize` 成功返回才验证了完整的图片识别链路。
容器运行时无需出站联网。Compose 默认禁止自动拉取镜像，后续启动和重建也可
离线进行；安装或更新时通过 `docker compose pull --policy always` 显式拉取。
模型与配置必须能被容器 UID 10001 读取；打包工具已将模型和配置设为可读权限。
如果手动复制模型，请保留文件的读取权限和目录的遍历权限。

离线部署包中包含 `image.tar` 时，用 `docker load --input image.tar` 导入镜像，
再使用 `docker compose up -d --pull never --wait --wait-timeout 180` 启动，
跳过 `docker compose pull`；Compose 不会尝试联网拉取镜像。

从另一台设备调用：

```bash
export FACEID_URL='http://服务器IP:8000'
export FACEID_API_KEY='与服务端一致的密钥'
curl --fail-with-body "$FACEID_URL/v1/model" \
  -H "Authorization: Bearer $FACEID_API_KEY"
curl --fail-with-body "$FACEID_URL/v1/recognize" \
  -H "Authorization: Bearer $FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

服务器防火墙或云安全组需允许调用方访问所选端口。公网使用 HTTPS；
接口格式及返回状态见 [API.md](API.md)。模型内含已训练的 50 个身份。
默认低置信度或质量不足返回 `review`，迁移部署不会改变这一规则。

## 4. 检查和维护

```bash
docker compose logs --tail 100 faceid
docker stats --no-stream
docker compose pull --policy always
docker compose up -d
docker compose restart faceid
docker compose down
```

健康检查请求 `/ready`。进程退出后可按 `unless-stopped` 策略重启；健康状态异常
本身不会触发自动重启。修改 `.env` 后用 `docker compose up -d --force-recreate`，
修改配置或替换模型后重启服务。备份部署目录和 `.env` 即可保留运行所需内容。

目标设备验收至少检查：`/ready`、带密钥的 `/v1/model`、一张已知身份照片的
`/v1/recognize`、缺少密钥返回 401、重启后的恢复。再记录内存和识别延迟，决定
是否调整线程或扩容。模型文件合计约 417 MiB，运行内存需另测；增加 worker
通常也会增加模型内存占用。

Docker 用法参考：[uv Docker 集成](https://docs.astral.sh/uv/guides/integration/docker/)、
[Actions 镜像发布](https://docs.docker.com/build/ci/github-actions/push-multi-registries/)、
[Compose 环境变量](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)、
[Compose 服务配置](https://docs.docker.com/reference/compose-file/services/)。
