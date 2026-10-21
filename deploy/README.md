# Linux 独立部署包

此目录包含已训练模型和配置，服务镜像由 GitHub Actions 发布到
`tano26/seiyuu-faceid`。运行时在目标服务器本地完成检测、
对齐与身份识别，不依赖开发电脑、训练图片、Python 安装、PyPI 或外部推理服务。
目标服务器需预先安装 Docker Engine 和 Docker Compose v2；镜像架构必须与服务器匹配。

## 启动

在本目录操作。先校验全部文件，再导入镜像：

```bash
sha256sum --check SHA256SUMS
cp .env.example .env
chmod 600 .env
```

用 `openssl rand -hex 32` 生成密钥，将其填写到 `.env` 的
`SEIYUU_FACEID_API_KEY=` 后。默认监听服务器所有网卡的 8000 端口；可在 `.env`
修改绑定地址和端口。镜像发布后拉取并启动：

```bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
curl --fail http://127.0.0.1:8000/ready
```

如果第一次加载模型超过 180 秒，先查看日志，再增加等待时间。
服务运行时不下载依赖或模型。Compose 默认禁止自动拉取镜像，后续启动和重建
也不需要连接 Docker Hub；安装或更新时才显式执行上面的拉取命令。
主机和容器中都需要能读取 `models/` 和 `configs/default.yaml`。

如果打包时额外包含了 `image.tar`，目标服务器也可完全离线启动：用
`docker load --input image.tar` 导入匹配架构的镜像，再运行
`docker compose up -d --pull never --wait --wait-timeout 180`，无需执行 `docker compose pull`。
Docker 和 Compose 自身需提前安装。更新镜像时可先备份 `.env` 中的镜像版本，
再拉取新版本并运行 `docker compose up -d`。

## 远程调用

调用设备设置自己的客户端变量；密钥与服务器 `.env` 中一致：

```bash
export FACEID_URL='http://服务器IP:8000'
export FACEID_API_KEY='你填写的密钥'
curl --fail-with-body "$FACEID_URL/v1/model" \
  -H "Authorization: Bearer $FACEID_API_KEY"
curl --fail-with-body "$FACEID_URL/v1/recognize" \
  -H "Authorization: Bearer $FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

允许调用方访问所选端口；公网服务使用 HTTPS 反向代理。完整接口见
[API.md](docs/API.md)，在线文档为 `/docs`。首页测试页面默认关闭。

## 运维

```bash
docker compose ps
docker compose logs --tail 100 faceid
docker stats --no-stream
docker compose down
```

正常退出后手动启动仍需 `docker compose up -d`；运行中的服务会按
`unless-stopped` 策略在崩溃或主机重启后恢复。健康检查标记异常，不会单凭
`unhealthy` 自动重启容器。修改配置后运行 `docker compose restart faceid`；
修改 `.env` 后运行 `docker compose up -d --force-recreate`。

当前为 CPU 推理、单 worker。普通测试服务器可先按 2 核、4 GiB 内存准备，
这是容量规划起点，最低内存、延迟和吞吐需要在目标服务器实测。模型权重约
417 MiB，不等于实际运行内存；多人脸图片会增加计算时间。
现有权重支持 50 个已训练身份，迁移无需重新训练。

配置和模型位于此目录下，备份整个目录即可迁移。部署包含本地模型，请按自己的
访问范围保存；模型来源与使用范围见 `models/selected/MODEL_CARD.md` 和
`manifest.json`。
