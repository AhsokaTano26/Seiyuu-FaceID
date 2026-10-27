# Linux 原生部署（无 Docker）

以下示例面向 Debian / Ubuntu 的 amd64 主机，以 `/opt/seiyuu-faceid` 为安装目录，
使用 systemd 管理一个 API worker。N150 + 8 GB 可先使用 `configs/n150.yaml`；
模型、FP32 精度、检测和接受阈值均保持不变。当前没有可访问的目标 Linux 主机，
下列流程尚未在该目标机实测，性能数据见 [N150 指南](N150.md) 中的范围说明。

Python 负责 API 和调度，主要模型计算已经由 ONNX Runtime 的原生代码执行。
可以直接去掉 Docker 部署这一层；是否降低延迟或内存需在同一主机对照测量，
现阶段没有依据承诺重写成其他语言能改善端到端性能。

## 安装代码、运行时和模型

准备包含本次原生部署文件及资源配置项的代码 checkout。模型不在 Git 仓库内，
需另外取得有使用权限的模型文件及配套条款；文件清单见
[Linux 部署指南](DEPLOY_LINUX.md#2-生成模型和部署配置包)。不要复制训练照片、
训练特征、测试数据或本机密钥。

先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)
安装 uv，并将可执行文件放到 `/usr/local/bin/uv`。下例假定该路径已可用，
不需要另行安装系统 Python 3.13。

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates openssl libgl1 libglib2.0-0 libgomp1
sudo useradd --system --user-group --home-dir /opt/seiyuu-faceid --no-create-home --shell /usr/sbin/nologin faceid
sudo git clone https://github.com/AhsokaTano26/Seiyuu-FaceID.git /opt/seiyuu-faceid
cd /opt/seiyuu-faceid
```

如果已存在 `faceid` 用户或已有代码目录，复用它们并跳过对应创建命令；确认代码
版本包含本指南和 `deploy/seiyuu-faceid.service`。将私有模型放入以下位置，
保留 `MODEL_CARD.md` 和 `ADAFACE_LICENSE` 等配套条款：

```text
/opt/seiyuu-faceid/models/det_500m.onnx
/opt/seiyuu-faceid/models/selected/adaface_ir101.onnx
/opt/seiyuu-faceid/models/selected/w600k_r50.onnx
/opt/seiyuu-faceid/models/selected/classifier.npz
/opt/seiyuu-faceid/models/selected/manifest.json
```

在仓库目录安装锁定的 API 依赖：

```bash
sudo env UV_PYTHON_INSTALL_DIR=/opt/seiyuu-faceid/.python \
  UV_CACHE_DIR=/opt/seiyuu-faceid/.uv-cache UV_LINK_MODE=copy \
  UV_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=automatic \
  /usr/local/bin/uv sync --locked --no-dev --extra api --python 3.13
sudo chown -R faceid:faceid /opt/seiyuu-faceid
sudo -u faceid /opt/seiyuu-faceid/.venv/bin/python -c \
  'import cv2, numpy, onnxruntime; from seiyuu_faceid.api.app import create_app'
```

这一步需要下载 Python 和依赖。托管 Python 固定放在 `.python`，避免 `.venv`
指向只有 root 可访问的个人缓存；`UV_LINK_MODE=copy` 避免安装包依赖缓存中的链接。
最后统一目录所有权，让服务用户能读取模型、配置、依赖并执行 Python。
运行时直接调用 `.venv`，不调用 uv 或自动下载运行时。相关变量见
[uv 官方环境变量说明](https://docs.astral.sh/uv/reference/environment/)。

## 启动服务

```bash
sudo install -m 0600 deploy/faceid-native.env.example /etc/seiyuu-faceid.env
sudoedit /etc/seiyuu-faceid.env
```

启动前必须将 `SEIYUU_FACEID_API_KEY` 填为自己的非空随机密钥，可用
`openssl rand -hex 32` 生成；空值会关闭 API 鉴权。环境文件由 systemd 读取，
无需让 `faceid` 用户读取该文件。不要把填写后的文件提交到仓库。

```bash
sudo install -m 0644 deploy/seiyuu-faceid.service /etc/systemd/system/seiyuu-faceid.service
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
sudo systemctl status seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
```

服务默认只监听 `127.0.0.1:8000`。需要远程调用时，可在现有反向代理后接入，
并保留 Bearer API 密钥验证；接口约定见 [API 文档](API.md)。日志使用
`sudo journalctl -u seiyuu-faceid.service -f` 查看。修改配置或环境文件后执行
`sudo systemctl restart seiyuu-faceid.service`。服务给请求 60 秒优雅退出时间，
systemd 的停止超时为 90 秒。

更新代码及依赖前先停止服务；在同一目录重新运行上面的 `uv sync` 和 `chown`
命令，再启动服务。模型仍由本机读取，不需要开发电脑在线。

## 在同一主机比较资源

先停止 API，避免测速进程和服务争抢 CPU、同时保留两份模型：

```bash
sudo systemctl stop seiyuu-faceid.service
cd /opt/seiyuu-faceid
sudo -u faceid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_target.py --config configs/n150.yaml \
  --synthetic --samples 8 --runs 5 --warmups 2 --repeats 2 \
  --output /tmp/n150-native-benchmark.json > /tmp/n150-native-stdout.json
sudo systemctl start seiyuu-faceid.service
```

工具要求输出路径尚不存在，每次复测请更换 `/tmp` 中的两个文件名。标准输出为
完整 JSON，保存在 `/tmp/n150-native-stdout.json`；`--output` 同时保存服务用户写入的副本。此基准不需要
API 密钥，不读取训练目录。它测量对齐裁剪的特征提取和分类，检测器加载但不运行，
不代表真人准确率或完整 HTTP 请求耗时。

若比较原生与 Docker，使用同一台机器、同一份模型、配置、输入、Python 和
ONNX Runtime 版本，以及相同线程环境变量；两种模式依次运行，避免竞争。查看报告
中的版本、CPU 时间、wall time 和峰值 RSS，并另外测真实 API 请求及服务空闲时
内存。容器总体内存与进程 RSS 的口径不同，不能直接当作 Docker 开销相减。
不根据本机 macOS 结果推断 N150、Linux 或 Docker 的性能差异。
