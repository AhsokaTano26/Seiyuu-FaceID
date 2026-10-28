# Python 原生服务部署包

这是当前提交的 Python 源码、锁定依赖、配置及 systemd 模板，不含模型或密钥。
模型来源与逐文件校验固定在 release/model-source.json。需要 Linux、uv 和运行 API 的 Python 3.13。
首次下载工具需要 Python 3.11+；可以在另一台机器下载后复制 models/。

先安装 OpenCV 所需的系统运行库。Debian / Ubuntu 执行：

~~~bash
sudo apt-get update
sudo apt-get install -y ca-certificates libgl1 libglib2.0-0 libgomp1
~~~

其他发行版需要提供等价的 OpenGL、GLib 与 OpenMP 运行库，否则导入 cv2 时会失败。

在解压目录中执行：

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
uv sync --locked --no-dev --extra api --python 3.13
export SEIYUU_FACEID_API_KEY='填写自己的随机密钥'
SEIYUU_FACEID_CONFIG=configs/n150.yaml uv run --no-sync uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
~~~

默认提供 HTTP API 和 /docs。N150 配置保留原模型与识别阈值；它没有 C++ 的 FTZ/DAZ 加速。
生产环境可将目录放在 /opt/seiyuu-faceid，按 deploy/seiyuu-faceid.service 设置 faceid 用户，
把 deploy/faceid-native.env.example 复制到 /etc/seiyuu-faceid.env 并填写密钥，
然后安装模板、daemon-reload、enable --now。服务用户需要读取代码、模型和 .venv；
环境文件保留 root:root、0600。服务模板默认关闭上传首页，修改环境变量后重启可开启。

首次安装 uv sync 需要网络。离线模型获取：
python3 tools/fetch_models.py --archive /路径/已下载的模型包.tar.gz。
解压和文件校验完成后运行不需要模型下载连接。接口见 docs/API.md。
