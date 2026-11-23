# C++ 推理 + Go HTTP — Linux amd64 部署包

[English](native.md) | [简体中文](native.zh-CN.md) | [日本語](native.ja.md)

这是 Linux x86_64 运行时、共享库、许可证和原模型参数的 JSON 导出。
目标机无需编译，也无需 Python/Docker 常驻。构建基线为 Ubuntu 22.04（glibc 2.35+）。
目标系统还需提供 libstdc++、libgcc、libm、libpthread 等标准系统运行库。
实际兼容性由目标机动态链接检查确认。其他架构不要使用此二进制。

ONNX 权重按 release/model-source.json 固定下载，不在每个应用包中重复存储。
首次下载工具需要 Python 3.11+，也可在其他机器运行后复制 models/。
这不是服务运行时的 Python 依赖。

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
export SEIYUU_FACEID_API_KEY='填写自己的随机密钥'
runtime/bin/faceid-server -model models/selected/native-model.json -addr 127.0.0.1:8000
~~~

native-model.json 默认开启 N150 已验证的 FTZ/DAZ 快速路径，保留 FP32 权重、
分类头及识别阈值。兼容浮点模式使用 models/selected/native-compatible.json，
同时保留对应 .sha256 文件。JSON 相对模型路径已经设置，保留目录布局。

要用 systemd，将解压内容放到 /opt/seiyuu-faceid/releases/版本号，
让 /opt/seiyuu-faceid/current 指向该目录。创建 root:root、0600 的
/etc/seiyuu-faceid/api.env，写入 SEIYUU_FACEID_API_KEY=自己的密钥，然后执行 —

~~~bash
sudo install -m 0644 deploy/seiyuu-faceid.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
~~~

模板监听 0.0.0.0:8000。启动前避免与旧容器或其他服务端口冲突。
本服务提供 API，没有上传首页和 /docs。接口字段见 [API 文档](../../docs/API.zh-CN.md)。

离线模型获取 — python3 tools/fetch_models.py --archive /路径/模型包.tar.gz。
模型、运行时和元信息均准备完成后，运行不依赖开发电脑或外部推理服务。
