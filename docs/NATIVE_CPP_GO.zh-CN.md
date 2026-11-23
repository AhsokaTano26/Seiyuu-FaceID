# C++ 推理核心 + Go HTTP

[English](NATIVE_CPP_GO.md) | [简体中文](NATIVE_CPP_GO.zh-CN.md) | [日本語](NATIVE_CPP_GO.ja.md)

本版本用于验证 Intel N150 / 8 GB 上原生运行时的收益。一个 Go 进程通过 cgo
直接调用 C++，SCRFD、AdaFace IR101、R50 加载一次。双骨干并行，HTTP 推理请求
依次执行。模型保持 FP32，使用原有分类头和 0.95 默认阈值。

算法原理、训练评估与各模块协作见 [识别技术框架详解](TECHNICAL_ARCHITECTURE.zh-CN.md)。

已在 N150 完成 Linux 构建与启动、1,051 张裁剪和 12 张原图的一致性对照，
以及完整三组 HTTP 基线。基线中同配置原生实现没有显示可靠提速，峰值 RSS 更高。
省 CPU 线程预设相对旧线程组的平均 HTTP 耗时增加约 37.9%。完整口径与后续
独立实验见 [N150 实机报告](N150_BENCHMARK.zh-CN.md)。2026-10-28 已在目标机使用
FTZ/DAZ 版本接管原 API，长期和压力场景仍待验收。

后续独立 FTZ/DAZ 候选已完成全量回归与 HTTP 测量，平均请求耗时约 0.681 秒，
历史留出 Top-1 保持 98.48%。该实现现已集成至 `native/`，构建后需在模型导出时
显式加 `--flush-denormals` 开启。仅 Linux x86_64 且 CPU 支持 FTZ/DAZ 时可开启，
Mac ARM 会明确拒绝。省略参数仍为默认关闭，兼容原有 JSON。结果、内存取舍及
分阶段比较边界见 [FTZ 实验章节](N150_BENCHMARK.zh-CN.md)。

这不是纯静态单文件程序。部署需要 `bin/faceid-server`、旁边的 `lib/` 共享库、
模型目录及元信息。服务运行时不需要 Python。导出模型参数和对照验证使用 Python。

<a id="build"></a>

## 构建

支持在 **Linux x86_64** 或 macOS arm64 本机编译。不把 Mac 构建的文件当成 Linux
可执行文件。构建机需要 C++17 编译器、Make、CMake ≥ 3.20、Go ≥ 1.24、curl、tar
和 SHA-256 工具。Linux 可安装 `build-essential cmake curl ca-certificates`，
Go 使用满足版本要求的安装包。

```bash
bash native/build.sh
```

脚本固定下载 ONNX Runtime 1.29.0、OpenCV 5.0.0 和 nlohmann/json 3.12.0，检查
下载文件 SHA-256，构建所需 OpenCV 模块和 C++ 核心，运行 CTest 与 Go 单元测试，
再生成 `dist/native-runtime/`。默认 4 个构建任务。内存紧张时使用 `JOBS=2`。
OpenCV 可能继续下载平台优化依赖。构建需要网络。依赖和构建产物均在忽略目录内。
依赖已构建且仅修改应用代码时，可用 `bash native/build.sh --app-only` 跳过 OpenCV
的配置和编译。改变依赖版本或构建选项后应运行完整构建。

```bash
uv run python tools/export_native_model.py --config configs/n150.yaml
```

N150 推荐使用已验证的快速配置（输出文件存在时不会覆盖）。

```bash
uv run python tools/export_native_model.py --config configs/n150.yaml \
  --flush-denormals --output models/selected/native-n150.json
```

后续启动的 `-model` 指向对应 JSON。该开关同时配置 ORT 会话线程和每次 C++ 调用的
FTZ/DAZ 模式。调用结束或异常时恢复线程浮点状态，适应 Go 的线程调度。模型权重、
分类头和 0.95 阈值保留。它改变次正规数语义，不是 FP16 或 INT8。已有 1,051 张
裁剪的准确率及判定保持一致，不能据此保证所有未知输入逐位相同。

准备步骤验证原始模型及分类器校验值，把已有 NPZ 参数无损转为
`models/selected/native-model.json`，并生成 `.sha256` 校验文件。ONNX 权重不变。
JSON 中模型路径相对于自身位置。部署时保留以下布局。

```text
models/
  det_500m.onnx
  selected/
    adaface_ir101.onnx
    w600k_r50.onnx
    classifier.npz
    native-model.json
    native-model.json.sha256
```

`classifier.npz` 用于核对原分类器来源。运行时不会用 Python 读取它。启动时 Go
同时验证 JSON 校验文件、原始模型哈希与模型标识。校验不通过直接退出。校验文件
用于发现损坏，不代表模型包经过数字签名认证。改变配置后需重新导出 JSON。

<a id="startup-and-api"></a>

## 启动与接口

```bash
export SEIYUU_FACEID_API_KEY='replace-with-your-key'
dist/native-runtime/bin/faceid-server \
  -model models/selected/native-model.json \
  -addr 127.0.0.1:8000
```

API 密钥环境变量沿用当前服务。不设置时不要求鉴权。默认监听本机。接口和上传
字段见 [API 文档](API.zh-CN.md)。`/health`、`/ready`、`/v1/model`、`/v1/detect`、
`/v1/recognize`、`/v1/recognize/batch`。此试验版没有首页 UI、Swagger、ReDoc
或 OpenAPI 路由。识别结果字段沿用现有 API。错误详情文本可能不同。

单文件 10 MiB、请求 81 MiB、单图 2000 万像素、每批 8 张。支持静态 JPEG、PNG、
WebP，应用 EXIF 方向。最多允许 8 个处于上传解析、等待或处理阶段的推理请求，超出返回
`503 server_busy`。排队取消返回 `408 request_cancelled`。开始运行的原生推理会完成后释放
资源。关闭服务时等待当前调用结束后销毁模型。试验版拒绝所有 APNG，包括单帧 APNG。

multipart 文件超过 32 KiB 内存阈值后会进入临时目录，请求结束清理，部署仍需预留
临时磁盘空间。`processing_ms` 包含文件读取、校验、解码和推理，不含 multipart
上传解析及排队。评估完整请求应另测客户端 HTTP 时间。HTTP 优雅关闭等待窗口为
90 秒，但原生调用完成前仍不能销毁模型，因此不保证整个进程严格在 90 秒内退出。

`model_id` 与原模型相同，`recognition_id` 按原生实现及完整配置生成，用于区分
实现与设置。阈值、候选数量支持按请求覆盖。

```bash
curl --fail http://127.0.0.1:8000/ready
curl --fail-with-body http://127.0.0.1:8000/v1/recognize \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
```

在目标机测 HTTP 延迟无需 Python，构建产物含 `faceid-bench`。

```bash
dist/native-runtime/bin/faceid-bench -url http://127.0.0.1:8000 \
  -image ./photo.jpg -warmups 3 -runs 20
```

它读取同一 API 密钥环境变量，输出平均 / P50 / P95 HTTP 耗时和平均服务端处理耗时。
不输出身份、文件路径或响应内容，也不把延迟当作 CPU 或内存指标。比较 Python
服务与原生服务时分别启动，使用相同图片和参数，避免两个模型实例争抢 CPU。

<a id="parity-validation"></a>

## 对照验证

模型未改变不等于输出必然相同。需检查解码、检测、五点对齐、归一化、线性头、
候选顺序及接受阈值。尤其 JPEG 解码和不同平台的数值实现都可能产生差异。

```bash
# Linux 使用 .so，macOS 使用 .dylib。
uv run --extra api python tools/validate_native.py \
  --library dist/native-runtime/lib/libfaceid_core.so \
  --model models/selected/native-model.json --config configs/n150.yaml \
  --crop-dir /path/to/aligned-112-crops \
  --image-dir /path/to/original-images --encoded --output /tmp/native-parity.json
```

验证工具只读取显式提供的目录，不默认扫描训练照片。报告汇总概率差与决策一致性，
不保存照片、身份列表或特征。它同时加载 Python 和 C++，因此它的进程内存不代表
原生服务内存。资源比较应分别启动独立 Python 与 Go 服务，用相同输入、线程配置、
预热和请求顺序，比较延迟和进程 RSS。历史留出集曾用于模型选择，不是新的独立泛化
保证。N150 已完成 Linux 构建、实际启动、全量裁剪及固定原图一致性验证，以及
三组 HTTP 基线。该机结果单列于 [实机报告](N150_BENCHMARK.zh-CN.md)，下文 Mac
记录仍保留其原平台与输入范围，不代替 N150 数据。

并列检测分数还有一个明确边界。C++ NMS 按倒序输入索引处理相同分数，Python 使用
NumPy 默认的不稳定排序。若重叠框恰好同分，可能选择不同关键点。不能据抽样回归
声称任意图片都逐位等价。是否作为默认运行时，仍应以目标机器和实际业务图片验收。

<a id="local-regression"></a>

## 本地回归记录

macOS ARM64，Python 与 C++ 在独立进程加载各自的 ONNX Runtime 1.29.0，避免
同一进程内动态库复用影响验证。对全部 1,051 张历史留出裁剪重新提取特征。

- Top-1、Top-5 顺序和 0.95 接受判定全部一致。无新增误判。
- 准确率均为 1,035 / 1,051 = **98.48%**。
- 特征最大绝对差 `2.24e-8`，分类概率最大绝对差 `3.58e-7`。不是逐位相同。
- 另测 12 张原图 / 14 张人脸。原始 BGR 和直接解码上传文件两条路径均通过。
  检测框、检测分数、质量分数一致，候选顺序和接受判定一致，概率最大差 `2.09e-7`。

本地私有明细保存在忽略目录 `data/experiments/native-cpp-go-20261027/`，不随模型
Release 发布。Python 全套 206 项测试、C++ CTest、Go 单元与 race 测试、实际模型
HTTP 集成测试通过。Mac 构建目录复制到其他目录后完成动态加载检查。这些是 Mac
阶段记录。后续 Linux 构建、N150 全量一致性及 HTTP 基线检查见 [实机报告](N150_BENCHMARK.zh-CN.md)，
其中单独列出目标机实际执行的测试范围，不把 206 项或 race 测试自动算入该范围。

<a id="local-http-resources"></a>

## 本地 HTTP 资源对照

同一台 Mac，分别启动独立 Python / Go 服务，均采用 N150 配置。另选 12 张原图，
每个服务预热一轮、计时四轮，共 48 个请求。各做三次新进程测试并交替运行顺序，
下表取三次进程结果的中位数。没有同时驻留两个服务实例。

| 指标 | Python HTTP | C++ + Go HTTP | 变化 |
| --- | ---: | ---: | ---: |
| 平均 HTTP 耗时 | 94.13 ms | 85.00 ms | 减少 9.7% |
| HTTP P95 | 113.83 ms | 103.15 ms | 减少 9.4% |
| 平均服务端处理耗时 | 92.14 ms | 82.69 ms | 减少 10.3% |
| 进程峰值 RSS | 1160.89 MiB | 1026.94 MiB | 减少 133.95 MiB / 11.5% |

三轮 HTTP 输出的检测、质量判定、候选顺序及接受结果均一致，概率最大差 `2.38e-7`。
HTTP 时间包含本机上传和响应，P95 使用 nearest-rank，服务端处理时间包含解码与推理。峰值 RSS 来自每个
服务进程结束后的 `wait4`，包括启动、模型加载和大图解码，不能与只测裁剪分类的
旧基准混用。这里测的是完整运行时迁移的收益，包含解码和数据处理路径变化，不能
全部归因为语言本身。它不是 N150 的耗时、内存或功耗预测。

构建产物和部署材料留在仓库工作流，不作为 GitHub 模型 Release 附件发布。

<a id="standard-dataset-retest"></a>

## 公开标准数据集复测

2026-10-28 在本机 M4 Pro / 24 GiB 上，对 LFW 512 张和 WIDER FACE validation
244 张固定抽样完成三轮对照，共 9,072 个计时 HTTP 请求。完整识别的 C++/Go 平均
耗时分别为 58.82 / 75.33 ms，较 Python 减少 1.83% / 2.67%。仅检测减少约
7.9% / 14.7%，峰值 RSS 中位数减少 3.8%。全部输出一致性检查通过。

这组 Mac 测试关闭 FTZ/DAZ，不替代 N150 的数据，也不是 LFW 验证准确率或 WIDER
检测 AP。输入、工作量、统计方法和原始依据见 [标准数据集测试报告](STANDARD_DATASET_BENCHMARK.zh-CN.md)。

随后完成[全量准确率评测](STANDARD_DATASET_ACCURACY.zh-CN.md)。LFW 6,000 对十折融合
验证准确率 99.13%，WIDER 完整验证集的 Easy / Medium / Hard AP（×100）为
85.49 / 79.48 / 48.84，原声优留出集仍为 98.48%。Python 与 C++ 评测指标和配对判定一致。

<a id="linux-systemd"></a>

## Linux systemd 部署

服务模板为 [`native/deploy/seiyuu-faceid.service`](../native/deploy/seiyuu-faceid.service)。
使用 `DynamicUser`、只读文件系统和独立临时目录，运行时无需 Python 或 Docker。
模板沿用 `0.0.0.0:8000`。可按需要改为本机地址。API 密钥由 root 权限的
`/etc/seiyuu-faceid/api.env` 提供，内容为 `SEIYUU_FACEID_API_KEY=已有密钥`，权限设为
`0600`，不要提交仓库。部署前确认该端口没有其他服务占用。

目录布局如下。模型与运行时放主盘，所有目录可由服务用户读取／遍历，程序可执行，
其余文件只读。服务用户不应有写入程序或模型的权限。

```text
/opt/seiyuu-faceid/
  current -> releases/<version>
  releases/<version>/
    runtime/bin/faceid-server
    runtime/lib/...
    runtime/licenses/...
    models/det_500m.onnx
    models/selected/{adaface_ir101.onnx,w600k_r50.onnx,classifier.npz}
    models/selected/native-model.json
    models/selected/native-model.json.sha256
```

这里的 `native-model.json` 取自带 `--flush-denormals` 导出的快速配置，重命名后重新
生成对应文件名的 SHA-256 sidecar，保留 JSON 字节和所有相对模型路径。准备好布局、
密钥文件及 `current` 链接后。

```bash
sudo install -m 0644 native/deploy/seiyuu-faceid.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
sudo systemctl status seiyuu-faceid.service
```

本次 N150 切换复用此前全量验证通过的核心库、Go 可执行文件及配置字节，未把 Mac
产物传给 Linux。先在临时端口验证，再停旧容器、关闭其自动重启、启用原端口的新
服务。API 密钥保留，其他容器及旧模型挂载保持原样，旧容器保留用于回退。上线后
再次验证 12 张原图／14 张人脸输出。结果见 [部署验收记录](N150_BENCHMARK.zh-CN.md)。

该原生服务仅提供前述 API 路由，`/docs`、`/redoc` 和上传首页不在其中。接口字段
查阅仓库 [API 文档](API.zh-CN.md)。如果旧部署提供 Swagger，此处不会自动保留该页面。

若由 Docker 切换到此服务，先停用 systemd 服务再恢复旧容器，避免端口冲突。

```bash
sudo systemctl disable --now seiyuu-faceid.service
docker update --restart=unless-stopped <原容器名>
docker start <原容器名>
curl --fail http://127.0.0.1:8000/ready
```
