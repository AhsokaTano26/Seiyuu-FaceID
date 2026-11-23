# C++ 核心 + Go HTTP 实验

[English](DESIGN.md) | [简体中文](DESIGN.zh-CN.md) | [日本語](DESIGN.ja.md)

这是经用户授权、面向 Intel N150 / 8 GB Linux 的原型。保留 Python 作为参考实现。运行时不需要 Python，单个 Go 进程通过 cgo 调用 C++ 共享库。三个常驻 ONNX Runtime CPU Session 和一个模型实例供请求串行使用，两个识别骨干并行运行。不执行 push 或 Release 发布。FTZ/DAZ 实现已通过目标机一致性与性能检查，并在 2026-10-28 选为 N150 服务。

## 接口

`include/faceid.h` 定义 C 所有权和错误边界。Go 在创建引擎前验证模型校验值。C++ 校验维度、有限参数和配置，并执行所有计算机视觉、推理及评分。

`tools/export_native_model.py` 把现有 NPZ 参数转换成 `models/selected/native-model.json`，只有这一步准备工作需要 Python。ONNX 和分类器来源文件保持不变。JSON schema 版本为 1。

- `model_id` 为 Python 排序后资产 JSON 的 SHA256，与 Python 保持一致。
- `source_assets` 为原始清单中的文件名到 SHA256 的映射。`detector_sha256` 为检测器 SHA256。
- `models` 含 `detector`、`adaface`、`r50` 相对于此 JSON 文件的路径。
- `class_names` 为有序字符串数组。
- `head` 含 float64 的 `mean` / `scale`，以及 float32 的 `coefficients`（按行）/ `intercept`。
- `runtime` 含 `intra_op_threads`、`detector_intra_op_threads`、`inter_op_threads`、`parallel_backbones`、`allow_spinning`、`enable_cpu_mem_arena` 和可选的 `flush_denormals`，后者严格要求 bool，缺省为 false。它仅在受支持的 Linux x86_64 上启用 FTZ/DAZ，在 Session 创建和每次推理调用前后完整恢复调用方 MXCSR，同时配置 ORT worker 池。
- `detector` 含 `input_size` [width,height]、`max_long_edge`、`score_threshold`、`nms_threshold`。
- `quality` 含 `min_face_size`、`min_sharpness`。
- `recognition` 含 `top_k`、`probability_threshold`。

`faceid_model` 返回 `model_id`、临时 `recognition_id` 标记（`native-v1:` + model ID）、`class_count`、`class_names` 和 `defaults`。Go 将 `recognition_id` 替换为 `native-v1\n` + 模型 JSON 原始字节的 SHA256，覆盖全部配置和检测器校验值，并补充上传限制和格式。JSON 空白变化也会改变 HTTP 识别 ID。

## 范围与验证

Go 实现健康检查、就绪检查、鉴权模型信息、检测、识别及批量识别。Multipart 支持 JPEG/PNG/WebP，每文件 10 MiB、每请求 81 MiB、每图 2000 万像素、每批 8 张。实现参数校验、等待期间取消、单次推理串行及优雅关闭。初始原型不包含 UI/OpenAPI。保留 EXIF 方向处理并拒绝动画。

检查模型损坏、畸形上传、鉴权、并发请求和关闭时的所有权。比较全部 1,051 张历史留出裁剪的特征、概率、top-1/top-5 和 0.95 判定，还需在原图上比较检测、对齐和解码。权重不变不足以证明结果不变。固定输入、预热、线程设置并重复运行，在独立进程中测试 Python 与原生服务。不能用本地 ARM 时间预测 N150 时间。私人测试输入和结果留在忽略的数据目录。

## 工作分工

1. 核心负责 C++ 流程与 C ABI、CMake 库及小型数值和错误测试。
2. HTTP 负责 Go net/http 服务、cgo 适配器和 API 行为测试。
3. 集成负责可复现的依赖和构建、模型导出器、Python/原生一致性及性能测量、Linux 部署说明。
