# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

面向用户私有整理的声优照片集、完全本地优先的人脸身份识别工具。运行时会把人脸转换为归一化 embedding，在本地图库中搜索，并返回 `recognized`（已识别）、`review`（待复核）或 `unknown`（未知）；它不是固定类别的 Softmax 分类器。

> **状态：**V1 本地 ONNX 流水线。可按需生成本地基线与阈值校准报告，但报告只适用于其对应数据集；每次更换数据或模型后，请重新校准阈值，再将其用于身份判断。

## 架构

```text
图片 → SCRFD 检测器 → 五点对齐 → ArcFace ONNX embedding
     → L2 归一化 → 本地图库 → 身份评分 → 阈值判断
```

训练和部署被明确分离：`training/` 可以逐步加入 PyTorch/MPS 工具；`src/seiyuu_faceid/runtime` 与日常 CLI 只依赖 NumPy、OpenCV、ONNX Runtime 和 YAML 解析。

## macOS 安装

需要 Python 3.12+（推荐 3.12）。创建隔离环境并安装运行时依赖：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements/runtime.txt
.venv/bin/python -m pip install -e .
```

将兼容的 ONNX 文件放到 `configs/default.yaml` 指定的路径：

```text
models/det_500m.onnx
models/w600k_mbf.onnx
```

已检查过的本地模型组合是 InsightFace `buffalo_sc`：具有五个关键点的 SCRFD-500MF 检测器，以及输入为 `112×112` 的 MobileFaceNet 识别模型。更换模型系列前请查看模型元数据；输入预处理取决于具体模型。

## 数据集目录

原始文件始终保持不变，目录名即 identity key：

```text
data/raw/hanazawa_kana/001.jpg
data/raw/minase_inori/001.jpg
```

处理后的人脸、拒绝记录、元数据和生成的图库均写入 `data/`。`raw` 中的身份目录只是待验证的标签：每张检测到的人脸都保留来源和检测器元数据，以便后续审核。多人图片只有在你明确选择了属于该身份目录的全部人脸后才会加入图库。

## 命令

```bash
# 只扫描一次原始图片；仅生成可随时丢弃的检测缓存。
python tools/scan_raw_faces.py

# 对齐、质量筛选、提取 embedding，并重建图库。
python tools/build_dataset.py
python tools/recognize.py path/to/image.jpg
python tools/batch_recognize.py path/to/photos
python tools/benchmark.py path/to/image.jpg
```

### 审核用于入库的群像

当 `data/raw/` 有群像时，先扫描一次：

```bash
.venv/bin/python tools/scan_raw_faces.py
.venv/bin/python tools/review_raw_faces.py
```

它会建立编号预览图和 `data/review/raw_faces/face_selections.jsonl`。对每条记录，打开其 `preview_image`，将 `"face_indexes": null` 换成从零开始的列表，例如 `[0, 2]`；若没有人脸属于该身份，使用 `[]`。之后正常建库：

```bash
.venv/bin/python tools/build_dataset.py
```

单人原图会自动入库。未做有效选择的多人图片会写入 `data/rejected/metadata.jsonl`，绝不进入图库。扫描生成 `data/runtime/detections.jsonl`，审核和建库都会复用；若原始数据集发生变化，缓存会自动视为过期。

所有命令都支持 `--config configs/default.yaml`。`build_dataset.py` 是通常的一步建库命令：复用扫描缓存（不存在时自动创建），写入对齐元数据，并立刻重建图库。`build_dataset.py --scan-only` 仅创建缓存。只有需要从既有对齐人脸重新生成 embedding 时，使用 `build_gallery.py`。使用 `--skip-gallery` 可单独检查数据集阶段。已检查的默认阈值来自当前独立的六身份 holdout；数据或模型有显著变化后必须重新校准。

### 独立评测与阈值校准

将带标签的评测图片与入库图片分开保存：

```text
data/gallery/<identity>/...
```

所有工具都把 `data/gallery/` 当作只读目录。生产图库从 `data/raw/` 构建，然后在不泄漏的情况下评测：

```bash
.venv/bin/python tools/evaluate_external.py
```

该命令会在 `data/runtime/evaluation/` 写入 embedding、每张图片的排除理由及报告。未审核的多人图片会从指标中排除。报告包括“留一身份”的未知人代理测试；在将任何阈值视为最终阈值前，请加入真实未知人员图片。

完成评测后，生成含 SVG 图表和 CSV 曲线的独立本地报告：

```bash
.venv/bin/python tools/generate_evaluation_report.py
```

默认输出目录为 `data/runtime/reports/evaluation_post_review/`。

## 本地网页界面

可选的 Streamlit 界面包含两个仅在本地运行的工作区：

- **图库管理**：列出 `data/raw/` 下的身份目录，可查看其中图片、创建身份目录、直接向所选目录上传参考图片。若图库已经存在，所选原图中的每张检测到的人脸都会自动匹配并显示结果。图片以带分页的缩略图网格呈现。
- **建库与审核**：独立页面。先点击**扫描全部原始图片**，再点击**开始逐题审核群像**，仅审核检测到多张人脸的图片。键盘数字键可切换对应人脸，`A` 选择全部，`Y` 接受可信建议，`Enter` 确认当前选择并进入下一张。建库会复用扫描结果，且在最后一张群像审核结束后开始。单人图片自动接受。
- **识别图片**：可识别一张或多张查询图片，而不写入 `raw`。将大型批量任务放入 `data/query/`（或输入其他本地目录）后点击**识别整个文件夹**；该目录及其子目录中的所有支持图片都会被处理，并以带分页的缩略图网格显示结果。临时上传适合小型临时检查。检测框为灰色，已选择／待审核为紫色或黄色，已知身份为绿色，未知身份为红色。结果可下载为 JSONL。

```bash
.venv/bin/python -m pip install -r requirements/web.txt
.venv/bin/streamlit run app.py
```

打开 Streamlit 输出的本地 URL（通常是 `http://localhost:8501`）。只要已放好配置的 ONNX 模型文件，网页即可完成首次建库。附带的 Streamlit 设置仅监听 `127.0.0.1`，并禁用了 Streamlit 使用统计。

## 训练

第一阶段运行时里特意不包含微调。后续训练将使用预训练的 MobileFaceNet 类 backbone 和 ArcFace head，仅导出 embedding backbone 为 ONNX，并验证 PyTorch 与 ONNX embedding 的一致性。训练依赖只属于 `requirements/training.txt`。

## Benchmark 与未来 N150 部署

`tools/benchmark.py` 输出包含检测、识别、端到端延迟、RSS 和吞吐量的 JSON。先记录 Mac 基线，再在 Intel N150 上执行完全相同的命令。Runtime 面向 CPU：单模型实例、受限的检测图片大小、不依赖 PyTorch，在当前小图库规模下也不使用向量数据库。

## 隐私

项目采用 local-first 设计：不会上传原始图片、embedding 或遥测数据，也不会调用云端人脸识别服务。Runtime 在导入 ONNX Runtime 前设置 `ORT_DISABLE_TELEMETRY=1`，以禁用其上传器和持久设备标识符。

## 模型许可证

代码使用 MIT 许可证；模型权重不随仓库发布，可能具有完全不同的许可证或使用限制。使用前请在 [models/README.zh-CN.md](models/README.zh-CN.md) 记录每个下载模型的来源和条款。
