# 文档图源与复现

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

技术框架与标准数据集准确率文档的插图由 [draw_documentation_figures.py](../../tools/draw_documentation_figures.py) 中的
Python 函数绘制，使用 Matplotlib，分别嵌入
[技术框架](../TECHNICAL_ARCHITECTURE.zh-CN.md) 和 [标准数据集准确率](../STANDARD_DATASET_ACCURACY.zh-CN.md)。
每幅图提供 **SVG、300 dpi PNG、PDF**，正文使用 SVG 以便放大。

## 复现

以下命令在仓库根目录运行。

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_documentation_figures.py
```

以下命令重绘指定图。

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_documentation_figures.py --only wider_pr_curves lfw_fold_accuracy
```

不运行模型，也不读取训练照片或原始人脸特征。绘图依赖与应用运行环境分离。
绘图数据为 [aggregate_metrics.json](accuracy/aggregate_metrics.json)，仅含汇总指标、
十折统计与 PR 曲线，源准确率报告、独立复算和曲线文件摘要随数据记录，
完整评测出处见 [准确率报告](../STANDARD_DATASET_ACCURACY.zh-CN.md)。

## 图目

函数名统一为 `draw_<文件名>()`，每个函数返回 Matplotlib Figure。
每行的 PNG 和 PDF 与 SVG 同名，便于在其他文档中插入。

| 文档 | 图意 | SVG | PNG | PDF |
| --- | --- | --- | --- | --- |
| 技术框架 | 完整识别与复核分支 | [SVG](architecture/recognition_pipeline.svg) | [PNG](architecture/recognition_pipeline.png) | [PDF](architecture/recognition_pipeline.pdf) |
| 技术框架 | SCRFD 多尺度输出与去重 | [SVG](architecture/scrfd_detector.svg) | [PNG](architecture/scrfd_detector.png) | [PDF](architecture/scrfd_detector.pdf) |
| 技术框架 | 五点对齐的几何关系 | [SVG](architecture/five_point_alignment.svg) | [PNG](architecture/five_point_alignment.png) | [PDF](architecture/five_point_alignment.pdf) |
| 技术框架 | 颜色、特征维度与融合分类 | [SVG](architecture/feature_fusion.svg) | [PNG](architecture/feature_fusion.png) | [PDF](architecture/feature_fusion.pdf) |
| 技术框架 | 离线训练和参数导出 | [SVG](architecture/training_pipeline.svg) | [PNG](architecture/training_pipeline.png) | [PDF](architecture/training_pipeline.pdf) |
| 技术框架 | 拟合集与历史留出集的口径 | [SVG](architecture/data_split.svg) | [PNG](architecture/data_split.png) | [PDF](architecture/data_split.pdf) |
| 技术框架 | FTZ/DAZ 浮点状态的作用域 | [SVG](architecture/denormal_scope.svg) | [PNG](architecture/denormal_scope.png) | [PDF](architecture/denormal_scope.pdf) |
| 技术框架 | Go、C ABI 与 C++ 的职责 | [SVG](architecture/native_runtime.svg) | [PNG](architecture/native_runtime.png) | [PDF](architecture/native_runtime.pdf) |
| 技术框架 | 双骨干的并行与汇合 | [SVG](architecture/inference_parallel.svg) | [PNG](architecture/inference_parallel.png) | [PDF](architecture/inference_parallel.pdf) |
| 技术框架 | 槽位、推理门与取消边界 | [SVG](architecture/request_lifecycle.svg) | [PNG](architecture/request_lifecycle.png) | [PDF](architecture/request_lifecycle.pdf) |
| 技术框架 | 资产校验与模型启动 | [SVG](architecture/model_startup.svg) | [PNG](architecture/model_startup.png) | [PDF](architecture/model_startup.pdf) |
| 准确率 | 三个任务及指标的区别 | [SVG](accuracy/benchmark_tasks.svg) | [PNG](accuracy/benchmark_tasks.png) | [PDF](accuracy/benchmark_tasks.pdf) |
| 准确率 | LFW 十折阈值选择与测试 | [SVG](accuracy/lfw_protocol.svg) | [PNG](accuracy/lfw_protocol.png) | [PDF](accuracy/lfw_protocol.pdf) |
| 准确率 | LFW 的正确、误判与漏检构成 | [SVG](accuracy/lfw_errors.svg) | [PNG](accuracy/lfw_errors.png) | [PDF](accuracy/lfw_errors.pdf) |
| 准确率 | LFW 逐折实测准确率 | [SVG](accuracy/lfw_fold_accuracy.svg) | [PNG](accuracy/lfw_fold_accuracy.png) | [PDF](accuracy/lfw_fold_accuracy.pdf) |
| 准确率 | WIDER 匹配、忽略与 AP 计算 | [SVG](accuracy/wider_protocol.svg) | [PNG](accuracy/wider_protocol.png) | [PDF](accuracy/wider_protocol.pdf) |
| 准确率 | 三个难度的真实 PR 曲线 | [SVG](accuracy/wider_pr_curves.svg) | [PNG](accuracy/wider_pr_curves.png) | [PDF](accuracy/wider_pr_curves.pdf) |
| 准确率 | 声优裁剪接受与复核的错误分解 | [SVG](accuracy/holdout_decisions.svg) | [PNG](accuracy/holdout_decisions.png) | [PDF](accuracy/holdout_decisions.pdf) |
| 准确率 | 真值准确率、实现一致性和独立复算 | [SVG](accuracy/validation_evidence.svg) | [PNG](accuracy/validation_evidence.png) | [PDF](accuracy/validation_evidence.pdf) |

## 绘图口径与验收

- 本机使用 **SimSun（宋体）**绘制中文、**Times New Roman** 绘制英文和普通数字，
  数学公式使用 Computer Modern MathText。SVG 使用字形轮廓，PDF 嵌入字体子集，
  没有随仓库复制字体文件。缺少指定字体时，脚本按明确候选顺序选择并打印实际字体。
- 结构图的节点位置和执行阶段是示意布局，箭头表示数据或调用依赖，框大小、
  横向间距不代表模型大小、样本占比或真实耗时。
- 五点图的右侧是代码中的真实 ArcFace 模板，左侧是已知相似变换构造的示意点，
  不是一张评测照片。两侧坐标轴各自保持等比例，原点均在左上角。
- LFW 折间曲线保留全部十折，不平滑，纵轴显示 97%–100.25% 的局部区间。
  WIDER PR 图保留全部 1,000 档结果，曲线终点是真实最大召回率，不补齐到 100%。
- 脚本检查节点文字边界和字符覆盖，核对十折均值、PR 召回单调性和曲线积分与 AP
  一致，导出无缺字警告。渲染后另查原尺寸、灰度预览及 PDF 字体嵌入。

[manifest.json](manifest.json) 记录绘图函数、实际依赖版本、字体及聚合数据和源码摘要。
插图保留在仓库文档中，不作为模型 Release 附件。

## 2026-11-03 模型版本准确率报告

[606 身份正式模型报告](../MODEL_ACCURACY_20261103.zh-CN.md) 使用独立的
[draw_model_accuracy_report.py](../../tools/draw_model_accuracy_report.py) 绘制下列图表。
主版本为 `models-5cc19e9d5b1a30aa`，历史参照为 50 身份的
`models-ca1a34fb7a63a38f`。两版使用不同测试集，图表不表示同条件下的性能提升。

| 图意 | PNG | PDF | SVG |
| --- | --- | --- | --- |
| 两版各自留出集的描述性指标 | [PNG](model_accuracy_20261103/model_version_metrics.png) | [PDF](model_accuracy_20261103/model_version_metrics.pdf) | [SVG](model_accuracy_20261103/model_version_metrics.svg) |
| 606 身份模型的阈值处置与错误分解 | [PNG](model_accuracy_20261103/model606_threshold_outcomes.png) | [PDF](model_accuracy_20261103/model606_threshold_outcomes.pdf) | [SVG](model_accuracy_20261103/model606_threshold_outcomes.svg) |
| 606 身份模型每个身份的样本量分布 | [PNG](model_accuracy_20261103/model606_sample_support.png) | [PDF](model_accuracy_20261103/model606_sample_support.pdf) | [SVG](model_accuracy_20261103/model606_sample_support.svg) |

以下命令在仓库根目录复现图表。

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_model_accuracy_report.py
```

输入为 [aggregate_metrics.json](model_accuracy_20261103/aggregate_metrics.json)，
仅含版本、指标、计数、分箱及证据摘要。
每图提供 400 dpi PNG 和两个矢量格式，使用 SimSun 与 Times New Roman。
源码、字体和图像摘要见 [figure_manifest.json](model_accuracy_20261103/figure_manifest.json)，
实际验收记录见 [quality_review.json](model_accuracy_20261103/quality_review.json)。


质量审核文件是文档多语化前的历史快照，其中报告校验值对应原始文档字节。翻译后的 Markdown 使用各语言正文，图表与数据摘要继续对应当时的生成记录。
