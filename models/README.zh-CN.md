# 模型包

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

默认路径使用以下模型包布局。

- `det_500m.onnx` 是 InsightFace buffalo_sc v0.7 的 SCRFD 检测器。[下载来源](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip)
- `selected/adaface_ir101.onnx` 是冻结的 AdaFace IR101/WebFace12M，转换自官方检查点与架构。输入为 BGR、NCHW、112×112，按 `(pixel/255 - 0.5)/0.5` 归一化，输出512维单位特征。[官方权重](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view)、[架构](https://github.com/mk-minchul/AdaFace)
- `selected/w600k_r50.onnx` 来自 InsightFace buffalo_m v0.7。输入为 RGB、NCHW、112×112，按 `pixel/127.5 - 1` 归一化，输出512维特征后再归一化。[下载来源](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip)
- `selected/classifier.npz` 存放本地拟合的标准化参数、逻辑回归系数和有序身份标签。AdaFace、R50 单位特征按此顺序拼接并除以√2。运行时复现 sklearn 的 float32 标准化和稳定多项 softmax。
- `selected/manifest.json` 记录来源URL和 SHA-256，加载模型时校验。
- `selected/evaluation.json`、`holdout_scores.npz` 与 `runtime_verification.json` 保存历史选定模型的评测证据。

历史模型包约416 MiB。权重由 Git 忽略并单独分发，迁移安装时应保留全部模型包文件。运行时依赖 NumPy 与 ONNX Runtime。

AdaFace 上游 MIT 声明保留在 `selected/ADAFACE_LICENSE`。InsightFace 代码采用 MIT，其公开预训练权重附有非商业研究用途条款。使用时同时遵守项目许可证和模型上游条款。

默认本地模型包与 `release/model-source.json` 固定的来源包含50个身份。单独发布的 `models-5cc19e9d5b1a30aa` 包含606个身份。选择版本时核对分类头 SHA-256 和 manifest，见[606身份报告](../docs/MODEL_ACCURACY_20261103.zh-CN.md)。
