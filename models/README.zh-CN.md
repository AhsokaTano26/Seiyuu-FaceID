# 模型清单

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

模型权重刻意不纳入本仓库。在添加任何权重前，请在下表记录准确下载链接、校验和、来源、版本与许可证。

默认配置需要：

| 用途 | 文件名 | 上游来源 | 许可证／限制 |
| --- | --- | --- | --- |
| 检测器 | `det_500m.onnx` | InsightFace [`buffalo_sc` v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip)，SCRFD-500MF | 上游预训练模型政策：仅限非商业研究。 |
| 识别器 | `w600k_mbf.onnx` | InsightFace [`buffalo_sc` v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip)，MobileFaceNet | 上游预训练模型政策：仅限非商业研究。 |

本地安装的 `buffalo_sc.zip` SHA-256 为：
`57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72`。

InsightFace 代码使用 MIT 许可证，但其发布的预训练模型并非如此：上游 README 声明，无论手动下载还是通过库下载，权重均仅可用于非商业研究。本项目仅对自身代码使用 MIT，且不授予任何模型权重的使用权。使用其他模型时，请在放入 `models/` 前新增一行并注明其准确条款。
