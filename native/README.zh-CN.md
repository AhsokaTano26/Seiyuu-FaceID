# C++ 推理 + Go HTTP

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

这是现有 FP32 模型的实验性替代运行时。Go 通过 cgo 在同一进程内调用 C++ 库。模型推理、检测、对齐和分类由 C++ 执行，HTTP、上传校验和鉴权由 Go 执行。Python 只用于模型准备和对照，服务运行时不需要 Python。

参见[构建、验证和部署说明](../docs/NATIVE_CPP_GO.zh-CN.md)。完整识别流程、训练及运行时设计理由见[技术架构指南](../docs/TECHNICAL_ARCHITECTURE.zh-CN.md)。现有 Python 运行时仍为参考实现。

Intel N150 目标机已通过 Linux 构建与启动检查、1,051 张历史留出裁剪和 12 张原图的一致性检查，以及完整三组 HTTP 基线。同配置下原生基线未显示可靠提速，峰值 RSS 更高。省 CPU 配置相对旧线程 Python 组的平均 HTTP 延迟增加约 37.9%。范围、结果和后续独立实验见 [N150 报告](../docs/N150_BENCHMARK.zh-CN.md)。经验证的 FTZ/DAZ 版本已在 2026-10-28 替换 N150 API 服务，长期运行和压力验收尚未完成。

后续独立 FTZ/DAZ 候选通过全量回归及三个新进程的 HTTP 测量，各进程平均请求耗时的中位数为 680.785 ms，历史留出 Top-1 仍为 98.48%。这是后续阶段实验，并未做同期交替 A/B。实现现已包含在本源码树。导出时加 `--flush-denormals`，可在受支持的 Linux x86_64 上开启。默认仍关闭以保持兼容，Mac ARM 会拒绝开启。实现、内存取舍和验证边界见 [FTZ 实验](../docs/N150_BENCHMARK.zh-CN.md)。

[systemd 模板](deploy/seiyuu-faceid.service) 运行单个原生进程，[部署指南](../docs/NATIVE_CPP_GO.zh-CN.md#linux-systemd) 说明布局、凭据、启动和回退。服务提供 API 端点，不包含 Python 的 Swagger 或上传 UI 路由。
