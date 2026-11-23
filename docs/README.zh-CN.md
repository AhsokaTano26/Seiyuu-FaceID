# 文档索引

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

本索引覆盖仓库维护的 Markdown 文档，通过顶部链接切换英文、简体中文和日文。英文使用 `.md`，中文使用 `.zh-CN.md`，日文沿用标准语言代码 `.ja.md`。

默认本地模型与 CI 固定来源包含50个身份，单独发布的606身份模型具有自己的报告和数据集。阅读指标时核对分类头校验值与测试协议。

第三方依赖目录、已生成部署目录与私人实验产物保留各自文件的声明和来源记录。历史计划按标注日期保存当时决策。下列指南说明仓库维护的接口和部署流程。

## 开始与使用

- [项目概览](../README.zh-CN.md)
- [HTTP API](API.zh-CN.md)
- [识别技术框架](TECHNICAL_ARCHITECTURE.zh-CN.md)
- [原生运行时入口](../native/README.zh-CN.md)
- [原生运行时设计](../native/DESIGN.zh-CN.md)

## 部署与发布

- [CI/CD、部署产物与模型发布触发](CI_CD.zh-CN.md)
- [Docker与离线部署](DEPLOY_LINUX.zh-CN.md)
- [Python/systemd部署](DEPLOY_NATIVE_LINUX.zh-CN.md)
- [C++/Go部署与验证](NATIVE_CPP_GO.zh-CN.md)
- [N150配置与测量](N150.zh-CN.md)
- [本地部署包](../deploy/README.zh-CN.md)
- [CI Docker包说明](../deploy/ci/docker.zh-CN.md)
- [CI Python包说明](../deploy/ci/python.zh-CN.md)
- [CI原生包说明](../deploy/ci/native.zh-CN.md)
- [模型发布输入](../release/README.zh-CN.md)

## 数据与训练

- [本地数据目录](../data/README.zh-CN.md)
- [模型来源与版本](../models/README.zh-CN.md)
- [分类头复现与搜索](../training/README.zh-CN.md)

## 评测与性能

- [606身份模型准确率](MODEL_ACCURACY_20261103.zh-CN.md)
- [历史50身份分类头对照](CLASSIFIER_OPTIMIZATION.zh-CN.md)
- [N150基线及FTZ/DAZ实测](N150_BENCHMARK.zh-CN.md)
- [LFW/WIDER FACE性能对照](STANDARD_DATASET_BENCHMARK.zh-CN.md)
- [LFW/WIDER FACE准确率协议](STANDARD_DATASET_ACCURACY.zh-CN.md)
- [图源与复现](figures/README.zh-CN.md)

## 历史实施计划

- [HTTP API实施，2026-10-21](superpowers/plans/2026-10-21-http-api.zh-CN.md)
- [样本量实验，2026-10-21](superpowers/plans/2026-10-21-sample-size.zh-CN.md)
- [分类头优化，2026-10-27](superpowers/plans/2026-10-27-classifier-optimization.zh-CN.md)
- [N150运行时计划，2026-10-27](superpowers/plans/2026-10-27-n150-runtime.zh-CN.md)
- [CI交付计划，2026-10-28](superpowers/plans/2026-10-28-ci-deployments.zh-CN.md)

独立的[人脸检测 OpenAPI 定义](detect.openapi.yaml)可导入API客户端。项目约定见 [AGENTS.md](../AGENTS.md)。
