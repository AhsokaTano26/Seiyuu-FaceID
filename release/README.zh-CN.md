# 模型发布输入

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

`model-source.json` 固定可下载的模型专用归档、整体 SHA-256，以及每个允许成员的摘要和大小。`training.json` 是核实该分类头实际使用的冻结训练、验证记录后导出的公开汇总，与分类头 SHA-256 及内置姓名顺序绑定。

CI 根据这些输入生成模型专用 Release。应用部署归档交付至 Actions Artifacts，容器镜像交付至 Docker Hub。当前来源固定为历史 v0.1.0 的模型归档，整个归档和成员摘要均已核对。

发布按内容寻址的 `models-<content-digest>` Release 后，会使用该权重和当前 main 的应用启动部署 CI。人工发布由 `release.published` 触发，CI 发布通过 `tools/publish_models.py --dispatch` 显式发送 `repository_dispatch`。生成的部署包固定该 Release 和模型摘要。模型触发的构建在交付部署文件后结束，避免递归发布模型。已有版本可通过工作流的 `model_tag` 输入重新构建。

公开元信息包含身份训练数汇总和来源。训练统计排除测试记录。模型 Release 排除照片、样本级标签、原始特征、评测数据与API密钥。

下载和更新流程见 [CI/CD 文档](../docs/CI_CD.zh-CN.md)，完整发布规则见[项目约定](../AGENTS.md)。
