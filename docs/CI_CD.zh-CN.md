# CI/CD — 三种部署方式与模型发布

[English](CI_CD.md) | [简体中文](CI_CD.zh-CN.md) | [日本語](CI_CD.ja.md)

工作流见 [ci.yml](../.github/workflows/ci.yml)。推送到 main、推送 v*.*.* 版本标签，
或在 main 手动运行后，先检查代码，再交付以下内容。Pull Request 只构建和检查，
不发布镜像、部署附件或模型。
发布 `models-<content-digest>` 模型 Release 也会启动部署 CI，使用该模型和当前 main 的应用提交。

| 交付方式 | 下载位置 | 内容与平台 |
| --- | --- | --- |
| Docker Compose | Actions 的 seiyuu-faceid-docker-提交SHA 产物。镜像在 Docker Hub | Compose、普通/N150 配置、固定镜像标签、模型下载工具。Linux amd64/arm64 |
| Python 原生 | Actions 的 seiyuu-faceid-python-提交SHA 产物 | Python 源码、uv.lock、配置、systemd 模板、模型下载工具。CI 检查 Linux amd64 |
| C++ / Go 原生 | Actions 的 seiyuu-faceid-native-linux-amd64-提交SHA 产物 | Linux amd64 可执行文件、共享库、许可证、FTZ 与兼容 JSON、systemd 模板、模型下载工具 |
| 模型 | GitHub Releases 的 models-内容摘要 版本 | ONNX 权重、分类头及必要元信息，独立于应用版本 |

三种部署包均附归档 SHA-256 和包内 SHA256SUMS。进入对应 Actions 运行页面底部的
Artifacts 下载，先解开外层下载文件，再按包中 README.md 操作。部署归档保留 14 天。
需要长期保存时自行下载归档，或从仓库复现构建。构建产物的使用方式见
[GitHub 官方说明](https://docs.github.com/en/actions/tutorials/store-and-share-data)。

部署包不重复携带约 400 MB ONNX 模型。它们都附带 release/model-source.json，
固定下载地址、整个归档摘要、允许的成员及每个文件的摘要和大小。
下载工具拒绝多余文件、路径穿越、链接、损坏模型以及覆盖不同的现有权重。
首次下载使用 Python 3.11+ 的标准库。也可以在另一台机器准备 models/ 后复制。
这是安装准备步骤，C++ / Go 服务运行时不需要 Python。离线时使用 —

~~~bash
python3 tools/fetch_models.py --archive /路径/模型归档.tar.gz
~~~

普通代码构建使用仓库固定的模型来源，目前由既有 v0.1.0 的模型专用附件引导获取，不下载该版本的历史部署附件。
模型发布触发的构建改为锁定新发布的 Release。所有任务共享经过验证的 `model-inputs`，其中包含来源锁定文件和训练汇总。
所有方式使用同一组 FP32 权重、分类头和识别阈值。模型文件需遵守随附的上游条款。

## 检查与发布顺序

1. prepare — 选择应用提交和模型输入。模型 Release 先验证附件白名单、GitHub 摘要、SHA256SUMS、归档内容标签、安全成员、包内外元信息一致性、分类头标签和实际训练样本数，再分发来源锁定文件。
2. quality — 锁定依赖检查、Ruff、全部 Python 测试、编译检查和 actionlint。
3. deployments — 获取并核对模型及训练统计，生成 Python 和 Compose 包。解压到新目录后，
   实际启动 Python HTTP 服务，检查就绪、鉴权、模型标识、检测与识别。验证两套 Compose。
4. native — 在 Ubuntu 22.04 构建 C++ / Go，执行 CTest、普通和 native Go 测试。
   导出兼容及 FTZ 参数。用固定随机人脸裁剪比较 Python/C++ 的特征、概率和判定。
   打包后在新目录启动原生 HTTP 服务并检查接口。
5. docker — 前置检查通过后，构建并发布 Python API 的 Linux amd64/arm64 镜像。
6. models — 仅在应用构建中执行。等待前述任务全部成功后生成或核对模型专用 Release，并显式请求模型部署 CI。模型触发的构建在交付部署文件后结束，避免循环发布。

随机裁剪比较用于实现回归检查，不构成准确率评测。真实数据准确率见
[标准数据集准确率报告](STANDARD_DATASET_ACCURACY.zh-CN.md)。原生运行时基于 glibc，
不是静态单文件程序。Ubuntu 22.04 构建提供 glibc 2.35 基线，其他系统需核对动态库兼容性。
默认 native-model.json 开启 Linux x86_64 FTZ/DAZ。
native-compatible.json 关闭该模式。保留相对模型目录布局及各 JSON 的 .sha256 文件。

Docker 镜像标签包括 main 的 latest、完整提交的 sha-标签和版本标签对应的语义版本。
部署包默认指向本次提交的 sha-标签。现有 Docker 镜像运行 Python API，没有原生 FTZ 路径。
模型构建还会发布 Docker 的 `models-<content-digest>` 标签，并在 Actions 产物名末尾附加模型标签。包内归档名继续使用应用提交，`BUILD.json` 和 `release/model-source.json` 绑定确切的模型归档摘要。
只有普通 main 构建更新 `latest`。模型从单独下载并校验的模型目录挂载。
C++ / Go 原生包默认只有 API，上传首页和 /docs 属于 Python 方案。

## 模型 Release 的边界

AGENTS.md 的发布约定继续适用。附件严格限定为 —

- seiyuu-faceid-models.tar.gz — models/ 目录中的模型与必要元信息。
- SHA256SUMS — 以上归档及配套附件的校验。
- TRAINING_SAMPLES.csv — 全部模型内置姓名及实际拟合样本数。
- TRAINING_PROVENANCE.json — 分类头摘要、冻结特征/划分文件的摘要及汇总统计。
- MODEL_LICENSES.md — 模型来源和上游使用条款。

不上传应用代码、部署归档、Compose、API 文档、照片、原始特征或评测数据。
Release 说明以模型下载、模型范围和完整身份/样本数表为主。
仓库固定来源的统计为 50 个身份、4,064 张实际拟合裁剪 —
3,574 张原训练集 + 490 张原验证集，排除 1,051 张测试集。
统计从本地当前模型使用的冻结划分导出，并与原发布 CSV、分类头标签顺序及哈希绑定核对。
CI 只接触公开汇总统计，不下载训练特征。

模型归档固定时间戳、文件顺序、权限和所有者信息，用归档 SHA-256 的前 16 位作为模型版本。
相同归档复用已有发布。应用代码单独变化不会重复上传权重。
新增发布先建立草稿、上传白名单文件，校验远端每个附件的大小和 GitHub 摘要后才公开。
失败留下草稿，可重跑补齐缺失附件。上传中断留下的零字节 starter 占位，
仅在草稿且名称属于白名单时清理重传。不同内容的已有附件和不完整的已公开版本会报错，
不会覆盖或静默修复。

## 凭据与后续更新

仓库需要 DOCKERHUB_TOKEN Secret，供 tano26/seiyuu-faceid 镜像发布使用。
模型发布使用 Actions 自带 GITHUB_TOKEN，仅 models 任务声明 `contents: write` 和 `actions: read`，用于发布、显式触发和查找已有验证运行。
其他任务只读，不需要另存 GitHub 长期密钥。流水线不会 SSH 到服务器或自动替换正在运行的服务。

更新应用后推送 main 即可。更新权重前，需要在本地确认实际训练来源，然后导出公开汇总 —

~~~bash
uv run python tools/export_training_metadata.py --output /tmp/training-new.json
~~~

该命令读取当前冻结数据与分类头，只导出汇总，拒绝交叉/重复划分、类别不匹配和错误总数。
检查新统计后更新 release/training.json。将已准备好的新模型专用归档提供给 CI，
并更新 release/model-source.json 的固定 GitHub 下载 URL、归档与逐成员摘要、大小。
模型来源归档必须已经可以下载，CI 不会访问开发电脑的忽略目录，也不会重新训练模型。
首次引导或更换来源可先手动发布模型专用来源版本，再由流水线生成经过验证的内容版本。

### 模型发布触发

人工发布通过 `release.published` 监听 `models-<content-digest>` 标签。公开前须上传完整的五个模型附件。
CI 发布使用 `tools/publish_models.py --dispatch`，发送类型为 `model-published`、携带 `client_payload.tag` 的 `repository_dispatch`。GitHub 会抑制由 GITHUB_TOKEN 产生的普通工作流事件，但允许显式 dispatch 启动工作流，规则见 [GitHub 官方说明](https://docs.github.com/en/actions/concepts/security/github_token)。
发布工具检查最近 100 次 CI 运行，仅复用本仓库可信模型事件中正在运行或已经成功的 `Model <tag>`，排除 PR 和 fork 运行。失败运行和未送达的触发可通过重跑发布工具重试，即使 Release 已经公开。
模型事件构建独立验证所选 Release，验证失败直接终止。同次运行的所有部署包均使用所选模型。后续普通代码构建仍以仓库固定来源为默认值，需要显式更新该来源才能更换默认模型。

为既有模型 Release 构建部署文件或重试失败运行时，在 main 的 **Run workflow** 中填写 `model_tag`，也可替换以下示例标签后执行。

~~~bash
gh workflow run ci.yml --ref main -f model_tag=models-0123456789abcdef
~~~

工作流和脚本须先进入 main。人工发布时，模型标签所指的提交也必须包含 `release.published` 监听，因为 GitHub 从标签提交读取该事件的工作流。早于监听配置的已有标签，使用上面的 `model_tag` 命令或显式 repository dispatch 到 main。规则见 [GitHub 工作流触发说明](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)。
运行名称为 `Model <tag>`。模型公开成功与部署 CI 成功是两个状态，部署前检查对应运行结果。

配置阶段完成本地验证。随后按用户授权提交、推送并验证云端流程。
完整云端构建、Docker Hub 推送及 Release 上传结果，以对应提交的 Actions 运行记录为准。
