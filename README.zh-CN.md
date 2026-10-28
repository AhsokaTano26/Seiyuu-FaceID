# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

使用本地声优照片训练的 50 类身份分类器。当前唯一模型是 **AdaFace IR101 + InsightFace R50 + 标准化 + 逻辑回归**，全部在本机通过 ONNX Runtime 推理。

完整原理见 [识别技术框架详解](docs/TECHNICAL_ARCHITECTURE.md)：从检测、质量检查、五点对齐、双骨干特征融合到分类与复核，附训练评估、C++ / Go 架构和 N150 性能证据。

## 使用

```bash
uv sync --locked --extra api --extra web --extra dev
uv run streamlit run app.py
uv run python tools/recognize.py path/to/photo.jpg
uv run python tools/batch_recognize.py path/to/photos
```

HTTP API 支持单图识别、批量识别、人脸检测和模型/健康状态查询：

```bash
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
curl --fail-with-body http://127.0.0.1:8000/v1/recognize -F 'file=@photo.jpg'
```

完整请求/响应、错误码、可选 Bearer 密钥及部署说明见 [API 文档](docs/API.md)；
启动后可访问 `http://127.0.0.1:8000/docs`。API 依赖为可选项，只使用网页时可省略 `--extra api`。

配置为 `configs/default.yaml`。本机模型放在 `models/selected/`，检测器为 `models/det_500m.onnx`；权重和私人照片不进入 Git，单独克隆仓库后需要放回完整模型包。

普通 Linux 服务器可使用 Docker 完全独立部署；镜像、模型与配置可打包后离线迁移，
运行时不依赖开发电脑或外部推理服务。步骤见 [Linux 部署文档](docs/DEPLOY_LINUX.md)。
GitHub Actions 在测试通过后发布到 `tano26/seiyuu-faceid`，需要仓库 Secret
`DOCKERHUB_TOKEN`；本地训练权重单独迁移和挂载。

CI/CD 在 main 或版本标签 push 后交付三种部署方式：Docker Compose、Python 原生、
C++ / Go 原生。部署包在 Actions Artifacts，镜像在 Docker Hub，模型及训练样本数清单
单独发布到 GitHub Releases。入口、检查及更新方式见 [CI/CD 文档](docs/CI_CD.md)。

图片依次经过 SCRFD 检测、五点对齐和两个冻结骨干；将 AdaFace、R50 的单位特征按此顺序拼接并除以 √2，再交给已训练的分类头。候选分数为 50 个已知类别内的概率。默认接受阈值为 `0.95`，通过质量检查且最高类别概率达到阈值才返回身份；低概率或质量不足的人脸返回 `review`，保留候选供人工复核。未校准图库外身份或角色图拒识能力。

随附配置已启用 `runtime.parallel_backbones`，两个骨干并行计算，每个骨干保持 2 个推理线程。资源受限的机器可设为 `false` 恢复顺序计算；旧配置缺少该项时仍采用顺序模式。模型权重和接受阈值不变。分类头对照实验与本机耗时见[优化报告](docs/CLASSIFIER_OPTIMIZATION.md)，复现命令见[训练说明](training/README.zh-CN.md)。

**N150 + 8 GB、Linux/Docker** 可使用 `configs/n150.yaml` 配合 `compose.n150.yaml`，以每骨干一个线程和关闭自旋减少 CPU 工作量。原 FP32 权重和判定规则保留；已完成的 N150 三组基线显示，该预设相对旧线程配置减少约 34.4% CPU 时间，但平均 HTTP 耗时增加约 37.9%，不能作为低延迟优化。部署和测速工具见 [N150 指南](docs/N150.md)，统计口径与后续独立实验见 [N150 实机报告](docs/N150_BENCHMARK.md)。

也可[原生部署为 Linux systemd 服务](docs/DEPLOY_NATIVE_LINUX.md)，使用相同模型和资源配置。

N150 当前采用 [C++ 推理核心 + Go HTTP](docs/NATIVE_CPP_GO.md)，运行时无需 Python。针对次正规数处理的 FTZ/DAZ 优化已集成，导出时使用 `--flush-denormals` 开启；平均 HTTP 耗时约 **0.68 秒**，1,051 张历史留出裁剪仍为 **98.48%**。2026-10-28 已在目标机切换为 systemd 原生服务，沿用原 API 端口与密钥。部署及回退见 [原生运行指南](docs/NATIVE_CPP_GO.md#linux-systemd-部署)，实测口径见 [N150 报告](docs/N150_BENCHMARK.md)。长期与压力场景验收仍待完成。

本机 M4 Pro 另用 LFW / WIDER FACE 共 756 张固定抽样完成公开数据对照：9,072 个计时请求，一致性检查全部通过；完整识别中 C++/Go 的平均耗时较 Python 减少约 1.8%–2.7%。输入、内存与准确率边界见 [标准数据集测试报告](docs/STANDARD_DATASET_BENCHMARK.md)。

后续[完整准确率评测](docs/STANDARD_DATASET_ACCURACY.md)：LFW 全部 6,000 对十折验证为 **99.13%**；WIDER 全部 3,226 张验证图片在当前部署参数下的 Easy / Medium / Hard AP 为 **85.49 / 79.48 / 48.84**（×100）；原声优 1,051 张留出 Top-1 复测仍为 **98.48%**。Python 与 C++ 得分及判定一致，三个任务的指标分别解释。

## 当前成绩

训练池包含 5,115 张已审核人脸、50 个身份；原始训练/验证/测试划分为 3,574/490/1,051 张，按来源与重复组隔离。当前分类头使用训练和验证合计 4,064 张拟合，测试人脸没有参与拟合。

| 指标 | 留出集结果 |
| --- | --- |
| Top-1 | **98.48%（1,035/1,051）** |
| Top-5 | 99.14% |
| Macro-F1 | 99.10%（49 个有测试样本的身份） |

这是多轮探索后按已观察到的最高留出分数选定的模型，成绩不是新独立测试集上的泛化保证。ONNX 部署版已重新提取全部 1,051 张测试人脸，Top-1 预测与原实验逐项一致。依据保存在 `models/selected/evaluation.json`、`holdout_scores.npz` 和 `runtime_verification.json`。

```bash
# 使用保留的固定特征复核指标和来源组隔离。
uv run python tools/evaluate.py
# 使用部署版重新提取全部测试特征。
uv run python tools/evaluate.py --reextract
uv run pytest
uv run ruff check .
```

## 数据和训练

原始照片保留在 `images/`、`data/raw/`。`data/training/` 保存本模型的冻结人脸裁剪、标签、来源组、划分与双骨干特征。未分类照片没有被作为未知人标签；角色图没有作为真人训练样本。

网页可管理、审核并整理 `data/raw/` 样本；这一步不更新已部署模型。新增身份或照片需要重新训练分类头。仅复现当前模型时：

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
```

该命令仅使用已冻结训练/验证特征，不覆盖部署模型。旧候选模型、实验环境和图库相似度匹配代码已移除。模型来源和上游条款见 [models/README.zh-CN.md](models/README.zh-CN.md)。
