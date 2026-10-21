# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

使用本地声优照片训练的 50 类身份分类器。当前唯一模型是 **AdaFace IR101 + InsightFace R50 + 标准化 + 逻辑回归**，全部在本机通过 ONNX Runtime 推理。

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

图片依次经过 SCRFD 检测、五点对齐和两个冻结骨干；将 AdaFace、R50 的单位特征按此顺序拼接并除以 √2，再交给已训练的分类头。候选分数为 50 个已知类别内的概率。默认接受阈值为 `0.95`，通过质量检查且最高类别概率达到阈值才返回身份；低概率或质量不足的人脸返回 `review`，保留候选供人工复核。未校准图库外身份或角色图拒识能力。

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
