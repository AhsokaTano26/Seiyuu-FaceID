# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

本地人脸检测与监督身份分类系统，采用 **SCRFD + AdaFace IR101 + InsightFace R50 + StandardScaler + 逻辑回归**，通过 ONNX Runtime 在 CPU 上推理。

## 模型版本

| 模型包 | 身份数 | 实际拟合人脸 | 留出人脸 | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| 默认本地 `models/selected/` 与 CI 固定来源 | 50 | 4,064 | 1,051 | 98.48%（1,035/1,051） |
| 已发布 `models-5cc19e9d5b1a30aa` | 606 | 49,858 | 8,759 | 99.38%（8,705/8,759） |

两版使用各自的数据集和评测历史。50身份分类头经过多轮历史留出集比较选定，606身份真值来自目录标签，尚未逐张核验。请结合[606身份报告](docs/MODEL_ACCURACY_20261103.zh-CN.md)与[历史优化报告](docs/CLASSIFIER_OPTIMIZATION.zh-CN.md)阅读指标。通过 `GET /v1/model` 与分类头校验值确认加载版本。

## 本地运行

```bash
uv sync --locked --extra api --extra web --extra dev
uv run streamlit run app.py
uv run python tools/recognize.py path/to/photo.jpg
uv run python tools/batch_recognize.py path/to/photos
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

完整模型包放在 `models/selected/`，检测器放在 `models/det_500m.onnx`。权重与私人数据由 Git 忽略，运行配置为 `configs/default.yaml`。API 和网页分别使用可选依赖。模型来源与上游条款见[模型说明](models/README.zh-CN.md)。

## HTTP API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health`、`/ready` | 进程与模型健康状态 |
| GET | `/v1/model` | 加载模型、身份范围与默认参数 |
| POST | `/api/v1/detect` | 内网服务检测有无人脸，返回框与置信度 |
| POST | `/v1/detect` | 返回检测框、分数和五点关键点 |
| POST | `/v1/recognize` | 单图身份分类 |
| POST | `/v1/recognize/batch` | 按序批量分类 |

```bash
curl --fail-with-body 'http://127.0.0.1:8000/api/v1/detect?score_threshold=0.7'   -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
curl --fail-with-body http://127.0.0.1:8000/v1/recognize   -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
```

内网检测接口复用已加载检测器，返回 `has_face`、`face_count` 和 `faces`。默认输入上限为2,000万像素，请求超时5秒，鉴权使用 `SEIYUU_FACEID_API_KEY`。同机容器调用需配置允许的私网子网。上传限制、配置和错误见 [API 文档](docs/API.zh-CN.md)，检测另附可导入的 [OpenAPI 定义](docs/detect.openapi.yaml)。Python 服务在 `/docs` 提供交互文档。

## 识别与数据

每张脸依次经过检测、质量检查、五点对齐和双冻结骨干。AdaFace、R50 单位特征按此顺序拼接并除以√2，再进行标准化与分类。分数表示内置身份范围内的类别概率，默认接受阈值为 `0.95`。质量不足或概率较低时返回 `review`，可用候选予以保留。未知人物与角色图的拒识性能需要另外验证。

历史数据池有5,115条记录，训练/验证/测试划分为3,574/490/1,051条，前两部分合计4,064条拟合分类头。历史 Top-5 为99.14%，Macro-F1 为99.10%，后者统计有测试样本的49个身份。历史验证中 ONNX 重新提取复现了全部 Top-1 预测，证据保存在对应模型包的 `evaluation.json`、`holdout_scores.npz` 与 `runtime_verification.json`。

未分类照片保留无标签状态，角色图排除在真人身份训练样本之外。

网页提供原图样本管理、人工选脸审核和识别。新增照片或身份需要独立准备划分并重新训练分类头。私人冻结特征位于 `data/training/`，实验位于 `data/experiments/`。历史照片清理已经移除了部分原图与裁剪，目录用途见[数据说明](data/README.zh-CN.md)，复现见[训练说明](training/README.zh-CN.md)。

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
uv run pytest
uv run ruff check .
```

训练写入新输出路径。特征评估依赖冻结归档，重新提取依赖相应裁剪。随附配置启用双骨干并行，每骨干2个推理线程。缺少此配置项的旧配置默认顺序执行。

## 部署与深入阅读

支持 Docker、Python/systemd 和 C++/Go/systemd。CI 部署归档放在 Actions Artifacts，镜像发布到 Docker Hub，模型与核实的训练样本数放在 GitHub Releases。

N150 + 8 GB 可用 `configs/n150.yaml` 和 `compose.n150.yaml` 减少 CPU 工作量。原预设实测 CPU 时间减少约34.4%，平均 HTTP 耗时增加约37.9%。后续原生 FTZ/DAZ 实验测得每请求约0.68秒，历史留出成绩98.48%。历史报告中长期与压力场景验收仍待完成，详见 [N150 报告](docs/N150_BENCHMARK.zh-CN.md)。

M4 Pro 上的 LFW/WIDER FACE 性能对照使用756张固定图片、9,072个计时请求，完整识别的原生 C++/Go 平均耗时减少约1.8%–2.7%。独立准确率协议中，LFW 全部6,000对为99.13%，WIDER 全部3,226张验证图片的 Easy/Medium/Hard AP 为85.49/79.48/48.84（×100）。[性能报告](docs/STANDARD_DATASET_BENCHMARK.zh-CN.md)与[准确率报告](docs/STANDARD_DATASET_ACCURACY.zh-CN.md)说明各自任务和范围。

[文档索引](docs/README.zh-CN.md)按架构、API、部署、训练、评测和历史实施计划整理。仓库维护的 Markdown 文档均在顶部提供语言切换。
