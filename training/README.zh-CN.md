# 复现默认分类器

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

可复现的默认分类头使用 AdaFace IR101 + R50 逻辑回归。冻结特征与来源组隔离的数据保存在本地 `data/training/`。最终分类头使用3,574条训练记录和490条验证记录拟合，1,051条留出记录独立保存。

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
```

训练命令使用 C=1、类别平衡权重和随机种子20261014，需要 sklearn 1.9.1，写入一个新的输出路径。部署模型单独管理。运行时使用 NumPy 和 ONNX Runtime。

新增照片或身份后，先审核标签并准备新的数据划分，再训练。网页的样本管理工作流负责数据准备，部署分类头需要单独更新。

## 分类头对照实验

```bash
uv run --extra training python tools/optimize_classifier.py --output data/experiments/head-search
uv run python tools/benchmark_classifier.py --threads 1 2 4 --output data/experiments/backbone-speed.json
```

选择尚不存在的输出路径。分类头搜索在原训练与验证池中，对固定的50种配置进行三折来源组件隔离验证。只有一个来源组件的身份始终进入训练，在 OOF 指标中排除。按合并 Top-1、受支持类别 Macro-F1、对数损失顺序选择。保存胜出配置后，才进行第二个开发集划分和一次历史测试比较，两者均保持胜出配置固定。产物包括协议、划分索引、概率与独立且兼容运行时的 `classifier.npz`。

历史测试集在此前选型中被重复使用，第二个开发集划分也复用相同样本。独立准确率估计需要新的测试集。10月27日实验中，候选在历史测试集上退步，因此继续使用原分类头。运行时改进来自双骨干并行计算，详见[实测报告](../docs/CLASSIFIER_OPTIMIZATION.zh-CN.md)。

606身份发布使用独立冻结数据，其计数、分类头摘要、本地证据路径和局限见[606身份准确率报告](../docs/MODEL_ACCURACY_20261103.zh-CN.md)。历史照片清理可能导致默认裁剪缺失。特征归档存在时可基于特征评估，重新提取需要匹配的裁剪文件。
