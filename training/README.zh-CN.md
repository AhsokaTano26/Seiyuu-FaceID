# Reproduce the selected classifier

Only AdaFace IR101 + R50 logistic regression is retained. Frozen feature extraction and source-group-disjoint data are stored locally in `data/training/`. The 3,574 training and 490 validation faces fit the final head; the 1,051 held-out faces are excluded.

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
```

The training command uses the selected C=1, balanced class weights and random seed 20261014. It requires sklearn 1.9.1 and writes only to a new output path. It does not overwrite the deployed model. Runtime uses NumPy and ONNX Runtime, without sklearn or PyTorch.

For new photos or identities, review the labels and prepare a new data split before training. The sample-management UI only prepares data; it does not silently modify the deployed classifier.

## 分类头对照实验

```bash
uv run --extra training python tools/optimize_classifier.py --output data/experiments/head-search
uv run python tools/benchmark_classifier.py --threads 1 2 4 --output data/experiments/backbone-speed.json
```

输出路径必须不存在。搜索工具在原训练＋验证池内，对固定的 50 组方案进行三折来源组件隔离验证；只有一个来源组件的身份始终用于训练，不计入 OOF 验证指标。按合并 OOF 的 Top-1、受支持类别 Macro-F1、对数损失依次选择，先保存胜出配置，再用第二个开发集划分和历史测试集检查，不据此重新挑选候选。结果保存协议、划分索引、概率和独立的 `classifier.npz`，不覆盖当前部署模型。

历史测试集曾参与调参，第二个开发集划分也复用了同一批样本，两者均不能视为新独立测试。10 月 27 日实验中，新候选在历史测试集上退步，因此保留原分类头；速度优化通过并行执行两个冻结骨干实现。详见[实测报告](../docs/CLASSIFIER_OPTIMIZATION.md)。
