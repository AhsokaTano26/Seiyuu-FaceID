# Reproduce the selected classifier

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The reproducible default head uses AdaFace IR101 + R50 logistic regression. Frozen feature extraction and source-group-disjoint data are stored locally in `data/training/`. The 3,574 training and 490 validation faces fit the final head. The 1,051 held-out faces are excluded.

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
```

The training command uses the selected C=1, balanced class weights and random seed 20261014. It requires sklearn 1.9.1 and writes only to a new output path. It does not overwrite the deployed model. Runtime uses NumPy and ONNX Runtime, without sklearn or PyTorch.

For new photos or identities, review the labels and prepare a new data split before training. The sample-management UI only prepares data. Deployment of a new head is a separate step.

## Compare candidate heads

```bash
uv run --extra training python tools/optimize_classifier.py --output data/experiments/head-search
uv run python tools/benchmark_classifier.py --threads 1 2 4 --output data/experiments/backbone-speed.json
```

Use new output paths. The head search compares 50 fixed configurations using three source-component-disjoint folds within the original train + validation pool. Single-component identities remain training-only and are excluded from OOF metrics. Selection uses pooled Top-1, then supported-class macro-F1, then log loss. The winner is saved before a second development split and one historical-test comparison. Neither comparison retunes the winner. Artifacts include the protocol, fold indices, scores and a separate runtime-compatible `classifier.npz`. Deployment of the exported head is a separate step.

The historical test was reused in earlier model selection. An independent accuracy estimate needs a fresh test set. The second development split reuses the same samples too. The October 27 experiment retained the deployed head because the candidate regressed on the historical test. The runtime improvement uses concurrent frozen backbones without changing the classifier. See [measured results](../docs/CLASSIFIER_OPTIMIZATION.md).

The 606-identity release uses a separate frozen dataset. Its counts, classifier checksum, local evidence paths and limitations are documented in the [606-identity evaluation report](../docs/MODEL_ACCURACY_20261103.md). Historical photo cleanup can leave the default crops absent. Feature-based evaluation remains available when its archives exist, and re-extraction requires the matching crops.
