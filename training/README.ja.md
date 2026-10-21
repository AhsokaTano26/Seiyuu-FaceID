# Reproduce the selected classifier

Only AdaFace IR101 + R50 logistic regression is retained. Frozen feature extraction and source-group-disjoint data are stored locally in `data/training/`. The 3,574 training and 490 validation faces fit the final head; the 1,051 held-out faces are excluded.

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
```

The training command uses the selected C=1, balanced class weights and random seed 20261014. It requires sklearn 1.9.1 and writes only to a new output path. It does not overwrite the deployed model. Runtime uses NumPy and ONNX Runtime, without sklearn or PyTorch.

For new photos or identities, review the labels and prepare a new data split before training. The sample-management UI only prepares data; it does not silently modify the deployed classifier.
