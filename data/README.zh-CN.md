# Local data

- `images/` at the repository root and `data/raw/`: original local photos; preserved.
- `aligned/`: reviewed, aligned faces and metadata produced by sample preparation.
- `training/`: the selected model's frozen 5,115 crops, trusted identity labels, source groups, train/validation/test indices and AdaFace + R50 features. Keep these for reproducibility.
- `review/`: human face selections and content/label review records; character artwork and unlabeled photos remain distinct from real-person identity labels.
- `gallery/`: original independent evaluation photos, retained as read-only source data. It is not a runtime model gallery.
- `query/`: read-only batch classification inputs.
- `runtime/detections.jsonl`: disposable face detection cache.

Private images, labels and generated model/data artifacts are excluded from Git. Superseded model experiments and generated reports have been removed. Evaluation of the selected model is `uv run python tools/evaluate.py`; `--reextract` verifies the deployed ONNX backbones as well.
