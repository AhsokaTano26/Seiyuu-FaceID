# Local data

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The following paths describe the dataset workflow. Private files are excluded from Git and may be absent in a checkout.

- `../images/` and `raw/` hold original local photos.
- `aligned/` holds reviewed, aligned faces and preparation metadata.
- `training/` holds the historical 50-identity model's frozen labels, source groups, train/validation/test indices and AdaFace + R50 features.
- `review/` holds human face selections and content/label review records. Character artwork and unlabeled photos retain their own review status.
- `gallery/` holds source evaluation photos.
- `query/` holds read-only batch classification inputs.
- `runtime/detections.jsonl` is a rebuildable detection cache.
- `experiments/` holds local experiment runs, including the separate 606-identity training artifacts.

Unclassified photos remain unlabeled, and character artwork is excluded from real-person training labels.

The historical pool has 5,115 face records. Its train/validation/test split is 3,574/490/1,051, and the final head fits 4,064 rows. The 606-identity run uses separate data and a separate head. See the [model report](../docs/MODEL_ACCURACY_20261103.md).

`uv run python tools/evaluate.py` evaluates the historical default head using saved features. `--reextract` also requires the original aligned crops. Local cleanup removed historical photographs and crops, so check that the required files exist before re-extraction. See the [training guide](../training/README.md) for new model preparation.
