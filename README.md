# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

For standalone Linux deployment, including offline image and model transfer, see
[the Linux deployment guide (中文)](docs/DEPLOY_LINUX.md).

Local supervised classification of 50 identities using **AdaFace IR101 + InsightFace R50 + StandardScaler + logistic regression**. This is the only deployed model; both frozen backbones run through ONNX Runtime on CPU.

```bash
uv sync --locked --extra api --extra web --extra dev
uv run streamlit run app.py
uv run python tools/recognize.py path/to/photo.jpg
uv run python tools/batch_recognize.py path/to/photos
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
uv run pytest
uv run ruff check .
```

Start the HTTP API and upload an image:

```bash
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
curl --fail-with-body http://127.0.0.1:8000/v1/recognize -F 'file=@photo.jpg'
```

The API supports recognition, batches, detection, model metadata and health checks.
See the [API guide](docs/API.md) for payloads, errors, optional Bearer authentication and deployment;
interactive documentation is available at `http://127.0.0.1:8000/docs`.
API dependencies are optional; omit `--extra api` for web-only use.

Place the complete private model bundle in `models/selected/` and the SCRFD detector at `models/det_500m.onnx`. Weights and private datasets are excluded from Git. Configuration: `configs/default.yaml`.

The classifier fits 4,064 training/validation faces. On the fixed 1,051-face holdout it achieves **98.48% Top-1 (1,035 correct)**, 99.14% Top-5, and 99.10% macro-F1 over the 49 supported identities. The deployed ONNX models reproduce every held-out Top-1 prediction. This model was chosen after repeated exploratory comparisons on that holdout; independent confirmation remains necessary.

Inputs undergo detection and five-point alignment. Unit AdaFace and R50 embeddings are concatenated in that order and divided by √2 before standardization and classification. Scores are probabilities among the known classes. The default probability threshold is 0.95. Faces passing quality checks return an identity only at or above that threshold; low-probability or low-quality faces return `review` with available candidates retained. Unknown-identity and artwork rejection have not been calibrated for this model.

Original photos, reviewed labels and the frozen source-group split are preserved. `data/training/` contains 5,115 face crops, labels and features. The web sample preparation workflow does not retrain the model. Unclassified photos are unlabeled; character artwork is excluded from real-person training labels.

Reproduce the selected head without overwriting the deployed bundle:

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
```

Superseded classifiers, experiments and prototype-gallery matching code have been removed. See [the Chinese guide](README.zh-CN.md) for evaluation details and [model provenance](models/README.md) for upstream sources and terms.
