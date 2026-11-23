# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Local face detection and supervised identity classification using **SCRFD + AdaFace IR101 + InsightFace R50 + StandardScaler + logistic regression**. Inference runs on CPU through ONNX Runtime.

## Model versions

| Bundle | Identities | Actual fit faces | Holdout faces | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Default local `models/selected/` and pinned CI source | 50 | 4,064 | 1,051 | 98.48% (1,035/1,051) |
| Published `models-5cc19e9d5b1a30aa` | 606 | 49,858 | 8,759 | 99.38% (8,705/8,759) |

These versions have different datasets and evaluation histories. The 50-identity head was selected after repeated comparisons on its historical holdout. The 606-identity truth labels come from folders and have not been individually verified. Read the corresponding [606-identity report](docs/MODEL_ACCURACY_20261103.md) and [historical optimization report](docs/CLASSIFIER_OPTIMIZATION.md). Check model identity through `GET /v1/model` and the classifier checksum.

## Run locally

```bash
uv sync --locked --extra api --extra web --extra dev
uv run streamlit run app.py
uv run python tools/recognize.py path/to/photo.jpg
uv run python tools/batch_recognize.py path/to/photos
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Install the complete bundle in `models/selected/` and the detector at `models/det_500m.onnx`. Weights and private datasets are excluded from Git. Runtime settings come from `configs/default.yaml`. API and web dependencies are optional extras. Model sources and upstream terms are documented in the [model guide](models/README.md).

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/ready` | Process and model health |
| GET | `/v1/model` | Loaded model, identities and defaults |
| POST | `/api/v1/detect` | Internal face-presence detection, boxes and confidence |
| POST | `/v1/detect` | Detection boxes, scores and five landmarks |
| POST | `/v1/recognize` | Single-image identity classification |
| POST | `/v1/recognize/batch` | Ordered batch classification |

```bash
curl --fail-with-body 'http://127.0.0.1:8000/api/v1/detect?score_threshold=0.7'   -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
curl --fail-with-body http://127.0.0.1:8000/v1/recognize   -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
```

The internal detection endpoint reuses the loaded detector and returns `has_face`, `face_count` and `faces`. Its defaults are a 20,000,000-pixel input limit and a 5-second request timeout. Authentication uses `SEIYUU_FACEID_API_KEY`. Configure a private client subnet for same-host container calls. The [API guide](docs/API.md) documents upload limits, configuration and errors. An importable [OpenAPI definition](docs/detect.openapi.yaml) covers detection. Python serves interactive docs at `/docs`.

## Recognition and data

Each face passes detection, quality checks, five-point alignment and two frozen backbones. The unit AdaFace and R50 vectors are concatenated in that order and divided by √2 before scaling and classification. Scores are probabilities over the built-in identities. The default acceptance threshold is `0.95`. Low quality or lower probability returns `review`, with candidates when available. Unknown-person and artwork rejection require additional validation.

The historical pool contains 5,115 records, split into 3,574/490/1,051 train/validation/test rows. The first two splits fit the 4,064-row head. Historical Top-5 is 99.14% and Macro-F1 is 99.10% over the 49 identities with test samples. ONNX re-extraction reproduced all recorded Top-1 predictions in the historical verification. The evidence is in `evaluation.json`, `holdout_scores.npz` and `runtime_verification.json` within that bundle.

Unclassified photos remain unlabeled. Character artwork is excluded from real-person training labels.

The web interface manages raw samples, human face review and recognition. Adding samples or identities requires a separately prepared split and classifier training. Private frozen features are in `data/training/`, and experiments are in `data/experiments/`. Historical photo cleanup removed some originals and crops. See the [data layout](data/README.md) and [training guide](training/README.md).

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
uv run pytest
uv run ruff check .
```

Training writes a new output path. Feature evaluation requires the frozen archives, and re-extraction requires the matching crops. The supplied configuration enables parallel backbones with two intra-op threads per backbone. Older configurations default to sequential execution.

## Deployment and further reading

Docker, Python/systemd and C++/Go/systemd are supported. CI deployment archives are in Actions Artifacts, images in Docker Hub, and model files with verified training counts in GitHub Releases.

For N150 + 8 GB, `configs/n150.yaml` and `compose.n150.yaml` reduce CPU work. The measured original profile traded about 34.4% less CPU time for about 37.9% longer mean HTTP latency. A later native FTZ/DAZ experiment measured about 0.68 seconds per request with the historical 98.48% holdout result. Long-running and stress acceptance were still pending in the historical report. See the [N150 report](docs/N150_BENCHMARK.md).

LFW/WIDER FACE performance comparison used 756 fixed images and 9,072 timed requests on M4 Pro. Full recognition averaged 1.8%–2.7% lower latency for native C++/Go. Separate accuracy protocols measured LFW 99.13% on all 6,000 pairs and WIDER Easy/Medium/Hard AP 85.49/79.48/48.84 (×100) over 3,226 validation images. The [performance](docs/STANDARD_DATASET_BENCHMARK.md) and [accuracy](docs/STANDARD_DATASET_ACCURACY.md) reports define each task and its scope.

The [documentation index](docs/README.md) groups architecture, API, deployment, training, evaluations and historical implementation plans. Use the language navigation at the top of every maintained Markdown document.
