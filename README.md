# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Local-first face identification for a privately curated voice-actor photo
collection. The runtime maps a face to a normalized embedding, searches a local
gallery, and returns `recognized`, `review`, or `unknown`; it is not a fixed
Softmax classifier.

> **Status:** V1 local ONNX pipeline. Local baseline and calibration reports
> are generated on demand and are dataset-specific; recalibrate after changing
> the data or model before treating a threshold as an identity decision.

## Architecture

```text
image → SCRFD detector → five-point alignment → ArcFace ONNX embedding
      → L2 normalization → local gallery → identity scoring → thresholds
```

Training and deployment are deliberately separated: `training/` may grow
PyTorch/MPS tooling, while `src/seiyuu_faceid/runtime` and the operational CLI
only require NumPy, OpenCV, ONNX Runtime, and YAML parsing.

## macOS setup

Python 3.12+ is required (3.12 is preferred). Create an isolated environment
and install runtime dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements/runtime.txt
.venv/bin/python -m pip install -e .
```

Place compatible ONNX files at the paths configured in `configs/default.yaml`:

```text
models/det_500m.onnx
models/w600k_mbf.onnx
```

The checked local model pair is InsightFace `buffalo_sc`: a SCRFD-500MF detector
with five landmarks and a MobileFaceNet `112×112` recognition model. Inspect
model metadata before changing model families; input preprocessing is
model-specific.

## Dataset layout

Raw files stay untouched. Directory names are identity keys:

```text
data/raw/hanazawa_kana/001.jpg
data/raw/minase_inori/001.jpg
```

Processed faces, rejections, metadata, and the generated gallery are written
under `data/`. A raw folder is only a proposed label: every detected face is
kept with source and detector metadata for later review. Images with several
faces are excluded from the gallery unless you explicitly select every face
that belongs to the folder's identity.

## Commands

```bash
# Scan raw images once. This only creates a disposable detection cache.
python tools/scan_raw_faces.py

# Align, quality-filter, extract embeddings, and rebuild Gallery.
python tools/build_dataset.py
python tools/recognize.py path/to/image.jpg
python tools/batch_recognize.py path/to/photos
python tools/benchmark.py path/to/image.jpg
```

### Reviewing group photos for enrollment

First scan raw images once, then run this when `data/raw/` contains group photos:

```bash
.venv/bin/python tools/scan_raw_faces.py
.venv/bin/python tools/review_raw_faces.py
```

It creates numbered previews and
`data/review/raw_faces/face_selections.jsonl`. For every record, open its
`preview_image` and replace `"face_indexes": null` with a zero-based list such
as `[0, 2]`; use `[]` if none of the faces belongs to that identity. Then build
normally:

```bash
.venv/bin/python tools/build_dataset.py
```

Single-face source images are enrolled automatically. A multi-face image with
no valid selection is written to `data/rejected/metadata.jsonl` and never
enters the gallery. The scan writes `data/runtime/detections.jsonl`; it is reused
by review and build, and is automatically considered stale if the raw dataset
changes.

All commands accept `--config configs/default.yaml`. `build_dataset.py` is the
normal single build step: it reuses the scan cache (or creates it if absent),
writes aligned metadata, and immediately rebuilds the Gallery. Use
`build_dataset.py --scan-only` to create only the cache. `build_gallery.py`
remains available when only embeddings need to be regenerated from existing
aligned faces. Use `--skip-gallery` to inspect the dataset stage alone. The
checked default thresholds were calibrated on the current independent six-identity
holdout; recalibrate after material data or model changes.

### Independent evaluation and threshold calibration

Keep labelled evaluation images separate from enrolment images:

```text
data/gallery/<identity>/...
```

`data/gallery/` is read-only for all tools. Build the production Gallery from
`data/raw/`, then evaluate it without leakage:

```bash
.venv/bin/python tools/evaluate_external.py
```

The command writes embeddings, per-image exclusion reasons and the report to
`data/runtime/evaluation/`. It excludes unreviewed multi-face images from the
metric. The report includes a leave-one-identity-out unknown-person proxy;
add real unknown-person images before treating any threshold as final.

Create a standalone local report with SVG figures and CSV curves after an
evaluation run:

```bash
.venv/bin/python tools/generate_evaluation_report.py
```

The default output is `data/runtime/reports/evaluation_post_review/`.

## Local web interface

The optional Streamlit interface has two local-only work areas:

- **图库管理** lists the identity folders under `data/raw/`, lets you inspect
  their images, create identity folders, and upload reference images directly
  into a selected folder. When a Gallery exists, every detected face in the
  selected raw image is matched automatically and its result is shown. Images
  are presented as a paginated thumbnail grid.
- **建库与审核** is a separate page. Click **扫描全部原始图片** once, then
  **开始逐题审核群像** to review only photos with multiple detected faces.
  Enter one-based IDs such as `1,3` and press Enter to save and continue; enter
  `A` or `all` to select every detected face. The Gallery build reuses this
  scan and starts once the final group photo is reviewed. Single-face images
  are accepted automatically.
- **识别图片** accepts one or multiple query images without saving them to
  `raw`. Put a large local batch in `data/query/` (or enter another local
  directory) and click **识别整个文件夹**; all supported images in that directory
  and all of its subdirectories are processed, and results are presented as a
  paginated thumbnail grid. Temporary uploads remain available for small
  ad-hoc checks. Detection boxes are gray,
  selected/review faces orange, known identities green, and unknown faces red.
  Results can be downloaded as JSONL.

```bash
.venv/bin/python -m pip install -r requirements/web.txt
.venv/bin/streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
The page can create the first Gallery itself, provided the configured ONNX
model files exist. The included Streamlit configuration listens only on
`127.0.0.1` and disables Streamlit usage statistics.

## Training

Fine-tuning is intentionally not part of the first runtime milestone. Future
training will use a pretrained MobileFaceNet-style backbone with an ArcFace
head, export only the embedding backbone to ONNX, and verify PyTorch/ONNX
embedding agreement. Training dependencies belong only in
`requirements/training.txt`.

## Benchmark and future N150 deployment

`tools/benchmark.py` emits JSON with detector, recognition, end-to-end latency,
RSS, and throughput. Record a Mac baseline first, then run the identical command
on the Intel N150. Runtime is CPU-oriented: a single model instance, bounded
detection image size, no PyTorch, and no vector database for this small gallery.

## Privacy

The project is local-first: it does not upload source images, embeddings, or
telemetry and does not call cloud face-recognition services. The runtime sets
`ORT_DISABLE_TELEMETRY=1` before importing ONNX Runtime, disabling its uploader
and persistent device identifier.

## Model licensing

Code is MIT licensed; model weights are not bundled and can have materially
different licenses or usage restrictions. Record each downloaded model's source
and terms in [`models/README.md`](models/README.md) before use.
