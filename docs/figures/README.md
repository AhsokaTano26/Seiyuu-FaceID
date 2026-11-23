# Figure sources and reproduction

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The Python functions in [draw_documentation_figures.py](../../tools/draw_documentation_figures.py) draw the Matplotlib illustrations used in the [architecture guide](../TECHNICAL_ARCHITECTURE.md) and [standard-dataset accuracy report](../STANDARD_DATASET_ACCURACY.md). Each figure has SVG, 300 dpi PNG and PDF versions. These documents embed SVG for scalable viewing.

## Reproduction

Run from the repository root.

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_documentation_figures.py
```

To redraw selected figures, run the following command.

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_documentation_figures.py --only wider_pr_curves lfw_fold_accuracy
```

Plot generation reads aggregate data and uses separate plotting dependencies. Its input is limited to aggregates, and model inference, training photos and raw face features are outside this workflow. The input [aggregate_metrics.json](accuracy/aggregate_metrics.json) contains summary metrics, ten-fold statistics and PR curves, together with source-report, independent-recalculation and curve-file digests. Full evaluation provenance is in the [accuracy report](../STANDARD_DATASET_ACCURACY.md).

## Figure catalog

Functions follow `draw_<filename>()` and each returns a Matplotlib Figure. PNG and PDF share the SVG basename for reuse.

| Document | Description | SVG | PNG | PDF |
| --- | --- | --- | --- | --- |
| Architecture | Recognition and review branches | [SVG](architecture/recognition_pipeline.svg) | [PNG](architecture/recognition_pipeline.png) | [PDF](architecture/recognition_pipeline.pdf) |
| Architecture | SCRFD multiscale outputs and suppression | [SVG](architecture/scrfd_detector.svg) | [PNG](architecture/scrfd_detector.png) | [PDF](architecture/scrfd_detector.pdf) |
| Architecture | Five-point alignment geometry | [SVG](architecture/five_point_alignment.svg) | [PNG](architecture/five_point_alignment.png) | [PDF](architecture/five_point_alignment.pdf) |
| Architecture | Color order, feature dimensions and fused classification | [SVG](architecture/feature_fusion.svg) | [PNG](architecture/feature_fusion.png) | [PDF](architecture/feature_fusion.pdf) |
| Architecture | Offline training and parameter export | [SVG](architecture/training_pipeline.svg) | [PNG](architecture/training_pipeline.png) | [PDF](architecture/training_pipeline.pdf) |
| Architecture | Fit and historical holdout scope | [SVG](architecture/data_split.svg) | [PNG](architecture/data_split.png) | [PDF](architecture/data_split.pdf) |
| Architecture | FTZ/DAZ floating-point state scope | [SVG](architecture/denormal_scope.svg) | [PNG](architecture/denormal_scope.png) | [PDF](architecture/denormal_scope.pdf) |
| Architecture | Responsibilities of Go, the C ABI and C++ | [SVG](architecture/native_runtime.svg) | [PNG](architecture/native_runtime.png) | [PDF](architecture/native_runtime.pdf) |
| Architecture | Parallel backbones and joining | [SVG](architecture/inference_parallel.svg) | [PNG](architecture/inference_parallel.png) | [PDF](architecture/inference_parallel.pdf) |
| Architecture | Queue slots, inference gate and cancellation | [SVG](architecture/request_lifecycle.svg) | [PNG](architecture/request_lifecycle.png) | [PDF](architecture/request_lifecycle.pdf) |
| Architecture | Asset verification and model startup | [SVG](architecture/model_startup.svg) | [PNG](architecture/model_startup.png) | [PDF](architecture/model_startup.pdf) |
| Accuracy | Three tasks and their metrics | [SVG](accuracy/benchmark_tasks.svg) | [PNG](accuracy/benchmark_tasks.png) | [PDF](accuracy/benchmark_tasks.pdf) |
| Accuracy | LFW ten-fold threshold selection and testing | [SVG](accuracy/lfw_protocol.svg) | [PNG](accuracy/lfw_protocol.png) | [PDF](accuracy/lfw_protocol.pdf) |
| Accuracy | LFW correct, error and missed-detection counts | [SVG](accuracy/lfw_errors.svg) | [PNG](accuracy/lfw_errors.png) | [PDF](accuracy/lfw_errors.pdf) |
| Accuracy | Measured LFW accuracy by fold | [SVG](accuracy/lfw_fold_accuracy.svg) | [PNG](accuracy/lfw_fold_accuracy.png) | [PDF](accuracy/lfw_fold_accuracy.pdf) |
| Accuracy | WIDER matching, ignored faces and AP | [SVG](accuracy/wider_protocol.svg) | [PNG](accuracy/wider_protocol.png) | [PDF](accuracy/wider_protocol.pdf) |
| Accuracy | Measured PR curves for three difficulty sets | [SVG](accuracy/wider_pr_curves.svg) | [PNG](accuracy/wider_pr_curves.png) | [PDF](accuracy/wider_pr_curves.pdf) |
| Accuracy | Accepted and review errors on voice-actor crops | [SVG](accuracy/holdout_decisions.svg) | [PNG](accuracy/holdout_decisions.png) | [PDF](accuracy/holdout_decisions.pdf) |
| Accuracy | Ground-truth accuracy, parity and independent recalculation | [SVG](accuracy/validation_evidence.svg) | [PNG](accuracy/validation_evidence.png) | [PDF](accuracy/validation_evidence.pdf) |

## Figure scope and validation

- The machine used SimSun for Chinese, Times New Roman for English and ordinary digits, and Computer Modern MathText for equations. SVG uses glyph outlines and PDF embeds font subsets. Font files are managed outside the repository. If a preferred font is missing, the script selects from an explicit fallback order and prints the actual choice.
- Diagram positions and processing stages are schematic. Arrows show data or call dependencies. Box size and spacing have no quantitative scale for model size, sample share or latency.
- The right side of the five-point figure is the actual ArcFace template. The left side uses a known synthetic similarity transform. Both panels use equal aspect ratios and top-left origins.
- The LFW curve keeps all ten folds without smoothing and uses a local 97%–100.25% vertical range. WIDER PR curves keep all 1,000 thresholds and stop at measured maximum recall.
- The script checks text bounds and glyph coverage, ten-fold means, monotonic recall and agreement between integrated PR and AP. Exports have no missing-glyph warnings. Separate rendering checks cover original-size views, grayscale and embedded PDF fonts.

[manifest.json](manifest.json) records functions, dependency versions, fonts and data/source digests. Figures are maintained with repository documentation and follow the model-only Release rules.

## Model accuracy report of 2026-11-03

The [606-identity model report](../MODEL_ACCURACY_20261103.md) uses [draw_model_accuracy_report.py](../../tools/draw_model_accuracy_report.py). Its primary model is `models-5cc19e9d5b1a30aa` and the historical 50-identity reference is `models-ca1a34fb7a63a38f`. Each uses its own test set. The figures provide descriptive results for those separate versions.

| Description | PNG | PDF | SVG |
| --- | --- | --- | --- |
| Descriptive metrics on each version’s holdout | [PNG](model_accuracy_20261103/model_version_metrics.png) | [PDF](model_accuracy_20261103/model_version_metrics.pdf) | [SVG](model_accuracy_20261103/model_version_metrics.svg) |
| 606-identity decisions and error breakdown | [PNG](model_accuracy_20261103/model606_threshold_outcomes.png) | [PDF](model_accuracy_20261103/model606_threshold_outcomes.pdf) | [SVG](model_accuracy_20261103/model606_threshold_outcomes.svg) |
| Per-identity sample support for the 606-identity model | [PNG](model_accuracy_20261103/model606_sample_support.png) | [PDF](model_accuracy_20261103/model606_sample_support.pdf) | [SVG](model_accuracy_20261103/model606_sample_support.svg) |

Reproduce from the repository root.

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_model_accuracy_report.py
```

The input [aggregate_metrics.json](model_accuracy_20261103/aggregate_metrics.json) contains versions, metrics, counts, bins and evidence digests. Every figure provides 400 dpi PNG and two vector formats, using SimSun and Times New Roman. Source, font and image digests are in [figure_manifest.json](model_accuracy_20261103/figure_manifest.json), with original validation in [quality_review.json](model_accuracy_20261103/quality_review.json).

The quality-review record is a snapshot taken before documentation localization. Its report checksum identifies the original report bytes. Localized Markdown has separate text, and the recorded figure and data digests retain their historical provenance.
