# Classification accuracy and speed optimization record (2026-10-27)

[English](CLASSIFIER_OPTIMIZATION.md) | [简体中文](CLASSIFIER_OPTIMIZATION.zh-CN.md) | [日本語](CLASSIFIER_OPTIMIZATION.ja.md)

This experiment retained the deployed classification head and enabled parallel AdaFace/R50 inference. The classification-head search produced no result worth replacing the existing model. Parallel execution reduced feature-extraction latency on this machine while preserving feature order, model weights, and the 0.95 acceptance threshold.

## Speed

Using local ONNX Runtime CPU inference, 16 crops from distinct source groups were fixed from the development pool, with three repetitions per crop. Sequential and parallel execution were interleaved and warmed up separately. Classification-head probability comparisons took place outside the timed interval.

| Intra-op threads per backbone | Sequential mean | Parallel mean | Sequential / parallel time ratio |
| --- | ---: | ---: | ---: |
| 1 | 101.31 ms | 56.40 ms | 1.80 |
| 2 | 78.39 ms | 54.22 ms | 1.45 |
| 4 | 58.42 ms | 40.47 ms | 1.44 |

The adopted configuration kept the existing 2 threads per backbone and reduced mean feature-extraction latency by **30.8%**. P95 fell from 86.32 ms to 61.04 ms. Sequential and parallel outputs were elementwise identical under each of the three thread settings, with 48/48 sampled Top-1 results matching for each setting. The 4-thread measurements were not used to change the default thread count. The local platform was macOS 26.6.2, ARM64, with ONNX Runtime 1.29.0.

These timings cover dual-backbone feature extraction from aligned crops. Image decoding, face detection, quality checks, and HTTP overhead are outside their scope, so they do not establish an increase in whole-API throughput. The results are local to this machine and require new measurements on Linux or machines with different CPU quotas.

The configuration field is `runtime.parallel_backbones`. The bundled `configs/default.yaml` sets it to `true`. Older configurations that omit it retain sequential execution. Each model has 2 intra-op threads, giving approximately 4 compute threads in parallel mode. Resource-constrained environments can set it to `false`. The executor is reused persistently and explicitly released when the API shuts down. If one backbone fails, both tasks are awaited before propagating the exception.

After integration into the actual runtime, the original sequential and new parallel computations were interleaved over all **1,051 historical holdout crops**.

- Features and classification probabilities were fully elementwise identical. Top-1 and acceptance/review decisions at the 0.95 threshold matched historical records.
- Top-1 remained **1,035/1,051 = 98.48%**, Top-5 99.14%, and Macro-F1 99.10%. The deployed classification-head SHA-256 was unchanged.
- Full-set mean feature-extraction time changed from **77.61 → 52.12 ms, a 32.8% reduction**, with P95 **79.35 → 55.40 ms**. These remain crop-level feature-extraction timings, including the first observation and excluding detection or HTTP.
- Relative to saved frozen features, newly extracted features differed by at most `9.16e-7`. Probabilities differed from historical saved values by at most `3.52e-6`. These historical numerical differences were identical in sequential and parallel modes and did not change identity or acceptance results.

## Classification-head experiment

The data comprised 5,115 reviewed faces and 50 identities. The original training and validation sets were merged into 4,064 development samples. The 1,051 historical test faces were excluded from this round of training, standardization, and candidate selection.

The development pool contained 2,513 source components. Components combined `sample_id`, `source_group`, original-image SHA, crop SHA, and optional photo-book grouping to check transitive duplication and development/test isolation. Two identities each had only one source component. They always remained in training and were excluded from OOF metrics, so OOF covered 4,062 faces and 48 identities. Every training fold still contained all 50 identities.

The candidate list and selection rule were saved before the run.

- Standardized LR used 6 C values × 2 class-weight options × 3 AdaFace/R50 weights, giving 36 configurations.
- Unstandardized LR used 4 C values × 2 class-weight options, giving 8 configurations.
- Source-component-equal-weight LR used 3 C values × 2 class-weight options, giving 6 configurations. Class balancing was calculated by component count to avoid applying crop-count-based weights again.

Three folds were isolated by source component, with preprocessing fitted only on the current training fold. Candidates were ranked by pooled OOF Top-1, supported-class Macro-F1, then log loss. Fixed candidate order was the final tie-breaker. The selected configuration was saved first. Only that candidate and the baseline were subsequently examined, without switching to another candidate based on later results.

The winning configuration was standardized LR with `C=0.1`, `class_weight=balanced`, and AdaFace weight 0.75. Block weights took effect after standardization and were folded into the exported scaling parameters, which the existing NumPy runtime could read directly. Its exported probabilities had a maximum error of 0 against the training implementation.

| Evaluation scope | Current head | New candidate |
| --- | ---: | ---: |
| Selection OOF, seed 20261027 | 3,977/4,062 (97.91%) | 3,986/4,062 (98.13%) |
| Second-split OOF, seed 20261028 | 3,981/4,062 (98.01%) | 3,988/4,062 (98.18%) |
| Historical test Top-1 | 1,035/1,051 (98.48%) | 1,033/1,051 (98.29%) |
| Historical test source-component macro-average accuracy | 98.65% | 98.36% |
| Historical test Macro-F1, 49 identities with ground truth | 99.10% | 96.96% |
| Historical test accepted at threshold 0.95 | 1,011 | 986 |
| Incorrectly accepted among them | 4 | 4 |

The candidate corrected none of the current head's 16 historical errors and introduced 2 additional errors. The number requiring human review rose from 40 to 65. The original head was therefore retained.

All 50 candidates shared the OOF data, introducing selection bias into the best OOF score. The second split reused the same samples and was not an independent test. The historical test set had also participated in earlier model selection, so these numbers remain exploratory comparisons and do not provide new evidence of generalization. The historical test set contained 676 source components, five of which accounted for 380 faces. Component-macro metrics were recorded as well, without treating every crop as independent. A paired component bootstrap with 10,000 repetitions and seed 20261029 yielded a descriptive 95% interval of −0.545 to 0.000 percentage points for the candidate's Top-1 change. This does not correct earlier selection bias.

No new ground-truth data were added for rejection of unknown people or character images, and no improvement in those accuracies is claimed. Further accuracy work should add reviewed photos from independent sources and a new test set before validating feature or classifier changes.

## Reproduction and local evidence

```bash
uv run --extra training python tools/optimize_classifier.py --output data/experiments/new-head-search
uv run python tools/benchmark_classifier.py --threads 1 2 4 --output data/experiments/new-speed.json
uv run python tools/evaluate.py --reextract
uv run pytest
uv run ruff check .
```

Output paths must not already exist. Models and detailed experimental data remain in ignored directories and are excluded from the repository and Releases.

- `data/experiments/classifier-search-20261027/` contains the protocol, candidate records, locked configuration, confirmation results, separate candidate head, historical probabilities, and comparisons.
- `data/experiments/classifier_speed_20261027.json` contains interleaved development-crop timings and elementwise parity checks.
- `data/experiments/runtime-parallel-20261027/` contains full-holdout verification scripts and results for the actual parallel runtime.

Code validation passed 131 tests, Ruff, and `git diff --check`. Tests cover component leakage, singleton identities, candidate selection independent of historical features, export parity, actual overlap of two threads, awaiting completion before propagating errors, repeated shutdown, and API resource cleanup. Test output contained two existing Starlette/httpx/AnyIO deprecation warnings.

The experimental method references [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). Per-model thread counts and CPU resource tradeoffs reference [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
