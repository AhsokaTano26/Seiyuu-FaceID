# Local standard-dataset performance benchmark with LFW and WIDER FACE

[English](STANDARD_DATASET_BENCHMARK.md) | [简体中文](STANDARD_DATASET_BENCHMARK.zh-CN.md) | [日本語](STANDARD_DATASET_BENCHMARK.ja.md)

This document records speed, resources, and parity. The subsequent evaluation completed **ten-fold accuracy on all 6,000 LFW pairs, detection AP on all 3,226 WIDER validation images, and a rerun on 1,051 voice-actor holdout faces**. See the [full accuracy report](STANDARD_DATASET_ACCURACY.md).

On 2026-10-28, Python and C++/Go were compared locally on an Apple M4 Pro / 24 GiB machine using **756 public-dataset images, 6 independent service processes, and 9,072 timed HTTP requests**. All three rounds of output parity checks and the independent statistical audit passed.

**Mean full-recognition latency fell by only about 1.8%–2.7%, and median peak RSS by about 3.8%.** Detection-only gains were larger, about 7.9%–14.7%. Full recognition on this machine remained dominated by model computation, leaving limited gains from language migration itself. This comparison used Mac ARM with FTZ/DAZ disabled. The N150 FTZ/DAZ results are recorded separately in the [N150 report](N150_BENCHMARK.md), and the two results must be kept distinct.

## Inputs and sources

| Dataset | Original scope | Fixed sample for this run | Image characteristics |
| --- | --- | --- | --- |
| Original LFW, non-funneled | 13,233 images, 5,749 identities | 1 image from each of 512 identities, 512 total | Original 250 × 250 JPEGs, usually with one main face and sometimes background faces |
| WIDER FACE validation | 3,226 images, 61 scene categories | 4 images per category, 244 total | Multiple people, small faces, occlusion, and other challenges. All selected images were 1,024 pixels wide and 351–2,014 pixels high |

LFW was downloaded from a [revision-pinned mirror](https://huggingface.co/datasets/DerrickUnleashed/LFW/tree/f7571fa9dd5a5c31ab781de5227c6f1b6171c0c5). Its archive SHA-256 matched the [scikit-learn dataset definition](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py). WIDER FACE used a [revision-pinned CUHK-CSE mirror](https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/833d07e7bf3860f294242312fe95eed0561eeb17/data). Dataset scope and intended uses are described on the [WIDER FACE project page](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html).

| File | SHA-256 |
| --- | --- |
| `lfw.tgz` | `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0` |
| `WIDER_val.zip` | `f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a` |
| `wider_face_split.zip` | `c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c` |

Sampling was fixed before inference using seed string `20261028-standard-benchmark-v1`, ordered by `SHA256(seed + ':' + relative path or identity name)`. LFW first selected 512 identities, then one image per identity. WIDER selected 4 images within each category. Images were not selected using model output and were not re-encoded, cropped, or resized as inputs. The existing application detection resizing, quality thresholds, and recognition pipeline were used.

Only selected images were extracted, totaling about 33.9 MiB. Including complete downloaded archives and results, the experiment directory occupied about 578 MiB. Data and per-image responses remain in a Git-ignored experiment directory and are excluded from the repository and model Releases.

## Measurement method

The environment was macOS 26.6.2 ARM64, Apple M4 Pro with 14 logical CPUs, and 24 GiB memory. It used Python 3.13.15, NumPy 2.5.2, ONNX Runtime 1.29.0, OpenCV Python 5.0.0.93, Pillow 12.3.0, FastAPI 0.154.0, and Uvicorn 0.54.0. The native program was a macOS ARM build with Go 1.26.2, C++, OpenCV 5.0.0, and ONNX Runtime 1.29.0.

Both implementations used the same FP32 models and classification head with the `configs/n150.yaml` thread settings. Each backbone had 1 thread, the two backbones ran in parallel, the detector had 2 threads, spinning was disabled, and the CPU arena was retained. Default Top-5 and the 0.95 threshold were identical. The native JSON did not enable `flush_denormals`. The current FTZ/DAZ implementation supports only Linux x86_64.

Each service process measured two paths for each dataset.

- `/v1/detect` covered upload, decoding, detection, and response.
- `/v1/recognize` covered upload, decoding, detection, quality checks, alignment and dual-backbone classification for qualifying faces, and response.

Each phase began with 10 untimed warmup requests, then requested every selected image in that dataset once. Each implementation ran in 3 fresh processes, in the order `Python → native → native → Python → Python → native`. Later rounds also varied dataset and endpoint-phase order. Only one tested service ran at a time, with one client issuing sequential requests. No concurrent load test or batch inference was performed. Totals were **9,072 timed and 240 warmup requests**.

Image bytes were loaded into client memory in advance. HTTP time measured local loopback upload through response completion, including client and connection overhead. It does not measure pure model time or public-network upload experience. Throughput was request count divided by elapsed time across the whole sequential phase, including loop and response-processing overhead. It does not establish maximum concurrent throughput.

CPU time came from service-process user + system time in macOS `proc_pid_rusage(RUSAGE_INFO_V2)`, converting Mach ticks to nanoseconds with the `125/3` ratio from `mach_timebase_info`. A calibration check against the measuring process's own `process_time_ns` preceded timing. CPU milliseconds per request include all service threads, exclude the client, and do not measure whole-machine energy use. Peak RSS came from `wait4` after each service exited and includes startup, all warmups, both datasets, and both endpoints. It was not divided into separate dataset-stage peaks.

The tables use **the median of the 3 process-level metrics**. P95 is the median of each process's nearest-rank P95. Requests from the three rounds were not pooled into a global P95. Normal background applications remained running. CPU frequency, system caches, and temperature were not controlled, so differences of a few percent should be interpreted within this workload.

## Full-recognition results

| Dataset | Implementation | Mean HTTP | P50 | P95 | Sequential throughput | CPU/request |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LFW, 512 images | Python | 59.91 ms | 54.46 ms | 93.99 ms | 16.69 images/s | 100.98 ms |
| LFW, 512 images | C++/Go | **58.82 ms** | 53.13 ms | 92.11 ms | 17.00 images/s | 100.57 ms |
| WIDER, 244 images | Python | 77.39 ms | 58.87 ms | 212.00 ms | 12.92 images/s | 130.35 ms |
| WIDER, 244 images | C++/Go | **75.33 ms** | 55.87 ms | 210.70 ms | 13.27 images/s | 128.87 ms |

Mean HTTP time decreased by **1.83%** on LFW and **2.67%** on WIDER. CPU time decreased by approximately **0.41%** and **1.13%**, respectively. These gains were small, with both implementations still performing the same dual-backbone computation.

Full-recognition work per round was identical for Python and C++/Go.

| Dataset | Detected faces | Classified | Blocked by quality | Images without faces | Most detected faces in one image |
| --- | ---: | ---: | ---: | ---: | ---: |
| LFW | 645 | 579 | 66 | 1 | 9 |
| WIDER | 1,219 | 363 | 856 | 14 | 105 |

WIDER's 75.33 ms covers the complete application pipeline including quality filtering. It does not mean that every annotated face in every image ran through both backbones. The WIDER sample annotated 2,892 faces, including 2,872 marked valid. Detection boxes were not matched under official evaluation rules in this run, so dividing these counts does not yield recall.

## Detection-only results

| Dataset | Implementation | Mean HTTP | P95 | Sequential throughput | CPU/request |
| --- | --- | ---: | ---: | ---: | ---: |
| LFW | Python | 15.95 ms | 16.57 ms | 62.64 images/s | 21.28 ms |
| LFW | C++/Go | **14.69 ms** | 15.32 ms | 68.02 images/s | 20.03 ms |
| WIDER | Python | 19.32 ms | 22.47 ms | 51.72 images/s | 25.19 ms |
| WIDER | C++/Go | **16.48 ms** | 18.68 ms | 60.44 images/s | 22.39 ms |

Mean HTTP time decreased by **7.90%** and **14.69%**, respectively. These changes include differences in image decoding, data processing, and HTTP paths. The full difference cannot be attributed to language alone.

## Memory and startup

| Implementation | Median peak RSS | Peak RSS of the three processes | Median startup to readiness |
| --- | ---: | --- | ---: |
| Python | 1,012.89 MiB | 1,012.89 / 1,043.66 / 1,002.08 MiB | 796.20 ms |
| C++/Go | **974.42 MiB** | 891.22 / 989.34 / 974.42 MiB | 466.89 ms |

Median peak RSS decreased by **38.47 MiB / 3.80%**, with substantial process-to-process variation. Native startup times were 1,344.69 / 395.80 / 466.89 ms, while Python times were 934.42 / 796.20 / 633.85 ms. Cold-cache conditions were not standardized, so the medians alone do not establish consistently faster cold starts.

## Output parity and accuracy limits

All **12 comparisons** across three rounds, two datasets, and two endpoints passed. Outputs were fully reproducible across processes for each implementation after excluding processing-time fields. Full recognition showed the following results.

- Face counts, boxes, and detection scores matched. Candidate identity order, quality decisions, and acceptance at 0.95 matched.
- The maximum absolute classification-probability difference was **`3.5762786865234375e-7`**. The maximum quality-score difference was about `2.22e-16`.
- Values were not bitwise identical. Existing validation-tool tolerances were retained, with 0.01 pixels for boxes/landmarks and `5e-5` for scores.
- In every round, all 942 classified faces required human review because of insufficient probability, while 922 more were blocked by quality checks. These inputs produced no automatic acceptances, which does not establish a zero false-identification rate for unknown identities.

This was a **system-performance and implementation-parity test on subsets of standard datasets**, without retraining the 50-class head. LFW names cannot directly serve as labels for that head. This run did not execute LFW's 6,000-pair, 10-fold verification protocol or calculate official WIDER Easy / Medium / Hard AP. The existing **98.48%** remained the score on the historical voice-actor holdout set and was not accuracy on these standard datasets.

## Evidence and review

Files are local under `data/experiments/standard-bench-20261028/`.

| File | Purpose |
| --- | --- |
| `downloads/`, `images/`, `manifest.json` | Verified archives, selected originals, fixed sample, and per-image hashes |
| `prepare.py`, `bench.py` | Sampling and independent HTTP-process test scripts |
| `results.json` | Six processes, per-request times, CPU/RSS, and complete parity summary |
| `*-private.json` | Raw per-image responses retained for local review and excluded from the repository |
| `audit.py`, `audit.json` | Independent recalculation of statistics, input/artifact verification, recursive response comparison, and 6 negative audit tests |
| `postflight.json` | Normal cleanup of all six processes and closure of all temporary listening ports |

The independent audit confirmed coverage of all **9,072 timed requests**, recalculated parity from raw responses, verified model, configuration, service, core-library, input, and script hashes, and checked that phase CPU times did not exceed process-lifetime CPU time. It also confirmed that all 756 sample paths and content hashes were unique.

| Evidence | SHA-256 |
| --- | --- |
| Sampling manifest | `0e83d3bf7850154b18ed0b5373076b225dc05b93a469e6f95b8f2e5ad19d3bde` |
| Final results | `76192b483dbb6122a35ce8fb7a4731f4d459e83170c6f2c936b7a5439a22b7db` |
| Local Go service | `edde232221d8aa9c34ebed69092008664110cfaa53454976109d3150137aefe9` |
| Local C++ core library | `3f6c37231af07ff3fae6b0496f67b2b5a377b9f4a8cc21c5360b6e8a08751d2d` |

Temporary services were stopped after testing. Models and decision thresholds were unchanged. No remote services were connected to or modified, and nothing was pushed.
