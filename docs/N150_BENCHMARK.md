# Intel N150 hardware validation of the native runtime, accuracy, and three-way performance

[English](N150_BENCHMARK.md) | [简体中文](N150_BENCHMARK.zh-CN.md) | [日本語](N150_BENCHMARK.ja.md)

This report records validation on 2026-10-27 on a Linux / Intel N150 / 8 GB machine. Linux native builds, dynamic-library loading, real-model interface checks, and Python/C++ parity on all historical holdout inputs were completed. **The three-way HTTP baseline and subsequent independent FTZ-candidate experiment were completed and independently audited.** The original baseline showed no reliable speed gain from changing language alone. After identifying subnormal-number handling overhead, the FTZ candidate achieved mean HTTP time of **680.785 ms**, retaining historical holdout accuracy of **1,035 / 1,051 = 98.48%**.

The candidate and earlier baseline used identical inputs in separate phases. They were not measured in a contemporaneous interleaved A/B experiment. Section 6 preserves the original three-way baseline. Section 8 separately records diagnosis, implementation, full regression, and candidate HTTP results. **The verified implementation was integrated into product source and enabled on N150 on 2026-10-28. Deployment acceptance checks are in Section 9.**

The original FP32 models, classification head, and acceptance threshold were retained to examine thread-configuration changes and C++/Go runtime migration separately. Algorithms and implementation principles are described in the [technical architecture](TECHNICAL_ARCHITECTURE.md), with build instructions in the [C++ + Go experimental guide](NATIVE_CPP_GO.md).

## 1. Verified environment

| Item | Environment for this run |
| --- | --- |
| Processor | Intel N150, 4 available logical CPUs |
| Memory | Nominally 8 GB, about 7.535 GiB visible to the system |
| System | Linux x86_64, kernel `6.8.0-142-generic`, glibc 2.39 |
| CPU governor snapshot | `powersave` on all four CPUs |
| Python reference environment | Python 3.12.3, NumPy 2.5.2, ONNX Runtime 1.29.0 |
| Python image dependencies | OpenCV 5.0.0, package `opencv-python 5.0.0.93`, Pillow 12.3.0 |
| Native-core dependencies | ONNX Runtime 1.29.0, OpenCV 5.0.0 |
| Storage | Builds, models, and test inputs prepared on the main SSD. Tests were not moved to the mechanical data disk |

`powersave` was the frequency-governor policy recorded at sampling time and does not mean the CPU was locked to a low frequency. Hardware records also include frequency and temperature snapshots. A single snapshot cannot establish constant frequency or temperature throughout testing or determine whether sustained downclocking occurred.

A process snapshot during formal measurement showed nice value 0 and CPU affinity `0–3` for the Python process under test. Neither its cgroup nor its ancestors imposed a hard CPU-time quota, and recorded quota-throttling counts were 0. This excludes single-core binding and cgroup quota limits at that snapshot, without replacing temperature, power, or whole-run scheduling monitoring.

Existing production containers stayed running, and background load was recorded before testing. In that snapshot, the existing FaceID service used about 970 MiB with low CPU utilization. This records background conditions and does not measure the tested service or establish an interference-free experiment. Existing services were not stopped, containers were not rebuilt, and data-disk mounts were not changed for this test.

## 2. Linux build and startup checks

Native builds completed on the target machine, with the following checks.

- Application shared-library dependencies resolved, and the service executable's help command exited normally.
- ELF relative loading used `$ORIGIN/../lib` to locate bundled shared libraries alongside the application.
- **2 C++ CTest tests passed**.
- Ordinary Go tests, build tests with the native core enabled, and real-model HTTP integration tests passed.
- **20 Python tests** related to model export and native validation passed.

These are the checks run on N150 during this round. The earlier complete 206-test Python result on Mac remains a historical record and does not mean all 206 were rerun on N150.

The native service comprises an executable, shared libraries, models, and configuration, so it is not a static single-file program. Loading checks establish operation in this machine's current system environment, without establishing compatibility with every Linux distribution or system-library version.

Numerical validation used a Python driver to launch separate reference and C API workers. The driver's Python dependencies prepared inputs and compared results and do not imply that the Go HTTP service needs Python at runtime. C API numerical checks also do not replace Go upload, authentication, rate-limiting, and interface integration tests. The two kinds of checks are recorded separately.

## 3. Fixing inputs, models, and configuration

An input manifest and hashes were generated before execution.

- **1,051 historical holdout face crops** supported full feature, classification-probability, and identity-result comparisons.
- **12 original images from holdout sources**, with **14 detected faces**, supported the complete decoding, detection, alignment, and classification comparison and served as fixed HTTP-performance inputs.

The 12 originals comprised 11 images with one detected face and 1 with three faces. Every qualifying face ran through both backbones. Request times and P95 therefore depend on input face counts, and this mixed-input P95 cannot represent the random latency tail of single-face photos.

Both implementations used the same file list, FP32 weights, original head, and ordered class names. Configuration, exported JSON, core library, service binary, and test-script hashes were also fixed to prevent mixing builds or inputs. Python YAML and native JSON naturally have different file hashes. The comparison checks exported configuration semantics, without requiring byte-identical files in different formats.

All groups retained these settings.

| Setting | Value |
| --- | --- |
| Detection-preprocessing maximum long side | 1920 pixels |
| SCRFD input | 640 × 640 |
| Detection / NMS threshold | 0.50 / 0.40 |
| Minimum face short side / sharpness | 48 pixels / Laplacian variance 20.0 |
| Default candidate count / acceptance threshold | 5 / 0.95 |
| Detector intra-op threads | 2 |
| Inter-op threads / CPU arena | 1 / enabled |

These 12 originals came from historical holdout sources. **The earlier Mac HTTP experiment used 12 originals from training sources.** Dividing Mac and N150 mean times therefore cannot establish a hardware-speed ratio for the same workload.

<a id="accuracy-and-parity"></a>

## 4. Completed accuracy and parity results

Full numerical comparison covered `python_n150` and `native_n150`. Both used 1 intra-op thread per backbone, parallel backbones, disabled spinning, and 2 detector threads. Python and C++ ran in separate workers to avoid dynamic-library interactions within one process.

Some correctness calculations overlapped builds or another correctness worker, so their elapsed times were excluded from performance conclusions. Formal HTTP timing began only after all build and correctness workers had exited.

### 4.1 All historical holdout crops

| Metric | Python N150 configuration | C++ N150 configuration |
| --- | ---: | ---: |
| Crops | 1,051 | 1,051 |
| Classes with test ground truth | 49 | 49 |
| Top-1 correct | 1,035 | 1,035 |
| Top-1 errors | 16 | 16 |
| Top-1 accuracy | 98.48% | 98.48% |
| Top-5 accuracy | 99.14% | 99.14% |
| Supported-class Macro-F1 | 99.10% | 99.10% |
| Accepted at threshold 0.95 | 1,011 | 1,011 |
| Incorrect acceptances among them | 4 | 4 |
| Requiring review | 40 | 40 |

All 1,051 crops matched in first candidate, ordered Top-5 candidates, and acceptance at 0.95. Class order also matched. C++ **introduced 0 errors and corrected 0 errors** relative to the reference.

| Numerical comparison | Measured maximum absolute difference |
| --- | ---: |
| 1024-dimensional features | `2.2351741790771484e-8` |
| Classification probabilities | `4.76837158203125e-7` |

Numerical-tolerance checks passed, although features and probabilities were not bitwise identical. Absolute tolerances were `2e-5` for features and `5e-6` for probabilities, with relative tolerance 0. Candidate order and acceptance were checked separately. Similar floating-point arrays alone cannot establish unchanged decisions.

Acceptance and review counts describe classification of aligned known-identity crops. Original-image detection failures and quality filtering are excluded, and the counts do not measure false acceptance of unknown identities. The 49 classes are those supported by ground truth in this holdout. The model still contains 50 identities.

### 4.2 Originals and direct decoding

The 12 originals and 14 detected faces were compared through both predecoded BGR and native direct decoding of compressed files.

- Face counts, boxes, detection scores, and quality scores matched.
- Candidate order and acceptance/review decisions matched for all 14 faces.
- Maximum absolute classification-probability difference was `2.384185791015625e-7` for both paths.
- Decoded image dimensions matched. Decoding results matched across Python workers and between workers and input preparation.

This covers the real original-image pipeline but includes only 12 images. It cannot cover all JPEG encodings, EXIF orientations, extreme dimensions, or overlapping-face cases.

### 4.3 Scope of conclusions

**On these N150 inputs, the C++ inference core in the C++/Go solution introduced no regression in identity or acceptance decisions.** The native Linux service loaded and passed the HTTP integration checks performed.

This provides no new independent generalization proof. The historical holdout had been used in model selection, and images may share source correlations. Both implementations returning the same incorrect identity still counts as parity without establishing correctness. Rejection of unknown people, identities outside the gallery, and arbitrary inputs requires separate data and metrics.

The full 1,051-crop comparison covers Python and C++ under the N150 configuration. The `python_original` group below restores different thread settings. HTTP tests checked all three groups on the 12 originals, without establishing a fresh full 1,051-crop check for the old-thread group.

## 5. Three-way HTTP performance protocol

### 5.1 Questions addressed by the groups

| Group | Implementation and environment | Backbone intra-op | Backbone scheduling | Spinning |
| --- | --- | ---: | --- | --- |
| `python_original` | Current Python source and locked environment reproducing old production thread settings | 2 threads per backbone | Sequential | On |
| `python_n150` | Same Python source, interpreter, and dependencies with N150 settings | 1 thread per backbone | Parallel | Off |
| `native_n150` | C++ core and Go HTTP with matching N150 settings | 1 thread per backbone | Parallel | Off |

In `python_original`, original **refers only to reproducing old runtime parameters**. It uses the current Python 3.12.3 environment and source. It does not directly measure the existing production Python 3.13 service or send benchmark requests to its production port. Both Python groups share interpreter and dependencies to separate thread-configuration effects from environment-version changes.

Comparing `python_original` with `python_n150` primarily examines configuration. Comparing `python_n150` with `native_n150` examines full runtime migration with the same model and scheduling. The latter still includes differences in decoding, object allocation, and HTTP implementation, so gains cannot all be attributed to C++ itself.

### 5.2 Requests and run order

The formal protocol was fixed before the first timed process began. Each group started three independent fresh service processes. Each used the same 12 originals for one full warmup round followed by four timed rounds, giving **12 warmup requests and 48 timed requests** per process.

The three repetition orders were as follows.

1. `python_original` → `python_n150` → `native_n150`.
2. `python_n150` → `native_n150` → `python_original`.
3. `native_n150` → `python_original` → `python_n150`.

There were nine tested processes and 432 timed requests. One tested service ran at a time, without overlapping builds or correctness jobs during formal timing. Production and other background services continued running with load recorded. Rotation reduces fixed-order effects without eliminating temperature, cache, or background-task variation.

Every timed request also underwent output-parity checks across the three configurations. Performance summaries must confirm completion status, process counts, request counts, and output comparisons together. Partial timing output alone cannot establish completion.

### 5.3 Metric calculation

| Metric | Definition and limits |
| --- | --- |
| HTTP wall time | Full time from local-client upload through JSON-response parsing, including local transfer and response handling |
| Server processing time | Service-recorded scope including reading, decoding, and inference, excluding preceding multipart parsing and queuing |
| Timed-interval CPU time | Increase in `/proc/<pid>/stat` `utime + stime` across 48 timed requests, divided by system clock frequency |
| Lifetime CPU time | Cumulative process CPU returned by `wait4`, including loading, warmup, and shutdown, stored separately from timed-interval CPU |
| Peak RSS | Peak across the full service-process lifetime, including model loading, warmup, and large-image handling, converting Linux KiB to MiB |
| Startup time | Fresh-process launch to readiness. Fresh processes do not imply cold filesystem caches |

Mean, P50, and P95 were calculated within each process's 48 requests, then the median of the three process metrics was taken within each group. Quantiles used nearest rank. After sorting 48 values, P50 takes the 24th and P95 the 46th. **The three processes' 144 requests were not pooled to recalculate P95.**

CPU time accumulates processor time across all relevant threads. It measures neither user wait nor electrical energy. Without whole-machine power sampling, CPU-time ratios cannot be presented as energy savings. Peak RSS also excludes other host and service memory and cannot establish whole-machine memory use.

## 6. Completed three-way baseline results

### 6.1 Completion and summary

The final report was marked `complete`. Each group completed three processes with 48 timed requests per process, totaling **432 timed requests**. Each repetition compared all pairs of configurations, giving nine comparisons. Face counts, candidate order, and decisions passed for every request.

An independent audit passed checks of completeness, counts, summary arithmetic, and relative changes. It did not reread private per-sample outputs or require better performance as a passing condition. Performance conclusions follow the measured data below.

Each table cell is the **median of three process-level metrics**. Times are in seconds, and CPU time is cumulative processor time.

| Metric | `python_original` | `python_n150` | `native_n150` |
| --- | ---: | ---: | ---: |
| Mean HTTP time, s/request | 8.277 | 11.415 | 11.401 |
| HTTP P50, s/request | 7.119 | 9.797 | 9.780 |
| HTTP P95, s/request | 21.000 | 29.207 | 29.165 |
| Mean server processing time, s/request | 8.252 | 11.393 | 11.380 |
| Timed-interval CPU time, CPU s/request | 20.937 | 13.731 | 13.710 |
| Mean cores occupied during timing | 2.529 | 1.203 | 1.203 |
| Process peak RSS, MiB | 751.27 | 758.95 | 906.97 |

The workload is the fixed mixture of 12 images, including one with three faces. Three-face computation affects P95, which cannot be interpreted as occasional random delays above twenty seconds for a fixed single-face input. The table also does not directly measure the existing production port.

### 6.2 Configuration changes reduce CPU time and increase wait

Relative to `python_original`, `python_n150` produced these changes.

- Timed CPU time decreased from **20.937 to 13.731 CPU s/request, down 34.4%**.
- Mean HTTP time increased from **8.277 to 11.415 s/request, up 37.9%**, with P95 up about 39.1%.
- Peak RSS increased from 751.27 to 758.95 MiB, up about 1.0%.

This N150 preset traded CPU work against latency on these inputs, without reducing both. The old-thread group occupied about 2.53 cores on average and the N150 preset about 1.20. Reduced cumulative CPU time does not imply shorter waits and cannot directly establish whole-machine energy savings.

### 6.3 Language migration did not establish faster execution or lower memory

Against `python_n150` with matching settings, `native_n150` had only about **0.12%** lower mean HTTP time, 0.14% lower P95, and 0.15% lower timed CPU time. The ranges between highest and lowest process mean times were about 0.24% of the Python median and 0.65% of the native median. The 0.12% intergroup difference was smaller than these repeated-process variations and cannot be called a reliable speedup.

Native peak RSS was **906.97 MiB**, about **148.02 MiB or 19.5%** above the matching Python result of **758.95 MiB**. Earlier Mac results therefore cannot establish reduced memory for this N150 native implementation.

Against the old-thread Python group, native mean HTTP time was still about 37.7% higher, peak RSS 20.7% higher, and timed CPU time 34.5% lower. These combined changes include scheduling and implementation differences and cannot all be attributed to Go or C++. The results support passed accuracy and decision regression checks, without establishing comprehensive superiority over the reference.

### 6.4 Environment and limits

Nine processes recorded 27 environmental snapshots before startup, after warmup, and after timed requests. Available memory ranged about 4,439.59–5,311.02 MiB, used swap stayed at 1.5 MiB, and working-directory free space was about 60.67 GiB. Frequency snapshots covered about 700–3600 MHz, with temperature readings from different thermal zones. Discrete samples cannot exclude transient frequency, temperature, or background-load changes within requests or establish the cause of elapsed times by themselves.

The three-process repeats describe this fixed workload and provide no guarantee of long-term stability, concurrent throughput, power use, or arbitrary user-image behavior. Mac and N150 used different HTTP originals, so ratios between their tables cannot establish a hardware-speed ratio.

The independent FTZ follow-up completed and audited its small-sample diagnosis, candidate-core checks, full numerical regression, and HTTP measurements, described in Section 8. It does not change this section's frozen three-way baseline.

## 7. Local review evidence

These repository-relative paths are local evidence in ignored directories. Cloning the repository does not include them.

| Evidence path | Contents |
| --- | --- |
| `data/experiments/n150-remote-20261027/benchmark-protocol.json` | Inputs, three-way protocol, timing, and publication scope fixed beforehand |
| `data/experiments/n150-remote-20261027/source-snapshot.json` | Source snapshot and package hashes identifying the build origin |
| `data/experiments/n150-remote-20261027/hardware.json` | Available CPU range, model, governor, frequency, and temperature snapshots |
| `data/experiments/n150-remote-20261027/cpu-constraints.json` | Process priority, CPU affinity, and cgroup quotas during formal measurement |
| `data/experiments/n150-remote-20261027/native-loader-check.json` | Linux artifact hashes, dynamic dependency resolution, and relative loading checks |
| `data/experiments/n150-remote-20261027/parity-linux.json` | Separate-worker numerical, classification, and acceptance results for all crops and originals |
| `data/experiments/n150-remote-20261027/background-before-http.log` | Background-service resources before formal HTTP testing |
| `data/experiments/n150-remote-20261027/http-three-way.json` | Nine processes, 432 timed requests, pairwise parity, and final performance summary |
| `data/experiments/n150-remote-20261027/baseline-audit.json` | Independent completeness and arithmetic audit, relative changes, and environment ranges |
| `data/experiments/n150-remote-20261027/denormal-probe-01.json` | Small-sample sequential-kernel diagnosis with FTZ/DAZ off and on |
| `data/experiments/n150-remote-20261027/denormal-weight-audit.json` | Subnormal counts in original top-level ONNX FP32 initializers and asset hashes |
| `data/experiments/n150-remote-20261027/ftz-build.json` | Separate candidate build, guard/CTest checks, and unchanged original-build record |
| `data/experiments/n150-remote-20261027/ftz-smoke.json` | Candidate C API startup check on three crops, outside HTTP and full-accuracy evaluation |
| `data/experiments/n150-remote-20261027/ftz-candidate-parity.json` | Full crop, BGR-original, and direct-decoding comparisons between the new core and frozen Python cache |
| `data/experiments/n150-remote-20261027/ftz-http.json` | Three candidate HTTP processes, 144 timed responses, actual core-library binding, and summary |
| `data/experiments/n150-remote-20261027/candidate-audit.json` | Independent audit of candidate-input report hashes, counts, summaries, and relative changes |
| `data/experiments/n150-remote-20261027/ftz-candidate/` | Original experimental patch, floating-point-state guard, and separate tests, integrated later on 2026-10-28 |

Public documentation retains aggregates only and excludes photos, identity lists, per-sample features, raw logs, server connection information, and keys. Model Releases continue to follow project rules and exclude test inputs, application-deployment materials, and private experiment records from attachments.

<a id="ftz-daz-follow-up"></a>

## 8. Independent FTZ / DAZ follow-up experiment

This was a separate experiment after completion of the three-way baseline. Numerical and kernel diagnosis on three crops, an independent candidate-core build, a small C API check, regression on all 1,051 crops and 12 originals, and **HTTP measurements in three fresh processes were all completed**. The final report passed independent audit. These results do not replace Section 6 or mean the experimental configuration had entered production at that time.

### 8.1 Why subnormal numbers warranted investigation

FP32 represents nonzero values very close to zero. Nonzero values below the smallest normal magnitude, `2^-126`, are called subnormal or denormal numbers. They support gradual underflow but may incur substantial computational cost on particular processors and instruction paths.

A read-only count of FP32 initializer tensors in the original ONNX files produced the following results.

| Model | Subnormal values | Initializer tensors containing them |
| --- | ---: | ---: |
| AdaFace IR101 | 202,104 | 24 |
| R50 | 30,184 | 25 |

The count covers all stored float32 elements in original top-level dense `FLOAT` ONNX initializers, excludes positive and negative zero, and retains `0 < abs(x) < 2^-126`. A bit-pattern check gives integers strictly between `0` and `0x00800000` after removing the sign bit. Counts include convolution weights, PReLU, and stored BatchNorm values. They do not count only convolution weights or trainable parameters and do not measure runtime activations or face features. The read-only audit and asset hashes are in `denormal-weight-audit.json`. These values are a diagnostic clue, but their counts alone cannot explain all elapsed time.

Intel's x86 floating-point control register MXCSR provides related modes. **FTZ** flushes subnormal results from applicable floating-point instructions to zero. **DAZ** treats subnormal input operands to those instructions as zero. Both affect SSE/AVX floating-point semantics and performance without unconditionally guaranteeing speed gains. [Intel FTZ / DAZ documentation](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-1/set-the-ftz-and-daz-flags.html)

The experiment kept original FP32 weight files and hashes unchanged, without FP16 or INT8 conversion or retraining. However, **floating-point subnormal-handling semantics changed**, so unchanged files do not guarantee unchanged accuracy. ONNX Runtime 1.29.0 provides `session.set_denormal_as_zero`, disabled by default, with an explicit warning that enabling it can affect model accuracy. [ORT 1.29.0 configuration-key definition](https://github.com/microsoft/onnxruntime/blob/v1.29.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)

### 8.2 Controlled three-crop diagnosis

Three fixed aligned crops were selected. Separate workers set that ORT option to `0` and `1`, executing disabled first and enabled second. Both backbones used one thread, disabled spinning, and sequential execution. Input preprocessing followed the reference. After warmup, three timed rounds yielded nine crop observations per group.

Timing surrounded only each backbone's `Session.run`. The table gives medians of nine observations. The combined-backbone row is the median sum of the two calls for each crop and **does not give total time to process all three crops**.

| Kernel scope | FTZ/DAZ off, s/crop | FTZ/DAZ on, s/crop | Off / on time ratio |
| --- | ---: | ---: | ---: |
| AdaFace | 9.614829 | 0.470143 | 20.45× |
| R50 | 1.857830 | 0.250646 | 7.41× |
| Both backbones sequentially | 11.470675 | 0.720563 | 15.919× |

Across the three crops, raw backbone outputs, fused features, and final probabilities were itemwise identical between modes. Top-1, ordered Top-5, and acceptance at 0.95 also matched. A separate runtime arithmetic probe confirmed that the main calling thread's handling of subnormal inputs and results changed.

This directly supports further investigation of subnormal handling as a performance factor on this machine. Timings exclude detection, large-image decoding, HTTP, and request queues and cover only three samples. Floating-point mode in these diagnostic processes also affected untimed preprocessing and classification, so these results do not isolate all numerical behavior of the final C++ service. Fixed ordering and a single diagnostic run also leave environmental differences unresolved.

### 8.3 Candidate-core management of thread floating-point state

Setting the option only during Session creation is insufficient. ORT 1.29.0 uses one-time initialization on the initial calling thread, while Session-owned thread pools receive the setting separately. Later callers cannot be assumed to inherit the required MXCSR state. [ORT 1.29.0 Session initialization](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/session/inference_session.cc)

Go goroutines do not have a fixed one-to-one relationship with operating-system threads. Thread-local state cannot rely on settings left by an earlier cgo call. [Go thread-binding documentation](https://pkg.go.dev/runtime#LockOSThread) The candidate therefore performs setup, inference, and restoration within the same C++ call.

1. At the start of each ORT Session creation and each `Model::run` scope, save the calling thread's complete MXCSR.
2. Explicitly enable or disable FTZ and DAZ according to experimental configuration, preserving other control bits. Apply the ORT Session option to its own thread-pool policy as well.
3. Restore the complete original state through an RAII guard on both normal return and exception unwinding. The persistent R50 worker uses the same `Model::run` guard.

This avoids retaining floating-point state across cgo calls or leaving this inference mode for subsequent Go calls. Image preprocessing and the head are outside the guard. The candidate does not enable `fast-math` and remains an experiment in floating-point mode within an explicitly bounded execution scope.

The guard's `true` mode currently supports only **Linux x86_64 with MXCSR FTZ/DAZ support**. Unsupported platforms, including the current Mac ARM machine, explicitly reject enabling it. Linux guard checks passed toggling, nesting, exception restoration, real subnormal arithmetic, and **repeated calls from 8 persistent threads**, as did candidate CTest. Guard tests validate tested state-management behavior without replacing full-model accuracy checks in ORT.

During this experiment, the candidate existed only as a patch, separate source copy, and independent shared library in ignored directories. Product `native/` source and original baseline artifacts stayed unchanged. The Go executable reused the unmodified build, with the candidate core and separate JSON enabling the experiment. `runtime.flush_denormals` is a new strictly boolean field in that core. **Adding this key to JSON alone cannot make an old core library run the experimental logic.** Subsequent product integration is in Section 9.

The candidate C API's three-crop check took about **0.492 seconds** per face, without detection or HTTP. Features and probabilities were identical to the native baseline cache. Against the Python cache, features matched and the maximum absolute probability difference was `1.1920928955078125e-7`, with candidate order and acceptance unchanged. This was only a gate to full validation. The candidate ran backbones in parallel, and the C API scope was broader than sequential-kernel diagnosis. Dividing their times cannot establish a roughly 23-fold FTZ-only speedup or an HTTP speedup.

### 8.4 Full regression passed

The candidate completed checks on all 1,051 crops and the same 12 originals. The reference was the previously completed frozen **Python, FTZ-off** cache, while the candidate launched a fresh **C++, FTZ-on** worker. Python reference workers were not rerun and Go HTTP was not tested at this step. Inputs, original model assets, class order, and the actual candidate library were bound through hashes.

| Full-crop check | Candidate result |
| --- | --- |
| Top-1 correct / errors | 1,035 / 16 |
| Top-1 accuracy | 98.47764% |
| New / corrected errors against reference cache | 0 / 0 |
| Top-1, ordered Top-5, and 0.95 decisions | Each 1,051 / 1,051 matched |
| Maximum absolute feature difference | `2.2351741790771484e-8` |
| Maximum absolute classification-probability difference | `4.76837158203125e-7` |
| Accepted / incorrectly accepted / review | 1,011 / 4 / 40, matching reference |

The 12 originals and 14 detected faces were checked through both predecoded BGR and candidate direct-decoding paths. Face counts, candidate order, and acceptance all matched. Maximum box, detection-score, and quality-score differences were 0. Maximum absolute probability differences were `2.384185791015625e-7` on both paths. Image-dimension and class-order checks also passed, with report `all_passed=true`.

This establishes no decision regression on the fixed inputs. Full-set features and probabilities still differed bitwise from Python. Exact parity in the three-crop diagnosis cannot be extended to full-set bitwise equivalence, and regression on historical data cannot provide a new independent accuracy guarantee. The cached-reference and staged-execution definitions must accompany the subsequent HTTP data.

### 8.5 Candidate HTTP completed and audited

The candidate completed three new HTTP service processes using the same 12 originals. Each process warmed up for 12 requests and timed 48, giving **36 warmup and 144 timed requests** in total. Statistics followed the baseline method. Process metrics were calculated separately before taking their median. P50/P95 used nearest rank, and timed CPU and lifetime CPU were kept separate.

| Metric | FTZ candidate, median of three process metrics |
| --- | ---: |
| Mean HTTP time | **680.785 ms/request** |
| HTTP P50 | 591.256 ms/request |
| HTTP P95 | 1607.468 ms/request |
| Mean server processing time | 659.466 ms/request |
| Timed-interval CPU time | **1032.083 CPU ms/request** |
| Mean occupied cores during timing | 1.516 |
| Process peak RSS | **885.836 MiB** |

Process mean HTTP times ranged from 676.934–683.204 ms. Inputs still included one three-face image, so P95 cannot represent the random latency tail for a fixed single-face input.

All **144 timed responses** were compared with frozen end-to-end Python original-image reference output. Image dimensions, face counts, candidate order, and acceptance passed parity checks. Maximum box, detection-score, and quality-score differences were 0, and maximum absolute probability difference was `2.384185791015625e-7`. The reference reused the Python original-image validation cache in input order. It was neither saved HTTP responses nor a Python service newly launched in this phase.

To prevent accidental loading of the old library, each process checked its own PID's core-library mapping after a successful `/v1/model` response and before warmup and CPU timing. File device number, inode, and candidate-library hash matched in **3 / 3 processes**. The final independent audit passed report-hash, request-count, numerical-regression linkage, and summary-arithmetic checks. It checked artifact binding in records without rereading remote binaries and did not require a speed threshold to pass.

The following comparisons use medians from earlier phases.

| Reference group | Earlier mean HTTP / CPU time | Candidate mean HTTP change | Candidate timed CPU change | Candidate peak RSS change |
| --- | --- | ---: | ---: | ---: |
| `python_original`, old threads reproduced in the current environment | 8.277 s / 20.937 CPU s per request | Down 91.775%, earlier time 12.158× candidate | Down 95.070% | Up 17.912%, 751.270 → 885.836 MiB |
| `native_n150`, native baseline | 11.401 s / 13.710 CPU s per request | Down 94.029%, earlier time 16.747× candidate | Down 92.472% | Down 2.330%, 906.969 → 885.836 MiB |

These ratios explicitly **compare a later-phase candidate with earlier baselines, without a contemporaneous interleaved A/B test**. Thermal state, frequency, and background-load drift may affect the exact ratios. Section 8.2 separately provides a controlled three-crop diagnosis changing only subnormal mode for identical sequential kernels, with a time ratio around 15.919×. This independently supports subnormal-handling overhead. The full HTTP change cannot be attributed to calling language, and reduced CPU time cannot directly establish whole-machine energy savings.

Together, full regression and HTTP results show preserved decisions on fixed inputs and substantial measured reductions in latency and CPU time. Peak RSS was still about 17.9% above the old-thread Python group, so lower memory was not achieved at the same time. This provides no guarantee for all images, unknown identities, long-term loads, or other hardware.

This describes the state at experiment completion on 2026-10-27, before any production switchover. Product integration, actual enablement, and deployment checks on 2026-10-28 are recorded separately in Section 9. Stress and long-term acceptance checks remain outstanding.

### 8.6 Post-test host checks

After all experiments, the original service's `/ready` returned 200. The IDs, PIDs, and start times of the existing 10 containers were unchanged, as were device numbers and inodes for 4 protected mounts. Benchmark ports were closed, and no inference workers from this round remained. Original-runtime and candidate-artifact hashes were verified separately, and the main disk still had about **60.59 GiB** free.

Local evidence is `data/experiments/n150-remote-20261027/postflight-final.json`, `completion-check.json`, and `candidate-final-hashes.txt` in the same ignored directory.

## 9. Fast version enabled on 2026-10-28

After the user selected the fastest version with unchanged accuracy on the existing validation set, the original experimental patch was applied unchanged to `native/src/faceid.cc` and `denormal_guard.hpp`, with thread-state tests added to CTest. `tools/export_native_model.py --flush-denormals` generates JSON explicitly enabling FTZ/DAZ and matching checksums. Omitting the option retains the disabled default for compatibility with old configurations and Mac ARM. Source-file hashes matched the original candidate, and the core build explicitly disabled `fast-math`.

Local integration validation passed **207 Python tests, 3 CTest tests, ordinary Go tests, and native-build tests**. Python retained two existing Starlette/httpx/AnyIO deprecation warnings, and the Mac linker reported duplicate rpath. Mac ARM guard tests checked rejection of unsupported enablement. The target machine separately rebuilt and passed real FTZ/DAZ arithmetic, complete MXCSR restoration, exception handling, and 8-persistent-thread tests.

Target deployment directly reused the Linux binaries verified in Section 8. Each copied artifact was checked.

| Artifact | SHA-256 |
| --- | --- |
| C++ core library | `9bdc3f7728c4dc7aa79002c3bb2d4cf0e6cc22aef2ab54d87a5310159849e158` |
| Go service | `cfc2163b46a2143ac0a393e370ae86e7487415149867eec2a7ac78998babeafe` |
| FTZ JSON, bytes preserved after renaming | `cb9c53d87f733f182168f2c3ef1465fa05828f43f8382e3f38698e9baad07190` |

Before deployment, production model ID, original-asset hashes, class order, and defaults `top_k=5` / `probability_threshold=0.95` were confirmed to match the candidate. A temporary instance first ran on a temporary port with the same systemd security configuration. After validation it stopped, and the native service took over the original `8000` port. Keys remained unchanged. The systemd service was enabled at boot, but startup was not checked through an actual host reboot.

Temporary and production instances each completed HTTP retests on **12 originals / 14 faces**. Boxes, detection scores, quality scores, candidate order, and acceptance matched the frozen Python reference for both, with maximum absolute probability difference `2.384185791015625e-7`. Actual core-library paths, device numbers, and inodes in `/proc/<pid>/maps` matched deployment files. Authentication-rejection and readiness checks passed.

The production instance averaged about **655.36 ms** across these 12 requests, with peak process RSS about **806.81 MiB** at completion. This deployment check includes the first request and differs from Section 8.5's three-process protocol with warmup and 48 timed requests. It cannot replace the formal benchmark or establish further speed or memory gains. All 1,051 crops were not rerun at this stage. The exact core library, Go binary, configuration bytes, and weights from Section 8.4 were reused.

The service is `seiyuu-faceid.service`, deployed on the main disk at `/opt/seiyuu-faceid/releases/20261028-ftz/`. The old container was stopped and automatic restart disabled, while its container, environment variables, and original model mounts were retained. The other 9 containers kept their IDs, PIDs, and start times. Main-disk free space was about **60.11 GiB**. Deployment and rollback instructions are in the [C++ / Go documentation](NATIVE_CPP_GO.md).

The deployment summary is saved in local ignored `data/experiments/n150-deploy-20261028/deployment.json`. Remote transaction records and rollback scripts remain only in a private maintenance directory on the target machine. Keys and private responses are not published. Stress, long-term operation, and broader business-image coverage still require separate acceptance checks.
