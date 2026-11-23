# C++ inference + Go HTTP

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Experimental alternative runtime for the existing FP32 models. Go calls a C++
library in the same process via cgo. Model inference, detection, alignment and
classification run in C++. HTTP, upload validation and authentication run in Go.
Python is used for model preparation and comparison only, not while serving.

See [build, validation and deployment notes](../docs/NATIVE_CPP_GO.md).
For the full recognition pipeline, training and runtime rationale, see the
[technical architecture guide](../docs/TECHNICAL_ARCHITECTURE.md).
The existing Python runtime remains the reference implementation.

Linux builds and startup checks have passed on the Intel N150 target, along with
parity checks on 1,051 historical holdout crops and 12 original images, and the
complete three-way HTTP baseline. The native baseline showed no reliable speed
gain over Python with matching settings and used more peak RSS. The CPU-saving
profile increased mean HTTP latency by about 37.9% versus the original-thread
Python group. See the [N150 report](../docs/N150_BENCHMARK.md) for scope, results
and separate follow-up experiments. The validated FTZ/DAZ version replaced the
N150 API service on 2026-10-28. Long-run and stress validation remain outstanding.

A separate FTZ/DAZ candidate has since passed the full regression and three fresh
HTTP runs, with a median process mean of 680.785 ms per request and unchanged
98.48% historical holdout Top-1 accuracy. This is a later-phase experiment, not
contemporaneous alternating A/B. The implementation is now included in this
source tree. Export with `--flush-denormals` to enable it on supported Linux
x86_64. The default stays off for compatibility and Mac ARM rejects enabling it.
See the [FTZ experiment](../docs/N150_BENCHMARK.md) for
the implementation, memory tradeoff and validation boundaries.

The [systemd template](deploy/seiyuu-faceid.service) runs a single native process.
the [deployment guide](../docs/NATIVE_CPP_GO.md#linux-systemd) covers layout,
credentials, startup and rollback. It serves API endpoints without the Python
Swagger or upload UI routes.
