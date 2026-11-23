# Implementation plan for three deployments and model delivery

[English](2026-10-28-ci-deployments.md) | [简体中文](2026-10-28-ci-deployments.zh-CN.md) | [日本語](2026-10-28-ci-deployments.ja.md)

This record describes the 2026-10-28 implementation.

The goal is to deliver Docker, Python native and C++/Go native deployments after main pushes, version-tag pushes and manual main runs, then publish validated models.

The design reuses Docker Hub multi-architecture images. Native and Compose archives are stored in Actions Artifacts. GitHub Releases contain models and required metadata. CI reads a version-controlled model URL and SHA-256 and uses public artifacts for installation.

At the planned stage, push, live deployment, historic Release overwrite and changes to models, thresholds or inference were outside the authorized work. Model notes list all built-in identities and actual train + validation counts, excluding test. Deployment archives fetch the pinned model separately.

- [x] Model delivery began with failing tests, followed by safe download, allowlisted extraction, checksums, training-count export, reproducible packaging and release-note generation. Frozen local data verified 50 classes and 4,064 fit rows.
- [x] Deployment artifacts include separate Python, Docker and Linux amd64 C++/Go archives, checksums, pinned sources and instructions. Native artifacts retain shared libraries, licenses and FTZ/compatible JSON.
- [x] Workflow preserves Python quality and Docker Hub jobs, adding native Linux builds, startup/request checks and artifact uploads. Publish model-only Releases after all deliveries succeed. Content-derived tags permit verified reuse. Draft uploads prevent partial public Releases and support recovery of interrupted placeholder uploads.
- [x] Verification covers Python tests, native/Go builds, workflow lint, model fetch from a clean directory, Python/Compose packaging and isolated ELF/JSON fixtures. Documentation states platform support, downloads and first cloud-run boundaries.

The design follows AGENTS.md and fetches approximately 400 MB of weights separately from each deployment archive. Weight changes update pinned sources and verified counts. Application-only pushes reuse a matching model Release.

Local verification recorded 235 Python passes. A clean tree without private models/data had 232 passes and 3 skips. Actionlint, Ruff and both Compose profiles passed. The model archive was downloaded under an allowlist, repackaged and compared file-by-file. A relocated Python package served its homepage, authentication, model metadata and inference correctly. Three CTest cases and both Go suites passed. Mac reinstall logged existing rpath rewrite diagnostics and exited 0. Inference/build code was retained. Independent review found and fixed homepage resources, draft lookup, interrupted upload placeholders and system-library wording, with regression checks.

Local Docker daemon was stopped, so new-container runtime validation was unavailable. Changes initially stayed local under the user's no-push instruction. Later authorization allowed commit, push and cloud verification. Results are recorded in the relevant Actions runs.

Before commit, trailing SVG whitespace, excess final blank lines and formula escape warnings were cleaned. Nineteen figures were regenerated through the same functions and source digests updated. Python compile verification exited 0.
