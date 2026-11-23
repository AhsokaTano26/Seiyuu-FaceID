# Documentation

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

This index covers the Markdown documentation maintained in this repository. Use the language links above to switch between English, Simplified Chinese and Japanese. English uses `.md`, Simplified Chinese uses `.zh-CN.md`, and Japanese uses the standard language code `.ja.md`.

The default local model and pinned CI source contain 50 identities. The separately published 606-identity model has its own report and dataset. Match the model checksum and test protocol when reading metrics.

Third-party dependency trees, generated deployment directories and private experiment artifacts keep the notices and provenance associated with their own files. Historical plans record decisions at their stated dates. The guides below describe the maintained interfaces and deployment workflow.

## Start and use

- [Project overview](../README.md)
- [HTTP API](API.md)
- [Recognition architecture](TECHNICAL_ARCHITECTURE.md)
- [Native runtime entry point](../native/README.md)
- [Native runtime design](../native/DESIGN.md)

## Deployment and releases

- [CI/CD, deployment artifacts and model publication triggers](CI_CD.md)
- [Docker and offline deployment](DEPLOY_LINUX.md)
- [Python/systemd deployment](DEPLOY_NATIVE_LINUX.md)
- [C++/Go deployment and validation](NATIVE_CPP_GO.md)
- [N150 configuration and measurement](N150.md)
- [Local deployment bundle](../deploy/README.md)
- [CI Docker bundle instructions](../deploy/ci/docker.md)
- [CI Python bundle instructions](../deploy/ci/python.md)
- [CI native bundle instructions](../deploy/ci/native.md)
- [Model release inputs](../release/README.md)

## Data and training

- [Local data layout](../data/README.md)
- [Model sources and bundle versions](../models/README.md)
- [Classifier reproduction and search](../training/README.md)

## Evaluation and performance

- [606-identity model evaluation](MODEL_ACCURACY_20261103.md)
- [Historical 50-identity head comparisons](CLASSIFIER_OPTIMIZATION.md)
- [N150 baseline and FTZ/DAZ measurements](N150_BENCHMARK.md)
- [LFW/WIDER FACE runtime comparison](STANDARD_DATASET_BENCHMARK.md)
- [LFW/WIDER FACE accuracy protocols](STANDARD_DATASET_ACCURACY.md)
- [Figure sources and reproduction](figures/README.md)

## Historical implementation plans

- [HTTP API implementation, 2026-10-21](superpowers/plans/2026-10-21-http-api.md)
- [Sample-count study, 2026-10-21](superpowers/plans/2026-10-21-sample-size.md)
- [Classifier optimization, 2026-10-27](superpowers/plans/2026-10-27-classifier-optimization.md)
- [N150 runtime plan, 2026-10-27](superpowers/plans/2026-10-27-n150-runtime.md)
- [CI delivery plan, 2026-10-28](superpowers/plans/2026-10-28-ci-deployments.md)

The standalone [face-detection OpenAPI definition](detect.openapi.yaml) can be imported into API clients. Project conventions are in [AGENTS.md](../AGENTS.md).
