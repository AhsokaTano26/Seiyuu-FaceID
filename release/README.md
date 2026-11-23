# Model release inputs

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

`model-source.json` pins a downloadable model-only archive, its SHA-256, and the digest and size of each allowed member. `training.json` is a public aggregate exported after verifying the frozen training/validation rows actually used to fit that head. It is bound to the classifier SHA-256 and built-in label order.

CI generates model-only Releases from these inputs. Application deployment archives go to Actions Artifacts and container images to Docker Hub. The source currently pins the historical v0.1.0 model archive, and both archive and member digests have been verified.

Publishing a content-addressed `models-<content-digest>` Release triggers deployment CI for those weights and the current main application. Human publication uses `release.published`; CI publication explicitly sends `repository_dispatch` with `tools/publish_models.py --dispatch`. Each resulting deployment bundle pins that Release and its model digest. Model-triggered builds finish after deployment delivery, preventing recursive model publication. Existing versions can be built again through the workflow's `model_tag` input.

The public metadata contains aggregate identity counts and provenance. Test rows are excluded from training counts. Photos, sample-level labels, raw features, evaluation data and API keys are excluded from model Releases.

See the [CI/CD guide](../docs/CI_CD.md) for downloads and updates, and [project conventions](../AGENTS.md) for the complete Release rules.
