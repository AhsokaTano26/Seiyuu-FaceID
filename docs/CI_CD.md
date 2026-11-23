# CI/CD with three deployment options and model releases

[English](CI_CD.md) | [简体中文](CI_CD.zh-CN.md) | [日本語](CI_CD.ja.md)

See the [ci.yml workflow](../.github/workflows/ci.yml).
A push to main, a v*.*.* version tag, or a manual run on main checks the code and then delivers the following artifacts.
Pull requests only build and validate. They do not publish images, deployment attachments, or models.
Publishing a `models-<content-digest>` Release also starts deployment CI for that model and the current main application revision.

| Delivery | Download location | Contents and platforms |
| --- | --- | --- |
| Docker Compose | Actions artifact `seiyuu-faceid-docker-<commit-sha>`, with the commit SHA. Images are on Docker Hub | Compose, standard/N150 configuration, fixed image tag, model downloader. Linux amd64/arm64 |
| Native Python | Actions artifact `seiyuu-faceid-python-<commit-sha>`, with the commit SHA | Python source, uv.lock, configuration, systemd template, model downloader. CI checks Linux amd64 |
| Native C++ / Go | Actions artifact `seiyuu-faceid-native-linux-amd64-<commit-sha>`, with the commit SHA | Linux amd64 executables, shared libraries, licenses, FTZ and compatible JSON exports, systemd template, model downloader |
| Models | GitHub Releases version `models-<content-digest>`, using a content digest | ONNX weights, classification head, and required metadata, independent of the application version |

All three deployment bundles include an archive SHA-256 and an internal SHA256SUMS file.
Download from Artifacts at the bottom of the corresponding Actions run page, unpack the outer download first, then follow the bundled README.md.
Deployment archives are retained for 14 days. Download and retain them yourself for long-term storage, or reproduce the build from the repository.
See [GitHub's artifact documentation](https://docs.github.com/en/actions/tutorials/store-and-share-data) for artifact usage.

Deployment bundles do not duplicate the approximately 400 MB ONNX models.
Each includes release/model-source.json, pinning the download URL, archive digest, allowed members, and each file's digest and size.
The downloader rejects extra files, path traversal, links, corrupt models, and overwriting different existing weights.
Initial download uses the Python 3.11+ standard library. Alternatively, prepare models/ on another machine and copy it over.
This is an installation preparation step. The C++ / Go runtime does not require Python.
For offline installation, use the following command.

~~~bash
python3 tools/fetch_models.py --archive /路径/模型归档.tar.gz
~~~

Normal code builds use the checked-in model source, currently bootstrapped from the existing v0.1.0 release.
Model-release builds pin the newly published Release instead. All jobs share verified `model-inputs` containing the source lock and training aggregate.
Historical deployment attachments from that release are not downloaded.
All deployment methods use the same FP32 weights, classification head, and recognition thresholds.
Follow the accompanying upstream terms when using the model files.

<a id="检查与发布顺序"></a>

## Validation and publication order

1. prepare selects the application commit and model inputs. For model Releases it verifies the asset allowlist, GitHub digests, SHA256SUMS, archive content tag, safe members, matching inner/outer metadata, classifier labels, and actual training counts before distributing the source lock.
2. quality checks locked dependencies, Ruff, all Python tests, compilation, and actionlint.
3. deployments retrieves and verifies models and training statistics, then generates Python and Compose bundles. After extracting into a fresh directory, it starts the Python HTTP service and checks readiness, authentication, model identifiers, detection, and recognition. Both Compose configurations are validated.
4. native builds C++ / Go on Ubuntu 22.04 and runs CTest, ordinary Go tests, and native Go tests. It exports compatible and FTZ parameters and compares Python/C++ features, probabilities, and decisions on fixed random face crops. After packaging, it starts the native HTTP service in a fresh directory and checks its endpoints.
5. docker builds and publishes Linux amd64/arm64 images for the Python API after prerequisite checks pass.
6. models runs only for application builds. After all preceding jobs succeed, it generates or verifies a model-only Release and explicitly requests model deployment CI. Model-triggered builds finish with deployment delivery, preventing a publication loop.

Random-crop comparisons check implementation regressions. They do not evaluate recognition accuracy.
For accuracy on real data, see the [standard dataset accuracy report](STANDARD_DATASET_ACCURACY.md).
The native runtime uses glibc and requires shared libraries. Ubuntu 22.04 builds provide a glibc 2.35 baseline.
Check dynamic library compatibility on other systems.
The default native-model.json enables Linux x86_64 FTZ/DAZ, while native-compatible.json disables it.
Preserve the relative model directory layout and each JSON file's .sha256 file.

Docker image tags include latest for main, a sha- tag with the full commit, and the semantic version corresponding to a version tag.
Deployment bundles default to the current commit's sha- tag.
Model builds also publish a Docker `models-<content-digest>` tag and append that model tag to Actions artifact names. Their internal archive names retain the application commit. `BUILD.json` and `release/model-source.json` bind the exact model archive digest.
Only ordinary main builds update `latest`. Models are mounted from the separately downloaded, verified model directory.
Existing Docker images run the Python API without the native FTZ path.
Native C++ / Go bundles provide the API only by default. The upload homepage and /docs belong to the Python option.

<a id="模型-release-的边界"></a>

## Model Release scope

The release rules in AGENTS.md remain in effect. Attachments are limited to the following files.

- seiyuu-faceid-models.tar.gz contains models and required metadata from models/.
- SHA256SUMS provides checksums for that archive and accompanying attachments.
- TRAINING_SAMPLES.csv lists every model-embedded name and actual fitting sample count.
- TRAINING_PROVENANCE.json records the classification head digest, frozen feature/split file digests, and aggregate statistics.
- MODEL_LICENSES.md describes model sources and upstream usage terms.

Do not upload application code, deployment archives, Compose, API documentation, photos, raw features, or evaluation data.
Release notes focus on model downloads, scope, and the complete identity/sample-count table.
The checked-in source statistics are 50 identities and 4,064 crops actually used for fitting.
These comprise 3,574 crops from the original training set plus 490 from the original validation set, excluding 1,051 test crops.
Statistics are exported from the frozen split used by the current local model and checked against the original release CSV, classifier label order, and hash bindings.
CI accesses only public aggregate statistics and does not download training features.

Model archives use fixed timestamps, file order, permissions, and ownership metadata.
The first 16 hexadecimal characters of the archive SHA-256 form the model version.
An identical archive reuses its existing release. Changes to application code alone do not upload weights again.
A new release starts as a draft, uploads only allowlisted files, and becomes public only after each remote attachment's size and GitHub digest are verified.
A failed run leaves a draft that can be rerun to fill missing attachments.
Zero-byte starter placeholders from interrupted uploads are removed and reuploaded only when the release is a draft and the filename is allowlisted.
Existing attachments with different content and incomplete public releases cause errors. They are neither overwritten nor silently repaired.

<a id="凭据与后续更新"></a>

## Credentials and subsequent updates

The repository needs a DOCKERHUB_TOKEN Secret for publishing tano26/seiyuu-faceid images.
Model publication uses Actions' built-in GITHUB_TOKEN. Only the models job declares `contents: write` and `actions: read` to publish, dispatch, and find existing validation runs.
Other jobs are read-only, and no separately stored long-lived GitHub key is required.
The pipeline does not SSH into servers or automatically replace running services.

For application updates, push to main.
Before updating weights, confirm actual training provenance locally and export the public summary.

~~~bash
uv run python tools/export_training_metadata.py --output /tmp/training-new.json
~~~

This command reads current frozen data and the classification head and exports only aggregates.
It rejects overlapping or duplicated splits, class mismatches, and incorrect totals.
After reviewing the new statistics, update release/training.json.
Provide CI with a prepared new model-only archive, then update the pinned GitHub download URL, archive and member digests, and sizes in release/model-source.json.
The source model archive must already be downloadable. CI does not access ignored directories on the development computer or retrain models.
For initial bootstrapping or a source change, manually publish a model-only source version first, then let the pipeline generate a verified content-addressed version.

### Model publication triggers

Human publication is handled by `release.published` for `models-<content-digest>` tags. The Release must already contain all five verified assets before publication.
CI publication uses `tools/publish_models.py --dispatch` to send `repository_dispatch` with type `model-published` and `client_payload.tag`. GitHub suppresses ordinary workflow events created with GITHUB_TOKEN, while explicit dispatch events can start workflows. See [GitHub's token rules](https://docs.github.com/en/actions/concepts/security/github_token).
The publisher checks the most recent 100 CI runs and reuses an active or successful `Model <tag>` run from trusted model events in this repository. PR and fork runs are excluded. Failed runs and missing dispatches can be retried by rerunning the publisher, including after the Release became public.
Model-event runs validate the selected Release in isolation and never fall back to the checked-in model if it is invalid. All deployment packages in that run use the selected model. The checked-in source remains the default for subsequent ordinary code builds until it is updated explicitly.

To build deployments for an existing model Release or retry a failed model run, select **Run workflow** on main and provide `model_tag`, or use the CLI below after replacing the example tag.

~~~bash
gh workflow run ci.yml --ref main -f model_tag=models-0123456789abcdef
~~~

The workflow and scripts must first be present on main. For human publication, the model tag must also point to a commit containing the `release.published` listener because GitHub loads that event's workflow from the tag commit. Existing tags that predate this listener use the manual `model_tag` command above or explicit repository dispatch to main. See [GitHub's workflow trigger rules](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows).
The resulting run is named `Model <tag>`. Model publication success and deployment CI success are separate states, so inspect that run before deploying.

Local validation is completed during configuration. Subsequent commit, push, and cloud workflow verification follow user authorization.
The Actions run for the corresponding commit is authoritative for complete cloud builds, Docker Hub pushes, and Release uploads.
