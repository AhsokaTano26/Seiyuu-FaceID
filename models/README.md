# Model bundles

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The default paths use the following bundle layout.

- `det_500m.onnx` is the SCRFD detector from [InsightFace buffalo_sc v0.7](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip).
- `selected/adaface_ir101.onnx` is the frozen AdaFace IR101/WebFace12M backbone, converted from the [official checkpoint](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view) and [architecture](https://github.com/mk-minchul/AdaFace). It takes BGR NCHW 112×112 input normalized as `(pixel/255 - 0.5)/0.5` and returns a 512-D unit embedding.
- `selected/w600k_r50.onnx` is the frozen R50 from [InsightFace buffalo_m v0.7](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip). It takes RGB NCHW 112×112 input normalized as `pixel/127.5 - 1`. Normalize its 512-D output.
- `selected/classifier.npz` contains locally fitted scaler, logistic regression coefficients and ordered identity labels. Concatenate AdaFace then R50 embeddings and divide by √2. Runtime reproduces sklearn's float32 scaling and stable multinomial softmax.
- `selected/manifest.json` records source URLs and SHA-256 checksums, verified during model loading.
- `selected/evaluation.json`, `holdout_scores.npz` and `runtime_verification.json` hold evaluation evidence for the selected model.

The bundle is approximately 416 MiB. Weights are excluded from Git and distributed separately. Preserve all bundle files when moving the installation. PyTorch and sklearn are not runtime dependencies.

AdaFace's upstream MIT notice is retained as `selected/ADAFACE_LICENSE`. InsightFace code is MIT-licensed. Its published pretrained weights are restricted to non-commercial research under the upstream model policy. The project's own MIT license does not change upstream weight terms.

The default local bundle and the source pinned in `release/model-source.json` contain 50 identities. The separately published `models-5cc19e9d5b1a30aa` contains 606 identities. Check the classifier SHA-256 and manifest when choosing a version. See the [606-identity report](../docs/MODEL_ACCURACY_20261103.md).
