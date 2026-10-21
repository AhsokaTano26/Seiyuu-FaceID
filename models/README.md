# Selected model

The app uses only this bundle:

- `det_500m.onnx`: SCRFD detector from [InsightFace buffalo_sc v0.7](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip).
- `selected/adaface_ir101.onnx`: frozen AdaFace IR101/WebFace12M backbone, converted from the [official checkpoint](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view) and [architecture](https://github.com/mk-minchul/AdaFace). Input: BGR, NCHW, 112×112, `(pixel/255 - 0.5)/0.5`; output: 512-D unit embedding.
- `selected/w600k_r50.onnx`: frozen R50 from [InsightFace buffalo_m v0.7](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip). Input: RGB, NCHW, 112×112, `pixel/127.5 - 1`; normalize its 512-D output.
- `selected/classifier.npz`: locally fitted scaler, logistic regression coefficients and ordered identity labels. Concatenate AdaFace then R50 embeddings and divide by √2. Runtime reproduces sklearn's float32 scaling and stable multinomial softmax.
- `selected/manifest.json`: source URLs and SHA-256 checksums, verified during model loading.
- `selected/evaluation.json`, `holdout_scores.npz`, `runtime_verification.json`: evaluation evidence for the selected model.

The bundle is approximately 416 MiB. It is local-only and excluded from Git. Preserve all bundle files when moving the installation. PyTorch and sklearn are not runtime dependencies.

AdaFace's upstream MIT notice is retained as `selected/ADAFACE_LICENSE`. InsightFace code is MIT; its published pretrained weights are restricted to non-commercial research under the upstream model policy. The project's own MIT license does not change upstream weight terms.
