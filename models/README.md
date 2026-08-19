# Model inventory

Weights are deliberately excluded from this repository. Do not add a weight
until its exact download URL, checksum, source, version, and license have been
recorded below.

The default config expects:

| Role | Filename | Upstream source | License / restriction |
| --- | --- | --- | --- |
| detector | `det_500m.onnx` | InsightFace [`buffalo_sc` v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip), SCRFD-500MF | Upstream pretrained-model policy: non-commercial research only. |
| recognizer | `w600k_mbf.onnx` | InsightFace [`buffalo_sc` v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip), MobileFaceNet | Upstream pretrained-model policy: non-commercial research only. |

The installed `buffalo_sc.zip` archive SHA-256 is
`57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72`.

InsightFace code is MIT, but its published pretrained models are not: the
upstream README states that manually downloaded and library-downloaded weights
are for non-commercial research. This project is MIT only for its own code and
does not grant any weight rights. For a different model, add a row with its
exact terms before putting it in `models/`.
