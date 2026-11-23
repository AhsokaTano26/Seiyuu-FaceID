# モデルバンドル

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

既定のパスでは次のバンドル構成を使用します。

- `det_500m.onnx` は InsightFace buffalo_sc v0.7 の SCRFD 検出器です。[配布元](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip)
- `selected/adaface_ir101.onnx` は固定 AdaFace IR101/WebFace12M で、公式チェックポイントとアーキテクチャから変換しています。BGR、NCHW、112×112を入力し、`(pixel/255 - 0.5)/0.5` で正規化して512次元単位特徴量を出力します。[公式重み](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view)、[アーキテクチャ](https://github.com/mk-minchul/AdaFace)
- `selected/w600k_r50.onnx` は InsightFace buffalo_m v0.7 の固定 R50 です。RGB、NCHW、112×112を入力し、`pixel/127.5 - 1` で正規化して512次元出力をさらに正規化します。[配布元](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip)
- `selected/classifier.npz` にローカルで学習した標準化パラメータ、ロジスティック回帰係数、順序付き人物ラベルを保存します。AdaFace、R50 の単位特徴量を順に連結して√2で割ります。実行時は sklearn の float32 標準化と安定化した多項 softmax を再現します。
- `selected/manifest.json` に出典URLとSHA-256を記録し、モデル読込時に検証します。
- `selected/evaluation.json`、`holdout_scores.npz`、`runtime_verification.json` に過去の選定モデルの評価記録を保存します。

過去のバンドルは約416 MiBです。重みは Git の管理対象外で別途配布します。環境を移す際はバンドル内の全ファイルを保持してください。実行時には NumPy と ONNX Runtime を使用します。

AdaFace の上流 MIT 表記を `selected/ADAFACE_LICENSE` に保持します。InsightFace のコードは MIT で、公開事前学習済み重みには非商用研究向け条件があります。プロジェクトと上流モデルの両方の利用条件に従ってください。

ローカルの既定バンドルと `release/model-source.json` が固定するモデルは50クラスです。別途公開した `models-5cc19e9d5b1a30aa` は606クラスです。版を選ぶ際は分類ヘッドのSHA-256とmanifestを確認してください。[606クラス報告](../docs/MODEL_ACCURACY_20261103.ja.md)に詳細があります。
