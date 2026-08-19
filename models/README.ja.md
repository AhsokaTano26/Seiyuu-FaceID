# モデル一覧

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

重みは意図的にこのリポジトリへ含めていません。重みを追加する前に、正確なダウンロード URL、チェックサム、提供元、版、ライセンスを以下に記録してください。

標準設定で必要なファイルは次のとおりです。

| 役割 | ファイル名 | 提供元 | ライセンス／制限 |
| --- | --- | --- | --- |
| 検出器 | `det_500m.onnx` | InsightFace [`buffalo_sc` v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip)、SCRFD-500MF | 上流の事前学習モデル方針：非商用研究に限定。 |
| 認識器 | `w600k_mbf.onnx` | InsightFace [`buffalo_sc` v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip)、MobileFaceNet | 上流の事前学習モデル方針：非商用研究に限定。 |

ローカルに配置した `buffalo_sc.zip` の SHA-256 は次の値です。
`57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72`

InsightFace のコードは MIT ですが、公開済みの事前学習モデルは MIT ではありません。上流 README は、手動・ライブラリ経由を問わずダウンロードした重みを非商用研究に限定しています。本プロジェクトの MIT は自作コードにのみ適用され、重みに関する権利は一切付与しません。別のモデルを使う場合は、`models/` に置く前に正確な利用条件を表へ追加してください。
