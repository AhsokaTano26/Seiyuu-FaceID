# ローカルデータ構成

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

- `raw/<identity>/`：ユーザーが用意した参照画像です。手動または Web UI で追加できますが、既存の元画像はツールで変更しません。
- `aligned/`：採用された `112×112` の顔クロップと `metadata.jsonl` です。
- `rejected/`：除外された画像・顔の理由を記録する JSONL です。
- `gallery/<identity>/`：独立評価専用の、ラベル付きユーザー画像です。ビルドツールはこのディレクトリに書き込みません。
- `runtime/`：本番用 embedding gallery、検出キャッシュ、評価レポートなど、再生成可能な出力です。
- `query/`：任意の一括認識入力ディレクトリです。ここに置いた画像は照会専用で、Gallery に追加されません。
- `cache/`：従来の破棄可能なキャッシュ場所です。新しい出力は `runtime/` を使用します。
- `review/raw_faces/`：集合写真の手動レビュー用の番号付きプレビューと選択マニフェストです。

これらのディレクトリは `.gitkeep` を除き、すべて Git の管理対象外です。
