# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

個人で整理した声優写真コレクション向けの、ローカルファーストな顔の本人識別ツールです。Runtime は顔を正規化 embedding に変換し、ローカル Gallery を検索して、`recognized`（認識済み）、`review`（要確認）、`unknown`（不明）を返します。固定クラスの Softmax 分類器ではありません。

> **状態：**V1 のローカル ONNX パイプラインです。ローカル baseline と閾値校正レポートはオンデマンドで生成できますが、結果は対象データセットにのみ有効です。データやモデルを変更したら、本人判定に使う前に閾値を再校正してください。

## アーキテクチャ

```text
画像 → SCRFD 検出器 → 5 点アラインメント → ArcFace ONNX embedding
    → L2 正規化 → ローカル Gallery → identity scoring → 閾値判定
```

学習とデプロイは明確に分離されています。`training/` には PyTorch/MPS ツールを追加できますが、`src/seiyuu_faceid/runtime` と通常の CLI は NumPy、OpenCV、ONNX Runtime、YAML 解析のみを必要とします。

## macOS のセットアップ

Python 3.12 以上（3.12 推奨）が必要です。仮想環境を作成して Runtime 依存をインストールします。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements/runtime.txt
.venv/bin/python -m pip install -e .
```

互換性のある ONNX ファイルを `configs/default.yaml` の指定先へ置きます。

```text
models/det_500m.onnx
models/w600k_mbf.onnx
```

確認済みのローカルモデル組み合わせは InsightFace `buffalo_sc` です。5 ランドマーク対応 SCRFD-500MF 検出器と、`112×112` の MobileFaceNet 認識モデルで構成されます。モデル系列を変更する前にメタデータを確認してください。入力前処理はモデルごとに異なります。

## データセット構成

元ファイルは変更しません。ディレクトリ名を identity key として扱います。

```text
data/raw/hanazawa_kana/001.jpg
data/raw/minase_inori/001.jpg
```

処理済みの顔、除外記録、メタデータ、生成された Gallery はすべて `data/` 以下に書き込みます。`raw` の identity ディレクトリは仮ラベルにすぎません。後から確認できるよう、検出した各顔に対してソース情報と検出器メタデータを保持します。複数人画像は、その identity に属する顔を明示的にすべて選択した場合のみ Gallery に登録されます。

## コマンド

```bash
# 元画像を一度だけスキャンします。破棄可能な検出キャッシュのみを作成します。
python tools/scan_raw_faces.py

# アラインメント、品質フィルタ、embedding 抽出、Gallery 再構築を実行します。
python tools/build_dataset.py
python tools/recognize.py path/to/image.jpg
python tools/batch_recognize.py path/to/photos
python tools/benchmark.py path/to/image.jpg
```

### 登録用の集合写真レビュー

`data/raw/` に集合写真がある場合は、まず一度スキャンします。

```bash
.venv/bin/python tools/scan_raw_faces.py
.venv/bin/python tools/review_raw_faces.py
```

番号付きプレビューと `data/review/raw_faces/face_selections.jsonl` が作成されます。各レコードの `preview_image` を開き、`"face_indexes": null` をゼロ始まりのリスト（例：`[0, 2]`）に置き換えます。その identity に属する顔がなければ `[]` を使用します。その後、通常どおりビルドします。

```bash
.venv/bin/python tools/build_dataset.py
```

単独人物の画像は自動登録されます。有効な選択がない複数人画像は `data/rejected/metadata.jsonl` に記録され、Gallery には入りません。スキャンは `data/runtime/detections.jsonl` を作成し、レビューとビルドで再利用します。元データセットが変わるとキャッシュは自動的に古いものとして扱われます。

すべてのコマンドは `--config configs/default.yaml` を受け付けます。通常の一段階ビルドは `build_dataset.py` です。スキャンキャッシュを再利用（なければ作成）し、アラインメント情報を書き込み、直ちに Gallery を再構築します。`build_dataset.py --scan-only` はキャッシュだけを作成します。既存のアラインメント顔から embedding だけを再生成する場合は `build_gallery.py` を使います。`--skip-gallery` ではデータセット段階だけを確認できます。確認済みの標準閾値は、現在の独立した 6 identity holdout に基づくものです。データやモデルに大きな変更があれば再校正してください。

### 独立評価と閾値校正

ラベル付き評価画像は、登録画像とは別に保存します。

```text
data/gallery/<identity>/...
```

すべてのツールは `data/gallery/` を読み取り専用として扱います。本番 Gallery は `data/raw/` から作成し、リークなしで評価します。

```bash
.venv/bin/python tools/evaluate_external.py
```

このコマンドは embedding、画像ごとの除外理由、レポートを `data/runtime/evaluation/` に書き込みます。未レビューの複数人画像は指標から除外されます。レポートには leave-one-identity-out による未知人物の代理評価も含まれます。閾値を最終値と見なす前に、実際の未知人物画像を追加してください。

評価後、SVG 図表と CSV 曲線を含む独立したローカルレポートを生成できます。

```bash
.venv/bin/python tools/generate_evaluation_report.py
```

標準の出力先は `data/runtime/reports/evaluation_post_review/` です。

## ローカル Web インターフェース

任意の Streamlit UI には、ローカル専用の 3 つの作業領域があります。

- **Gallery 管理**：`data/raw/` の identity ディレクトリを一覧し、画像の確認、identity ディレクトリ作成、選択したディレクトリへの参照画像アップロードができます。Gallery が存在する場合、選択した元画像で検出された各顔は自動照合され、結果が表示されます。画像はページネーション付きサムネイルグリッドで表示されます。
- **ビルドとレビュー**：独立ページです。まず**全元画像をスキャン**し、続けて**集合写真を順にレビュー**を実行すると、複数の顔が検出された写真だけをレビューできます。数字キーで該当する顔を切り替え、`A` で全選択、`Y` で信頼できる候補を採用、`Enter` で確定して次へ進みます。ビルドはスキャン結果を再利用し、最後の集合写真をレビューすると開始されます。単独人物の画像は自動的に採用されます。
- **画像を認識**：`raw` へ保存せずに、1 枚または複数の照会画像を認識できます。大規模な一括処理は `data/query/`（または別のローカルディレクトリ）に置き、**フォルダ全体を認識**をクリックします。対象ディレクトリとすべてのサブディレクトリ内の対応画像が処理され、結果はページネーション付きサムネイルグリッドで表示されます。一時アップロードは小規模な確認向けです。検出枠は灰色、選択済み／要レビューは紫または黄色、既知 identity は緑、不明は赤です。結果は JSONL でダウンロードできます。

```bash
.venv/bin/python -m pip install -r requirements/web.txt
.venv/bin/streamlit run app.py
```

Streamlit が表示するローカル URL（通常は `http://localhost:8501`）を開きます。設定済み ONNX ファイルがあれば、UI 自体で最初の Gallery を作成できます。付属の Streamlit 設定は `127.0.0.1` のみを待ち受け、Streamlit の利用統計を無効化します。

## 学習

最初の Runtime マイルストーンには意図的にファインチューニングを含めません。今後の学習では、事前学習済み MobileFaceNet 系 backbone と ArcFace head を使用し、embedding backbone のみを ONNX としてエクスポートし、PyTorch と ONNX の embedding 一致を検証します。学習依存は `requirements/training.txt` にのみ含めます。

## Benchmark と将来の N150 デプロイ

`tools/benchmark.py` は、検出・認識・エンドツーエンド遅延、RSS、スループットを含む JSON を出力します。まず Mac の baseline を記録し、その後 Intel N150 で同じコマンドを実行します。Runtime は CPU 向けです。モデルは単一インスタンス、検出用画像サイズは制限し、PyTorch に依存せず、この小規模 Gallery ではベクトルデータベースも使用しません。

## プライバシー

本プロジェクトはローカルファーストです。元画像、embedding、テレメトリをアップロードせず、クラウド顔認識サービスも呼び出しません。Runtime は ONNX Runtime の import 前に `ORT_DISABLE_TELEMETRY=1` を設定し、アップローダーと永続デバイス識別子を無効にします。

## モデルライセンス

コードは MIT ライセンスですが、モデル重みは同梱せず、異なるライセンスや利用制限を持つ場合があります。使用前に各ダウンロードモデルの提供元と条件を [models/README.ja.md](models/README.ja.md) に記録してください。
