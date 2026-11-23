# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

**SCRFD + AdaFace IR101 + InsightFace R50 + StandardScaler + ロジスティック回帰**によるローカル顔検出・教師あり人物分類システムです。ONNX Runtime を使い CPU で推論します。

## モデルの版

| バンドル | クラス数 | 学習に使用した顔 | ホールドアウト | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| 既定の `models/selected/` と CI が固定する取得元 | 50 | 4,064 | 1,051 | 98.48%（1,035/1,051） |
| 公開版 `models-5cc19e9d5b1a30aa` | 606 | 49,858 | 8,759 | 99.38%（8,705/8,759） |

各版は別のデータセットと評価履歴を持ちます。50クラスヘッドは過去のホールドアウトで繰り返し比較して選定しました。606クラスの正解はディレクトリ名に基づき、画像ごとの確認は未完了です。[606クラス報告](docs/MODEL_ACCURACY_20261103.ja.md)と[過去の最適化報告](docs/CLASSIFIER_OPTIMIZATION.ja.md)を併せて読んでください。`GET /v1/model` と分類ヘッドのチェックサムで使用中の版を確認します。

## ローカルで起動

```bash
uv sync --locked --extra api --extra web --extra dev
uv run streamlit run app.py
uv run python tools/recognize.py path/to/photo.jpg
uv run python tools/batch_recognize.py path/to/photos
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

完全なバンドルを `models/selected/`、検出器を `models/det_500m.onnx` に配置します。重みと非公開データは Git の管理対象外です。実行設定は `configs/default.yaml` です。API と未分類写真はラベルなしで保持し、キャラクター画像は実人物の学習ラベルから除きます。

ウェブ画面はそれぞれ任意の追加依存を使用します。出典と上流の条件は[モデル情報](models/README.ja.md)を参照してください。

## HTTP API

| メソッド | パス | 用途 |
| --- | --- | --- |
| GET | `/health`、`/ready` | プロセスとモデルの状態 |
| GET | `/v1/model` | 使用中のモデル、人物一覧、既定値 |
| POST | `/api/v1/detect` | 内部サービス向けの顔の有無、矩形、信頼度 |
| POST | `/v1/detect` | 検出矩形、スコア、5点ランドマーク |
| POST | `/v1/recognize` | 単一画像の人物分類 |
| POST | `/v1/recognize/batch` | 順序を保持した一括分類 |

```bash
curl --fail-with-body 'http://127.0.0.1:8000/api/v1/detect?score_threshold=0.7'   -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
curl --fail-with-body http://127.0.0.1:8000/v1/recognize   -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
```

内部検出 API は読み込み済み検出器を再利用し、`has_face`、`face_count`、`faces` を返します。既定の入力上限は2,000万画素、要求タイムアウトは5秒です。認証は `SEIYUU_FACEID_API_KEY` を使用します。同一ホストのコンテナから呼ぶ場合は許可するプライベートサブネットを設定します。入力制限、設定、エラーは [APIガイド](docs/API.ja.md)に記載しています。検出にはインポート用の [OpenAPI定義](docs/detect.openapi.yaml)もあります。Python サービスの `/docs` で対話式ドキュメントを開けます。

## 認識とデータ

各顔に対して検出、品質確認、5点位置合わせ、二つの固定バックボーンを実行します。AdaFace、R50 の単位特徴量を順に連結して√2で割り、標準化と分類へ渡します。スコアは組み込み人物間のクラス確率で、既定の採用しきい値は `0.95` です。品質不足や低確率の場合は `review` を返し、利用可能な候補を保持します。未知人物とキャラクター画像の拒否性能は別途検証が必要です。

過去のプールは5,115件で、学習/検証/テスト分割は3,574/490/1,051件です。最初の二つの4,064件で分類ヘッドを学習しています。過去の Top-5 は99.14%、Macro-F1 は正解テスト画像がある49クラスに対して99.10%です。過去の ONNX 再抽出検証では全 Top-1 予測が再現されました。対応するバンドルの `evaluation.json`、`holdout_scores.npz`、`runtime_verification.json` に記録があります。

ウェブ画面は元画像の管理、人手による顔確認、認識を提供します。写真や人物を追加する際は、別途分割を準備して分類ヘッドを再学習します。非公開の固定特徴量は `data/training/`、実験は `data/experiments/` にあります。過去の整理で一部の元写真と顔画像は削除済みです。[データ配置](data/README.ja.md)と[学習手順](training/README.ja.md)を参照してください。

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
uv run pytest
uv run ruff check .
```

学習は新しい出力先へ書き込みます。特徴量評価には固定アーカイブ、再抽出には対応する顔画像が必要です。付属設定はバックボーンごとに2推論スレッドを使って並列処理します。この項目を持たない旧設定は逐次実行です。

## 配備と詳しい資料

Docker、Python/systemd、C++/Go/systemd に対応します。CI の配布アーカイブは Actions Artifacts、イメージは Docker Hub、モデルと確認済み学習件数は GitHub Releases にあります。

N150 + 8 GB では `configs/n150.yaml` と `compose.n150.yaml` により CPU 処理量を減らせます。初期プロファイルの実測は CPU 時間が約34.4%減り、平均 HTTP 時間が約37.9%増えました。その後のネイティブ FTZ/DAZ 実験は要求あたり約0.68秒で、過去のホールドアウト成績は98.48%でした。過去の報告時点では長時間運転と負荷試験の受け入れ確認が未完了です。[N150報告](docs/N150_BENCHMARK.ja.md)を参照してください。

M4 Pro での LFW/WIDER FACE 性能比較は固定756画像と計時9,072要求を使用し、完全認識のネイティブ C++/Go 平均時間は約1.8%–2.7%短くなりました。独立した精度評価では LFW 全6,000ペアが99.13%、WIDER 全3,226検証画像の Easy/Medium/Hard AP が85.49/79.48/48.84（×100）でした。[性能報告](docs/STANDARD_DATASET_BENCHMARK.ja.md)と[精度報告](docs/STANDARD_DATASET_ACCURACY.ja.md)に各タスクと範囲を記載しています。

[ドキュメント索引](docs/README.ja.md)にアーキテクチャ、API、配備、学習、評価、過去の実装計画をまとめています。管理対象の Markdown 文書の上部から言語を切り替えられます。
