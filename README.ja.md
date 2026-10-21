# seiyuu-faceid

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Linux サーバーへの独立デプロイとモデル・イメージのオフライン転送は
[Linux デプロイ手順（中国語）](docs/DEPLOY_LINUX.md)を参照してください。

現在のモデルは **AdaFace IR101 + InsightFace R50 + 標準化 + ロジスティック回帰** に統一されています。50 人の既知クラスをローカルの ONNX Runtime で分類します。

```bash
uv sync --locked --extra api --extra web --extra dev
uv run streamlit run app.py
uv run python tools/recognize.py path/to/photo.jpg
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
```

HTTP API は単一画像・複数画像の識別、顔検出、モデル情報、ヘルスチェックに対応します。

```bash
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
curl --fail-with-body http://127.0.0.1:8000/v1/recognize -F 'file=@photo.jpg'
```

リクエスト、レスポンス、エラー、任意の Bearer 認証は [API ガイド](docs/API.md) を参照してください。
起動後の対話型ドキュメントは `http://127.0.0.1:8000/docs` です。
ウェブ画面のみを利用する場合、`--extra api` は不要です。

モデルは `models/selected/`、検出器は `models/det_500m.onnx` に配置します。重みと個人データは Git の対象外です。

4,064 件で学習し、固定留保データ 1,051 件で **Top-1 98.48%（1,035 件正解）**、Top-5 99.14%、Macro-F1 99.10%（評価対象49クラス）でした。ONNX 版の全 Top-1 予測は元の実験と一致します。同じ留保データを繰り返し探索して選定したため、独立データでの性能保証ではありません。

原画像、確認済みラベル、分割、人顔切り出しは保存済みです。未分類画像は未ラベルであり、キャラクター画像は真人の学習対象に含めません。出力は既知クラス内の確率です。未知人物・イラストの拒否性能は校正されていません。既定の確率しきい値は 0.95 です。確率がしきい値未満、または品質不足の場合は `review` になります。

ウェブ画面でのサンプル整理はモデルを更新しません。旧モデルと実験は削除しました。詳細は [English](README.md) または [简体中文](README.zh-CN.md)、出典は [モデル情報](models/README.md) を参照してください。
