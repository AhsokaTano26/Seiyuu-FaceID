# Seiyuu FaceID HTTP API

[English](API.md) | [简体中文](API.zh-CN.md) | [日本語](API.ja.md)

HTTP サービスは実装済みです。本書ではインターフェースの仕様と呼び出し方法を説明します。
ローカルの SCRFD 検出器と AdaFace IR101 + R50 分類器を再利用し、外部の認識サービスは呼び出しません。

<a id="安装与启动"></a>

## インストールと起動

プロジェクトのルートで次を実行します。

```bash
uv sync --locked --extra api --extra web --extra dev
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

`models/det_500m.onnx` と完全な `models/selected/` モデル一式が必要です。
重みは Git に含まれないため、リポジトリのクローンだけではモデルを実行できません。
各プロセスは起動時に一度だけモデルを読み込み、分類器ファイルを検証します。
設定やモデルの欠落、検証失敗があれば起動に失敗し、成功したように見せる応答は返しません。

- ベース URL は `http://127.0.0.1:8000` です。
- 最小限のテストページは `/` です。画像を選択すると認識または検出を実行できます。複数画像は自動的に一括認識します。キーとパラメーターは「オプション設定」にあります。
- 対話型ドキュメントは `/docs`、ReDoc は `/redoc`、OpenAPI は `/openapi.json` です。
- 既定の設定はプロジェクトの `configs/default.yaml` です。
- `SEIYUU_FACEID_CONFIG` で設定ファイルの場所を変更できます。相対パスは起動ディレクトリから解決します。
- `SEIYUU_FACEID_API_KEY` は任意の Bearer キーです。未設定の場合は認証を要求しません。
- `SEIYUU_FACEID_SERVE_UI=0` はトップページのテスト画面を無効にし、API とインターフェースのドキュメントだけを提供します。

テストページを無効にし、ほかのマシンからサーバーの到達可能な IP で接続できるようにするには次を実行します。

```bash
SEIYUU_FACEID_SERVE_UI=0 uv run --extra api uvicorn seiyuu_faceid.api.app:create_app \
  --factory --host 0.0.0.0 --port 8000
```

呼び出し側は `http://<reachable-server-ip>:8000` のプレースホルダーをサーバーの到達可能な IP に置き換えます。
`0.0.0.0` は待受用のアドレスであり、接続先には指定できません。

```bash
export SEIYUU_FACEID_API_KEY='replace-with-your-key'
uv run uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

キーを有効にすると、すべての `/v1/` エンドポイントと `/api/v1/detect` で
`Authorization: Bearer replace-with-your-key` ヘッダーが必要です。ヘルスチェックとドキュメントにキーは不要です。
ドキュメントには非公開の人物一覧を含めません。人物一覧は認証付きモデルエンドポイントで取得します。

<a id="接口一览"></a>

## エンドポイント一覧

| メソッド | パス | 用途 |
| --- | --- | --- |
| GET | `/health` | プロセスの生存確認 |
| GET | `/ready` | モデルの準備状態 |
| GET | `/v1/model` | モデル識別子、クラス、既定値、アップロード制限 |
| POST | `/api/v1/detect` | 同一ホストのサービス向け顔有無検出。人数、境界ボックス、検出信頼度を返す |
| POST | `/v1/detect` | 顔位置と5点ランドマークのみを検出 |
| POST | `/v1/recognize` | 画像1枚の検出と人物認識 |
| POST | `/v1/recognize/batch` | 複数画像を認識し、アップロード順に各画像の結果を返す |

<a id="图片与参数约定"></a>

## 画像とパラメーターの仕様

- `multipart/form-data` でアップロードします。単一画像のフィールドは `file`、一括処理では `files` を繰り返します。
- 実際のファイル内容に基づいて JPEG、PNG、WebP を受け付けます。拡張子や申告 MIME 型には依存しません。
- アニメーション画像は拒否します。EXIF の向きを自動適用し、グレースケールや透過画像は3チャンネルへ変換します。
- 1ファイルの上限は **10 MiB（10,485,760 バイト）** です。デコード前に寸法を確認し、上限は **20,000,000 ピクセル**です。一括処理は最大 **8枚**、HTTP リクエスト全体は最大 **81 MiB** です。
- サーバー上のファイルパス、画像 URL、Base64 JSON は受け付けません。アップロードしたファイルをプロジェクトのデータセットへ保存しません。multipart パーサーが一時ファイルを使う場合は、リクエスト終了時に閉じます。
- 座標原点は EXIF の向きを適用した画像の左上です。単位はピクセルで、`bbox` は `[x1, y1, x2, y2]` です。5点の順序は左目、右目、鼻先、左口角、右口角です。
- すべての認識エンドポイントでクエリパラメーター `top_k`（整数、1–50）と `probability_threshold`（有限数、0–1）を指定できます。省略時は YAML の既定値を使い、現在は `5` と `0.95` です。変更はそのリクエストだけに適用され、ほかのリクエストや読み込み済みモデルには影響しません。
- `processing_ms` は画像のデコードと推論時間を含みます。アップロードと待ち行列の時間は含まず、性能を保証する値ではありません。

<a id="健康检查"></a>

## ヘルスチェック

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

`GET /health` は HTTP 200 と次の本文を返します。

```json
{"status": "ok"}
```

`GET /ready` はモデルの準備完了時に HTTP 200 と次の本文を返します。

```json
{"status": "ready"}
```

モデルがまだ利用できない場合、`/ready` は HTTP 503 とエラーコード `model_unavailable` を返します。
起動に失敗するとプロセスが終了するため、ヘルスチェックにも接続できません。

<a id="同机服务的人脸检测"></a>

## 同一ホストのサービス向け顔検出

Python と C++/Go の両サービスが `POST /api/v1/detect` に対応します。フィールド `file` で画像1枚をアップロードします。
サービスは起動時に読み込んだ SCRFD を再利用し、各画像につき検出を一度だけ実行します。
任意パラメーター `score_threshold` の範囲は `0 < 値 ≤ 1` で、そのリクエストだけに適用します。
省略するとモデル設定の `detector.score_threshold` を使います。同梱設定の値は `0.50` です。

```bash
curl --fail-with-body 'http://127.0.0.1:8000/api/v1/detect?score_threshold=0.7' \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

```json
{
  "has_face": true,
  "face_count": 1,
  "faces": [{"bbox": [38.2, 27.1, 104.5, 119.7], "detector_score": 0.93}]
}
```

検出スコアがしきい値を満たし、重複するボックスを除去した後、顔1つにつき1項目を返します。
座標は EXIF の向きを補正した元画像のピクセル値です。
条件を満たす顔がなければ HTTP 200 と `{"has_face":false,"face_count":0,"faces":[]}` を返します。
声優の人物ラベル、鮮明度、認識確率はこの判定に使いません。

| 設定 | 既定値 | 作用 |
| --- | ---: | --- |
| `SEIYUU_FACEID_DETECT_MAX_PIXELS` | 20000000 | 新エンドポイントが受け付ける幅×高さの上限。1まで引き下げ可能。寸法確認後にデコードする。1ファイルの上限は引き続き10 MiB |
| `detector.max_long_edge` | 1920 | 共有検出器で長辺の大きすぎる画像を縦横比を保って縮小する。ネイティブサービスはエクスポート済みモデル設定を使用 |
| `SEIYUU_FACEID_DETECT_TIMEOUT_MS` | 5000 | 新エンドポイントがリクエストヘッダーを受信してから、アップロード、待機、デコード、推論で共有するミリ秒単位の予算。正の整数 |
| `SEIYUU_FACEID_DETECT_CLIENT_CIDR` | 空 | 既定では IPv4/IPv6 ループバックを許可。同一ホストのコンテナーのプライベートアドレスまたはサブネットを1つ追加で許可できる。例は `172.29.0.3/32` |

同一ホストのプロセスは `127.0.0.1` で呼び出し、既存の Bearer キーを使用します。
Docker Compose では上表の環境変数を `.env` で渡します。
ホストから公開ポート経由で呼び出す場合は `FACEID_BIND_ADDRESS=127.0.0.1` を設定し、コンテナーから見えるホストのブリッジゲートウェイを単一アドレス CIDR で許可します。
同一ホストのコンテナー同士で直接呼ぶ場合は、専用ネットワーク上の呼び出し元コンテナーのアドレスまたはサブネットを指定します。
systemd では既存の `EnvironmentFile` に設定し、変更後に該当サービスを再起動します。
外部公開するリバースプロキシではこのパスを除外してください。ローカルプロキシ自身が許可済み送信元アドレスを持つ場合があるためです。

デプロイ用プローブは引き続き `/health` と `/ready` を使います。両エンドポイントは推論中も応答できます。
タイムアウト時は HTTP 503 と `detection_timeout` を返します。待ち行列内のリクエストは待機を終了します。
開始済みの同期推論は完了後にモデル実行スロットを解放します。
検出リクエストの状態と所要時間はサービスログで確認できます。Python のロガーは `seiyuu_faceid.api.detection`、レベルは INFO です。

<a id="模型信息"></a>

## モデル情報

`GET /v1/model` は次のフィールドを返します。

- `model_id` は分類モデル資源の SHA-256 識別子です。
- `recognition_id` はモデルと既定の実行設定を合わせた識別子です。
- `class_count` と `class_names` は学習済み人物数と完全な人物一覧です。
- `defaults` は `top_k` と `probability_threshold` です。
- `limits` は `max_upload_bytes`、`max_request_bytes`、`max_image_pixels`、`max_batch_size` です。
- `supported_formats` は `["JPEG", "PNG", "WEBP"]` です。

```bash
curl --fail http://127.0.0.1:8000/v1/model \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY"
```

<a id="单图识别"></a>

## 単一画像の認識

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/recognize \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" \
  -F 'file=@photo.jpg'

curl --fail-with-body \
  'http://127.0.0.1:8000/v1/recognize?top_k=3&probability_threshold=0.98' \
  -F 'file=@photo.jpg'
```

次は HTTP 200 の応答例です。値は説明用です。

```json
{
  "model_id": "model-sha256",
  "image": {"filename": "photo.jpg", "width": 1000, "height": 1200},
  "face_count": 1,
  "processing_ms": 120.5,
  "faces": [
    {
      "bbox": [100.0, 120.0, 300.0, 340.0],
      "detector_score": 0.89,
      "quality_score": 1.0,
      "quality_reasons": [],
      "status": "recognized",
      "identity": "上坂堇",
      "probability_threshold": 0.95,
      "decision_reason": null,
      "score_type": "class_probability",
      "candidates": [{"identity": "上坂堇", "score": 0.9999}]
    }
  ]
}
```

`candidates` は確率の降順で最大 `top_k` 件を返します。例ではほかの候補を省略しています。
顔が検出されなくても HTTP 200 を返し、`face_count: 0`、`faces: []` となります。

| フィールド/状態 | 意味 |
| --- | --- |
| `recognized` | 品質検査に合格し、最大クラス確率が今回のしきい値以上。`identity` が認識した人物 |
| `review` + `low_probability` | 最大クラス確率が不足。`identity: null` で候補を保持 |
| `review` + `quality_rejected` | 顔が小さすぎる、またはぼやけすぎている。`identity: null` で候補は空 |
| `quality_reasons` | `face_too_small`、`too_blurry` を含む場合がある |
| `score_type` | `class_probability`。固定された既知クラス内の確率 |

現在のモデルは50人に固定されています。クラス確率から、その人が既知の人物に属する確実性は得られません。
未知の人物やキャラクター画像を拒否する校正は未実施です。
高いスコアだけで、任意にアップロードされた人物の実際の身元を判定することはできません。

<a id="仅检测"></a>

## 検出のみ

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/detect -F 'file=@photo.jpg'
```

トップレベルのフィールドは認識応答と同じです。各顔は次の形式です。

```json
{
  "bbox": [100.0, 120.0, 300.0, 340.0],
  "detector_score": 0.89,
  "landmarks": [[150.0, 180.0], [250.0, 180.0], [200.0, 230.0],
                [160.0, 280.0], [240.0, 280.0]]
}
```

このエンドポイントでは人物分類と顔品質による拒否を行いません。検出できた小さな顔も返します。

<a id="批量识别"></a>

## 一括認識

```bash
curl --fail-with-body 'http://127.0.0.1:8000/v1/recognize/batch?top_k=3' \
  -F 'files=@photo1.jpg' -F 'files=@photo2.png'
```

HTTP 200 は `{"results": [...]}` を返します。各項目は単一画像の応答と同じ形式で、0始まりの `index` を含みます。
同じファイル名の画像を統合しません。個別の空ファイル、未対応形式、破損、上限超過はその位置にエラーを返し、ほかの画像の処理は継続します。

```json
{
  "results": [
    {
      "index": 0,
      "filename": "broken.jpg",
      "error": {"code": "invalid_image", "message": "Could not decode image."}
    }
  ]
}
```

一括枚数超過、リクエスト本文超過、不正なパラメーター、必須フィールドの欠落、認証失敗ではリクエスト全体が失敗し、推論を開始しません。
一括処理中にモデルエラーが発生すると、リクエスト全体で HTTP 500 を返します。

<a id="错误格式"></a>

## エラー形式

HTTP エラーは `{"error": {"code": "...", "message": "..."}}` に統一します。
パラメーター検証時は `details` を追加し、各項目に `field` と `message` を含めます。
画像、キー、ローカルパスは応答に再掲しません。

| HTTP | code | 条件 |
| --- | --- | --- |
| 400 | `empty_file` | 空ファイル |
| 400 | `invalid_image` | 破損した画像、またはデコード不能 |
| 400 | `bad_request` | 不正な multipart またはヘッダー |
| 401 | `unauthorized` | キー有効時に有効な Bearer キーがない |
| 403 | `internal_only` | 新しい検出エンドポイントで接続元が未許可 |
| 404 | `not_found` | ルートが存在しない |
| 405 | `method_not_allowed` | HTTP メソッドが不正 |
| 413 | `file_too_large` | 1ファイルが10 MiBを超過 |
| 413 | `request_too_large` | 本文が81 MiBを超過。Content-Length がなくてもストリームを計数 |
| 413 | `image_too_large` | 対象エンドポイントのピクセル上限を超過 |
| 415 | `unsupported_image` | JPEG/PNG/WebP 以外、またはアニメーション画像 |
| 422 | `validation_error` | フィールド欠落、不正なパラメーター、一括枚数が1–8以外 |
| 500 | `internal_error` | 推論エラーまたは予期しないサービスエラー。詳細なスタックはサービスログだけに記録 |
| 503 | `model_unavailable` | モデルが未準備 |
| 503 | `detection_timeout` | 新しい検出エンドポイントでリクエスト期限を超過 |

<a id="python-http-客户端"></a>

## Python HTTP クライアント

クライアント側で `httpx` を別途インストールします。サービスにキーを設定した場合は headers に Authorization を追加します。

```python
from pathlib import Path
import httpx

photo = Path("photo.jpg")
with photo.open("rb") as handle:
    response = httpx.post(
        "http://127.0.0.1:8000/v1/recognize",
        files={"file": (photo.name, handle, "application/octet-stream")},
        params={"top_k": 5},
        timeout=60,
    )
response.raise_for_status()
for face in response.json()["faces"]:
    print(face["bbox"], face["status"], face["identity"], face["candidates"])
```

<a id="部署与验证"></a>

## デプロイと検証

通常の Linux サーバーへの完全な独立デプロイと、イメージおよびモデルのオフライン移行は [Linux デプロイ](DEPLOY_LINUX.ja.md) を参照してください。

既定ではローカルだけで待ち受けます。LAN 接続には `--host 0.0.0.0` を明示し、キーを設定します。
インターネット公開時はリバースプロキシで HTTPS、アクセス制御、リクエスト制限を提供します。クロスオリジン接続は既定で無効です。
まず1 workerで運用することを推奨します。各 worker は約415 MiBの重みを個別に読み込み、さらに推論用メモリを使用します。
プロセス内の推論はスレッドプールで直列実行します。ヘルスチェックは推論完了を待ちません。

```bash
uv run pytest
uv run ruff check .
uv pip check
```

API の自動テストは HTTP 契約と異常入力を対象にします。ローカルに非公開の重みがある場合は、実モデルの統合テストも別途実行します。
CI は非公開の写真やモデルをダウンロード、コミットしません。

フレームワークの使用方法は [FastAPI ファイルアップロード](https://fastapi.tiangolo.com/tutorial/request-files/) と [Uvicorn 起動設定](https://uvicorn.dev/settings/) を参照してください。
