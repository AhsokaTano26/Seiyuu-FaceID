# Linux サーバーへの完全な独立デプロイ

[English](DEPLOY_LINUX.md) | [简体中文](DEPLOY_LINUX.zh-CN.md) | [日本語](DEPLOY_LINUX.ja.md)

Docker を使わないインストールと systemd 設定は [Linux ネイティブデプロイ](DEPLOY_NATIVE_LINUX.ja.md) を参照してください。

引き続き Python + FastAPI + ONNX Runtime を使い、まず CPU 版をデプロイすることを推奨します。
認識はすべて対象サーバーで実行し、開発用コンピューターや外部モデル API を呼び出しません。再学習は不要です。
実行には次の構成要素が必要です。

- Docker サービスイメージには Python、ロック済み API 依存関係、プロジェクトコード、Linux ネイティブライブラリーを含めます。
- ローカルモデルには SCRFD 検出器、AdaFace IR101、R50、分類ヘッド、モデルマニフェストを用意します。
- 設定と Compose で待受ポート、キー、ヘルスチェック、ログ、再起動ポリシーを定義します。

対象ホストに Docker Engine と Docker Compose v2 を事前にインストールします。
テスト環境は2コア、4 GiBメモリを出発点とします。最低構成は未検証で、実際の容量は対象 CPU、画像内の顔数、同時実行数に依存します。
同梱設定はモデルごとに推論2スレッド、サービス1 workerで、認識用の2バックボーンを並列実行します。
認識段階では2モデル合計で最大約4計算スレッドを使用します。
2コアまたは CPU 割当の少ないホストでは `runtime.parallel_backbones: false`、または `runtime.intra_op_threads` を1にして実測してください。
ローカルの並列ベンチマークは対象 Linux ホストのスループットを示しません。[最適化報告](CLASSIFIER_OPTIMIZATION.ja.md) を参照してください。
現在の実装は CPU のみを使用します。GPU を搭載したサーバーでも GPU アクセラレーションは自動的に有効になりません。

対象が **Intel N150 + 8 GB、Linux/Docker** の場合は、追加された `configs/n150.yaml` と `compose.n150.yaml` を使用できます。
このプリセットはスレッドの負荷を減らし、ORT のスピン待機を無効にします。元の FP32 モデルと認識規則は維持します。
リソースと遅延のトレードオフ、対象機の計測コマンドは [N150 ガイド](N150.ja.md) にあります。

<a id="1-通过-github-actions-发布到-docker-hub"></a>

## 1. GitHub Actions から Docker Hub へ公開

ワークフローはリポジトリの `.github/workflows/ci.yml`、イメージリポジトリは [`tano26/seiyuu-faceid`](https://hub.docker.com/r/tano26/seiyuu-faceid) です。
GitHub の [Actions Secrets](https://github.com/AhsokaTano26/Seiyuu-FaceID/settings/secrets/actions) に `DOCKERHUB_TOKEN` を追加します。
値には当該 Docker リポジトリへの push 権限を持つアクセストークンを設定します。
ログイン名は `tano26` に固定されているため、ユーザー名変数の追加設定は不要です。

ワークフローと Dockerfile を含むコードを `main` に push すると、lint、テスト、コンパイル検査の後にイメージをビルドして push します。
Actions で `CI and Docker Hub` を手動実行することもできます。
PR は品質検査だけを実行し、Docker Hub にログイン、公開しません。

| トリガー | イメージタグ |
| --- | --- |
| `main` への push、または `main` 上で手動実行 | `latest`、`sha-<full-commit-sha>`。後者は完全なコミット SHA |
| `v0.1.0` などのバージョンタグを push | `0.1.0`、`sha-<full-commit-sha>`。後者は完全なコミット SHA |

バージョンタグは `latest` を上書きしません。
ワークフローは `linux/amd64` と `linux/arm64` を公開し、対象サーバーは pull 時に一致するアーキテクチャを自動選択します。
Python 依存関係は `uv.lock` に従い、API だけを有効にしてインストールします。
学習、Streamlit、テスト用依存関係は含めません。
各アーキテクチャのビルド時にネイティブライブラリーと API モジュールのインポートを検証します。
イメージビルドに学習済み重みは不要です。

Docker ビルドコンテキストは許可リスト方式で、`models/`、写真、実験データ、`.env`、ローカルの `.venv` を含めません。
GitHub Runner にもこれらの非公開モデルはありません。
モデルは次節の手順で別途配布し、対象サーバーで読み取り専用マウントします。
実行時に開発用コンピューターからファイルを読みません。

<a id="2-生成模型和部署配置包"></a>

## 2. モデルとデプロイ設定パッケージの生成

モデルは `.gitignore` で除外されるため、Git のクローンだけでは不十分です。
現在の完全なモデルを持つマシンでパッケージを作成します。必要なファイルは次のとおりです。

```text
models/det_500m.onnx
models/selected/adaface_ir101.onnx
models/selected/w600k_r50.onnx
models/selected/classifier.npz
models/selected/manifest.json
models/selected/MODEL_CARD.md
models/selected/ADAFACE_LICENSE
```

Python 標準ライブラリーだけに依存するパッケージツールを使用します。

```bash
python3 tools/package_deployment.py \
  --output dist/seiyuu-faceid-deployment
tar -C dist -czf dist/seiyuu-faceid-deployment.tar.gz seiyuu-faceid-deployment
```

ツールは4つのモデルファイルの SHA-256 を確認し、パッケージ全体の `SHA256SUMS` を生成します。既存の出力ディレクトリの上書きは拒否します。
出力にはモデル、設定、Compose、キーが空の設定例、独立実行の説明を含めます。
学習データ、テスト写真、評価特徴量、ローカルキーは含めません。
イメージタグを変更する場合は、パッケージの `.env` の `FACEID_IMAGE` も更新します。

生成した `.tar.gz` を対象サーバーへ転送します。
サービスイメージを Docker Hub から一度取得すれば独立して実行できます。
対象ホストに Python、uv、学習依存関係をインストールする必要はなく、PyPI にも接続しません。
Docker と Compose 自体は事前にインストールしてください。

対象サーバーから Docker Hub に接続できない場合は、接続可能なマシンで対象アーキテクチャのイメージを取得、エクスポートし、ツールの `--image-tar` 引数でパッケージに含めます。

```bash
docker pull --platform linux/amd64 tano26/seiyuu-faceid:latest
mkdir -p dist
docker image save --output dist/image-linux-amd64.tar tano26/seiyuu-faceid:latest
python3 tools/package_deployment.py \
  --image-tar dist/image-linux-amd64.tar \
  --output dist/seiyuu-faceid-offline
tar -C dist -czf dist/seiyuu-faceid-offline.tar.gz seiyuu-faceid-offline
```

サーバーで `uname -m` を実行してアーキテクチャを確認します。
`x86_64` は `linux/amd64`、`aarch64` は `linux/arm64` に対応します。
オフライン用イメージは対象アーキテクチャに合わせてください。

<a id="3-在目标-linux-服务器启动"></a>

## 3. 対象 Linux サーバーで起動

```bash
tar -xzf seiyuu-faceid-deployment.tar.gz
cd seiyuu-faceid-deployment
sha256sum --check SHA256SUMS
cp .env.example .env
chmod 600 .env
```

`openssl rand -hex 32` でキーを生成し、`.env` の `SEIYUU_FACEID_API_KEY=` の後へ記入します。
既定では全ネットワークインターフェースの8000番ポートで待ち受けます。
ポートを変更する場合は `FACEID_PORT` を編集します。
既存の HTTPS リバースプロキシを使う場合は `FACEID_BIND_ADDRESS` を `127.0.0.1` に変更できます。
本番では `FACEID_IMAGE` を `latest` から固定のバージョンまたは SHA タグへ変更できます。
イメージ公開後に取得してサービスを起動します。

```bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
docker compose ps
curl --fail http://127.0.0.1:8000/ready
```

準備完了時の応答は `{"status":"ready"}` です。
`/ready` の成功はモデル読み込みの完了を示します。
画像認識の全経路は `/v1/recognize` が正常応答して初めて確認できます。
コンテナー実行時の外向き通信は不要です。
Compose は既定で自動 pull を禁止するため、その後の起動や再作成もオフラインで実行できます。
インストールや更新時だけ `docker compose pull --policy always` で明示的に取得します。
モデルと設定はコンテナー UID 10001 から読み取れる必要があります。
パッケージツールは読み取り権限を設定済みです。
手動コピー時はファイルの読み取り権限とディレクトリの探索権限を維持してください。

オフラインパッケージに `image.tar` がある場合は `docker load --input image.tar` で読み込み、`docker compose up -d --pull never --wait --wait-timeout 180` で起動します。
`docker compose pull` は省略します。Compose はネットワーク経由の pull を試みません。

別の端末からは次のように呼び出します。

```bash
export FACEID_URL='http://服务器IP:8000'
export FACEID_API_KEY='与服务端一致的密钥'
curl --fail-with-body "$FACEID_URL/v1/model" \
  -H "Authorization: Bearer $FACEID_API_KEY"
curl --fail-with-body "$FACEID_URL/v1/recognize" \
  -H "Authorization: Bearer $FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

サーバーのファイアウォールまたはクラウドのセキュリティグループで、呼び出し元から選択ポートへの接続を許可します。
公開ネットワークでは HTTPS を使用します。形式と応答状態は [API.md](API.ja.md) を参照してください。
モデルには学習済みの50人が含まれます。
低信頼度または品質不足では既定で `review` を返します。移行してもこの規則は変わりません。

<a id="4-检查和维护"></a>

## 4. 確認と保守

```bash
docker compose logs --tail 100 faceid
docker stats --no-stream
docker compose pull --policy always
docker compose up -d
docker compose restart faceid
docker compose down
```

ヘルスチェックは `/ready` を呼び出します。
プロセス終了後は `unless-stopped` ポリシーで再起動できます。
ヘルス状態の異常だけでは自動再起動しません。
`.env` 変更後は `docker compose up -d --force-recreate` を使います。
設定変更やモデル置換後はサービスを再起動します。
デプロイディレクトリと `.env` をバックアップすれば実行に必要な内容を保存できます。

対象機の受け入れ確認では、少なくとも `/ready`、キー付き `/v1/model`、既知人物写真の `/v1/recognize`、キーなしの401応答、再起動後の復旧を確認します。
続いてメモリと認識遅延を記録し、スレッド調整や容量増加の要否を決めます。
モデルファイルは合計約417 MiBで、実行メモリは別途計測が必要です。
worker を増やすと通常はモデルのメモリ使用量も増えます。

Docker の参考資料は [uv Docker 統合](https://docs.astral.sh/uv/guides/integration/docker/)、[Actions イメージ公開](https://docs.docker.com/build/ci/github-actions/push-multi-registries/)、[Compose 環境変数](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)、[Compose サービス設定](https://docs.docker.com/reference/compose-file/services/) です。
