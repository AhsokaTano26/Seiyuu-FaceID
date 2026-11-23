# Linux 独立デプロイパッケージ

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

このディレクトリには学習済みモデルと設定を含みます。
サービスイメージは GitHub Actions から `tano26/seiyuu-faceid` へ公開します。
実行時の検出、位置合わせ、人物認識はすべて対象サーバー内で完結します。
開発用コンピューター、学習写真、ホストへの Python インストール、PyPI、外部推論サービスには依存しません。
サーバーには Docker Engine と Docker Compose v2 を事前にインストールします。
イメージのアーキテクチャをサーバーに合わせてください。

<a id="启动"></a>

## 起動

このディレクトリで操作します。まず全ファイルを検証し、その後イメージを読み込みます。

```bash
sha256sum --check SHA256SUMS
cp .env.example .env
chmod 600 .env
```

`openssl rand -hex 32` でキーを生成し、`.env` の `SEIYUU_FACEID_API_KEY=` の後に記入します。
既定ではサーバーの全インターフェースの8000番ポートで待ち受けます。
バインドアドレスとポートは `.env` で変更できます。
イメージ公開後に取得して起動します。

```bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
curl --fail http://127.0.0.1:8000/ready
```

初回のモデル読み込みが180秒を超える場合は、ログを確認してから待機時間を延ばします。
実行時に依存関係やモデルを取得しません。
Compose は既定で自動 pull を禁止し、以降の起動や再作成にも Docker Hub 接続は不要です。
インストールや更新時だけ、上記の pull コマンドを明示的に実行します。
ホストとコンテナーの両方で `models/` と `configs/default.yaml` を読み取れる必要があります。

パッケージに `image.tar` も含む場合は、対象サーバーを完全オフラインで起動できます。
`docker load --input image.tar` で対応アーキテクチャのイメージを読み込み、`docker compose up -d --pull never --wait --wait-timeout 180` を実行します。
`docker compose pull` は不要です。Docker と Compose 自体は事前にインストールします。
更新時は先に `.env` のイメージバージョンを保存し、新版を取得して `docker compose up -d` を実行できます。

<a id="远程调用"></a>

## 遠隔からの呼び出し

呼び出し端末でクライアント変数を設定します。キーはサーバーの `.env` と同じ値にします。

```bash
export FACEID_URL='http://服务器IP:8000'
export FACEID_API_KEY='你填写的密钥'
curl --fail-with-body "$FACEID_URL/v1/model" \
  -H "Authorization: Bearer $FACEID_API_KEY"
curl --fail-with-body "$FACEID_URL/v1/recognize" \
  -H "Authorization: Bearer $FACEID_API_KEY" \
  -F 'file=@photo.jpg'
```

選択したポートへの呼び出し元の接続を許可します。
公開サービスでは HTTPS リバースプロキシを使用します。
完全な仕様は [API.md](../docs/API.ja.md)、オンラインドキュメントは `/docs` です。
トップページのテスト画面は既定で無効です。

<a id="运维"></a>

## 運用

```bash
docker compose ps
docker compose logs --tail 100 faceid
docker stats --no-stream
docker compose down
```

正常停止後に手動で起動する場合も `docker compose up -d` が必要です。
実行中サービスは `unless-stopped` ポリシーでクラッシュやホスト再起動から復旧します。
ヘルスチェックは異常を表示しますが、`unhealthy` だけでコンテナーを自動再起動しません。
設定変更後は `docker compose restart faceid`、`.env` 変更後は `docker compose up -d --force-recreate` を実行します。

現在は CPU 推論、1 workerです。
通常のテストサーバーは容量計画の出発点として2コア、4 GiBメモリを用意できます。
最小メモリ、遅延、スループットは対象サーバーで実測してください。
重みは約417 MiBで、実行時メモリとは異なります。
複数の顔を含む画像は計算時間が増えます。
既存の重みは学習済み50人に対応し、移行時の再学習は不要です。

N150 + 8 GB の Linux ホストでは、対応コードを含む新イメージで同梱の CPU 節約設定を使用できます。

```bash
docker compose -f compose.yaml -f compose.n150.yaml up -d --wait --wait-timeout 180
```

元の FP32 モデルと並列デュアルバックボーンを維持し、各バックボーンを1スレッド、スピン無効にします。
実際の遅延、CPU、メモリは対象機で測定してください。
操作は [N150 ガイド](../docs/N150.ja.md) にあります。

設定とモデルはこのディレクトリ内にあるため、全体のバックアップで移行できます。
ローカルモデルを含むため、意図したアクセス範囲で保管してください。
出所と利用範囲は `models/selected/MODEL_CARD.md` と `manifest.json` を参照してください。
