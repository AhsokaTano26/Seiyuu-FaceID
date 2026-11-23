# Docker Compose デプロイパッケージ

[English](docker.md) | [简体中文](docker.zh-CN.md) | [日本語](docker.ja.md)

Docker Engine と Compose v2 を事前にインストールします。モデルとキーは含めません。
.env.example は今回のコミットの Docker Hub イメージタグを固定し、後日の latest の誤使用を防ぎます。
イメージは Linux amd64 / arm64 に対応し、Python API を実行します。
高速な C++ / Go ネイティブ版には、別の native-linux-amd64 成果物を選択してください。

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
cp .env.example .env
chmod 600 .env
~~~

初回のモデル取得には Python 3.11+ が必要です。
別のマシンで取得して models/ をコピーすることもできます。
事前取得したアーカイブを使う場合は --archive /path/to/models.tar.gz のパスを置き換えて追加します。
openssl rand -hex 32 で自分専用のキーを生成し、.env の SEIYUU_FACEID_API_KEY に記入して次を実行します。

~~~bash
docker compose pull --policy always
docker compose up -d --wait --wait-timeout 180
curl --fail http://127.0.0.1:8000/ready
~~~

N150 の CPU 節約スレッド設定は `docker compose -f compose.yaml -f compose.n150.yaml up -d --wait --wait-timeout 180` で起動します。
ネイティブ FTZ/DAZ は有効になりません。

完全オフラインの場合は同じアーキテクチャのイメージを事前に docker save で出力し、対象機で docker load してから起動します。
[API ドキュメント](../../docs/API.ja.md) を参照してください。
アップロード用トップページは既定で無効です。Python のインターフェースドキュメントは /docs にあります。
