# Python ネイティブサービスのデプロイパッケージ

[English](python.md) | [简体中文](python.zh-CN.md) | [日本語](python.ja.md)

今回のコミットの Python ソース、ロック済み依存関係、設定、systemd テンプレートを含みます。モデルとキーは含めません。
モデルの出所とファイルごとの検証は release/model-source.json に固定しています。
Linux、uv、API 実行用の Python 3.13 が必要です。
初回取得ツールは Python 3.11+ が必要です。別のマシンで取得して models/ をコピーしても構いません。

最初に OpenCV に必要なシステムランタイムライブラリーをインストールします。
Debian / Ubuntu では次を実行します。

~~~bash
sudo apt-get update
sudo apt-get install -y ca-certificates libgl1 libglib2.0-0 libgomp1
~~~

ほかのディストリビューションでは相当する OpenGL、GLib、OpenMP ランタイムを用意してください。
不足すると cv2 のインポートが失敗します。

展開したディレクトリで次を実行します。

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
uv sync --locked --no-dev --extra api --python 3.13
export SEIYUU_FACEID_API_KEY='填写自己的随机密钥'
SEIYUU_FACEID_CONFIG=configs/n150.yaml uv run --no-sync uvicorn seiyuu_faceid.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
~~~

既定で HTTP API と /docs を提供します。
N150 設定は元のモデルと認識しきい値を維持し、C++ の FTZ/DAZ 高速化はありません。
本番ではディレクトリを /opt/seiyuu-faceid に置き、deploy/seiyuu-faceid.service に従って faceid ユーザーを設定します。
deploy/faceid-native.env.example を /etc/seiyuu-faceid.env へコピーしてキーを記入し、テンプレートのインストール、daemon-reload、enable --now を実行します。
サービスユーザーはコード、モデル、.venv を読み取れる必要があります。
環境ファイルは root:root、0600 を維持します。
テンプレートは既定でアップロード用トップページを無効にします。環境変数を変更し再起動すれば有効にできます。

初回の uv sync インストールにはネットワークが必要です。
オフライン取得は `python3 tools/fetch_models.py --archive /path/to/downloaded-models.tar.gz` のパスを取得済みアーカイブに置き換えて実行します。
展開とファイル検証の完了後はモデル取得用の接続を必要としません。
[API ドキュメント](../../docs/API.ja.md) を参照してください。
