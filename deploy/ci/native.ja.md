# C++ 推論 + Go HTTP の Linux amd64 デプロイパッケージ

[English](native.md) | [简体中文](native.zh-CN.md) | [日本語](native.ja.md)

Linux x86_64 のランタイム、共有ライブラリー、ライセンス、元のモデルパラメーターの JSON エクスポートを含みます。
対象機でコンパイルする必要はなく、Python や Docker の常駐も不要です。
ビルド基準は Ubuntu 22.04、glibc 2.35+ です。
対象システムには libstdc++、libgcc、libm、libpthread などの標準ランタイムライブラリーも必要です。
実際の互換性は対象機の動的リンク検査で確認します。
ほかのアーキテクチャではこのバイナリーを使用しないでください。

ONNX 重みは release/model-source.json で固定した場所から取得し、各アプリパッケージに重複保存しません。
初回取得ツールには Python 3.11+ が必要です。
別のマシンで実行し models/ をコピーしても構いません。Python はこの準備工程だけで必要です。

~~~bash
sha256sum --check SHA256SUMS
python3 tools/fetch_models.py
export SEIYUU_FACEID_API_KEY='填写自己的随机密钥'
runtime/bin/faceid-server -model models/selected/native-model.json -addr 127.0.0.1:8000
~~~

native-model.json は N150 で検証済みの FTZ/DAZ 高速経路を既定で有効にし、FP32 重み、分類ヘッド、認識しきい値を維持します。
互換浮動小数点モードは models/selected/native-compatible.json を使い、対応する .sha256 ファイルも残します。
JSON の相対モデルパスは設定済みです。ディレクトリ配置を維持してください。

systemd を使う場合は展開内容を `/opt/seiyuu-faceid/releases/<version>` に置き、末尾をバージョン番号に置き換えます。
/opt/seiyuu-faceid/current をそのディレクトリに向けます。
root:root、0600 の /etc/seiyuu-faceid/api.env を作り、`SEIYUU_FACEID_API_KEY=<your-key>` に自分のキーを記入して次を実行します。

~~~bash
sudo install -m 0644 deploy/seiyuu-faceid.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
~~~

テンプレートは 0.0.0.0:8000 で待ち受けます。起動前に旧コンテナーやほかのサービスとのポート競合を避けます。
API を提供し、アップロード用トップページと /docs はありません。
フィールドは [API ドキュメント](../../docs/API.ja.md) を参照してください。

オフラインでモデルを取得する場合は `python3 tools/fetch_models.py --archive /path/to/models.tar.gz` のパスを置き換えて実行します。
モデル、ランタイム、メタデータの準備完了後は、開発用コンピューターや外部推論サービスに依存せず実行できます。
