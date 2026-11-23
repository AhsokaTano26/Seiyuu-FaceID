# Linux ネイティブデプロイ（Docker なし）

[English](DEPLOY_NATIVE_LINUX.md) | [简体中文](DEPLOY_NATIVE_LINUX.zh-CN.md) | [日本語](DEPLOY_NATIVE_LINUX.ja.md)

以下の例は Debian / Ubuntu の amd64 ホストを対象とし、`/opt/seiyuu-faceid` にインストールして systemd で API 1 workerを管理します。
N150 + 8 GB ではまず `configs/n150.yaml` を使用できます。
モデル、FP32 精度、検出しきい値、受け入れしきい値は維持します。
本手順の作成時点では対象 Linux ホストに接続できず、以下の手順は対象機で未検証でした。
性能データの適用範囲は [N150 ガイド](N150.ja.md) の説明を参照してください。

Python は API とスケジューリングを担当し、主要なモデル計算はすでに ONNX Runtime のネイティブコードで実行しています。
Docker を使わずにデプロイできます。遅延やメモリが減るかは同一ホストで比較計測する必要があります。
この段階では、ほかの言語への書き直しでエンドツーエンド性能が向上すると保証できる根拠はありません。

<a id="安装代码运行时和模型"></a>

## コード、ランタイム、モデルのインストール

今回のネイティブデプロイ用ファイルとリソース設定項目を含むコードの checkout を用意します。
モデルは Git リポジトリに含まれないため、使用権限のあるモデルと付随する利用条件を別途取得します。
ファイル一覧は [Linux デプロイガイド](DEPLOY_LINUX.ja.md#2-生成模型和部署配置包) を参照してください。
学習写真、学習特徴量、テストデータ、ローカルキーはコピーしないでください。

まず [uv 公式インストール手順](https://docs.astral.sh/uv/getting-started/installation/) に従って uv をインストールし、実行ファイルを `/usr/local/bin/uv` に配置します。
以下はこのパスが利用可能な前提です。システムに Python 3.13 を別途インストールする必要はありません。

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates openssl libgl1 libglib2.0-0 libgomp1
sudo useradd --system --user-group --home-dir /opt/seiyuu-faceid --no-create-home --shell /usr/sbin/nologin faceid
sudo git clone https://github.com/AhsokaTano26/Seiyuu-FaceID.git /opt/seiyuu-faceid
cd /opt/seiyuu-faceid
```

`faceid` ユーザーやコードディレクトリがすでにある場合は再利用し、対応する作成コマンドを省略します。
コードに本ガイドと `deploy/seiyuu-faceid.service` が含まれていることを確認します。
非公開モデルを次の場所に配置し、`MODEL_CARD.md` と `ADAFACE_LICENSE` などの付随条件を維持してください。

```text
/opt/seiyuu-faceid/models/det_500m.onnx
/opt/seiyuu-faceid/models/selected/adaface_ir101.onnx
/opt/seiyuu-faceid/models/selected/w600k_r50.onnx
/opt/seiyuu-faceid/models/selected/classifier.npz
/opt/seiyuu-faceid/models/selected/manifest.json
```

リポジトリのディレクトリでロック済み API 依存関係をインストールします。

```bash
sudo env UV_PYTHON_INSTALL_DIR=/opt/seiyuu-faceid/.python \
  UV_CACHE_DIR=/opt/seiyuu-faceid/.uv-cache UV_LINK_MODE=copy \
  UV_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=automatic \
  /usr/local/bin/uv sync --locked --no-dev --extra api --python 3.13
sudo chown -R faceid:faceid /opt/seiyuu-faceid
sudo -u faceid /opt/seiyuu-faceid/.venv/bin/python -c \
  'import cv2, numpy, onnxruntime; from seiyuu_faceid.api.app import create_app'
```

この段階では Python と依存関係をダウンロードします。
管理対象の Python は `.python` に固定し、`.venv` が root だけにアクセス可能な個人キャッシュを参照することを防ぎます。
`UV_LINK_MODE=copy` でインストール済みパッケージがキャッシュ内のリンクに依存することを避けます。
最後にディレクトリの所有者を統一し、サービスユーザーがモデル、設定、依存関係を読み、Python を実行できるようにします。
実行時は `.venv` を直接呼び出し、uv の起動やランタイムの自動ダウンロードは行いません。
関連変数は [uv 公式環境変数リファレンス](https://docs.astral.sh/uv/reference/environment/) にあります。

<a id="启动服务"></a>

## サービスの起動

```bash
sudo install -m 0600 deploy/faceid-native.env.example /etc/seiyuu-faceid.env
sudoedit /etc/seiyuu-faceid.env
```

起動前に `SEIYUU_FACEID_API_KEY` を自分専用の空でないランダムキーへ設定します。
`openssl rand -hex 32` で生成できます。空値は API 認証を無効にします。
環境ファイルは systemd が読むため、`faceid` ユーザーに読み取り権限を与える必要はありません。
記入済みファイルをリポジトリへコミットしないでください。

```bash
sudo install -m 0644 deploy/seiyuu-faceid.service /etc/systemd/system/seiyuu-faceid.service
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
sudo systemctl status seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
```

既定では `127.0.0.1:8000` だけで待ち受けます。
遠隔から呼び出す場合は既存のリバースプロキシを通し、Bearer API キー認証を維持します。
仕様は [API ドキュメント](API.ja.md) を参照してください。
ログは `sudo journalctl -u seiyuu-faceid.service -f` で確認します。
設定や環境ファイルの変更後は `sudo systemctl restart seiyuu-faceid.service` を実行します。
リクエストの正常終了猶予は60秒、systemd の停止タイムアウトは90秒です。

コードと依存関係の更新前にサービスを停止します。
同じディレクトリで上記の `uv sync` と `chown` を再実行し、サービスを起動します。
モデルは引き続きローカルから読み込むため、開発用コンピューターの常時接続は不要です。

<a id="在同一主机比较资源"></a>

## 同一ホストでリソースを比較

ベンチマークとサービスの CPU 競合、およびモデル2組の同時保持を避けるため、先に API を停止します。

```bash
sudo systemctl stop seiyuu-faceid.service
cd /opt/seiyuu-faceid
sudo -u faceid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_target.py --config configs/n150.yaml \
  --synthetic --samples 8 --runs 5 --warmups 2 --repeats 2 \
  --output /tmp/n150-native-benchmark.json > /tmp/n150-native-stdout.json
sudo systemctl start seiyuu-faceid.service
```

ツールは未作成の出力パスを要求します。再計測ごとに `/tmp` の2つのファイル名を変更します。
標準出力は完全な JSON で、`/tmp/n150-native-stdout.json` に保存します。
`--output` はサービスユーザーが書き込むコピーも保存します。
このベンチマークに API キーは不要で、学習ディレクトリも読みません。
位置合わせ済みクロップの特徴抽出と分類を測定します。
検出器は読み込まれますが実行しません。実人物の精度や HTTP リクエスト全体の所要時間を表すものではありません。

ネイティブと Docker を比較する場合は、同じマシン、モデル、設定、入力、Python と ONNX Runtime のバージョン、スレッド環境変数を使います。
競合を避けるため2つの方式を順番に実行します。
レポートのバージョン、CPU 時間、wall time、ピーク RSS を確認し、実際の API リクエストとサービス待機時メモリも別途測定します。
コンテナー全体のメモリとプロセス RSS は集計範囲が異なるため、差を Docker の負荷として直接扱えません。
ローカル macOS の結果から N150、Linux、Docker の性能差を推定しないでください。
