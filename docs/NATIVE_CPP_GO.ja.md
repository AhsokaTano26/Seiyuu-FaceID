# C++ 推論コア + Go HTTP

[English](NATIVE_CPP_GO.md) | [简体中文](NATIVE_CPP_GO.zh-CN.md) | [日本語](NATIVE_CPP_GO.ja.md)

この版は Intel N150 / 8 GB でネイティブ実行系の効果を検証するためのものです。1つの Go プロセスが cgo 経由で C++ を直接呼び出し、SCRFD、AdaFace IR101、R50 を1回だけ読み込みます。両バックボーンは並列、HTTP 推論リクエストは直列に実行します。モデルは FP32、既存の分類ヘッドと既定の 0.95 しきい値を維持します。

アルゴリズム、学習評価、各モジュールの連携は [認識技術ガイド](TECHNICAL_ARCHITECTURE.ja.md) を参照してください。

N150 では Linux ビルドと起動、1,051 枚の切り抜きと 12 枚の元画像の一致性、完全な3組の HTTP ベースラインを確認しました。同設定のネイティブ版は確かな高速化を示さず、ピーク RSS は増加しました。省 CPU スレッド設定の平均 HTTP 時間は旧スレッド組より約 37.9% 増えました。範囲全体と後続の独立実験は [N150 実機レポート](N150_BENCHMARK.ja.md) にあります。2026-10-28 に対象機の既存 API を FTZ/DAZ 版へ切り替えました。長期稼働と負荷の受け入れ確認は残っています。

後の独立した FTZ/DAZ 候補は全件回帰と HTTP 測定を完了し、平均リクエスト時間は約 0.681 秒、過去のホールドアウト Top-1 は 98.48% を維持しました。この実装は `native/` に統合済みで、ビルド後のモデル出力時に `--flush-denormals` を明示して有効化します。Linux x86_64 と FTZ/DAZ 対応 CPU が必要で、Mac ARM は明示的に拒否します。省略時は既定で無効となり、既存 JSON と互換です。結果、メモリのトレードオフ、段階間比較の制約は [FTZ 実験](N150_BENCHMARK.ja.md) を参照してください。

デプロイには `bin/faceid-server`、隣接する `lib/` 共有ライブラリ、モデルディレクトリ、メタデータが必要です。完全静的な単一ファイル構成ではありません。提供時に Python は不要ですが、モデルパラメーターの出力と比較検証には使います。

<a id="build"></a>

## ビルド

**Linux x86_64** または macOS arm64 の実機でビルドできます。Mac の成果物を Linux 実行ファイルとしては扱えません。C++17 コンパイラー、Make、CMake ≥ 3.20、Go ≥ 1.24、curl、tar、SHA-256 ツールが必要です。Linux では `build-essential cmake curl ca-certificates` と、版条件を満たす Go を導入できます。

```bash
bash native/build.sh
```

スクリプトは ONNX Runtime 1.29.0、OpenCV 5.0.0、nlohmann/json 3.12.0 を固定して取得し、SHA-256 を検証します。必要な OpenCV モジュールと C++ コアをビルドし、CTest と Go 単体テストを実行して `dist/native-runtime/` を生成します。既定は 4 並列で、メモリが少ない場合は `JOBS=2` を使います。OpenCV が追加のプラットフォーム最適化依存を取得する場合もあり、ネットワークが必要です。依存と成果物は無視対象ディレクトリ内に置きます。依存の構築後にアプリコードだけを変更した場合は `bash native/build.sh --app-only` で OpenCV の構成とコンパイルを省けます。依存版やビルドオプションを変えた場合は完全ビルドを実行します。

```bash
uv run python tools/export_native_model.py --config configs/n150.yaml
```

N150 には検証済みの高速設定を推奨します。既存の出力ファイルは上書きしません。

```bash
uv run python tools/export_native_model.py --config configs/n150.yaml \
  --flush-denormals --output models/selected/native-n150.json
```

起動時の `-model` は対応する JSON を指定します。このフラグは ORT セッションスレッドと各 C++ 呼び出しの FTZ/DAZ を設定し、完了時や例外時にスレッドの浮動小数点状態を復元して Go のスケジューリングに対応します。重み、分類ヘッド、0.95 しきい値は維持します。非正規化数の動作が変わりますが、FP16 や INT8 へ変換しません。1,051 枚の切り抜きで精度と判定は一致したものの、すべての未知入力のビット一致を保証しません。

準備処理は元モデルと分類器のチェックサムを検証し、既存 NPZ パラメーターを損失なく `models/selected/native-model.json` へ変換して `.sha256` を生成します。ONNX の重みは不変です。JSON のモデルパスは自身の位置からの相対パスです。デプロイ時は次の配置を維持します。

```text
models/
  det_500m.onnx
  selected/
    adaface_ir101.onnx
    w600k_r50.onnx
    classifier.npz
    native-model.json
    native-model.json.sha256
```

`classifier.npz` は元分類器の出典確認に使い、実行時に Python で読みません。Go は起動時に JSON のチェックサム、元モデルのハッシュ、モデル識別子を検証し、不合格なら終了します。チェックサムは破損検出用であり、モデルパッケージの電子署名認証ではありません。設定変更後は JSON を再出力します。

<a id="startup-and-api"></a>

## 起動と API

```bash
export SEIYUU_FACEID_API_KEY='replace-with-your-key'
dist/native-runtime/bin/faceid-server \
  -model models/selected/native-model.json \
  -addr 127.0.0.1:8000
```

API キーの環境変数は既存サービスと同じです。未設定なら認証を要求しません。既定ではローカルで待ち受けます。エンドポイントとアップロードフィールドは [API 文書](API.ja.md) にあります。ルートは `/health`、`/ready`、`/v1/model`、`/v1/detect`、`/v1/recognize`、`/v1/recognize/batch` です。この試作版にはトップ画面、Swagger、ReDoc、OpenAPI ルートはありません。認識フィールドは既存 API を継承し、エラー詳細の文言は異なる場合があります。

制限は1ファイル 10 MiB、1リクエスト 81 MiB、1画像 2000 万画素、1バッチ 8 枚です。静止 JPEG、PNG、WebP に対応し、EXIF の向きを適用します。アップロード解析、待機、処理中の推論リクエストを合計 8 件まで受け入れ、超過時は `503 server_busy`、待機キャンセルは `408 request_cancelled` を返します。開始済みのネイティブ推論は完了後にリソースを解放します。終了時は現在の呼び出し完了後にモデルを破棄します。1フレームを含むすべての APNG を拒否します。

multipart ファイル内容は 32 KiB のメモリしきい値を超えると一時ディレクトリへ書き出し、リクエスト終了時に削除するため、一時ディスク領域が必要です。`processing_ms` はファイル読み取り、検証、デコード、推論を含み、multipart 解析と待機は除外します。完全リクエストはクライアントの HTTP 時間を別途測定します。HTTP の正常終了待機枠は 90 秒ですが、ネイティブ呼び出し終了までモデルを破棄できず、プロセス全体の厳密な 90 秒以内の終了は保証しません。

`model_id` は元モデルと同じです。`recognition_id` はネイティブ実装と全設定を反映し、実装や設定を区別します。しきい値と候補数はリクエスト単位で上書きできます。

```bash
curl --fail http://127.0.0.1:8000/ready
curl --fail-with-body http://127.0.0.1:8000/v1/recognize \
  -H "Authorization: Bearer $SEIYUU_FACEID_API_KEY" -F 'file=@photo.jpg'
```

対象機の HTTP 遅延測定には Python は不要で、成果物に `faceid-bench` を含みます。

```bash
dist/native-runtime/bin/faceid-bench -url http://127.0.0.1:8000 \
  -image ./photo.jpg -warmups 3 -runs 20
```

同じ API キー環境変数を読み、HTTP 平均 / P50 / P95 とサーバー処理時間の平均を出力します。人物、ファイルパス、応答内容は表示せず、遅延を CPU やメモリの指標として扱いません。Python とネイティブを比較する際は別々に起動し、同じ画像とパラメーターを使ってモデル2個が CPU を奪い合わないようにします。

<a id="parity-validation"></a>

## 一致性検証

モデルが不変でも出力の一致は保証されません。デコード、検出、5点アライメント、正規化、線形ヘッド、候補順序、受理しきい値を確認します。特に JPEG デコードとプラットフォームごとの数値実装は差を生む可能性があります。

```bash
# Linux は .so、macOS は .dylib を使います。
uv run --extra api python tools/validate_native.py \
  --library dist/native-runtime/lib/libfaceid_core.so \
  --model models/selected/native-model.json --config configs/n150.yaml \
  --crop-dir /path/to/aligned-112-crops \
  --image-dir /path/to/original-images --encoded --output /tmp/native-parity.json
```

検証ツールは明示されたディレクトリだけを読み、既定で学習写真を走査しません。確率差と判定一致を集計し、写真、人物一覧、特徴は保存しません。Python と C++ を同時に読み込むため、このプロセスメモリはネイティブサービスのメモリではありません。資源比較では独立した Python と Go サービスを起動し、入力、スレッド設定、予熱、リクエスト順をそろえて遅延と RSS を測定します。過去のホールドアウトはモデル選択に使われ、新たな独立汎化保証ではありません。N150 では Linux ビルド、実起動、全切り抜きと固定元画像の一致性、3組の HTTP ベースラインを完了しました。[実機レポート](N150_BENCHMARK.ja.md) に個別掲載し、以下の Mac 記録は元のプラットフォームと入力範囲を保ちます。

検出同点には明確な境界があります。C++ NMS は入力索引の逆順、Python は NumPy 既定の不安定ソートを使います。重なる矩形が同点なら別のランドマークを選ぶ可能性があり、標本回帰で任意画像のビット一致を主張できません。既定実行系への採用には、対象機と実際の業務画像による受け入れ確認が必要です。

<a id="local-regression"></a>

## ローカル回帰記録

macOS ARM64 で Python と C++ が別プロセスに各 ONNX Runtime 1.29.0 を読み込み、同一プロセスでの動的ライブラリ再利用による影響を避けました。過去のホールドアウト全 1,051 枚から特徴を再抽出しました。

- Top-1、Top-5 順序、0.95 受理判定が全件一致し、新たな誤認はありませんでした。
- 両方の精度は 1,035 / 1,051 = **98.48%** でした。
- 特徴の最大絶対差は `2.24e-8`、分類確率は `3.58e-7` で、ビット単位では同一ではありません。
- 別の元画像 12 枚 / 顔 14 件で、生 BGR とアップロードファイル直接デコードの両経路が合格しました。矩形、検出スコア、品質スコア、候補順序、受理判定は一致し、確率最大差は `2.09e-7` でした。

ローカルの私有詳細は無視対象の `data/experiments/native-cpp-go-20261027/` に置き、モデル Release へ含めません。Python 全 206 テスト、C++ CTest、Go 単体と race テスト、実モデル HTTP 統合テストが合格しました。Mac ビルドディレクトリを別の場所に複製して動的ロードも確認しました。これらは Mac 段階の記録です。後続の Linux ビルド、N150 全件一致性、HTTP ベースラインは [実機レポート](N150_BENCHMARK.ja.md) を参照してください。対象機で実行したテスト範囲を別記し、206 テストや race テストを自動的に含めません。

<a id="local-http-resources"></a>

## ローカル HTTP 資源比較

同じ Mac で独立した Python / Go サービスを N150 設定で起動し、別に選んだ元画像 12 枚を使いました。1巡予熱後、4巡、計 48 リクエストを計時しました。各実装で新プロセスを3回、交互順で測定し、表は3プロセスの中央値です。両サービスは同時常駐させていません。

| 指標 | Python HTTP | C++ + Go HTTP | 変化 |
| --- | ---: | ---: | ---: |
| 平均 HTTP 時間 | 94.13 ms | 85.00 ms | 9.7% 減 |
| HTTP P95 | 113.83 ms | 103.15 ms | 9.4% 減 |
| 平均サーバー処理時間 | 92.14 ms | 82.69 ms | 10.3% 減 |
| プロセスのピーク RSS | 1160.89 MiB | 1026.94 MiB | 133.95 MiB / 11.5% 減 |

HTTP 3回分の出力で検出、品質判定、候補順序、受理結果が一致し、確率最大差は `2.38e-7` でした。HTTP 時間はローカルのアップロードと応答を含み、P95 は nearest-rank、サーバー処理時間はデコードと推論を含みます。ピーク RSS は各プロセス終了後の `wait4` から取得し、起動、モデル読み込み、大画像デコードを含みます。切り抜き分類だけの旧基準と混用できません。実行系全体の移行効果にはデコードとデータ処理経路の変更を含み、すべてを言語へ帰属できません。N150 の時間、メモリ、消費電力の予測値でもありません。

成果物とデプロイ資料はリポジトリのワークフローに置き、GitHub モデル Release の添付物にはしません。

<a id="standard-dataset-retest"></a>

## 公開標準データセットでの再測定

2026-10-28 にローカル M4 Pro / 24 GiB で、LFW 512 枚と WIDER FACE validation 244 枚の固定標本を3回比較し、計 9,072 HTTP リクエストを計時しました。完全認識の C++/Go 平均はそれぞれ 58.82 / 75.33 ms で、Python より 1.83% / 2.67% 減りました。検出のみは約 7.9% / 14.7% 減、ピーク RSS 中央値は 3.8% 減でした。全出力一致性検査が合格しました。

この Mac 測定は FTZ/DAZ を無効にし、N150 とは別の範囲です。LFW 検証精度や WIDER 検出 AP を測った値ではありません。入力、負荷、統計手法、元の根拠は [標準データセット測定レポート](STANDARD_DATASET_BENCHMARK.ja.md) にあります。

後の [全精度評価](STANDARD_DATASET_ACCURACY.ja.md) では、LFW 6,000 ペアの10分割融合検証精度が 99.13%、WIDER 完全検証集合の Easy / Medium / Hard AP（×100）が 85.49 / 79.48 / 48.84 でした。元の声優ホールドアウトは 98.48% を維持し、Python と C++ の指標とペア判定は一致しました。

<a id="linux-systemd-部署"></a>
<a id="linux-systemd"></a>

## Linux systemd デプロイ

テンプレートは [`native/deploy/seiyuu-faceid.service`](../native/deploy/seiyuu-faceid.service) です。`DynamicUser`、読み取り専用ファイルシステム、専用一時ディレクトリを使い、提供時に Python や Docker は不要です。テンプレートは `0.0.0.0:8000` を使い、必要に応じてローカルアドレスに変えられます。root 所有の `/etc/seiyuu-faceid/api.env` に `SEIYUU_FACEID_API_KEY=existing-key` を置き、権限を `0600` にしてリポジトリへ含めません。事前にポートが空いていることを確認します。

以下の配置を使います。モデルと実行系は主ディスクに置き、サービスユーザーは全ディレクトリの読み取りと通過、プログラム実行が可能な状態にします。その他のファイルは読み取り専用にし、プログラムやモデルへの書き込み権限を与えません。

```text
/opt/seiyuu-faceid/
  current -> releases/<version>
  releases/<version>/
    runtime/bin/faceid-server
    runtime/lib/...
    runtime/licenses/...
    models/det_500m.onnx
    models/selected/{adaface_ir101.onnx,w600k_r50.onnx,classifier.npz}
    models/selected/native-model.json
    models/selected/native-model.json.sha256
```

`native-model.json` は `--flush-denormals` で出力した高速設定です。改名後は JSON バイト列と全相対モデルパスを保ち、新ファイル名に対応する SHA-256 sidecar を再生成します。配置、キーファイル、`current` リンクを準備してから実行します。

```bash
sudo install -m 0644 native/deploy/seiyuu-faceid.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/seiyuu-faceid.service
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuu-faceid.service
curl --fail http://127.0.0.1:8000/ready
sudo systemctl status seiyuu-faceid.service
```

今回の N150 切り替えは、全件検証済みのコア、Go 実行ファイル、設定バイト列を再利用し、Mac 成果物を Linux へ転送しませんでした。一時ポートで検証後、旧コンテナーを停止して自動再起動を無効化し、元ポートで新サービスを有効化しました。API キー、他のコンテナー、旧モデルのマウントは維持し、旧コンテナーを復旧用に残しました。稼働後に元画像 12 枚 / 顔 14 件を再確認しました。[デプロイ受け入れ記録](N150_BENCHMARK.ja.md) を参照してください。

ネイティブサービスは前述の API のみを提供し、`/docs`、`/redoc`、アップロードトップ画面は含みません。フィールドはリポジトリの [API 文書](API.ja.md) を参照してください。旧デプロイに Swagger があっても自動継承しません。

このサービスから旧 Docker デプロイへ戻す場合は、ポート競合を避けるため先に systemd サービスを無効化します。

```bash
sudo systemctl disable --now seiyuu-faceid.service
docker update --restart=unless-stopped <original-container-name>
docker start <original-container-name>
curl --fail http://127.0.0.1:8000/ready
```
