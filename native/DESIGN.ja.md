# C++ コア + Go HTTP 実験

[English](DESIGN.md) | [简体中文](DESIGN.zh-CN.md) | [日本語](DESIGN.ja.md)

利用者の承認を受けた Intel N150 / 8 GB Linux 向けの試作です。Python を参照実装として維持します。実行時には Python を使わず、単一の Go プロセスが cgo 経由で C++ 共有ライブラリを呼びます。3つの常駐 ONNX Runtime CPU Session と1つのモデルを使い、リクエストを直列化しながら両認識バックボーンを並列実行します。push と Release 公開は行いません。FTZ/DAZ 実装は対象機の一致性と性能確認に合格し、2026-10-28 に N150 サービスへ採用されました。

## インターフェース

`include/faceid.h` は C の所有権とエラーの境界です。Go はエンジン作成前にモデルのチェックサムを検証します。C++ は次元、有限パラメーター、設定を検証し、すべての画像処理、推論、スコア計算を実行します。

`tools/export_native_model.py` は既存 NPZ パラメーターを `models/selected/native-model.json` に変換します。Python が必要なのはこの準備だけです。ONNX と分類器の元ファイルは変更しません。JSON schema の版は 1 です。

- `model_id` は Python のキーをソートした資産 JSON の SHA256 で、Python と同一です。
- `source_assets` は元のマニフェストのファイル名と SHA256 の対応表です。`detector_sha256` は検出器の SHA256 です。
- `models` は JSON からの相対パスである `detector`、`adaface`、`r50` を含みます。
- `class_names` は順序付き文字列配列です。
- `head` は float64 の `mean` / `scale` と、float32 の `coefficients`（行）/ `intercept` を含みます。
- `runtime` は `intra_op_threads`、`detector_intra_op_threads`、`inter_op_threads`、`parallel_backbones`、`allow_spinning`、`enable_cpu_mem_arena`、省略可能な `flush_denormals` を含みます。後者は厳密な bool で、欠落時は false です。対応 Linux x86_64 だけで FTZ/DAZ を有効化し、Session 作成と各推論呼び出しの前後で呼び出し元 MXCSR を完全復元します。ORT worker プールも設定します。
- `detector` は `input_size` [width,height]、`max_long_edge`、`score_threshold`、`nms_threshold` を含みます。
- `quality` は `min_face_size`、`min_sharpness` を含みます。
- `recognition` は `top_k`、`probability_threshold` を含みます。

`faceid_model` は `model_id`、暫定 `recognition_id` マーカー（`native-v1:` + model ID）、`class_count`、`class_names`、`defaults` を返します。Go は `recognition_id` を `native-v1\n` + モデル JSON の生バイト列の SHA256 に置き換え、全設定と検出器チェックサムを反映し、アップロード制限と形式も追加します。JSON の空白だけを変えても HTTP 認識 ID が変わります。

## 範囲と検証

Go はヘルス、準備完了、認証付きモデル情報、検出、認識、バッチ認識を実装します。Multipart JPEG/PNG/WebP、1ファイル 10 MiB、1リクエスト 81 MiB、1画像 2000 万画素、1バッチ 8 枚です。パラメーター検証、待機中キャンセル、単一推論、正常終了を実装します。UI/OpenAPI は初期試作の範囲外です。EXIF の向きを維持し、アニメーションを拒否します。

モデル破損、不正アップロード、認証、同時リクエスト、終了時所有権を検査します。過去のホールドアウト全 1,051 枚の embedding、確率、top-1/top-5、0.95 判定を比較し、元写真では検出、アライメント、デコードも比較します。重み不変だけを結果不変の根拠にはしません。Python とネイティブは別プロセスで、入力、予熱、スレッド設定を固定し、反復測定します。ローカル ARM 時間で N150 時間を予測できません。私有テスト入力と結果は無視対象データディレクトリに置きます。

## 作業分担

1. コアは C++ パイプラインと C ABI、CMake ライブラリ、小さな数値およびエラーテストを担当します。
2. HTTP は Go net/http サービス、cgo アダプター、API 動作テストを担当します。
3. 統合は再現可能な依存とビルド、モデル出力、Python/ネイティブ一致性と性能測定、Linux デプロイ説明を担当します。
