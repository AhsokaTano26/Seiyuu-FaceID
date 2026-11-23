# 図のソースと再現

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

[draw_documentation_figures.py](../../tools/draw_documentation_figures.py) の Python 関数が Matplotlib で図を描き、[技術ガイド](../TECHNICAL_ARCHITECTURE.ja.md)と[標準データセット精度報告](../STANDARD_DATASET_ACCURACY.ja.md)で使用します。各図は SVG、300 dpi PNG、PDF を提供し、文書には拡大可能な SVG を埋め込みます。

## 再現

リポジトリのルートで実行します。

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_documentation_figures.py
```

指定した図のみを再描画する場合は次を実行します。

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_documentation_figures.py --only wider_pr_curves lfw_fold_accuracy
```

描画は集計データを読み、専用の描画依存関係を使用します。入力を集計に限定し、モデル推論・学習写真・生の顔特徴量は処理対象外です。[aggregate_metrics.json](accuracy/aggregate_metrics.json) に集計指標、10分割統計、PR曲線、元報告・独立再計算・曲線ファイルのチェックサムを保存しています。評価の由来は[精度報告](../STANDARD_DATASET_ACCURACY.ja.md)を参照してください。

## 図の一覧

関数は `draw_<ファイル名>()` の形式で Matplotlib Figure を返します。PNG と PDF は SVG と同じファイル名本体を使用します。

| 文書 | 内容 | SVG | PNG | PDF |
| --- | --- | --- | --- | --- |
| アーキテクチャ | 認識と要確認の分岐 | [SVG](architecture/recognition_pipeline.svg) | [PNG](architecture/recognition_pipeline.png) | [PDF](architecture/recognition_pipeline.pdf) |
| アーキテクチャ | SCRFDの複数スケール出力と重複抑制 | [SVG](architecture/scrfd_detector.svg) | [PNG](architecture/scrfd_detector.png) | [PDF](architecture/scrfd_detector.pdf) |
| アーキテクチャ | 5点位置合わせの幾何 | [SVG](architecture/five_point_alignment.svg) | [PNG](architecture/five_point_alignment.png) | [PDF](architecture/five_point_alignment.pdf) |
| アーキテクチャ | 色順序・特徴次元・融合分類 | [SVG](architecture/feature_fusion.svg) | [PNG](architecture/feature_fusion.png) | [PDF](architecture/feature_fusion.pdf) |
| アーキテクチャ | オフライン学習とパラメータ出力 | [SVG](architecture/training_pipeline.svg) | [PNG](architecture/training_pipeline.png) | [PDF](architecture/training_pipeline.pdf) |
| アーキテクチャ | 学習と過去ホールドアウトの範囲 | [SVG](architecture/data_split.svg) | [PNG](architecture/data_split.png) | [PDF](architecture/data_split.pdf) |
| アーキテクチャ | FTZ/DAZ浮動小数点状態の範囲 | [SVG](architecture/denormal_scope.svg) | [PNG](architecture/denormal_scope.png) | [PDF](architecture/denormal_scope.pdf) |
| アーキテクチャ | Go・C ABI・C++の役割 | [SVG](architecture/native_runtime.svg) | [PNG](architecture/native_runtime.png) | [PDF](architecture/native_runtime.pdf) |
| アーキテクチャ | 二つの骨格の並列処理と合流 | [SVG](architecture/inference_parallel.svg) | [PNG](architecture/inference_parallel.png) | [PDF](architecture/inference_parallel.pdf) |
| アーキテクチャ | 待機枠・推論ゲート・取消境界 | [SVG](architecture/request_lifecycle.svg) | [PNG](architecture/request_lifecycle.png) | [PDF](architecture/request_lifecycle.pdf) |
| アーキテクチャ | 資源検証とモデル起動 | [SVG](architecture/model_startup.svg) | [PNG](architecture/model_startup.png) | [PDF](architecture/model_startup.pdf) |
| 精度 | 三つのタスクと指標 | [SVG](accuracy/benchmark_tasks.svg) | [PNG](accuracy/benchmark_tasks.png) | [PDF](accuracy/benchmark_tasks.pdf) |
| 精度 | LFWの10分割しきい値選択とテスト | [SVG](accuracy/lfw_protocol.svg) | [PNG](accuracy/lfw_protocol.png) | [PDF](accuracy/lfw_protocol.pdf) |
| 精度 | LFWの正解・誤判定・未検出件数 | [SVG](accuracy/lfw_errors.svg) | [PNG](accuracy/lfw_errors.png) | [PDF](accuracy/lfw_errors.pdf) |
| 精度 | LFWの分割ごとの実測精度 | [SVG](accuracy/lfw_fold_accuracy.svg) | [PNG](accuracy/lfw_fold_accuracy.png) | [PDF](accuracy/lfw_fold_accuracy.pdf) |
| 精度 | WIDERの対応付け・除外・AP計算 | [SVG](accuracy/wider_protocol.svg) | [PNG](accuracy/wider_protocol.png) | [PDF](accuracy/wider_protocol.pdf) |
| 精度 | 三難易度の実測PR曲線 | [SVG](accuracy/wider_pr_curves.svg) | [PNG](accuracy/wider_pr_curves.png) | [PDF](accuracy/wider_pr_curves.pdf) |
| 精度 | 声優顔画像の採用・要確認の誤り内訳 | [SVG](accuracy/holdout_decisions.svg) | [PNG](accuracy/holdout_decisions.png) | [PDF](accuracy/holdout_decisions.pdf) |
| 精度 | 真値精度・実装一致・独立再計算 | [SVG](accuracy/validation_evidence.svg) | [PNG](accuracy/validation_evidence.png) | [PDF](accuracy/validation_evidence.pdf) |

## 図の範囲と検証

- 作成環境では中国語に SimSun、英語と通常数字に Times New Roman、数式に Computer Modern MathText を使用しました。SVG は字形輪郭、PDF はフォントサブセットを埋め込みます。フォント本体は別途管理します。指定フォントがない場合、明示的な候補順で選び実際のフォントを表示します。
- 図中の位置と処理段階は模式的です。矢印はデータや呼び出しの依存を示します。箱の寸法や間隔にはモデル量・サンプル比・時間の数量的な尺度を持たせていません。
- 5点図の右は実際の ArcFace テンプレート、左は既知の相似変換から合成した点です。両方とも等縮尺と左上原点を使います。
- LFW は10分割すべてを平滑化せず表示し、縦軸は97%–100.25%の部分範囲です。WIDER PR は1,000しきい値すべてを保持し、実測最大再現率で終わります。
- スクリプトは文字境界、字形網羅、10分割平均、再現率の単調性、PR積分とAPの一致を確認します。出力時に欠字警告はありません。別のレンダリングで元サイズ、グレースケール、PDF埋込フォントを確認しました。

[manifest.json](manifest.json) に関数、依存版、フォント、データとソースのチェックサムがあります。図はリポジトリ文書と管理し、モデル専用Releaseの規約に従います。

## 2026-11-03のモデル精度報告

[606クラス報告](../MODEL_ACCURACY_20261103.ja.md)は [draw_model_accuracy_report.py](../../tools/draw_model_accuracy_report.py) を使います。主モデルは `models-5cc19e9d5b1a30aa`、過去の50クラス参照モデルは `models-ca1a34fb7a63a38f` です。各版は固有のテスト集合を使用し、図はそれぞれの記述結果を示します。

| 内容 | PNG | PDF | SVG |
| --- | --- | --- | --- |
| 各版のホールドアウト記述指標 | [PNG](model_accuracy_20261103/model_version_metrics.png) | [PDF](model_accuracy_20261103/model_version_metrics.pdf) | [SVG](model_accuracy_20261103/model_version_metrics.svg) |
| 606クラスの判定と誤り内訳 | [PNG](model_accuracy_20261103/model606_threshold_outcomes.png) | [PDF](model_accuracy_20261103/model606_threshold_outcomes.pdf) | [SVG](model_accuracy_20261103/model606_threshold_outcomes.svg) |
| 606クラスモデルの人物別サンプル数 | [PNG](model_accuracy_20261103/model606_sample_support.png) | [PDF](model_accuracy_20261103/model606_sample_support.pdf) | [SVG](model_accuracy_20261103/model606_sample_support.svg) |

リポジトリのルートで再現します。

```bash
uv run --no-project --python 3.13 \
  --with matplotlib==3.10.9 --with numpy==2.5.4 \
  python tools/draw_model_accuracy_report.py
```

入力の [aggregate_metrics.json](model_accuracy_20261103/aggregate_metrics.json) は版、指標、件数、区間集計、証拠のチェックサムを含みます。各図は400 dpi PNGと二つのベクトル形式で、SimSunとTimes New Romanを使用します。ソース、フォント、図のチェックサムは [figure_manifest.json](model_accuracy_20261103/figure_manifest.json)、元の検証は [quality_review.json](model_accuracy_20261103/quality_review.json) にあります。

品質レビューは文書を多言語化する前の記録です。報告チェックサムは元の文書バイト列を識別します。翻訳Markdownには別の文章を使用し、記録済みの図とデータのチェックサムは過去の由来を保持します。
