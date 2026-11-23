# 既定分類器の再現

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

再現可能な既定分類ヘッドは AdaFace IR101 + R50 とロジスティック回帰を使用します。固定特徴量とソースグループを分離したデータはローカルの `data/training/` にあります。最終ヘッドは学習3,574件と検証490件で学習し、ホールドアウト1,051件を別に保持します。

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
```

学習コマンドは C=1、クラス均衡重み、乱数シード20261014を使用します。sklearn 1.9.1 が必要で、新しい出力先へ書き込みます。配備モデルは別途管理します。実行時は NumPy と ONNX Runtime を使用します。

写真や人物を追加する際は、ラベルを確認して新しいデータ分割を準備した後に学習します。ウェブ画面のサンプル管理はデータ準備を担当し、配備する分類ヘッドは別途更新します。

## 分類ヘッドの比較実験

```bash
uv run --extra training python tools/optimize_classifier.py --output data/experiments/head-search
uv run python tools/benchmark_classifier.py --threads 1 2 4 --output data/experiments/backbone-speed.json
```

未作成の出力先を指定してください。探索では元の学習・検証プール内の固定50構成を、ソース成分が重ならない3分割で検証します。成分が一つしかない人物は常に学習に含め、OOF指標から除きます。統合Top-1、真値が存在するクラスのMacro-F1、対数損失の順で選択します。選定構成を保存した後、別の開発分割と過去のテスト集合で一度比較し、選定構成は固定します。成果物はプロトコル、分割インデックス、確率、独立した実行時互換の `classifier.npz` です。

過去のテスト集合は以前のモデル選択で再利用されており、二つ目の開発分割も同じサンプルを使います。独立した精度推定には新しいテスト集合が必要です。10月27日の実験では候補の過去テスト成績が低下したため、元のヘッドを継続使用しました。実行時の改善は二つの固定バックボーンの並列処理によるものです。[実測報告](../docs/CLASSIFIER_OPTIMIZATION.ja.md)を参照してください。

606クラス公開版は別の固定データを使用します。件数、分類ヘッドのチェックサム、ローカル証拠のパス、制約は[606クラス評価報告](../docs/MODEL_ACCURACY_20261103.ja.md)を参照してください。過去の写真整理により既定の顔画像が欠けている場合があります。特徴量アーカイブがあれば特徴量に基づく評価が可能で、再抽出には対応する顔画像が必要です。
