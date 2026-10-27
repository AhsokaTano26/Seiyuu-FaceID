# Reproduce the selected classifier

Only AdaFace IR101 + R50 logistic regression is retained. Frozen feature extraction and source-group-disjoint data are stored locally in `data/training/`. The 3,574 training and 490 validation faces fit the final head; the 1,051 held-out faces are excluded.

```bash
uv run --extra training python tools/train_classifier.py --output /tmp/classifier.npz
uv run python tools/evaluate.py
uv run python tools/evaluate.py --reextract
```

The training command uses the selected C=1, balanced class weights and random seed 20261014. It requires sklearn 1.9.1 and writes only to a new output path. It does not overwrite the deployed model. Runtime uses NumPy and ONNX Runtime, without sklearn or PyTorch.

For new photos or identities, review the labels and prepare a new data split before training. The sample-management UI only prepares data; it does not silently modify the deployed classifier.

## 分類ヘッドの比較

```bash
uv run --extra training python tools/optimize_classifier.py --output data/experiments/head-search
uv run python tools/benchmark_classifier.py --threads 1 2 4 --output data/experiments/backbone-speed.json
```

出力には未使用のパスを指定します。元の訓練＋検証データを出典グループ単位で三分割し、固定の 50 候補を比較します。出典グループが一つしかない人物は訓練にのみ使用し、OOF 指標には含めません。Top-1、対応クラスの Macro-F1、対数損失の順に候補を選び、選択を保存してから別の開発分割と過去のテスト集合で確認します。結果を見て候補を選び直すことはありません。既存モデルを上書きせず、独立した `classifier.npz` と実験記録を保存します。

過去のテスト集合は以前のモデル選択に使われており、二つ目の開発分割も同じ標本を再利用します。独立した汎化評価ではありません。10 月 27 日の比較では候補が過去のテスト集合で悪化したため、既存ヘッドを維持しました。速度改善は二つの固定バックボーンの並列実行で行います。[測定結果](../docs/CLASSIFIER_OPTIMIZATION.md)を参照してください。
