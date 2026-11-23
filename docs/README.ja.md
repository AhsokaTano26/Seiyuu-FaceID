# ドキュメント索引

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

この索引はリポジトリで管理する Markdown 文書を対象にしています。上部のリンクで英語、簡体字中国語、日本語を切り替えられます。英語は `.md`、中国語は `.zh-CN.md`、日本語は標準言語コードの `.ja.md` を使用します。

ローカルの既定モデルと CI が固定する取得元は50クラスです。別途公開した606クラスモデルには固有の報告とデータセットがあります。指標を読む際は分類ヘッドのチェックサムと評価プロトコルを確認してください。

第三者の依存ディレクトリ、生成済みの配布物、非公開実験の成果物は、それぞれのファイルに関する表記と由来を保持します。過去の実装計画には記載日付の判断を保存しています。以下のガイドは管理対象のインターフェースと配備手順を説明します。

## 利用の開始

- [プロジェクト概要](../README.ja.md)
- [HTTP API](API.ja.md)
- [認識アーキテクチャ](TECHNICAL_ARCHITECTURE.ja.md)
- [ネイティブ実行環境](../native/README.ja.md)
- [ネイティブ設計](../native/DESIGN.ja.md)

## 配備と公開

- [CI/CD、デプロイ成果物とモデル公開のトリガー](CI_CD.ja.md)
- [Dockerとオフライン配備](DEPLOY_LINUX.ja.md)
- [Python/systemd配備](DEPLOY_NATIVE_LINUX.ja.md)
- [C++/Goの配備と検証](NATIVE_CPP_GO.ja.md)
- [N150の設定と計測](N150.ja.md)
- [ローカル配布バンドル](../deploy/README.ja.md)
- [CI Dockerバンドル手順](../deploy/ci/docker.ja.md)
- [CI Pythonバンドル手順](../deploy/ci/python.ja.md)
- [CIネイティブバンドル手順](../deploy/ci/native.ja.md)
- [モデル公開の入力](../release/README.ja.md)

## データと学習

- [ローカルデータ配置](../data/README.ja.md)
- [モデルの出典と版](../models/README.ja.md)
- [分類器の再現と探索](../training/README.ja.md)

## 評価と性能

- [606クラスモデルの精度](MODEL_ACCURACY_20261103.ja.md)
- [過去の50クラス分類ヘッド比較](CLASSIFIER_OPTIMIZATION.ja.md)
- [N150の基準値とFTZ/DAZ実測](N150_BENCHMARK.ja.md)
- [LFW/WIDER FACE実行性能比較](STANDARD_DATASET_BENCHMARK.ja.md)
- [LFW/WIDER FACE精度評価](STANDARD_DATASET_ACCURACY.ja.md)
- [図のソースと再現](figures/README.ja.md)

## 過去の実装計画

- [HTTP API実装、2026-10-21](superpowers/plans/2026-10-21-http-api.ja.md)
- [サンプル数実験、2026-10-21](superpowers/plans/2026-10-21-sample-size.ja.md)
- [分類器最適化、2026-10-27](superpowers/plans/2026-10-27-classifier-optimization.ja.md)
- [N150実行環境計画、2026-10-27](superpowers/plans/2026-10-27-n150-runtime.ja.md)
- [CI配布計画、2026-10-28](superpowers/plans/2026-10-28-ci-deployments.ja.md)

単独の[顔検出 OpenAPI 定義](detect.openapi.yaml)を API クライアントへインポートできます。プロジェクト規約は [AGENTS.md](../AGENTS.md) にあります。
