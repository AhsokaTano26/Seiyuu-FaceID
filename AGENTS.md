# 项目约定

[English](#en) | [简体中文](#zh-CN) | [日本語](#ja)

<a id="zh-CN"></a>

## 文档语言

仓库维护的 Markdown 文档提供英文、简体中文和日文。英文使用 `.md`，中文使用 `.zh-CN.md`，日文使用 `.ja.md`。每份文档在标题后链接三个语言版本，保持命令、API字段、指标和评测范围一致。新增或整理文档时同步更新索引。本指令文件在同一文件中并列三语。

## GitHub Release

- Release 只发布模型文件及必要的配套元信息：校验文件、模型来源与使用条款、身份及训练样本数清单。
- 不在 Release 附件中发布部署包、应用代码、Docker Compose 或 API 文档。部署材料保留在仓库中，容器镜像发布到 Docker Hub。
- Release 说明以模型下载和模型范围为主，列出所有已训练声优／身份的模型内置姓名及每人的实际训练样本数，并注明统计口径。
- 样本数必须核实当前发布模型实际使用的数据；测试集不计入训练样本数。
- Release 说明不写 Linux 部署介绍、API 功能介绍、镜像架构说明、快速部署命令或密钥配置步骤。
- 不发布训练照片、原始特征、评测数据或 API 密钥。

以上为用户确认的发布约定，后续发布均遵守。

<a id="en"></a>

## Documentation languages

Maintain repository Markdown documentation in three languages. English uses `.md`, Simplified Chinese uses `.zh-CN.md`, and Japanese uses `.ja.md`. Each document provides links to all three versions after its title. Keep commands, API fields, metrics and evaluation scope consistent across translations. Update the documentation index when adding or reorganizing documents. This instruction file contains all three languages together.

## GitHub Releases

- Publish model files and necessary metadata only, including checksums, model sources and terms, and identity/training-count lists.
- Keep deployment packages, application code, Compose files and API documentation in the repository or Actions Artifacts. Publish container images to Docker Hub.
- Release notes focus on model downloads and model scope. List every built-in identity name and the actual training count for each identity, and explain how counts were obtained.
- Verify counts against data used by the model being released. Exclude the test set from training counts.
- Release notes cover model delivery and scope. Deployment instructions, API overviews, image architectures, quick-start deployment commands and key setup belong in repository documentation.
- Keep training photos, raw features, evaluation data and API keys out of Releases.

These release conventions were confirmed by the user and apply to subsequent releases.

<a id="ja"></a>

## ドキュメントの言語

管理対象の Markdown 文書は英語、簡体字中国語、日本語を用意します。英語は `.md`、中国語は `.zh-CN.md`、日本語は `.ja.md` を使用します。各文書はタイトルの後に3言語へのリンクを置き、コマンド、APIフィールド、指標と評価範囲を一致させます。文書を追加・整理した際は索引も更新します。この指示ファイルには3言語を併記します。

## GitHub Release の規約

- 公開するのはモデルと必要なメタデータです。チェックサム、モデル出典と利用条件、人物名と学習件数の一覧を含めます。
- 配備パッケージ、アプリケーションコード、Compose、API文書はリポジトリまたは Actions Artifacts で管理します。コンテナイメージは Docker Hub に公開します。
- Release説明はモデルのダウンロードと対応範囲を中心とし、全組み込み人物名、実際の学習件数、集計方法を記載します。
- 公開対象モデルの学習に使ったデータで件数を確認し、テスト集合を学習件数から除きます。
- Release説明にはモデル配布と範囲を記載します。配備手順、API概要、イメージのアーキテクチャ、簡易配備コマンドとキー設定はリポジトリ文書に置きます。
- 学習写真、生の特徴量、評価データ、APIキーは Release の対象外です。

以上はユーザーが確認した公開規約で、以後の公開にも適用します。
