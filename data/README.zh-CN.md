# 本地数据目录

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

- `raw/<identity>/`：用户提供的参考图片。可手动添加或通过网页界面上传；工具绝不修改现有原始文件。
- `aligned/`：通过审核的 `112×112` 人脸裁剪图与 `metadata.jsonl`。
- `rejected/`：记录被跳过图片或人脸的 JSONL 元数据。
- `gallery/<identity>/`：用户提供的、带标签的独立评测图片，仅用于独立验证。建库工具绝不会写入此目录。
- `runtime/`：可随时重新生成的产物，包括生产 embedding 图库、检测缓存和评估报告。
- `query/`：可选的批量识别输入目录。其中的图片只作为查询输入，绝不会加入图库。
- `cache/`：历史遗留的可丢弃缓存目录；新产物统一写入 `runtime/`。
- `review/raw_faces/`：群像人工审核所需的本地编号预览图与选择清单。

上述目录均被 Git 忽略，仅保留 `.gitkeep` 占位文件。
