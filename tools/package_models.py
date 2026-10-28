#!/usr/bin/env python3
"""Build a reproducible model-only Release and its identity/count documentation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from tools._artifacts import archive_tree, write_checksums
    from tools.fetch_models import REQUIRED_FILES, load_source, sha256, verify_files
else:
    from _artifacts import archive_tree, write_checksums
    from fetch_models import REQUIRED_FILES, load_source, sha256, verify_files

ASSET_NAMES = frozenset(
    {
        "seiyuu-faceid-models.tar.gz",
        "SHA256SUMS",
        "TRAINING_SAMPLES.csv",
        "TRAINING_PROVENANCE.json",
        "MODEL_LICENSES.md",
    }
)


def validate_training(root: Path) -> dict:
    source = load_source(root / "release/model-source.json")
    verify_files(root, source)
    manifest = json.loads((root / "models/selected/manifest.json").read_text("utf-8"))
    for filename, expected in manifest["assets"].items():
        if filename not in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz"):
            raise ValueError("Unexpected model manifest asset.")
        if sha256(root / "models/selected" / filename) != expected:
            raise ValueError("Model asset checksum differs from manifest.")
    if sha256(root / "models/det_500m.onnx") != manifest["detector"]["sha256"]:
        raise ValueError("Detector checksum differs from manifest.")
    training = json.loads((root / "release/training.json").read_text("utf-8"))
    with np.load(root / "models/selected/classifier.npz", allow_pickle=False) as head:
        names = head["class_names"].tolist()
    identities = training["identities"]
    sizes = training["split_sizes"]
    counts = [row["training_samples"] for row in identities]
    if (
        training.get("schema_version") != 1
        or training["classifier_sha256"] != manifest["assets"]["classifier.npz"]
        or training["fit_splits"] != ["train", "validation"]
        or training["count_unit"] != "aligned_face_crop"
        or [row["name"] for row in identities] != names
        or len(names) != manifest["classes"]
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n or any(ord(c) < 32 for c in n) for n in names)
        or any(type(n) is not int or n <= 0 for n in counts)
        or any(
            type(sizes[k]) is not int or sizes[k] < 0
            for k in ("train", "validation", "test_excluded")
        )
        or sum(counts) != manifest["training_faces"]
        or sum(counts) != sizes["train"] + sizes["validation"]
    ):
        raise ValueError("Training counts, model labels or classifier binding do not match.")
    with (root / "models/TRAINING_SAMPLES.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if [(r["identity"], int(r["training_samples"])) for r in csv_rows] != list(zip(names, counts)):
        raise ValueError("Published source counts differ from verified training metadata.")
    return training


def release_notes(training: dict, download_base: str) -> str:
    identities = training["identities"]
    total = sum(row["training_samples"] for row in identities)
    rows = "\n".join(
        f"| {row['name'].replace('|', '&#124;')} | {row['training_samples']} |"
        for row in identities
    )
    sizes = training["split_sizes"]
    return (
        "# 模型下载与范围\n\n"
        f"[模型文件]({download_base}/seiyuu-faceid-models.tar.gz) · "
        f"[SHA-256 校验]({download_base}/SHA256SUMS) · "
        f"[身份及样本数]({download_base}/TRAINING_SAMPLES.csv) · "
        f"[来源与使用条款]({download_base}/MODEL_LICENSES.md)\n\n"
        f"已训练 **{len(identities)} 个身份**，实际拟合 **{total:,} 张有效人脸裁剪**。\n\n"
        f"统计口径：原训练集 {sizes['train']:,} 张与验证集 {sizes['validation']:,} 张合并拟合；"
        f"测试集 {sizes['test_excluded']:,} 张不计入训练样本数。姓名按模型内置标签顺序列出。"
        "冻结特征/划分文件的校验摘要和分类头绑定见 TRAINING_PROVENANCE.json；"
        "预训练骨干的上游样本不计入本分类头的训练样本数。\n\n"
        "| 声优／身份 | 实际训练样本数（张） |\n| --- | ---: |\n"
        f"{rows}\n| **合计** | **{total:,}** |\n\n"
        "模型包包含检测器、双骨干、分类头及必要元信息。"
        "权重使用范围遵守随附上游条款。\n"
    )


def package_models(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.absolute()
    if output.exists():
        raise FileExistsError(output)
    training = validate_training(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".models-release-", dir=output.parent) as tmp:
        stage, payload = Path(tmp) / "release", Path(tmp) / "payload"
        stage.mkdir()
        for name in sorted(REQUIRED_FILES - {"models/SHA256SUMS"}):
            target = payload / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        provenance = payload / "models/TRAINING_PROVENANCE.json"
        provenance.write_text(
            json.dumps(training, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_checksums(payload / "models")
        archive_tree(payload, stage / "seiyuu-faceid-models.tar.gz")
        for name in ("TRAINING_SAMPLES.csv", "TRAINING_PROVENANCE.json", "MODEL_LICENSES.md"):
            shutil.copyfile(payload / "models" / name, stage / name)
        write_checksums(stage)
        tag = "models-" + sha256(stage / "seiyuu-faceid-models.tar.gz")[:16]
        stage.rename(output)
    repository_url = load_source(root / "release/model-source.json")["url"].split(
        "/releases/download/", 1
    )[0]
    return {
        "tag": tag,
        "notes": release_notes(training, f"{repository_url}/releases/download/{tag}"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    result = package_models(args.root, args.output)
    args.notes.write_text(result["notes"], encoding="utf-8")
    args.metadata.write_text(json.dumps({"tag": result["tag"]}) + "\n", encoding="utf-8")
    print(result["tag"])


if __name__ == "__main__":
    main()
