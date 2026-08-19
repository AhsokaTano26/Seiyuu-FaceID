#!/usr/bin/env python3
"""Create a self-contained, local evaluation report with curves and figures."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from seiyuu_faceid.dataset.metadata import read_jsonl
from seiyuu_faceid.gallery.index import Gallery
from seiyuu_faceid.gallery.scoring import search_gallery
from seiyuu_faceid.runtime.config import AppConfig, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--queries", default="data/runtime/evaluation/external_queries.npz"
    )
    parser.add_argument(
        "--records", default="data/runtime/evaluation/external_records.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="data/runtime/reports/evaluation_post_review"
    )
    args = parser.parse_args()
    config = load_config(_bootstrap.ROOT / args.config)
    output_dir = (_bootstrap.ROOT / args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    curves_dir = output_dir / "curves"
    figures_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    queries, expected, sources = _load_queries((_bootstrap.ROOT / args.queries).resolve())
    gallery = Gallery.load(config.gallery_vectors_path, config.gallery_metadata_path)
    rankings, scores = _score_queries(gallery, queries, config)
    predicted = [ranking[0] for ranking in rankings]
    correct = np.asarray(
        [actual == guess for actual, guess in zip(expected, predicted, strict=True)], dtype=bool
    )
    top_five = np.asarray(
        [actual in ranking[:5] for actual, ranking in zip(expected, rankings, strict=True)], dtype=bool
    )
    proxy_scores = _open_set_proxy_scores(gallery, queries, expected, config)
    pair_scores, pair_labels = _sample_verification_pairs(queries, expected)
    roc = _verification_curve(pair_scores, pair_labels)
    threshold_curve = _threshold_curve(scores, correct, proxy_scores)
    per_identity = _per_identity(expected, predicted, rankings)
    exclusion_counts = _exclusion_counts((_bootstrap.ROOT / args.records).resolve())
    references_per_identity = Counter(gallery.reference_identities)

    _write_csv(curves_dir / "verification_roc.csv", roc)
    _write_csv(curves_dir / "threshold_operating_points.csv", threshold_curve)
    _write_csv(curves_dir / "per_identity_accuracy.csv", per_identity)
    _write_csv(
        curves_dir / "score_samples.csv",
        _score_rows(expected, sources, predicted, scores, correct, proxy_scores),
    )

    _write_roc_svg(figures_dir / "01_verification_roc.svg", roc)
    _write_distribution_svg(
        figures_dir / "02_score_distribution.svg", scores[correct], scores[~correct], proxy_scores, config
    )
    _write_threshold_svg(figures_dir / "03_threshold_tradeoff.svg", threshold_curve, config)
    _write_bar_svg(
        figures_dir / "04_per_identity_accuracy.svg",
        per_identity,
        "各身份 Top-1 准确率（独立评估）",
        "top1_accuracy",
        "Top-1 准确率",
    )
    _write_bar_svg(
        figures_dir / "05_reference_counts.svg",
        [
            {"identity": identity, "references": count}
            for identity, count in sorted(references_per_identity.items())
        ],
        "每个身份的已入库参考脸数量",
        "references",
        "参考脸数量",
    )

    metrics = _metrics(
        config,
        expected,
        correct,
        top_five,
        scores,
        proxy_scores,
        pair_scores,
        pair_labels,
        per_identity,
        exclusion_counts,
        references_per_identity,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(metrics), encoding="utf-8"
    )
    (output_dir / "index.html").write_text(_html_index(metrics), encoding="utf-8")
    print(f"Wrote evaluation report: {output_dir}")
    return 0


def _load_queries(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with np.load(path) as values:
        return (
            np.asarray(values["embeddings"], dtype=np.float32),
            [str(value) for value in values["expected"]],
            [str(value) for value in values["sources"]],
        )


def _score_queries(
    gallery: Gallery, queries: np.ndarray, config: AppConfig
) -> tuple[list[list[str]], np.ndarray]:
    rankings: list[list[str]] = []
    scores: list[float] = []
    for query in queries:
        candidates = search_gallery(
            gallery,
            query,
            top_k=config.top_k,
            candidate_search_top_k=config.candidate_search_top_k,
            aggregation=config.aggregation,
            verification_top_k=config.verification_top_k,
        )
        rankings.append([candidate.identity for candidate in candidates])
        scores.append(candidates[0].score)
    return rankings, np.asarray(scores, dtype=np.float32)


def _open_set_proxy_scores(
    gallery: Gallery, queries: np.ndarray, expected: list[str], config: AppConfig
) -> np.ndarray:
    excluded: dict[str, Gallery] = {}
    for identity in sorted(set(expected)):
        keep = np.asarray([item != identity for item in gallery.reference_identities], dtype=bool)
        identities = [
            item
            for item, is_kept in zip(gallery.reference_identities, keep, strict=True)
            if is_kept
        ]
        metadata = [
            item
            for item, is_kept in zip(gallery.reference_metadata, keep, strict=True)
            if is_kept
        ]
        excluded[identity] = Gallery.from_references(gallery.references[keep], identities, metadata)
    return np.asarray(
        [
            search_gallery(
                excluded[identity],
                query,
                top_k=config.top_k,
                candidate_search_top_k=config.candidate_search_top_k,
                aggregation=config.aggregation,
                verification_top_k=config.verification_top_k,
            )[0].score
            for query, identity in zip(queries, expected, strict=True)
        ],
        dtype=np.float32,
    )


def _sample_verification_pairs(
    queries: np.ndarray, expected: list[str], max_negative: int = 25_000
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, identity in enumerate(expected):
        grouped[identity].append(index)
    same_pairs = [pair for indexes in grouped.values() for pair in combinations(indexes, 2)]
    rng = np.random.default_rng(20260819)
    negative_pairs: list[tuple[int, int]] = []
    while len(negative_pairs) < min(max_negative, len(expected) * (len(expected) - 1) // 2):
        left, right = sorted(rng.choice(len(expected), size=2, replace=False).tolist())
        if expected[left] != expected[right]:
            negative_pairs.append((left, right))
    pairs = same_pairs + negative_pairs
    left = np.asarray([queries[first] for first, _ in pairs], dtype=np.float32)
    right = np.asarray([queries[second] for _, second in pairs], dtype=np.float32)
    return np.einsum("ij,ij->i", left, right), np.asarray(
        [True] * len(same_pairs) + [False] * len(negative_pairs), dtype=bool
    )


def _verification_curve(scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.linspace(-0.05, 1.0, 301)
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        accepted = scores >= threshold
        tar = float(accepted[labels].mean())
        far = float(accepted[~labels].mean())
        curve.append(
            {
                "threshold": float(threshold),
                "tar": tar,
                "far": far,
                "frr": 1 - tar,
                "balanced_accuracy": (tar + 1 - far) / 2,
            }
        )
    return curve


def _threshold_curve(
    scores: np.ndarray, correct: np.ndarray, proxy_scores: np.ndarray
) -> list[dict[str, float]]:
    curve: list[dict[str, float]] = []
    for threshold in np.linspace(0.0, 0.9, 181):
        accepted = scores >= threshold
        curve.append(
            {
                "threshold": float(threshold),
                "known_accepted_rate": float(accepted.mean()),
                "known_correct_recognized_rate": float((accepted & correct).mean()),
                "known_recognized_precision": float(correct[accepted].mean())
                if accepted.any()
                else 1.0,
                "proxy_unknown_false_recognized_rate": float((proxy_scores >= threshold).mean()),
            }
        )
    return curve


def _per_identity(
    expected: list[str], predicted: list[str], rankings: list[list[str]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for identity in sorted(set(expected)):
        indexes = [index for index, item in enumerate(expected) if item == identity]
        rows.append(
            {
                "identity": identity,
                "samples": len(indexes),
                "top1_accuracy": float(
                    np.mean([predicted[index] == identity for index in indexes])
                ),
                "top5_accuracy": float(
                    np.mean([identity in rankings[index][:5] for index in indexes])
                ),
            }
        )
    return rows


def _exclusion_counts(path: Path) -> dict[str, int]:
    records = read_jsonl(path)
    return dict(Counter(str(record.get("reason", "accepted")) for record in records))


def _score_rows(
    expected: list[str],
    sources: list[str],
    predicted: list[str],
    scores: np.ndarray,
    correct: np.ndarray,
    proxy_scores: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "source_image": source,
            "expected_identity": actual,
            "predicted_identity": guess,
            "top1_score": float(score),
            "top1_correct": bool(is_correct),
            "open_set_proxy_score": float(proxy_score),
        }
        for source, actual, guess, score, is_correct, proxy_score in zip(
            sources, expected, predicted, scores, correct, proxy_scores, strict=True
        )
    ]


def _metrics(
    config: AppConfig,
    expected: list[str],
    correct: np.ndarray,
    top_five: np.ndarray,
    scores: np.ndarray,
    proxy_scores: np.ndarray,
    pair_scores: np.ndarray,
    pair_labels: np.ndarray,
    per_identity: list[dict[str, object]],
    exclusion_counts: dict[str, int],
    references_per_identity: Counter[str],
) -> dict[str, object]:
    high = float(config.high_threshold or 0.0)
    low = float(config.low_threshold or 0.0)
    recognized = scores >= high
    review = (scores >= low) & ~recognized
    return {
        "scope": {
            "enrollment": "data/raw (manually reviewed group-photo selections included)",
            "evaluation": "data/gallery (independent labelled images; never enrolled)",
            "identities": len(set(expected)),
            "reference_faces": int(sum(references_per_identity.values())),
            "references_per_identity": dict(sorted(references_per_identity.items())),
        },
        "evaluation_input": {
            "total_images": int(sum(exclusion_counts.values())),
            "eligible_single_faces": len(expected),
            "exclusions": exclusion_counts,
        },
        "identification": {
            "top1_accuracy": float(correct.mean()),
            "top5_accuracy": float(top_five.mean()),
            "recognized_rate": float(recognized.mean()),
            "recognized_precision": float(correct[recognized].mean()),
            "review_rate": float(review.mean()),
            "unknown_rate": float((scores < low).mean()),
            "per_identity": per_identity,
        },
        "thresholds": {"high": high, "low": low, "aggregation": config.aggregation},
        "open_set_proxy": {
            "method": "Remove each query's true identity from the Gallery before scoring.",
            "false_recognized_rate": float((proxy_scores >= high).mean()),
            "review_rate": float(((proxy_scores >= low) & (proxy_scores < high)).mean()),
            "unknown_rate": float((proxy_scores < low).mean()),
            "p95": float(np.quantile(proxy_scores, 0.95)),
            "p99_5": float(np.quantile(proxy_scores, 0.995)),
        },
        "pair_verification": {
            "pairs": int(len(pair_scores)),
            "same_person_pairs": int(pair_labels.sum()),
            "different_person_pairs": int((~pair_labels).sum()),
            "roc_auc": _auc(pair_scores, pair_labels),
            "same_mean_similarity": float(pair_scores[pair_labels].mean()),
            "different_mean_similarity": float(pair_scores[~pair_labels].mean()),
        },
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[np.argsort(scores, kind="stable")] = np.arange(len(scores), dtype=np.float64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    return float((ranks[labels].sum() - positive * (positive - 1) / 2) / (positive * negative))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg_canvas(title: str) -> tuple[list[str], callable]:
    width, height = 960, 540
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;fill:#1f2937}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#4b5563;stroke-width:1.2}.line{fill:none;stroke-width:3}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="70" y="38" font-size="22" font-weight="700">{html.escape(title)}</text>',
    ]

    def close() -> str:
        return "".join(parts) + "</svg>"

    return parts, close


def _write_roc_svg(path: Path, rows: list[dict[str, float]]) -> None:
    parts, close = _svg_canvas("验证对 ROC 曲线")
    left, top, right, bottom = 88, 72, 910, 470
    _axes(parts, left, top, right, bottom, "FAR（误接受率）", "TAR（通过率）")
    points = " ".join(
        f"{left + row['far'] * (right - left):.1f},{bottom - row['tar'] * (bottom - top):.1f}"
        for row in reversed(rows)
    )
    parts.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#9ca3af" stroke-dasharray="6 6"/>',
            f'<polyline class="line" points="{points}" stroke="#2563eb"/>',
            '<text x="720" y="92" font-size="13">蓝线：验证对 ROC</text>',
        ]
    )
    path.write_text(close(), encoding="utf-8")


def _write_distribution_svg(
    path: Path,
    correct_scores: np.ndarray,
    wrong_scores: np.ndarray,
    proxy_scores: np.ndarray,
    config: AppConfig,
) -> None:
    parts, close = _svg_canvas("身份得分分布与阈值")
    left, top, right, bottom = 88, 72, 910, 470
    maximum = max(len(correct_scores), len(wrong_scores), len(proxy_scores))
    _axes(parts, left, top, right, bottom, "Top-1 得分", "样本数", x_min=0.0, x_max=0.9, y_max=maximum)
    bins = np.linspace(0.0, 0.9, 37)
    for values, color, name in [
        (correct_scores, "#16a34a", "正确 Top-1"),
        (wrong_scores, "#f59e0b", "错误 Top-1"),
        (proxy_scores, "#dc2626", "未知代理"),
    ]:
        counts, _ = np.histogram(values, bins=bins)
        points = " ".join(
            f"{left + (bins[index] / 0.9) * (right - left):.1f},{bottom - (count / maximum) * (bottom - top):.1f}"
            for index, count in enumerate(counts)
        )
        parts.append(f'<polyline class="line" points="{points}" stroke="{color}"/>')
        legend_x = 590 + ["#16a34a", "#f59e0b", "#dc2626"].index(color) * 105
        parts.append(f'<text x="{legend_x}" y="92" font-size="13" fill="{color}">{name}</text>')
    for value, color, label in [
        (float(config.low_threshold or 0), "#f59e0b", "低阈值"),
        (float(config.high_threshold or 0), "#7c3aed", "高阈值"),
    ]:
        x = left + (value / 0.9) * (right - left)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{color}" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x + 4:.1f}" y="{bottom - 8}" font-size="12">{label} {value:.2f}</text>')
    path.write_text(close(), encoding="utf-8")


def _write_threshold_svg(path: Path, rows: list[dict[str, float]], config: AppConfig) -> None:
    parts, close = _svg_canvas("阈值：已知覆盖率与未知误认率")
    left, top, right, bottom = 88, 72, 910, 470
    _axes(parts, left, top, right, bottom, "阈值", "比例", x_min=0.0, x_max=0.9)
    for field, color, label in [
        ("known_correct_recognized_rate", "#16a34a", "已知正确识别率"),
        ("proxy_unknown_false_recognized_rate", "#dc2626", "未知代理误认率"),
    ]:
        points = " ".join(
            f"{left + row['threshold'] / 0.9 * (right - left):.1f},{bottom - row[field] * (bottom - top):.1f}"
            for row in rows
        )
        parts.append(f'<polyline class="line" points="{points}" stroke="{color}"/>')
        legend_x = 610 if color == "#16a34a" else 760
        parts.append(f'<text x="{legend_x}" y="92" font-size="13" fill="{color}">{label}</text>')
    for value, color, label in [
        (float(config.low_threshold or 0), "#f59e0b", "低"),
        (float(config.high_threshold or 0), "#7c3aed", "高"),
    ]:
        x = left + value / 0.9 * (right - left)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{color}" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x + 4:.1f}" y="{bottom - 8}" font-size="12">{label} {value:.2f}</text>')
    path.write_text(close(), encoding="utf-8")


def _write_bar_svg(
    path: Path, rows: list[dict[str, object]], title: str, field: str, y_label: str
) -> None:
    parts, close = _svg_canvas(title)
    left, top, right, bottom = 88, 72, 910, 470
    values = [float(row[field]) for row in rows]
    percentage = field.endswith("accuracy")
    y_max = 1.0 if percentage else max(values) * 1.15
    _axes(parts, left, top, right, bottom, "身份", y_label, y_max=y_max)
    step = (right - left) / len(rows)
    for index, row in enumerate(rows):
        value = float(row[field])
        x = left + index * step + step * 0.18
        width = step * 0.64
        height = value / y_max * (bottom - top)
        y = bottom - height
        label = html.escape(str(row["identity"]))
        displayed = f"{value:.1%}" if percentage else str(int(value))
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="#2563eb"/>',
                f'<text x="{x + width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12">{displayed}</text>',
                f'<text x="{x + width / 2:.1f}" y="{bottom + 23}" text-anchor="middle" font-size="12">{label}</text>',
            ]
        )
    path.write_text(close(), encoding="utf-8")


def _axes(
    parts: list[str],
    left: int,
    top: int,
    right: int,
    bottom: int,
    x_label: str,
    y_label: str,
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_max: float = 1.0,
) -> None:
    parts.extend(
        [
            f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="none" stroke="#4b5563"/>',
            f'<text x="{(left + right) / 2}" y="522" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="20" y="{(top + bottom) / 2}" transform="rotate(-90 20 {(top + bottom) / 2})" text-anchor="middle" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    for index in range(6):
        fraction = index / 5
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        x_value = x_min + fraction * (x_max - x_min)
        y_value = fraction * y_max
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text x="{x:.1f}" y="{bottom + 43}" text-anchor="middle" font-size="12">{x_value:.2f}</text>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{y_value:.2f}</text>',
            ]
        )


def _summary_markdown(metrics: dict[str, object]) -> str:
    identification = metrics["identification"]
    proxy = metrics["open_set_proxy"]
    verification = metrics["pair_verification"]
    scope = metrics["scope"]
    evaluation = metrics["evaluation_input"]
    assert isinstance(identification, dict)
    assert isinstance(proxy, dict)
    assert isinstance(verification, dict)
    assert isinstance(scope, dict)
    assert isinstance(evaluation, dict)
    exclusions = evaluation["exclusions"]
    assert isinstance(exclusions, dict)
    return f"""# seiyuu-faceid 独立评估报告（群像审核后）

## 结论

- 生产图库使用 `data/raw`，共 **{scope['reference_faces']}** 张参考脸、**{scope['identities']}** 个身份；`data/gallery` 始终仅作为独立评估集。
- {evaluation['total_images']} 张评估原图中，自动评估只纳入标签无歧义的 **{evaluation['eligible_single_faces']}** 张单人脸。排除项：多人脸待选择 {exclusions.get('selection_required', 0)} 张、未检出人脸 {exclusions.get('no_face_detected', 0)} 张、质量不合格 {exclusions.get('quality_rejected', 0)} 张。
- 闭集身份识别 Top-1 为 **{identification['top1_accuracy']:.2%}**，Top-5 为 **{identification['top5_accuracy']:.2%}**。这是真实独立图片的结果，不是训练集准确率。
- 当前阈值 `high={metrics['thresholds']['high']:.2f}`、`low={metrics['thresholds']['low']:.2f}` 下，直接识别覆盖率为 **{identification['recognized_rate']:.2%}**，其中精度为 **{identification['recognized_precision']:.2%}**；其余进入 Review 或 Unknown。
- 以“移除真实身份后搜索”的未知人代理测试中，误识别为已知人物的比例约 **{proxy['false_recognized_rate']:.2%}**。这不是实际陌生人集，后续仍应补充真实未知人物图片做最终阈值确认。
- 验证对 ROC AUC 为 **{verification['roc_auc']:.4f}**，同一人平均相似度 **{verification['same_mean_similarity']:.4f}**，不同人平均相似度 **{verification['different_mean_similarity']:.4f}**。

## 重要限制与下一步

1. 伊藤美来的独立评估照片中含有 Pyxis 图集，存在身份混杂或强风格差异；该类评估图不应进入训练。
2. 增加人工审核的群像后，参考脸从 1,108 增至 1,886，但独立 Top-1 未发生虚高改善。这说明当前优先事项是标签纯度与跨时期覆盖，而非盲目继续扩容。
3. 目前不建议直接进行小样本 ArcFace 微调：6 个身份、约 1,886 张参考脸的条件下容易过拟合。应先清理独立评估标签，再以该报告重新比较预训练与微调模型。

## 文件说明

- `metrics.json`：机器可读结论与指标。
- `curves/`：ROC、阈值扫描、逐身份准确率、逐图得分 CSV。
- `figures/`：SVG 图，可直接在浏览器中打开。
- `index.html`：本目录图表总览。
"""


def _html_index(metrics: dict[str, object]) -> str:
    identification = metrics["identification"]
    proxy = metrics["open_set_proxy"]
    assert isinstance(identification, dict)
    assert isinstance(proxy, dict)
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>seiyuu-faceid 评估报告</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;margin:36px;color:#172033;max-width:1080px}}h1{{margin-bottom:4px}}.metrics{{display:flex;gap:14px;flex-wrap:wrap}}.metric{{border:1px solid #d9dee8;padding:14px 18px;border-radius:8px;min-width:150px}}.metric strong{{display:block;font-size:25px}}img{{width:100%;border:1px solid #e5e7eb;margin:16px 0 28px}}small{{color:#667085}}</style>
<h1>seiyuu-faceid 独立评估报告</h1><small>群像人工审核后生成；评估图片未参与建库。</small>
<div class="metrics"><div class="metric"><span>Top-1</span><strong>{identification['top1_accuracy']:.2%}</strong></div><div class="metric"><span>Top-5</span><strong>{identification['top5_accuracy']:.2%}</strong></div><div class="metric"><span>直接识别精度</span><strong>{identification['recognized_precision']:.2%}</strong></div><div class="metric"><span>未知代理误认</span><strong>{proxy['false_recognized_rate']:.2%}</strong></div></div>
<h2>验证对 ROC</h2><img src="figures/01_verification_roc.svg" alt="验证对 ROC 曲线">
<h2>得分分布</h2><img src="figures/02_score_distribution.svg" alt="得分分布">
<h2>阈值取舍</h2><img src="figures/03_threshold_tradeoff.svg" alt="阈值曲线">
<h2>逐身份准确率</h2><img src="figures/04_per_identity_accuracy.svg" alt="逐身份准确率">
<h2>参考脸数量</h2><img src="figures/05_reference_counts.svg" alt="参考脸数量">
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
