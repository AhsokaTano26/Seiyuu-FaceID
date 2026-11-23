#!/usr/bin/env python3
"""Plot version-bound model metrics from public aggregate snapshots, without inference.

Run from the repository root:
uv run --no-project --python 3.13 --with matplotlib==3.10.9 --with numpy==2.5.4 \
    python tools/draw_model_accuracy_report.py
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seiyuu-model-report-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Patch
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/model_accuracy_20261103"
INK = "#263746"
BLUE = "#456D90"
GREEN = "#628575"
RED = "#AC5F53"
AMBER = "#B1844B"


def select_font(candidates):
    for family in candidates:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=[family]), fallback_to_default=False
            )
            face = FT2Font(path)
            if face.family_name.casefold() == family.casefold():
                return family, path, set(face.get_charmap())
        except (ValueError, OSError, RuntimeError):
            continue
    raise RuntimeError(f"No permitted font available: {candidates}")


def style():
    latin = select_font(
        ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "Nimbus Roman", "Times"]
    )
    cjk = select_font(["SimSun", "Songti SC", "STSong", "Source Han Serif SC", "Noto Serif CJK SC"])
    plt.rcParams.update(
        {
            "font.family": [latin[0], cjk[0]],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": True,
            "axes.axisbelow": True,
            "hatch.linewidth": 0.35,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "seiyuu-model-accuracy-20261103",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    print(f"Fonts: Latin={latin[0]}; Chinese={cjk[0]}")
    if latin[0] != "Times New Roman" or cjk[0] != "SimSun":
        print("FONT FALLBACK: preferred family unavailable; permitted fallback selected")
    return latin, cjk


def axes_style(ax, *, horizontal=False):
    ax.spines[["top", "right"]].set_visible(False)
    if horizontal:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.grid(axis="x" if horizontal else "y", color="#E1E6E9", linewidth=0.5)


def draw_version_metrics(data):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.65))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.25, top=0.76, wspace=0.65)
    labels = ["Top-1 准确率", "Top-5 准确率", "自动接受准确率", "自动接受覆盖率"]
    keys = ["top1_accuracy", "top5_accuracy", "accepted_precision", "coverage"]
    versions = [("old", BLUE, "#DEE8F0", ""), ("new", GREEN, "#E1ECE5", "///")]
    for index, (key, edge, fill, hatch) in enumerate(versions):
        model = data[key]
        ax = axes[index]
        values = [100 * model["metrics"][metric] for metric in keys]
        ax.barh(
            np.arange(4),
            values,
            height=0.58,
            color=fill,
            edgecolor=edge,
            linewidth=0.7,
            hatch=hatch,
        )
        for position, value in enumerate(values):
            ax.text(5, position, f"{value:.2f}%", va="center", fontsize=10)
        ax.set(yticks=np.arange(4), yticklabels=labels, xlim=(0, 100), xticks=[0, 50, 100])
        ax.invert_yaxis()
        ax.set_xlabel("比例 / %")
        axes_style(ax, horizontal=True)
        x = 0.035 if index == 0 else 0.55
        name = "历史" if index == 0 else "本次"
        fig.text(x, 0.95, f"({'ab'[index]}) {name}模型：{model['class_count']} 身份", va="top")
        fig.text(x, 0.88, model["tag"], va="top", fontsize=8.5)
        fig.text(
            x, 0.82, f"各自留出测试：{model['metrics']['samples']:,} 张人脸", va="top", fontsize=8.5
        )
    fig.text(
        0.035, 0.11, "两版的身份集合、测试样本与标签审核情况不同；仅描述各自结果。", fontsize=8.5
    )
    fig.text(
        0.035,
        0.045,
        "自动接受阈值均为 0.95；这些数值不能作为同条件下性能提升的证据。",
        fontsize=8.5,
    )
    return fig


def draw_threshold_outcomes(data):
    model = data["new"]
    metrics = model["metrics"]
    outcomes = metrics["outcomes"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.7))
    fig.subplots_adjust(left=0.26, right=0.96, bottom=0.15, top=0.78, hspace=1.55)
    fig.text(0.04, 0.97, f"606 身份模型｜{model['tag']}", va="top", fontsize=9.5)
    first, second = axes
    categories = [
        (outcomes["accepted_correct"], "正确自动返回", GREEN, ""),
        (outcomes["accepted_wrong"], "自动误接受", RED, "xxx"),
        (metrics["review"], "需复核", AMBER, "///"),
    ]
    left = 0
    handles, labels = [], []
    for count, name, color, hatch in categories:
        share = 100 * count / metrics["samples"]
        first.barh(
            0,
            share,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0,
            hatch=hatch,
            height=0.5,
        )
        left += share
        handles.append(Patch(facecolor=color, edgecolor="white", hatch=hatch))
        labels.append(f"{name}\n{count:,}（{share:.2f}%）")
    error_center = (
        100 * (outcomes["accepted_correct"] + outcomes["accepted_wrong"] / 2) / metrics["samples"]
    )
    first.annotate(
        "误接受 15 张（0.17%）",
        xy=(error_center, 0.25),
        xytext=(69, 0.48),
        fontsize=8.5,
        color=RED,
        arrowprops={"arrowstyle": "-", "color": RED, "linewidth": 0.65},
    )
    first.set(xlim=(0, 100), ylim=(-0.6, 0.6), yticks=[], xticks=[0, 20, 40, 60, 80, 100])
    first.set_xlabel("占全部测试人脸的比例 / %")
    first.text(-0.31, 1.48, "(a) 全部测试人脸的处置（N=8,759）", transform=first.transAxes)
    first.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.55),
        ncol=3,
        columnspacing=1.1,
        handlelength=1.1,
        handletextpad=0.45,
    )
    axes_style(first, horizontal=True)
    errors = [outcomes["accepted_wrong"], outcomes["review_wrong"]]
    second.barh(
        [0, 1],
        errors,
        height=0.5,
        color=[RED, AMBER],
        edgecolor="white",
        linewidth=0.7,
        hatch="///",
    )
    for i, count in enumerate(errors):
        second.text(count + 0.7, i, f"{count} / 54（{100 * count / 54:.2f}%）", va="center")
    second.set(
        yticks=[0, 1],
        yticklabels=["自动接受的错误", "需复核的错误"],
        xlim=(0, 54),
        xticks=[0, 10, 20, 30, 40, 50],
    )
    second.invert_yaxis()
    second.set_xlabel("Top-1 错误人脸数 / 张")
    second.text(-0.31, 1.35, "(b) 54 张 Top-1 错误人脸中的处置", transform=second.transAxes)
    axes_style(second, horizontal=True)
    fig.text(0.04, 0.025, "复核只表示需要人工检查，不代表这些错误已被人工纠正。", fontsize=8.5)
    return fig


def draw_sample_support(data):
    support = data["new_support"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.3))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.36, top=0.79, wspace=0.28)
    fig.text(0.035, 0.97, f"606 身份模型｜{data['new']['tag']}", va="top", fontsize=9.5)
    ymax = 25 * math.ceil(max(max(support["fit_histogram"]), max(support["test_histogram"])) / 25)
    for index, (name, color, hatch) in enumerate([("fit", BLUE, ""), ("test", GREEN, "///")]):
        ax = axes[index]
        labels, counts = support[name + "_bin_labels"], support[name + "_histogram"]
        positions = np.arange(len(labels))
        ax.bar(
            positions,
            counts,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            width=0.75,
        )
        for position, count in zip(positions, counts, strict=True):
            ax.text(position, count + 3, str(count), ha="center", fontsize=8.5)
        ax.set(xticks=positions, xticklabels=labels, ylim=(0, ymax + 15))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.set_ylabel("身份数")
        ax.set_xlabel("每个身份的人脸数 / 张")
        ax.text(
            0,
            1.13,
            "(a) 实际拟合样本" if index == 0 else "(b) 留出测试样本",
            transform=ax.transAxes,
        )
        axes_style(ax)
    fig.text(0.035, 0.11, "每人拟合样本中位数：49 张；每人测试样本中位数：8 张。", fontsize=8.5)
    fig.text(
        0.035,
        0.035,
        "208 / 606 个身份的测试样本不超过 5 张，个别身份的稳定性证据有限。",
        fontsize=8.5,
    )
    return fig


def export(fig, name, output, fonts):
    fig.canvas.draw()
    allowed = fonts[0][2] | fonts[1][2]
    glyphs = set()
    for text in fig.findobj(Text):
        if text.get_visible():
            glyphs.update(c for c in text.get_text() if not c.isspace())
    missing = glyphs - {chr(code) for code in allowed}
    if missing:
        raise ValueError(f"Uncovered figure glyphs: {missing}")
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(
            path, dpi=400, bbox_inches="tight", pad_inches=0.06, transparent=extension == "svg"
        )
        outputs[extension] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    plt.close(fig)
    return {"outputs": outputs, "glyph_coverage": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT / "aggregate_metrics.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    assert data["new"]["metrics"]["top1_correct"] == 8705
    assert data["new"]["metrics"]["samples"] == 8759
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = style()
    figures = {}
    for function, name in [
        (draw_version_metrics, "model_version_metrics"),
        (draw_threshold_outcomes, "model606_threshold_outcomes"),
        (draw_sample_support, "model606_sample_support"),
    ]:
        figures[name] = export(function(data), name, args.output, fonts)
    manifest = {
        "primary_model_tag": data["primary_model_tag"],
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "fonts": {
            key: {"family": font[0], "file": Path(font[1]).name}
            for key, font in zip(("latin", "chinese"), fonts, strict=True)
        },
        "figures": figures,
    }
    (args.output / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
