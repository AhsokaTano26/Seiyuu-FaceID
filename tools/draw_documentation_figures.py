#!/usr/bin/env python3
"""Draw the architecture and accuracy documentation figures with Python functions.

Requires Matplotlib 3.10+ and NumPy; never runs inference or reads source photos.
Example: uv run --no-project --python 3.13 --with matplotlib==3.10.9 --with numpy==2.5.4 \
    python tools/draw_documentation_figures.py
"""
# Cache locations and the Agg backend must be configured before plotting imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seiyuu-doc-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "seiyuu-doc-font-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

PALETTE = {
    "process": ("#eef3f7", "#456d90"),
    "model": ("#f0eff6", "#77678c"),
    "pass": ("#edf4ef", "#628575"),
    "review": ("#fbf2e8", "#ae7948"),
    "data": ("#f4f5f6", "#7b8188"),
}
INK = "#253746"
OUTPUT = ROOT / "docs/figures"
DATA = json.loads((OUTPUT / "accuracy/aggregate_metrics.json").read_text(encoding="utf-8"))
assert DATA["source"]["python_native_metrics_equal"]


def fonts():
    def select(names):
        for name in names:
            try:
                path = font_manager.findfont(
                    font_manager.FontProperties(family=[name]), fallback_to_default=False
                )
                face = FT2Font(path)
                if face.family_name.casefold() == name.casefold():
                    return name, path, set(face.get_charmap())
            except (ValueError, OSError, RuntimeError):
                continue
        raise RuntimeError(f"Required font unavailable: {names}")

    latin = select(
        ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "Nimbus Roman", "Times"]
    )
    cjk = select(["SimSun", "Songti SC", "STSong", "Source Han Serif SC", "Noto Serif CJK SC"])
    return latin, cjk


LATIN, CJK = fonts()
plt.rcParams.update(
    {
        "font.family": [LATIN[0], CJK[0]],
        "font.size": 12,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": True,
        "axes.linewidth": 0.7,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
        "svg.hashsalt": "seiyuu-faceid-doc-figures-v1",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
BOX_TEXT = []


class Diagram:
    """Small explicit-coordinate drawing helper; arrows retain the documented dependencies."""

    def __init__(self, height=7, width=12):
        self.fig = plt.figure(figsize=(8.8, height * 0.66))
        self.ax = self.fig.add_axes([0.025, 0.025, 0.95, 0.95])
        self.ax.set(xlim=(0, width), ylim=(0, height))
        self.ax.axis("off")
        self.nodes = {}

    def node(self, key, x, y, label, w=3.1, h=0.85, kind="process", size=12):
        fill, stroke = PALETTE[kind]
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=0.9,
            edgecolor=stroke,
            facecolor=fill,
            zorder=3,
        )
        self.ax.add_patch(box)
        self.nodes[key] = (x, y, w, h)
        lines = label.split("\n")
        spacing = min(0.29, (h - 0.18) / max(1, len(lines)))
        for i, line in enumerate(lines):
            if "$" in line and re.search(r"[\u3400-\u9fff]", line):
                raise ValueError("Keep CJK labels and MathText on separate lines")
            text = self.ax.text(
                x,
                y + ((len(lines) - 1) / 2 - i) * spacing,
                line,
                fontsize=size,
                color=INK,
                ha="center",
                va="center",
                zorder=4,
            )
            BOX_TEXT.append((self.fig, text, box))
        return key

    def port(self, key, side):
        x, y, w, h = self.nodes[key]
        return {
            "top": (x, y + h / 2),
            "bottom": (x, y - h / 2),
            "left": (x - w / 2, y),
            "right": (x + w / 2, y),
        }[side]

    def edge(self, a, b, start="bottom", end="top", via=(), label=None, dashed=False):
        points = []
        for point in [self.port(a, start), *via, self.port(b, end)]:
            if not points or point != points[-1]:
                points.append(point)
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        self.ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=11,
                lw=0.9,
                color="#687582",
                linestyle="--" if dashed else "-",
                zorder=2,
            )
        )
        if label:
            x, y, text = label
            self.ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=10.5,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                zorder=5,
            )

    def note(self, x, y, text, size=11):
        self.ax.text(x, y, text, fontsize=size, color="#58636e", ha="center", va="center")


def draw_recognition_pipeline():
    d = Diagram(10)
    d.node("upload", 6, 9.3, "上传与格式检查\nJPEG / PNG / WebP", w=4)
    d.node("detect", 6, 7.9, "解码与 SCRFD\n人脸框、检测分数、五点", w=4)
    d.node("empty", 1.9, 7.9, "未检测到脸\n返回空人脸列表", kind="data", w=3)
    d.node("quality", 6, 6.5, "逐脸检查质量\n短边 ≥ 48 px，清晰度 ≥ 20", w=4)
    d.node("bad", 1.9, 5.1, "review\nquality_rejected", kind="review", w=3)
    d.node("align", 6, 5.1, "五点对齐\n112 × 112", w=3.2)
    d.node("features", 6, 3.7, "AdaFace + R50\n单位特征拼接、除以 √2", kind="model", w=4)
    d.node("head", 6, 2.3, "标准化、线性头、softmax\n50 类概率与 Top-5", kind="model", w=4)
    d.node("yes", 3.1, 0.7, "recognized\n接受首位身份", kind="pass", w=3.9)
    d.node("review", 8.7, 0.7, "review / low_probability\n保留候选供复核", kind="review", w=4.4)
    for a, b in [
        ("upload", "detect"),
        ("detect", "quality"),
        ("quality", "align"),
        ("align", "features"),
        ("features", "head"),
    ]:
        d.edge(a, b)
    d.edge("detect", "empty", "left", "right", label=(3.55, 8.48, "无脸"))
    d.edge("quality", "bad", "left", "top", via=((1.9, 6.5),), label=(2.8, 6.75, "不合格"))
    d.edge("head", "yes", via=((6, 1.45), (3.1, 1.45)), label=(3.15, 1.65, "最高概率 ≥ 0.95"))
    d.edge("head", "review", via=((6, 1.45), (8.7, 1.45)), label=(8.75, 1.65, "最高概率 < 0.95"))
    return d.fig


def draw_scrfd_detector():
    d = Diagram(8.5)
    d.node("image", 6, 7.8, "原图像素 → 等比缩放与补边\n640 × 640 RGB 输入", w=5)
    d.node("s8", 2, 6.2, "步长 8\n分数 / 框距离 / 五点", kind="model", w=3.4)
    d.node("s16", 6, 6.2, "步长 16\n分数 / 框距离 / 五点", kind="model", w=3.4)
    d.node("s32", 10, 6.2, "步长 32\n分数 / 框距离 / 五点", kind="model", w=3.4)
    d.node("decode", 6, 4.6, "九组张量共同解码\n网格中心 + 步长", w=4.4)
    d.node("score", 6, 3.2, "保留检测分数 ≥ 0.50\nNMS 抑制 IoU > 0.40 的重叠框", w=5.7)
    d.node("original", 6, 1.8, "撤销补边和缩放\n映射回原图中的框与关键点", w=5, kind="pass")
    for key in ["s8", "s16", "s32"]:
        x = d.nodes[key][0]
        d.edge("image", key, via=((6, 7.05), (x, 7.05)))
        d.edge(key, "decode", via=((x, 5.4), (6, 5.4)))
    d.edge("decode", "score")
    d.edge("score", "original")
    d.note(6, 0.55, "长边 1920 是工作图上限；640 × 640 才是网络输入。")
    return d.fig


def draw_five_point_alignment():
    spec = importlib.util.spec_from_file_location(
        "faceid_alignment", ROOT / "src/seiyuu_faceid/vision/alignment.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    target = mod.ARC_FACE_TEMPLATE_112.astype(float)
    angle = np.deg2rad(22)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = (target - np.array([56, 56])) @ rotation.T * 1.15 + np.array([74, 67])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), layout="constrained")
    for ax, points, limit, caption in [
        (axes[0], source, 150, "(a) 合成原图五点：已知相似变换"),
        (axes[1], target, 112, "(b) 真实 ArcFace 112 模板"),
    ]:
        ax.set_aspect("equal")
        ax.set(xlim=(0, limit), ylim=(limit, 0), xlabel="横向像素 / px", ylabel="纵向像素 / px")
        for i, (x, y) in enumerate(points, 1):
            ax.scatter(x, y, s=24, color="#456d90", marker="o", zorder=3)
            ax.annotate(str(i), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=11)
        ax.plot(points[[0, 1, 2, 0], 0], points[[0, 1, 2, 0], 1], color="#a7b4c0", lw=0.8)
        ax.plot(points[[3, 4], 0], points[[3, 4], 1], color="#a7b4c0", lw=0.8)
        ax.text(0.5, 1.08, caption, transform=ax.transAxes, ha="center", fontsize=11)
        ax.grid(alpha=0.14)
    return fig


def draw_feature_fusion():
    d = Diagram(9.4)
    d.node("crop", 6, 8.7, "112 × 112 BGR 对齐图", w=4.4)
    d.node("a", 2.9, 7.1, "AdaFace IR101\nBGR 像素归一化", w=4.1, kind="model")
    d.node("r", 9.1, 7.1, "R50\n先转换为 RGB，再归一化", w=4.5, kind="model")
    d.node("ua", 2.9, 5.7, "512 维单位特征\nAdaFace", w=4.1)
    d.node("ur", 9.1, 5.7, "512 维单位特征\nR50", w=4.1)
    d.node("f", 6, 4.25, "按 AdaFace、R50 顺序拼接\n除以 √2，得到 1024 维特征", w=6)
    d.node("scale", 6, 2.8, "固定均值与尺度进行标准化\n" r"$x=(f-\mu)/\sigma$", w=5)
    d.node(
        "linear",
        6,
        1.4,
        "线性头：50 × 1024 权重 + 50 偏置\n稳定 softmax → 50 类概率",
        w=7,
        kind="pass",
    )
    for key in ["a", "r"]:
        x = d.nodes[key][0]
        d.edge("crop", key, via=((6, 8), (x, 8)))
    d.edge("a", "ua")
    d.edge("r", "ur")
    for key in ["ua", "ur"]:
        x = d.nodes[key][0]
        d.edge(key, "f", via=((x, 4.95), (6, 4.95)))
    d.edge("f", "scale")
    d.edge("scale", "linear")
    return d.fig


def draw_training_pipeline():
    d = Diagram(7.4)
    d.node("photos", 6, 6.7, "审核照片、身份标签与来源分组", w=6)
    d.node(
        "features",
        6,
        5.3,
        "冻结划分、对齐图和双骨干特征\n骨干权重不参与本地训练",
        w=6,
        kind="model",
    )
    d.node("train", 3, 3.8, "开发池：4064 张\n拟合标准化和逻辑回归", w=4.8)
    d.node("test", 9, 3.8, "历史留出：1051 张\n保留真值用于评估", w=4.6, kind="data")
    d.node("params", 3, 2.3, "classifier.npz\n类别顺序与分类参数", w=4.5)
    d.node("eval", 9, 2.3, "历史留出评估\n检查 Top-1、Top-5 与接受判定", w=4.8, kind="pass")
    d.node("export", 3, 0.8, "原始 ONNX + 校验元信息\n导出 JSON，供 C++ / Go 使用", w=5.4)
    d.edge("photos", "features")
    for key in ["train", "test"]:
        x = d.nodes[key][0]
        d.edge("features", key, via=((6, 4.6), (x, 4.6)))
    d.edge("train", "params")
    d.edge("test", "eval")
    d.edge("params", "eval", "right", "left")
    d.edge("params", "export")
    return d.fig


def draw_data_split():
    d = Diagram(6.5)
    d.node("all", 6, 5.8, "5115 张已审核人脸 / 50 个身份\n按来源组件隔离划分", w=6)
    for key, x, label in [
        ("train", 2, "原训练集\n3574 张"),
        ("val", 6, "原验证集\n490 张"),
        ("test", 10, "历史留出集\n1051 张"),
    ]:
        d.node(key, x, 4.3, label, w=3.2, kind="data")
        d.edge("all", key, via=((6, 5.05), (x, 5.05)))
    d.node("fit", 4, 2.6, "合并 4064 张\n拟合均值、尺度、W、b", w=5)
    d.node("eval", 9.5, 1, "使用冻结参数进行评估\n1035 / 1051 = 98.48%", w=4.7, kind="pass")
    for key in ["train", "val"]:
        x = d.nodes[key][0]
        d.edge(key, "fit", via=((x, 3.45), (4, 3.45)))
    d.edge("test", "eval", via=((10, 2), (9.5, 2)))
    d.edge("fit", "eval", "bottom", "left", via=((4, 1),))
    return d.fig


def draw_denormal_scope():
    d = Diagram(8.4)
    d.node("enter", 6, 7.7, "Session 构造或 Model::run\nLinux x86_64 支持检查已通过", w=6)
    d.node("save", 6, 6.25, "保存本线程完整 MXCSR", w=4.8)
    d.node("on", 2.8, 4.8, "配置开启\n设置 DAZ / FTZ", w=4.2, kind="model")
    d.node("off", 9.2, 4.8, "配置关闭\n清除 DAZ / FTZ", w=4.2, kind="data")
    d.node("run", 6, 3.3, "执行 ORT 构造或推理\nSession 选项另行设置其线程池", w=6)
    d.node("restore", 6, 1.85, "正常返回或异常展开\nRAII 恢复完整 MXCSR", w=5.5, kind="pass")
    d.edge("enter", "save")
    for key in ["on", "off"]:
        x = d.nodes[key][0]
        d.edge("save", key, via=((6, 5.5), (x, 5.5)))
        d.edge(key, "run", via=((x, 4.05), (6, 4.05)))
    d.edge("run", "restore")
    d.note(6, 0.5, "每次调用分别设置和恢复；图像预处理与分类头在此作用域外。")
    return d.fig


def draw_native_runtime():
    d = Diagram(8)
    d.node("client", 6, 7.3, "HTTP 客户端", w=3.4, kind="data")
    d.node("go", 6, 5.9, "Go：路由、鉴权、上传限制\n有界槽位与单推理门", w=6)
    d.node("cabi", 6, 4.5, "同一进程地址空间\ncgo → C ABI", w=5)
    d.node("cpp", 6, 3.1, "C++：解码、检测、对齐\n特征融合、分类与阈值决策", w=6, kind="model")
    d.node("models", 2.5, 1.5, "常驻 ORT Session\nSCRFD / AdaFace / R50", w=4.4, kind="data")
    d.node("json", 9, 1.5, "C ABI 结果 JSON\nGo 组织响应并释放缓冲区", w=5, kind="pass")
    for a, b in [("client", "go"), ("go", "cabi"), ("cabi", "cpp")]:
        d.edge(a, b)
    d.edge("cpp", "models", via=((6, 2.3), (2.5, 2.3)))
    d.edge("cpp", "json", via=((6, 2.3), (9, 2.3)))
    d.edge(
        "json",
        "go",
        "right",
        "right",
        via=((11.8, 1.5), (11.8, 5.9)),
        label=(10.5, 6.18, "返回结果"),
    )
    return d.fig


def draw_inference_parallel():
    d = Diagram(6.8)
    # Columns express execution stages, not elapsed-time proportions.
    for y, label in [(5.8, "Go HTTP"), (4.1, "C++ 调用线程"), (2.35, "持久 R50 worker")]:
        d.ax.axhline(y - 0.68, color="#e6e9ec", lw=0.7, zorder=0)
        d.ax.text(0.15, y, label, ha="left", va="center", fontsize=11, color=INK)
    d.node("request", 3.25, 5.8, "上传校验\n取得推理门", w=2.5)
    d.node("response", 10, 5.8, "释放结果缓冲区\n返回响应", w=3.2, kind="pass")
    d.node("prep", 3.25, 4.1, "解码、检测\n质量检查、对齐", w=2.6)
    d.node("ada", 6.5, 4.1, "AdaFace Run", w=2.6, kind="model")
    d.node("r50", 6.5, 2.35, "R50 Run", w=2.6, kind="model")
    d.node("join", 10, 4.1, "等待双骨干完成\n拼接、分类、决策", w=3.2)
    d.edge("request", "prep")
    d.edge("prep", "ada", "right", "left")
    d.edge("prep", "r50", "bottom", "left", via=((3.25, 2.35),))
    d.edge("ada", "join", "right", "left")
    d.edge("r50", "join", "right", "bottom", via=((10, 2.35),))
    d.edge("join", "response", "top", "bottom")
    d.note(6.5, 1.1, "同一张对齐图供两个骨干读取；分类要等待两段特征齐备。")
    d.note(6, 0.4, "布局只表示依赖与并行关系，不表示实测耗时比例。", size=10.5)
    return d.fig


def draw_request_lifecycle():
    d = Diagram(8.4)
    d.node("request", 6, 7.7, "推理请求到达", w=4)
    d.node("slot", 6, 6.3, "取得在途槽位？\n解析、等待和处理合计上限 8", w=5.3)
    d.node("busy", 1.6, 6.3, "槽位已满\n503 server_busy", w=2.7, kind="review")
    d.node("parse", 6, 4.9, "解析上传并检查输入\n文件、格式与像素限制", w=5.3)
    d.node("wait", 6, 3.5, "等待单推理门\n最多 1 个请求执行原生推理", w=5.3)
    d.node("cancel", 1.8, 2.05, "等待期间取消\n408 + 资源清理", w=3, kind="review")
    d.node("run", 8, 2.05, "取得执行资格\n调用 C++ / ORT", w=5)
    d.node("done", 8, 0.65, "当前调用完成\n释放推理门、槽位和临时文件", w=5.9, kind="pass")
    for a, b in [("request", "slot"), ("slot", "parse"), ("parse", "wait")]:
        d.edge(a, b)
    d.edge("slot", "busy", "left", "right")
    d.edge("wait", "cancel", "left", "top", via=((1.8, 3.5),))
    d.edge("wait", "run", via=((6, 2.8), (8, 2.8)))
    d.edge("run", "done")
    return d.fig


def draw_model_startup():
    d = Diagram(8)
    d.node("start", 6, 7.3, "Go 服务启动", w=4)
    d.node("json", 6, 5.9, "读取导出 JSON 与 .sha256\n验证 JSON 内容摘要", w=6)
    d.node(
        "assets",
        2.5,
        4.4,
        "AdaFace / R50 / classifier.npz\n以及检测器 ONNX",
        w=4.5,
        kind="data",
        size=11,
    )
    d.node("verify", 8.8, 4.4, "验证模型标识\n及四项来源资产摘要", w=4.9)
    d.node("fail", 2.5, 2.8, "校验失败\n退出启动", w=4, kind="review")
    d.node("engine", 8.8, 2.8, "全部通过\n创建 C++ 引擎、读取分类参数", w=5)
    d.node("ready", 8.8, 1.1, "三个 ORT Session 初始化成功\n/ready 返回 ready", w=5.2, kind="pass")
    d.edge("start", "json")
    d.edge("json", "verify", via=((6, 5.15), (8.8, 5.15)))
    d.edge("assets", "verify", "right", "left")
    d.edge("verify", "fail", "bottom", "right", via=((8.8, 3.6), (4.9, 3.6), (4.9, 2.8)))
    d.edge("verify", "engine")
    d.edge("engine", "ready")
    return d.fig


def draw_benchmark_tasks():
    d = Diagram(6.4)
    for y, input_text, task, metric in [
        (
            5.3,
            "LFW：6000 个规定配对\n涉及 7701 张不同图片",
            "双骨干特征的同人验证\n每折独立选择距离阈值",
            "验证准确率\n5948 / 6000 = 99.13%",
        ),
        (
            3.2,
            "WIDER：3226 张验证图\n官方难度标注",
            "SCRFD 检测\n按 IoU 与真实框匹配",
            "Easy / Medium / Hard AP\n85.49 / 79.48 / 48.84",
        ),
        (
            1.1,
            "声优：1051 张留出人脸\n对应已训练的 50 类",
            "双骨干 + 原分类头\n与真实身份标签比较",
            "Top-1 分类准确率\n1035 / 1051 = 98.48%",
        ),
    ]:
        key = str(y)
        d.node(key + "a", 2, y, input_text, w=3.55, kind="data", size=11)
        d.node(key + "b", 6, y, task, w=3.6, size=11)
        d.node(key + "c", 10, y, metric, w=3.6, kind="pass", size=11)
        d.edge(key + "a", key + "b", "right", "left")
        d.edge(key + "b", key + "c", "right", "left")
    return d.fig


def draw_lfw_protocol():
    d = Diagram(8.7)
    d.node("pairs", 6, 8, "官方 pairs.txt：6000 对\n10 折，每折 300 同人 + 300 异人", w=7)
    d.node("train", 2.9, 6.4, "其余 9 折：5400 对\n特征距离 + 训练折真值", w=4.9, kind="data")
    d.node("test", 9.1, 6.4, "当前测试折：600 对\n仅计算特征距离", w=4.9, kind="data")
    d.node("choose", 2.9, 4.85, "只用训练折选择阈值\n0, 0.01, …, 3.99", w=4.9)
    d.node("predict", 9.1, 4.85, "使用冻结阈值\n判断同人 / 异人", w=4.9)
    d.node("truth", 2.9, 3.2, "当前测试折真值\n只进入评分步骤", w=4.9, kind="data")
    d.node("score", 9.1, 3.2, "核对 600 对判定\n漏检涉及的配对计错", w=4.9)
    d.node(
        "repeat",
        6,
        1.55,
        "依次轮换全部 10 折\n合计 5948 / 6000，报告折间标准差",
        w=7.9,
        kind="pass",
    )
    for key in ["train", "test"]:
        x = d.nodes[key][0]
        d.edge("pairs", key, via=((6, 7.2), (x, 7.2)))
    d.edge("train", "choose")
    d.edge("test", "predict")
    d.edge("choose", "predict", "right", "left", label=(6, 5.1, "距离阈值"))
    d.edge("predict", "score")
    d.edge("truth", "score", "right", "left")
    d.edge("score", "repeat", via=((9.1, 2.4), (6, 2.4)))
    d.note(6, 0.45, "测试折真值不进入阈值选择；声优 50 类分类头不参与本任务。")
    return d.fig


def draw_lfw_errors():
    lfw = DATA["lfw"]["models"]["fusion"]
    missed = lfw["missing_pairs"]
    total = lfw["pairs"]
    correct = lfw["correct"]
    tp = sum(f["tp"] for f in lfw["folds"])
    tn = sum(f["tn"] for f in lfw["folds"])
    fp = sum(f["fp"] for f in lfw["folds"])
    fn = sum(f["fn"] for f in lfw["folds"])
    assert tp + tn == correct and fp + fn + missed + correct == total
    d = Diagram(7.2)
    d.node("all", 6, 6.5, f"全部 {total} 对\n不删除漏检样本", w=5)
    d.node("miss", 2.6, 4.9, f"漏检涉及 {missed} 对\n全部计错", w=4.5, kind="review")
    d.node("valid", 9.3, 4.9, f"检测成功 {total - missed} 对\n进行特征距离判断", w=4.7)
    d.node("correct", 6.3, 3.3, f"判对 {correct} 对\nTP {tp} + TN {tn}", w=4.1, kind="pass")
    d.node("error", 9.8, 1.65, f"判错 {fp + fn} 对\nFN {fn} + FP {fp}", w=3.2, kind="review")
    d.node(
        "total",
        3.2,
        1.65,
        f"总错误 {total - correct} 对\n漏检 {missed} + 判错 {fp + fn}",
        w=4.8,
        kind="review",
    )
    d.edge("all", "miss", via=((6, 5.7), (2.6, 5.7)))
    d.edge("all", "valid", via=((6, 5.7), (9.3, 5.7)))
    d.edge("valid", "correct", via=((9.3, 4.1), (6.3, 4.1)))
    d.edge("valid", "error", "right", "right", via=((11.8, 4.9), (11.8, 1.65)))
    d.edge("miss", "total", via=((2.6, 3.6), (3.2, 3.6)))
    d.edge("error", "total", "left", "right")
    d.note(6, 0.45, "统计单位是“配对”：10 张漏检图片影响 18 对。")
    return d.fig


def draw_lfw_fold_accuracy():
    rows = DATA["lfw"]["models"]["fusion"]["folds"]
    x = np.arange(1, 11)
    y = np.array([r["accuracy"] for r in rows]) * 100
    mean = DATA["lfw"]["models"]["fusion"]["accuracy"] * 100
    assert np.isclose(y.mean(), mean, rtol=0, atol=1e-12)
    fig, ax = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    ax.plot(x, y, "o-", color="#456d90", ms=5, label="Python 与 C++ 判定一致")
    ax.axhline(mean, color="#777777", ls="--", lw=1, label=f"十折平均 {mean:.2f}%")
    for i, value in zip(x, y, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (i, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set(
        xlabel="测试折编号（每折 600 对）",
        ylabel="验证准确率 / %",
        xticks=x,
        xlim=(0.5, 10.5),
        ylim=(97, 100.25),
    )
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    return fig


def draw_wider_protocol():
    d = Diagram(8.2)
    d.node("images", 2.8, 7.5, "3226 张验证图\n当前部署参数运行 SCRFD", w=4.7, kind="data")
    d.node("gt", 9.2, 7.5, "官方真实框与难度索引\nEasy / Medium / Hard", w=4.7, kind="data")
    d.node("pred", 2.8, 5.9, "保留 16264 个检测框\n全验证集分数归一化", w=4.7)
    d.node("difficulty", 9.2, 5.9, "分别确定当前难度目标\n其他标注按忽略规则处理", w=4.7)
    d.node("match", 6, 4.3, "IoU ≥ 0.5 匹配\n区分命中、重复检测与忽略框", w=6)
    d.node("pr", 6, 2.7, "1000 档阈值累积 PR\n计算精度包络积分 AP", w=6)
    d.node(
        "ap",
        6,
        1.1,
        "Easy 85.49 / Medium 79.48 / Hard 48.84\n两种实现的全部 PR 档位一致",
        w=9,
        kind="pass",
    )
    d.edge("images", "pred")
    d.edge("gt", "difficulty")
    for key in ["pred", "difficulty"]:
        x = d.nodes[key][0]
        d.edge(key, "match", via=((x, 5.1), (6, 5.1)))
    d.edge("match", "pr")
    d.edge("pr", "ap")
    return d.fig


def draw_wider_pr_curves():
    fig, ax = plt.subplots(figsize=(8.8, 4.7), layout="constrained")
    for key, color, style, marker in [
        ("easy", "#456d90", "-", "o"),
        ("medium", "#628575", "--", "s"),
        ("hard", "#b97a48", "-.", "^"),
    ]:
        row = DATA["pr_curves"][key]
        recall = np.asarray(row["recall"])
        precision = np.asarray(row["precision"])
        ap = DATA["wider"]["subsets"][key]["ap"] * 100
        assert (
            len(recall) == len(precision) == 1000
            and np.isfinite(recall).all()
            and np.isfinite(precision).all()
        )
        assert np.all(np.diff(recall) >= 0)
        padded_recall = np.r_[0.0, recall, 1.0]
        envelope = np.maximum.accumulate(np.r_[0.0, precision, 0.0][::-1])[::-1]
        changes = np.flatnonzero(np.diff(padded_recall) != 0)
        reconstructed_ap = np.sum(np.diff(padded_recall)[changes] * envelope[changes + 1])
        assert np.isclose(reconstructed_ap * 100, ap, rtol=0, atol=1e-10)
        ax.plot(
            recall,
            precision,
            color=color,
            ls=style,
            lw=1.4,
            label=f"{key.capitalize()}  AP = {ap:.2f}",
        )
        ax.plot(recall[-1], precision[-1], marker=marker, color=color, ms=5)
        ax.annotate(
            f"{recall[-1] * 100:.2f}%",
            (recall[-1], precision[-1]),
            xytext=(0, -20),
            ha="center",
            textcoords="offset points",
            fontsize=10,
            color=color,
        )
    ax.set(xlabel="召回率 Recall", ylabel="精度 Precision", xlim=(0, 1), ylim=(0, 1.03))
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left")
    return fig


def draw_holdout_decisions():
    h = DATA["holdout"]
    accepted = h["accepted"]
    wrong = h["accepted_wrong"]
    correct = h["top1_correct"]
    accepted_correct = accepted - wrong
    review_correct = correct - accepted_correct
    review_wrong = h["review"] - review_correct
    d = Diagram(6.5)
    d.node("all", 6, 5.8, f"已对齐历史留出人脸 {h['images']} 张\n质量过滤不在此裁剪分类实验内", w=7)
    d.node("auto", 3, 4.2, f"最高概率 ≥ 0.95\n自动接受 {accepted} 张", w=4.9, kind="pass")
    d.node("review", 9, 4.2, f"最高概率 < 0.95\n人工复核 {h['review']} 张", w=4.9, kind="review")
    d.node("autoc", 1.7, 2.6, f"Top-1 正确\n{accepted_correct} 张", w=2.7, kind="pass")
    d.node("autow", 4.5, 2.6, f"Top-1 错误\n{wrong} 张", w=2.5, kind="review")
    d.node("reviewc", 7.6, 2.6, f"Top-1 正确\n{review_correct} 张", w=2.7, kind="data")
    d.node("revieww", 10.5, 2.6, f"Top-1 错误\n{review_wrong} 张", w=2.5, kind="data")
    for key in ["auto", "review"]:
        x = d.nodes[key][0]
        d.edge("all", key, via=((6, 5), (x, 5)))
    for a, children in [("auto", ["autoc", "autow"]), ("review", ["reviewc", "revieww"])]:
        for key in children:
            x = d.nodes[key][0]
            d.edge(a, key, via=((d.nodes[a][0], 3.4), (x, 3.4)))
    d.note(
        6,
        1.15,
        f"总 Top-1 正确：{accepted_correct} + {review_correct} = {correct} / {h['images']}"
        f" = {h['top1_accuracy'] * 100:.2f}%",
    )
    d.note(6, 0.45, "复核样本的 Top-1 可与标签核对；转人工本身不等于自动识别成功。", size=10.5)
    return d.fig


def draw_validation_evidence():
    d = Diagram(8.7)
    d.node("fixed", 6, 8, "固定输入、模型资产与参数\n记录 SHA-256", w=6, kind="data")
    d.node("py", 2.9, 6.45, "Python 独立进程\n特征、检测框、分类概率", w=4.9)
    d.node("cpp", 9.1, 6.45, "C++ 独立进程\n特征、检测框、分类概率", w=4.9)
    d.node("saved", 6, 4.9, "保存两组原始预测结果\n之后分别核对正确性与一致性", w=6, kind="data")
    d.node("truth", 1.8, 3.3, "真实标签\n与官方检测框", w=3, kind="data")
    d.node("accuracy", 6, 3.3, "对真值评分\n计算准确率和 AP", w=4.1, kind="pass")
    d.node("parity", 9.6, 1.65, "对两个实现比较\n检查数值和决策是否变化", w=4.5)
    d.node("audit", 3.2, 0.7, "独立参考复算\n十折与全部 PR 档位核对", w=5.3, kind="pass")
    for key in ["py", "cpp"]:
        x = d.nodes[key][0]
        d.edge("fixed", key, via=((6, 7.2), (x, 7.2)))
        d.edge(key, "saved", via=((x, 5.65), (6, 5.65)))
    d.edge("truth", "accuracy", "right", "left")
    d.edge("saved", "accuracy")
    d.edge("saved", "parity", "right", "top", via=((9.6, 4.9),))
    d.edge("accuracy", "audit", via=((6, 1.8), (3.2, 1.8)))
    return d.fig


FIGURES = {
    "recognition_pipeline": ("architecture", draw_recognition_pipeline),
    "scrfd_detector": ("architecture", draw_scrfd_detector),
    "five_point_alignment": ("architecture", draw_five_point_alignment),
    "feature_fusion": ("architecture", draw_feature_fusion),
    "training_pipeline": ("architecture", draw_training_pipeline),
    "data_split": ("architecture", draw_data_split),
    "denormal_scope": ("architecture", draw_denormal_scope),
    "native_runtime": ("architecture", draw_native_runtime),
    "inference_parallel": ("architecture", draw_inference_parallel),
    "request_lifecycle": ("architecture", draw_request_lifecycle),
    "model_startup": ("architecture", draw_model_startup),
    "benchmark_tasks": ("accuracy", draw_benchmark_tasks),
    "lfw_protocol": ("accuracy", draw_lfw_protocol),
    "lfw_errors": ("accuracy", draw_lfw_errors),
    "lfw_fold_accuracy": ("accuracy", draw_lfw_fold_accuracy),
    "wider_protocol": ("accuracy", draw_wider_protocol),
    "wider_pr_curves": ("accuracy", draw_wider_pr_curves),
    "holdout_decisions": ("accuracy", draw_holdout_decisions),
    "validation_evidence": ("accuracy", draw_validation_evidence),
}


def validate_figure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    errors = []
    for candidate, text, box in BOX_TEXT:
        if candidate is not fig:
            continue
        inner = box.get_window_extent(renderer)
        extent = text.get_window_extent(renderer)
        if (
            extent.x0 < inner.x0 + 2
            or extent.x1 > inner.x1 - 2
            or extent.y0 < inner.y0 + 2
            or extent.y1 > inner.y1 - 2
        ):
            errors.append(text.get_text())
    union = LATIN[2] | CJK[2]
    for text in fig.findobj(matplotlib.text.Text):
        content = text.get_text()
        if "$" in content:
            continue
        missing = {c for c in content if not c.isspace() and ord(c) not in union}
        if missing:
            errors.append("missing glyphs: " + "".join(sorted(missing)))
    if errors:
        raise ValueError("Figure text does not fit or glyphs are absent: " + repr(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=FIGURES)
    args = parser.parse_args()
    selected = args.only or list(FIGURES)
    records = []
    print(f"Fonts: Latin={LATIN[0]}; CJK={CJK[0]} (preferred SimSun)")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for name in selected:
            group, draw = FIGURES[name]
            fig = draw()
            validate_figure(fig)
            folder = OUTPUT / group
            folder.mkdir(parents=True, exist_ok=True)
            for suffix in ("svg", "png", "pdf"):
                metadata = (
                    {"Date": None}
                    if suffix == "svg"
                    else (
                        {
                            "CreationDate": None,
                            "ModDate": None,
                            "Creator": "Seiyuu-FaceID documentation figures",
                        }
                        if suffix == "pdf"
                        else None
                    )
                )
                path = folder / f"{name}.{suffix}"
                fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08, metadata=metadata)
                if suffix == "svg":
                    path.write_text(
                        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                    )
            records.append(
                {"name": name, "group": group, "function": draw.__name__, "text_fit_checked": True}
            )
            plt.close(fig)
            print(f"Rendered {group}/{name}: SVG, 300-dpi PNG, PDF")
        if captured:
            for warning in captured:
                print(str(warning.message))
            raise RuntimeError("Resolve rendering warnings before delivery")
    if not args.only:
        manifest = {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "font_family_order": [LATIN[0], CJK[0]],
            "cjk_fallback": CJK[0] != "SimSun",
            "latin_fallback": LATIN[0] != "Times New Roman",
            "aggregate_data_sha256": hashlib.sha256(
                (OUTPUT / "accuracy/aggregate_metrics.json").read_bytes()
            ).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "figures": records,
        }
        (OUTPUT / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
