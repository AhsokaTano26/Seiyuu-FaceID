# Training boundary

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

This directory is intentionally reserved for Phase 5. It may depend on PyTorch,
MPS, and scikit-learn, but runtime code must not import it. Fine-tuning starts
only after a measured pretrained-ONNX baseline and threshold calibration.
