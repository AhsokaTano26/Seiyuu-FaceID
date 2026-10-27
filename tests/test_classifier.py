from __future__ import annotations

import numpy as np
import pytest

from seiyuu_faceid.vision.classifier import LinearIdentityHead


def make_head(tmp_path):
    path = tmp_path / "classifier.npz"
    np.savez(
        path,
        mean=np.array([1.0, 2.0]),
        scale=np.array([2.0, 4.0]),
        coefficients=np.array([[1.0, 1.0], [-1.0, 1.0]]),
        intercept=np.array([-1.0, 0.0]),
        class_names=np.array(["zeta", "alpha"]),
    )
    return LinearIdentityHead(path)


def test_head_applies_training_scaler_and_preserves_class_order(tmp_path):
    head = make_head(tmp_path)
    scores = head.predict_proba(np.array([[3.0, 6.0]], dtype=np.float32))
    np.testing.assert_allclose(scores, [[0.7310585786, 0.2689414214]], atol=1e-8)
    candidates = head.candidates(np.array([3.0, 6.0], dtype=np.float32), top_k=2)
    assert [item.identity for item in candidates] == ["zeta", "alpha"]
    assert candidates[0].score == pytest.approx(0.7310585786)


def test_head_softmax_is_stable_and_rejects_invalid_features(tmp_path):
    head = make_head(tmp_path)
    scores = head.predict_proba(np.array([[10001.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(scores, [[1.0, 0.0]])
    with pytest.raises(ValueError, match="features"):
        head.predict_proba(np.array([[np.nan, 2.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="features"):
        head.predict_proba(np.zeros((1, 3), dtype=np.float32))


def fake_classifier(tmp_path, monkeypatch, *, parallel, adaface, r50):
    import hashlib
    import json

    import seiyuu_faceid.vision.classifier as module

    bundle = tmp_path / ("parallel" if parallel else "sequential")
    bundle.mkdir(exist_ok=True)
    for name in ("adaface_ir101.onnx", "w600k_r50.onnx"):
        (bundle / name).write_bytes(name.encode())
    np.savez(
        bundle / "classifier.npz",
        mean=np.zeros(1024),
        scale=np.ones(1024),
        coefficients=np.zeros((2, 1024)),
        intercept=np.zeros(2),
        class_names=np.array(["alice", "bob"]),
    )
    assets = {
        name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        for name in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")
    }
    (bundle / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                feature_order=["adaface_ir101", "w600k_r50"],
                feature_dimensions=1024,
                assets=assets,
                classes=2,
            )
        )
    )

    class Recognizer:
        def __init__(self, path, **kwargs):
            self.operation = adaface if path.name.startswith("adaface") else r50

        def embed(self, image):
            return self.operation(image)

    monkeypatch.setattr(module, "ONNXRecognizer", Recognizer)
    return module.IdentityClassifier(bundle, parallel_backbones=parallel)


def test_parallel_backbones_overlap_preserve_order_dtype_and_reuse_executor(tmp_path, monkeypatch):
    from threading import Barrier, get_ident

    barrier = Barrier(2, timeout=2)
    threads = set()

    def operation(value, concurrent):
        def embed(image):
            if concurrent:
                threads.add(get_ident())
                barrier.wait()
            return np.full(512, value, dtype=np.float32)

        return embed

    sequential = fake_classifier(
        tmp_path, monkeypatch, parallel=False, adaface=operation(1, False), r50=operation(2, False)
    )
    parallel = fake_classifier(
        tmp_path, monkeypatch, parallel=True, adaface=operation(1, True), r50=operation(2, True)
    )
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    try:
        assert sequential._executor is None
        executor = parallel._executor
        expected = sequential.embed(image)
        for _ in range(3):
            actual = parallel.embed(image)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.float32
            assert actual[0] < actual[512]
            assert parallel._executor is executor
        assert len(threads) == 2
        np.testing.assert_array_equal(
            parallel.head.predict_proba(actual[None]), sequential.head.predict_proba(expected[None])
        )
    finally:
        parallel.close()
        sequential.close()
    parallel.close()
    with pytest.raises(RuntimeError, match="closed"):
        parallel.embed(image)
    with pytest.raises(RuntimeError, match="closed"):
        sequential.embed(image)


def test_parallel_failure_waits_for_other_backbone_before_propagating(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Barrier, Event

    started = Barrier(2, timeout=2)
    failed, release, completed = Event(), Event(), Event()

    def fail(image):
        started.wait()
        failed.set()
        raise ValueError("AdaFace failed")

    def finish(image):
        started.wait()
        assert release.wait(2)
        completed.set()
        return np.ones(512, dtype=np.float32)

    classifier = fake_classifier(tmp_path, monkeypatch, parallel=True, adaface=fail, r50=finish)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(classifier.embed, np.zeros((112, 112, 3), dtype=np.uint8))
            try:
                assert failed.wait(2)
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.05)
            finally:
                release.set()
            with pytest.raises(ValueError, match="AdaFace failed"):
                pending.result(timeout=2)
            assert completed.is_set()
    finally:
        classifier.close()
