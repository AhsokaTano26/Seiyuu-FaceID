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
