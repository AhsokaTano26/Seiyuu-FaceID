from __future__ import annotations

import numpy as np
import pytest

from seiyuu_faceid.vision.recognizer import RecognizerError, l2_normalize


def test_l2_normalize_returns_unit_vector() -> None:
    result = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.allclose(result, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(RecognizerError, match="zero-norm"):
        l2_normalize(np.zeros(4, dtype=np.float32))
