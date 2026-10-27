"""Keep the standalone timing comparison paired and inference-equivalent."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "classifier_benchmark", TOOLS / "benchmark_classifier.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_subset_has_one_crop_per_development_group_and_is_reproducible():
    rows = [{"source_group": group} for group in ["a", "a", "b", "c", "test"]]
    first = benchmark.select_indices(rows, np.array([0, 1, 2, 3]), count=3, seed=17)
    assert set(first) <= {0, 1, 2, 3}
    assert len({rows[i]["source_group"] for i in first}) == 3
    assert first == benchmark.select_indices(rows, np.array([0, 1, 2, 3]), count=3, seed=17)
    with pytest.raises(ValueError, match="source groups"):
        benchmark.select_indices(rows, np.array([0, 1]), count=2, seed=17)


def test_parallel_submits_both_backbones_before_waiting():
    pending = []

    class Executor:
        def submit(self, function, image):
            pending.append((function, image))

            def result():
                assert len(pending) == 2
                return function(image)

            return SimpleNamespace(result=result)

    classifier = SimpleNamespace(
        adaface=SimpleNamespace(embed=lambda image: np.array([1, 2], dtype=np.float32)),
        r50=SimpleNamespace(embed=lambda image: np.array([3, 4], dtype=np.float32)),
    )
    actual = benchmark.parallel_embed(classifier, np.zeros((2, 2, 3)), Executor())
    np.testing.assert_array_equal(
        actual, np.arange(1, 5, dtype=np.float32) / np.float32(np.sqrt(2))
    )


def test_modes_interleave_and_report_probability_parity(monkeypatch):
    order = []

    def embed(image):
        order.append("sequential")
        return np.array([image.item(), 1], dtype=np.float32)

    def parallel(classifier, image, executor):
        order.append("parallel")
        return np.array([image.item(), 1], dtype=np.float32)

    head = SimpleNamespace(predict_proba=lambda features: np.tile([.25, .75], (len(features), 1)))
    monkeypatch.setattr(benchmark, "parallel_embed", parallel)
    report = benchmark.benchmark_modes(
        SimpleNamespace(embed=embed, head=head), [np.array(1), np.array(2)], warmups=0, runs=2
    )
    assert order == ["sequential", "parallel", "parallel", "sequential",
                     "parallel", "sequential", "sequential", "parallel"]
    assert report["parity"]["embedding_exact"]
    assert report["parity"]["probability_exact"]
    assert report["parity"]["top1_matches"] == 4
    assert report["sequential"]["observations"] == 4
    assert report["parallel"]["p95_ms"] >= 0
