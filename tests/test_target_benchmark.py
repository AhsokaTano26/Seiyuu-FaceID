"""Resource reports remain comparable and do not publish private model outputs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("benchmark_target", TOOLS / "benchmark_target.py")
benchmark = importlib.util.module_from_spec(SPEC)


@pytest.fixture(autouse=True)
def load_tool():
    SPEC.loader.exec_module(benchmark)


def test_metrics_include_interpolated_percentile_and_core_usage():
    result = benchmark.summarize([1, 2, 3, 4], [2, 4, 6, 8])
    assert result["wall_ms"] == pytest.approx({"mean": 2.5, "median": 2.5, "p95": 3.85})
    assert result["cpu_ms"]["mean"] == 5
    assert result["cpu_core_equivalents"] == 2
    assert benchmark.rss_mib(2048, "Linux") == 2
    assert benchmark.rss_mib(2097152, "Darwin") == 2


def test_parity_checks_ranking_threshold_and_feature_tolerance():
    reference = {"embeddings": [[1, 0]], "probabilities": [[0.95, 0.05]]}
    same = benchmark.compare_parity(reference, reference, 0.95)
    assert same["top1_matches"] == same["top5_matches"] == same["threshold_matches"] == 1
    changed = {"embeddings": [[1, 1e-7]], "probabilities": [[0.9499999, 0.0500001]]}
    result = benchmark.compare_parity(reference, changed, 0.95)
    assert result["embedding_allclose"]
    assert not result["embedding_exact"]
    assert result["threshold_matches"] == 0
    assert np.isclose(result["embedding_max_abs_difference"], 1e-7)


def test_supervisor_uses_fresh_children_and_rotates_order(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        return SimpleNamespace(stdout="{\"ok\": true}")

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    results = benchmark.run_children({"repeats": 2})
    assert len(results) == len(calls) == 16
    assert calls[0][1]["profile"]["name"] == "baseline2seqspinOn"
    assert calls[8][1]["profile"]["name"] == "current2parspinOn"
    assert all(command == [sys.executable, str(TOOLS / "benchmark_target.py"), "--worker"]
               for command, _ in calls)


def worker_rows():
    rows = []
    for profile in benchmark.PROFILES:
        rows.append({"profile": {"name": profile[0]}, "model_id": "model-hash",
                     "corpus_sha256": "pixel-hash", "samples": 1, "classes": 2,
                     "probability_threshold": 0.95, "platform": "test", "cpu_model": "test",
                     "logical_cpus": 4, "python": "test", "versions": {},
                     "embeddings": [[1, 0]], "probabilities": [[0.96, 0.04]],
                     "wall_ms": {"mean": 3}, "cpu_ms": {"mean": 4}, "peak_rss_mib": 100})
    return rows


def test_report_preserves_metadata_but_discards_private_arrays():
    spec = {"samples": 1, "runs": 1, "warmups": 1, "repeats": 1,
            "idle_seconds": 0, "seed": 1, "synthetic": True,
            "crop_dir": "/private/crops", "config": "/private/config"}
    report = benchmark.build_report(spec, worker_rows())
    assert report["model_id"] == "model-hash"
    assert report["results"][0]["parity"]["threshold_matches"] == 1
    serialized = json.dumps(report)
    assert "embeddings" not in serialized
    assert "probabilities" not in serialized
    assert "/private" not in serialized


def test_main_stdout_is_sanitized_json_identical_to_saved_report(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv",
                        ["benchmark_target.py", "--synthetic", "--output", str(output)])
    monkeypatch.setattr(benchmark, "run_children", lambda spec: worker_rows())
    assert benchmark.main() == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == json.loads(output.read_text())
    assert "embeddings" not in stdout
    assert "probabilities" not in stdout


def test_worker_failure_surfaces_reason_without_child_traceback(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="Traceback:\nValueError: bad crop\n")

    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    with pytest.raises(RuntimeError,
                       match="baseline2seqspinOn, repeat 0 failed: ValueError: bad crop"):
        benchmark.run_children({"repeats": 1})


def test_cli_help_and_existing_output_do_not_load_models(tmp_path):
    command = [sys.executable, str(TOOLS / "benchmark_target.py")]
    help_result = subprocess.run(command + ["--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    output = tmp_path / "existing.json"
    output.write_text("keep")
    result = subprocess.run(command + ["--synthetic", "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "exists" in result.stderr
    assert output.read_text() == "keep"


@pytest.mark.parametrize("extra", [["--samples", "65"], ["--idle-seconds", "nan"],
                                        ["--idle-seconds", "6"], ["--runs", "0"]])
def test_cli_rejects_unbounded_requests(tmp_path, extra):
    result = subprocess.run([sys.executable, str(TOOLS / "benchmark_target.py"),
                             "--synthetic", "--output", str(tmp_path / "new.json"), *extra],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "new.json").exists()
