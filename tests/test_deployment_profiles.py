"""The small-CPU profile changes resource policy without changing recognition rules."""

from dataclasses import asdict
from pathlib import Path

import yaml

from seiyuu_faceid.runtime.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_n150_profile_keeps_models_and_accuracy_settings():
    baseline = asdict(load_config(ROOT / "configs/default.yaml"))
    efficient = asdict(load_config(ROOT / "configs/n150.yaml"))
    resource_fields = {
        "intra_op_threads", "inter_op_threads", "parallel_backbones",
        "allow_spinning", "enable_cpu_mem_arena",
        "detector_intra_op_threads",
    }
    assert {k: v for k, v in baseline.items() if k not in resource_fields} == {
        k: v for k, v in efficient.items() if k not in resource_fields
    }
    assert efficient["intra_op_threads"] == 1
    assert efficient["inter_op_threads"] == 1
    assert efficient["parallel_backbones"] is True
    assert efficient["allow_spinning"] is False
    assert efficient["detector_intra_op_threads"] == 2


def test_n150_compose_overrides_config_at_the_existing_mount_target():
    service = yaml.safe_load((ROOT / "compose.n150.yaml").read_text())["services"]["faceid"]
    assert service["environment"]["OMP_NUM_THREADS"] == "1"
    volumes = service["volumes"]
    assert volumes == [{
        "type": "bind", "source": "./configs/n150.yaml",
        "target": "/app/configs/default.yaml", "read_only": True,
        "bind": {"create_host_path": False},
    }]


def test_docker_image_carries_only_the_explicit_target_benchmark_tools():
    dockerfile = (ROOT / "Dockerfile").read_text()
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert "COPY tools/_bootstrap.py tools/benchmark_target.py /app/tools/" in dockerfile
    assert "!tools/benchmark_target.py" in ignore
    assert "!tools/_bootstrap.py" in ignore
    assert "!tools/**" not in ignore
    assert "!configs/n150.yaml" in ignore
