#!/usr/bin/env python3
"""Create model-free deployment archives for Actions, never for model Releases."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

if __package__:
    from tools._artifacts import archive_tree, resolve_documentation_links, write_checksums
    from tools.fetch_models import load_source, sha256
else:
    from _artifacts import archive_tree, resolve_documentation_links, write_checksums
    from fetch_models import load_source, sha256

COMMON = (
    "LICENSE",
    "tools/fetch_models.py",
    "release/model-source.json",
    "release/training.json",
    "docs/API.md",
    "docs/API.zh-CN.md",
    "docs/API.ja.md",
)
PROFILES = ("configs/default.yaml", "configs/n150.yaml")


def copy_files(root: Path, stage: Path, names) -> None:
    for name in names:
        source, target = root / name, stage / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing or unsafe deployment input: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def check_elf(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(20)
    if len(header) < 20 or header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\x3e\x00":
        raise ValueError(f"Expected a Linux amd64 ELF file: {path.name}")


def copy_native(root: Path, stage: Path, runtime: Path, model_dir: Path) -> None:
    for name in ("faceid-server", "faceid-bench"):
        source = runtime / "bin" / name
        check_elf(source)
        copy_files(runtime, stage / "runtime", [f"bin/{name}"])
        (stage / "runtime/bin" / name).chmod(0o755)
    libraries = sorted((runtime / "lib").glob("*.so*"))
    if not libraries or not (runtime / "lib/libfaceid_core.so").is_file():
        raise ValueError("Missing native shared libraries.")
    (stage / "runtime/lib").mkdir(parents=True)
    for source in libraries:
        target = stage / "runtime/lib" / source.name
        if source.is_symlink():
            link = source.readlink()
            if link.is_absolute() or source.resolve().parent != (runtime / "lib").resolve():
                raise ValueError("A native library symlink escapes the runtime.")
            target.symlink_to(link)
        else:
            check_elf(source)
            shutil.copyfile(source, target)
            target.chmod(0o755)
    licenses = [
        p.relative_to(runtime).as_posix() for p in (runtime / "licenses").rglob("*") if p.is_file()
    ]
    if not licenses:
        raise ValueError("Native runtime license notices are missing.")
    copy_files(runtime, stage / "runtime", licenses)
    source_lock = load_source(root / "release/model-source.json")
    expected = {
        n: source_lock["files"][f"models/selected/{n}"]["sha256"]
        for n in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")
    }
    for origin, filename, fast in (
        ("native-n150.json", "native-model.json", True),
        ("native-model.json", "native-compatible.json", False),
    ):
        source = model_dir / origin
        data = json.loads(source.read_text("utf-8"))
        if (
            data["source_assets"] != expected
            or data["detector_sha256"] != source_lock["files"]["models/det_500m.onnx"]["sha256"]
            or data["runtime"].get("flush_denormals", False) is not fast
        ):
            raise ValueError("Native configuration does not match the pinned models/profile.")
        target = stage / "models/selected" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.with_name(filename + ".sha256").write_text(
            f"{sha256(target)}  {filename}\n",
            encoding="utf-8",
        )
    copy_files(root, stage, ("native/deploy/seiyuu-faceid.service",))
    service = stage / "deploy/seiyuu-faceid.service"
    service.parent.mkdir(exist_ok=True)
    (stage / "native/deploy/seiyuu-faceid.service").rename(service)
    (stage / "native/deploy").rmdir()
    (stage / "native").rmdir()


def package_ci_bundle(
    root: Path,
    kind: str,
    output: Path,
    *,
    revision: str,
    runtime: Path | None = None,
    native_models: Path | None = None,
) -> Path:
    if kind not in ("python", "docker", "native") or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A supported bundle kind and complete Git commit SHA are required.")
    platform = "native-linux-amd64" if kind == "native" else kind
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"seiyuu-faceid-{platform}-{revision}.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    with tempfile.TemporaryDirectory(prefix=".ci-bundle-", dir=output) as tmp:
        stage = Path(tmp)
        copy_files(root, stage, COMMON)
        for suffix in ("", ".zh-CN", ".ja"):
            document = (root / f"deploy/ci/{kind}{suffix}.md").read_text("utf-8")
            for language in ("", ".zh-CN", ".ja"):
                document = document.replace(
                    f"]({kind}{language}.md)", f"](README{language}.md)"
                )
            document = document.replace("](../../docs/", "](docs/")
            (stage / f"README{suffix}.md").write_text(document, encoding="utf-8")
        (stage / "BUILD.json").write_text(
            json.dumps(
                {
                    "revision": revision,
                    "deployment": platform,
                    "model_archive_sha256": load_source(root / "release/model-source.json")[
                        "sha256"
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if kind == "python":
            copy_files(root, stage, ("src/seiyuu_faceid/api/test.html",))
            copy_files(
                root,
                stage,
                (
                    *PROFILES,
                    "pyproject.toml",
                    "uv.lock",
                    "app.py",
                    "deploy/seiyuu-faceid.service",
                    "deploy/faceid-native.env.example",
                ),
            )
            copy_files(
                root, stage, [p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")]
            )
        elif kind == "docker":
            copy_files(root, stage, (*PROFILES, "compose.yaml", "compose.n150.yaml"))
            env = (
                (root / "deploy/.env.example")
                .read_text("utf-8")
                .replace(
                    "FACEID_IMAGE=tano26/seiyuu-faceid:latest",
                    f"FACEID_IMAGE=tano26/seiyuu-faceid:sha-{revision}",
                )
            )
            (stage / ".env.example").write_text(env, encoding="utf-8")
        else:
            if runtime is None:
                raise ValueError("A freshly built Linux runtime is required.")
            copy_native(root, stage, runtime, native_models or root / "models/selected")
        resolve_documentation_links(
            stage, sorted(stage.glob("README*.md")) + sorted((stage / "docs").glob("*.md")),
            revision,
        )
        write_checksums(stage)
        archive_tree(stage, archive)
    archive.with_name(archive.name + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n",
        encoding="utf-8",
    )
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--kind", choices=("python", "docker", "native"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--native-models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        package_ci_bundle(
            args.root,
            args.kind,
            args.output,
            revision=args.revision,
            runtime=args.runtime,
            native_models=args.native_models,
        )
    )


if __name__ == "__main__":
    main()
