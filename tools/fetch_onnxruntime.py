#!/usr/bin/env python3
"""Prepare the ONNX Runtime C++ SDK from Microsoft's pinned official NuGet package."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

VERSION = "1.29.0"
PACKAGE = f"microsoft.ml.onnxruntime.{VERSION}.nupkg"
URL = f"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/{VERSION}/{PACKAGE}"
# Published packageHash from NuGet's catalog, not a digest obtained from a mirror.
SHA512 = (
    "009e9845974a786b6923603b231f82ea1c5f1a84c74f15a8f5f659c7e8f93cd486"
    "3c120f2c0f31fd3a7a5879c1a3d587b6bfb3db732100b4e6464f18b62ebf60"
)


def verify_package(path: Path, expected: str) -> None:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("ONNX Runtime NuGet package SHA-512 mismatch.")


def install_nuget(
    package: Path, destination: Path, platform: str, *, expected_sha512: str = SHA512
) -> None:
    if platform not in ("linux-x64", "osx-arm64"):
        raise ValueError("Unsupported native SDK platform.")
    if destination.exists():
        raise FileExistsError(destination)
    verify_package(package, expected_sha512)
    library = "libonnxruntime.so" if platform == "linux-x64" else "libonnxruntime.dylib"
    runtime = f"runtimes/{platform}/native/"
    prefix = "build/native/include/"
    with zipfile.ZipFile(package) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.filename.startswith(prefix):
                name = info.filename[len(prefix) :]
                if "/" in name or "\\" in name or not name.endswith((".h", ".inc")):
                    raise ValueError("Unsafe ONNX Runtime header path.")
                target = "include/" + name
            elif info.filename in (
                runtime + library,
                runtime + "libonnxruntime_providers_shared.so",
                "LICENSE",
                "ThirdPartyNotices.txt",
                "Privacy.md",
            ):
                target = info.filename.removeprefix(runtime)
            else:
                continue
            if target in members or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Duplicate or linked ONNX Runtime SDK member.")
            members[target] = info
        required = {
            "include/onnxruntime_c_api.h",
            "include/onnxruntime_cxx_api.h",
            "include/onnxruntime_cxx_inline.h",
            library,
            "LICENSE",
            "ThirdPartyNotices.txt",
            "Privacy.md",
        }
        if not required <= members.keys():
            raise ValueError("Incomplete ONNX Runtime SDK headers, library or notices.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".onnxruntime-", dir=destination.parent) as tmp:
            stage = Path(tmp) / "sdk"
            for name, info in members.items():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            libraries = stage / "lib"
            libraries.mkdir()
            if platform == "linux-x64":
                versioned = f"libonnxruntime.so.{VERSION}"
                links = ("libonnxruntime.so", "libonnxruntime.so.1")
            else:
                versioned = f"libonnxruntime.{VERSION}.dylib"
                links = ("libonnxruntime.dylib", "libonnxruntime.1.dylib")
            (stage / library).rename(libraries / versioned)
            for link in links:
                (libraries / link).symlink_to(versioned)
            provider = stage / "libonnxruntime_providers_shared.so"
            if provider.exists():
                provider.rename(libraries / provider.name)
            (stage / "VERSION_NUMBER").write_text(VERSION + "\n", encoding="utf-8")
            stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("linux-x64", "osx-arm64"), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    package = args.cache / PACKAGE
    if not package.exists():
        partial = package.with_suffix(".nupkg.partial")
        print(f"Downloading official ONNX Runtime {VERSION} SDK from NuGet.", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--silent",
                "--show-error",
                URL,
                "-o",
                str(partial),
            ],
            check=True,
        )
        verify_package(partial, SHA512)
        partial.rename(package)
    install_nuget(package, args.destination, args.platform)
    print(f"Prepared verified ONNX Runtime {VERSION} SDK: {args.platform}")


if __name__ == "__main__":
    main()
