"""Native builds use the pinned official NuGet SDK without GitHub downloads."""

import hashlib
import stat
import zipfile

import pytest


@pytest.fixture
def nuget(tmp_path):
    files = {
        "build/native/include/onnxruntime_c_api.h": b"#define ORT_API_VERSION 29\n",
        "build/native/include/onnxruntime_cxx_api.h": b"C++ API header",
        "build/native/include/onnxruntime_cxx_inline.h": b"C++ inline header",
        "runtimes/linux-x64/native/libonnxruntime.so": b"Linux runtime",
        "runtimes/linux-x64/native/libonnxruntime_providers_shared.so": b"provider library",
        "runtimes/osx-arm64/native/libonnxruntime.dylib": b"macOS runtime",
        "LICENSE": b"MIT license",
        "ThirdPartyNotices.txt": b"upstream notices",
        "Privacy.md": b"upstream privacy notice",
    }
    return tmp_path / "runtime.nupkg", files


def write_package(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return hashlib.sha512(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "platform,filename,content,link",
    [
        ("linux-x64", "libonnxruntime.so.1.29.0", b"Linux runtime", "libonnxruntime.so"),
        ("osx-arm64", "libonnxruntime.1.29.0.dylib", b"macOS runtime", "libonnxruntime.dylib"),
    ],
)
def test_nuget_installs_only_the_requested_platform_with_headers_and_notices(
    nuget, tmp_path, platform, filename, content, link
):
    from tools.fetch_onnxruntime import install_nuget

    package, files = nuget
    digest = write_package(package, files)
    target = tmp_path / "sdk"
    install_nuget(package, target, platform, expected_sha512=digest)
    assert (target / "lib" / filename).read_bytes() == content
    assert (target / "lib" / link).resolve() == (target / "lib" / filename).resolve()
    assert (target / "include/onnxruntime_c_api.h").read_bytes() == b"#define ORT_API_VERSION 29\n"
    for name in ("LICENSE", "ThirdPartyNotices.txt", "Privacy.md"):
        assert (target / name).read_bytes() == files[name]
    assert (target / "VERSION_NUMBER").read_text().strip() == "1.29.0"
    assert not (target / "runtimes").exists()


@pytest.mark.parametrize(
    "damage", ["checksum", "library", "header", "notice", "traversal", "symlink"]
)
def test_invalid_nuget_never_installs_a_partial_sdk(nuget, tmp_path, damage):
    from tools.fetch_onnxruntime import install_nuget

    package, files = nuget
    if damage == "library":
        del files["runtimes/linux-x64/native/libonnxruntime.so"]
    elif damage == "header":
        del files["build/native/include/onnxruntime_c_api.h"]
    elif damage == "notice":
        del files["ThirdPartyNotices.txt"]
    elif damage == "traversal":
        files["build/native/include/../../outside"] = b"escape"
    digest = write_package(package, files)
    if damage == "checksum":
        digest = "0" * 128
    elif damage == "symlink":
        with zipfile.ZipFile(package, "a") as archive:
            link = zipfile.ZipInfo("build/native/include/evil.h")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "../../../outside")
        digest = hashlib.sha512(package.read_bytes()).hexdigest()
    target = tmp_path / "sdk"
    with pytest.raises(ValueError):
        install_nuget(package, target, "linux-x64", expected_sha512=digest)
    assert not target.exists()
    assert not (tmp_path / "outside").exists()
