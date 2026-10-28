"""Deterministic tar and checksum helpers for public CI deliverables."""

import gzip
import tarfile
from pathlib import Path

if __package__:
    from tools.fetch_models import sha256
else:
    from fetch_models import sha256


def write_checksums(directory: Path) -> None:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(directory).as_posix()}\n" for p in files),
        encoding="utf-8",
    )


def archive_tree(source: Path, target: Path, *, prefix: str = "") -> None:
    """Stable gzip headers and tar ownership/timestamps; retain library symlinks."""
    with (
        target.open("xb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
    ):
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(source.rglob("*")):
                if path.is_dir():
                    continue
                name = path.relative_to(source).as_posix()
                info = tar.gettarinfo(str(path), arcname=f"{prefix}{name}")
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.pax_headers = {}
                if info.isfile():
                    info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                elif info.issym():
                    tar.addfile(info)
                else:
                    raise ValueError(f"Unsupported artifact file: {path}")
