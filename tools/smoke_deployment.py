#!/usr/bin/env python3
"""Start an unpacked deployment and verify readiness, authentication and image requests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def smoke(root: Path, kind: str, log: Path) -> None:
    root = root.resolve()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    env = dict(
        os.environ,
        SEIYUU_FACEID_API_KEY="ci-local-smoke-key",
        SEIYUU_FACEID_SERVE_UI="1",
        SEIYUU_FACEID_CONFIG=str(root / "configs/n150.yaml"),
        PYTHONPATH=str(root / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if kind == "native":
        command = [
            str(root / "runtime/bin/faceid-server"),
            "-model",
            str(root / "models/selected/native-model.json"),
            "-addr",
            f"127.0.0.1:{port}",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "seiyuu_faceid.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as output:
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{kind} service exited; inspect {log}")
                try:
                    with urllib.request.urlopen(endpoint + "/ready", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{kind} service did not become ready; inspect {log}")
                time.sleep(0.5)
            try:
                urllib.request.urlopen(endpoint + "/v1/model", timeout=10)
            except urllib.error.HTTPError as error:
                if error.code != 401:
                    raise
            else:
                raise RuntimeError("The deployed API did not enforce its configured key.")
            request = urllib.request.Request(
                endpoint + "/v1/model",
                headers={
                    "Authorization": "Bearer ci-local-smoke-key",
                },
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                metadata = json.load(response)
            if kind == "python":
                with urllib.request.urlopen(endpoint + "/", timeout=30) as response:
                    if response.status != 200 or b"<html" not in response.read().lower():
                        raise ValueError("The unpacked Python upload page is missing.")
            manifest = json.loads((root / "models/selected/manifest.json").read_text("utf-8"))
            expected = hashlib.sha256(
                json.dumps(manifest["assets"], sort_keys=True).encode()
            ).hexdigest()
            if metadata["model_id"] != expected:
                raise ValueError("Deployed model identity differs from the verified source.")
            image = io.BytesIO()
            Image.new("RGB", (640, 480), (128, 128, 128)).save(image, format="PNG")
            boundary = "seiyuu-ci-smoke"
            body = (
                (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="file"; filename="blank.png"\r\n'
                    "Content-Type: image/png\r\n\r\n"
                ).encode()
                + image.getvalue()
                + f"\r\n--{boundary}--\r\n".encode()
            )
            for route in ("/v1/detect", "/v1/recognize", "/api/v1/detect"):
                request = urllib.request.Request(
                    endpoint + route,
                    data=body,
                    headers={
                        "Authorization": "Bearer ci-local-smoke-key",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.load(response)
                if result["face_count"] != 0 or result["faces"] != []:
                    raise ValueError("Unexpected blank-image inference result.")
                if route == "/api/v1/detect" and result["has_face"] is not False:
                    raise ValueError("Blank image was reported as containing a face.")
            if process.poll() is not None:
                raise RuntimeError("Service exited during the smoke requests.")
            print(f"{kind}: readiness, model identity, auth, detect and recognize passed.")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("python", "native"), required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.root, args.kind, args.log)


if __name__ == "__main__":
    main()
