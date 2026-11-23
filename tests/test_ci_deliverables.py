"""Exercise the deployment archives and resumable model publisher contracts."""

import hashlib
import io
import json
import tarfile
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b" * 40


@pytest.mark.parametrize("kind", ["python", "docker"])
def test_deployment_archive_has_pinned_model_fetcher_without_private_files(tmp_path, kind):
    from tools.package_ci_deployments import package_ci_bundle

    archive = package_ci_bundle(ROOT, kind, tmp_path, revision=REVISION)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert "tools/fetch_models.py" in names
        assert "release/model-source.json" in names
        assert "SHA256SUMS" in names
        for suffix in ("", ".zh-CN", ".ja"):
            assert f"docs/API{suffix}.md" in names
            readme = tar.extractfile(f"README{suffix}.md").read().decode()
            assert f"](docs/API{suffix}.md)" in readme
            assert "](README.zh-CN.md)" in readme
            assert "](README.ja.md)" in readme
            api = tar.extractfile(f"docs/API{suffix}.md").read().decode()
            assert f"/blob/{REVISION}/docs/DEPLOY_LINUX{suffix}.md)" in api
        assert not any(
            n.endswith((".onnx", ".npz")) or n.startswith(("data/", "images/")) for n in names
        )
        assert not any(n == ".env" or n.startswith(".git/") for n in names)
        if kind == "python":
            assert "src/seiyuu_faceid/api/app.py" in names and "uv.lock" in names
            assert "src/seiyuu_faceid/api/test.html" in names
            assert "deploy/seiyuu-faceid.service" in names
        else:
            assert "compose.yaml" in names and "compose.n150.yaml" in names
            env = tar.extractfile(".env.example").read().decode()
            assert "tano26/seiyuu-faceid:sha-" + REVISION in env
        for line in tar.extractfile("SHA256SUMS").read().decode().splitlines():
            expected, name = line.split("  ")
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected


def elf():
    value = bytearray(64)
    value[:6] = b"\x7fELF\x02\x01"
    value[18:20] = (62).to_bytes(2, "little")
    return bytes(value)


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "licenses").mkdir()
    for filename in ("faceid-server", "faceid-bench"):
        p = root / "bin" / filename
        p.write_bytes(elf())
        p.chmod(0o755)
    (root / "lib/libfaceid_core.so").write_bytes(elf())
    (root / "lib/libonnxruntime.so.1.29.0").write_bytes(elf())
    (root / "lib/libonnxruntime.so.1").symlink_to("libonnxruntime.so.1.29.0")
    (root / "licenses/runtime-LICENSE").write_text("license")
    return root


@pytest.fixture
def native_models(tmp_path):
    source = json.loads((ROOT / "release/model-source.json").read_text())
    directory = tmp_path / "exported-models"
    directory.mkdir()
    document = {
        "model_id": "same-model",
        "source_assets": {
            name: source["files"][f"models/selected/{name}"]["sha256"]
            for name in ("adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz")
        },
        "detector_sha256": source["files"]["models/det_500m.onnx"]["sha256"],
        "runtime": {},
    }
    (directory / "native-model.json").write_text(json.dumps(document))
    document["runtime"]["flush_denormals"] = True
    (directory / "native-n150.json").write_text(json.dumps(document))
    return directory


def test_native_bundle_preserves_libraries_and_offers_fast_and_compatible_configs(
    tmp_path, runtime, native_models
):
    from tools.package_ci_deployments import package_ci_bundle

    archive = package_ci_bundle(
        ROOT,
        "native",
        tmp_path / "out",
        revision=REVISION,
        runtime=runtime,
        native_models=native_models,
    )
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert "runtime/bin/faceid-server" in names
        for suffix in ("", ".zh-CN", ".ja"):
            assert f"docs/API{suffix}.md" in names
            readme = tar.extractfile(f"README{suffix}.md").read().decode()
            assert f"](docs/API{suffix}.md)" in readme
            assert "](README.zh-CN.md)" in readme
            assert "](README.ja.md)" in readme
        assert tar.getmember("runtime/bin/faceid-server").mode == 0o755
        library = tar.getmember("runtime/lib/libonnxruntime.so.1")
        assert library.issym() and library.linkname == "libonnxruntime.so.1.29.0"
        assert "runtime/licenses/runtime-LICENSE" in names
        assert "deploy/seiyuu-faceid.service" in names
        fast = json.load(tar.extractfile("models/selected/native-model.json"))
        compat = json.load(tar.extractfile("models/selected/native-compatible.json"))
        assert fast["runtime"]["flush_denormals"] is True
        assert not compat["runtime"].get("flush_denormals", False)
        assert fast["model_id"] == compat["model_id"]
        assert not any(name.endswith((".onnx", ".npz", ".dylib")) for name in names)


@pytest.mark.parametrize("attack", ["wrong_arch", "external_link"])
def test_native_package_rejects_wrong_platform_and_escaping_libraries(
    tmp_path, runtime, native_models, attack
):
    from tools.package_ci_deployments import package_ci_bundle

    if attack == "wrong_arch":
        (runtime / "bin/faceid-server").write_bytes(b"Mac executable")
    else:
        (runtime / "lib/evil.so").symlink_to("/etc/passwd")
    with pytest.raises(ValueError):
        package_ci_bundle(
            ROOT,
            "native",
            tmp_path / "out",
            revision=REVISION,
            runtime=runtime,
            native_models=native_models,
        )
    assert not list((tmp_path / "out").glob("*.tar.gz"))


class ReleaseService:
    """External API double; product code still validates every artifact and remote digest."""

    def __init__(self, existing=None, fail_upload=False):
        self.release = existing
        self.fail_upload = fail_upload
        self.created = False
        self.published = False
        self.uploaded = []
        self.deleted = []

    def get(self, tag):
        return self.release

    def create(self, tag, revision, notes):
        self.created = True
        self.release = {"draft": True, "assets": [], "html_url": "https://example/model"}
        return self.release

    def upload(self, tag, paths):
        if self.fail_upload:
            raise RuntimeError("network unavailable")
        for path in paths:
            self.uploaded.append(path.name)
            self.release["assets"].append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "state": "uploaded",
                    "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

    def publish(self, tag):
        self.published = True
        self.release["draft"] = False

    def delete_failed(self, tag, asset_id):
        self.deleted.append(asset_id)
        self.release["assets"] = [a for a in self.release["assets"] if a.get("id") != asset_id]


@pytest.fixture
def release_assets(tmp_path):
    from tools.package_models import ASSET_NAMES

    for name in ASSET_NAMES:
        (tmp_path / name).write_bytes(name.encode())
    return tmp_path


def test_publisher_finishes_draft_only_after_all_assets_verify(release_assets):
    from tools.publish_models import publish_models

    service = ReleaseService()
    result = publish_models(service, "models-" + "a" * 16, REVISION, "model notes", release_assets)
    assert result == "https://example/model"
    assert service.created and service.published
    assert len(service.uploaded) == 5
    repeat = ReleaseService(service.release)
    publish_models(repeat, "models-" + "a" * 16, REVISION, "model notes", release_assets)
    assert not repeat.created and not repeat.published and not repeat.uploaded


def test_failed_upload_leaves_release_unpublished_and_can_resume(release_assets):
    from tools.publish_models import publish_models

    failed = ReleaseService(fail_upload=True)
    with pytest.raises(RuntimeError):
        publish_models(failed, "models-" + "a" * 16, REVISION, "notes", release_assets)
    assert failed.created and not failed.published and failed.release["draft"]
    retry = ReleaseService(failed.release)
    publish_models(retry, "models-" + "a" * 16, REVISION, "notes", release_assets)
    assert not retry.created and retry.published


@pytest.mark.parametrize("draft", [True, False])
def test_empty_failed_upload_is_recovered_only_in_a_draft(release_assets, draft):
    from tools.publish_models import publish_models

    service = ReleaseService(
        {
            "draft": draft,
            "assets": [
                {
                    "id": 123,
                    "name": "seiyuu-faceid-models.tar.gz",
                    "state": "starter",
                    "size": 0,
                    "digest": None,
                }
            ],
            "html_url": "https://example/model",
        }
    )
    if draft:
        publish_models(service, "models-" + "a" * 16, REVISION, "notes", release_assets)
        assert service.deleted == [123] and service.published
        assert "seiyuu-faceid-models.tar.gz" in service.uploaded
    else:
        with pytest.raises(ValueError):
            publish_models(service, "models-" + "a" * 16, REVISION, "notes", release_assets)
        assert not service.deleted and not service.uploaded


@pytest.mark.parametrize(
    "condition", ["unexpected_local", "unexpected_remote", "remote_corrupt", "published_incomplete"]
)
def test_publisher_rejects_extra_assets_or_different_published_bytes(release_assets, condition):
    from tools.publish_models import publish_models

    service = ReleaseService()
    if condition == "unexpected_local":
        (release_assets / "deployment.tar.gz").write_bytes(b"not a model")
    else:
        service.release = {"draft": False, "assets": [], "html_url": "https://example/model"}
        if condition == "unexpected_remote":
            service.release["assets"].append({"name": "compose.yaml"})
        elif condition == "remote_corrupt":
            service.release["assets"].append(
                {
                    "name": "TRAINING_SAMPLES.csv",
                    "digest": "sha256:" + "0" * 64,
                    "state": "uploaded",
                    "size": 1,
                }
            )
    with pytest.raises(ValueError):
        publish_models(service, "models-" + "a" * 16, REVISION, "notes", release_assets)
    assert not service.created and not service.published and not service.uploaded


def test_github_adapter_finds_draft_when_tag_endpoint_returns_404(monkeypatch):
    from tools.publish_models import GitHubReleases

    tag = "models-" + "a" * 16
    draft = {"id": 7, "tag_name": tag, "draft": True, "assets": []}

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, **kwargs):
        url = request.full_url
        if url.endswith("/tags/" + tag):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if url.endswith("?per_page=100&page=1"):
            return Response(json.dumps([{"tag_name": f"other-{i}"} for i in range(100)]).encode())
        if url.endswith("?per_page=100&page=2"):
            return Response(json.dumps([draft]).encode())
        raise AssertionError(f"Unexpected GitHub request: {url}")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert GitHubReleases("owner/project", "test-token").get(tag) == draft


def test_real_github_adapter_completes_draft_upload_publish_cycle(release_assets, monkeypatch):
    from tools.publish_models import GitHubReleases, publish_models

    tag = "models-" + "a" * 16
    state = {"release": None, "created": 0}

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, **kwargs):
        url = request.full_url
        release = state["release"]
        if request.get_method() == "POST":
            payload = json.loads(request.data)
            assert payload["draft"] is True and payload["target_commitish"] == REVISION
            state["created"] += 1
            state["release"] = {
                "id": 7,
                "tag_name": tag,
                "draft": True,
                "assets": [],
                "html_url": "https://example/model",
            }
            return Response(json.dumps(state["release"]).encode())
        if request.get_method() == "PATCH":
            assert url.endswith("/releases/7")
            assert json.loads(request.data)["draft"] is False
            release["draft"] = False
            return Response(json.dumps(release).encode())
        if url.endswith("/tags/" + tag):
            if release is None or release["draft"]:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return Response(json.dumps(release).encode())
        if url.endswith("?per_page=100&page=1"):
            return Response(json.dumps([] if release is None else [release]).encode())
        raise AssertionError(f"Unexpected GitHub request: {url}")

    def upload(command, **kwargs):
        assert command[:5] == ["gh", "release", "upload", tag, "--repo"]
        assert command[5] == "owner/project" and "--clobber" not in command
        for filename in command[6:]:
            path = Path(filename)
            state["release"]["assets"].append(
                {
                    "name": path.name,
                    "state": "uploaded",
                    "size": path.stat().st_size,
                    "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("subprocess.run", upload)
    service = GitHubReleases("owner/project", "test-token")
    assert (
        publish_models(service, tag, REVISION, "notes", release_assets) == "https://example/model"
    )
    assert state["created"] == 1 and state["release"]["draft"] is False
    publish_models(service, tag, REVISION, "notes", release_assets)
    assert state["created"] == 1


@pytest.mark.parametrize(
    "existing",
    [
        "success",
        "in_progress",
        "failure",
        "other_model",
        "none",
        "pull_request",
        "fork",
        "feature",
        "manual",
        "release",
    ],
)
def test_model_dispatch_reuses_successful_or_running_validation_and_retries_failures(
    monkeypatch, existing
):
    from tools.publish_models import GitHubReleases

    tag = "models-" + "a" * 16
    requests = []

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, **kwargs):
        requests.append(request)
        if request.get_method() == "GET":
            assert "/actions/workflows/ci.yml/runs?" in request.full_url
            runs = (
                []
                if existing == "none"
                else [
                    {
                        "event": {
                            "pull_request": "pull_request",
                            "manual": "workflow_dispatch",
                            "release": "release",
                        }.get(existing, "repository_dispatch"),
                        "head_repository": {
                            "full_name": "other/fork" if existing == "fork" else "owner/project"
                        },
                        "head_branch": (
                            tag
                            if existing == "release"
                            else "feature"
                            if existing == "feature"
                            else "main"
                        ),
                        "display_title": "Model "
                        + ("models-" + "b" * 16 if existing == "other_model" else tag),
                        "status": "in_progress" if existing == "in_progress" else "completed",
                        "conclusion": "failure" if existing == "failure" else "success",
                    }
                ]
            )
            return Response(json.dumps({"workflow_runs": runs}).encode())
        assert request.full_url == "https://api.github.com/repos/owner/project/dispatches"
        assert json.loads(request.data) == {
            "event_type": "model-published",
            "client_payload": {"tag": tag},
        }
        response = Response()
        response.status = 204
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    dispatched = GitHubReleases("owner/project", "test-token").dispatch_model(tag)
    assert dispatched is (existing not in ("success", "in_progress", "manual", "release"))
    assert len(requests) == (2 if dispatched else 1)
