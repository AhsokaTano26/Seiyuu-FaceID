#!/usr/bin/env python3
"""Publish only validated model assets; resume drafts and never overwrite public releases."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

if __package__:
    from tools.fetch_models import sha256
    from tools.package_models import ASSET_NAMES
else:
    from fetch_models import sha256
    from package_models import ASSET_NAMES


def publish_models(service, tag: str, revision: str, notes: str, directory: Path) -> str:
    if not re.fullmatch(r"models-[0-9a-f]{16}", tag) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid content tag or source revision.")
    paths = {p.name: p for p in directory.iterdir()}
    if set(paths) != ASSET_NAMES or any(p.is_symlink() or not p.is_file() for p in paths.values()):
        raise ValueError("Release assets must be the exact model-only allowlist.")
    expected = {n: ("sha256:" + sha256(p), p.stat().st_size) for n, p in paths.items()}

    def check(release):
        assets = release["assets"]
        if len({a["name"] for a in assets}) != len(assets):
            raise ValueError("Duplicate remote Release assets.")
        if any(a["name"] not in expected for a in assets):
            raise ValueError("Unexpected remote asset; refusing to alter the Release.")
        for asset in assets:
            want_digest, want_size = expected[asset["name"]]
            if (
                asset.get("state") != "uploaded"
                or asset.get("digest") != want_digest
                or asset.get("size") != want_size
            ):
                raise ValueError("Remote model asset differs; refusing to overwrite.")
        return {a["name"] for a in assets}

    release = service.get(tag)
    if release is None:
        release = service.create(tag, revision, notes)
    # GitHub may retain an empty `starter` asset after an upstream upload 502.
    # Only these incomplete placeholders in a draft may be removed and retried.
    failed = [
        a
        for a in release["assets"]
        if release["draft"]
        and a.get("name") in expected
        and a.get("state") == "starter"
        and a.get("size") == 0
        and type(a.get("id")) is int
    ]
    remaining = [a for a in release["assets"] if a not in failed]
    present = check({**release, "assets": remaining})
    for asset in failed:
        service.delete_failed(tag, asset["id"])
    missing = ASSET_NAMES - present
    if not release["draft"]:
        if missing:
            raise ValueError("Published model Release is incomplete; refusing to modify.")
        return release["html_url"]
    if missing:
        service.upload(tag, [paths[n] for n in sorted(missing)])
    release = service.get(tag)
    if check(release) != ASSET_NAMES:
        raise ValueError("Incomplete model upload; keep the Release as a draft.")
    service.publish(tag)
    return release["html_url"]


class GitHubReleases:
    def __init__(self, repository: str, token: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not token:
            raise ValueError("A repository and GH_TOKEN are required.")
        self.repository = repository
        self.token = token

    def request(
        self, endpoint: str, data: dict | None = None, *, method="GET", namespace="releases"
    ):
        payload = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repository}/{namespace}{endpoint}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "seiyuu-faceid-release",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 204:
                    return None
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404 and method == "GET":
                return None
            raise RuntimeError(f"GitHub Release API failed with HTTP {error.code}.") from None

    def get(self, tag):
        published = self.request(f"/tags/{tag}")
        if published is not None:
            return published
        # The tag endpoint only finds published releases. Authenticated listing
        # also includes drafts, whose tag may not exist as a Git ref yet.
        matches = []
        page = 1
        while True:
            releases = self.request(f"?per_page=100&page={page}")
            if not isinstance(releases, list):
                raise RuntimeError("Could not list GitHub model Release drafts.")
            matches.extend(release for release in releases if release["tag_name"] == tag)
            if len(releases) < 100:
                break
            page += 1
        if len(matches) > 1:
            raise ValueError("Multiple model Release drafts share a tag; refusing to publish.")
        return matches[0] if matches else None

    def create(self, tag, revision, notes):
        return self.request(
            "",
            {
                "tag_name": tag,
                "target_commitish": revision,
                "name": f"FaceID 模型 {tag[7:]}",
                "body": notes,
                "draft": True,
                "prerelease": False,
            },
            method="POST",
        )

    def upload(self, tag, paths):
        # gh streams the large archive; no --clobber, and no shell interpolation.
        subprocess.run(
            ["gh", "release", "upload", tag, "--repo", self.repository, *map(str, paths)],
            check=True,
        )

    def publish(self, tag):
        release = self.get(tag)
        self.request(f"/{release['id']}", {"draft": False, "make_latest": "true"}, method="PATCH")

    def delete_failed(self, tag, asset_id):
        self.request(f"/assets/{asset_id}", method="DELETE")

    def dispatch_model(self, tag: str) -> bool:
        if not re.fullmatch(r"models-[0-9a-f]{16}", tag):
            raise ValueError("Invalid model tag for workflow dispatch.")
        # Repeated app builds reuse a successful or active validation of these model bytes.
        # Failed dispatches remain retryable even after the Release is already public.
        runs = self.request("?per_page=100", namespace="actions/workflows/ci.yml/runs")
        for run in runs["workflow_runs"]:
            event, branch = run.get("event"), run.get("head_branch")
            trusted = (run.get("head_repository") or {}).get("full_name") == self.repository and (
                (event in ("repository_dispatch", "workflow_dispatch") and branch == "main")
                or (event == "release" and branch == tag)
            )
            if trusted and run["display_title"] == f"Model {tag}":
                if run["status"] != "completed" or run["conclusion"] == "success":
                    return False
                break
        self.request(
            "",
            {"event_type": "model-published", "client_payload": {"tag": tag}},
            method="POST",
            namespace="dispatches",
        )
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--dispatch", action="store_true", help="Ensure model deployment CI is triggered"
    )
    args = parser.parse_args()
    service = GitHubReleases(
        os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GH_TOKEN", "")
    )
    tag = json.loads(args.metadata.read_text())["tag"]
    result = publish_models(
        service,
        tag,
        args.revision,
        args.notes.read_text(encoding="utf-8"),
        args.directory,
    )
    print(result)
    if args.dispatch:
        dispatched = service.dispatch_model(tag)
        print("Model CI dispatched." if dispatched else "Model CI already running or successful.")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(f"模型发布：{result}\n")


if __name__ == "__main__":
    main()
