"""AT-233. The temporary Atlas cluster the cluster tiers run against.

``scripts/atlas_ci_cluster.py`` creates and deletes real, billed infrastructure, so what it decides is tested
here with a fake Atlas CLI: what it creates, what it deletes and refuses to delete, and what it prints.
Whether Atlas accepts the request is the weekly run's job.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "atlas_ci_cluster.py"
NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture
def script() -> ModuleType:
    """The script, imported from its path: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("atlas_ci_cluster", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeAtlas:
    """Record every Atlas CLI call and answer the three that return something."""

    def __init__(self, *, clusters: Sequence[dict[str, Any]] = (), fail: dict[str, str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.created: dict[str, Any] | None = None
        self.clusters = list(clusters)
        self.fail = fail or {}

    def __call__(self, arguments: Sequence[str]) -> str:
        call = list(arguments)
        self.calls.append(call)
        for prefix, stderr in self.fail.items():
            if " ".join(call).startswith(prefix):
                raise subprocess.CalledProcessError(1, ["atlas", *call], output="", stderr=stderr)
        if call[:2] == ["clusters", "list"]:
            return json.dumps({"results": self.clusters})
        if call[:2] == ["clusters", "create"]:
            self.created = json.loads(Path(call[call.index("--file") + 1]).read_text())
        if call[:3] == ["clusters", "connectionStrings", "describe"]:
            return json.dumps({"standardSrv": "mongodb+srv://strands-ci-1.abcde.mongodb.net"})
        return ""

    def named(self, *prefix: str) -> list[list[str]]:
        return [call for call in self.calls if call[: len(prefix)] == list(prefix)]


@pytest.fixture
def atlas(script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeAtlas:
    """A fake CLI, no real search wait, and GitHub's output file in a temporary directory."""
    monkeypatch.setattr(script, "_wait_for_search", lambda uri: None)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("ATLAS_CI_CLUSTER", "strands-ci-42-1")
    return FakeAtlas()


class TestWhatUpCreates:
    """AT-233. An M10 on the newest release, with what automated embedding needs, and nothing that outlives it."""

    def test_the_cluster_is_an_m10_on_the_newest_release_with_disk_auto_scaling(
        self, script: ModuleType, atlas: FakeAtlas
    ) -> None:
        script.up(atlas, now=NOW)

        config = atlas.created
        assert config is not None and config["name"] == "strands-ci-42-1"
        assert config["versionReleaseSystem"] == "CONTINUOUS"
        assert config["terminationProtectionEnabled"] is False
        region = config["replicationSpecs"][0]["regionConfigs"][0]
        assert region["electableSpecs"] == {"instanceSize": "M10", "nodeCount": 3}
        assert region["autoScaling"]["diskGB"]["enabled"] is True

    def test_the_user_and_the_access_list_entry_expire_on_their_own(self, script: ModuleType, atlas: FakeAtlas) -> None:
        script.up(atlas, now=NOW)

        expiry = (NOW + script.LIFETIME).strftime("%Y-%m-%dT%H:%M:%SZ")
        (user,) = atlas.named("dbusers", "create")
        (entry,) = atlas.named("accessLists", "create")
        assert user[user.index("--deleteAfter") + 1] == expiry
        assert user[user.index("--scope") + 1] == "strands-ci-42-1"
        assert "--currentIp" in entry and entry[entry.index("--deleteAfter") + 1] == expiry

    def test_the_password_is_masked_before_it_is_handed_on(
        self, script: ModuleType, atlas: FakeAtlas, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.up(atlas, now=NOW)

        (user,) = atlas.named("dbusers", "create")
        password = user[user.index("--password") + 1]
        printed = capsys.readouterr().out
        outputs = (tmp_path / "outputs").read_text()

        lines = printed.splitlines()
        assert lines[0] == f"::add-mask::{password}", "the password must be masked before any command runs"
        assert f"::add-mask::{_uri(outputs)}" in lines
        assert [line for line in lines if password in line] == [lines[0], f"::add-mask::{_uri(outputs)}"]
        assert "name=strands-ci-42-1\n" in outputs and password in _uri(outputs)

    def test_a_cluster_left_over_from_an_earlier_run_is_deleted_first(
        self, script: ModuleType, atlas: FakeAtlas
    ) -> None:
        old = (NOW - script.LIFETIME - datetime.timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
        recent = (NOW - datetime.timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
        atlas.clusters = [
            {"name": "strands-ci-1-1", "createDate": old},
            {"name": "strands-ci-2-1", "createDate": recent},
            {"name": "production", "createDate": old},
        ]

        script.up(atlas, now=NOW)

        assert atlas.named("clusters", "delete") == [["clusters", "delete", "strands-ci-1-1", "--force"]]

    def test_a_name_without_the_prefix_is_refused(
        self, script: ModuleType, atlas: FakeAtlas, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLAS_CI_CLUSTER", "production")

        with pytest.raises(ValueError, match="strands-ci-"):
            script.up(atlas, now=NOW)
        assert atlas.named("clusters", "create") == []


class TestWhatDownDeletes:
    """AT-233. Only this script's clusters, and an absent one is a finished teardown, not a failure."""

    def test_the_cluster_and_its_user_are_deleted(self, script: ModuleType) -> None:
        atlas = FakeAtlas()

        script.down("strands-ci-42-1", atlas)

        assert atlas.calls == [
            ["clusters", "delete", "strands-ci-42-1", "--force"],
            ["dbusers", "delete", "strands-ci-42-1", "--force"],
        ]

    def test_a_cluster_without_the_prefix_is_never_deleted(self, script: ModuleType) -> None:
        atlas = FakeAtlas()

        with pytest.raises(ValueError, match="refusing"):
            script.down("production", atlas)
        assert atlas.calls == []

    def test_a_cluster_that_was_never_created_is_not_an_error(self, script: ModuleType) -> None:
        atlas = FakeAtlas(fail={"clusters delete": "Error: CLUSTER_NOT_FOUND", "dbusers delete": "not found"})

        script.down("strands-ci-42-1", atlas)

    def test_any_other_delete_failure_is_reported(self, script: ModuleType) -> None:
        atlas = FakeAtlas(fail={"clusters delete": "Error: UNEXPECTED_ERROR"})

        with pytest.raises(subprocess.CalledProcessError):
            script.down("strands-ci-42-1", atlas)


def test_the_connection_string_carries_the_user_and_refuses_anything_but_srv(script: ModuleType) -> None:
    """AT-233."""
    uri = script.connection_uri("mongodb+srv://c.abcde.mongodb.net", "strands-ci-1", "ab12")

    assert uri == "mongodb+srv://strands-ci-1:ab12@c.abcde.mongodb.net/?retryWrites=true&w=majority"
    with pytest.raises(ValueError):
        script.connection_uri("mongodb://c.abcde.mongodb.net", "u", "p")


def _uri(outputs: str) -> str:
    return next(line.removeprefix("uri=") for line in outputs.splitlines() if line.startswith("uri="))


def test_a_failing_command_shows_what_the_atlas_cli_said(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AT-233. A failure without the CLI's own message cannot be diagnosed from a CI log."""

    def failing(*arguments: Any, **options: Any) -> Any:
        raise subprocess.CalledProcessError(
            1, ["atlas", "clusters", "create"], output="", stderr="Error: INVALID_ATTRIBUTE"
        )

    monkeypatch.setattr(script.subprocess, "run", failing)

    with pytest.raises(subprocess.CalledProcessError):
        script._atlas(["clusters", "create"])
    assert "INVALID_ATTRIBUTE" in capsys.readouterr().err
