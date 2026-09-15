"""The release workflow, asserted by reading it, because nothing in this suite can run it.

Covers ``AT-134`` through ``AT-137`` (``REQ-E-134`` to ``REQ-E-137``). Same principle as
``test_ci.py``: a workflow is a YAML document on disk, so these are unit-tier tests with no tier and
no network, and they read the file the way ``test_packaging.py`` reads ``pyproject.toml``.

**Why this file is worth more than the CI one.** ``ci-python.yml`` is wrong for as long as it takes
somebody to open a pull request. This file is wrong until a tag is pushed, and the failure it
produces is either a release that does not happen or -- far worse -- a wheel on PyPI that cannot be
replaced, only yanked. There is no rehearsal: the first execution of this workflow is a real upload
of a real version number that is then permanently spent.

**What this file cannot tell you.** It asserts the workflow says what it should say. Whether the
``pypi`` environment exists, whether it has required reviewers, and whether PyPI has a trusted
publisher bound to this repository and this filename are all settings outside the repository, and no
test here can see any of them. A green run of this file means the workflow is correct and says
nothing about whether an upload would succeed. That limit is stated rather than implied, because a
green test file about publishing reads like evidence publishing works.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

# tests/ -> python/ -> the repository root, which is where `.github/` belongs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-python.yml"

#: The tag prefix ``pyproject.toml``'s ``[tool.hatch.version.raw-options]`` derives the version from.
#: A second language in the same repository versions independently off its own prefix, and the whole
#: point of a prefix is that one workflow must not answer the other's tag.
TAG_PREFIX = "python-v"

#: The action that performs the upload. Named as data rather than matched loosely, because the
#: assertions below are about what this specific action is and is not handed.
PUBLISH_ACTION = "pypa/gh-action-pypi-publish"


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The parsed release workflow.

    Fails rather than skips when the file is absent, for the reason ``test_ci.py``'s fixture gives:
    absence is the defect these tests exist to catch, and a skip reports the same green as a pass.
    """
    assert WORKFLOW_PATH.is_file(), f"REQ-E-134: no release workflow at {WORKFLOW_PATH}"
    parsed = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(parsed, dict), "the workflow did not parse as a mapping"
    return parsed


@pytest.fixture(scope="module")
def raw() -> str:
    """The unparsed text, for the two questions YAML cannot answer.

    ``yaml.safe_load`` discards comments, and one assertion below is about a string that must not
    appear anywhere at all -- including in a comment, where a commented-out ``password:`` is a
    suggestion to the next person rather than a safeguard.
    """
    assert WORKFLOW_PATH.is_file(), f"REQ-E-134: no release workflow at {WORKFLOW_PATH}"
    return WORKFLOW_PATH.read_text()


def _jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The job mapping, which every assertion here starts from."""
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and jobs, "the workflow defines no jobs"
    return jobs


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    """One job's steps, as mappings."""
    return [step for step in job.get("steps", []) if isinstance(step, dict)]


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The ``on:`` block, under whichever key YAML produced for it.

    ``on`` is a YAML 1.1 boolean, so ``safe_load`` turns the unquoted key into ``True``. Handled
    rather than worked around in the workflow file, because quoting the key there to suit a test
    would be the test dictating the document's style.
    """
    for key in ("on", True):
        if key in workflow:
            value = workflow[key]
            assert isinstance(value, dict), f"the `on:` block is {type(value).__name__}, not a mapping"
            return value
    raise AssertionError("the workflow declares no triggers")


def _step_uses(job: dict[str, Any], action: str) -> list[dict[str, Any]]:
    """Every step in ``job`` invoking ``action``, ignoring the version after the ``@``."""
    return [step for step in _steps(job) if str(step.get("uses", "")).split("@")[0] == action]


class TestOnlyThePythonTagTriggersARelease:
    """AT-134 / REQ-E-134 -- what starts an upload, and everything that must not.

    The repository is laid out to hold a second language beside ``python/``, and the version scheme
    is per-prefix. A trigger broad enough to catch the other language's tag publishes this package
    from a commit nobody released, with a version derived from somebody else's tag.
    """

    def test_a_tag_push_is_the_only_trigger(self, workflow: dict[str, Any]) -> None:
        """No ``workflow_dispatch``, no ``release``, no branch push.

        Each of those is a second way to reach an irreversible upload, and a second way is a second
        thing to reason about when asking "what could have published this".
        """
        triggers = _triggers(workflow)

        assert set(triggers) == {"push"}, f"the only trigger must be a tag push, found {sorted(triggers)}"
        push = triggers["push"]
        assert isinstance(push, dict) and set(push) == {"tags"}, (
            f"the push trigger must be narrowed to tags alone, found {push}"
        )

    def test_every_tag_pattern_carries_the_python_prefix(self, workflow: dict[str, Any]) -> None:
        """The prefix is what keeps two languages' releases independent in one repository."""
        patterns = _triggers(workflow)["push"]["tags"]

        assert patterns, "the tag trigger lists no patterns"
        for pattern in patterns:
            assert pattern.startswith(TAG_PREFIX), (
                f"tag pattern {pattern!r} does not start with {TAG_PREFIX!r}, so a tag for another "
                f"language in this repository could start a Python release"
            )

    def test_another_languages_tag_matches_no_pattern(self, workflow: dict[str, Any]) -> None:
        """Asserted by matching, not by reading the prefix off the front a second time.

        ``fnmatch`` is the semantics GitHub applies to a tag filter, so this asks the question the
        way the platform answers it. The previous test could pass on a pattern like
        ``python-v*`` alongside a stray ``*`` -- this one could not.
        """
        patterns = _triggers(workflow)["push"]["tags"]
        foreign = ["typescript-v1.0.0", "v1.0.0", "1.0.0", "release-1.0.0"]

        for tag in foreign:
            matched = [p for p in patterns if fnmatch.fnmatch(tag, p)]
            assert matched == [], f"tag {tag!r} would start a Python release via {matched}"

    def test_the_packages_own_tag_does_match(self, workflow: dict[str, Any]) -> None:
        """The control. A filter that matches nothing is airtight and never releases anything."""
        patterns = _triggers(workflow)["push"]["tags"]
        ours = f"{TAG_PREFIX}0.1.0"

        assert any(fnmatch.fnmatch(ours, p) for p in patterns), (
            f"no pattern in {patterns} matches {ours!r}, so no tag would ever publish"
        )


class TestNothingHereIsALongLivedCredential:
    """AT-135 / REQ-E-135 -- the security property trusted publishing is actually about.

    Trusted publishing is not "a nicer way to store the token". It is the absence of a token: PyPI
    verifies an OIDC assertion naming this repository, this workflow filename and this environment,
    then mints a credential good for one upload. Nothing in the repository can be stolen, because
    nothing in the repository is a credential.

    The coupling to the filename is a security property rather than an inconvenience, and it is the
    reason renaming this file silently breaks releases until the publisher on PyPI is updated.
    """

    def test_the_publish_action_is_handed_no_password(self, workflow: dict[str, Any]) -> None:
        """A ``password:`` input turns trusted publishing back into a stored token, silently.

        The action accepts one and prefers it. So a token added "temporarily" to debug a failing
        upload does not error, does not warn, and quietly reintroduces exactly what OIDC removed.
        """
        for name, job in _jobs(workflow).items():
            for step in _step_uses(job, PUBLISH_ACTION):
                supplied = step.get("with", {}) or {}
                assert "password" not in supplied, (
                    f"job {name!r} hands {PUBLISH_ACTION} a `password:`, which overrides trusted "
                    f"publishing with a stored token"
                )

    def test_no_step_reads_a_secret(self, raw: str) -> None:
        """``secrets.*`` appears nowhere, comments included.

        Read from the raw text rather than the parsed document on purpose: a commented-out secret
        reference is a recipe the next person under release pressure will uncomment.
        """
        references = re.findall(r"secrets\.[A-Za-z_][A-Za-z0-9_]*", raw)

        assert references == [], (
            f"the release workflow references {sorted(set(references))}; publishing here is over OIDC "
            f"and no long-lived credential should exist to reference"
        )

    def test_the_oidc_permission_is_raised_on_exactly_one_job(self, workflow: dict[str, Any]) -> None:
        """``id-token: write`` is what exchanges for the upload credential, so its blast radius is
        every job that holds it. One job needs it.
        """
        holders = [
            name for name, job in _jobs(workflow).items() if (job.get("permissions") or {}).get("id-token") == "write"
        ]

        assert len(holders) == 1, f"`id-token: write` is held by {holders}, and exactly one job should hold it"

    def test_the_workflow_default_grants_no_write(self, workflow: dict[str, Any]) -> None:
        """The top-level default is read-only, so a job that raises nothing inherits nothing."""
        default = workflow.get("permissions")

        assert default == {"contents": "read"}, (
            f"the workflow-level permissions are {default!r}; they should be read-only so that raising "
            f"a permission is always a visible, per-job decision"
        )

    def test_the_publishing_job_is_bound_to_an_environment(self, workflow: dict[str, Any]) -> None:
        """The environment is half of what PyPI verifies, and where required reviewers live.

        This is the step that makes a tag insufficient on its own: a human approves the deployment
        before the upload runs. It is the last reversible moment in the whole process.
        """
        publishers = [(name, job) for name, job in _jobs(workflow).items() if _step_uses(job, PUBLISH_ACTION)]
        assert publishers, f"no job uses {PUBLISH_ACTION}"

        for name, job in publishers:
            environment = job.get("environment")
            assert environment, f"job {name!r} publishes to PyPI without an `environment:` to gate it"
            declared = environment.get("name") if isinstance(environment, dict) else environment
            assert declared, f"job {name!r} declares an environment with no name"


class TestThePublishedArtifactIsTheVerifiedArtifact:
    """AT-136 / REQ-E-136 -- the version check, enforced at the only moment it still matters.

    A version on PyPI is spent permanently: yanking hides it, nothing replaces it. So the check that
    the wheel's version equals the tag has to sit between the build and the upload, and the upload
    has to consume the artifact that check ran against rather than rebuilding one of its own.

    The failure this prevents is specific and quiet. ``hatch-vcs`` derives the version from git
    metadata; a shallow clone has no tags, so the version falls back rather than failing, and the
    artifact that reaches PyPI is a ``0.0.0`` or a ``.dev`` that can never be taken back.
    """

    def test_the_upload_runs_only_after_the_build(self, workflow: dict[str, Any]) -> None:
        """``needs:`` is what serialises them; without it both jobs start at once."""
        jobs = _jobs(workflow)
        publishers = [name for name, job in jobs.items() if _step_uses(job, PUBLISH_ACTION)]

        for name in publishers:
            needs = jobs[name].get("needs")
            needs = [needs] if isinstance(needs, str) else (needs or [])
            assert needs, f"job {name!r} publishes without depending on any other job"

    def test_every_job_reaches_the_upload_through_the_gate(self, workflow: dict[str, Any]) -> None:
        """Walk ``needs:`` from the publishing job and require the gate to be an ancestor.

        Asserted transitively rather than as one edge, because the chain is what carries the
        property: the gate can be two jobs away and still hold, and a publishing job rewired to
        depend only on the build would pass a direct-edge check while skipping every gate.
        """
        jobs = _jobs(workflow)
        publishers = [name for name, job in jobs.items() if _step_uses(job, PUBLISH_ACTION)]
        assert publishers, f"no job uses {PUBLISH_ACTION}"

        def ancestors(name: str) -> set[str]:
            needs = jobs[name].get("needs")
            needs = [needs] if isinstance(needs, str) else (needs or [])
            found = set(needs)
            for parent in needs:
                found |= ancestors(parent)
            return found

        def jobs_running(fragment: str) -> set[str]:
            return {
                name for name, job in jobs.items() if any(fragment in str(step.get("run", "")) for step in _steps(job))
            }

        # Every gate, not merely the first one: the floors and the contract tier live in separate jobs,
        # and an ancestry check that accepts either would pass on a chain that skips the other.
        required = {
            "the coverage floors": jobs_running("hatch run cov"),
            "the contract tier": jobs_running("-m contract"),
        }
        for label, owners in required.items():
            assert owners, f"no job runs {label}"

        for name in publishers:
            reachable = ancestors(name)
            for label, owners in required.items():
                assert reachable & owners, (
                    f"job {name!r} can publish without {label} ({sorted(owners)}) having run: its "
                    f"ancestors are {sorted(reachable)}"
                )

    def test_the_checkout_that_builds_is_not_shallow(self, workflow: dict[str, Any]) -> None:
        """``fetch-depth: 0``, because a shallow clone is how the version silently becomes wrong.

        Asserted on every checkout in the file rather than on the build job's alone: the gate
        computes coverage on the same tree, and a version fallback there would be just as invisible.
        """
        for name, job in _jobs(workflow).items():
            for step in _step_uses(job, "actions/checkout"):
                depth = (step.get("with") or {}).get("fetch-depth")
                assert depth == 0, (
                    f"job {name!r} checks out with fetch-depth {depth!r}; hatch-vcs reads the tag from "
                    f"git metadata and a shallow clone has none, so the version falls back instead of failing"
                )

    def test_the_version_check_compares_versions_rather_than_strings(self, workflow: dict[str, Any]) -> None:
        """``Version(...)``, so ``python-v0.1.0-rc1`` and ``python-v0.1.0rc1`` are one answer.

        A string comparison is the tempting shape and it is wrong in both directions: it fails a
        release over PEP 440 spelling, and a maintainer who has just been failed over spelling is
        the person most likely to replace the check with something weaker.
        """
        runs = [str(step.get("run", "")) for job in _jobs(workflow).values() for step in _steps(job)]
        checks = [block for block in runs if "GITHUB_REF_NAME" in block]

        assert checks, "no step compares the built artifact against the tag"
        for block in checks:
            assert "Version(" in block, (
                "the tag check does not parse either side as a PEP 440 version, so it is comparing strings"
            )
            assert TAG_PREFIX in block, f"the tag check does not strip the {TAG_PREFIX!r} prefix before comparing"


class TestTheReleaseGateIsNoWeakerThanCI:
    """AT-137 / REQ-E-137 -- a tag must clear at least what a pull request clears.

    The gate exists because the two runs answer different questions. CI answers "is this branch
    healthy" and can be re-run on a fix. The release gate answers "is this the artifact we are about
    to spend a version number on", and a failure discovered afterwards has no re-run.

    So every gate that guards a pull request guards a tag, and the contract tier is among them. That
    last clause is new. It could not be written while the contract tier needed a credential, and a
    tag is the worst possible moment to discover a credential is missing -- which is what the comment
    in this file used to say, correctly, about a situation that no longer exists. A service container
    is not a credential, the tier's own definition admits a plain ``mongod``, and the whole tier costs
    about eighteen seconds.
    """

    def test_the_gate_runs_every_static_check(self, workflow: dict[str, Any]) -> None:
        """Formatting, linting and typing, by the script names ``pyproject.toml`` defines."""
        runs = "\n".join(str(step.get("run", "")) for job in _jobs(workflow).values() for step in _steps(job))

        for script in ("format-check", "lint", "typecheck"):
            assert f"hatch run {script}" in runs, f"no step runs `hatch run {script}`"

    def test_nothing_runs_the_rewriting_formatter(self, workflow: dict[str, Any]) -> None:
        """``hatch run format`` rewrites the checkout and exits 0, so a gate built on it cannot fail."""
        runs = [str(step.get("run", "")) for job in _jobs(workflow).values() for step in _steps(job)]
        offenders = [block for block in runs if re.search(r"hatch run format(?!-check)", block)]

        assert offenders == [], f"these steps run the rewriting formatter instead of the checking one: {offenders}"

    def test_the_gate_enforces_both_coverage_floors(self, workflow: dict[str, Any]) -> None:
        """``hatch run cov`` is the aggregate floor and the per-module floor, and it runs the unit tier."""
        runs = "\n".join(str(step.get("run", "")) for job in _jobs(workflow).values() for step in _steps(job))

        assert "hatch run cov" in runs, "no step enforces the coverage floors"

    def test_the_gate_runs_the_contract_tier(self, workflow: dict[str, Any]) -> None:
        """The tier that needs a real server, which a tag now clears before spending a version.

        Selected rather than implied: ``hatch run test`` without ``-m contract`` reports the tier as
        skipped when no deployment answers, and a skip reads as green. The selector is what makes an
        absent server fail the release instead of quietly passing it.
        """
        runs = [str(step.get("run", "")) for job in _jobs(workflow).values() for step in _steps(job)]
        selected = [block for block in runs if re.search(r"hatch run test\b.*-m\s+(\"|')?contract", block)]

        assert selected, "no step runs `hatch run test -m contract`, so a tag never touches a real server"

    def test_the_contract_tier_gets_its_server_from_a_container(self, workflow: dict[str, Any]) -> None:
        """Every job that sets ``MONGODB_URI`` gets the value from a ``services:`` container.

        The same assertion ``ci-python.yml`` carries, for the same reason: a release workflow holding
        a cluster credential is the long-lived secret this whole design removes, arriving by the back
        door as a convenience.
        """
        for name, job in _jobs(workflow).items():
            uri = (job.get("env") or {}).get("MONGODB_URI")
            if uri is None:
                continue
            assert "secrets." not in str(uri), f"job {name!r} takes MONGODB_URI from a secret: {uri!r}"
            assert job.get("services"), (
                f"job {name!r} sets MONGODB_URI to {uri!r} but declares no `services:` container to answer it"
            )
