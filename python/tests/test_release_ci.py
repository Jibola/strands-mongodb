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
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_ci import _DEFAULT_TIMEOUT_MINUTES, _TIMEOUT_CEILING_MINUTES

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    # The same alias `test_examples.py` and `test_migration.py` use, for the same reason.
    import tomli as tomllib

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


#: The release document, beside the package README rather than at the repository root: it is about
#: releasing the Python distribution, and a second language in this repository releases on its own tag.
RELEASING_PATH = PROJECT_ROOT / "RELEASING.md"

#: The changelog, in the same place and for the same reason. ``pyproject.toml`` points ``readme`` at
#: ``python/README.md``, and these two are that document's neighbours.
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"

#: The marker both READMEs carry above their install blocks while nothing is on PyPI. It is the
#: repository's own statement about its publication state, and ``AT-164`` makes the changelog answer to it.
UNPUBLISHED_MARKER = "<!-- skip: these reach PyPI for a package that is not published yet -->"

#: ``## [Unreleased]`` or ``## [0.1.0] - 2026-09-16``. The bracketed form is Keep a Changelog's, and the
#: date group is what distinguishes a section that describes a release from one that describes a plan.
_CHANGELOG_HEADING = re.compile(r"^##\s+\[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>\S+))?\s*$", re.MULTILINE)

#: The version table in the repository-root README -- ``| `0.1.0` | ... |``. Read rather than duplicated,
#: for the reason ``AT-161`` gives about that file: it is the landing page, and it is what a reader believes.
_VERSION_ROW = re.compile(r"^>?\s*\|\s*`(?P<version>\d+\.\d+\.\d+)`\s*\|", re.MULTILINE)

#: What a reader of a red run needs to be told, once, in the first line.
_RELEASING_DEFECT = (
    "REQ-E-157: the release workflow exists and is asserted, but nothing in this tree tells a person how to run it"
)
_CHANGELOG_DEFECT = "REQ-E-158: the repository publishes a version table and describes no version"


def _document(path: Path, requirement: str) -> str:
    """Read a release document, failing with the defect named rather than with a ``FileNotFoundError``.

    A fixture would do this too, and the ``workflow`` fixture above is one. The difference matters while
    the file is absent: a fixture that fails reports every test in the class as an *error*, and an error
    reads as "the suite is broken" where a failure reads as "the thing under test is missing". These
    documents are missing exactly once -- before they are written -- and that is the run whose first line
    should say which file and which requirement.
    """
    assert path.is_file(), f"{requirement}: no document at {path}"
    return path.read_text()


class TestSomebodyCanActuallyPerformARelease:
    """AT-163 / REQ-E-157 -- the release document, held to the workflow rather than to itself.

    ``AT-134``…``AT-137`` assert that ``release-python.yml`` says what it must say. Nothing asserted that
    a person could follow it. Four jobs run on a tag and only the last one does anything irreversible,
    which means the cost of not knowing the order is paid exactly once, at the only moment it cannot be
    undone.

    Every assertion here reads its expectation out of the parsed workflow. A document that restates the
    workflow passes the day it is written and drifts silently afterwards, which is the failure mode the
    root ``README.md`` demonstrated for thirteen tasks with a passing test over it.
    """

    def test_it_names_every_job_the_workflow_runs(self, workflow: dict[str, Any]) -> None:
        """AT-163 -- job names come from the YAML, so a fifth job fails here instead of being forgotten."""
        releasing = _document(RELEASING_PATH, _RELEASING_DEFECT)
        jobs = _jobs(workflow)
        missing = sorted(
            name for name, job in jobs.items() if name not in releasing and str(job.get("name", "")) not in releasing
        )

        assert missing == [], (
            f"the release document does not mention these jobs: {missing}. A reader following it would "
            f"not know they run, and one of them is the last gate before an upload that can only be yanked"
        )

    def test_it_names_the_tag_that_starts_a_release(self, workflow: dict[str, Any]) -> None:
        """AT-163 -- the prefix is read from the trigger, not from this file's own constant."""
        releasing = _document(RELEASING_PATH, _RELEASING_DEFECT)
        patterns = [str(tag) for tag in (_triggers(workflow).get("push") or {}).get("tags") or []]
        assert patterns, "the workflow declares no tag trigger, which AT-134 should have caught first"

        for pattern in patterns:
            prefix = pattern.rstrip("*")
            assert prefix in releasing, (
                f"the release document never mentions {prefix!r}, which is the only thing that starts a "
                f"release; a reader has no way to know what to tag"
            )

    def test_it_names_the_environment_the_upload_waits_on(self, workflow: dict[str, Any]) -> None:
        """AT-163 -- the protected environment is where a human approves, and it is a settings artifact.

        It is the one part of the release a reader cannot discover by reading the repository, because
        the environment and its required reviewers live in repository settings rather than in a file.
        A release document that omits it describes a process that stops without explanation.
        """
        releasing = _document(RELEASING_PATH, _RELEASING_DEFECT)
        environments = [
            job["environment"]["name"] if isinstance(job.get("environment"), dict) else job.get("environment")
            for job in _jobs(workflow).values()
            if job.get("environment")
        ]
        assert environments, "no job binds an environment, which AT-135 should have caught first"

        for environment in environments:
            assert str(environment) in releasing, (
                f"the release document never mentions the {environment!r} environment, where the release "
                f"waits for a human and where it stops if nobody is watching"
            )


class TestTheChangelogAndTheRepositoryAgreeAboutWhatExists:
    """AT-164 / REQ-E-158 -- two documents that both claim to say what a version contains.

    The version table lives on the repository's landing page; the changelog is the file a packager
    reads. Nothing derives either from the other, which is the same condition that let the ``dev``
    extra and the hatch environment drift until ``AT-160`` compared them.

    The second test is the one that earns its place. Both READMEs carry a marker saying the package is
    not on PyPI. That marker becomes false the moment the name is claimed, and until now the only thing
    tracking it was a line in a planning document outside this repository. Here it is a gate: the
    changelog may not date a release while the repository says none has happened.
    """

    def test_its_versions_are_the_versions_the_landing_page_publishes(self) -> None:
        """AT-164 -- compared as sets in both directions; ``Unreleased`` is not a version."""
        changelog = _document(CHANGELOG_PATH, _CHANGELOG_DEFECT)
        readme = (REPO_ROOT / "README.md").read_text()

        documented = {
            match.group("version")
            for match in _CHANGELOG_HEADING.finditer(changelog)
            if match.group("version").lower() != "unreleased"
        }
        published = {match.group("version") for match in _VERSION_ROW.finditer(readme)}
        assert published, "the landing page's version table did not parse, so this test proves nothing"

        assert documented == published, (
            f"the changelog and the landing page disagree about which versions exist. Only in the "
            f"changelog: {sorted(documented - published)}. Only on the landing page: "
            f"{sorted(published - documented)}"
        )

    def test_nothing_is_dated_while_the_repository_says_nothing_is_published(self) -> None:
        """AT-164 -- the install-block marker and the changelog change on the same day or neither does.

        Both directions are asserted. While the marker is present no section may carry a date, and once
        it is gone at least one must — so the test fails whichever half of the pair is forgotten, which
        is the only arrangement that survives the claim being made by a person in a hurry.
        """
        changelog = _document(CHANGELOG_PATH, _CHANGELOG_DEFECT)
        readmes = {
            "README.md": (REPO_ROOT / "README.md").read_text(),
            "python/README.md": (PROJECT_ROOT / "README.md").read_text(),
        }
        unpublished = {name for name, text in readmes.items() if UNPUBLISHED_MARKER in text}
        dated = {
            match.group("version"): match.group("date")
            for match in _CHANGELOG_HEADING.finditer(changelog)
            if match.group("date")
        }

        if unpublished:
            assert dated == {}, (
                f"{sorted(unpublished)} still say the package is not published, and the changelog dates "
                f"{dated} as released. One of the two is wrong, and a dated release nobody can install is "
                f"the more expensive way to be wrong"
            )
        else:
            assert dated, (
                "no README carries the unpublished marker any more, so the package is claimed on PyPI, "
                "and the changelog still dates nothing as released"
            )

    def test_a_source_download_carries_it(self) -> None:
        """AT-164 -- the sdist ships the changelog, on the same argument ``test_migration.py`` makes.

        A changelog that exists only on GitHub is read by people who were already going to find the
        repository. The reader it is written for is the one holding a source download and deciding
        whether to upgrade, and ``include`` is what decides whether they have it.
        """
        include = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"][
            "sdist"
        ]["include"]
        assert CHANGELOG_PATH.name in include, (
            f"sdist include is {include}, so a source download of this package contains no changelog and "
            f"its reader has no way to know what changed between the version they have and this one"
        )


class TestNoReleaseJobCanRunForSixHours:
    """AT-165 / REQ-E-159 — the release half of the same requirement, with one job that is not like the others.

    ``test_ci.py``'s class carries the argument and the measured numbers; the ceiling is imported from it
    rather than written twice, because two copies of a number nothing compares is the drift ``AT-160``
    exists to catch.

    The job worth a sentence is ``publish``. It sits behind the ``pypi`` environment's required reviewers,
    and a budget looks like it would cut the approval off — it does not. A job pending a deployment
    protection rule has not started, so ``timeout-minutes`` has not begun counting; GitHub cancels an
    unapproved run after thirty days on its own clock, and that is not a number this file can set. The
    budget here bounds the upload once somebody has said yes, which takes seconds.
    """

    def test_every_job_declares_one(self, workflow: dict[str, Any]) -> None:
        """AT-165 — including the two that have never run, since nothing has ever been released."""
        undeclared = sorted(name for name, job in _jobs(workflow).items() if job.get("timeout-minutes") is None)
        assert undeclared == [], (
            f"these release jobs declare no timeout-minutes and so inherit GitHub's "
            f"{_DEFAULT_TIMEOUT_MINUTES}: {undeclared}. This workflow has never run — there is no release "
            f"— so a hang here would be discovered by a maintainer watching a tag they just pushed"
        )

    def test_none_of_them_is_the_default_in_disguise(self, workflow: dict[str, Any]) -> None:
        """AT-165 — the same ceiling, and none of these jobs has a cluster tier's excuse for needing it."""
        for name, job in _jobs(workflow).items():
            budget = job.get("timeout-minutes")
            assert isinstance(budget, int) and not isinstance(budget, bool), (
                f"release job {name!r} declares timeout-minutes {budget!r}, which YAML did not read as an integer"
            )
            assert 0 < budget <= _TIMEOUT_CEILING_MINUTES, (
                f"release job {name!r} budgets {budget} minutes against a ceiling of "
                f"{_TIMEOUT_CEILING_MINUTES}. Nothing here runs a cluster tier, so there is no job in this "
                f"workflow with a reason to approach it"
            )


#: A git object name. What makes a reference immutable is that it names the content rather than a place
#: the content currently sits.
_DIGEST = re.compile(r"\A[0-9a-f]{40}\Z")


class TestTheActionThatUploadsIsPinnedToContentRatherThanToAPlace:
    """``AT-170`` / ``REQ-E-164`` -- the one mutable reference that matters.

    ``pypa/gh-action-pypi-publish@release/v1`` is a branch: not a tag, not a digest. It runs in the only
    job in the repository that raises ``id-token: write``, the token PyPI exchanges for an upload
    credential for ``strands-mongodb``, so whoever controls that branch controls what runs there and the
    blast radius is every install of this package.

    **This is not a repository-wide pinning policy, and must not become one.** MongoDB's own zizmor
    configuration is ``actions/*: ref-pin`` with ``mongodb-labs/drivers-github-tools/*: hash-pin``, so
    ``actions/checkout@v7`` throughout these workflows is correct by that standard and moving it to
    digests would be a deviation from it. PyMongo hash-pins this one action and ref-pins the rest. The
    scope of this gate is therefore the job holding the credential, not the file.

    **What a pin costs.** A digest does not update itself. ``REQ-E-166`` is what keeps it current, and the
    two requirements are worth less apart than together.
    """

    def test_exactly_one_job_holds_the_publishing_credential(self, workflow: dict[str, Any]) -> None:
        """AT-170 -- the vacuity guard, and the premise the scope of this gate rests on."""
        holders = [
            name for name, job in workflow["jobs"].items() if (job.get("permissions") or {}).get("id-token") == "write"
        ]
        assert len(holders) == 1, (
            f"{len(holders)} jobs raise id-token: write ({holders}). This gate is scoped to the job that "
            f"holds the upload credential, so which job that is has to be unambiguous"
        )

    def test_every_action_it_runs_is_pinned_to_a_digest(self, workflow: dict[str, Any]) -> None:
        """AT-170 -- the finding."""
        for name, job in workflow["jobs"].items():
            if (job.get("permissions") or {}).get("id-token") != "write":
                continue
            uses = [step["uses"] for step in job.get("steps") or [] if step.get("uses")]
            assert uses, f"job {name!r} holds the upload credential and runs no action, which cannot be right"
            unpinned = [reference for reference in uses if not _DIGEST.match(reference.partition("@")[2])]
            assert unpinned == [], (
                f"job {name!r} holds the credential PyPI exchanges for an upload token and runs {unpinned} "
                f"from a mutable reference. A branch or a tag can be moved; a digest names the content"
            )

    def test_the_reader_can_tell_a_digest_from_a_branch(self) -> None:
        """AT-170 -- the matcher, against the two references this repository actually uses."""
        assert _DIGEST.match("dc37677b2e1c63e2034f94d8a5b11f265b73ba33")
        assert not _DIGEST.match("release/v1")
        assert not _DIGEST.match("v7")
        assert not _DIGEST.match("v7.0.0")
