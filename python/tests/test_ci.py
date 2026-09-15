"""The CI configuration, asserted by reading it, because nothing in this suite can run it.

Covers ``AT-104`` through ``AT-109`` (``REQ-E-104`` to ``REQ-E-109``), plus ``AT-130``, ``AT-131`` and
``AT-138``. There is no tier here and no network: a workflow is a YAML document on disk, so these are
unit-tier tests that read it the way ``test_packaging.py`` reads ``pyproject.toml``.

``AT-138`` is the odd one out and reads in the opposite direction. The others ask what the workflow says;
it asks what this suite reads, and then whether the workflow's path filters cover the answer. That answer
is not a list held here -- it is walked out of the test modules, because a file joins it the moment
somebody writes a ``Path`` expression naming it, and nothing about writing one prompts opening a workflow.

**What this file cannot tell you.** It asserts the workflow says what it should say. Whether GitHub then
runs it, whether a runner has ``hatch``, and whether the matrix resolves are all unobservable from here, and
the first real answer to any of them arrives on the first pull request. That limit is worth stating rather
than implying, because a green test file about CI reads like evidence CI works.

What it does catch is the failure mode that actually happened: a gate that was specified and present in no
file, for a long stretch, while a test in this suite named the missing mechanism as another item's only
mitigation (``AT-052``). A workflow asserted by tests cannot go missing quietly.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import ChecklistGate

# Imported rather than reimplemented, for the reason its own docstring gives: "was this run narrowed?" has
# one right answer, and a copy of it drifts the first time pytest grows another selection flag.
from test_gate import _narrowing

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    tomllib = None  # type: ignore[assignment]

# tests/ -> python/ -> the repository root, which is where `.github/` belongs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci-python.yml"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

#: Every directory ``pyproject.toml``'s ``files`` key hands to mypy, which is also everything the matrix's
#: oldest interpreter has to be able to compile.
SOURCE_ROOTS = ("src", "tests", "scripts", "examples")

#: The comprehension expressions. A ``comprehension`` node is the ``for`` clause inside one of these, and its
#: ``is_async`` flag is what ``async for`` sets.
_COMPREHENSION_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: The weekly canary slot the extension template ships and ``AT-052`` is written against
#: (``extension-template/.github/workflows/ci-python.yml``). Kept identical rather than moved to a
#: quieter hour: the point of matching is that a maintainer who knows the template knows this file.
CANARY_CRON = "0 6 * * 1"

#: The two conformance checks the canary runs, one per protocol. Written as data
#: here, never in a docstring in this file: the gate in ``conftest.py`` indexes ids out of docstrings, so an
#: id in one of these would register a test about YAML as a carrier of a protocol check.
PROTOCOL_CHECK_IDS = ("AT-034", "AT-052")


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The parsed workflow.

    Fails rather than skips when the file is absent, which is the whole point: absence is the defect these
    tests were written to catch, and a skip would have reported the same green as a passing gate.
    """
    assert WORKFLOW_PATH.is_file(), f"REQ-E-104: no CI configuration at {WORKFLOW_PATH}"
    parsed = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(parsed, dict), "the workflow did not parse as a mapping"
    return parsed


def _run_blocks(node: Any) -> list[str]:
    """Every ``run:`` string anywhere in the document, found by walking rather than by regex.

    A regex over the raw text would also match the word inside a comment or a step name, and the claims
    below are about what executes.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_run_blocks(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_run_blocks(item))
    return found


def _values_under(node: Any, wanted: str) -> list[str]:
    """Every string value stored under ``wanted``, at any depth."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == wanted:
                if isinstance(value, str):
                    found.append(value)
                elif isinstance(value, dict):
                    found.extend(str(item) for item in value.values())
            found.extend(_values_under(value, wanted))
    elif isinstance(node, list):
        for item in node:
            found.extend(_values_under(item, wanted))
    return found


def _all_strings(node: Any) -> list[str]:
    """Every string anywhere in the document, keys included.

    Used instead of reading the raw text where the claim is about the whole file, so that a word inside a
    ``#`` comment cannot fail a test about what the workflow does.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                found.append(key)
            found.extend(_all_strings(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_all_strings(item))
    elif isinstance(node, str):
        found.append(node)
    return found


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The ``on:`` block, looked up under both spellings it can arrive as.

    YAML 1.1 resolves a bare ``on`` key to the boolean ``True``, so ``workflow["on"]`` raises ``KeyError``
    against an idiomatic workflow file. GitHub's own parser has the same heritage and reads the key as the
    trigger name anyway, so the file keeps the unquoted spelling every other workflow uses and the quirk is
    absorbed here. Worth a helper rather than a one-line ``get``: the failure it prevents is a test that
    looks broken while the workflow is correct.
    """
    for key in ("on", True):
        if key in workflow:
            block = workflow[key]
            assert isinstance(block, dict), f"the triggers parsed as {type(block).__name__}, not a mapping"
            return block
    raise AssertionError(f"the workflow declares no triggers; its keys are {sorted(map(str, workflow))}")


def _pyproject() -> dict[str, Any]:
    """The manifest, parsed. Callers must gate on ``tomllib`` being importable."""
    assert tomllib is not None, "tomllib is unavailable; the caller should have skipped"
    return tomllib.loads(PYPROJECT_PATH.read_text())


_NEEDS_TOMLLIB = pytest.mark.skipif(
    tomllib is None,
    reason="tomllib is 3.11+, and this compares the workflow against pyproject.toml",
)


#: Every module the suite collects, walked for the paths it names. ``conftest.py`` is included: it reads
#: ``.env`` from two locations, and one of them is above the package directory.
_TEST_MODULES = sorted(Path(__file__).resolve().parent.glob("*.py"))

#: The prefix the trigger has watched since the workflow was written. A path under it needs nothing from
#: ``AT-138``; a path outside it is what this is about.
_PACKAGE_DIRECTORY = "python"

#: Paths outside ``python/`` the walk is known to find today, checked as a subset of what it returns so
#: that a walk which silently resolved nothing cannot pass as a clean result. Not an equality: adding a
#: read should fail the test below, which demands the trigger be extended, rather than this one.
_READ_AT_LEAST = (
    ".github/workflows/ci-python.yml",
    ".github/workflows/release-python.yml",
    "AGENTS.md",
    "CONTRIBUTING.md",
    ".gitignore",
)


def _glob_to_regex(pattern: str) -> str:
    """One GitHub path filter, as the regular expression it stands for.

    Two wildcards with different reach, which is the whole reason this is not ``fnmatch``: ``*`` stops at a
    separator and ``**`` crosses them, so ``python/*`` and ``python/**`` are different filters and only the
    second one watches ``python/tests/test_ci.py``.
    """
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            crosses = pattern[index + 1 : index + 2] == "*"
            out.append(".*" if crosses else "[^/]*")
            index += 2 if crosses else 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(char))
            index += 1
    return "".join(out)


def _watched(path: str, patterns: list[str]) -> bool:
    """Whether a trigger declaring ``patterns`` runs on a change to ``path``.

    A directory is asked about through a probe child rather than about itself, because a filter is written
    about files: ``docs/**`` matches ``docs/guide.md`` and does not match the string ``docs``. Anything the
    walk names that is not a directory on disk is asked about directly.
    """
    probed = f"{path}/a-file-underneath" if (REPO_ROOT / path).is_dir() else path
    return any(re.fullmatch(_glob_to_regex(pattern), probed) for pattern in patterns)


def _paths_named_by(module: Path) -> set[str]:
    """Every repository-relative path one test module names, resolved from its syntax rather than run.

    The module is parsed, its module-level names are resolved to paths, and every complete ``/`` chain
    anywhere in it is evaluated against them. Four forms carry all of it: ``Path(__file__)``, ``.resolve()``,
    ``.parent``, and ``/`` with a string on the right.

    Two results are dropped rather than returned, and both are dropped because no trigger could watch them.
    A chain that walks above the repository root -- ``test_migration.py`` reaches a checkout beside this one
    -- resolves to nothing. So does the root itself, which is a directory a name is built from and not a
    file anything reads.

    Returns:
        Repository-relative paths, slash-separated, with no leading slash.
    """
    tree = ast.parse(module.read_text())
    here = (_PACKAGE_DIRECTORY, "tests", module.name)
    known: dict[str, tuple[str, ...] | None] = {}

    def evaluate(node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = evaluate(node.left)
            right = node.right
            if left is None or not isinstance(right, ast.Constant) or not isinstance(right.value, str):
                return None
            return left + tuple(part for part in right.value.split("/") if part)
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            base = evaluate(node.value)
            return None if not base else base[:-1]
        if isinstance(node, ast.Call):
            called = node.func
            if isinstance(called, ast.Attribute) and called.attr == "resolve":
                return evaluate(called.value)
            is_this_file = (
                isinstance(called, ast.Name)
                and called.id == "Path"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "__file__"
            )
            return here if is_this_file else None
        if isinstance(node, ast.Name):
            return known.get(node.id)
        return None

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    known[target.id] = evaluate(statement.value)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            known[statement.target.id] = evaluate(statement.value) if statement.value else None

    chains = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)]
    # The left operand of a longer chain is a prefix of it. `_ROOT / ".github"` is a step towards a file
    # rather than a path anything opens, and demanding the trigger watch `.github` would be a different
    # requirement than the one REQ-E-138 states.
    partial = {id(node.left) for node in chains}
    resolved = {evaluate(node) for node in chains if id(node) not in partial}
    return {"/".join(parts) for parts in resolved if parts}


def _paths_outside_the_package() -> set[str]:
    """Every path the suite names that ``python/**`` does not already cover.

    One exemption, and it is derived rather than listed: a path ``.gitignore`` accounts for cannot be in a
    commit, so no change to it can trigger anything. ``conftest.py`` names ``.env`` beside the repository
    root, and a contributor creates that file or does not.
    """
    ignored = (REPO_ROOT / ".gitignore").read_text()
    named: set[str] = set()
    for module in _TEST_MODULES:
        named |= _paths_named_by(module)
    return {path for path in named if path.split("/")[0] != _PACKAGE_DIRECTORY and Path(path).name not in ignored}


class TestTheGatesEveryChangeMustPass:
    """AT-104 / REQ-E-104 -- the four things a pull request has to clear before anyone reads it.

    Three static gates and the unit tier. All four already exist as hatch scripts and have been run by
    hand at the end of every change; what has never existed is anything that runs them when
    nobody remembers to.
    """

    def test_the_three_static_gates_all_run(self, workflow: dict[str, Any]) -> None:
        """Formatting, linting and typing, each by the script name ``pyproject.toml`` defines."""
        runs = "\n".join(_run_blocks(workflow))

        for script in ("format-check", "lint", "typecheck"):
            assert f"hatch run {script}" in runs, f"no step runs `hatch run {script}`"

    def test_nothing_runs_the_rewriting_formatter(self, workflow: dict[str, Any]) -> None:
        """The distinction the extension template gets wrong, and the reason REQ-E-104 names the form.

        ``hatch run format`` is ``ruff format``: it rewrites the files in place and exits 0, so a job built
        on it reports success on unformatted code and leaves the changes in a runner that is about to be
        thrown away. ``format-check`` is ``ruff format --check``, which exits 1 and changes nothing. Only
        one of the two is a gate.
        """
        offenders = [block for block in _run_blocks(workflow) if re.search(r"hatch run format(?!-check)", block)]

        assert offenders == [], (
            f"these steps run the rewriting formatter instead of the checking one: {offenders}; "
            f"`ruff format` exits 0 after rewriting, so the step can never fail"
        )

    def test_a_job_runs_the_unit_tier(self, workflow: dict[str, Any]) -> None:
        """``hatch run test`` with no selector, which is the unit tier plus whatever tiers have variables.

        No tier variable is set for this job, so `conftest.py`'s gate skips the three that need a backend
        and the run is the 680-test unit tier. The contract tier has a job of its own, beside this one, with
        a service container and ``MONGODB_URI`` pointed at it.
        """
        assert any("hatch run test" in block for block in _run_blocks(workflow)), "no step runs the test suite"

    def test_both_change_triggers_watch_the_package_directory(self, workflow: dict[str, Any]) -> None:
        """A push or a pull request that touches ``python/`` runs the gates.

        Asserted for both triggers rather than one: a repository that only gates pull requests gates
        nothing on a branch that is pushed to directly, and this one is about to change owners.
        """
        triggers = _triggers(workflow)

        for event in ("pull_request", "push"):
            assert event in triggers, f"no {event} trigger; the file declares {sorted(map(str, triggers))}"
            paths = triggers[event]["paths"]
            assert any(entry.startswith("python/") for entry in paths), (
                f"the {event} trigger watches {paths}, which does not include the package directory"
            )

    def test_the_workflow_watches_itself(self, workflow: dict[str, Any]) -> None:
        """A path filter that omits the workflow file means changes to the gates are ungated.

        Small point, and it is the one that makes the first pull request against this file testable at all.
        """
        triggers = _triggers(workflow)
        relative = str(WORKFLOW_PATH.relative_to(REPO_ROOT))

        for event in ("pull_request", "push"):
            paths = triggers[event]["paths"]
            assert relative in paths, f"the {event} trigger watches {paths}, which does not include {relative}"


class TestTheTriggerWatchesEverythingTheSuiteReads:
    """``AT-138`` / ``REQ-E-138`` -- the files this suite asserts about that no change trigger watched.

    ``release-python.yml`` arrived with eighteen tests that read it, in one pull request. CI ran, because
    the tests are under ``python/``. The next change to the workflow alone runs nothing: it matches
    neither declared filter, and its own trigger is a pushed tag. A pull request with no checks looks the
    same as one whose checks passed.

    The same was already true of ``AGENTS.md``, ``CONTRIBUTING.md`` and ``.gitignore``. ``AT-113`` resolves
    every command, variable, marker, version and path those documents name against the repository, and
    could only ever have caught a stale claim on some later change that happened to touch ``python/``.

    The list is found rather than held. A file enters the suite's reach when somebody writes one ``Path``
    expression in a test module, and nothing about writing it prompts opening the workflow -- so a curated
    list would be extended by exactly the person who forgot to extend the filter.
    """

    def test_the_walk_finds_the_files_the_suite_is_known_to_read(self) -> None:
        """AT-138 -- the vacuity guard: a walk that resolved nothing would pass the test below.

        A subset rather than an equality. Adding a read is supposed to fail the trigger test, which names
        the fix, rather than this one, which would only say the constant is out of date.
        """
        found = _paths_outside_the_package()
        missing = [path for path in _READ_AT_LEAST if path not in found]
        assert missing == [], f"the walk did not resolve paths the suite reads: {missing}. It found {sorted(found)}"

    def test_both_change_triggers_watch_every_file_the_suite_reads(self, workflow: dict[str, Any]) -> None:
        """AT-138 -- ``REQ-E-138``: a file the suite makes a claim about, and no job to check the claim.

        Both triggers, for the reason the package-directory test gives: a repository that gates only pull
        requests gates nothing on a branch pushed to directly.
        """
        triggers = _triggers(workflow)
        found = sorted(_paths_outside_the_package())

        for event in ("pull_request", "push"):
            patterns = triggers[event]["paths"]
            unwatched = [path for path in found if not _watched(path, patterns)]
            assert unwatched == [], (
                f"the {event} trigger watches {patterns}, so a change to {unwatched} runs no test -- "
                "and the suite asserts something about each of them"
            )

    def test_the_matcher_reads_a_filter_the_way_github_does(self) -> None:
        """AT-138 -- the test above passes on everything if the matcher answers yes to everything.

        The two wildcards and the boundary between them, plus the prefix that looks like a match and is a
        different directory.
        """
        assert _watched("AGENTS.md", ["AGENTS.md"])
        assert _watched("python/tests/test_ci.py", ["python/**"])
        assert not _watched("AGENTS.md", ["python/**"])
        assert not _watched("python-notes/x.md", ["python/**"]), "`python/**` is not a prefix match on the name"
        assert not _watched("a/b/c.md", ["a/*/x.md"]), "`*` does not cross a separator"
        assert not _watched("AGENTS.md", []), "an empty filter watches nothing"


class TestTheMatrixCoversEveryVersionTheClassifiersClaim:
    """AT-105 / REQ-E-105 -- the matrix and the classifiers, compared as sets in both directions.

    Equality rather than containment. A matrix narrower than the claim ships support for an untested
    interpreter; a matrix wider than the claim tests one the package says it does not support, and the
    honest fix for that is a new classifier, not a quieter test.
    """

    @_NEEDS_TOMLLIB
    def test_the_matrix_equals_the_versions_the_classifiers_claim(self, workflow: dict[str, Any]) -> None:
        """Read from both files, so neither list can be edited alone."""
        claimed = {
            classifier.rsplit(" :: ", 1)[1]
            for classifier in _pyproject()["project"]["classifiers"]
            if classifier.startswith("Programming Language :: Python :: 3.")
        }
        assert claimed, "pyproject.toml claims no specific Python versions, so there is nothing to match"

        matrix = {str(version) for version in workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]}

        assert matrix == claimed, (
            f"the matrix tests {sorted(matrix)} and the classifiers claim {sorted(claimed)}; "
            f"untested but claimed: {sorted(claimed - matrix)}, tested but unclaimed: {sorted(matrix - claimed)}"
        )

    def test_every_version_in_the_matrix_is_quoted(self, workflow: dict[str, Any]) -> None:
        """Unquoted ``3.10`` is the float ``3.1`` in YAML, and ``setup-python`` would resolve it to 3.1.

        This is the one way the matrix can be wrong while looking right, and it is invisible in a diff. It
        gets its own test rather than riding on the set comparison above, because the set comparison would
        report it as a missing version and send the reader looking for the wrong mistake.
        """
        entries = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]

        unquoted = [entry for entry in entries if not isinstance(entry, str)]
        assert unquoted == [], f"quote these, YAML read them as numbers: {unquoted}"


class TestBothCoverageFloorsAreEnforced:
    """AT-106 / REQ-E-106 -- the aggregate floor and the per-module floor, in the one job that measures once.

    Measured in CI on the unit tier alone: the aggregate is 98.10% and the weakest module is 89.47%,
    against floors of 90% and 70%. So the job passes with no connection string, which is why the
    floors were enforceable before any tier had a deployment. In CI rather than locally, because the
    aggregate moves by a line or two with the interpreter -- 3.14 reaches 98.18% -- and the number
    worth writing down is the one the gate actually reads.
    """

    def test_a_job_runs_the_coverage_gate(self, workflow: dict[str, Any]) -> None:
        assert any("hatch run cov" in block for block in _run_blocks(workflow)), (
            "no step runs `hatch run cov`, so neither coverage floor is enforced anywhere automatic"
        )

    @_NEEDS_TOMLLIB
    def test_the_coverage_script_is_still_both_floors(self, workflow: dict[str, Any]) -> None:
        """The workflow delegates, so the delegate is asserted here too.

        ``hatch run cov`` is two commands. If someone drops the second, the workflow still says `cov` and
        the per-module floor silently stops being enforced, with nothing in the diff to notice. Reading the
        script definition is what closes that.
        """
        script = _pyproject()["tool"]["hatch"]["envs"]["default"]["scripts"]["cov"]

        assert any("--cov-fail-under=90" in command for command in script), (
            f"the `cov` script no longer carries the aggregate floor: {script}"
        )
        assert any("check_module_coverage.py" in command for command in script), (
            f"the `cov` script no longer runs the per-module floor: {script}"
        )

    def test_the_coverage_job_measures_once_rather_than_per_interpreter(self, workflow: dict[str, Any]) -> None:
        """Four runs of the same measurement produce one number and three copies of it."""
        job = workflow["jobs"]["coverage"]

        assert "strategy" not in job, f"the coverage job has a matrix: {job['strategy']}"


class TestTheWeeklyCanaryAgainstTheLatestSDK:
    """AT-107 / REQ-E-107 -- the cron named as the only mitigation for an unpinned SDK, never written.

    ``AT-052``'s own docstring says the marker "selects it for the CI cron that resolves the latest published
    ``strands-agents`` instead of the pinned one". That sentence was in the suite for a long time and the
    cron it describes was in no file. The test was therefore running against the pinned SDK, which is the one
    version it cannot detect drift in.
    """

    def test_the_schedule_is_the_weekly_slot_the_template_uses(self, workflow: dict[str, Any]) -> None:
        crons = [entry["cron"] for entry in _triggers(workflow)["schedule"]]

        assert crons == [CANARY_CRON], f"the schedule is {crons}, not the template's weekly {CANARY_CRON!r}"

    def test_the_canary_runs_only_on_the_schedule(self, workflow: dict[str, Any]) -> None:
        """Otherwise every pull request resolves the latest SDK and the pinned floor is never the thing tested."""
        condition = workflow["jobs"]["sdk-compat"].get("if")

        assert condition == "github.event_name == 'schedule'", (
            f"the canary's condition is {condition!r}, so it runs outside the cron"
        )

    def test_the_change_triggered_jobs_sit_out_the_schedule(self, workflow: dict[str, Any]) -> None:
        """The mirror of the above. A cron that also runs the pinned jobs reports nothing new, weekly."""
        for name in ("lint", "test", "coverage"):
            condition = workflow["jobs"][name].get("if")
            assert condition == "github.event_name != 'schedule'", (
                f"the {name} job's condition is {condition!r}, so the weekly cron runs it too"
            )

    def test_the_canary_upgrades_the_sdk_before_it_tests_anything(self, workflow: dict[str, Any]) -> None:
        """Order is the assertion. Upgrading after the tests run tests the pinned version and reports green."""
        runs = _run_blocks(workflow["jobs"]["sdk-compat"])

        upgrades = [index for index, block in enumerate(runs) if "--upgrade strands-agents" in block]
        assert upgrades, f"no step upgrades strands-agents; the canary's steps run {runs}"

        selects = [index for index, block in enumerate(runs) if "-m compat" in block]
        assert selects, f"no step selects the compat tier; the canary's steps run {runs}"

        assert min(upgrades) < min(selects), (
            f"the upgrade happens at step {min(upgrades)} and the compat tier runs at step {min(selects)}, "
            f"so the tier ran against the pinned SDK"
        )

    def test_the_canary_typechecks_against_the_upgraded_sdk(self, workflow: dict[str, Any]) -> None:
        """The memory store's conformance check asserts a mypy result, so mypy has to see the new version.

        That check is ``AT-052``, and it is named down here rather than in the line above deliberately. The
        gate in ``conftest.py`` indexes ids out of first paragraphs and lets a test's own declaration
        override a class's, per id and across the whole suite. With the id in the summary line, this test
        about a YAML file became the only carrier of ``AT-052``, and the protocol check's own class lost it.
        Measured while auditing the release checklist: three rows cited the id and all three resolved here.

        The Protocol subset is satisfied by assignment under mypy. A member added upstream surfaces there and
        nowhere else, which makes the typecheck the load-bearing half of the canary rather than a formality.
        """
        runs = _run_blocks(workflow["jobs"]["sdk-compat"])

        assert any("hatch run typecheck" in block for block in runs), (
            f"the canary never typechecks, so an added Protocol member goes unreported; its steps run {runs}"
        )

    def test_the_canary_runs_the_contract_tier_against_the_upgraded_sdk(self, workflow: dict[str, Any]) -> None:
        """The clause of this item that was unmet for eight tasks, and the reason it was unmet is gone.

        The canary is specified to run the contract tier beside both protocol checks. ``REQ-E-107``
        deferred that clause on the grounds that the tier "would cost the contract tier a deployment",
        which was true of the two deployments considered then -- a repository secret and a provisioned
        cluster. A service container is neither, so the clause is now free and the deferral no longer
        has a reason behind it.

        Asserted after the upgrade for the same reason the compat selector is: a tier that runs before
        ``pip install --upgrade`` tests the pinned SDK and reports green about the wrong version.
        """
        job = workflow["jobs"]["sdk-compat"]
        runs = _run_blocks(job)

        assert "MONGODB_URI" in (job.get("env") or {}), (
            "the canary sets no MONGODB_URI, so every contract test in it skips and the cron reports a pass"
        )

        upgrades = [index for index, block in enumerate(runs) if "--upgrade strands-agents" in block]
        selects = [index for index, block in enumerate(runs) if "-m contract" in block]
        assert selects, f"the canary never selects the contract tier; its steps run {runs}"
        assert min(upgrades) < min(selects), (
            f"the upgrade happens at step {min(upgrades)} and the contract tier runs at step {min(selects)}, "
            f"so the tier ran against the pinned SDK"
        )


class TestTheCompatTierSelectsBothProtocolChecks:
    """AT-119 / REQ-E-117 -- the canary runs ``-m compat``, so a check without that marker never runs.

    Two conformance checks run against the newest published SDK, one per protocol.
    Auditing the markers found that only one of them carried it, so a job specified to catch
    upstream drift in both protocols could only ever see it in one. The marker is the selection, which is
    why it is asserted here rather than left to a reader of two decorator lines.

    Its own class rather than a method of the canary class above, because the gate's precedence rule works
    per id over the whole suite: inside that class this test would also have been indexed as a carrier of
    the canary's own id, and a failure about markers would have been reported against the cron.
    """

    def test_both_protocol_checks_carry_the_marker(
        self, checklist_gate: ChecklistGate, request: pytest.FixtureRequest
    ) -> None:
        """Read from the collected session rather than by grepping for a decorator line.

        The question is which tests ``-m compat`` selects, and the marker as pytest resolved it is the only
        thing that answers it. Reading the marker also covers the case a grep would miss, where the id moves
        to a test outside the marked class.
        """
        if narrowing := _narrowing(request.config):
            pytest.skip(f"this run collected a subset ({narrowing}), so an absent marker cannot be told apart")

        collected = {item.nodeid: item for item in request.session.items}
        unmarked: dict[str, list[str]] = {}
        for identifier in PROTOCOL_CHECK_IDS:
            carriers = checklist_gate.by_id.get(identifier, ())
            assert carriers, f"no collected test declares {identifier}, so the canary has nothing to select"
            missing = [
                node
                for node in carriers
                if node not in collected or collected[node].get_closest_marker("compat") is None
            ]
            if missing:
                unmarked[identifier] = sorted(missing)

        assert unmarked == {}, (
            f"tests carrying a protocol-check id that '-m compat' does not select, so the weekly canary "
            f"never runs them against the newest published SDK: {unmarked}"
        )


class TestTheContractTierRunsOnEveryChange:
    """AT-130 / REQ-E-127 -- the tier a service container satisfies, gated on every change rather than on a secret.

    ``conftest.py`` defines the tier as "any real MongoDB, including a plain ``mongod``", and this package
    opens no transaction, no session and no change stream, so nothing in it needs a replica set. A GitHub
    Actions service container is therefore a real MongoDB by the tier's own definition, and it needs no
    secret, no provisioning step and no access list.

    The value of running it here is arrival time. The first run that pointed ``MONGODB_URI`` at a plain
    deployment found one class relying on the wider tier; that finding is worth having on a pull request
    rather than much later, and this job is what makes it arrive there.
    """

    @staticmethod
    def _jobs_setting(workflow: dict[str, Any], variable: str) -> dict[str, dict[str, Any]]:
        """Every job whose ``env:`` block sets ``variable``, keyed by job name."""
        return {name: job for name, job in workflow["jobs"].items() if variable in (job.get("env") or {})}

    def test_every_job_setting_the_contract_variable_runs_the_tier(self, workflow: dict[str, Any]) -> None:
        """Both halves, because either alone is a job that reports green without asserting anything.

        A job that sets the variable and never selects the tier runs the unit tier twice. A job that selects
        the tier with no variable set reports every contract test as skipped, which a green check reads as
        passed -- the exact failure ``tests/test_tier_gating.py`` spends five of its six classes on.

        Two jobs set it: the one that gates a change, and the weekly canary, which runs the same tier
        against the newest published SDK. Asserted over all of them rather than over a count, so a third
        cannot arrive half-wired.
        """
        setting = self._jobs_setting(workflow, "MONGODB_URI")
        assert setting, "no job sets MONGODB_URI, so the contract tier runs nowhere"

        silent = {name: _run_blocks(job) for name, job in setting.items()}
        silent = {name: runs for name, runs in silent.items() if not any("-m contract" in block for block in runs)}
        assert silent == {}, f"these jobs set MONGODB_URI and never select the tier it is for: {silent}"

    def test_every_deployment_is_a_service_container_rather_than_a_credential(self, workflow: dict[str, Any]) -> None:
        """The half that makes this free, and the half a future edit is most likely to undo.

        A maintainer who wanted the tier run against a cluster would reach for a repository secret, which
        puts a live credential in a repository this package is moving out of. The container is the answer
        that needs no such decision, so the image is asserted rather than left as a comment.
        """
        for name, job in self._jobs_setting(workflow, "MONGODB_URI").items():
            services = job.get("services") or {}
            images = [service.get("image", "") for service in services.values()]
            assert any(image.split(":")[0] == "mongo" for image in images), (
                f"the {name} job declares no MongoDB service container; its services are {services}"
            )

            uri = job["env"]["MONGODB_URI"]
            assert "secrets." not in uri, (
                f"the {name} job reaches for a credential rather than the container beside it: {uri!r}"
            )
            assert "localhost" in uri or "127.0.0.1" in uri, (
                f"the {name} job's connection string is {uri!r}, which does not point at that container"
            )

    def test_the_tier_gates_a_change_rather_than_only_the_cron(self, workflow: dict[str, Any]) -> None:
        """Pull requests are specified to gate on this tier, so weekly alone is the wrong cadence.

        The canary runs it too, and that is a different claim: weekly against the newest SDK, which is
        ``AT-107``'s subject rather than this one's. What this asserts is that a change does not merge
        without it.
        """
        conditions = {name: job.get("if") for name, job in self._jobs_setting(workflow, "MONGODB_URI").items()}

        assert "github.event_name != 'schedule'" in conditions.values(), (
            f"no job runs the contract tier on a change; the jobs that set MONGODB_URI are {conditions}"
        )


class TestTheTiersThatNeedAClusterRunOnlyOnTheSchedule:
    """AT-131 / REQ-E-128 -- the other half, where a secret is unavoidable and a pull request is the wrong trigger.

    ``integration`` needs mongot and ``autoembed`` needs a cluster with a registered embedding model, so
    neither can be stood up by a container. Both therefore run against a repository secret, on the cron, for
    two reasons that point the same way: a pull request from a fork receives no secret, so the tiers would
    skip on exactly the contributions that most need checking and report the skip as a pass; and they cost
    33 minutes of wall clock against a cloud cluster when measured, which is not a gate anybody keeps.

    The trigger is what this class asserts, rather than the intent behind it.
    """

    #: The two variables that can only be satisfied by a cluster, and so can only arrive as secrets.
    CLUSTER_VARIABLES = ("MONGODB_ATLAS_URI", "MONGODB_AUTOEMBED_URI")

    def test_every_job_holding_a_cluster_secret_runs_only_on_the_schedule(self, workflow: dict[str, Any]) -> None:
        """One condition per job rather than per variable: a second job is as exposed as a second key."""
        holders = {
            name: job
            for name, job in workflow["jobs"].items()
            if any(variable in (job.get("env") or {}) for variable in self.CLUSTER_VARIABLES)
        }
        assert holders, f"no job supplies {self.CLUSTER_VARIABLES}, so neither tier runs anywhere"

        wrong = {
            name: job.get("if") for name, job in holders.items() if job.get("if") != "github.event_name == 'schedule'"
        }
        assert wrong == {}, f"these jobs hold a cluster secret outside the cron: {wrong}"

    def test_both_cluster_tiers_are_actually_selected(self, workflow: dict[str, Any]) -> None:
        """A secret supplied and a marker never selected is a credential in a repository for nothing."""
        holders = [
            job
            for job in workflow["jobs"].values()
            if any(variable in (job.get("env") or {}) for variable in self.CLUSTER_VARIABLES)
        ]
        runs = "\n".join(block for job in holders for block in _run_blocks(job))

        for marker in ("integration", "autoembed"):
            assert marker in runs, f"no step in a cluster job selects the {marker} tier; they run {runs!r}"

    def test_no_trigger_hands_a_fork_the_secrets(self, workflow: dict[str, Any]) -> None:
        """``pull_request_target`` runs a fork's pull request with the base repository's secrets.

        It is the one trigger that would defeat everything above, and it is easy to reach for: it is what
        somebody adds when they want a fork's pull request to see a check that needs a secret. Asserted on
        the trigger list rather than left to review.
        """
        triggers = sorted(str(key) for key in _triggers(workflow))

        assert "pull_request_target" not in triggers, (
            f"the workflow triggers on {triggers}, and pull_request_target runs a fork's code with our secrets"
        )


class TestNoCredentialReachesACommandLine:
    """AT-108 / REQ-E-108 -- a rule written while there is nothing to break it.

    Written while there was no secret in the file, so the tests below passed on an empty set for two tasks.
    That was the point of writing them then: the change that eventually introduced one -- the scheduled
    Atlas job two classes above -- should meet a failing test rather than put a masked value in a public
    build log. The set is no longer empty, so a rule about a hypothetical is now a rule about two keys.
    """

    def test_no_command_or_step_name_interpolates_a_secret(self, workflow: dict[str, Any]) -> None:
        """GitHub echoes a step's command above its output, and masking there is best effort."""
        for scope, values in (("run", _run_blocks(workflow)), ("name", _values_under(workflow, "name"))):
            offenders = [value for value in values if "secrets." in value]
            assert offenders == [], f"these {scope} values interpolate a secret: {offenders}"

    def test_every_secret_reference_sits_under_env(self, workflow: dict[str, Any]) -> None:
        """The positive form of the rule: a secret may appear, in exactly one kind of place.

        Checked against the parsed document rather than the file's text, so a secret named in a comment
        cannot fail a test about what the workflow does.
        """
        referenced = [value for value in _all_strings(workflow) if "secrets." in value]
        permitted = set(_values_under(workflow, "env"))

        stray = [value for value in referenced if value not in permitted]
        assert stray == [], f"these secret references are not env values: {stray}"

    def test_every_tier_variable_the_workflow_sets_is_set_under_env(self, workflow: dict[str, Any]) -> None:
        """No `export` in a shell line, which is how a value ends up in a log with no masking at all.

        Covers the four names `conftest.py` and `test_packaging.py` read between them, whether or not this
        workflow sets any of them today. It sets one.
        """
        tier_variables = (
            "MONGODB_URI",
            "MONGODB_ATLAS_URI",
            "MONGODB_AUTOEMBED_URI",
            "STRANDS_MONGODB_BUILD_TIER",
        )
        commands = _run_blocks(workflow)
        environments = set(_values_under(workflow, "env"))

        for name in tier_variables:
            in_commands = [block for block in commands if name in block]
            assert in_commands == [], f"{name} appears in a command rather than under env:: {in_commands}"

        assert any("STRANDS_MONGODB_BUILD_TIER" in key for key in _all_strings(workflow)), (
            "no job switches the build tier on"
        )
        assert environments, "the workflow declares no env: block at all"


class TestTheBuildTierRunsSomewhere:
    """AT-109 / REQ-E-109 -- the last tier in the suite with nothing that runs it.

    ``test_packaging.py:66`` turns it off by default because it builds a wheel and an sdist and resolves
    dependencies from PyPI. The variable is the one tier gate that is not a connection string, so a runner
    with network is all it needs and it never waited on a deployment decision at all.
    """

    def test_exactly_one_job_switches_the_build_tier_on(self, workflow: dict[str, Any]) -> None:
        """One, because the artifact is the same artifact on every interpreter."""
        enabling = [
            name for name, job in workflow["jobs"].items() if "STRANDS_MONGODB_BUILD_TIER" in (job.get("env") or {})
        ]

        assert len(enabling) == 1, f"the build tier is switched on in {enabling}, and it should be in exactly one"

    def test_the_build_tier_is_not_in_the_matrix_job(self, workflow: dict[str, Any]) -> None:
        """Four wheels built from one commit, and three of them discarded unread."""
        matrix_job = workflow["jobs"]["test"]

        assert "STRANDS_MONGODB_BUILD_TIER" not in (matrix_job.get("env") or {}), (
            "the matrix job builds the same artifact once per interpreter"
        )

    def test_the_tier_is_switched_on_with_a_value_the_gate_reads_as_set(self, workflow: dict[str, Any]) -> None:
        """The gate is ``not os.environ.get(...)``, so an empty string reads as off and a skip looks like a pass."""
        values = [
            job["env"]["STRANDS_MONGODB_BUILD_TIER"]
            for job in workflow["jobs"].values()
            if "STRANDS_MONGODB_BUILD_TIER" in (job.get("env") or {})
        ]

        assert values, "no job switches the build tier on"
        for value in values:
            assert str(value).strip(), f"the build tier is set to {value!r}, which the gate reads as unset"


class TestNothingNeedsMoreThanTheOldestInterpreterInTheMatrix:
    """AT-110 / REQ-E-110 — the floor's grammar, checked from whatever interpreter is running.

    ``AT-105`` asserts the matrix equals the classifiers, which claim 3.10 as the floor. This asserts the
    weaker-looking thing that actually broke: that the tree can be *parsed* there. An audit found an
    ``async for`` comprehension nested inside a plain one at ``test_examples.py:384``, legal from 3.11 and a
    ``SyntaxError`` on 3.10, sitting inside a triple-quoted string that a child interpreter runs. No linter
    parses that string and no type checker sees it, so the first thing to compile it was Python 3.10 itself.

    **Why the obvious guard is not the one used here.** ``ast.parse(source, feature_version=(3, 10))`` reads
    like the answer and is not: measured on 3.14, it accepts the nested form, because ``feature_version``
    gates a short list of constructs and this is not among them. The floor's grammar is not reachable from a
    newer interpreter through the standard library, so this checks the one construct structurally instead.

    That is the honest limit. It catches this construct and no other, and the 3.10 leg of the matrix stays
    the only thing that would find a different one. Which is an argument for keeping the matrix.
    """

    @staticmethod
    def _has_async_clause(comprehension: ast.expr) -> bool:
        generators = getattr(comprehension, "generators", [])
        return any(generator.is_async for generator in generators)

    @classmethod
    def _rejected_by_the_floor(cls, tree: ast.AST) -> list[int]:
        """Line numbers of comprehensions Python 3.10 refuses to compile.

        The rule is exact rather than conservative, measured on 3.10.11 against all four shapes: an
        asynchronous comprehension is rejected when the comprehension enclosing it is synchronous, because
        that outer comprehension is its own synchronous scope. Enclosed by an *asynchronous* comprehension it
        is accepted, and unenclosed inside an ``async def`` it is accepted.
        """
        offenders: list[int] = []

        def visit(node: ast.AST, enclosing: ast.expr | None) -> None:
            if isinstance(node, _COMPREHENSION_TYPES):
                if cls._has_async_clause(node) and enclosing is not None and not cls._has_async_clause(enclosing):
                    offenders.append(node.lineno)
                enclosing = node
            for child in ast.iter_child_nodes(node):
                visit(child, enclosing)

        visit(tree, None)
        return offenders

    def test_the_detector_matches_what_the_floor_actually_rejects(self) -> None:
        """The sweep below reports nothing once the tree is clean, so this is what keeps it from being vacuous.

        All four shapes were compiled on 3.10.11 to fix the rule. Only the first is an error there.
        """
        # Joined at runtime rather than written whole, because the sweep below walks the string literals in
        # this file too and this one is precisely what it looks for. Explicit ``+`` rather than adjacency:
        # adjacent literals are folded during parsing, so the sweep would see one string and flag it. Each
        # half alone is a ``SyntaxError``, which the sweep skips.
        rejected = "async def m():\n    return [[e async for e in s()] " + "for _ in r()]\n"
        accepted = (
            "async def m():\n"
            "    a = [[e async for e in s()] async for _ in t()]\n"
            "    b = [e async for e in s()]\n"
            "    c = [[e for e in s()] for _ in r()]\n"
            "    return a, b, c\n"
        )

        assert self._rejected_by_the_floor(ast.parse(rejected)) == [2]
        assert self._rejected_by_the_floor(ast.parse(accepted)) == []

    def test_no_module_uses_a_construct_the_floor_rejects(self) -> None:
        """Every ``.py`` file mypy is pointed at, parsed as itself."""
        for root in SOURCE_ROOTS:
            for path in sorted((PROJECT_ROOT / root).rglob("*.py")):
                offenders = self._rejected_by_the_floor(ast.parse(path.read_text()))

                assert not offenders, f"{path.relative_to(PROJECT_ROOT)} lines {offenders} will not compile on 3.10"

    def test_no_embedded_script_uses_a_construct_the_floor_rejects(self) -> None:
        """And every multi-line string in those files that happens to be Python, which is the half that broke.

        Prose docstrings mostly fail to parse and are skipped, so the count is not the point; the three script
        constants handed to ``python -c`` are (``test_examples.py``, ``test_extras.py``, ``test_tier_gating.py``).
        The detector's own test above is what proves a clean sweep means something.
        """
        for root in SOURCE_ROOTS:
            for path in sorted((PROJECT_ROOT / root).rglob("*.py")):
                for node in ast.walk(ast.parse(path.read_text())):
                    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                        continue
                    if "\n" not in node.value:
                        continue
                    try:
                        embedded = ast.parse(node.value)
                    except SyntaxError:
                        continue

                    offenders = self._rejected_by_the_floor(embedded)

                    assert not offenders, (
                        f"{path.relative_to(PROJECT_ROOT)}: the string at line {node.lineno} will not compile "
                        f"on 3.10, at its own lines {offenders}"
                    )
