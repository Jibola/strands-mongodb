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
    # `tomli` is the parser `tomllib` was adopted from, and 2.x is the API `tomllib` shipped, so the
    # alias makes the two interchangeable and every caller below reads one name. It is a dev dependency
    # declared without a version marker on purpose; `pyproject.toml` says why, and the reason is mypy
    # rather than this branch.
    import tomli as tomllib

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
    """The manifest, parsed.

    Returns:
        The parsed document.
    """
    return tomllib.loads(PYPROJECT_PATH.read_text())


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
    ".github/dependabot.yml",
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


#: The triggers only somebody with write access to this repository can cause. An allowlist rather than a
#: list of the dangerous ones, so a trigger nobody has thought about counts as reachable by a stranger
#: until somebody decides otherwise. ``push`` and ``workflow_dispatch`` both require write access;
#: ``schedule`` is pulled by a clock and by no party at all.
TRIGGERS_NEEDING_WRITE_ACCESS = frozenset({"push", "schedule", "workflow_dispatch"})

#: The triggers a change arrives on. A job gated off both of these does not gate a merge, and a job gated
#: on either of them runs on somebody's commit rather than on a decision to run it.
CHANGE_TRIGGERS = frozenset({"pull_request", "push"})

#: One comparison of ``github.event_name`` against a quoted event name. The whole grammar this workflow
#: uses, joined by ``||``; anything else is refused rather than guessed at.
_EVENT_COMPARISON = re.compile(r"github\.event_name\s*(==|!=)\s*'([a-z_]+)'\Z")


def _admitted_events(condition: str | None, triggers: frozenset[str]) -> frozenset[str]:
    """Which of the declared triggers actually reach a job carrying this ``if:`` expression.

    An interpreter rather than a string comparison, because that is the question every caller below has.
    A literal can only answer "is this condition still spelled the way somebody spelled it when they
    wrote the test down", which is a different question with the same green.

    Refuses what it cannot read. A condition outside the grammar raises here instead of being treated as
    reachable, because an unread condition is exactly the case where a wrong answer is most expensive:
    the two readings differ by whether a secret-holding job is open to a fork.
    """
    if condition is None:
        return triggers

    admitted: set[str] = set()
    for term in str(condition).split("||"):
        match = _EVENT_COMPARISON.fullmatch(term.strip())
        assert match is not None, (
            f"this reader does not understand {term.strip()!r} in {condition!r}. Extend it rather than "
            "letting a condition nothing reads count as reachable"
        )
        operator, event = match.groups()
        admitted |= {t for t in triggers if (t == event) == (operator == "==")}
    return frozenset(admitted)


#: One ``${{ ... }}`` interpolation. The two concurrency keys are the only place this workflow computes a
#: value instead of stating one, so the grammar below stays as small as they are.
_INTERPOLATION = re.compile(r"\$\{\{(.+?)\}\}")

#: A context read: ``github.`` and a dotted path. The paths this workflow reads are a closed set, and one
#: it does not read is refused rather than rendered as the empty string -- an unknown key quietly becoming
#: "" is precisely how a group expression collapses two events into one group with nobody the wiser.
_CONTEXT_READ = re.compile(r"github\.([a-z_.]+)\Z")

#: ``github.x == 'literal'``, the only comparison either concurrency key performs.
_CONTEXT_EQUALITY = re.compile(r"github\.([a-z_.]+)\s*==\s*'([a-z_]+)'\Z")


def _context_read(term: str, context: dict[str, str]) -> str:
    """One ``github.x`` lookup, refusing a key this reader holds no value for."""
    match = _CONTEXT_READ.fullmatch(term.strip())
    assert match is not None, (
        f"this reader does not understand {term.strip()!r}. Extend it rather than letting an expression "
        "nothing reads decide which runs share a group"
    )
    key = match.group(1)
    assert key in context, f"{term.strip()!r} reads a context key this reader has no value for: {sorted(context)}"
    return context[key]


def _render(template: object, context: dict[str, str]) -> str:
    """Render a workflow expression against one run's context.

    The same reasoning as :func:`_admitted_events`, applied to the other half of the file that decides
    whether a job runs. Comparing the expression to the string somebody typed answers "is this still
    spelled that way". What the tests below ask is "do these two runs land in the same group", and
    nothing but evaluating it answers that.
    """

    def one(match: re.Match[str]) -> str:
        terms = match.group(1).split("||")
        if len(terms) == 1:
            equality = _CONTEXT_EQUALITY.fullmatch(terms[0].strip())
            if equality is not None:
                key, literal = equality.groups()
                assert key in context, f"{terms[0].strip()!r} reads a key this reader has no value for"
                return "true" if context[key] == literal else "false"
            return _context_read(terms[0], context)
        # ``||`` is first-non-empty in a workflow expression, and only a lookup can be empty. An equality
        # inside one would be read as its truthy spelling "false", so the reader refuses it in _context_read.
        for term in terms:
            value = _context_read(term, context)
            if value:
                return value
        return ""

    return _INTERPOLATION.sub(one, str(template))


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


class TestEveryJobCanBeStartedOnDemand:
    """``AT-139`` / ``REQ-E-139`` -- the mirror of the class above, asked about jobs instead of files.

    ``AT-138`` asks whether a change to an asserted file reaches a job. This asks whether a declared job
    can be started by a person, and whether a declared trigger reaches anything at all.

    It is not hypothetical. ``sdk-compat`` and ``atlas`` were gated on ``github.event_name ==
    'schedule'`` against one weekly cron, and in the first thirty-four runs of this workflow neither had
    executed once: eighteen on ``pull_request``, sixteen on ``push``, none on the schedule. Both reported
    ``skipping`` on every pull request, which sits in the checks list beside a pass in the same colour.
    The SDK canary had never run, and the two secrets the other job holds had never been used.

    On demand rather than merely reachable, because a path existed the whole time -- it just took a week
    and could not be walked twice. A job you can only reach by waiting cannot be debugged: the answer to
    "did that fix it" is another week away.
    """

    #: The one trigger no person can pull. Everything else in an ``on:`` block is caused by somebody.
    UNPULLABLE = "schedule"

    def _declared(self, workflow: dict[str, Any]) -> frozenset[str]:
        return frozenset(str(key) for key in _triggers(workflow))

    def test_every_job_can_be_started_by_a_person(self, workflow: dict[str, Any]) -> None:
        """AT-139 -- a job only the clock reaches is a job nobody can use to answer a question."""
        declared = self._declared(workflow)
        on_demand = declared - {self.UNPULLABLE}

        stranded = {
            name: job.get("if")
            for name, job in workflow["jobs"].items()
            if not (_admitted_events(job.get("if"), declared) & on_demand)
        }
        assert stranded == {}, (
            f"the triggers a person can pull are {sorted(on_demand)}, and these jobs answer to none of "
            f"them: {stranded}. Each can only be run by waiting for the cron, once"
        )

    def test_every_trigger_reaches_some_job(self, workflow: dict[str, Any]) -> None:
        """AT-139 -- the other direction, and a guard rather than a finding.

        A trigger gating no job still produces a run, and that run is green because it did nothing. An
        absent check is a hole; a green one over an empty job list is a wrong answer.
        """
        declared = self._declared(workflow)
        conditions = [job.get("if") for job in workflow["jobs"].values()]

        idle = sorted(
            trigger
            for trigger in declared
            if not any(trigger in _admitted_events(condition, declared) for condition in conditions)
        )
        assert idle == [], f"these triggers start a run in which every job is skipped: {idle}"

    def test_the_reader_interprets_the_condition_rather_than_matching_it(self) -> None:
        """AT-139 -- the vacuity guard. Both tests above pass on everything if the reader admits everything."""
        declared = frozenset({"push", "pull_request", "schedule", "workflow_dispatch"})

        assert _admitted_events(None, declared) == declared, "an ungated job runs on every declared trigger"
        assert _admitted_events("github.event_name == 'schedule'", declared) == {"schedule"}
        assert _admitted_events("github.event_name != 'schedule'", declared) == declared - {"schedule"}
        assert _admitted_events(
            "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'", declared
        ) == {"schedule", "workflow_dispatch"}
        assert _admitted_events("github.event_name == 'release'", declared) == frozenset(), (
            "an event no trigger declares reaches nothing, which is the shape of the defect above"
        )

    def test_the_reader_refuses_a_condition_it_cannot_read(self) -> None:
        """AT-139 -- the failure mode of a reader is answering anyway, and a wrong yes here reads as green."""
        for condition in ("success()", "github.ref == 'refs/heads/main'", "github.event_name == 'push' && true"):
            with pytest.raises(AssertionError, match="does not understand"):
                _admitted_events(condition, frozenset({"push", "schedule"}))


class TestOneRunNeverCancelsAnUnrelatedOne:
    """``AT-159`` / ``REQ-E-152`` -- the concurrency group, which had three unrelated events inside it.

    ``github.ref`` is ``refs/heads/main`` for a push to ``main``, for a dispatch against ``main`` and for
    the weekly cron alike. Keying the group on it put all three in one group, and ``cancel-in-progress:
    true`` then made whichever started last kill the others.

    The job that loses is always the same one. ``Atlas tiers`` measured 36 minutes, it is the only job
    here that cannot be re-run cheaply, and the two events that start it -- the cron and a dispatch --
    are exactly the two with no second chance: the cron's is a week away, and a dispatch is a button
    somebody pressed because they wanted an answer now. A merge landing during either run destroyed it,
    and a cancelled run sits in the checks list looking like a failure.

    Superseding is right for a pull request and wrong everywhere else, which is the asymmetry these tests
    pin. A second push to a branch makes the first run's verdict worthless -- it grades a commit nobody
    will merge. A push, a dispatch and a cron each grade a commit that is already on ``main``, where the
    cancelled run is the only verdict that commit was ever going to get.
    """

    #: One run's context. A push to ``main`` and a dispatch against ``main`` agree on every value here
    #: except ``event_name`` and ``run_id``, which is the whole of the defect above.
    BASE_CONTEXT = {
        "workflow": "CI - Python",
        "event_name": "push",
        "ref": "refs/heads/main",
        "run_id": "35076582911",
        "sha": "eb9ac2d6",
        "event.pull_request.number": "",
    }

    def _context(self, **overrides: str) -> dict[str, str]:
        return {**self.BASE_CONTEXT, **overrides}

    def _group(self, workflow: dict[str, Any], **overrides: str) -> str:
        return _render(workflow["concurrency"]["group"], self._context(**overrides))

    def _pull_request(self, number: str, run_id: str) -> dict[str, str]:
        return self._context(
            event_name="pull_request",
            ref=f"refs/pull/{number}/merge",
            run_id=run_id,
            **{"event.pull_request.number": number},
        )

    def _supersedes(self, workflow: dict[str, Any], context: dict[str, str]) -> bool:
        """Whether a newer run cancels an in-progress one, for a run in this context."""
        rendered = _render(workflow["concurrency"]["cancel-in-progress"], context)
        assert rendered in {"true", "false"}, (
            f"cancel-in-progress rendered {rendered!r} for a {context['event_name']} run. A literal there "
            "applies one answer to every event, and the events differ: see this class's docstring"
        )
        return rendered == "true"

    def test_no_two_events_against_main_share_a_group(self, workflow: dict[str, Any]) -> None:
        """AT-159 -- the finding. Three events, one ref, and one of them cancelling another."""
        groups = {
            event: self._group(workflow, event_name=event, run_id=str(index))
            for index, event in enumerate(("push", "workflow_dispatch", "schedule"), start=1)
        }
        assert len(set(groups.values())) == len(groups), (
            f"these events against main share a concurrency group: {groups}. A dispatch or the weekly "
            "cron is cancelled by any merge that lands while it runs, and Atlas tiers runs 36 minutes"
        )

    def test_two_dispatches_do_not_share_a_group(self, workflow: dict[str, Any]) -> None:
        """AT-159 -- two presses of a "run CI now" button are two questions, not one asked twice."""
        first = self._group(workflow, event_name="workflow_dispatch", run_id="1")
        second = self._group(workflow, event_name="workflow_dispatch", run_id="2")
        assert first != second, f"both dispatches land in {first!r}, so the second cancels the first"

    def test_two_runs_of_one_pull_request_do_share_a_group(self, workflow: dict[str, Any]) -> None:
        """AT-159 -- the other direction, and the reason the key is not simply unique per run.

        Coalescing a pull request's runs is the behaviour worth keeping. Losing it would run the full
        matrix against every intermediate commit of a branch somebody is still pushing to.
        """
        first = _render(workflow["concurrency"]["group"], self._pull_request("20", run_id="1"))
        second = _render(workflow["concurrency"]["group"], self._pull_request("20", run_id="2"))
        assert first == second, f"two runs of one pull request landed in {first!r} and {second!r}"

    def test_two_pull_requests_do_not_share_a_group(self, workflow: dict[str, Any]) -> None:
        """AT-159 -- the guard on the test above, which a constant group expression also passes."""
        twenty = _render(workflow["concurrency"]["group"], self._pull_request("20", run_id="1"))
        twenty_one = _render(workflow["concurrency"]["group"], self._pull_request("21", run_id="2"))
        assert twenty != twenty_one, f"both pull requests land in {twenty!r}, so one cancels the other"

    def test_only_a_pull_request_run_is_superseded(self, workflow: dict[str, Any]) -> None:
        """AT-159 -- ``REQ-E-152``: the asymmetry, asserted in both directions."""
        assert self._supersedes(workflow, self._pull_request("20", run_id="1")), (
            "a second push to a pull request should cancel the first run, which grades a commit nobody will merge"
        )
        kept = [
            event
            for event in ("push", "workflow_dispatch", "schedule")
            if self._supersedes(workflow, self._context(event_name=event))
        ]
        assert kept == [], (
            f"a newer run cancels an in-progress one for {kept}, and each of those grades a commit "
            "already on main, where the cancelled run is the only verdict that commit will ever get"
        )

    def test_the_reader_resolves_the_fallback_rather_than_the_first_term(self) -> None:
        """AT-159 -- the vacuity guard. Every test above passes if the reader renders everything to ""."""
        template = "${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}"

        on_a_pull_request = _render(template, {**self.BASE_CONTEXT, "event.pull_request.number": "20"})
        assert on_a_pull_request == "CI - Python-20", on_a_pull_request

        on_a_push = _render(template, self.BASE_CONTEXT)
        assert on_a_push == "CI - Python-35076582911", on_a_push

        assert _render("${{ github.event_name == 'push' }}", self.BASE_CONTEXT) == "true"
        assert _render("${{ github.event_name == 'pull_request' }}", self.BASE_CONTEXT) == "false"

    def test_the_reader_refuses_an_expression_it_cannot_read(self) -> None:
        """AT-159 -- a reader that answers anyway turns an unread expression into a green test."""
        for template in (
            "${{ github.head_ref }}",
            "${{ contains(github.ref, 'main') }}",
            "${{ github.event.pull_request.number || github.ref || github.sha }}-x",
        ):
            with pytest.raises(AssertionError, match="does not understand|no value for"):
                _render(template, {"workflow": "CI - Python", "event_name": "push", "run_id": "1"})


class TestTheMatrixCoversEveryVersionTheClassifiersClaim:
    """AT-105 / REQ-E-105 -- the matrix and the classifiers, compared as sets in both directions.

    Equality rather than containment. A matrix narrower than the claim ships support for an untested
    interpreter; a matrix wider than the claim tests one the package says it does not support, and the
    honest fix for that is a new classifier, not a quieter test.
    """

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

    def test_the_canary_never_runs_on_a_change(self, workflow: dict[str, Any]) -> None:
        """Otherwise every pull request resolves the latest SDK and the pinned floor is never the thing tested.

        Off the change triggers rather than on the cron alone. Those are different claims, and it is the
        first one that carries the reason: a maintainer starting the canary by hand has decided to resolve
        the latest SDK, which is the job's entire point, while a contributor pushing a commit has not.
        """
        declared = frozenset(str(key) for key in _triggers(workflow))
        on_a_change = sorted(_admitted_events(workflow["jobs"]["sdk-compat"].get("if"), declared) & CHANGE_TRIGGERS)

        assert on_a_change == [], (
            f"the canary runs on {on_a_change}, so an ordinary change resolves the latest SDK and the "
            "pinned floor is never the version under test"
        )

    def test_the_change_triggered_jobs_sit_out_the_schedule(self, workflow: dict[str, Any]) -> None:
        """The mirror of the above. A cron that also runs the pinned jobs reports nothing new, weekly."""
        declared = frozenset(str(key) for key in _triggers(workflow))

        for name in ("lint", "test", "coverage"):
            admitted = _admitted_events(workflow["jobs"][name].get("if"), declared)
            assert "schedule" not in admitted, (
                f"the {name} job's condition is {workflow['jobs'][name].get('if')!r}, so the weekly cron "
                "runs it too and reports the same answer it reported on the commit"
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
        declared = frozenset(str(key) for key in _triggers(workflow))
        conditions = {name: job.get("if") for name, job in self._jobs_setting(workflow, "MONGODB_URI").items()}
        on_a_change = {
            name: sorted(_admitted_events(condition, declared) & CHANGE_TRIGGERS)
            for name, condition in conditions.items()
        }

        assert any(on_a_change.values()), (
            f"no job runs the contract tier on a change; the jobs that set MONGODB_URI are {conditions}"
        )


class TestTheTiersThatNeedAClusterAreNeverHandedToAStranger:
    """AT-131 / REQ-E-128, and ``AT-140`` / ``REQ-E-140`` -- where a secret is unavoidable and a pull request
    is the wrong trigger.

    ``integration`` needs mongot and ``autoembed`` needs a cluster with a registered embedding model, so
    neither can be stood up by a container. Both therefore run against a repository secret, and off
    ``pull_request`` in particular, for two reasons that point the same way: a pull request from a fork
    receives no secret, so the tiers would skip on exactly the contributions that most need checking and
    report the skip as a pass; and they cost 33 minutes of wall clock against a cloud cluster when
    measured, which is not a gate anybody keeps. ``REQ-E-165`` later added ``push`` to the same job, which
    this class permits and ``AT-171`` requires: a push to ``main`` needs write access, so it is not a
    trigger a stranger can pull.

    This was once asserted as ``if:`` equalling ``"github.event_name == 'schedule'"``. That was right
    about the file and wrong about the claim, in both directions: it broke on the change that added a
    second trigger exactly as safe as the cron, and it would have passed a condition naming some other
    safe-sounding event that a stranger can in fact cause. The claim is about who can pull the trigger,
    so that is what is asked now.
    """

    #: The two variables that can only be satisfied by a cluster, and so can only arrive as secrets.
    CLUSTER_VARIABLES = ("MONGODB_ATLAS_URI", "MONGODB_AUTOEMBED_URI")

    def _exposure(self, workflow: dict[str, Any]) -> dict[str, list[str]]:
        """Per secret-holding job, the declared triggers a party without write access can cause."""
        declared = frozenset(str(key) for key in _triggers(workflow))
        reachable_by_a_stranger = declared - TRIGGERS_NEEDING_WRITE_ACCESS

        exposed = {}
        for name, job in workflow["jobs"].items():
            if not any(variable in (job.get("env") or {}) for variable in self.CLUSTER_VARIABLES):
                continue
            through = _admitted_events(job.get("if"), declared) & reachable_by_a_stranger
            if through:
                exposed[name] = sorted(through)
        return exposed

    def test_no_trigger_a_stranger_can_cause_reaches_a_cluster_secret(self, workflow: dict[str, Any]) -> None:
        """AT-140 -- one condition per job rather than per variable: a second job is as exposed as a second key."""
        holders = [
            name
            for name, job in workflow["jobs"].items()
            if any(variable in (job.get("env") or {}) for variable in self.CLUSTER_VARIABLES)
        ]
        assert holders, f"no job supplies {self.CLUSTER_VARIABLES}, so neither tier runs anywhere"

        assert self._exposure(workflow) == {}, (
            f"these jobs supply a cluster credential under a trigger somebody without write access to this "
            f"repository can cause: {self._exposure(workflow)}"
        )

    def test_the_exposure_check_can_actually_see_an_exposure(self, workflow: dict[str, Any]) -> None:
        """AT-140 -- the vacuity guard, because the test above asserts that something is empty.

        Two conditions that would hand a key to a stranger, and the second is the one worth writing down:
        it never mentions a pull request at all. ``!=`` against the cron admits every other declared
        trigger, which is how this would arrive -- as somebody copying the gate off ``lint``.
        """
        for condition in (None, "github.event_name != 'schedule'"):
            rigged = {
                **workflow,
                "jobs": {name: {**job, "if": condition} for name, job in workflow["jobs"].items()},
            }
            assert self._exposure(rigged) != {}, (
                f"a cluster job gated on {condition!r} is reachable from a pull request, and this missed it"
            )

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


#: GitHub's default job timeout, in minutes. Not a setting anybody chose — it is what a workflow gets when
#: it declares nothing, and six hours of a hung runner is what it costs.
_DEFAULT_TIMEOUT_MINUTES = 360

#: The ceiling this repository holds itself to. `Atlas tiers` is the long pole at 36 minutes observed, so
#: anything above an hour is either a job nobody measured or the default wearing a number.
_TIMEOUT_CEILING_MINUTES = 60


class TestNoJobCanRunForSixHours:
    """AT-165 / REQ-E-159 — every job declares how long it is allowed to take.

    Nothing in either workflow set ``timeout-minutes``, so every job inherited GitHub's 360. The failure
    that matters is not a slow job; it is a *hung* one, which looks identical to a running one until the
    six hours are up. This repository has three places where that is a real shape rather than a
    hypothetical: ``initialize()`` polls a search index until it is queryable, the cluster tiers talk to
    Atlas over the network, and ``REQ-E-124`` exists because a mongot that is merely unreachable is a
    thing that happens here.

    The budgets are per job rather than one number at the top, because the jobs are not alike. Measured
    over the last eleven runs of this workflow, maxima: ``Lint`` 52s, the five ``Test`` legs 196s,
    ``Contract tier`` 92s, ``Coverage floors`` 263s, ``Latest SDK compatibility`` 145s, and ``Atlas
    tiers`` 2,170s. One ceiling above the slowest would give the fastest job a fifty-fold margin, which is
    a budget that never fires.
    """

    def test_every_job_declares_one(self, workflow: dict[str, Any]) -> None:
        """AT-165 — read from the job mapping, so a job added later has to answer the question too."""
        jobs = workflow["jobs"]
        assert jobs, "the workflow defines no jobs, so this test proves nothing"

        undeclared = sorted(name for name, job in jobs.items() if job.get("timeout-minutes") is None)
        assert undeclared == [], (
            f"these jobs declare no timeout-minutes and so inherit GitHub's {_DEFAULT_TIMEOUT_MINUTES}: "
            f"{undeclared}. A job that hangs rather than fails holds a runner for six hours, and on a "
            f"cluster tier that is indistinguishable from one that is merely waiting on a search index"
        )

    def test_none_of_them_is_the_default_in_disguise(self, workflow: dict[str, Any]) -> None:
        """AT-165 — a budget has to be a number somebody chose, which means it has to be small enough to fire."""
        budgets = {name: job.get("timeout-minutes") for name, job in workflow["jobs"].items()}

        for name, budget in budgets.items():
            assert isinstance(budget, int) and not isinstance(budget, bool), (
                f"job {name!r} declares timeout-minutes {budget!r}, which YAML did not read as an integer"
            )
            assert 0 < budget <= _TIMEOUT_CEILING_MINUTES, (
                f"job {name!r} budgets {budget} minutes. The slowest job in this workflow is the Atlas tier "
                f"at 36 minutes observed, so anything above {_TIMEOUT_CEILING_MINUTES} is the "
                f"{_DEFAULT_TIMEOUT_MINUTES}-minute default with a different number on it"
            )


#: The other workflow. Three of the four gates below are about a property no single file can have on its
#: own: a credential left in a checkout is left in whichever job checked out, and there are nine of them
#: across the two files.
RELEASE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-python.yml"

#: The automated update source for the actions these workflows run. Outside ``python/``, so ``AT-138``
#: requires both change triggers to watch it once this module names it.
DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"


def _checkout_steps(parsed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every ``actions/checkout`` step in a parsed workflow, paired with the job that runs it.

    Matched on the action rather than the reference, so a version bump does not quietly empty the walk.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    for job_name, job in (parsed.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            uses = step.get("uses")
            if isinstance(uses, str) and uses.split("@")[0] == "actions/checkout":
                found.append((job_name, step))
    return found


def _every_checkout() -> list[tuple[str, str, dict[str, Any]]]:
    """Every checkout in the repository, as ``(workflow file, job, step)``."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for path in (WORKFLOW_PATH, RELEASE_WORKFLOW_PATH):
        parsed = yaml.safe_load(path.read_text())
        assert isinstance(parsed, dict), f"{path.name} did not parse as a mapping"
        out.extend((path.name, job, step) for job, step in _checkout_steps(parsed))
    return out


class TestNoCheckoutLeavesACredentialBehind:
    """``AT-169`` / ``REQ-E-163`` -- the token ``actions/checkout`` writes into ``.git/config``.

    By default the action stores an ``http.extraheader`` carrying the job's ``GITHUB_TOKEN`` in the
    checked-out repository's config and leaves it there for the rest of the job. Nothing in either
    workflow pushes, tags or comments, so no step after a checkout has any use for it -- and the contract
    tier runs a shipped example as a subprocess in an environment where ``pip`` has just resolved a
    dependency tree.

    **What this is worth, stated plainly.** Both workflows declare ``permissions: contents: read``, so the
    persisted credential is read-scoped, and the one job that raises ``id-token: write`` checks nothing
    out. This is defence in depth, not a live escalation, and the requirement says so.
    """

    def test_the_walk_actually_finds_the_checkouts(self) -> None:
        """AT-169 -- a gate over an empty list is a gate that passes for the wrong reason."""
        checkouts = _every_checkout()
        assert len(checkouts) >= 9, (
            f"the walk found {len(checkouts)} checkouts across the two workflows and there were nine when "
            f"this was written; a matcher that resolves nothing reports a clean result"
        )

    def test_every_checkout_refuses_to_persist_the_credential(self) -> None:
        """AT-169 -- read from the step, so a checkout added later has to answer the question too."""
        offenders = [
            f"{file}:{job}"
            for file, job, step in _every_checkout()
            if (step.get("with") or {}).get("persist-credentials") is not False
        ]
        assert offenders == [], (
            f"these checkouts leave the job's GITHUB_TOKEN readable in .git/config for every later step: "
            f"{offenders}. Nothing in either workflow pushes, so set `persist-credentials: false`"
        )


#: The two variables that name a real cluster. A job that sets either is a job this gate is about.
_CLUSTER_VARIABLES = ("MONGODB_ATLAS_URI", "MONGODB_AUTOEMBED_URI")


def _cluster_jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every job that hands itself a cluster credential, keyed by job name.

    Identified by the variables it sets rather than by its key, so renaming the job does not silently
    empty this gate.
    """
    return {
        name: job
        for name, job in workflow["jobs"].items()
        if any(variable in (job.get("env") or {}) for variable in _CLUSTER_VARIABLES)
    }


class TestTheClusterTiersRunOnEveryMerge:
    """``AT-171`` / ``REQ-E-165`` -- a seven-day window on the branch tags are cut from.

    The job admitted ``schedule`` and ``workflow_dispatch`` and nothing else. A push to ``main`` triggers
    this workflow but did not trigger this job, so a merge received no ``integration`` and no
    ``autoembed`` coverage until the following Monday -- and ``release-python.yml`` runs only the gate and
    contract tiers, so a tag could be published having never exercised ``$vectorSearch`` with
    ``autoEmbed``, this package's default search mode, against a cluster at all.

    The reason the job stays off the *change* triggers is unchanged: a fork's pull request receives no
    secret, ``conftest.py`` reports both tiers as skipped, and a skip and a pass are the same colour. A
    push to ``main`` is not a fork.
    """

    def test_there_is_a_job_that_needs_a_cluster(self, workflow: dict[str, Any]) -> None:
        """AT-171 -- the vacuity guard. Every assertion below is over this set."""
        assert _cluster_jobs(workflow), (
            f"no job sets any of {_CLUSTER_VARIABLES}, so the three assertions below are about nothing"
        )

    def test_a_merge_to_main_runs_them(self, workflow: dict[str, Any]) -> None:
        """AT-171 -- the finding. A commit on main is the commit a tag is cut from."""
        triggers = frozenset(_triggers(workflow))
        for name, job in _cluster_jobs(workflow).items():
            admitted = _admitted_events(job.get("if"), triggers)
            assert "push" in admitted, (
                f"job {name!r} runs on {sorted(admitted)} and not on a push to main, so a merge gets no "
                f"cluster coverage until the next cron and a tag can be cut from a commit no cluster saw"
            )

    def test_they_stay_off_the_trigger_a_fork_can_pull(self, workflow: dict[str, Any]) -> None:
        """AT-171 -- widening this job must not widen it to the one event that hands a stranger the wheel."""
        triggers = frozenset(_triggers(workflow))
        for name, job in _cluster_jobs(workflow).items():
            admitted = _admitted_events(job.get("if"), triggers)
            assert "pull_request" not in admitted, (
                f"job {name!r} admits pull_request. A fork's pull request receives no secret, so both "
                f"tiers would skip and report the same green as a run that passed"
            )

    def test_a_person_can_still_start_them(self, workflow: dict[str, Any]) -> None:
        """AT-171 -- the property AT-139 holds workflow-wide, asserted here because this gate edits the condition."""
        triggers = frozenset(_triggers(workflow))
        for name, job in _cluster_jobs(workflow).items():
            admitted = _admitted_events(job.get("if"), triggers)
            assert "workflow_dispatch" in admitted, (
                f"job {name!r} no longer admits workflow_dispatch, so the only way to ask this question "
                f"is to merge something or wait for Monday"
            )


#: The ecosystem whose versions nothing in this repository raises by hand.
_ACTIONS_ECOSYSTEM = "github-actions"

#: The ecosystem this repository deliberately does not configure. `pyproject.toml` declares floors with no
#: upper bounds and a paragraph of rationale for each; an automated pull request that raises one would
#: narrow the range this package promises to work on while looking like housekeeping.
_REFUSED_ECOSYSTEMS = frozenset({"pip", "uv", "poetry"})


@pytest.fixture(scope="module")
def dependabot() -> dict[str, Any]:
    """The parsed update configuration.

    Fails rather than skips when absent, for the reason the workflow fixture gives: absence is the defect.
    """
    assert DEPENDABOT_PATH.is_file(), f"REQ-E-166: no update configuration at {DEPENDABOT_PATH}"
    parsed = yaml.safe_load(DEPENDABOT_PATH.read_text())
    assert isinstance(parsed, dict), "the update configuration did not parse as a mapping"
    return parsed


class TestTheActionsHaveAnUpdateSourceAndTheFloorsDoNot:
    """``AT-172`` / ``REQ-E-166`` -- two halves, and the second one is a refusal.

    Nothing raises an action version in this repository, and ``REQ-E-164`` has just added a digest that
    will not raise itself. Both peer repositories run Dependabot on ``github-actions`` with an identical
    seven-day cooldown -- the window in which a yanked or compromised release is usually caught.

    The refusal is the half specific to this package. A library ships its floors: ``strands-agents>=1.48.0``
    and the three beside it are the contract, not an artefact of how a requirements file was written. A
    bot that raises one revokes a promise, so the Python ecosystems are asserted *absent* rather than
    merely left out, because leaving something out is not a decision anybody can see.
    """

    def test_it_watches_the_actions(self, dependabot: dict[str, Any]) -> None:
        """AT-172 -- the first half. Nine checkouts and a digest, and nothing raising any of them."""
        ecosystems = [entry.get("package-ecosystem") for entry in dependabot.get("updates") or []]
        assert _ACTIONS_ECOSYSTEM in ecosystems, (
            f"the update configuration watches {ecosystems} and not {_ACTIONS_ECOSYSTEM!r}, so the action "
            f"references in both workflows -- including the digest REQ-E-164 pins -- go stale unwatched"
        )

    def test_every_entry_waits_before_proposing_a_release(self, dependabot: dict[str, Any]) -> None:
        """AT-172 -- a cooldown, because the day a release lands is the day a bad one is still a release."""
        for entry in dependabot.get("updates") or []:
            ecosystem = entry.get("package-ecosystem")
            cooldown = (entry.get("cooldown") or {}).get("default-days")
            assert isinstance(cooldown, int) and cooldown >= 7, (
                f"the {ecosystem!r} entry declares default-days {cooldown!r}. Both peer repositories wait "
                f"seven, which is the window a yanked release is usually caught in"
            )

    def test_nothing_proposes_a_new_dependency_floor(self, dependabot: dict[str, Any]) -> None:
        """AT-172 -- the refusal, asserted rather than assumed."""
        configured = {entry.get("package-ecosystem") for entry in dependabot.get("updates") or []}
        overreach = sorted(configured & _REFUSED_ECOSYSTEMS)
        assert overreach == [], (
            f"the update configuration watches {overreach}, which proposes new dependency floors. The "
            f"floors in pyproject.toml are the range this package promises to work on and each one has a "
            f"reason written beside it; raising one is a decision, not maintenance"
        )


#: The marker expression that selects the tier which runs a shipped example as a subprocess.
_CONTRACT_SELECTION = "-m contract"

#: What the gate has to run. `--porcelain` is the machine-readable form and, unlike a bare `git status`,
#: it is empty on a clean tree -- which is what makes it usable as a condition.
_TREE_QUERY = "git status --porcelain"


def _jobs_running_shipped_code() -> list[tuple[str, str, dict[str, Any]]]:
    """Every job in the repository that selects the contract tier, as ``(workflow file, job, job body)``.

    Found by what the job runs rather than by its name, because two of the three are not called
    ``contract``.
    """
    out: list[tuple[str, str, dict[str, Any]]] = []
    for path in (WORKFLOW_PATH, RELEASE_WORKFLOW_PATH):
        parsed = yaml.safe_load(path.read_text())
        for name, job in (parsed.get("jobs") or {}).items():
            if any(_CONTRACT_SELECTION in block for block in _run_blocks(job)):
                out.append((path.name, name, job))
    return out


class TestTheTierThatRunsShippedCodeLeavesNothingBehind:
    """``AT-173`` / ``REQ-E-167`` -- an example that writes a file beside itself.

    ``-m contract`` selects the storage quickstart alongside the tier, so one of the two examples in
    ``examples/`` is executed as a subprocess against a real server on every change rather than merely
    linted. A shipped example that leaves a file behind is a defect every reader inherits by copying it,
    and nothing in this repository observed whether one did.

    **Every job that selects the tier, not just the one called ``contract``.** Three do: the contract job
    here, the SDK canary, and the release gate's own contract job. A rule with an exception is a rule
    somebody has to remember.

    **What it cannot catch.** ``--porcelain`` respects ``.gitignore``, so the caches and coverage files
    that file already names stay invisible. What is left is a path nobody declared, which is the shape a
    stray artifact actually has.
    """

    def test_there_are_jobs_that_run_shipped_code(self) -> None:
        """AT-173 -- the vacuity guard. The count has moved before: the release workflow's own contract
        job was the third, and was added after the other two.
        """
        found = _jobs_running_shipped_code()
        assert len(found) >= 3, (
            f"the walk found {len(found)} jobs selecting {_CONTRACT_SELECTION!r} and there were three when "
            f"this was written; a matcher that resolves nothing reports a clean result"
        )

    def test_each_of_them_asks_whether_the_tree_is_still_clean(self) -> None:
        """AT-173 -- the gate itself."""
        offenders = [
            f"{file}:{name}"
            for file, name, job in _jobs_running_shipped_code()
            if not any(_TREE_QUERY in block for block in _run_blocks(job))
        ]
        assert offenders == [], (
            f"these jobs run a shipped example as a subprocess and never ask what it left behind: "
            f"{offenders}. Add a step running `{_TREE_QUERY}` and failing on any output"
        )

    def test_the_question_is_asked_after_the_tier_rather_than_before(self) -> None:
        """AT-173 -- a clean tree before the run is the state every job starts in and proves nothing."""
        for file, name, job in _jobs_running_shipped_code():
            steps = job.get("steps") or []
            ran = next(
                (index for index, step in enumerate(steps) if _CONTRACT_SELECTION in (step.get("run") or "")),
                None,
            )
            asked = next(
                (index for index, step in enumerate(steps) if _TREE_QUERY in (step.get("run") or "")),
                None,
            )
            assert ran is not None and asked is not None, f"{file}:{name} is missing one of the two steps"
            assert asked > ran, (
                f"{file}:{name} checks the working tree at step {asked} and runs the tier at step {ran}. "
                f"Asked first, it reports the state the runner handed the job"
            )
