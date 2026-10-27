"""Shared fixtures and tier gating.

Four test tiers, distinguished by the infrastructure they need. A tier whose backend is absent
**skips with the missing variable named**. It never silently passes, and it never fails for lack of
a database, because both of those teach the reader to ignore the result:

===============  =========================  ==============================================
Marker           Environment variable       Backend
===============  =========================  ==============================================
(none)           --                         none; mocked driver
``contract``     ``MONGODB_URI``            any real MongoDB, including a plain ``mongod``
``integration``  ``MONGODB_ATLAS_URI``      Atlas, or an Atlas CLI local deployment
``autoembed``    ``MONGODB_AUTOEMBED_URI``  a cluster with a registered embedding model
===============  =========================  ==============================================

Those three variables are read from the environment, and any the environment lacks are filled from a
``.env`` file at ``python/.env`` or the repository root. Nothing else in that file is read, an existing
value is never replaced, and an empty value means "no backend for this tier", which is how a run opts
out of a configured deployment without editing anything.

``autoembed`` exists as a separate gate rather than as part of ``integration`` because an Atlas CLI
local deployment runs vector and full-text search but registers no embedding models, so
``$vectorSearch`` with ``autoEmbed`` cannot be exercised there. Folding it into ``integration``
would report the package's default search mode as covered when it was never run.

**A ``contract`` test may not run a search query, and only review enforces this.** "Any real MongoDB"
is meant literally. On ``mongo:8`` (version 8.3.8, ``modules: []``), ``$search``, ``$vectorSearch``,
``$listSearchIndexes``, ``list_search_indexes()`` and ``create_search_index`` all return
``OperationFailure`` code 31082, while ordinary index creation and every CRUD path work normally.
``mongo:6``, ``mongo:7`` and an enterprise ``mongod`` 7.0.12 refuse the same five calls with a different
code, so the tier boundary is the same on all four. The contract tier therefore covers writes, ordinary
indexes and the dedup constraint, and anything needing mongot belongs in ``integration``. A test that
needs search but carries only the ``contract`` marker fails on a plain deployment, so point
``MONGODB_URI`` at one when adding a contract test.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import subprocess
import sys
import textwrap
import uuid
from collections.abc import AsyncIterator, Callable, Generator, Iterable, Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

if TYPE_CHECKING:
    from pymongo import AsyncMongoClient

# tests/ -> python/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"

# Marker -> the environment variable that supplies its backend. Order is widest-gate first so the
# skip message names the most permissive variable a test could be satisfied by.
TIER_VARIABLES: dict[str, str] = {
    "contract": "MONGODB_URI",
    "integration": "MONGODB_ATLAS_URI",
    "autoembed": "MONGODB_AUTOEMBED_URI",
}

# Where a developer's connection strings may already be written down. `python/.env` first, because
# the nearer file is the one someone working inside this directory would create, then the repository
# root. Both are `.gitignore`d, and `test_tier_gating` asserts that, because the file holds a connection
# string with a password in it.
ENV_FILE_LOCATIONS: tuple[Path, ...] = (_PROJECT_ROOT / ".env", _PROJECT_ROOT.parent / ".env")


def parse_env_file(text: str) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines, with no interpolation and no shell semantics.

    The value is everything after the *first* ``=``, because a connection string carries ``=`` in its
    query string and splitting on every one of them truncates the URI. One layer of matching quotes is
    stripped; nothing is unescaped, since ``%40`` in a generated Atlas password is part of the value the
    driver expects to receive.

    Args:
        text: Contents of a ``.env`` file.

    Returns:
        Every well-formed assignment in the file, later lines winning over earlier ones. Blank lines,
        ``#`` comments and lines with no ``=`` are skipped, which is what makes a commented-out URI
        stay commented out.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name.strip()] = value
    return values


def fill_tier_variables_from_env(
    environ: MutableMapping[str, str], locations: Iterable[Path] = ENV_FILE_LOCATIONS
) -> list[str]:
    """Fill the three tier variables from the first ``.env`` that exists, without overriding anything.

    Only ``TIER_VARIABLES``' own names are read. Every other key in the file is ignored, which is the
    difference between reading configuration and importing an environment: a ``.env`` written for an
    application can carry ``PATH`` or a cloud credential, and a harness that assimilates all of it
    changes how the interpreter resolves programs for the rest of the session.

    A key already **present** in ``environ`` is left alone, and presence rather than truthiness is the
    test on purpose. The gate reads an empty value as unset, so ``MONGODB_URI= pytest`` stays the way to
    run a tier unbacked on a machine whose ``.env`` configures it.

    Args:
        environ: Mapping to fill, ``os.environ`` in the one call that matters.
        locations: Candidate paths, in preference order. The first that exists is the only one read,
            since two files are a mistake to notice rather than a merge to perform.

    Returns:
        The variables this call set, in file order. Empty when no file exists, which is the normal case
        in CI and not a condition worth reporting.
    """
    wanted = set(TIER_VARIABLES.values())
    for path in locations:
        if not path.is_file():
            continue
        filled = []
        for name, value in parse_env_file(path.read_text()).items():
            if name in wanted and name not in environ:
                environ[name] = value
                filled.append(name)
        return filled
    return []


# At import time, so the collection hook and every fixture below see the same environment. Three keys of
# hand-written parsing rather than a `python-dotenv` dependency; `fill_tier_variables_from_env` says why.
fill_tier_variables_from_env(os.environ)

# One database per test session, dropped on teardown. Concurrent sessions (a local run beside CI,
# or pytest-xdist workers) must not share a namespace, and a fixed name makes that collision
# invisible: the second session sees the first session's documents and passes for the wrong reason.
_SESSION_SUFFIX = uuid.uuid4().hex[:8]


def pytest_configure(config: pytest.Config) -> None:
    """Register the contract checklist gate, and make the subprocess tests count toward coverage.

    Several tests assert on import-time behavior, which is only observable in a fresh interpreter,
    so they run their assertions in a child process. Nothing a child executes is measured unless
    ``coverage`` starts there too, and the ``.pth`` hook that starts it does nothing without
    ``COVERAGE_PROCESS_START``. pytest-cov 7.1 does not set that variable, so the parent sets it here
    for its children to inherit.

    Left unset, the missing-extra branches in ``storage.py`` and ``memory_store.py`` and the
    ``__dir__`` in ``__init__.py`` read as uncovered while being fully tested, and the per-module
    floor in ``scripts/check_module_coverage.py`` would fail on them. The likely response to that is
    an exemption entry, which would hide the real gap the floor exists to find.

    Only set when coverage is actually collecting. Otherwise a plain ``pytest`` run would leave
    ``.coverage.*`` data files behind from every subprocess it starts.

    The gate is registered as a plugin instance rather than written as module-level hooks; see
    :class:`ChecklistGate` for why.

    Args:
        config: The active pytest configuration.
    """
    if getattr(config.option, "cov_source", None):
        os.environ["COVERAGE_PROCESS_START"] = str(_PROJECT_ROOT / "pyproject.toml")
    config.pluginmanager.register(ChecklistGate(), ChecklistGate.name)


# Acceptance-test ids as this suite writes them: ``AT-``, up to three digits and an optional letter
# suffix, plus two other prefixes the pattern also accepts. ``AT-GATE`` deliberately does not match,
# so the gate cannot end up listed as its own evidence.
_AT_ID = re.compile(r"\b((?:AT|CH|MP)-[0-9]{1,3}[a-z]*)")


class RunOutcome(NamedTuple):
    """What one test did in this session.

    Attributes:
        outcome: ``passed``, ``failed`` or ``skipped``, as pytest reported it.
        detail: The skip reason or the phase that failed, so the gate's message can carry it.
    """

    outcome: str
    detail: str


def _declared_ids(item: pytest.Item) -> tuple[set[str], set[str]]:
    """Return the acceptance-test ids declared on a collected test, and those on its class.

    Both are load-bearing in this suite. Some ids sit on a test, some on the class holding a group of
    them, and a few on both.

    Only each docstring's **first paragraph** is read, and that is not a stylistic preference. This suite
    declares coverage in the summary line, then discusses other tiers in the body: the two unit classes
    in ``test_memory_store.py`` covering the metadata merge both name the search-result checklist id
    several paragraphs down, to say that the assertion lives on the integration tier and not in them.
    Reading whole docstrings would turn that sentence into a coverage claim, and the gate would then
    report the item as covered, and pass, in a run where the integration tier was deselected.

    Module docstrings are not read at all, for the same reason one step further out.

    Reading collected items rather than parsing the files is what keeps a fake's ids out of the index.
    ``FakeAggregationCursor`` carries an id in its docstring and is never collected, so it cannot
    satisfy a checklist item on its own.

    Args:
        item: A collected test.

    Returns:
        A pair of id sets, the test's own first and its class's second. Either may be empty.
    """
    owner = getattr(item, "cls", None)
    function = getattr(item, "obj", None)
    own = set(_AT_ID.findall(_summary(getattr(function, "__doc__", ""))))
    inherited = set(_AT_ID.findall(_summary(getattr(owner, "__doc__", ""))))
    return own, inherited


def _summary(docstring: str | None) -> str:
    """Return a docstring's first paragraph, where this suite declares what a test covers.

    Args:
        docstring: A docstring, or ``None`` for a test or class that has none.

    Returns:
        Everything up to the first blank line, or an empty string.
    """
    return (docstring or "").split("\n\n", 1)[0]


class ChecklistGate:
    """Collects the two things the contract checklist's release gate needs and a test cannot see by itself.

    A test can assert what it observes. It cannot observe which other tests were collected, nor what
    they did, and the gate's whole claim is about the run around it: that every checklist item has a
    test, and that the release-gate subset ran and passed rather than quietly skipping. Both facts live
    in pytest hooks, so they are gathered here and read through fixtures.

    Registered from :func:`pytest_configure` rather than by defining the hooks at module level, for two
    reasons. ``pytest_collection_modifyitems`` may only be defined once per module and this conftest
    already uses it for tier gating, and ``test_tier_gating.py`` re-exports that gating hook into a
    nested session, which must not inherit the gate's reordering.
    """

    name = "strands_mongodb_checklist_gate"

    def __init__(self) -> None:
        self.by_id: dict[str, tuple[str, ...]] = {}
        self.outcomes: dict[str, RunOutcome] = {}

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        """Index the collected tests by acceptance-test id, then force the gate to run last.

        An id declared by a test wins over the same id on any class, anywhere. A class docstring only
        carries the ids no test declares for itself, which is how a class covering one behaviour across
        several tests is meant to read. Without that precedence, a failure in the contract ``read`` test
        would be blamed on the checklist items about ``write`` and ``list`` ordering too, because the
        contract class in ``test_storage.py`` declares all three ids in one summary line.

        ``trylast`` is what makes the reordering survive a plugin that shuffles collection order.
        Nothing in the current pinned set does, but a gate run before the tests it measures would report
        every one of them as never having run.
        """
        own: dict[str, list[str]] = {}
        inherited: dict[str, list[str]] = {}
        for item in items:
            declared, from_class = _declared_ids(item)
            for identifier in declared:
                own.setdefault(identifier, []).append(item.nodeid)
            for identifier in from_class:
                inherited.setdefault(identifier, []).append(item.nodeid)
        self.by_id = {
            identifier: tuple(own.get(identifier) or inherited.get(identifier) or ())
            for identifier in set(own) | set(inherited)
        }

        gate = [item for item in items if item.get_closest_marker("gate") is not None]
        if gate:
            items[:] = [item for item in items if item.get_closest_marker("gate") is None] + gate

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record one outcome per node id, across the three phases of a test.

        A test that skips in setup never reaches its call phase, and a test that passes its call phase
        can still fail in teardown, so neither phase alone is the result. A recorded failure is never
        overwritten by a later phase's success.
        """
        if report.when == "setup" and report.passed:
            return
        if report.when == "teardown" and not report.failed:
            return
        if self.outcomes.get(report.nodeid, RunOutcome("", "")).outcome == "failed":
            return
        if report.skipped and isinstance(report.longrepr, tuple):
            detail = str(report.longrepr[2])
        elif report.failed:
            detail = f"{report.when} failed"
        else:
            detail = ""
        self.outcomes[report.nodeid] = RunOutcome(report.outcome, detail)


@pytest.fixture(scope="session")
def checklist_gate(request: pytest.FixtureRequest) -> ChecklistGate:
    """The registered :class:`ChecklistGate`, holding this run's id index and outcomes.

    Returns:
        The plugin instance :func:`pytest_configure` registered.

    Raises:
        pytest.fail.Exception: If the plugin is not registered, which means the gate is measuring a
            session it was never installed in, and a green result would mean nothing.
    """
    plugin = request.config.pluginmanager.get_plugin(ChecklistGate.name)
    if not isinstance(plugin, ChecklistGate):
        pytest.fail(f"the {ChecklistGate.name} plugin is not registered; the checklist gate cannot see this run")
    return plugin


def _run_mypy(
    source: str,
    tmp_path: Path,
    *,
    env: dict[str, str],
    cwd: Path,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Write a snippet to ``tmp_path`` and type check it in a subprocess.

    Args:
        source: Python source. Dedented, so a triple-quoted snippet can stay indented at its call
            site.
        tmp_path: Directory to hold the snippet and mypy's cache.
        env: Full environment for the subprocess.
        cwd: Working directory, which determines the config file mypy discovers.
        extra_args: Additional mypy arguments, appended before the snippet path.

    Returns:
        The completed process, with mypy's notes and errors on ``stdout``.
    """
    snippet = tmp_path / "snippet.py"
    snippet.write_text(textwrap.dedent(source))
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", f"--cache-dir={tmp_path / '.mypy_cache'}"]
        + list(extra_args)
        + [str(snippet)],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )


@pytest.fixture
def run_mypy(tmp_path: Path) -> Callable[[str], subprocess.CompletedProcess[str]]:
    """Type check a source snippet against this working tree.

    Several requirements are promises to a type checker rather than to the interpreter: a
    ``Required`` key, a narrow ``Literal``, the class a lazily exported name resolves to. None of
    them can be asserted at runtime. This runs the checker the project ships with and lets the test
    read its output.

    ``MYPYPATH`` names ``src`` rather than relying on the editable install, so the result describes
    the code in this tree and not whichever build the venv last resolved. The subprocess runs in the
    project root, so mypy finds ``[tool.mypy]`` in ``pyproject.toml`` and checks at the declared 3.10
    floor. An explicit file argument overrides the ``files`` list there.

    Use :func:`run_mypy_as_installed` instead for anything a **consumer** of the built package would
    see. Putting ``src`` on the path is what lets mypy analyze this package at all, so this fixture
    cannot observe whether the distribution advertises its types.

    Returns:
        A callable taking Python source and returning the completed mypy process, with its notes and
        errors on ``stdout``.
    """

    def run(source: str) -> subprocess.CompletedProcess[str]:
        return _run_mypy(
            source,
            tmp_path,
            env={**os.environ, "MYPYPATH": str(_SRC)},
            cwd=_PROJECT_ROOT,
        )

    return run


@pytest.fixture
def run_mypy_as_installed(tmp_path: Path) -> Callable[[str], subprocess.CompletedProcess[str]]:
    """Type check a snippet the way a consumer's own project would.

    Three differences from :func:`run_mypy`, each removing something a consumer's project would not have:

    * No ``MYPYPATH``. The package has to be found through the install, as a dependency, not as a
      directory of source files that happens to be on the path.
    * A generated config rather than ours, with ``ignore_missing_imports`` off. Our own setting is
      ``true``, which is right for third-party libraries without stubs and wrong here: it turns "I
      could not read this package's types" into silence, and every symbol from it into ``Any``.
    * ``tmp_path`` as the working directory, so no config or source tree of ours is discovered by
      proximity.

    That combination is what makes the PEP 561 marker observable. Without ``py.typed`` mypy finds the
    package, declines to read it, and reports ``module is installed, but missing library stubs or
    py.typed marker``.

    Returns:
        A callable taking Python source and returning the completed mypy process.
    """
    config = tmp_path / "mypy.ini"
    config.write_text("[mypy]\npython_version = 3.10\nignore_missing_imports = false\n")
    env = {key: value for key, value in os.environ.items() if key != "MYPYPATH"}

    def run(source: str) -> subprocess.CompletedProcess[str]:
        return _run_mypy(
            source,
            tmp_path,
            env=env,
            cwd=tmp_path,
            extra_args=("--config-file", str(config)),
        )

    return run


@pytest.fixture(scope="session")
def test_database_name() -> str:
    """Name of this session's throwaway database.

    Returns:
        A name unique to this session, of the form ``strands_mongodb_test_<8 hex chars>``.
    """
    return f"strands_mongodb_test_{_SESSION_SUFFIX}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None]:
    """Turn an unreachable search process into a skip when it surfaces while a fixture is building.

    See :func:`pytest_runtest_call` for why both phases are wrapped and why this is a hook rather than
    another call site.

    Args:
        item: The test being set up. Unused: what is inspected is the failure, not the test.

    Yields:
        Control to the rest of the setup phase.
    """
    try:
        return (yield)
    except BaseException as error:
        # The same two lines the two search helpers hold, for the same reason: the guard returns for
        # anything that is not the outage, and then this re-raises it untouched.
        skip_if_search_is_unreachable(error)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None]:
    """Turn an unreachable search process into a skip wherever in the test body it surfaces.

    The two search helpers also call :func:`skip_if_search_is_unreachable`, but that alone misses the
    usual shape of the outage: ``add`` and ``initialize`` provision the search indexes, so on a
    deployment whose mongot is gone the failure arrives on the *arrange* line, before any search runs.

    Adding the guard to every arrange line would mean touching every cluster test and would still miss
    the next one. A hook sees the failure wherever it stands, and the match it delegates to stays strict
    (two codes, one of them qualified by a sentence this package cannot produce), so nothing that could
    be a package regression is affected.

    **Teardown is deliberately not wrapped.** By then the test has its result, and converting a
    teardown failure into a skip would discard an answer the suite already has.

    Args:
        item: The test being run. Unused: what is inspected is the failure, not the test.

    Yields:
        Control to the test body.
    """
    try:
        return (yield)
    except BaseException as error:
        # The same two lines the two search helpers hold, for the same reason: the guard returns for
        # anything that is not the outage, and then this re-raises it untouched.
        skip_if_search_is_unreachable(error)
        raise


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test whose tier has no configured backend as skipped, naming the variable.

    A collection hook rather than an autouse fixture, and the difference is not stylistic. An
    autouse function-scoped fixture runs *after* any session-scoped fixture the test requests, so
    ``contract_uri`` would be instantiated first and raise before the gate ever ran. The result would
    be an **error**, not a skip: a missing backend reported as a broken suite.

    Collection-time gating has no such ordering problem, because nothing has been set up yet.

    A test carrying two tier markers needs both variables. Markers describe requirements, not
    alternatives.
    """
    for item in items:
        for marker, variable in TIER_VARIABLES.items():
            if item.get_closest_marker(marker) is None:
                continue
            if not os.environ.get(variable):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{marker} tier needs {variable} to be set to a deployment this test may write to"
                    )
                )
                break


_CREDENTIAL = re.compile(r"://[^@/]+@")
"""The userinfo component of a connection string, which is what must not be printed."""


class RedactedURI(str):
    """A connection string whose ``repr`` hides its password.

    A ``str`` subclass rather than a wrapper, because everything downstream is entitled to treat this as
    the string it is: pymongo builds a client from it, tests compare it, f-strings interpolate it.
    Only ``repr`` changes, and ``repr`` is what reaches the log: pytest prints every fixture a failing
    test requested using ``repr``, so without this the failure header of any test holding a URI would
    carry the password into the run log.

    ``__str__`` deliberately keeps the credential, because the client needs it to connect.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the value with its userinfo replaced, quoted as a ``repr`` is expected to be."""
        return repr(_CREDENTIAL.sub("://<redacted>@", str(self)))


def _require(variable: str) -> RedactedURI:
    """Read a gate variable that ``_tier_gate`` has already proven present.

    Args:
        variable: Name of the environment variable to read.

    Returns:
        The variable's value, wrapped so that no report prints its password.

    Raises:
        RuntimeError: If the variable is unset, which means a fixture was requested by a test
            carrying no matching tier marker. That is a wiring bug in the test, not a missing
            backend, so it must fail rather than skip.
    """
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"{variable} is unset but a fixture requiring it was requested. Add the matching tier "
            f"marker to the test so it is gated instead of failing here."
        )
    return RedactedURI(value)


@pytest.fixture(scope="session")
def contract_uri() -> str:
    """Connection string for a real MongoDB deployment of any kind."""
    return _require("MONGODB_URI")


@pytest.fixture(scope="session")
def atlas_uri() -> str:
    """Connection string for a deployment with Atlas Search available."""
    return _require("MONGODB_ATLAS_URI")


@pytest.fixture(scope="session")
def autoembed_uri() -> str:
    """Connection string for a cluster with a registered Atlas embedding model."""
    return _require("MONGODB_AUTOEMBED_URI")


HOST_UNREACHABLE: Final[int] = 6
"""``HostUnreachable``, the server error code that says a process in the deployment did not answer.

On an Atlas CLI local deployment with mongot stopped, ``$vectorSearch`` raises
``OperationFailure('Error connecting to localhost:28000 (127.0.0.1:28000) :: caused by ::
Connection refused', 6)``. The 28000 is mongot, which the CLI runs as a separate process beside
``mongod``, so the database answers every other call and only search fails.
"""

COMMAND_FAILED: Final[int] = 125
"""``CommandFailed``, which is what the *other* channel to the same process answers with.

On ``mongod`` 8.0 with nothing listening on its ``mongotHost`` port, one outage gives two answers:
``$search`` raises code 6 naming the host and port that refused, while ``$listSearchIndexes`` and
``createSearchIndexes`` raise ``OperationFailure('... :: caused by :: Error connecting to Search Index
Management service.', 125)``, with no host, no port, and a code that is general rather than a network
answer. mongod reaches mongot over two channels and reports them
differently, so a guard that knows one code knows half of the outage.
"""

SEARCH_MANAGEMENT_UNREACHABLE: Final[str] = "Error connecting to Search Index Management service"
"""The sentence that makes :data:`COMMAND_FAILED` mean the outage rather than any failed command.

125 on its own is far too wide to skip on: it is the server's general "that command did not work out",
and a suite that skipped on it would hide results it exists to report. This sentence is the part no
change to this package can produce: the host it stands for comes from a server parameter, and the
package neither sets it nor knows it. The tests of this guard assert both halves of the pairing.
"""


def _chain(error: BaseException) -> list[BaseException]:
    """Return ``error`` and every cause under it, outermost first.

    Walked to the end rather than read two links deep, because the ``auto`` path's "no search tier
    could be prepared" error has a cause of its own: this package's error wrapping this package's error
    wrapping the driver's, and the code is on the last of the three.

    Args:
        error: The outermost exception.

    Returns:
        The chain, outermost first. Bounded, and self-referencing chains terminate rather than spin.
    """
    chain: list[BaseException] = []
    link: BaseException | None = error
    while link is not None and not any(link is seen for seen in chain):
        chain.append(link)
        link = link.__cause__
    return chain


def _is_search_unreachable(error: BaseException) -> bool:
    """Whether one link says the deployment's search process did not answer.

    Two conditions, because one outage has two reports (:data:`COMMAND_FAILED`). The first is a code on
    its own, since 6 is a network answer and nothing this package does can produce one. The second is a
    code *and* a sentence, since 125 alone is the server's general "that command failed".

    Args:
        error: One link of the chain.

    Returns:
        ``True`` if this link is the outage, ``False`` for anything else, including every link with no
        code at all.
    """
    code = getattr(error, "code", None)
    if code == HOST_UNREACHABLE:
        return True
    return code == COMMAND_FAILED and SEARCH_MANAGEMENT_UNREACHABLE in str(error)


def skip_if_search_is_unreachable(error: BaseException) -> None:
    """Skip the running test if ``error`` is the deployment's search process refusing connections.

    With mongot down, every search-shaped test fails and the report reads as though this package had
    broken searching. A failure the operator has to fix is not one the suite should assert on, and a
    skip that names the process tells the operator to restart mongot.

    **Two reports of one outage.** Code 6 is what the query channel answers with. Every call that
    provisions or reads an index goes over mongod's second channel to the same process and comes back
    as 125 with a sentence instead, so a guard on code 6 alone misses them. Both conditions live in
    :func:`_is_search_unreachable`.

    **Still deliberately narrow.** 6 matches on its own because it is a network answer this package
    cannot provoke; 125 matches only beside :data:`SEARCH_MANAGEMENT_UNREACHABLE`, because on its own
    it is the server's general "that command failed" and skipping on it would hide results the suite
    exists to report. And it stays called from the search helpers, so a write that fails the same way
    still fails.

    The whole chain is read rather than the first two links, because the callers hold three different
    depths of the same failure: ``test_pipeline.py`` drives the driver and catches the
    ``OperationFailure`` itself, ``test_memory_store.py`` catches this package's ``StorageError`` with
    the driver's chained, and the ``auto`` path adds a third link on top. One walk
    reads all three, and the skip message is built from the links that say something new, so the
    namespace and the host that refused both survive however deep they sit.

    Args:
        error: The exception a search call raised. Anything else returns, so the caller re-raises.

    Raises:
        Skipped: via ``pytest.skip``, when any link in the chain is one of the two reports.
    """
    if not any(_is_search_unreachable(link) for link in _chain(error)):
        return
    # Both links, rather than whichever one was caught. This package's error names the namespace and
    # the operation; the driver's names the host and port that refused, which is the process to go and
    # restart. Building the message from the chain rather than trusting the wrapper to have embedded its
    # cause is what keeps both in the message for every caller.
    chain: list[str] = []
    for link in _chain(error):
        text = str(link)
        if not any(text in seen for seen in chain):
            chain.append(text)
    pytest.skip(
        f"this deployment's search process is unreachable, so the failure is infrastructure rather "
        f"than a package regression and asserting on it would report the wrong thing: "
        f"{' :: caused by :: '.join(chain)}"
    )


async def _drop_database(uri: str, database: str) -> None:
    """Drop a database, including any search indexes on its collections.

    Args:
        uri: Connection string for the deployment holding the database.
        database: Name of the database to drop.
    """
    from pymongo import AsyncMongoClient

    client: AsyncMongoClient[Any] = AsyncMongoClient(uri)
    try:
        db = client[database]
        for name in await db.list_collection_names():
            # A dropped collection can leave a search index building on mongot, which then holds a
            # name we may want to reuse. Drop the indexes by name first, and tolerate a deployment
            # with no search support, where list_search_indexes is not available at all.
            try:
                async for index in await db[name].list_search_indexes():
                    await db[name].drop_search_index(index["name"])
            except Exception:  # noqa: BLE001 - teardown must not mask the test's own result
                pass
        await client.drop_database(database)
    finally:
        await client.close()


@pytest.fixture(scope="session", autouse=True)
def _drop_test_databases(test_database_name: str) -> Iterator[None]:
    """Drop this session's database from every configured deployment, on the way out.

    Session-scoped and synchronous on purpose. It runs ``asyncio.run`` in a fresh loop rather than
    holding a client across the session, because sharing one client across every test's loop is a
    pattern the package itself handles, and the harness must not depend on the behavior under test.
    """
    yield
    for variable in ("MONGODB_URI", "MONGODB_ATLAS_URI", "MONGODB_AUTOEMBED_URI"):
        uri = os.environ.get(variable)
        if not uri:
            continue
        try:
            asyncio.run(_drop_database(uri, test_database_name))
        except Exception as error:  # noqa: BLE001 - report, never fail the session on cleanup
            print(f"warning: could not drop {test_database_name} on {variable}: {error!r}")


@pytest.fixture
async def contract_database(contract_uri: str, test_database_name: str) -> AsyncIterator[Any]:
    """An ``AsyncDatabase`` on a real deployment, with a per-test client.

    A fresh client per test keeps each test on its own event loop and makes cross-test pollution
    through a shared pool impossible.
    """
    from pymongo import AsyncMongoClient

    client: AsyncMongoClient[Any] = AsyncMongoClient(contract_uri)
    try:
        yield client[test_database_name]
    finally:
        await client.close()


@pytest.fixture
async def atlas_database(atlas_uri: str, test_database_name: str) -> AsyncIterator[Any]:
    """An ``AsyncDatabase`` on a deployment with Atlas Search available."""
    from pymongo import AsyncMongoClient

    client: AsyncMongoClient[Any] = AsyncMongoClient(atlas_uri)
    try:
        yield client[test_database_name]
    finally:
        await client.close()


@pytest.fixture
def unique_collection_name(request: pytest.FixtureRequest) -> str:
    """A collection name unique to the requesting test.

    Derived from the test's own name so a failure points at its data, with a random suffix so a
    reran test never inherits residue from the run before it.

    Returns:
        A collection name safe to create and drop.
    """
    stem = request.node.name.replace("[", "_").replace("]", "").replace("-", "_")
    return f"{stem}_{uuid.uuid4().hex[:6]}"[:100]


def rendered(span: ReadableSpan) -> str:
    """Return every string a span carries, joined, ready for a substring scan.

    Written from the ``ReadableSpan`` surface rather than from the setters, so a value reaches the scan the
    same way whatever route put it there. In opentelemetry-sdk 1.44.0, ``attributes`` and an
    event's ``attributes`` are both ``None`` on a span that never set any, and ``status.description`` is
    ``None`` until a status carries one.

    ``resource.attributes`` is deliberately not read. Those come from the host's ``Resource``, which this
    package never touches, and they carry a ``service.instance.id`` that changes per process.

    Args:
        span: A finished span, as the exporter received it.

    Returns:
        One string holding the span's name, its status description, its instrumentation scope, and the
        keys and values of its own attributes, of every event, and of every link.
    """
    parts = [span.name, str(span.status.description or "")]
    parts += [f"{key}={value!r}" for key, value in (span.attributes or {}).items()]
    for event in span.events:
        parts.append(event.name)
        parts += [f"{key}={value!r}" for key, value in (event.attributes or {}).items()]
    for link in span.links:
        parts += [f"{key}={value!r}" for key, value in (link.attributes or {}).items()]
    if span.instrumentation_scope is not None:
        parts.append(span.instrumentation_scope.name)
    return "\n".join(parts)


class _ForwardingProcessor(SpanProcessor):
    """The one processor this harness adds to the global provider, sending each span to whichever test is reading.

    One, not one per test. A processor cannot be removed from a provider once added, so a fresh
    ``SimpleSpanProcessor`` per test would leave every earlier one in place, and by the end of the suite
    each span would be offered to hundreds of shut-down exporters.
    """

    def __init__(self) -> None:
        self.exporter: InMemorySpanExporter | None = None

    def on_end(self, span: ReadableSpan) -> None:
        if self.exporter is not None:
            self.exporter.export((span,))


_FORWARDER: Final = _ForwardingProcessor()
_INSTALLED = False


@contextlib.contextmanager
def capturing_spans() -> Iterator[InMemorySpanExporter]:
    """Point the harness's processor at a fresh in-memory exporter for the body of the ``with``.

    The processor goes onto the *global* tracer provider, because that is the only provider package code
    would ever reach: ``trace.get_tracer(__name__)`` and ``trace.get_current_span()`` both read it, so a
    provider kept local to this test would leave a leak writing somewhere nobody was watching.

    Two paths, because both states occur. A fresh interpreter has a ``ProxyTracerProvider`` and takes
    ``set_tracer_provider``; a suite that already configured tracing has a real one and takes the
    processor. As of 1.44.0, importing ``strands`` does not install a provider, so the first path is the
    one the unit tier takes. Either way it happens once per process.
    """
    global _INSTALLED
    if not _INSTALLED:
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.add_span_processor(_FORWARDER)
        else:
            installed = TracerProvider()
            installed.add_span_processor(_FORWARDER)
            trace.set_tracer_provider(installed)
        _INSTALLED = True
    exporter = InMemorySpanExporter()
    _FORWARDER.exporter = exporter
    try:
        yield exporter
    finally:
        _FORWARDER.exporter = None


@pytest.fixture
def exported_spans() -> Iterator[InMemorySpanExporter]:
    """An in-memory exporter receiving every span the package records during the test. See ``capturing_spans``."""
    with capturing_spans() as exporter:
        yield exporter
