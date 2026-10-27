"""Tests for the test harness itself: does the tier gate actually gate?

``conftest.py``'s ``pytest_collection_modifyitems`` hook is the single mechanism keeping an
unconfigured backend from being reported as a pass. It can fail in both directions: broken open,
every gated test skips while the suite still reports green; broken the other way, a missing database
looks like a broken test suite.

So the gate gets its own tests. ``pytester`` runs a nested pytest session with a synthetic test
file, which is the only way to observe skip behavior from the outside, since a test cannot watch
itself be skipped. The ``.env`` loader, which decides what the gate sees in the first place, is
tested here too.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    ENV_FILE_LOCATIONS,
    HOST_UNREACHABLE,
    TIER_VARIABLES,
    _require,
    fill_tier_variables_from_env,
    parse_env_file,
    skip_if_search_is_unreachable,
)

from strands_mongodb import _indexes

pytest_plugins = ["pytester"]

EXPECT_SEARCH_VARIABLE = "STRANDS_MONGODB_EXPECT_SEARCH"
"""How a caller says "the deployment I gave you has a search process, so hold me to it".

Set by one leg of the contract tier's two and left empty by the other. One-directional on purpose: set
means the search process must be there, and unset means nothing, because a developer running this tier
against a cluster has not promised anything and should not be failed for exceeding the promise.
"""

# How a test says "this machine has no backend for that tier". Deleting the variable is not enough
# because the real conftest reads `.env`: every nested session re-imports that module, which re-fills any
# tier variable the file names and the environment lacks, so a `delenv` here would be undone before the
# gate ran. An empty value is present, therefore never filled, and the gate reads it as unset because it
# tests truthiness. That keeps these tests hermetic whether or not the checkout has a `.env`.
_NO_BACKEND = ""

# The nested session must exercise the real conftest, not a copy that could drift from it. It is
# loaded under a distinct module name because the nested file is itself named `conftest`, so a plain
# `from conftest import ...` resolves to the partially-initialized importer and fails circularly.
#
# Only the gating fixtures are re-exported. `_drop_test_databases` is deliberately left out: it is
# session-scoped autouse and would try to connect to whatever URI a test sets, blocking for the
# 30-second server-selection timeout against an address that need not exist.
_LOAD_REAL_CONFTEST = """
import importlib.util

_spec = importlib.util.spec_from_file_location("strands_mongodb_conftest_under_test", {conftest!r})
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)

TIER_VARIABLES = _real.TIER_VARIABLES
pytest_collection_modifyitems = _real.pytest_collection_modifyitems
test_database_name = _real.test_database_name
contract_uri = _real.contract_uri
atlas_uri = _real.atlas_uri
autoembed_uri = _real.autoembed_uri
unique_collection_name = _real.unique_collection_name
"""


@pytest.fixture
def gated(pytester: pytest.Pytester, request: pytest.FixtureRequest) -> pytest.Pytester:
    """A nested pytest project that reuses this suite's real ``conftest.py``."""
    conftest = str(request.config.rootpath / "tests" / "conftest.py")
    pytester.makeconftest(_LOAD_REAL_CONFTEST.format(conftest=conftest))
    pytester.makeini(
        """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        markers =
            contract: needs a real MongoDB deployment of any kind.
            integration: needs Atlas or an Atlas CLI local deployment with mongot.
            autoembed: needs a cluster with a registered Atlas embedding model.
        """
    )
    return pytester


class TestGateSkipsWhenUnconfigured:
    """A tier with no backend skips, and the reason names the variable to set."""

    @pytest.mark.parametrize(
        ("marker", "variable"),
        [
            ("contract", "MONGODB_URI"),
            ("integration", "MONGODB_ATLAS_URI"),
            ("autoembed", "MONGODB_AUTOEMBED_URI"),
        ],
    )
    def test_skips_naming_the_missing_variable(
        self,
        gated: pytest.Pytester,
        monkeypatch: pytest.MonkeyPatch,
        marker: str,
        variable: str,
    ) -> None:
        """The skip reason is actionable, not just present.

        ``pytest.skip("no database")`` would satisfy a "does it skip" test while telling the reader
        nothing. Asserting the variable name is in the reason is what makes the report useful.
        """
        monkeypatch.setenv(variable, _NO_BACKEND)
        gated.makepyfile(
            f"""
            import pytest

            @pytest.mark.{marker}
            def test_needs_a_backend():
                raise AssertionError("this body must never run")
            """
        )

        result = gated.runpytest("-rs")

        result.assert_outcomes(skipped=1, passed=0, failed=0)
        result.stdout.fnmatch_lines([f"*{variable}*"])

    def test_skips_rather_than_errors_when_the_test_requests_a_session_scoped_fixture(
        self, gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session-scoped backend fixture does not turn the skip into an error.

        Pytest instantiates higher-scoped fixtures first, so a gate implemented as a function-scoped
        autouse fixture would let the session-scoped ``contract_uri`` be built, and raise, before the
        gate ran. The outcome would be **errors**, not skips.

        That distinction is the point of the gate. An error says the suite is broken and someone
        should fix the tests; a skip says the environment lacks a backend and names the variable to
        set.
        """
        monkeypatch.setenv("MONGODB_URI", _NO_BACKEND)
        gated.makepyfile(
            """
            import pytest

            @pytest.mark.contract
            def test_wants_a_database(contract_uri):
                raise AssertionError("this body must never run")
            """
        )

        result = gated.runpytest("-rs")

        result.assert_outcomes(skipped=1, errors=0, failed=0, passed=0)
        result.stdout.fnmatch_lines(["*MONGODB_URI*"])

    def test_class_level_marker_gates_every_method(
        self, gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Markers applied to a class reach its methods.

        The harness smoke tests mark the class, not each method. If ``get_closest_marker`` did not
        walk up to the class, every one of them would run ungated.
        """
        monkeypatch.setenv("MONGODB_URI", _NO_BACKEND)
        gated.makepyfile(
            """
            import pytest

            @pytest.mark.contract
            class TestNeedsABackend:
                def test_one(self):
                    raise AssertionError("must never run")

                def test_two(self):
                    raise AssertionError("must never run")
            """
        )

        result = gated.runpytest()

        result.assert_outcomes(skipped=2, passed=0, errors=0)


class TestGateAllowsWhenConfigured:
    """A tier with its backend configured runs."""

    def test_runs_when_the_variable_is_set(self, gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
        """The complement of the skip test.

        Without this, a gate that skips unconditionally passes every skip assertion above. Both
        directions are needed or the gate is only half tested.
        """
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
        gated.makepyfile(
            """
            import pytest

            @pytest.mark.contract
            def test_needs_a_backend():
                assert True
            """
        )

        result = gated.runpytest()

        result.assert_outcomes(passed=1, skipped=0, failed=0)

    def test_unmarked_test_always_runs(self, gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
        """The unit tier is ungated, so it runs with every backend variable absent."""
        for variable in ("MONGODB_URI", "MONGODB_ATLAS_URI", "MONGODB_AUTOEMBED_URI"):
            monkeypatch.setenv(variable, _NO_BACKEND)
        gated.makepyfile(
            """
            def test_pure_unit():
                assert True
            """
        )

        result = gated.runpytest()

        result.assert_outcomes(passed=1, skipped=0)


class TestMarkersAreRequirementsNotAlternatives:
    """Two markers mean two backends, not either one."""

    def test_a_test_with_two_markers_needs_both_variables(
        self, gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``autoembed`` test also carries ``integration``, and one variable must not satisfy both.

        Setting only ``MONGODB_ATLAS_URI`` must not run the AutoEmbeddings acceptance test against a
        deployment with no registered model, where ``$vectorSearch`` returns an empty result rather
        than an error and the test would report a pass.
        """
        monkeypatch.setenv("MONGODB_ATLAS_URI", "mongodb://localhost:27017")
        monkeypatch.setenv("MONGODB_AUTOEMBED_URI", _NO_BACKEND)
        gated.makepyfile(
            """
            import pytest

            @pytest.mark.integration
            @pytest.mark.autoembed
            def test_needs_both():
                raise AssertionError("this body must never run")
            """
        )

        result = gated.runpytest("-rs")

        result.assert_outcomes(skipped=1, passed=0)
        result.stdout.fnmatch_lines(["*MONGODB_AUTOEMBED_URI*"])


class TestFixtureMisuseFailsRatherThanSkips:
    """Requesting a backend fixture without the matching marker is a wiring bug."""

    def test_unmarked_test_requesting_a_gated_fixture_fails(
        self, gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It must fail loudly, not skip.

        A skip here would hide a test that intended to hit a real database and was silently never
        running. The distinction matters: a missing backend is the environment's fault and should
        skip, a missing marker is the test's fault and should fail.
        """
        monkeypatch.setenv("MONGODB_URI", _NO_BACKEND)
        gated.makepyfile(
            """
            def test_forgot_the_marker(contract_uri):
                assert contract_uri
            """
        )

        result = gated.runpytest()

        result.assert_outcomes(errors=1, skipped=0, passed=0)
        result.stdout.fnmatch_lines(["*MONGODB_URI is unset*"])


class TestAnUnreachableSearchProcessSkipsRatherThanFails:
    """AT-127 — the second gate, which reports infrastructure as infrastructure.

    The gate above decides whether a tier runs at all, from a variable read before collection. This one
    covers what the variable cannot see: a deployment that is configured and reachable for everything
    except the separate process that answers searches. Without it, an unreachable search process would
    be reported as failing tests in this package with nothing wrong with them.

    Every claim here is about the boundary between "skip this" and "fail this". A guard that skipped on
    any search failure would hide the regressions the tests exist to find.
    """

    @staticmethod
    def _storage_error(code: int | None) -> Exception:
        """Build the failure shape ``test_memory_store.py`` catches: this package's error, driver chained."""
        from pymongo.errors import OperationFailure
        from strands.types.exceptions import StorageError

        cause = OperationFailure("Error connecting to localhost:28000 :: Connection refused", code=code)
        error = StorageError("strands_mongodb: searching 'q' in testdb.memory failed: connection refused")
        error.__cause__ = cause
        return error

    def test_the_documented_code_is_the_one_the_deployment_raised(self) -> None:
        """``HostUnreachable`` is 6 in the server's error table, and 6 is what the server reports."""
        assert HOST_UNREACHABLE == 6

    def test_a_chained_host_unreachable_skips_and_the_message_carries_both_halves(self) -> None:
        """``test_memory_store.py``'s shape: this package's ``StorageError``, the driver's code chained.

        ``pytest.skip`` raises ``Skipped``, which derives from ``BaseException`` rather than
        ``Exception``, so this catches it explicitly rather than letting an ``except Exception`` in the
        helper's caller decide. The skip message carries both halves: the namespace comes from this
        package's own sentence and the host and port from the driver's, and a message carrying one
        without the other sends the reader to the wrong place.
        """
        with pytest.raises(BaseException, match="search process is unreachable") as raised:
            skip_if_search_is_unreachable(self._storage_error(HOST_UNREACHABLE))

        assert raised.typename == "Skipped"
        assert "testdb.memory" in str(raised.value)
        assert "localhost:28000" in str(raised.value)

    def test_an_unchained_host_unreachable_skips_too(self) -> None:
        """``test_pipeline.py``'s shape, where the driver's error is not wrapped in anything.

        One helper serves two files only if it reads both shapes, and this is the one a guard written
        against a chained cause alone would miss.
        """
        from pymongo.errors import OperationFailure

        with pytest.raises(BaseException, match="search process is unreachable") as raised:
            skip_if_search_is_unreachable(OperationFailure("Error connecting to localhost:28000", code=6))

        assert raised.typename == "Skipped"

    def test_the_index_management_channel_is_recognised_too(self) -> None:
        """``AT-167`` -- one outage, two codes, because mongod reaches mongot over two channels.

        On a deployment running ``mongod`` with nothing on its ``mongotHost`` port, ``$search`` answers
        with 6 and the host and port that refused; ``$listSearchIndexes`` and ``createSearchIndexes``
        answer with **125**, ``CommandFailed``, *"Error connecting to Search Index Management
        service."*: a different code, a different sentence, no host in it. A guard that knew only 6
        would catch the query surface and miss every call that provisions or reads an index.
        """
        from pymongo.errors import OperationFailure

        with pytest.raises(BaseException, match="search process is unreachable") as raised:
            skip_if_search_is_unreachable(
                OperationFailure(
                    "Executor error during aggregate command :: caused by :: Error connecting to "
                    "Search Index Management service.",
                    code=125,
                )
            )

        assert raised.typename == "Skipped"

    def test_a_command_failed_that_names_something_else_is_left_to_fail(self) -> None:
        """The other half of it, and the reason 125 is matched with its message rather than alone.

        ``CommandFailed`` is a general code. Unlike 6, which is a network answer, it is what the
        server says about a command that did not work out. Matching the number on its own would skip
        on a whole class of real results. The sentence naming the Search Index Management service is
        the part this package cannot cause: the host it names comes from a server parameter.
        """
        from pymongo.errors import OperationFailure

        try:
            skip_if_search_is_unreachable(OperationFailure("Executor error during aggregate command", code=125))
        except BaseException as outcome:  # noqa: BLE001 - a skip is a BaseException, and it is the defect
            pytest.fail(
                f"the guard skipped on a bare CommandFailed: {outcome!r}. 125 without the sentence that "
                f"names the index management service is an ordinary command failure, and skipping on it "
                f"would hide results the suite exists to report"
            )

    def test_a_reason_three_links_down_is_still_found(self) -> None:
        """``AT-167`` -- the chain is walked to the end, not read two links deep.

        The ``auto`` path's *"no search tier could be prepared"* error has a cause, and that cause is
        itself this package's error wrapping the driver's. So the code sits three links from what the
        caller catches, and a guard reading ``error`` and ``error.__cause__`` alone
        stops one link short of every failure on that path.
        """
        from pymongo.errors import OperationFailure
        from strands.types.exceptions import StorageError

        driver = OperationFailure("Error connecting to localhost:27027", code=HOST_UNREACHABLE)
        tier = StorageError("strands_mongodb: creating text_index failed")
        tier.__cause__ = driver
        outer = StorageError("strands_mongodb: no search tier could be prepared on testdb.memory")
        outer.__cause__ = tier

        with pytest.raises(BaseException, match="search process is unreachable") as raised:
            skip_if_search_is_unreachable(outer)

        assert raised.typename == "Skipped"

    @pytest.mark.parametrize("code", [8, 26, 31082, None], ids=["unknown-error", "namespace", "unsupported", "none"])
    def test_every_other_failure_is_left_to_fail(self, code: int | None) -> None:
        """Every other code, or none, is left to fail.

        8 is ``UnknownError`` and is what a comparison inverted to ``!=`` would let through. 26 and 31082 are both
        real and both mean the package asked for something the deployment does not have, which is a
        result the suite must report rather than skip. ``None`` is a network error that never reached a
        server and therefore carries no code at all.
        """
        try:
            skip_if_search_is_unreachable(self._storage_error(code))
        except BaseException as outcome:  # noqa: BLE001 - a skip is a BaseException, and it is the defect
            pytest.fail(
                f"the guard skipped on code {code}: {outcome!r}. Only HostUnreachable is infrastructure, "
                f"and a skip on anything else turns a result the suite has to report into a clean-looking run"
            )


class TestTheOutageSkipsWhereverItSurfaces:
    """``AT-167`` -- the guard reaches the arrange phase, not only the act phase.

    A search outage often surfaces before the search itself: on ``await store.add(...)``, the line
    before the search, or in a fixture. ``add`` calls ``initialize`` and ``initialize`` provisions the
    search indexes, so on a deployment with no mongot the outage arrives during setup and never reaches
    a search helper that holds the guard.

    A hook around the setup and call phases sees the failure wherever it surfaces, and the match stays
    exactly as strict, which keeps a package regression a failure.
    """

    @staticmethod
    def _outage() -> BaseException:
        from pymongo.errors import OperationFailure

        return OperationFailure(
            "Executor error during aggregate command :: caused by :: Error connecting to Search "
            "Index Management service.",
            code=125,
        )

    @pytest.mark.parametrize("phase", ["setup", "call"])
    def test_an_outage_escaping_either_phase_becomes_a_skip(self, phase: str) -> None:
        """Both phases, because a fixture that provisions indexes fails in the first one.

        A failure in setup is reported as an *error* rather than a failure, which reads as "the suite is
        broken", the wrong reading for something an operator fixes with a restart.
        """
        import conftest

        wrapper = getattr(conftest, f"pytest_runtest_{phase}")(item=None)
        next(wrapper)

        with pytest.raises(BaseException, match="search process is unreachable") as raised:
            wrapper.throw(self._outage())

        assert raised.typename == "Skipped"

    @pytest.mark.parametrize("phase", ["setup", "call"])
    def test_every_other_failure_comes_out_as_it_went_in(self, phase: str) -> None:
        """The hook is a filter, not a net: anything else passes through unchanged.

        An ``AssertionError`` is what a real defect looks like arriving here, and it has to survive the
        wrapper with the same type and the same message, or a real failure could be reported as a
        skip.
        """
        import conftest

        wrapper = getattr(conftest, f"pytest_runtest_{phase}")(item=None)
        next(wrapper)

        with pytest.raises(AssertionError, match="two results, got one") as raised:
            wrapper.throw(AssertionError("expected two results, got one"))

        assert raised.typename == "AssertionError"


class TestTheEnvFileFillsTierVariables:
    """A URI written down in `.env` is configuration the harness should find.

    The gate above decides what happens when a variable is missing. This decides which variables exist,
    which is the earlier question: without it, a cluster named in `.env` would leave every tier reporting
    "not configured".
    """

    def test_only_the_three_tier_variables_are_read(self, tmp_path: Path) -> None:
        """A `.env` is not an environment to import wholesale.

        The narrowing is the point. A file written for an application deployment can carry `PATH`, a
        cloud credential or a model key, and a harness that assimilates all of it changes how the
        interpreter resolves programs for the rest of the session.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "MONGODB_URI=mongodb://one\n"
            "MONGODB_ATLAS_URI=mongodb://two\n"
            "MONGODB_AUTOEMBED_URI=mongodb://three\n"
            "PATH=/nowhere\n"
            "AWS_SECRET_ACCESS_KEY=nope\n"
        )
        environ: dict[str, str] = {}

        filled = fill_tier_variables_from_env(environ, locations=[env_file])

        assert sorted(filled) == sorted(TIER_VARIABLES.values())
        assert environ == {
            "MONGODB_URI": "mongodb://one",
            "MONGODB_ATLAS_URI": "mongodb://two",
            "MONGODB_AUTOEMBED_URI": "mongodb://three",
        }

    def test_a_value_already_in_the_environment_is_never_replaced(self, tmp_path: Path) -> None:
        """The file fills gaps. CI exports its own URIs and must behave as if the file did not exist."""
        env_file = tmp_path / ".env"
        env_file.write_text("MONGODB_URI=mongodb://from-the-file\nMONGODB_ATLAS_URI=mongodb://also-the-file\n")
        environ = {"MONGODB_URI": "mongodb://from-the-shell"}

        filled = fill_tier_variables_from_env(environ, locations=[env_file])

        assert filled == ["MONGODB_ATLAS_URI"]
        assert environ["MONGODB_URI"] == "mongodb://from-the-shell"

    def test_an_explicitly_empty_value_means_no_backend_and_survives(self, tmp_path: Path) -> None:
        """`MONGODB_URI= pytest` is how a developer runs the unbacked tier deliberately.

        The gate reads an empty value as unset, so the two rules have to disagree about emptiness: the
        gate asks whether the value is truthy, and the loader asks whether the key is present. Were the
        loader to test truthiness too, opting out of a configured backend would be impossible from the
        command line, and `.env` would become mandatory rather than a default.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("MONGODB_URI=mongodb://from-the-file\n")
        environ = {"MONGODB_URI": ""}

        filled = fill_tier_variables_from_env(environ, locations=[env_file])

        assert filled == []
        assert environ["MONGODB_URI"] == ""

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("MONGODB_URI=mongodb://host/?a=1&b=2", "mongodb://host/?a=1&b=2"),
            ('MONGODB_URI="mongodb://host/"', "mongodb://host/"),
            ("MONGODB_URI='mongodb://host/'", "mongodb://host/"),
            ("  MONGODB_URI = mongodb://host/  ", "mongodb://host/"),
            ("MONGODB_URI=mongodb://user:p%40ss@host/", "mongodb://user:p%40ss@host/"),
        ],
        ids=["equals-in-the-value", "double-quoted", "single-quoted", "padded", "escaped-password"],
    )
    def test_the_value_is_everything_after_the_first_equals(self, line: str, expected: str) -> None:
        """A connection string carries `=` in its query string, so splitting on every `=` truncates it.

        The escaped-password case is the one that would fail silently: a URI-encoded `@` or `:` is
        ordinary in a generated Atlas password, and a parser that unescapes or re-splits produces a
        string that connects to the wrong place or not at all.
        """
        assert parse_env_file(line) == {"MONGODB_URI": expected}

    def test_comments_blanks_and_malformed_lines_are_skipped(self) -> None:
        """A hand-edited file has commented-out URIs in it. None of them is a variable."""
        parsed = parse_env_file(
            "# the old cluster\n"
            "#MONGODB_URI=mongodb://retired\n"
            "\n"
            "   \n"
            "this line has no equals sign\n"
            "MONGODB_URI=mongodb://current\n"
        )

        assert parsed == {"MONGODB_URI": "mongodb://current"}

    def test_the_first_existing_location_wins(self, tmp_path: Path) -> None:
        """Two files are a mistake, not a merge. The nearer one is the one someone is editing."""
        first = tmp_path / "python" / ".env"
        first.parent.mkdir()
        first.write_text("MONGODB_URI=mongodb://nearer\n")
        second = tmp_path / ".env"
        second.write_text("MONGODB_URI=mongodb://further\nMONGODB_ATLAS_URI=mongodb://further-atlas\n")
        environ: dict[str, str] = {}

        fill_tier_variables_from_env(environ, locations=[first, second])

        assert environ == {"MONGODB_URI": "mongodb://nearer"}

    def test_no_file_anywhere_is_the_normal_case(self, tmp_path: Path) -> None:
        """CI has no `.env`. Absence is not a warning, an error, or a reason to skip differently."""
        environ: dict[str, str] = {}

        filled = fill_tier_variables_from_env(environ, locations=[tmp_path / "absent" / ".env"])

        assert filled == []
        assert environ == {}

    def test_the_locations_are_the_package_directory_then_the_repository_root(self) -> None:
        """Both are `.gitignore`d, and that is asserted rather than trusted.

        A harness that read credentials from a path the repository would commit would encourage
        committing them, and a `.env` typically holds a connection string with a password in it.
        """
        package_directory = Path(__file__).resolve().parent.parent
        assert list(ENV_FILE_LOCATIONS) == [package_directory / ".env", package_directory.parent / ".env"]

        ignored = (package_directory.parent / ".gitignore").read_text().splitlines()
        assert ".env" in [line.strip() for line in ignored]

    def test_the_loader_ran_at_import_time(self) -> None:
        """Wiring, not parsing: the fixtures and the collection hook both read `os.environ`.

        Conditional because the file is `.gitignore`d and therefore absent on CI, which is the case the
        assertion has to tolerate rather than the case it is for.
        """
        present = [path for path in ENV_FILE_LOCATIONS if path.is_file()]
        if not present:
            pytest.skip("no .env in this checkout, which is the CI configuration")

        for name in parse_env_file(present[0].read_text()):
            if name in TIER_VARIABLES.values():
                assert name in os.environ, f"{name} is in {present[0]} but was never loaded"


class TestAFailureReportNeverCarriesThePassword:
    """AT-103 — the connection string the harness reads never reaches a test report with its password.

    The package keeps credentials out of its own output: pymongo strips them from its ``repr`` and
    nothing in the package formats a config. The value pytest prints for a failing test is a different
    one: the URI the harness read out of the environment before any package code ran. Pytest prints every
    fixture a failing test requested, so without redaction the user and password would appear in the
    fixture header of every failure.

    Two halves, because the mechanism and the outcome fail differently. The first asserts what the value
    does; the second runs a session that fails on purpose and reads the report, which is the artifact
    that reaches a CI log.
    """

    _SECRET = "Sup3rS3cret"
    _URI = f"mongodb+srv://alice:{_SECRET}@cluster0.example.mongodb.net/"

    def test_the_repr_redacts_while_the_value_still_connects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-103, the mechanism. Redacting ``str`` too would fail every connection instead."""
        monkeypatch.setenv("MONGODB_URI", self._URI)

        value = _require("MONGODB_URI")

        assert self._SECRET not in repr(value)
        assert self._SECRET not in f"{value!r}"
        assert "cluster0.example.mongodb.net" in repr(value), "the host is what makes a failure diagnosable"
        # The value itself is unchanged, so it still connects; only its repr is redacted.
        assert value == self._URI
        assert str(value) == self._URI

    def test_a_failing_test_does_not_print_the_password_in_its_fixture_header(
        self, gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AT-103, the outcome. The report is what reaches a CI log, so the report is what is asserted.

        The nested test both requests the fixture and fails: pytest prints every fixture a failing test
        requested, using ``repr``.
        """
        monkeypatch.setenv("MONGODB_URI", self._URI)
        gated.makepyfile(
            """
            import pytest

            @pytest.mark.contract
            def test_fails_while_holding_the_uri(contract_uri):
                assert contract_uri
                assert False, "failing on purpose, to read what the report prints"
            """
        )

        result = gated.runpytest()

        result.assert_outcomes(failed=1)
        assert self._SECRET not in result.stdout.str(), "the password reached the report"
        result.stdout.fnmatch_lines(["*cluster0.example.mongodb.net*"])


@pytest.mark.contract
class TestALegPromisedASearchProcessGotOne:
    """``AT-187`` -- the guard on the half of the contract tier that can pass vacuously.

    The contract tier runs twice, over a deployment with a search process and one without, because the
    rules about which search tier a store resolves are all true for a trivial reason where nothing can
    search: the store resolves no tier at all and every branch about fusion takes its quiet side. The leg
    with mongot is what reads the other side.

    That leg's value depends entirely on it really having mongot. If the image changed, or the service
    came up before its search process did, or a copied job used the wrong image, the tests would take
    the same quiet branches and the job would still be green.

    So the promise is made explicit in the environment and read back here. This belongs with the other
    tests of the harness rather than with the tests that use it: what it asserts is that the gate is
    being fed what it was told it would be fed.
    """

    async def test_the_deployment_has_the_search_process_the_environment_promised(
        self, contract_uri: str, test_database_name: str, unique_collection_name: str, contract_database: Any
    ) -> None:
        """``AT-187`` -- with the variable set, the absence of mongot is a failure rather than a skip."""
        if not os.environ.get(EXPECT_SEARCH_VARIABLE):
            pytest.skip(f"{EXPECT_SEARCH_VARIABLE} is unset, so this deployment promised nothing")

        support = await _indexes.search_support(contract_database[unique_collection_name])

        assert support.available, (
            f"{EXPECT_SEARCH_VARIABLE} is set, so MONGODB_URI was supposed to name a deployment with a "
            f"search process, and the probe answered unavailable with server error code {support.code}. "
            "Every test that branches on what this deployment can search has just taken the branch for a "
            "deployment that cannot, and reported success"
        )
