"""Connection resolution, ownership and lifecycle: ``REQ-E-010`` through ``REQ-E-017a``.

Everything here is about the client, not about queries, so almost none of it needs a database.
``AsyncMongoClient`` construction performs no I/O and validates nothing (verified on pymongo 4.17.0:
``AsyncMongoClient("not-a-uri-at-all")`` succeeds, reading the string as a hostname), which is what
lets the ownership, registry and attribution tests run in the mock tier at full speed.

Two tests do need a real deployment, and for opposite reasons. ``AT-011`` has to observe that a
closed client actually refuses work, and ``AT-017`` has to run two real event loops. Neither can be
faked without asserting the fake.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode
from pymongo import AsyncMongoClient
from pymongo.driver_info import DriverInfo
from strands.types.exceptions import StorageError

from strands_mongodb import _client
from strands_mongodb.memory_store import MongoDBMemoryStore
from strands_mongodb.storage import MongoDBStorage

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from opentelemetry.sdk.trace import ReadableSpan

    from strands_mongodb.types import MongoDBConfig

# A password that no other string in the suite contains, so a sweep for it cannot match by accident.
LEAKY_URI = "mongodb://appuser:SUPERSECRET@db.example.com:27017/appdb"


async def _is_closed(client: AsyncMongoClient[Any]) -> bool:
    """Report whether a client has been closed, by observing it rather than reading a flag.

    ``client._closed`` would answer this in one attribute, and it is private. What a *caller* sees is
    ``InvalidOperation: Cannot use AsyncMongoClient after close``, raised before any network attempt,
    so a probe against a dead port separates the two states without a database and without a wait: a
    live client reaches server selection and times out, a closed one never gets there.

    Args:
        client: The client to probe. It is left in whatever state it was already in.

    Returns:
        ``True`` if the client refuses work because it was closed.
    """
    from pymongo.errors import InvalidOperation

    try:
        await client.admin.command("ping")
    except InvalidOperation:
        return True
    except Exception:  # noqa: BLE001 - any other failure means the client was still willing to try
        return False
    return False


@pytest.fixture(autouse=True)
def _clear_registry() -> Any:
    """Empty the module-level registry around every test.

    The registry is process-global by design (``REQ-E-012``), which makes it shared mutable state
    between tests. Clearing it before *and* after means a test never inherits a client another test
    created, and a test that leaves one behind cannot make a later assertion about identity pass for
    the wrong reason.

    Reaching into ``_REGISTRY`` is deliberate. The alternative is a ``reset_registry()`` helper in
    ``_client.py``, which would be a function shipped to users that exists only for this fixture. A
    private module's private dict is the more honest seam: it reads as a test touching internals,
    because that is what it is.
    """
    _client._REGISTRY.clear()
    yield
    _client._REGISTRY.clear()


class TestConstructionRequiresAWayToReachTheDatabase:
    """AT-015 / REQ-E-015 — the error that has to arrive before anything else does.

    Nothing upstream raises here. On pymongo 4.17.0 a string with no scheme is read as a hostname and
    an unresolvable host is not an error until an operation runs, so without this check a config
    missing its connection string surfaces as a server-selection timeout against ``localhost`` some
    seconds later, in a stack frame that names neither field.
    """

    def test_neither_client_nor_connection_string_raises_value_error(self) -> None:
        """The `IF` half of ``REQ-E-015``."""
        with pytest.raises(ValueError) as caught:
            _client.resolve({"database": "appdb"})

        assert isinstance(caught.value, ValueError)

    def test_the_message_names_both_fields_the_caller_could_have_set(self) -> None:
        """ "Missing configuration" tells the reader nothing they did not already know.

        Both names, because either one fixes it, and a message naming only ``connection_string``
        pushes a caller who already holds an ``AsyncMongoClient`` toward building a second one.
        """
        with pytest.raises(ValueError, match="connection_string") as caught:
            _client.resolve({"database": "appdb"})

        assert "client" in str(caught.value)

    def test_the_message_is_attributed_to_this_package(self) -> None:
        """House style from ``BedrockKnowledgeBaseStore``: the class or module leads the message.

        A bare ``ValueError`` about a missing key, arriving from inside an ``Agent`` construction that
        touches several backends, does not say which one is unhappy.
        """
        with pytest.raises(ValueError, match="strands.mongodb|MongoDB"):
            _client.resolve({"database": "appdb"})

    def test_no_client_is_registered_by_a_rejected_config(self) -> None:
        """The `before any I/O` half, asserted through the registry rather than by timing.

        A rejected config that still left a registry entry behind would mean the constructor got far
        enough to build something, which is the failure this clause exists to prevent.
        """
        with pytest.raises(ValueError):
            _client.resolve({"database": "appdb"})

        assert _client._REGISTRY == {}


class TestOwnershipDecidesWhoMayClose:
    """AT-010, AT-011 / REQ-E-010, REQ-E-011 — the asymmetry the reference implementation shows.

    The defect in the reference implementation was one line: ``close()`` calling
    ``self.client.close()`` unconditionally. In an agent process a single ``AsyncMongoClient`` is
    typically shared across memory, storage and the application's own queries, so closing an injected
    one tears down a pool three unrelated subsystems are still using. The bug does not show up in the
    closing component. It shows up somewhere else, later.
    """

    async def test_an_injected_client_survives_our_close(self) -> None:
        """REQ-E-010 — `SHALL NOT close it, at any point, including on its own close()`."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/", serverSelectionTimeoutMS=50)
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        _client.resolve(config)
        await _client.close(config)

        assert await _is_closed(injected) is False

    async def test_an_injected_client_survives_close_called_twice(self) -> None:
        """Idempotence on the path where a mistake is unrecoverable.

        ``close()`` is the kind of method that gets called from a ``finally`` and again from an
        ``atexit``. The second call must not find a reason to close what the first one correctly left
        alone, and it must not raise for a config it no longer has an entry for.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/", serverSelectionTimeoutMS=50)
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        _client.resolve(config)
        await _client.close(config)
        await _client.close(config)

        assert await _is_closed(injected) is False

    async def test_closing_a_config_that_never_resolved_is_a_no_op(self) -> None:
        """``close()`` runs on the unhappy path, so it must not raise on a config ``resolve`` rejected.

        The realistic shape is a ``try``/``finally`` around construction: ``resolve`` raises
        ``ValueError`` for a config missing both connection fields, and the ``finally`` calls ``close``
        on that same config. A ``close`` that re-raised there would replace the actionable
        ``REQ-E-015`` message with a second copy of itself thrown from teardown.
        """
        await _client.close({"database": "appdb"})

        assert _client._REGISTRY == {}

    async def test_a_client_we_constructed_is_closed_by_our_close(self) -> None:
        """REQ-E-011 — the other half, which the same ownership flag has to get right.

        Written alongside the injected case on purpose. A ``close()`` that closes nothing passes every
        assertion above while leaking a connection pool per store.
        """
        config: MongoDBConfig = {
            "database": "appdb",
            "connection_string": "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50",
        }
        connection = _client.resolve(config)

        await _client.close(config)

        assert await _is_closed(connection.client) is True

    def test_ownership_is_recorded_from_which_field_supplied_the_client(self) -> None:
        """The flag itself, because the two tests above cannot distinguish `right` from `lucky`.

        A ``close()`` hardcoded to never close passes the injected tests; one hardcoded to always
        close passes the constructed test. This is the field that has to carry the difference.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")

        assert _client.resolve({"database": "appdb", "client": injected}).owned is False
        assert _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:2/"}).owned is True

    def test_an_injected_client_wins_over_a_connection_string(self) -> None:
        """REQ-E-010's `WHEN the caller supplies client` outranks REQ-E-011's `WHILE client is absent`.

        Both fields set is not an error, and it is not far-fetched: a config assembled from a
        template's defaults plus a caller's override arrives exactly this way. The client the caller
        handed us is the one they expect their queries to run on.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {
            "database": "appdb",
            "client": injected,
            "connection_string": "mongodb://127.0.0.1:2/",
        }

        connection = _client.resolve(config)

        assert connection.client is injected
        assert connection.owned is False


class TestOneClientPerConnectionIdentity:
    """AT-012 / REQ-E-012 — a documented property, and the reason it is a property and not a nicety.

    A ``Storage`` reaches five SDK subsystems and a ``MemoryStore`` is registered per memory surface,
    so one agent can construct a double-digit number of these from the same config. Each
    ``AsyncMongoClient`` carries its own connection pool, defaulting to 100 sockets, plus a topology
    monitor per server. Without sharing, a config that reads like one database connection becomes
    however many the wiring happened to produce.
    """

    def test_a_hundred_resolutions_of_one_config_yield_one_client(self) -> None:
        """The count from the acceptance test, asserted on object identity rather than on a cache-hit counter."""
        config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

        clients = {id(_client.resolve(config).client) for _ in range(100)}

        assert len(clients) == 1

    def test_separate_config_dicts_with_the_same_connection_string_still_share(self) -> None:
        """`same connection identity`, not `same dict`.

        Each store is configured from its own literal, so identity keyed on the config object would
        share nothing in the case the requirement is about. This is the test that fails if the registry
        is keyed on ``id(config)``.

        Both dicts are bound to names and kept alive, which is the only reason this test can detect
        that. Passed as temporaries, the first is freed as soon as ``resolve`` returns and CPython
        hands the second its address, so ``id(config)`` collides by accident and a registry keyed on it
        passes. Found by mutation: the version of this test without the local variables did pass
        against exactly that implementation.
        """
        first_config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}
        second_config: MongoDBConfig = {"database": "otherdb", "connection_string": "mongodb://127.0.0.1:1/"}

        first = _client.resolve(first_config)
        second = _client.resolve(second_config)

        assert id(first_config) != id(second_config), "the configs must be distinct objects for this to mean anything"
        assert first.client is second.client

    def test_different_connection_strings_do_not_share(self) -> None:
        """The negative half. A registry with one global entry passes every test above."""
        first = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})
        second = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:2/"})

        assert first.client is not second.client

    def test_the_same_uri_with_different_client_args_does_not_share(self) -> None:
        """``client_args`` is part of the identity, and this is the test that says so.

        Added after a mutation that dropped ``client_args`` from the digest survived the whole suite.
        ``_identity``'s docstring already claimed the behavior and the code already implemented it, so
        the gap was invisible to review: the claim was simply unasserted.

        What the mutation would have shipped is a silent misconfiguration rather than a crash. The
        second config hits the cache, its arguments are discarded, and the caller gets a client with
        someone else's timeouts. Asserted on the timeout that reaches ``client.options``, not only on
        object identity, because sharing is the failure and the wrong timeout is the symptom a user
        would eventually report.
        """
        uri = "mongodb://127.0.0.1:1/"
        patient: MongoDBConfig = {
            "database": "appdb",
            "connection_string": uri,
            "client_args": {"connectTimeoutMS": 20000},
        }
        impatient: MongoDBConfig = {
            "database": "appdb",
            "connection_string": uri,
            "client_args": {"connectTimeoutMS": 500},
        }

        first = _client.resolve(patient)
        second = _client.resolve(impatient)

        assert first.client is not second.client
        assert first.client.options.pool_options.connect_timeout == 20.0
        assert second.client.options.pool_options.connect_timeout == 0.5

    def test_client_args_and_no_client_args_are_different_identities(self) -> None:
        """The boundary case the digest has to get right: ``{}`` and absent must key the same.

        ``config.get("client_args") or {}`` collapses both to the same JSON, which is deliberate. An
        empty dict asks for nothing, so it describes the same client as omitting the key, and treating
        them apart would build a second pool for a config that differs only in verbosity.
        """
        uri = "mongodb://127.0.0.1:1/"
        absent = _client.resolve({"database": "appdb", "connection_string": uri})
        empty = _client.resolve({"database": "appdb", "connection_string": uri, "client_args": {}})
        populated = _client.resolve(
            {"database": "appdb", "connection_string": uri, "client_args": {"connectTimeoutMS": 500}}
        )

        assert absent.client is empty.client
        assert populated.client is not absent.client

    def test_two_injected_clients_are_kept_apart_by_identity(self) -> None:
        """Injected clients key on ``id(client)``, by design.

        Two callers can inject two clients built from the same URI, and they are not
        interchangeable: they may differ in credentials, read preference, or timeouts, none of which
        the registry inspects. Deduplicating them would hand one caller the other's client.
        """
        first_client: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        second_client: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")

        first = _client.resolve({"database": "appdb", "client": first_client})
        second = _client.resolve({"database": "appdb", "client": second_client})

        assert first.client is first_client
        assert second.client is second_client
        assert len(_client._REGISTRY) == 2

    def test_closing_one_config_releases_its_registry_entry(self) -> None:
        """A registry that only ever grows is a leak, and it also breaks reuse after ``close()``.

        The entry has to go, so a later ``resolve`` of the same config builds a fresh client rather
        than handing back the closed one.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}
        first = _client.resolve(config)

        asyncio.run(_client.close(config))
        second = _client.resolve(config)

        assert second.client is not first.client


class TestDriverAttributionReachesTheServer:
    """AT-013a, AT-013b / REQ-E-013 — driver attribution, and the API that was named does not exist.

    Attribution is how MongoDB counts Strands usage in server logs and Atlas telemetry, so a silent
    failure here is invisible to us and to the user both.

    ``REQ-E-013`` originally named ``append_client_metadata``. That function is real, but it lives in
    ``pymongo_search_utils``, which the ``[storage]`` extra does not install, and ``_client.py`` is
    shared with ``storage.py``. Its entire body is a call to ``client.append_metadata``, so these tests
    assert that call and the metadata it produces.
    """

    def test_a_client_we_construct_carries_the_attribution(self) -> None:
        """AT-013a — asserted on the resulting metadata, not on the call.

        ``driver`` is not a named parameter of ``AsyncMongoClient.__init__``. It arrives through
        ``**kwargs``, so no type checker and no signature inspection will catch it being misspelled,
        renamed upstream, or dropped. The pool metadata is the only thing that actually proves it
        landed.
        """
        connection = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        metadata = connection.client.options.pool_options.metadata

        assert "Strands" in metadata["driver"]["name"]

    def test_the_attribution_extends_pymongos_own_rather_than_replacing_it(self) -> None:
        """The metadata is a pipe-delimited chain, and PyMongo's own entry has to stay in it.

        Overwriting it would misreport the driver to the server, which is worse than sending nothing:
        it makes a Strands install look like it is not using PyMongo at all.
        """
        connection = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        name = connection.client.options.pool_options.metadata["driver"]["name"]

        assert name.startswith("PyMongo")
        assert name.endswith("Strands")

    def test_the_driver_name_is_exactly_strands(self) -> None:
        """The name is a data key, not a label, so it is pinned rather than merely searched for.

        Every assertion above uses ``in`` or ``endswith``, which all pass on ``StrandsMongoDB`` or
        ``strands``. Server logs and Atlas telemetry group by this string, and the comparison in
        ``_update_metadata`` is case-insensitive but exact otherwise, so changing it orphans every
        record collected under the old one.

        The ``DriverInfo`` type is asserted alongside it because ``append_metadata`` raises ``TypeError``
        on anything else, and the injected path passes it through unexamined.
        """
        info = _client._driver_info()

        assert isinstance(info, DriverInfo)
        assert info.name == "Strands"

    def test_the_reported_version_is_the_installed_package_version(self) -> None:
        """A version pinned to a literal would report the release it was typed in, forever."""
        from importlib.metadata import version

        connection = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        versions = connection.client.options.pool_options.metadata["driver"]["version"]

        assert versions.endswith(version("strands-mongodb"))

    def test_an_injected_client_is_attributed_through_append_metadata(self) -> None:
        """AT-013b — the second path, which is the only one available for a client we did not build."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")

        _client.resolve({"database": "appdb", "client": injected})

        assert "Strands" in injected.options.pool_options.metadata["driver"]["name"]

    def test_repeated_attribution_of_one_client_is_idempotent(self) -> None:
        """A characterization test of pymongo, not an assertion about our control flow.

        Labeled that way because it cannot fail on anything we write, and a reader is entitled to know
        which tests are load bearing. Confirmed by mutation: removing the cache-hit early return from
        ``resolve`` leaves this passing.

        ``PoolOptions._update_metadata`` returns early when the name is already in the pipe-delimited
        chain, comparing case-insensitively, so appending ``Strands`` twice is a no-op and only a
        different name stacks. The spec asserted the opposite until this was read rather than inferred.

        Kept because we rely on it. Resolving one config repeatedly is the normal case
        (``REQ-E-012``), and if a future pymongo drops the dedupe, the attribution string starts
        growing once per store and this is the test that says so.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        for _ in range(5):
            _client.resolve(config)

        name = injected.options.pool_options.metadata["driver"]["name"]

        assert name.count("Strands") == 1

    def test_the_installed_pymongo_implements_append_metadata_as_a_method(self) -> None:
        """AT-013b's second half — the floor is what lets the call be unguarded.

        Below pymongo 4.14 ``append_metadata`` is not a method. It resolves through
        ``AsyncMongoClient.__getattr__`` to a ``Database`` of that name, which is truthy and not
        callable, so ``hasattr`` says yes and calling it raises ``TypeError``. That shape is why the
        declared floor is ``>=4.14.0`` rather than a runtime probe: attribution is bookkeeping a caller
        did not ask for, and a package has no business narrating its own bookkeeping to them.

        Asserted on the installed driver rather than on the specifier, which
        ``test_pymongo_floor_is_4_14_on_both_runtime_extras`` covers. Both are needed: the specifier is a
        promise to a resolver, and this is the fact the code depends on.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")

        assert callable(inspect.getattr_static(type(injected), "append_metadata", None))


class TestRetryWritesDisabledIsWorthSaying:
    """AT-016 / REQ-E-016 — a warning rather than a refusal.

    The upsert shapes in ``storage.py`` are idempotent precisely so that the driver's own retry can
    replay them. Turning ``retryWrites`` off does not corrupt anything; it removes the layer that
    absorbs a primary stepdown, and the write surfaces to the caller instead. That is a legitimate
    choice, so this warns and proceeds.

    Read from ``client.options.retry_writes`` rather than parsed out of the connection string. See the
    widening recorded on ``REQ-E-016``: it catches the setting whether it came from the URI, from
    ``client_args``, or from a client the caller injected, and it means we ship no URI parser.
    """

    def test_a_connection_string_disabling_retries_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The URI route, which is the one the requirement was written about."""
        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/?retryWrites=false"})

        assert "retryWrites" in caplog.text

    def test_the_warning_says_what_the_consequence_is(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning that only restates the setting gives the reader nothing to decide with.

        They set it deliberately. What they may not know is that this package's write shapes were
        designed around the retry it removes.
        """
        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/?retryWrites=false"})

        assert "retr" in caplog.text.lower()
        assert len(caplog.text) > 60, caplog.text

    def test_client_args_disabling_retries_warns_too(self, caplog: pytest.LogCaptureFixture) -> None:
        """The route the original wording missed, and the reason the requirement was widened."""
        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            _client.resolve(
                {
                    "database": "appdb",
                    "connection_string": "mongodb://127.0.0.1:1/",
                    "client_args": {"retryWrites": False},
                }
            )

        assert "retryWrites" in caplog.text

    def test_an_injected_client_with_retries_disabled_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The hazard does not depend on who built the client."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/", retryWrites=False)

        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            _client.resolve({"database": "appdb", "client": injected})

        assert "retryWrites" in caplog.text

    def test_a_shared_config_warns_once_no_matter_how_many_stores_resolve_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One warning per connection, not one per store, which is what makes it readable.

        A ``Storage`` reaches five SDK subsystems, so the same config is resolved many times over one
        agent's construction. A hundred identical warnings in the log is the same as none: the reader
        scrolls past them.

        This is the assertion the cache-hit early return in ``resolve`` actually protects. The
        attribution it also skips turns out to be idempotent anyway (see
        ``test_repeated_attribution_of_one_client_is_idempotent``), so without this test nothing would
        fail if that early return were deleted.
        """
        config: MongoDBConfig = {
            "database": "appdb",
            "connection_string": "mongodb://127.0.0.1:1/?retryWrites=false",
        }

        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            for _ in range(100):
                _client.resolve(config)

        assert len([r for r in caplog.records if "retryWrites" in r.getMessage()]) == 1

    def test_the_default_configuration_warns_about_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning every user sees on a correct config is noise, and it trains them to filter us out."""
        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        assert caplog.records == []


class TestThePasswordNeverLeavesTheConfig:
    """AT-014 / REQ-E-014 — the credential sweep, and the test that keeps it closed.

    ``MongoDBConfig`` is a plain ``dict`` holding a URI with an inline password. Every ordinary Python
    habit leaks it: an f-string over the config in a debug line, a ``KeyError`` naming the key it
    could not find, a dataclass ``__repr__`` that includes the field, a traceback capturing locals.
    None of those is a bug anyone writes on purpose, which is why the guard has to be a test rather
    than a review note.

    Checked what pymongo already handles, so this asserts only what is ours: ``repr`` of an
    ``AsyncMongoClient`` built from ``LEAKY_URI`` prints ``host=['db.example.com:27017']`` and no
    password, and an invalid URI option raises no exception to leak into.

    Scope note: ``AT-013`` specifies this sweep across all five ``Storage`` methods and both memory
    methods. :func:`assert_no_leak` is written to be extended there, and the sweep here covers every
    surface the connection layer exposes.
    """

    @staticmethod
    def assert_no_leak(*subjects: object) -> None:
        """Assert the password appears in none of the given renderings.

        Args:
            *subjects: Objects whose ``repr`` must not contain the password, plus any strings to check
                directly. Strings are checked as themselves rather than as their ``repr``, so a log
                record or an exception message can be passed straight in.

        Raises:
            AssertionError: Naming the leaking subject, without reproducing the password in the
                failure message, which would leak it into CI output.
        """
        for index, subject in enumerate(subjects):
            text = subject if isinstance(subject, str) else repr(subject)
            assert "SUPERSECRET" not in text, f"subject {index} ({type(subject).__name__}) leaked the password"

    def test_resolving_a_leaky_config_logs_no_password_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """DEBUG, because that is the level whose whole purpose is saying more than usual.

        A package that redacts at ``info`` and interpolates the config at ``debug`` has not solved
        this. Debug logging is enabled precisely when something is already going wrong, which is when
        the logs get pasted into a ticket.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": LEAKY_URI}

        with caplog.at_level(logging.DEBUG, logger="strands_mongodb"):
            _client.resolve(config)

        self.assert_no_leak(caplog.text)

    def test_a_leaky_config_that_also_disables_retries_redacts_the_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The one log line ``_client.py`` is *required* to emit, on a config that carries a secret.

        ``REQ-E-016`` and ``REQ-E-014`` meet here, and the natural way to write the warning is to
        include the connection string so the reader knows which client it is about.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": f"{LEAKY_URI}?retryWrites=false"}

        with caplog.at_level(logging.DEBUG, logger="strands_mongodb"):
            _client.resolve(config)

        assert "retryWrites" in caplog.text
        self.assert_no_leak(caplog.text)

    def test_nothing_the_registry_holds_renders_the_password(self) -> None:
        """The registry key is the live hazard, because it is derived from the connection string.

        A ``dict`` keyed on a URI leaks on ``repr(registry)``, in a ``KeyError``, and in any debugger
        or crash reporter that walks module globals. The identity semantics ``REQ-E-012`` needs
        survive a digest of the string, so the key does not have to be the string itself.

        ``_Connection`` is a dataclass, so its generated ``__repr__`` prints every field it has. This
        is the assertion that fails if a future field holds the URI.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": LEAKY_URI}
        connection = _client.resolve(config)

        self.assert_no_leak(connection, _client._REGISTRY, *_client._REGISTRY.keys())

    def test_a_client_pymongo_refuses_to_build_is_reported_without_the_password(self) -> None:
        """Exception messages, on the one failure ``resolve`` can hit while holding a live URI.

        ``client_args`` is forwarded to ``AsyncMongoClient`` unvalidated, so a bad entry raises from
        inside our own call. pymongo's own message does not echo the URI (checked: ``bogusOption``
        yields ``Unknown option: bogusOption``), which is what makes this a test of our handling rather
        than of theirs: the risk is a wrapper that adds context by interpolating the config.
        """
        config: Any = {"database": "appdb", "connection_string": LEAKY_URI, "client_args": {"bogusOption": 1}}

        with pytest.raises(Exception) as caught:  # noqa: PT011 - the type is pymongo's to choose
            _client.resolve(config)

        self.assert_no_leak(str(caught.value), caught.value)

    def test_the_traceback_of_a_failed_construction_holds_no_password(self) -> None:
        """Locals travel with a traceback, which is what makes this hard to get right by inspection.

        Any frame between the caller and the raise that binds the URI to a name puts it in the
        traceback, and from there into every crash reporter the user runs with locals enabled.

        The ``else`` clause is not decoration. An earlier version of this test used a config nothing
        rejects, so the ``except`` body never ran and it passed while asserting nothing at all.
        """
        import traceback

        config: Any = {"database": "appdb", "connection_string": LEAKY_URI, "client_args": {"bogusOption": 1}}

        try:
            _client.resolve(config)
        except Exception as error:
            self.assert_no_leak("".join(traceback.format_exception(error)))
        else:
            pytest.fail("expected pymongo to reject client_args={'bogusOption': 1}")

    def test_the_password_is_not_in_the_module_logger_name_or_any_handler_state(self) -> None:
        """A cheap sweep over what the module keeps at rest, after a leaky config has been through it.

        ``repr`` of a module's globals is what a traceback renderer, a memory dump, or an error
        reporter with ``include_locals`` will walk.
        """
        _client.resolve({"database": "appdb", "connection_string": LEAKY_URI})

        self.assert_no_leak(*(value for value in vars(_client).values() if not callable(value)))


# Port 1 is reserved and never listening, and 50 ms is what makes "cannot connect" fast enough to run in
# the unit tier instead of waiting out pymongo's 30-second default. The shape is ``test_storage.py``'s
# ``UNREACHABLE`` carrying ``LEAKY_URI``'s password, because ``AT-117`` needs both properties at once: a
# config that holds a secret, and calls that reach their own error handling rather than stopping at
# validation.
LEAKY_UNREACHABLE = "mongodb://appuser:SUPERSECRET@127.0.0.1:1/?serverSelectionTimeoutMS=50"


def _rendered(span: ReadableSpan) -> str:
    """Return every string a span carries, joined, ready for a substring scan.

    Written from the ``ReadableSpan`` surface rather than from the setters, so a value reaches the scan the
    same way whatever route put it there. Verified against opentelemetry-sdk 1.44.0: ``attributes`` and an
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


def _assert_no_span_leaks(spans: Sequence[ReadableSpan]) -> None:
    """Assert no span renders the password.

    Args:
        spans: The finished spans to scan.

    Raises:
        AssertionError: Naming the offending span, without reproducing the password: a failure here is
            read in CI output, which is one of the places the password must not reach.
    """
    for span in spans:
        assert "SUPERSECRET" not in _rendered(span), f"span {span.name!r} leaked the password"


@pytest.fixture
def exported_spans() -> Iterator[InMemorySpanExporter]:
    """Put an in-memory exporter behind the *global* tracer provider, and take it out again.

    Global, because that is the only provider package code would ever reach:
    ``trace.get_tracer(__name__)`` and ``trace.get_current_span()`` both read it, so a provider kept local
    to this test would leave a leak writing somewhere nobody was watching.

    Two paths, because both states occur. A fresh interpreter has a ``ProxyTracerProvider`` and takes
    ``set_tracer_provider``; a suite that already configured tracing has a real one and takes another
    processor. Measured on 1.44.0: importing ``strands`` does not install a provider, so the first path is
    the one the unit tier takes.

    ``set_tracer_provider`` cannot be undone, so teardown shuts the exporter down and leaves the provider
    in place. A shut-down ``InMemorySpanExporter`` records nothing, and no other test in this suite emits
    or reads a span.
    """
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(processor)
    else:
        installed = TracerProvider()
        installed.add_span_processor(processor)
        trace.set_tracer_provider(installed)
    try:
        yield exporter
    finally:
        processor.shutdown()


class TestNoSpanCarriesThePassword:
    """AT-117 / REQ-E-114 — the fourth surface a password can reach, and the only one no test had looked at.

    ``REQ-E-114`` asks that a config holding an inline password survive ``repr()``, ``str()``, log output and OTel
    span attributes. The first three are closed: ``TestThePasswordNeverLeavesTheConfig`` above covers
    construction, the registry, debug logging, exception messages and tracebacks, ``test_storage.py`` and
    ``test_memory_store.py`` extend the sweep across the five storage methods and the four memory methods,
    and both of those pass the store instances themselves as subjects, which is what covers ``repr``. No
    test in the suite had ever created a span.

    Measured before writing this: ``src/strands_mongodb/`` contains no occurrence of ``opentelemetry``,
    ``otel``, ``span`` or ``tracer``. The package emits no spans of its own, and that is deliberate rather
    than pending, since a partner package must not bring an OTel dependency or configure
    exporters of its own.

    So this is a tripwire for a leak that does not exist yet, which is the case for writing it now. A
    package that may not configure an exporter, and later wants to record an operation name and a duration,
    has one place to put them: the span already current on the calling thread. Whoever adds that line will
    have the config in scope, and will be adding it to a module whose tests are all about queries.
    """

    @pytest.mark.parametrize("surface", ["name", "attribute", "event", "status"])
    def test_the_scan_catches_a_span_leaking_on_any_surface(
        self, surface: str, exported_spans: InMemorySpanExporter
    ) -> None:
        """The positive control, one span per surface, each leaking in exactly one place.

        Without this the test below is close to vacuous: the package emits no spans, so a scan that read
        the wrong field, or an exporter that received nothing, would pass just as well as a clean package.
        Parametrized rather than one span leaking four ways, because a single span carrying the password in
        every field passes against a scan that only reads one of them.
        """
        tracer = trace.get_tracer("at117.control")

        with tracer.start_as_current_span("leak-SUPERSECRET" if surface == "name" else "control") as span:
            if surface == "attribute":
                span.set_attribute("connection_string", LEAKY_UNREACHABLE)
            elif surface == "event":
                span.add_event("connection resolved", {"uri": LEAKY_UNREACHABLE})
            elif surface == "status":
                span.set_status(Status(StatusCode.ERROR, f"cannot reach {LEAKY_UNREACHABLE}"))

        finished = exported_spans.get_finished_spans()
        assert len(finished) == 1, "the control needs exactly one span, or it is testing the wrong one"
        with pytest.raises(AssertionError, match="leaked the password"):
            _assert_no_span_leaks(finished)

    async def test_exercising_the_package_emits_no_span_that_carries_the_password(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        """Both stores and the client, driven from inside an enclosing span, on a leaky config.

        The enclosing span is what a host application supplies: an agent invocation is already inside one
        by the time a store is called. It is also what makes this assertion non-empty, and it is the span a
        future ``trace.get_current_span().set_attribute(...)`` in this package would write to.

        Every call fails, and that is the surface worth covering. A store built once at start-up and
        failing on every call afterwards is the shape ``REQ-E-014`` exists for, and a failure is when a
        package is most tempted to say which connection it was talking about.

        Measured on opentelemetry-sdk 1.44.0: the exercise exports exactly one span, the enclosing one.
        The count is not asserted, because instrumentation arriving later is a change to review rather than
        a failure of this requirement, and ``REQ-E-114`` is about what a span carries.

        Six calls rather than all nine public methods, and the reason is a measurement: a call against a
        closed port costs 0.500s whatever the URI asks for, because pymongo's server selection cannot
        conclude before one heartbeat interval and ``MIN_HEARTBEAT_INTERVAL`` is 0.5s. Verified with raw
        pymongo at ``serverSelectionTimeoutMS`` of both 50 and 10: 0.504s each time. So the list keeps one
        call per distinct path, a find, an upsert, an aggregation, the index path and both memory
        directions, and drops ``delete``, ``list`` and ``delete_prefix``, which are variants of paths
        already here. That is 3.0s of the unit tier instead of 4.5s.
        """
        tracer = trace.get_tracer("at117.host")
        config: MongoDBConfig = {"database": "appdb", "connection_string": LEAKY_UNREACHABLE}
        storage = MongoDBStorage(config, collection="at117")
        store = MongoDBMemoryStore(
            name="at117",
            config=config,
            collection="at117_memory",
            scope="at117",
            writable=True,
        )

        try:
            with tracer.start_as_current_span("host.agent.invocation"):
                _client.resolve(config)
                for call in ("read", "search"):
                    with contextlib.suppress(StorageError):
                        await getattr(storage, call)("session/at117")
                with contextlib.suppress(StorageError):
                    await storage.write("session/at117", b'{"kind": "snapshot"}')
                with contextlib.suppress(StorageError):
                    await store.initialize()
                with contextlib.suppress(StorageError):
                    await store.add("a memory this test writes")
                with contextlib.suppress(StorageError):
                    await store.search("memory")
                store.get_tools()
        finally:
            await storage.close()
            await store.close()

        finished = exported_spans.get_finished_spans()
        assert finished, "no span was exported, so this test proved nothing"
        _assert_no_span_leaks(finished)

    @pytest.mark.contract
    async def test_the_succeeding_path_emits_no_span_that_carries_the_password(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        exported_spans: InMemorySpanExporter,
    ) -> None:
        """The same exercise where every call works, which is the half the failing path cannot cover.

        The instrumentation a later version would add records an operation name, a duration and a result count.
        Two of those three are only known once a call has succeeded, so a leak could plausibly live on a
        line that a failing exercise never reaches. This is that line's coverage.

        Same recipe as ``test_memory_store.py``'s succeeding half: an injected client that reaches the real
        deployment, alongside a ``connection_string`` that still holds a secret. Nothing else produces a
        config that both works and carries a password, and that combination is what ``REQ-E-014`` and
        ``REQ-E-114`` are both about.

        ``search`` is allowed to fail. A deployment without a search process refuses it, and this test is
        about spans rather than about search tiers.
        """
        client: AsyncMongoClient[Any] = AsyncMongoClient(contract_uri)
        # The real host, reached with the real client, described by a URI carrying a password the server
        # would reject. Written inline rather than borrowing `test_memory_store.py`'s helper: reaching
        # across modules for two lines ties their collection order together.
        scheme, _, rest = contract_uri.partition("://")
        config: dict[str, Any] = {
            "client": client,
            "connection_string": f"{scheme}://appuser:SUPERSECRET@{rest.partition('@')[2] or rest}",
            "database": test_database_name,
        }
        storage = MongoDBStorage(config, collection=unique_collection_name)  # type: ignore[arg-type]
        store = MongoDBMemoryStore(
            name="at117",
            config=config,  # type: ignore[arg-type]
            collection=f"{unique_collection_name}_memory",
            scope="at117",
            writable=True,
        )
        tracer = trace.get_tracer("at117.host")

        try:
            with tracer.start_as_current_span("host.agent.invocation"):
                await storage.write("session/at117", b'{"kind": "snapshot"}')
                await storage.read("session/at117")
                await storage.list("session/")
                with contextlib.suppress(StorageError):
                    await storage.search("snapshot")
                await storage.delete("session/at117")
                with contextlib.suppress(StorageError):
                    await store.add("a memory this test writes")
                with contextlib.suppress(StorageError):
                    await store.search("memory")
        finally:
            await storage.close()
            await store.close()
            # `close()` never closes an injected client (`REQ-E-010`), which is the asymmetry that lets
            # this config hold a secret at all.
            await client.close()

        finished = exported_spans.get_finished_spans()
        assert finished, "no span was exported, so this test proved nothing"
        _assert_no_span_leaks(finished)


class TestTheLockFollowsTheEventLoop:
    """AT-017 / REQ-E-017 — the event-loop boundary, and a failure mode worse than the requirement implies.

    A synchronous ``Agent`` runs each invocation on a fresh event loop, so a lock created during the
    first invocation is asked to work during the second on a loop that no longer exists.

    Two things about this are sharper than ``REQ-E-017`` states, both verified on 3.14.5:

    * **It only bites under contention.** ``asyncio.Lock`` has bound its loop lazily since 3.10:
      ``acquire()`` on an unlocked lock returns without calling ``_get_loop()``, so an uncontended
      lock crosses loops silently. A test that acquires it once per loop passes against a completely
      broken implementation. Both halves below therefore force a waiter.
    * **It does not recover.** The ``RuntimeError`` is raised after ``self._locked`` is already set,
      so the failing lock is left locked forever and every later invocation fails too. The
      user-visible symptom is an agent that works once and then hangs or raises on every subsequent
      call, which reads like a connection problem rather than a lock problem.
    """

    @staticmethod
    async def _contend(connection: Any) -> None:
        """Acquire the connection's lock from two tasks at once, forcing a waiter future.

        The waiter is the whole point. ``asyncio.Lock`` binds to a loop when it creates one, and never
        otherwise, so this is the minimum work that makes the binding observable.

        Args:
            connection: The resolved connection whose lock to contend.
        """

        async def hold() -> None:
            async with connection.get_lock():
                await asyncio.sleep(0.01)

        await asyncio.gather(hold(), hold())

    def test_a_shared_config_works_across_two_successive_event_loops(self) -> None:
        """The requirement, asserted with two real ``asyncio.run`` calls as the acceptance test specifies.

        ``asyncio.run`` creates a loop, runs, and closes it. Nothing short of that reproduces the
        failure: one loop with two tasks is the case that already works, and it is what a naive
        version of this test would do.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}
        connection = _client.resolve(config)

        asyncio.run(self._contend(connection))
        asyncio.run(self._contend(connection))

    def test_the_same_config_resolved_again_in_the_new_loop_also_works(self) -> None:
        """The realistic shape: each invocation resolves the config for itself and hits the cache.

        The cache hit is what makes this interesting. A fresh resolution would return a fresh lock and
        never exercise the rebinding, so an implementation that rebinds nothing would pass.

        Both resolutions here run with no loop running, which is why they still share one entry:
        ``REQ-E-017a`` splits the registry per running loop, and neither of these calls has one. The
        entry a caller shares across loops is the loop-free slot, not a loop's own.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

        asyncio.run(self._contend(_client.resolve(config)))
        asyncio.run(self._contend(_client.resolve(config)))

    def test_the_lock_is_replaced_rather_than_reused_when_the_loop_changes(self) -> None:
        """The mechanism, because both tests above pass on a lock that is simply never contended.

        A ``get_lock()`` returning a brand-new lock every call would also pass them, and would
        serialize nothing. This pins the actual contract: same object within a loop, different object
        across loops.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}
        connection = _client.resolve(config)

        async def bind_and_return() -> asyncio.Lock:
            await self._contend(connection)
            return connection.get_lock()

        first = asyncio.run(bind_and_return())
        second = asyncio.run(bind_and_return())

        assert first is not second

    def test_the_lock_is_stable_within_one_loop(self) -> None:
        """The other half of the contract, and the one that makes the lock a lock.

        Two coroutines on the same loop must receive the same object, or nothing is mutually excluded.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}
        connection = _client.resolve(config)

        async def two_views() -> tuple[asyncio.Lock, asyncio.Lock]:
            return connection.get_lock(), connection.get_lock()

        first, second = asyncio.run(two_views())

        assert first is second


class TestTheClientFollowsTheEventLoop:
    """AT-091 / REQ-E-017a — the event-loop boundary again, on the half ``AT-017`` left alone.

    ``AT-017`` rebinds the lock and stops there, and the class above it says so in its own name. The
    client is the other per-loop object and the one that actually raises: measured on pymongo 4.17.0,
    ``AsyncMongoClient`` records the running loop on its first operation and every later operation from
    a different loop raises ``RuntimeError: Cannot use AsyncMongoClient in different event loop``.

    Why the whole existing suite passed with this broken: every fixture is function-scoped, and every
    test that builds a store closes it, which pops the registry entry before the next loop asks for one.
    The failure needs a store that **outlives** one loop, which is what a synchronous ``Agent`` is and
    what no test had.

    These assertions are on the registry rather than on a write, because the observable difference is
    which client comes back. ``AT-091b`` in ``test_storage.py`` makes the same claim against a real
    deployment, where the wrong answer is a ``RuntimeError`` instead of a failed identity check.
    """

    CONFIG: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

    @staticmethod
    def _resolve_on_a_fresh_loop(config: MongoDBConfig) -> Any:
        """Resolve a config inside its own ``asyncio.run``, then let that loop close.

        ``asyncio.run`` is the point rather than an implementation detail: it creates a loop, runs, and
        closes it, which is exactly what ``strands/_async.py`` does for every synchronous entry point.

        Args:
            config: The config to resolve.

        Returns:
            The connection that was resolved on that now-dead loop.
        """

        async def body() -> Any:
            return _client.resolve(config)

        return asyncio.run(body())

    def test_two_successive_loops_get_two_clients(self) -> None:
        """The requirement. One client cannot serve both, so resolving twice must not share one."""
        first = self._resolve_on_a_fresh_loop(self.CONFIG)
        second = self._resolve_on_a_fresh_loop(self.CONFIG)

        assert first.client is not second.client

    def test_the_dead_loops_entry_is_dropped_rather_than_accumulated(self) -> None:
        """The registry's bound. Ten loops must not leave ten entries in a process-global dict."""
        for _ in range(10):
            self._resolve_on_a_fresh_loop(self.CONFIG)

        assert len(_client._REGISTRY) == 1, sorted(str(key) for key in _client._REGISTRY)

    def test_one_loop_still_shares_one_client_across_many_resolutions(self) -> None:
        """``REQ-E-012`` unchanged, which is the thing a per-loop key could plausibly break.

        Resolving inside one loop has to keep returning the same connection, or every store in an
        ordinary async application opens its own pool.
        """

        async def resolve_thrice() -> list[Any]:
            return [_client.resolve(self.CONFIG) for _ in range(3)]

        connections = asyncio.run(resolve_thrice())

        assert len({id(connection.client) for connection in connections}) == 1

    def test_a_client_built_outside_any_loop_is_adopted_rather_than_replaced(self) -> None:
        """The ``None`` slot, and the reason it exists.

        Construction from module-level code has no running loop to key on, and the client it builds is
        unbound until its first operation. Handing that client to the first loop that asks keeps the
        common arrangement, construct at import and use on one loop, at one pool. Building a second
        would leave the first open, unused, and never closed.
        """
        before = _client.resolve(self.CONFIG)

        async def resolve_inside() -> Any:
            return _client.resolve(self.CONFIG)

        after = asyncio.run(resolve_inside())

        assert after.client is before.client
        assert len(_client._REGISTRY) == 1

    def test_two_live_loops_each_get_their_own_client(self) -> None:
        """The case that is not sequential, and the one the storage example hits.

        ``SnapshotSessionManager.initialize`` is a synchronous hook that runs its read on a fresh loop
        in a worker thread while the caller's loop is still running. Both loops are alive at once, so
        replacing one entry with the other would hand a live loop a client bound elsewhere.
        """
        from concurrent.futures import ThreadPoolExecutor

        async def outer() -> tuple[Any, Any, Any]:
            mine = _client.resolve(self.CONFIG)

            def in_a_worker() -> Any:
                async def body() -> Any:
                    return _client.resolve(self.CONFIG)

                return asyncio.run(body())

            with ThreadPoolExecutor() as executor:
                theirs = executor.submit(in_a_worker).result()
            # Resolved again after the worker's loop closed: the caller's own loop must still get its
            # own client back, not the one the worker built and abandoned.
            return mine, theirs, _client.resolve(self.CONFIG)

        mine, theirs, again = asyncio.run(outer())

        assert theirs.client is not mine.client
        assert mine.client is again.client

    def test_an_injected_client_is_never_split_across_loops(self) -> None:
        """The carve-out. We cannot build a second copy of a client we were handed.

        A caller who injects one client and drives it from two loops gets pymongo's ``RuntimeError``,
        and that is the right owner for the mistake. What this package must not do is quietly open a
        second pool the caller never asked for and cannot close.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        first = self._resolve_on_a_fresh_loop(config)
        second = self._resolve_on_a_fresh_loop(config)

        assert first.client is injected
        assert second.client is injected
        assert len(_client._REGISTRY) == 1

    def test_close_releases_every_loops_slot_for_the_identity(self) -> None:
        """``close()`` is per config, and a per-loop key must not turn that into per config per loop.

        A caller who closes once and finds a pool still registered has a leak no further call can reach,
        because nothing in the public API names a loop. Synchronous, like the tests above it, because a
        closing loop is half of what is being asserted and pytest-asyncio would supply the loop instead.
        """
        self._resolve_on_a_fresh_loop(self.CONFIG)

        async def resolve_then_close() -> None:
            _client.resolve(self.CONFIG)
            assert len(_client._REGISTRY) >= 1
            await _client.close(self.CONFIG)

        asyncio.run(resolve_then_close())

        assert _client._REGISTRY == {}


class TestConcurrentClosesOnOneIdentity:
    """AT-091d / REQ-E-017c — three stores shutting down at once, which is the ordinary shape.

    ``examples/memory_recall.py write`` ends with
    ``asyncio.gather(curator.close(), writable.close(), curated.close())``. Three stores, one connection
    identity, one call each. That raised ``KeyError`` against the live cluster and never once in the
    suite, because every test until now closed one config at a time.

    Reproducing it needs two things at the same time: more than one slot for the identity, which
    ``REQ-E-017a`` introduced, and a real suspension inside the first caller's ``client.close()``, which
    is what lets the second caller run and drop a slot the first one still has on its list. A dead loop's
    slot is the cheapest second slot, and the closed client below is the cheapest way to make one.
    """

    CONFIG: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

    @staticmethod
    def _slot_for_a_dead_loop() -> None:
        """Leave one entry behind whose loop has closed, without going through ``resolve``.

        A plain ``asyncio.run`` that resolves and returns would leave exactly this, and ``resolve``
        reaps it on the next miss. Built by hand so the reap cannot race the test into passing.
        """
        dead = asyncio.new_event_loop()
        dead.close()
        client: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        identity = _client._identity(TestConcurrentClosesOnOneIdentity.CONFIG)
        _client._REGISTRY[(identity, dead)] = _client._Connection(client=client, owned=True)

    def test_three_concurrent_closes_release_every_slot_without_raising(self) -> None:
        """The failure this reproduces is a ``KeyError`` from a teardown path, which is the worst place."""

        async def body() -> None:
            _client.resolve(self.CONFIG)
            self._slot_for_a_dead_loop()
            assert len(_client._REGISTRY) == 2

            await asyncio.gather(*(_client.close(self.CONFIG) for _ in range(3)))

        asyncio.run(body())

        assert _client._REGISTRY == {}

    def test_a_slot_removed_mid_iteration_is_treated_as_already_released(self) -> None:
        """The same hazard from the other direction: ``resolve``'s reap, not a second ``close``.

        ``close()`` snapshots the identity's keys and then suspends on the first client's ``endSessions``.
        Anything that removes a slot during that window has to leave the rest of the iteration standing.
        """

        async def body() -> None:
            _client.resolve(self.CONFIG)
            self._slot_for_a_dead_loop()
            keys = list(_client._REGISTRY)

            closing = asyncio.ensure_future(_client.close(self.CONFIG))
            # One turn of the loop is enough: `close()` reaches its first `await` and the entry the
            # snapshot's second key names is still there for this line to take.
            await asyncio.sleep(0)
            _client._REGISTRY.pop(keys[-1], None)
            await closing

        asyncio.run(body())

        assert _client._REGISTRY == {}
