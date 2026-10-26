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
import sys
import threading
import traceback
from typing import TYPE_CHECKING, Any, Final

import pymongo
import pytest
from conftest import rendered
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode
from pymongo import AsyncMongoClient
from pymongo.driver_info import DriverInfo
from strands.types.exceptions import StorageError

from strands_mongodb import _client
from strands_mongodb.memory_store import MongoDBMemoryStore
from strands_mongodb.storage import MongoDBStorage

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        """The metadata a caller ends up holding, whichever layer kept it to one entry.

        Two layers now keep it there and this test cannot tell them apart, which is why
        :class:`TestAttributionHappensAtMostOncePerClient` counts the calls instead.
        ``PoolOptions._update_metadata`` returns early when the name is already in the pipe-delimited
        chain -- from pymongo 4.14.1 onward, measured; **not** on the declared floor 4.14.0 -- and
        ``_already_attributed`` does the same comparison a layer up, on every version.

        Kept for the outcome rather than the mechanism. Resolving one config repeatedly is the normal
        case (``REQ-E-012``), and this is the test that says so in the terms a caller reads: the driver
        name they hand us comes back naming Strands once.
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


def _app_name_of(client: AsyncMongoClient[Any]) -> str | None:
    """The ``appName`` the handshake will send, read where pymongo keeps it after parsing every source."""
    application = client.options.pool_options.metadata.get("application")
    return None if application is None else application["name"]


def _built_app_name(connection_string: str, **client_args: Any) -> str | None:
    config: MongoDBConfig = {"database": "appdb", "connection_string": connection_string}
    if client_args:
        config["client_args"] = client_args
    return _app_name_of(_client.resolve(config).client)


class TestAppNameAttribution:
    """AT-215 to AT-222 / REQ-E-210 to REQ-E-215, and the three behaviours that must not change.

    ``appName`` is the handshake field that server logs, ``$currentOp`` and the profiler show. Setting it
    to ``strands-mongodb``, or adding that as a suffix to the caller's own value, is how this package's usage
    gets counted at the server. Every assertion reads the value pymongo actually parsed, because the kwarg,
    the URI and their precedence all meet inside pymongo, and only the parsed value is what gets sent.
    """

    def test_a_uri_without_an_app_name_gets_the_package_name(self) -> None:
        """AT-215."""
        assert _built_app_name("mongodb://127.0.0.1:1/") == "strands-mongodb"

    def test_an_empty_app_name_counts_as_none(self) -> None:
        """AT-215. An empty value has nothing worth keeping, and ``-strands-mongodb`` would be a typo."""
        assert _built_app_name("mongodb://127.0.0.1:1/", appname="") == "strands-mongodb"

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("?appName=billing", "billing-strands-mongodb"),
            ("?APPNAME=billing", "billing-strands-mongodb"),
            ("?retryWrites=true&appName=billing&w=majority", "billing-strands-mongodb"),
            ("?appName=billing%20api", "billing api-strands-mongodb"),
        ],
    )
    def test_a_uri_app_name_is_kept_and_suffixed(self, query: str, expected: str) -> None:
        """AT-216. Decoded the way pymongo decodes it, so the server never shows a ``%20``."""
        assert _built_app_name(f"mongodb://127.0.0.1:1/{query}") == expected

    def test_the_last_uri_app_name_wins_as_it_does_in_pymongo(self) -> None:
        """AT-216. pymongo keeps the last of a repeated key, so the suffix goes on the value it would have sent."""
        assert _client._app_name("mongodb://h/?appName=first&appName=second", {}) == "second-strands-mongodb"

    @pytest.mark.parametrize("key", ["appname", "appName", "APPNAME"])
    def test_a_client_args_app_name_is_suffixed_in_any_case(self, key: str) -> None:
        """AT-217. Passing the key and the package's own kwarg together would be a ``TypeError`` on every resolve."""
        assert _built_app_name("mongodb://127.0.0.1:1/", **{key: "billing"}) == "billing-strands-mongodb"

    def test_client_args_wins_over_the_uri_as_it_does_in_pymongo(self) -> None:
        """AT-217."""
        assert _built_app_name("mongodb://127.0.0.1:1/?appName=from-uri", appname="from-args") == (
            "from-args-strands-mongodb"
        )

    @pytest.mark.parametrize("value", ["strands-mongodb", "billing-strands-mongodb"])
    def test_a_value_that_already_carries_the_suffix_is_unchanged(self, value: str) -> None:
        """AT-218. A value copied out of a server log back into a URI must not grow a second suffix."""
        assert _built_app_name(f"mongodb://127.0.0.1:1/?appName={value}") == value

    def test_an_over_long_value_is_shortened_so_the_suffix_fits(self) -> None:
        """AT-219. pymongo raises at 129 bytes, so keeping the value whole would break a working config."""
        name = _built_app_name("mongodb://127.0.0.1:1/", appname="a" * 200)

        assert name == "a" * 112 + "-strands-mongodb"

    def test_a_value_that_exactly_fits_is_not_shortened(self) -> None:
        """AT-219. The boundary itself: 112 bytes plus 16 is exactly the limit."""
        assert _built_app_name("mongodb://127.0.0.1:1/", appname="a" * 112) == "a" * 112 + "-strands-mongodb"

    def test_shortening_never_splits_a_multibyte_character(self) -> None:
        """AT-219. ``€`` is three bytes and 112 is not a multiple of three, so a byte-level cut would split one."""
        euro = _built_app_name("mongodb://127.0.0.1:1/", appname="€" * 100)

        assert euro == "€" * 37 + "-strands-mongodb"
        assert len(euro.encode()) <= 128

    def test_an_injected_client_keeps_its_own_app_name(self) -> None:
        """AT-220 / REQ-E-214. pymongo cannot change it after construction, and the requester ruled it stays."""
        mine: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/?appName=mine")
        bare: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")

        assert _app_name_of(_client.resolve({"database": "appdb", "client": mine}).client) == "mine"
        assert _app_name_of(_client.resolve({"database": "appdb", "client": bare}).client) is None

    def test_deriving_it_reads_no_credentials_and_resolves_no_dns(self) -> None:
        """AT-221 / REQ-E-215. An ``srv`` host that cannot exist proves no lookup ran: the call returns."""
        assert _client._app_name(LEAKY_URI, {}) == "strands-mongodb"
        assert _client._app_name("mongodb+srv://u:p@cluster.example.invalid/?appName=x", {}) == "x-strands-mongodb"
        assert "SUPERSECRET" not in _client._app_name(LEAKY_URI + "?appName=svc", {})

    def test_other_client_args_pass_through_and_the_callers_dict_is_untouched(self) -> None:
        """AT-222. The key is removed from a copy, never from the dict the caller owns."""
        client_args: dict[str, Any] = {"appname": "billing", "maxPoolSize": 7}
        connection = _client.resolve(
            {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/", "client_args": client_args}
        )

        assert connection.client.options.pool_options.max_pool_size == 7
        assert client_args == {"appname": "billing", "maxPoolSize": 7}

    def test_two_configs_with_the_same_uri_still_share_one_client(self) -> None:
        """AT-222. The identity is the configuration as given, not the derived name."""
        config: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/?appName=x"}
        twin: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/?appName=x"}

        assert _client.resolve(config).client is _client.resolve(twin).client

    @pytest.mark.contract
    async def test_the_server_sees_the_app_name(self, contract_uri: str) -> None:
        """AT-224 / REQ-E-210. The real boundary: the server's own record of this connection's handshake.

        ``$currentOp`` lists the aggregation that is asking, and that operation carries the ``appName`` its
        connection sent. A deployment that refuses ``$currentOp`` to this user cannot answer, so it skips.
        """
        connection = _client.resolve({"database": "admin", "connection_string": contract_uri})
        try:
            try:
                cursor = await connection.client.admin.aggregate(
                    [{"$currentOp": {}}, {"$match": {"appName": {"$regex": "strands-mongodb$"}}}]
                )
                ops = await cursor.to_list()
            except pymongo.errors.OperationFailure as error:
                pytest.skip(f"$currentOp is not available to this user: {error.code}")
        finally:
            await _client.close({"database": "admin", "connection_string": contract_uri})

        assert ops, "the server recorded no operation from a connection whose appName ends in strands-mongodb"


class TestAttributionHappensAtMostOncePerClient:
    """``AT-190`` / ``REQ-E-185`` -- the floor's pymongo stacks what the code assumed it deduped.

    :func:`_client._attribute_injected_client` runs once per registry slot, and ``close`` releases the
    slot, so a caller who closes a store and resolves the same client again is attributed again. Whether
    that mattered was answered from the installed 4.17.0, where ``PoolOptions._update_metadata`` compares
    the incoming name against the pipe-delimited chain case-insensitively and returns early. A comment in
    ``resolve`` said so in as many words.

    Measured across the declared range instead, one venv per release: **4.14.0 does not have that
    comparison and 4.14.1 onward does.** 4.14.0 is the floor, and the ``Dependency floors`` job installs
    exactly it. On the floor each cycle appends another ``Strands`` and another version to the handshake
    document, and against ``mongo:8`` the twenty-second cycle ends with every new connection failing --
    ``OperationFailure: The client metadata document must be less than or equal to 512 bytes``. A dead
    client in a process that did nothing wrong, and nothing in the package's own behaviour to see it by.

    So the guard is this package's now, and these tests count the calls *this package* makes rather than
    reading the chain the driver may have deduped for us. Counting is the whole point: on 4.14.1 upward
    the chain looks identical either way, so a test that read the chain would have passed on the floor
    while the floor was broken -- which is what happened.
    """

    @staticmethod
    def _record_appends(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record every ``append_metadata`` name, and let the real call proceed.

        Patched on the class rather than the instance, because ``AsyncMongoClient.__getattr__`` treats an
        unknown attribute as a database name and an instance attribute would be a fresh surface rather
        than the method under test. The original is called through, so the metadata a later assertion
        reads is the metadata pymongo would really have built.

        Args:
            monkeypatch: The fixture that undoes the patch when the test ends.

        Returns:
            A list that grows by one driver name per call, in call order.
        """
        appended: list[str] = []
        original = AsyncMongoClient.append_metadata

        def record(self: AsyncMongoClient[Any], driver_info: DriverInfo) -> None:
            appended.append(driver_info.name)
            original(self, driver_info)

        monkeypatch.setattr(AsyncMongoClient, "append_metadata", record)
        return appended

    async def test_closing_and_resolving_again_does_not_attribute_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-190 -- the cycle that killed the client on the floor, run three times.

        Three rather than twenty-two because the failure this asserts is the second append, not the
        twenty-second: the server's limit is where an unbounded chain is finally noticed, and the bug is
        that the chain grows at all. Removing the guard fails this on every supported pymongo, which the
        earlier reading of the chain could not do.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        appended = self._record_appends(monkeypatch)
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        for _ in range(3):
            _client.resolve(config)
            await _client.close(config)

        assert appended == ["Strands"], f"attributed {len(appended)} times across three cycles: {appended}"
        assert injected.options.pool_options.metadata["driver"]["name"].count("Strands") == 1

    def test_a_client_the_caller_attributed_first_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-190 -- the same guard from the other side, where the existing entry is not ours to blame.

        A caller who runs their own ``append_metadata`` with this name -- copying the snippet from the
        migration guide, or wrapping two Strands packages -- reaches ``resolve`` with the chain already
        naming Strands. Appending a second entry there is the same unbounded growth arriving through a
        supported call rather than through our own cycle.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(DriverInfo("Strands", "theirs"))
        appended = self._record_appends(monkeypatch)

        _client.resolve({"database": "appdb", "client": injected})

        assert appended == [], f"attributed a client that already named Strands: {appended}"

    def test_the_comparison_ignores_case_the_way_the_driver_does(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-190 -- ``strands`` and ``Strands`` are one entry, because server-side grouping folds them.

        This mirrors ``_update_metadata`` from 4.14.1 rather than inventing a rule: a case-sensitive guard
        here would append beside the caller's lowercase entry and produce the growth on a chain the driver
        would have refused to extend.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(DriverInfo("strands", "theirs"))
        appended = self._record_appends(monkeypatch)

        _client.resolve({"database": "appdb", "client": injected})

        assert appended == [], f"a lowercase entry read as a different driver: {appended}"

    def test_a_longer_name_that_starts_with_ours_is_not_ours(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-190 -- the vacuity guard, and the reason the comparison splits on the delimiter.

        ``"Strands" in chain`` is the obvious way to write this guard and it is wrong: a client already
        carrying ``StrandsToolkit`` has not been attributed to Strands, and skipping the append there
        loses the attribution entirely rather than duplicating it. Whole entries, so this client is still
        attributed and a test that could only ever see a skip cannot pass by accident.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(DriverInfo("StrandsToolkit", "theirs"))
        appended = self._record_appends(monkeypatch)

        _client.resolve({"database": "appdb", "client": injected})

        assert appended == ["Strands"], f"a different driver's name was read as ours: {appended}"
        assert injected.options.pool_options.metadata["driver"]["name"].endswith("StrandsToolkit|Strands")


class TestAttributionThatSrvResolutionDiscardsIsReapplied:
    """AT-195 / ``REQ-E-189`` — the one path where ``REQ-E-185``'s guard keeps a loss instead of a chain.

    ``append_metadata`` writes into the client's ``ClientOptions``. Resolving a ``mongodb+srv://`` URI
    assigns a **new** ``ClientOptions`` and rebuilds the topology settings from it, and that happens on the
    client's first connection rather than at construction -- so for an injected SRV client the append lands
    before the resolution and is discarded by it. ``_already_attributed`` then reads a chain with no
    ``Strands`` in it, which is true, so it is not the cause of the loss; what it is, without this, is the
    reason the loss is permanent.

    Telemetry rather than correctness, and it is a ``P3`` for that reason: no database operation behaves
    differently either way. What it costs is MongoDB's own usage counting for exactly the deployments that
    matter most, since ``mongodb+srv://`` is what Atlas hands out.

    The re-check is bounded in both directions. It runs only for injected clients, only on the
    registry-hit path, and it is a string split over metadata already in memory. And it stops: an append
    that does not become visible clears the flag, so a future driver that ignores ``append_metadata``
    costs one extra append rather than the per-operation chain growth ``REQ-E-185`` exists to prevent.
    """

    @staticmethod
    def _lose_the_attribution(client: AsyncMongoClient[Any]) -> None:
        """Strip ``Strands`` from the chain in place, which is what the new ``ClientOptions`` amounts to.

        Written on the metadata rather than by re-running SRV resolution, so this half of the class stays
        a test of the branch. The test below runs the real resolution and asserts the loss is real.
        """
        driver = client.options.pool_options.metadata["driver"]
        driver["name"] = "|".join(entry for entry in driver["name"].split("|") if entry.casefold() != "strands")
        # pymongo 4.19 also dedupes on a private set of appended drivers, which a new ``ClientOptions`` starts empty.
        getattr(client.options.pool_options, "_PoolOptions__appended_drivers", set()).clear()

    def test_an_attribution_that_was_discarded_is_applied_again_on_the_next_resolve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The finding. The second resolve is a registry hit, which before this returned without looking."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}
        _client.resolve(config)
        self._lose_the_attribution(injected)
        appended = TestAttributionHappensAtMostOncePerClient._record_appends(monkeypatch)

        _client.resolve(config)

        assert appended == ["Strands"], f"the hit path did not re-apply a lost attribution: {appended}"
        assert _client._already_attributed(injected) is True

    def test_a_client_that_kept_it_is_not_appended_to_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``REQ-E-185`` still holds, and this is the test that says the re-check did not repeal it.

        Ten hits on a client that kept its attribution, which on the 4.14.0 floor would be ten entries in
        the handshake document if the re-check appended unconditionally instead of reading first.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}
        _client.resolve(config)
        appended = TestAttributionHappensAtMostOncePerClient._record_appends(monkeypatch)

        for _ in range(10):
            _client.resolve(config)

        assert appended == [], f"a client that already named Strands was appended to: {appended}"

    def test_an_append_that_does_not_become_visible_stops_the_re_checking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound, and the one that makes this safe to add rather than merely correct.

        A driver whose ``append_metadata`` does nothing observable is not a driver to keep calling: the
        chain it is not growing here is the chain it *is* growing on 4.14.0, once per resolve, and
        ``REQ-E-185`` measured where that ends. Five resolves, two appends, and the second one is what
        proves the first was not simply skipped.
        """
        appended: list[str] = []

        def refuse(self: AsyncMongoClient[Any], driver_info: DriverInfo) -> None:
            appended.append(driver_info.name)

        monkeypatch.setattr(AsyncMongoClient, "append_metadata", refuse)
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        for _ in range(5):
            _client.resolve(config)

        assert appended == ["Strands", "Strands"], f"the re-check did not stop: {appended}"

    async def test_srv_resolution_really_does_discard_it_and_the_next_resolve_recovers_it(self) -> None:
        """The premise, against pymongo's own SRV path with only the DNS lookup replaced.

        Everything above is a test of the branch, and every one of them would pass against a package
        whose attribution was never lost -- the loss is the assumption the requirement rests on, and it
        belongs to the driver rather than to this package. So this runs the real ``_resolve_srv`` on a
        real client, with ``_parse_srv`` returning what a lookup would have returned, and asserts both
        halves: that the append is gone afterwards, and that resolving the client again brings it back.

        Measured on pymongo 4.17.0 while writing it: ``PyMongo|c|async|Strands`` before, and
        ``PyMongo|c|async`` after.
        """
        from unittest import mock

        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb+srv://cluster0.example.mongodb.net/", connect=False)
        config: MongoDBConfig = {"database": "appdb", "client": injected}
        _client.resolve(config)
        assert _client._already_attributed(injected) is True, "the append did not take, so nothing below means anything"

        resolved = {
            "nodelist": [("host1.example.mongodb.net", 27017)],
            "options": {"replicaSet": "rs0", "tls": True},
        }
        with mock.patch("pymongo.asynchronous.uri_parser._parse_srv", new=mock.AsyncMock(return_value=resolved)):
            await injected._resolve_srv()

        assert _client._already_attributed(injected) is False, (
            "SRV resolution kept the attribution, so either the driver changed or this requirement is "
            "describing something that no longer happens"
        )

        _client.resolve(config)

        assert _client._already_attributed(injected) is True
        assert injected.options.pool_options.metadata["driver"]["name"].endswith("Strands")


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


class TestTheCredentialsTheUriRedactionDoesNotCover:
    """AT-177 / ``REQ-E-171`` — credential leakage again, on the half ``AT-014`` could not see.

    ``AT-014`` above asserts that the password does not render, and it passed for four reasons, only
    three of which were ours. The fourth was pymongo: ``AsyncMongoClient.__repr__`` prints every
    non-default option it was built with, and the *password in the URI* is the one it has always
    redacted. A credential passed as any other option is not covered by that.

    Two options carry a secret past that redaction, and each was rendered on every pymongo release from
    this package's ``>=4.14.0`` floor to 4.18.1 with the exact call :meth:`_resolve` makes:

    * ``tlsCertificateKeyFilePassword='PEMPASSPHRASE'`` — a password by name, printed verbatim by every
      one of them;
    * ``authmechanismproperties={'AWS_SESSION_TOKEN': 'SESSIONTOKENSECRET'}`` — a live STS token, printed
      verbatim through 4.17.0 and rendered ``'<redacted>'`` from 4.18.0 on.

    Both halves of that second line are asserted, because CI installs both: the ``Dependency floors`` job
    resolves the floor, where the token leaks, and the matrix jobs resolve the newest, where it does not.
    A package that supports its own floor cannot treat the newest driver's redaction as the behaviour.

    ``_Connection`` is a dataclass, so its generated ``__repr__`` rendered the client, and the registry's
    values are ``_Connection`` objects, so ``repr(_REGISTRY)`` rendered every one of them. A debug line
    interpolating the connection, a crash reporter walking module globals, or an exception rendering its
    arguments therefore published whichever of these the user configured. ``client`` now carries
    ``field(repr=False)``, which is the smallest fix that keeps the attribute and removes the rendering.

    Reported by a production-readiness review. The class is separate from ``AT-014``'s rather than folded
    into it because the hazard is a different one: not our interpolation of a config we hold, but a
    dependency's ``__repr__`` reached through a dataclass we generate.
    """

    TOKEN: Final[str] = "SESSIONTOKENSECRET"
    PEM: Final[str] = "PEMPASSPHRASE"
    # Port 1 is reserved and never listening. Nothing here makes a call, so this only has to be a URI a
    # client can be built from; `LEAKY_UNREACHABLE` below carries a password, which is the other class's
    # subject and would make a failure here ambiguous about which secret escaped.
    UNREACHABLE: Final[str] = "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50"
    #: The release that began redacting ``AWS_SESSION_TOKEN``, bisected across 4.14.0, 4.15.0, 4.16.0,
    #: 4.17.0, 4.18.0 and 4.18.1 rather than read from a changelog.
    TOKEN_REDACTED_FROM: Final[tuple[int, int]] = (4, 18)

    @classmethod
    def assert_no_secret(cls, *subjects: object) -> None:
        """Assert neither measured secret appears in any of the given renderings.

        Args:
            *subjects: Objects whose ``repr`` must not carry a secret, plus any strings to check as
                themselves. Same contract as ``AT-014``'s :meth:`~TestThePasswordNeverLeavesTheConfig.assert_no_leak`,
                over both measured values, whether or not the installed driver redacts either.

        Raises:
            AssertionError: Naming the leaking subject and which secret leaked, without reproducing the
                value, because this message is read in CI output.
        """
        for index, subject in enumerate(subjects):
            text = subject if isinstance(subject, str) else repr(subject)
            for label, secret in (("session token", cls.TOKEN), ("certificate key password", cls.PEM)):
                assert secret not in text, f"subject {index} ({type(subject).__name__}) leaked the {label}"

    @staticmethod
    def _resolve(**client_args: Any) -> _client._Connection:
        """Resolve a connection whose client was built with the given driver options.

        Args:
            **client_args: Forwarded to ``AsyncMongoClient`` through ``client_args``, unvalidated, which
                is the documented route a user reaches these options by.

        Returns:
            The connection the registry now holds.
        """
        config: Any = {
            "database": "appdb",
            "connection_string": TestTheCredentialsTheUriRedactionDoesNotCover.UNREACHABLE,
            "client_args": client_args,
        }
        return _client.resolve(config)

    def test_the_driver_prints_the_certificate_key_password_which_is_why_this_class_exists(self) -> None:
        """The hazard, asserted rather than remembered, on the object we hand the driver options to.

        This is also the vacuity guard for every test below: if pymongo ever starts redacting this one,
        those tests keep passing while asserting nothing, and this one fails and says why. The
        certificate key password is what the guard rests on rather than the session token, because it is
        the option no release in the supported range redacts;
        :meth:`test_the_session_tokens_redaction_is_the_drivers_and_moves_between_releases` carries the
        one that does.
        """
        connection = self._resolve(
            tlsCertificateKeyFilePassword=self.PEM,
            authMechanism="MONGODB-AWS",
            authMechanismProperties=f"AWS_SESSION_TOKEN:{self.TOKEN}",
        )

        rendered = repr(connection.client)

        assert self.PEM in rendered

    def test_the_session_tokens_redaction_is_the_drivers_and_moves_between_releases(self) -> None:
        """The second credential, whose exposure is a fact about the installed driver rather than about us.

        Written as a bisection result rather than as a single measurement because the suite runs against
        both ends of the supported range in the same workflow, and a premise asserted on one end is a
        test that fails on the other for a reason that is nobody's defect. Whichever way this goes, the
        tests below are unaffected: ``REQ-E-171``'s fix removes our rendering of the client, so the
        driver's choice about this option never reaches a caller through us.
        """
        connection = self._resolve(
            tlsCertificateKeyFilePassword=self.PEM,
            authMechanism="MONGODB-AWS",
            authMechanismProperties=f"AWS_SESSION_TOKEN:{self.TOKEN}",
        )

        rendered = repr(connection.client)

        if pymongo.version_tuple[:2] >= self.TOKEN_REDACTED_FROM:
            assert self.TOKEN not in rendered, "pymongo stopped redacting the session token again"
            assert "<redacted>" in rendered
        else:
            assert self.TOKEN in rendered, "pymongo began redacting the session token below 4.18.0"

    def test_neither_reaches_the_connection_or_the_registry(self) -> None:
        """The outcome ``REQ-E-171`` states, on both renderings the review named.

        The registry is included because it is the worse of the two: it is a module global, so it is what
        a crash reporter walking ``vars(_client)`` renders, and it holds every connection in the process
        rather than the one a frame happens to be looking at.
        """
        connection = self._resolve(
            tlsCertificateKeyFilePassword=self.PEM,
            authMechanism="MONGODB-AWS",
            authMechanismProperties=f"AWS_SESSION_TOKEN:{self.TOKEN}",
        )

        self.assert_no_secret(connection, _client._REGISTRY, *_client._REGISTRY.values())

    def test_a_debug_line_interpolating_the_connection_is_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """The route a leak actually takes, written the way it gets written.

        Nothing in the package logs a connection today. The point is that the next line that does cannot
        publish a credential by doing so, because ``%s`` and ``%r`` of a dataclass are its ``__repr__``.
        """
        connection = self._resolve(tlsCertificateKeyFilePassword=self.PEM)

        with caplog.at_level(logging.DEBUG, logger="strands_mongodb"):
            logging.getLogger("strands_mongodb").debug("connection=%r", connection)

        assert "connection=" in caplog.text
        self.assert_no_secret(caplog.text)

    def test_an_error_that_names_the_connection_carries_no_secret_into_its_traceback(self) -> None:
        """The review's third route: an exception rendering the object it was given.

        The standard renderer prints no locals, so what this asserts is the interpolation rather than the
        frame: a message built from the connection, formatted the way a crash reporter and a log handler
        both format it. That is the shape every error in this package already uses -- ``_target`` and a
        namespace go into the message -- so the next one that reaches for the connection instead is one
        edit away, and this is what makes that edit safe.
        """
        connection = self._resolve(
            authMechanism="MONGODB-AWS", authMechanismProperties=f"AWS_SESSION_TOKEN:{self.TOKEN}"
        )

        try:
            raise StorageError(f"strands_mongodb: pretending something failed while holding {connection!r}")
        except StorageError as error:
            rendered = "".join(traceback.format_exception(error))

        assert "_Connection(" in rendered, "the message did not render the connection, so nothing was checked"
        self.assert_no_secret(rendered)

    def test_the_fields_that_are_not_secrets_still_render(self) -> None:
        """``repr=False`` on the one field, not a hand-written ``__repr__`` that hides the object.

        A dataclass whose ``repr`` says nothing is a debugging regression traded for a security fix, and
        the fields listed here are the ones worth having in a frame: which client owns closing it, which
        loop it belongs to, what the deployment can do, and which collections have been vetted.
        """
        connection = self._resolve()

        rendered = repr(connection)

        assert rendered.startswith("_Connection(")
        assert "client=" not in rendered
        for field_name in ("owned=", "lock=", "loop=", "capabilities=", "binary_keys="):
            assert field_name in rendered, f"{field_name} stopped rendering"

    def test_the_client_is_still_reachable_as_an_attribute(self) -> None:
        """The property ``repr=False`` must not have cost, and the whole module reads it this way."""
        connection = self._resolve()

        assert connection.client is not None
        assert isinstance(_client._Connection.__dataclass_fields__["client"].repr, bool)
        assert _client._Connection.__dataclass_fields__["client"].repr is False


# Port 1 is reserved and never listening, and 50 ms is what makes "cannot connect" fast enough to run in
# the unit tier instead of waiting out pymongo's 30-second default. The shape is ``test_storage.py``'s
# ``UNREACHABLE`` carrying ``LEAKY_URI``'s password, because ``AT-117`` needs both properties at once: a
# config that holds a secret, and calls that reach their own error handling rather than stopping at
# validation.
LEAKY_UNREACHABLE = "mongodb://appuser:SUPERSECRET@127.0.0.1:1/?serverSelectionTimeoutMS=50"


def _assert_no_span_leaks(spans: Sequence[ReadableSpan]) -> None:
    """Assert no span renders the password.

    Args:
        spans: The finished spans to scan.

    Raises:
        AssertionError: Naming the offending span, without reproducing the password: a failure here is
            read in CI output, which is one of the places the password must not reach.
    """
    for span in spans:
        assert "SUPERSECRET" not in rendered(span), f"span {span.name!r} leaked the password"


class TestNoSpanCarriesThePassword:
    """AT-117 / REQ-E-114 — the fourth surface a password can reach, and the only one no test had looked at.

    ``REQ-E-114`` asks that a config holding an inline password survive ``repr()``, ``str()``, log output and OTel
    span attributes. The first three are closed: ``TestThePasswordNeverLeavesTheConfig`` above covers
    construction, the registry, debug logging, exception messages and tracebacks, ``test_storage.py`` and
    ``test_memory_store.py`` extend the sweep across the five storage methods and the four memory methods,
    and both of those pass the store instances themselves as subjects, which is what covers ``repr``. No
    test in the suite had ever created a span.

    Written before the package emitted spans of its own. It now does, through ``_telemetry``, and
    ``test_telemetry.py`` holds the sweep over them; this class still owns the case it was written for, a
    password reaching a span by any route at all.

    It stays a tripwire. The package now records on the span current at each call, and the config with the
    password in it is in scope at every one of those call sites.
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

        The package's spans record an operation name, a duration and a result count.
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


class _RecordingLock:
    """A stand-in for ``_REGISTRY_LOCK`` that says whether it is held and how deeply.

    Wraps a real ``RLock`` rather than faking one, so the mutual exclusion under test is the real thing
    and only the observation is added. Depth is counted here rather than read off the lock because
    ``RLock.locked()`` arrived in Python 3.14 and this package's floor is 3.10, and ``_is_owned()`` is
    private; a wrapper needs neither and works on every version in the matrix.

    Attributes:
        depth: How many times it is currently held by the thread inside it.
        peak: The deepest nesting seen, which is what makes the reentrancy claim testable.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.depth = 0
        self.peak = 0

    def __enter__(self) -> _RecordingLock:
        self._lock.acquire()
        self.depth += 1
        self.peak = max(self.peak, self.depth)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.depth -= 1
        self._lock.release()

    @property
    def held(self) -> bool:
        """Whether it is held right now."""
        return self.depth > 0


class _WatchfulRegistry(dict[Any, Any]):
    """A registry that records whether the lock was held each time it was iterated.

    ``__iter__`` and ``items`` are the two entry points the module uses to walk it, and walking a dict
    another thread is writing is the ``RuntimeError`` half of ``AT-178``. Recording rather than raising,
    because what is being asserted is that every walk happened under the lock, not that any one did.

    Attributes:
        walks: One entry per iteration, ``True`` if the lock was held for it.
    """

    def __init__(self, lock: _RecordingLock) -> None:
        super().__init__()
        self._lock = lock
        self.walks: list[bool] = []

    def __iter__(self) -> Any:
        self.walks.append(self._lock.held)
        return super().__iter__()

    def items(self) -> Any:
        self.walks.append(self._lock.held)
        return super().items()


class _RacingRegistry(dict[Any, Any]):
    """A registry that loses every entry between the snapshot and the removal.

    Stands in for the other thread in the reviewer's reproduction, without needing two threads to line up:
    ``items()`` hands back a list of what was there and then empties the dict, so the caller's snapshot
    names keys that are already gone. A ``del`` over that list is a ``KeyError``; ``pop(key, None)`` is a
    no-op, which is the outcome ``REQ-E-172`` asks for.
    """

    def items(self) -> Any:
        snapshot = list(super().items())
        for key, _ in snapshot:
            super().pop(key, None)
        return snapshot


class _ClosedByHand:
    """The smallest thing ``close()`` will await, recording what the lock was doing while it waited.

    Attributes:
        held_while_closing: Whether ``_REGISTRY_LOCK`` was held at the moment this was awaited.
    """

    def __init__(self, lock: _RecordingLock) -> None:
        self._lock = lock
        self.held_while_closing: bool | None = None

    async def close(self) -> None:
        """Record and suspend, which is what ``endSessions`` does."""
        self.held_while_closing = self._lock.held
        await asyncio.sleep(0)


class TestTheRegistryIsThreadSafeAndForgetsTolerantly:
    """AT-178 / ``REQ-E-172`` — two threads, one process-global dict, and a crash in the wrong place.

    ``REQ-E-017a`` keys the registry per event loop, and the loops a synchronous ``Agent`` creates are the
    reason it has to. What that requirement did not say is which *thread* the loop belongs to. Each loop
    runs on one thread, so nothing on a loop can race another coroutine here — but two threads each
    driving their own loop are not serialised by anything, and the dict they both reach is a module global.

    Two failures followed, and the review reproduced the first with overlapping event-loop resolutions:

    * ``_reap_closed_loops`` snapshotted the entries to drop and then removed them by ``del``. Two threads
      that snapshot the same dead loop leave the second one deleting a key that is gone: ``KeyError``.
    * The snapshot itself walked a dict another thread could be inserting into, which CPython answers with
      ``RuntimeError: dictionary changed size during iteration``.

    Both surfaced inside ``resolve``, on a caller's first use of a connection, about an entry belonging to
    a loop that caller never had. ``close()`` had already reached the ``pop(key, None)`` half of the fix
    for ``REQ-E-017c``, from one loop rather than from two threads; this is the same conclusion drawn
    across threads, plus the lock, which nothing on one loop needs.

    The lock is a ``threading.RLock``, and both of those words are load-bearing. A thread lock, because an
    ``asyncio`` one serialises coroutines on one loop and these callers are on different loops -- and
    because ``resolve`` is synchronous, called from ``MongoDBStorage.__init__``, where a coroutine is not
    available. Reentrant, because ``resolve`` holds it across ``_register``, which takes it too.

    What it must never do is span an ``await``. A thread lock held across a suspension blocks the thread
    the loop runs on, so the coroutine that would release it cannot be resumed by the loop waiting on it:
    a deadlock, in ``close()``, on a teardown path. The last test here is what keeps that closed.
    """

    CONFIG: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

    @staticmethod
    def _dead_loop_key() -> tuple[Any, Any]:
        """Leave one entry behind whose loop has closed, and return its key.

        Built by hand rather than by resolving and returning, so a reap cannot race the test into passing.
        Same idiom as ``AT-091d``'s helper below.

        Returns:
            The key of the entry that was added.
        """
        dead = asyncio.new_event_loop()
        dead.close()
        client: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        identity = _client._identity(TestTheRegistryIsThreadSafeAndForgetsTolerantly.CONFIG)
        key = (identity, dead)
        _client._REGISTRY[key] = _client._Connection(client=client, owned=True)
        return key

    def test_every_walk_of_the_registry_happens_under_the_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``RuntimeError`` half, asserted as the invariant that prevents it rather than as a race.

        A test that runs threads and hopes to catch a dict mid-write is a test that passes on a broken
        build most of the time. What can be asserted deterministically is that no walk happens outside the
        lock, which is the property that makes the interleaving impossible.
        """
        lock = _RecordingLock()
        registry = _WatchfulRegistry(lock)
        monkeypatch.setattr(_client, "_REGISTRY_LOCK", lock)
        monkeypatch.setattr(_client, "_REGISTRY", registry)
        self._dead_loop_key()

        _client.resolve(self.CONFIG)

        assert registry.walks, "the registry was never walked, so this asserted nothing"
        assert all(registry.walks), f"{registry.walks.count(False)} of {len(registry.walks)} walks were unlocked"
        assert lock.depth == 0, "the lock was still held after resolve returned"

    def test_the_lock_is_reentrant_because_resolve_holds_it_across_register(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why an ``RLock`` and not a ``Lock``: a plain one would deadlock this thread against itself.

        The nesting is the thing being asserted. Splitting the two functions so it does not happen would
        put ``_register``'s check and its insert on either side of a release, which is the defect.
        """
        lock = _RecordingLock()
        monkeypatch.setattr(_client, "_REGISTRY_LOCK", lock)

        _client.resolve(self.CONFIG)

        assert lock.peak >= 2, "resolve no longer nests, so the reentrancy this lock pays for is unused"

    def test_a_key_another_thread_already_reaped_is_not_a_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``KeyError`` half, made deterministic: the snapshot is stale by construction.

        The reviewer reproduced this with two worker threads. A registry that empties itself during the
        walk is the same stale snapshot with none of the timing, and it fails against ``del`` every time.
        """
        lock = _RecordingLock()
        monkeypatch.setattr(_client, "_REGISTRY_LOCK", lock)
        monkeypatch.setattr(_client, "_REGISTRY", _RacingRegistry())
        self._dead_loop_key()

        _client._reap_closed_loops()

        assert _client._REGISTRY == {}

    def test_eight_threads_reaping_at_once_raise_nothing(self) -> None:
        """The reviewer's reproduction, and it does reproduce -- reliably, which took one adjustment.

        Written the obvious way it passed on the broken build every time: a reap over a handful of entries
        finishes inside one GIL slice, so the threads never actually interleave and the test asserts that
        two things did not happen at once rather than that they are safe when they do. Two changes fix
        that, and both describe a machine under load rather than a machine in a test:

        * 300 dead entries, so a single reap is long enough to be interrupted part-way;
        * ``sys.setswitchinterval(1e-6)``, so the interpreter preempts inside it.

        Measured on the unfixed build with both in place: six of the eight threads raised, every run, and
        the first exception was ``RuntimeError: dictionary changed size during iteration`` rather than the
        ``KeyError`` the review named. Both are the same defect seen from either side -- one thread walking
        while another writes, one thread deleting what another already deleted -- and the iteration is what
        fails first, because it fails before the removals begin.

        The switch interval is process-global, so it is restored in a ``finally``: leaving it at a
        microsecond makes every threaded test after this one crawl.
        """
        errors: list[BaseException] = []
        start = threading.Barrier(8)
        for _ in range(300):
            self._dead_loop_key()

        def worker() -> None:
            try:
                start.wait(timeout=10)
                for _ in range(30):
                    _client._reap_closed_loops()
            except BaseException as error:  # noqa: BLE001 - the claim is that nothing at all escapes
                errors.append(error)

        original = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
        finally:
            sys.setswitchinterval(original)

        assert errors == [], f"{len(errors)} of 8 threads raised, first: {errors[0]!r}"
        assert not any(thread.is_alive() for thread in threads), "a worker did not finish, which reads as a deadlock"
        assert _client._REGISTRY == {}, "the reap left a dead loop's entry behind"

    def test_close_releases_the_lock_before_it_waits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The deadlock the fix could have introduced, and the reason ``close()`` takes the lock twice.

        ``close()`` snapshots under the lock and pops under the lock, and does neither while awaiting
        ``client.close()``. Holding it across that suspension would block the thread the loop runs on, so
        a second coroutine needing the lock could not be resumed by the loop that is waiting for it. The
        symptom would be a process that never exits, from a ``finally`` and an ``atexit``.
        """
        lock = _RecordingLock()
        monkeypatch.setattr(_client, "_REGISTRY_LOCK", lock)
        client = _ClosedByHand(lock)

        async def body() -> None:
            identity = _client._identity(self.CONFIG)
            loop = asyncio.get_running_loop()
            _client._REGISTRY[(identity, loop)] = _client._Connection(client=client, owned=True)  # type: ignore[arg-type]

            await _client.close(self.CONFIG)

        asyncio.run(body())

        assert client.held_while_closing is False, "the lock was held across an await inside close()"
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


class _ALoopThatClosesBetweenTheCheckAndTheSchedule:
    """A foreign loop that answers ``is_closed()`` with ``False`` and then refuses the schedule.

    Which is what a live loop does when it closes in between -- the window is two statements wide and
    nothing here holds that loop still. Faked rather than raced, because a test that reproduces it by
    timing would reproduce it rarely and would not fail when the fix was removed.

    ``asyncio.run_coroutine_threadsafe`` reaches the loop through exactly one call, so this is the whole
    surface it needs: the ``RuntimeError`` it raises is the one a closed loop raises, and it is not a
    ``PyMongoError``, so nothing in the driver catches it on the way out.
    """

    def __init__(self) -> None:
        self.schedules = 0

    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, *args: Any, **kwargs: Any) -> None:
        self.schedules += 1
        raise RuntimeError("Event loop is closed")


class _CountsItsCloses:
    """A stand-in client whose ``close()`` is a coroutine, so an un-awaited one is visible."""

    def __init__(self) -> None:
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1


class TestALoopClosingMidTeardownDoesNotStrandTheOtherSlots:
    """AT-194 / ``REQ-E-188`` — the one statement in ``close()`` that could raise past the loop it is in.

    ``close()`` releases every slot for an identity (``REQ-E-017a``), and a slot on another live loop is
    closed by scheduling onto that loop. Between the ``is_closed()`` that decides it is worth scheduling
    and the schedule itself, that loop may close. The ``RuntimeError`` is not a ``PyMongoError``, so it
    escapes the driver's handling and this function's: it leaves ``close()`` from inside the loop over
    the remaining keys, and every slot after the failing one stays in the registry for the life of the
    process -- from a path that runs in a ``finally`` and in an ``atexit``, where there is no later call
    to reach them.

    The un-awaited coroutine is the second half. ``client.close()`` has already been called by then, and
    a coroutine nobody awaits raises ``RuntimeWarning`` at collection -- which this suite configures as
    an error, so the failure would surface in whatever test happened to trigger the collection.

    Nothing here claims the client gets closed. It cannot be: closing needs the loop that owns its
    sockets and that loop is gone. This is the closed loop's case arriving one statement earlier, and what
    the fix is about is the slots that are still releasable.
    """

    CONFIG: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

    def test_the_running_loops_slot_is_released_after_the_foreign_schedule_fails(self) -> None:
        """The finding. The dead loop's slot is first in the registry, so it is first in the snapshot."""
        hostile = _ALoopThatClosesBetweenTheCheckAndTheSchedule()
        stranded = _CountsItsCloses()

        async def body() -> None:
            identity = _client._identity(self.CONFIG)
            _client._REGISTRY[(identity, hostile)] = _client._Connection(  # type: ignore[index]
                client=_CountsItsCloses(),  # type: ignore[arg-type]
                owned=True,
            )
            _client._REGISTRY[(identity, asyncio.get_running_loop())] = _client._Connection(
                client=stranded,  # type: ignore[arg-type]
                owned=True,
            )

            await _client.close(self.CONFIG)

        asyncio.run(body())

        assert hostile.schedules == 1, "the schedule was not attempted, so this test proves nothing"
        assert stranded.closes == 1, "a slot after the failing one was never reached"
        assert _client._REGISTRY == {}

    def test_the_coroutine_nobody_can_await_is_closed_rather_than_dropped(self) -> None:
        """The second half, asserted on the warning rather than on a flag.

        Collected explicitly instead of waiting for the garbage collector to do it at some later test's
        expense, which is exactly the shape that made this worth fixing rather than tolerating.
        """
        import gc
        import warnings

        hostile = _ALoopThatClosesBetweenTheCheckAndTheSchedule()

        async def body() -> None:
            identity = _client._identity(self.CONFIG)
            _client._REGISTRY[(identity, hostile)] = _client._Connection(  # type: ignore[index]
                client=_CountsItsCloses(),  # type: ignore[arg-type]
                owned=True,
            )

            await _client.close(self.CONFIG)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(body())
            gc.collect()

        never_awaited = [warning for warning in caught if "never awaited" in str(warning.message)]
        assert never_awaited == [], f"{never_awaited[0].message}" if never_awaited else ""

    def test_a_foreign_loop_that_is_still_open_is_still_scheduled_onto(self) -> None:
        """The vacuity guard. A fix that stopped scheduling altogether would pass both tests above and
        would leak a pool on every loop but the running one, which is what ``REQ-E-017a`` exists for."""
        other = asyncio.new_event_loop()
        thread = threading.Thread(target=other.run_forever, daemon=True)
        thread.start()
        closed = threading.Event()

        class _SignalsWhenClosed(_CountsItsCloses):
            async def close(self) -> None:
                await super().close()
                closed.set()

        client = _SignalsWhenClosed()

        async def body() -> None:
            identity = _client._identity(self.CONFIG)
            _client._REGISTRY[(identity, other)] = _client._Connection(
                client=client,  # type: ignore[arg-type]
                owned=True,
            )

            await _client.close(self.CONFIG)

        try:
            asyncio.run(body())
            assert closed.wait(timeout=5), "the client was never closed on the loop that owns it"
        finally:
            other.call_soon_threadsafe(other.stop)
            thread.join(timeout=5)
            other.close()

        assert client.closes == 1
        assert _client._REGISTRY == {}
