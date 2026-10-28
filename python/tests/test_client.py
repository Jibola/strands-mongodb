"""Connection resolution, ownership and lifecycle.

Everything here is about the client, not about queries, so almost none of it needs a database.
``AsyncMongoClient`` construction performs no I/O and validates nothing (on pymongo 4.17.0,
``AsyncMongoClient("not-a-uri-at-all")`` succeeds, reading the string as a hostname), which is what
lets the ownership, registry and attribution tests run in the unit tier at full speed.
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

    The registry is process-global by design, so that one connection identity shares one client,
    which makes it shared mutable state between tests. Clearing it before *and* after means a test
    never inherits a client another test created, and a test that leaves one behind cannot make a
    later assertion about identity pass for the wrong reason.

    Reaching into ``_REGISTRY`` is deliberate. The alternative is a ``reset_registry()`` helper in
    ``_client.py``, which would be a function shipped to users that exists only for this fixture. A
    private module's private dict is the clearer seam: it reads as a test touching internals.
    """
    _client._REGISTRY.clear()
    yield
    _client._REGISTRY.clear()


class TestConstructionRequiresAWayToReachTheDatabase:
    """AT-015 — the error that has to arrive before anything else does.

    Nothing upstream raises here. On pymongo 4.17.0 a string with no scheme is read as a hostname and
    an unresolvable host is not an error until an operation runs, so without this check a config
    missing its connection string surfaces as a server-selection timeout against ``localhost`` some
    seconds later, in a stack frame that names neither field.
    """

    def test_neither_client_nor_connection_string_raises_value_error(self) -> None:
        """A config with neither ``client`` nor ``connection_string`` raises ``ValueError``."""
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
        """The rejection happens before any I/O, asserted through the registry rather than by timing.

        A rejected config that still left a registry entry behind would mean the constructor got far
        enough to build something.
        """
        with pytest.raises(ValueError):
            _client.resolve({"database": "appdb"})

        assert _client._REGISTRY == {}


class TestOwnershipDecidesWhoMayClose:
    """AT-010, AT-011 — a client the caller injected is never closed; one the package built is.

    In an agent process a single ``AsyncMongoClient`` is typically shared across memory, storage and the
    application's own queries, so a ``close()`` that closed an injected client unconditionally would tear
    down a pool three unrelated subsystems are still using. The failure would not show up in the closing
    component, but somewhere else, later.
    """

    async def test_an_injected_client_survives_our_close(self) -> None:
        """An injected client is never closed, including by the store's own ``close()``."""
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
        missing-connection message with a second copy of itself thrown from teardown.
        """
        await _client.close({"database": "appdb"})

        assert _client._REGISTRY == {}

    async def test_a_client_we_constructed_is_closed_by_our_close(self) -> None:
        """A client the package constructed is closed by its ``close()``, through the same ownership flag.

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
        """When both are set, the injected ``client`` wins over ``connection_string``.

        Both fields set is not an error, and it is not far-fetched: a config assembled from a
        template's defaults plus a caller's override arrives exactly this way. The client the caller
        handed over is the one they expect their queries to run on.
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
    """AT-012 — one client per connection identity, and why that matters.

    A ``Storage`` reaches five SDK subsystems and a ``MemoryStore`` is registered per memory surface,
    so one agent can construct a double-digit number of these from the same config. Each
    ``AsyncMongoClient`` carries its own connection pool, defaulting to 100 sockets, plus a topology
    monitor per server. Without sharing, a config that reads like one database connection becomes
    however many the wiring happened to produce.
    """

    def test_a_hundred_resolutions_of_one_config_yield_one_client(self) -> None:
        """Asserted on object identity rather than on a cache-hit counter."""
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
        would pass.
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
        """``client_args`` is part of the identity.

        Leaving it out would be a silent misconfiguration rather than a crash: the second config would
        hit the cache, its arguments would be discarded, and the caller would get a client with someone
        else's timeouts. Asserted on the timeout that reaches ``client.options``, not only on object
        identity, because sharing is the failure and the wrong timeout is the symptom a user would see.
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
        """Injected clients key on ``id(client)``.

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
    """AT-013a, AT-013b — every client carries ``DriverInfo(name="Strands-MongoDB", version=...)``.

    Attribution is how MongoDB counts Strands usage in server logs and Atlas telemetry, so a silent
    failure here would be invisible to MongoDB and to the user both. The version is the installed
    package version.

    ``pymongo_search_utils.append_client_metadata`` does the same job, but it lives in the ``[memory]``
    extra, and ``_client.py`` must also work under a ``[storage]``-only install. Its entire body is a
    call to ``client.append_metadata``, so these tests assert that call and the metadata it produces.
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

        Overwriting it would misreport the driver to the server, making a Strands install look like it
        is not using PyMongo at all.
        """
        connection = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        name = connection.client.options.pool_options.metadata["driver"]["name"]

        assert name.startswith("PyMongo")
        assert name.endswith("|Strands-MongoDB")

    def test_the_driver_name_is_exactly_strands_mongodb(self) -> None:
        """AT-230 — the name is a data key, not a label, so it is pinned exactly.

        Every assertion above uses ``in`` or ``endswith``, which all pass on ``StrandsMongoDB`` or
        ``strands``. Server logs and Atlas telemetry group by this string, so changing it would orphan
        every record collected under the old one.

        The ``DriverInfo`` type is asserted alongside it because ``append_metadata`` raises ``TypeError``
        on anything else, and the injected path passes it through unexamined.
        """
        info = _client._driver_info()

        assert isinstance(info, DriverInfo)
        assert info.name == "Strands-MongoDB"

    def test_the_reported_version_is_the_installed_package_version(self) -> None:
        """A version pinned to a literal would report the release it was typed in, forever."""
        from importlib.metadata import version

        connection = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        versions = connection.client.options.pool_options.metadata["driver"]["version"]

        assert versions.endswith(version("strands-mongodb"))

    def test_an_injected_client_is_attributed_through_append_metadata(self) -> None:
        """AT-013b — the second path, which is the only one available for a client the package did not build."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")

        _client.resolve({"database": "appdb", "client": injected})

        assert injected.options.pool_options.metadata["driver"]["name"].endswith("|Strands-MongoDB")

    def test_repeated_attribution_of_one_client_is_idempotent(self) -> None:
        """The metadata a caller ends up holding, whichever layer kept it to one entry.

        Two layers can keep it there and this test cannot tell them apart, which is why
        :class:`TestAttributionHappensAtMostOncePerClient` counts the calls instead. From pymongo 4.14.1
        on, ``PoolOptions._update_metadata`` returns early when the name alone is already in the
        pipe-delimited chain; 4.14.0, the declared floor, does not. ``_already_attributed`` checks a
        layer up, on every version, and treats an entry as a duplicate only when name and version both
        match.

        This asserts the outcome rather than the mechanism. Resolving one config repeatedly is the
        normal case, and the driver name the caller's client carries names Strands once.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        for _ in range(5):
            _client.resolve(config)

        name = injected.options.pool_options.metadata["driver"]["name"]

        assert name.count(_client._DRIVER_NAME) == 1

    def test_the_installed_pymongo_implements_append_metadata_as_a_method(self) -> None:
        """AT-013b's second half — the floor is what lets the call be unguarded.

        Below pymongo 4.14 ``append_metadata`` is not a method. It resolves through
        ``AsyncMongoClient.__getattr__`` to a ``Database`` of that name, which is truthy and not
        callable, so ``hasattr`` says yes and calling it raises ``TypeError``. That is why the declared
        floor is ``>=4.14.0`` rather than a runtime probe with a warning for the caller.

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


class TestAppNameIsTheApplications:
    """AT-231 — the package never sets, defaults or rewrites ``appName``.

    The handshake specification makes ``client.application.name`` the application's own field. A wrapping
    library is attributed in ``driver.name`` and ``driver.version`` instead, which server logs and
    ``$currentOp`` show as ``clientMetadata.driver``. Every assertion reads the value pymongo parsed, because
    the URI, ``client_args`` and their precedence all meet inside pymongo.
    """

    def test_no_app_name_is_sent_when_none_is_configured(self) -> None:
        connection = _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        assert _app_name_of(connection.client) is None

    @pytest.mark.parametrize("query", ["?appName=billing", "?APPNAME=billing", "?retryWrites=true&appName=billing"])
    def test_a_uri_app_name_is_sent_exactly_as_given(self, query: str) -> None:
        connection = _client.resolve({"database": "appdb", "connection_string": f"mongodb://127.0.0.1:1/{query}"})

        assert _app_name_of(connection.client) == "billing"

    def test_a_client_args_app_name_is_sent_exactly_as_given(self) -> None:
        config: MongoDBConfig = {
            "database": "appdb",
            "connection_string": "mongodb://127.0.0.1:1/",
            "client_args": {"appname": "billing", "maxPoolSize": 7},
        }
        connection = _client.resolve(config)

        assert _app_name_of(connection.client) == "billing"
        assert connection.client.options.pool_options.max_pool_size == 7

    def test_an_injected_client_keeps_its_own_app_name(self) -> None:
        mine: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/?appName=mine")
        bare: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")

        assert _app_name_of(_client.resolve({"database": "appdb", "client": mine}).client) == "mine"
        assert _app_name_of(_client.resolve({"database": "appdb", "client": bare}).client) is None

    @pytest.mark.contract
    async def test_the_server_sees_the_driver_entry_and_the_applications_name(self, contract_uri: str) -> None:
        """AT-231 against a real server: its own record of this connection's handshake.

        ``$currentOp`` lists the aggregation that is asking, with the ``clientMetadata`` its connection sent.
        A deployment that refuses ``$currentOp`` to this user cannot answer, so it skips.
        """
        separator = "&" if "?" in contract_uri else "?"
        config: MongoDBConfig = {"database": "admin", "connection_string": f"{contract_uri}{separator}appName=billing"}
        connection = _client.resolve(config)
        try:
            try:
                cursor = await connection.client.admin.aggregate(
                    [{"$currentOp": {}}, {"$match": {"appName": "billing"}}]
                )
                ops = await cursor.to_list()
            except pymongo.errors.OperationFailure as error:
                pytest.skip(f"$currentOp is not available to this user: {error.code}")
        finally:
            await _client.close(config)

        assert ops, "the server recorded no operation from a connection whose appName is exactly 'billing'"
        driver = ops[0]["clientMetadata"]["driver"]
        assert driver["name"].split("|")[-1] == _client._DRIVER_NAME
        assert driver["version"].split("|")[-1] == _client._package_version()


class TestAttributionHappensAtMostOncePerClient:
    """``AT-190`` -- the package appends its attribution to an injected client at most once.

    :func:`_client._attribute_injected_client` runs once per registry slot, and ``close`` releases the
    slot, so a caller who closes a store and resolves the same client again reaches the append again.
    From pymongo 4.14.1 on, ``PoolOptions._update_metadata`` drops an append whose name alone is already
    in the pipe-delimited chain, compared case-insensitively. 4.14.0, the declared floor that the
    ``Dependency floors`` job installs, has no such check: each cycle appends another name and version to
    the handshake document until it passes the server's 512-byte client metadata limit, and every new
    connection then fails with ``OperationFailure``.

    So the guard is this package's own. Following the MongoDB handshake specification, an existing entry
    is a duplicate only when its name and version both match this package's, compared case-sensitively.
    These tests count the calls *this package* makes rather than reading the chain, because on 4.14.1 and
    later the driver's own dedupe makes the chain look the same either way.
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
        """AT-190 -- the close-and-resolve cycle, run three times, appends once.

        Three cycles are enough because the failure this asserts is the second append: the server's
        limit is where an unbounded chain is finally noticed, but the defect is that the chain grows at
        all. Removing the guard fails this on every supported pymongo.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        appended = self._record_appends(monkeypatch)
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        for _ in range(3):
            _client.resolve(config)
            await _client.close(config)

        assert appended == [_client._DRIVER_NAME], f"attributed {len(appended)} times across three cycles: {appended}"
        assert injected.options.pool_options.metadata["driver"]["name"].count(_client._DRIVER_NAME) == 1

    def test_a_client_already_carrying_our_exact_entry_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-190 -- an entry the caller appended with this package's name and version counts as this package's.

        A caller who runs their own ``append_metadata`` with this package's ``DriverInfo`` reaches
        ``resolve`` with the chain already carrying the entry, so the package does not append it again.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(_client._driver_info())
        appended = self._record_appends(monkeypatch)

        _client.resolve({"database": "appdb", "client": injected})

        assert appended == [], f"attributed a client that already carried this package's entry: {appended}"

    def test_our_name_with_another_version_is_a_different_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-232 -- per the handshake specification, an entry is a duplicate only when every field is equal.

        So the package offers the append. pymongo 4.14.1 and later then drop it, because the driver's own
        dedupe compares the name alone; that is the driver's behaviour, and the chain is not asserted here.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(DriverInfo(_client._DRIVER_NAME, "0.0.0-another"))
        appended = self._record_appends(monkeypatch)

        _client.resolve({"database": "appdb", "client": injected})

        assert appended == [_client._DRIVER_NAME], f"another version was read as this one: {appended}"

    def test_the_comparison_is_case_sensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-232 -- the handshake specification compares fields with case-sensitive equality."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(DriverInfo(_client._DRIVER_NAME.lower(), _client._package_version()))
        appended = self._record_appends(monkeypatch)

        _client.resolve({"database": "appdb", "client": injected})

        assert appended == [_client._DRIVER_NAME], f"a lowercase entry was read as this package's: {appended}"

    def test_a_longer_name_that_starts_with_ours_is_not_ours(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AT-190 -- the comparison splits on the delimiter and matches whole entries.

        A substring test would read ``Strands-MongoDBToolkit`` as this package and skip the append, losing
        the attribution. Matching whole entries means this client is still attributed.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(DriverInfo(f"{_client._DRIVER_NAME}Toolkit", _client._package_version()))
        appended = self._record_appends(monkeypatch)

        _client.resolve({"database": "appdb", "client": injected})

        assert appended == [_client._DRIVER_NAME], f"a different driver's name was read as ours: {appended}"
        assert injected.options.pool_options.metadata["driver"]["name"].endswith(
            f"{_client._DRIVER_NAME}Toolkit|{_client._DRIVER_NAME}"
        )

    def test_entries_are_paired_from_the_right(self) -> None:
        """AT-232. pymongo's own name has three entries and its version one, so pairing from the left would
        match this package's name against another library's version."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        injected.append_metadata(DriverInfo("OtherLibrary", _client._package_version()))
        injected.append_metadata(DriverInfo(_client._DRIVER_NAME, "0.0.0-another"))

        assert _client._already_attributed(injected) is False

        injected.append_metadata(DriverInfo("ThirdLibrary", "1.0"))
        bare: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        bare.append_metadata(_client._driver_info())
        bare.append_metadata(DriverInfo("ThirdLibrary", "1.0"))

        assert _client._already_attributed(bare) is True


class TestAttributionThatSrvResolutionDiscardsIsReapplied:
    """AT-195 — an injected client's attribution that SRV resolution discards is applied again.

    ``append_metadata`` writes into the client's ``ClientOptions``. Resolving a ``mongodb+srv://`` URI
    assigns a **new** ``ClientOptions`` and rebuilds the topology settings from it, and that happens on the
    client's first connection rather than at construction, so for an injected SRV client the append lands
    before the resolution and is discarded by it. Without a re-check on the next resolve, the loss would be
    permanent.

    No database operation behaves differently either way. What attribution supports is MongoDB's usage
    counting, and ``mongodb+srv://`` is what Atlas hands out.

    The re-check is bounded in both directions. It runs only for injected clients, only on the
    registry-hit path, and it is a string split over metadata already in memory. And it stops: an append
    that does not become visible clears the flag, so a driver that ignores ``append_metadata`` costs one
    extra append rather than unbounded chain growth.
    """

    @staticmethod
    def _lose_the_attribution(client: AsyncMongoClient[Any]) -> None:
        """Strip this package's entry from both chains in place, which is what the new ``ClientOptions`` amounts to.

        Written on the metadata rather than by re-running SRV resolution, so this half of the class stays
        a test of the branch. The test below runs the real resolution and asserts the loss is real.
        """
        driver = client.options.pool_options.metadata["driver"]
        names, versions = driver["name"].split("|"), driver["version"].split("|")
        if names[-1] == _client._DRIVER_NAME:
            names.pop()
            versions.pop()
        driver["name"], driver["version"] = "|".join(names), "|".join(versions)
        # pymongo 4.19 also dedupes on a private set of appended drivers, which a new ``ClientOptions`` starts empty.
        getattr(client.options.pool_options, "_PoolOptions__appended_drivers", set()).clear()

    def test_an_attribution_that_was_discarded_is_applied_again_on_the_next_resolve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second resolve is a registry hit, and it still checks for and restores a lost attribution."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}
        _client.resolve(config)
        self._lose_the_attribution(injected)
        appended = TestAttributionHappensAtMostOncePerClient._record_appends(monkeypatch)

        _client.resolve(config)

        assert appended == [_client._DRIVER_NAME], f"the hit path did not re-apply a lost attribution: {appended}"
        assert _client._already_attributed(injected) is True

    def test_a_client_that_kept_it_is_not_appended_to_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The re-check keeps the at-most-once guarantee: it reads before it appends.

        Ten hits on a client that kept its attribution, which on the 4.14.0 floor would be ten entries in
        the handshake document if the re-check appended unconditionally.
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
        """The bound that keeps the re-check safe.

        A driver whose ``append_metadata`` does nothing observable is not a driver to keep calling: the
        chain may not be visible here and still be growing, as it would on 4.14.0, once per resolve.
        Five resolves, two appends, and the second one is what proves the first was not simply skipped.
        """
        appended: list[str] = []

        def refuse(self: AsyncMongoClient[Any], driver_info: DriverInfo) -> None:
            appended.append(driver_info.name)

        monkeypatch.setattr(AsyncMongoClient, "append_metadata", refuse)
        injected: AsyncMongoClient[Any] = AsyncMongoClient("mongodb://127.0.0.1:1/")
        config: MongoDBConfig = {"database": "appdb", "client": injected}

        for _ in range(5):
            _client.resolve(config)

        assert appended == [_client._DRIVER_NAME, _client._DRIVER_NAME], f"the re-check did not stop: {appended}"

    async def test_srv_resolution_really_does_discard_it_and_the_next_resolve_recovers_it(self) -> None:
        """The premise, against pymongo's own SRV path with only the DNS lookup replaced.

        Everything above is a test of the branch, and every one of them would pass against a package
        whose attribution was never lost. The loss is the premise, and it belongs to the driver rather
        than to this package. So this runs the real ``_resolve_srv`` on a real client, with ``_parse_srv``
        returning what a lookup would have returned, and asserts both halves: that the append is gone
        afterwards, and that resolving the client again brings it back.

        On pymongo 4.17.0 the name chain is ``PyMongo|c|async|Strands-MongoDB`` before and
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
            "SRV resolution kept the attribution, so either the driver changed or this test is "
            "describing something that no longer happens"
        )

        _client.resolve(config)

        assert _client._already_attributed(injected) is True
        assert injected.options.pool_options.metadata["driver"]["name"].endswith(_client._DRIVER_NAME)


class TestRetryWritesDisabledIsWorthSaying:
    """AT-016 — disabled ``retryWrites`` gets a warning rather than a refusal.

    The upsert shapes in ``storage.py`` are idempotent precisely so that the driver's own retry can
    replay them. Turning ``retryWrites`` off does not corrupt anything; it removes the layer that
    absorbs a primary stepdown, and the write surfaces to the caller instead. That is a legitimate
    choice, so this warns and proceeds.

    Read from ``client.options.retry_writes`` rather than parsed out of the connection string, so it
    catches the setting whether it came from the URI, from ``client_args``, or from a client the caller
    injected, and the package ships no URI parser.
    """

    def test_a_connection_string_disabling_retries_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The URI route."""
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
        """The ``client_args`` route."""
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
        attribution it also skips is idempotent anyway (see
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
        """A warning every user sees on a correct config is noise, and it teaches them to filter the package out."""
        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            _client.resolve({"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"})

        assert caplog.records == []


class TestThePasswordNeverLeavesTheConfig:
    """AT-014 — a password in the connection string never reaches a log, a repr, an error or a traceback.

    ``MongoDBConfig`` is a plain ``dict`` holding a URI with an inline password. Ordinary Python habits
    would expose it: an f-string over the config in a debug line, a ``KeyError`` naming the key it could
    not find, a dataclass ``__repr__`` that includes the field, a traceback capturing locals. None of
    those is written on purpose, which is why the guarantee is a test.

    pymongo already handles its own part, so this asserts only what is the package's: ``repr`` of an
    ``AsyncMongoClient`` built from ``LEAKY_URI`` prints ``host=['db.example.com:27017']`` and no
    password, and an invalid URI option raises no exception that could carry it.

    This class covers the client module. ``test_storage.py`` and ``test_memory_store.py`` run the same
    sweep across the store methods.
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
        """DEBUG, because that is the level whose purpose is saying more than usual.

        Redacting at ``info`` while interpolating the config at ``debug`` would not be enough. Debug
        logging is enabled when something is already going wrong, which is when the logs get pasted
        into a ticket.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": LEAKY_URI}

        with caplog.at_level(logging.DEBUG, logger="strands_mongodb"):
            _client.resolve(config)

        self.assert_no_leak(caplog.text)

    def test_a_leaky_config_that_also_disables_retries_redacts_the_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The one log line ``_client.py`` is *required* to emit, on a config that carries a secret.

        The retry warning and the credential guarantee meet here, and the natural way to write the
        warning would be to include the connection string so the reader knows which client it is about.
        """
        config: MongoDBConfig = {"database": "appdb", "connection_string": f"{LEAKY_URI}?retryWrites=false"}

        with caplog.at_level(logging.DEBUG, logger="strands_mongodb"):
            _client.resolve(config)

        assert "retryWrites" in caplog.text
        self.assert_no_leak(caplog.text)

    def test_nothing_the_registry_holds_renders_the_password(self) -> None:
        """The registry key is derived from the connection string, and never renders it.

        A ``dict`` keyed on a URI would expose it through ``repr(registry)``, in a ``KeyError``, and in any
        debugger or crash reporter that walks module globals. The one-client-per-identity semantics
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
        inside the package's own call. pymongo's own message does not echo the URI (``bogusOption``
        yields ``Unknown option: bogusOption``), so this tests the package's handling: a wrapper that
        added context by interpolating the config would expose the password.
        """
        config: Any = {"database": "appdb", "connection_string": LEAKY_URI, "client_args": {"bogusOption": 1}}

        with pytest.raises(Exception) as caught:  # noqa: PT011 - the type is pymongo's to choose
            _client.resolve(config)

        self.assert_no_leak(str(caught.value), caught.value)

    def test_the_traceback_of_a_failed_construction_holds_no_password(self) -> None:
        """Locals travel with a traceback, which is what makes this hard to get right by inspection.

        Any frame between the caller and the raise that binds the URI to a name puts it in the
        traceback, and from there into every crash reporter the user runs with locals enabled.

        The ``else`` clause makes sure the ``except`` body runs: with a config nothing rejects, the test
        would pass while asserting nothing at all.
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
    """AT-177 — credentials passed as driver options never reach a repr of the connection or registry.

    ``AsyncMongoClient.__repr__`` prints every non-default option the client was built with, and the
    password in the URI is the one value it always redacts. Two other options carry a secret, and on the
    supported pymongo range (``>=4.14.0``) the driver renders them with the exact call :meth:`_resolve`
    makes:

    * ``tlsCertificateKeyFilePassword='PEMPASSPHRASE'``, a password by name, printed verbatim by every
      release in the range;
    * ``authmechanismproperties={'AWS_SESSION_TOKEN': 'SESSIONTOKENSECRET'}``, a live STS token, printed
      verbatim before 4.18.0 and rendered ``'<redacted>'`` from 4.18.0 on.

    Both behaviours of the token are asserted, because CI installs both: the ``Dependency floors`` job
    resolves the floor and the matrix jobs resolve the newest driver. The package supports its floor, so
    the newest driver's redaction is not something it can rely on.

    ``_Connection`` is a dataclass and the registry's values are ``_Connection`` objects, so a generated
    ``__repr__`` that included the client would carry these secrets into a debug line interpolating the
    connection, a crash reporter walking module globals, or an exception rendering its arguments. The
    ``client`` field carries ``field(repr=False)``, which keeps the attribute and removes the rendering.

    This is separate from :class:`TestThePasswordNeverLeavesTheConfig` because the route is different:
    not the package interpolating a config it holds, but a dependency's ``__repr__`` reached through a
    dataclass the package defines.
    """

    TOKEN: Final[str] = "SESSIONTOKENSECRET"
    PEM: Final[str] = "PEMPASSPHRASE"
    # Port 1 is reserved and never listening. Nothing here makes a call, so this only has to be a URI a
    # client can be built from; `LEAKY_UNREACHABLE` below carries a password, which is the other class's
    # subject and would make a failure here ambiguous about which secret escaped.
    UNREACHABLE: Final[str] = "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50"
    #: The first pymongo release that redacts ``AWS_SESSION_TOKEN`` in the client's ``repr``.
    TOKEN_REDACTED_FROM: Final[tuple[int, int]] = (4, 18)

    @classmethod
    def assert_no_secret(cls, *subjects: object) -> None:
        """Assert neither secret appears in any of the given renderings.

        Args:
            *subjects: Objects whose ``repr`` must not carry a secret, plus any strings to check as
                themselves. Same contract as :meth:`TestThePasswordNeverLeavesTheConfig.assert_no_leak`,
                over both values, whether or not the installed driver redacts either.

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
        """The driver's own ``repr`` of the client carries the certificate key password.

        This is the vacuity guard for every test below: if pymongo ever starts redacting this option,
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
        """The session token's exposure depends on the installed driver, not on this package.

        Asserted per release because the suite runs against both ends of the supported range in the same
        workflow. Whichever way this goes, the tests below are unaffected: the connection never renders
        its client, so the driver's choice about this option never reaches a caller through the package.
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
        """Neither secret appears in the repr of the connection or of the registry.

        The registry matters most: it is a module global, so it is what
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
        """A debug line that interpolates the connection carries no credential.

        Nothing in the package logs a connection. Any line that does still cannot expose a credential,
        because ``%s`` and ``%r`` of a dataclass are its ``__repr__``.
        """
        connection = self._resolve(tlsCertificateKeyFilePassword=self.PEM)

        with caplog.at_level(logging.DEBUG, logger="strands_mongodb"):
            logging.getLogger("strands_mongodb").debug("connection=%r", connection)

        assert "connection=" in caplog.text
        self.assert_no_secret(caplog.text)

    def test_an_error_that_names_the_connection_carries_no_secret_into_its_traceback(self) -> None:
        """An exception whose message renders the connection carries no credential.

        The standard renderer prints no locals, so what this asserts is the interpolation rather than the
        frame: a message built from the connection, formatted the way a crash reporter and a log handler
        both format it. Errors in this package already put ``_target`` and a namespace into the message,
        and an error that rendered the connection would be just as safe.
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

        A dataclass whose ``repr`` said nothing would be harder to debug, and the fields listed here are
        the ones worth having in a frame: which client owns closing it, which
        loop it belongs to, what the deployment can do, and which collections have been vetted.
        """
        connection = self._resolve()

        rendered = repr(connection)

        assert rendered.startswith("_Connection(")
        assert "client=" not in rendered
        for field_name in ("owned=", "lock=", "loop=", "capabilities=", "binary_keys="):
            assert field_name in rendered, f"{field_name} stopped rendering"

    def test_the_client_is_still_reachable_as_an_attribute(self) -> None:
        """``repr=False`` keeps the attribute, which the whole module reads."""
        connection = self._resolve()

        assert connection.client is not None
        assert isinstance(_client._Connection.__dataclass_fields__["client"].repr, bool)
        assert _client._Connection.__dataclass_fields__["client"].repr is False


# Port 1 is reserved and never listening, and 50 ms is what makes "cannot connect" fast enough to run in
# the unit tier instead of waiting out pymongo's 30-second default. The shape is ``test_storage.py``'s
# ``UNREACHABLE`` carrying ``LEAKY_URI``'s password, because ``TestNoSpanCarriesThePassword`` needs both
# properties at once: a config that holds a secret, and calls that reach their own error handling rather
# than stopping at validation.
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
    """AT-117 — a password in the config never reaches an OpenTelemetry span.

    A config holding an inline password must not expose it through ``repr()``, ``str()``, log output or
    span attributes. ``TestThePasswordNeverLeavesTheConfig`` above covers construction, the registry, debug
    logging, exception messages and tracebacks; ``test_storage.py`` and ``test_memory_store.py`` extend
    the sweep across the storage and memory methods and pass the store instances themselves as subjects,
    which covers ``repr``. This class covers spans.

    ``test_telemetry.py`` checks the package's own spans attribute by attribute. This class checks that
    the password reaches no span by any route at all, which matters because the package records on the
    span current at each call, and the config with the password in it is in scope at every one of those
    call sites.
    """

    @pytest.mark.parametrize("surface", ["name", "attribute", "event", "status"])
    def test_the_scan_catches_a_span_leaking_on_any_surface(
        self, surface: str, exported_spans: InMemorySpanExporter
    ) -> None:
        """The positive control, one span per surface, each leaking in exactly one place.

        Without this the test below could be vacuous: a scan that read the wrong field, or an exporter
        that received nothing, would pass just as well as a clean package.
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
        failing on every call afterwards is a common shape, and a failure is when an error message is
        most likely to name the connection it was talking about.

        The span count is not asserted, because what this test is about is what a span carries.

        Six calls rather than all nine public methods: a call against a closed port cannot fail faster
        than pymongo's minimum heartbeat interval of 0.5s, whatever ``serverSelectionTimeoutMS`` asks
        for. So the list keeps one call per distinct path, a find, an upsert, an aggregation, the index
        path and both memory directions, and drops ``delete``, ``list`` and ``delete_prefix``, which are
        variants of paths already here.
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

        The package's spans record an operation name, a duration and a result count. Two of those three
        are only known once a call has succeeded, so the code that records them is only reached here.

        Same recipe as ``test_memory_store.py``'s succeeding half: an injected client that reaches the real
        deployment, alongside a ``connection_string`` that still holds a secret. Nothing else produces a
        config that both works and carries a password, and that combination is what this test needs.

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
            # `close()` never closes an injected client, which is the asymmetry that lets this config hold
            # a secret at all.
            await client.close()

        finished = exported_spans.get_finished_spans()
        assert finished, "no span was exported, so this test proved nothing"
        _assert_no_span_leaks(finished)


class TestTheLockFollowsTheEventLoop:
    """AT-017 — the connection's lock works across successive event loops.

    A synchronous ``Agent`` runs each invocation on a fresh event loop, so a lock created during the
    first invocation is asked to work during the second on a loop that no longer exists.

    Two properties of ``asyncio.Lock`` shape these tests (CPython 3.14):

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
        """A shared config works across two real ``asyncio.run`` calls.

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

        Both resolutions here run with no loop running, which is why they still share one entry: the
        registry is split per running loop, and neither of these calls has one. The entry a caller
        shares across loops is the loop-free slot, not a loop's own.
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
    """AT-091 — each event loop gets its own client for a connection identity.

    ``TestTheLockFollowsTheEventLoop`` covers the lock. The client is the other per-loop object and the
    one that raises: on pymongo 4.17.0, ``AsyncMongoClient`` records the running loop on its first
    operation and every later operation from a different loop raises
    ``RuntimeError: Cannot use AsyncMongoClient in different event loop``.

    Tests that build a store and close it within one loop cannot see this, because closing pops the
    registry entry before the next loop asks for one. The failure needs a store that **outlives** one
    loop, which is what a synchronous ``Agent`` is.

    These assertions are on the registry rather than on a write, because the observable difference is
    which client comes back. ``test_storage.py`` makes the same claim against a real deployment, where
    the wrong answer is a ``RuntimeError`` instead of a failed identity check.
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
        """The registry stays bounded. Ten loops must not leave ten entries in a process-global dict."""
        for _ in range(10):
            self._resolve_on_a_fresh_loop(self.CONFIG)

        assert len(_client._REGISTRY) == 1, sorted(str(key) for key in _client._REGISTRY)

    def test_one_loop_still_shares_one_client_across_many_resolutions(self) -> None:
        """One loop still shares one client, which is the thing a per-loop key could plausibly break.

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
        """An injected client is never copied: the package cannot build a second copy of a client it was handed.

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
    another thread is writing raises ``RuntimeError``. Recording rather than raising,
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

    Stands in for a second thread reaping the same entries, without needing two threads to line up:
    ``items()`` hands back a list of what was there and then empties the dict, so the caller's snapshot
    names keys that are already gone. A ``del`` over that list is a ``KeyError``; ``pop(key, None)`` is a
    no-op, which is the required outcome.
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
    """AT-178 — the registry is safe to use from several threads at once.

    The registry is keyed per event loop, because a synchronous ``Agent`` creates loops of its own. Each
    loop runs on one thread, so nothing on a loop can race another coroutine here, but two threads each
    driving their own loop are not serialised by anything, and the dict they both reach is a module global.

    Without a lock and tolerant removal, two failures are possible, both inside ``resolve`` on a caller's
    first use of a connection, about an entry belonging to a loop that caller never had:

    * Two threads that snapshot the same dead loop's entry and remove it by ``del`` leave the second one
      deleting a key that is gone: ``KeyError``. Removal is ``pop(key, None)``.
    * A snapshot that walks a dict another thread is inserting into gets
      ``RuntimeError: dictionary changed size during iteration``. Every walk happens under the lock.

    The lock is a ``threading.RLock``. A thread lock, because an ``asyncio`` one serialises coroutines on
    one loop and these callers are on different loops, and because ``resolve`` is synchronous, called
    from ``MongoDBStorage.__init__``, where a coroutine is not available. Reentrant, because ``resolve``
    holds it across ``_register``, which takes it too.

    It must never span an ``await``. A thread lock held across a suspension blocks the thread the loop
    runs on, so the coroutine that would release it cannot be resumed by the loop waiting on it: a
    deadlock, in ``close()``, on a teardown path. The last test here guards against that.
    """

    CONFIG: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

    @staticmethod
    def _dead_loop_key() -> tuple[Any, Any]:
        """Leave one entry behind whose loop has closed, and return its key.

        Built by hand rather than by resolving and returning, so a reap cannot race the test into passing.
        Same idiom as ``TestConcurrentClosesOnOneIdentity``'s helper below.

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
        put ``_register``'s check and its insert on either side of a release, which would be a race.
        """
        lock = _RecordingLock()
        monkeypatch.setattr(_client, "_REGISTRY_LOCK", lock)

        _client.resolve(self.CONFIG)

        assert lock.peak >= 2, "resolve no longer nests, so the reentrancy this lock pays for is unused"

    def test_a_key_another_thread_already_reaped_is_not_a_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``KeyError`` half, made deterministic: the snapshot is stale by construction.

        A registry that empties itself during the walk is the stale snapshot two racing threads would
        produce, with none of the timing, and it fails against ``del`` every time.
        """
        lock = _RecordingLock()
        monkeypatch.setattr(_client, "_REGISTRY_LOCK", lock)
        monkeypatch.setattr(_client, "_REGISTRY", _RacingRegistry())
        self._dead_loop_key()

        _client._reap_closed_loops()

        assert _client._REGISTRY == {}

    def test_eight_threads_reaping_at_once_raise_nothing(self) -> None:
        """Eight threads reaping the same dead entries concurrently raise nothing and leave nothing behind.

        A reap over a handful of entries finishes inside one GIL slice, so the threads would never
        actually interleave. Two settings make them interleave, and both describe a machine under load:

        * 300 dead entries, so a single reap is long enough to be interrupted part-way;
        * ``sys.setswitchinterval(1e-6)``, so the interpreter preempts inside it.

        With both in place, an unlocked registry fails this reliably, usually first with
        ``RuntimeError: dictionary changed size during iteration`` and otherwise with ``KeyError``: one
        thread walking while another writes, or one thread deleting what another already deleted.

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
        """``close()`` never holds the lock across an ``await``, which is why it takes the lock twice.

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
    """AT-091d — three stores shutting down at once, which is the ordinary shape.

    ``examples/memory_recall.py write`` ends with
    ``asyncio.gather(curator.close(), writable.close(), curated.close())``. Three stores, one connection
    identity, one call each, and none of them may raise.

    Exercising it needs two things at the same time: more than one slot for the identity, which the
    per-loop registry provides, and a real suspension inside the first caller's ``client.close()``, which
    is what lets the second caller run and drop a slot the first one still has on its list. A dead loop's
    slot is the cheapest second slot, and the closed loop below is the cheapest way to make one.
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
        """No ``KeyError`` escapes a teardown path, where there is no later call to recover."""

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

    That is what a live loop does when it closes in between: the window is two statements wide and
    nothing here holds that loop still. Faked rather than raced, because a test that relied on timing
    would hit the window rarely and would not reliably fail without the handling.

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
    """AT-194 — a foreign loop closing during ``close()`` does not stop the other slots being released.

    ``close()`` releases every slot for an identity, and a slot on another live loop is closed by
    scheduling onto that loop. Between the ``is_closed()`` that decides it is worth scheduling and the
    schedule itself, that loop may close. The ``RuntimeError`` is not a ``PyMongoError``, so the driver
    does not handle it, and if it left ``close()`` every slot after the failing one would stay in the
    registry for the life of the process, from a path that runs in a ``finally`` and in an ``atexit``,
    where there is no later call to reach them.

    The un-awaited coroutine is the second half. ``client.close()`` has already been called by then, and
    a coroutine nobody awaits raises ``RuntimeWarning`` at collection, which this suite configures as an
    error, so the failure would surface in whatever test happened to trigger the collection.

    Nothing here claims the client gets closed. It cannot be: closing needs the loop that owns its
    sockets and that loop is gone, the same as for any client whose loop has closed. What is asserted is
    that the slots that are still releasable are released.
    """

    CONFIG: MongoDBConfig = {"database": "appdb", "connection_string": "mongodb://127.0.0.1:1/"}

    def test_the_running_loops_slot_is_released_after_the_foreign_schedule_fails(self) -> None:
        """The running loop's slot is released even though the failing foreign slot comes first in the snapshot."""
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

        Collected explicitly instead of waiting for the garbage collector to do it during some later
        test, where the warning would be attributed to the wrong place.
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
        """The vacuity guard. Handling that stopped scheduling altogether would pass both tests above and
        would leak a pool on every loop but the running one."""
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
