"""Constructing the two public classes: signature, connection sharing, ownership, teardown.

``test_client.py`` asserts the connection contract against ``_client.resolve`` directly. This file
asserts the same contract through the constructors a user actually calls: *100 stores* yield one client,
a bad config is rejected *at construction time*, and *two ``asyncio.run`` calls sharing one config* both
succeed. A registry that behaves perfectly while ``MongoDBStorage.__init__`` forgets to consult it
satisfies none of them.

Nothing here reaches the network. ``AsyncMongoClient`` construction performs no I/O, so a URI
pointing at a closed port is enough to build and register a client, and every test below runs in the
mock tier with no backend configured.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

import pytest
from pymongo import AsyncMongoClient
from strands.memory.types import _has_method

from strands_mongodb import _client
from strands_mongodb.memory_store import MongoDBMemoryStore
from strands_mongodb.storage import MongoDBStorage

if TYPE_CHECKING:
    from collections.abc import Callable

    from strands_mongodb.types import MongoDBConfig, MongoDBMemoryStoreConfig

# Port 1 is reserved and never listening, so anything that reaches the network here fails loudly
# instead of finding a stray local mongod and passing for the wrong reason.
#
# The timeout keeps the file fast. `_is_closed` distinguishes a closed client from a live one by
# probing both, and a live client against a dead port takes `serverSelectionTimeoutMS` to give up,
# which is 30 seconds at pymongo's default. Setting it in the URI covers the clients this package
# constructs as well.
UNREACHABLE = "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50"


@pytest.fixture(autouse=True)
def _clear_registry() -> Any:
    """Isolate each test from the process-global connection registry.

    Same justification as the fixture of the same name in ``test_client.py``: the registry is
    module-level by design (one pool per connection identity), so without this a test that constructs a store leaves a
    connection behind for the next one, and ``len(_REGISTRY) == 1`` assertions become order
    dependent.
    """
    _client._REGISTRY.clear()
    yield
    _client._REGISTRY.clear()


def _storage_config(**overrides: Any) -> MongoDBConfig:
    """Build a minimal valid connection config.

    Args:
        **overrides: Fields to set or replace.

    Returns:
        A config with ``database`` and ``connection_string`` set.
    """
    config: dict[str, Any] = {"database": "appdb", "connection_string": UNREACHABLE}
    config.update(overrides)
    return config  # type: ignore[return-value]


def _store_config(**overrides: Any) -> MongoDBMemoryStoreConfig:
    """Build a minimal valid memory-store config, nesting the connection half under ``config``.

    Args:
        **overrides: Top-level store fields to set or replace. Pass ``config=`` to replace the
            connection half.

    Returns:
        A config with ``name`` and a valid nested ``config``.
    """
    store: dict[str, Any] = {"name": "facts", "config": _storage_config()}
    store.update(overrides)
    return store  # type: ignore[return-value]


class TestTheConstructorSignatures:
    """One calling convention per SDK surface, each matching the SDK class it sits beside.

    The asymmetry is deliberate, and it is the part most likely to be "tidied" by a later reader into a
    single convention, which would be a breaking change on whichever surface lost.
    Asserted against ``inspect.signature`` rather than by calling, so the test describes the promise
    and not one call that happens to work.
    """

    def test_a_storage_takes_its_connection_config_positionally(self) -> None:
        """``MongoDBStorage(config, /, *, ...)``, matching ``S3Storage``'s resource-first shape."""
        parameters = list(inspect.signature(MongoDBStorage.__init__).parameters.values())

        assert parameters[1].name == "config"
        assert parameters[1].kind is inspect.Parameter.POSITIONAL_ONLY

    def test_the_storage_options_are_keyword_only_with_the_documented_defaults(self) -> None:
        """``collection``, ``prefix`` and ``ttl_seconds`` are per-instance, so they are not in the config.

        Keyword-only because ``MongoDBStorage(config, "sessions")`` reads as if the second argument
        were a key or a prefix, and the three of them are not in an order anyone would remember.
        """
        parameters = inspect.signature(MongoDBStorage.__init__).parameters

        for name, default in (("collection", "storage"), ("prefix", ""), ("ttl_seconds", None)):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
            assert parameters[name].default == default, name

    def test_passing_the_storage_config_by_keyword_is_rejected(self) -> None:
        """Positional-only is enforced by the interpreter, not merely documented.

        ``config=`` would otherwise be absorbed as an unexpected keyword and the failure would be a
        ``TypeError`` about ``config`` either way, so this asserts the parameter really is marked
        positional-only rather than trusting the same message twice.
        """
        with pytest.raises(TypeError):
            MongoDBStorage(config=_storage_config())  # type: ignore[misc, call-arg]

    def test_a_memory_store_takes_its_whole_config_as_keyword_arguments(self) -> None:
        """``MongoDBMemoryStore(**config)``, matching ``BedrockKnowledgeBaseStore``.

        The SDK's ``MemoryManager`` never constructs a store, so this convention exists for the
        caller's benefit: ``name=`` and ``writable=`` read as fields of the store, and they are
        declared in a ``TypedDict`` that extends the SDK's own ``MemoryStoreConfig``.
        """
        parameters = list(inspect.signature(MongoDBMemoryStore.__init__).parameters.values())

        assert len(parameters) == 2, "self plus one **kwargs parameter, nothing positional"
        assert parameters[1].kind is inspect.Parameter.VAR_KEYWORD


class TestOneClientPerConfigAcrossBothSurfaces:
    """AT-012 — N instances from one config, one connection pool.

    The count of 100 matters. A multi-tenant application builds one store per user
    per request, and each ``AsyncMongoClient`` opens its own pool with its own monitoring threads, so
    the failure mode is not "slightly wasteful": it is a connection count that grows with traffic
    until the deployment refuses new connections.
    """

    def test_100_storages_from_one_config_share_one_client(self) -> None:
        """One config object passed 100 times yields one pool."""
        config = _storage_config()

        storages = [MongoDBStorage(config) for _ in range(100)]

        clients = {id(storage._connection.client) for storage in storages}
        assert len(clients) == 1
        assert len(_client._REGISTRY) == 1

    def test_100_memory_stores_from_one_config_share_one_client(self) -> None:
        """The memory surface reaches the same registry, through its nested ``config`` key."""
        connection = _storage_config()

        stores = [MongoDBMemoryStore(**_store_config(config=connection, name=f"facts{index}")) for index in range(100)]

        clients = {id(store._connection.client) for store in stores}
        assert len(clients) == 1
        assert len(_client._REGISTRY) == 1

    def test_a_storage_and_a_memory_store_sharing_a_config_share_one_client(self) -> None:
        """Stores and storages share one registry, and the two surfaces are separate modules.

        The likely way to fail this is per-module state: a registry defined in ``storage.py`` and
        another in ``memory_store.py`` would pass both tests above and still open two pools for an
        agent that uses memory and session storage against one database, which is the common case.
        """
        connection = _storage_config()

        storage = MongoDBStorage(connection)
        store = MongoDBMemoryStore(**_store_config(config=connection))

        assert storage._connection.client is store._connection.client
        assert len(_client._REGISTRY) == 1

    def test_two_storages_differing_only_by_collection_share_one_client(self) -> None:
        """A per-instance option is not part of the connection identity.

        Sessions and transcripts land in different collections of one database, and that must not
        cost a second pool. ``collection`` is a constructor argument rather than a config field
        precisely so it stays out of the identity computed from the config.
        """
        connection = _storage_config()

        sessions = MongoDBStorage(connection, collection="sessions")
        transcripts = MongoDBStorage(connection, collection="transcripts")

        assert sessions._connection.client is transcripts._connection.client
        assert len(_client._REGISTRY) == 1


class TestABadConfigIsRejectedAtConstruction:
    """AT-015 — the failure happens where the caller can see it.

    The check runs at construction time because of upstream behavior.
    pymongo validates nothing about a connection string: ``AsyncMongoClient("not-a-uri")`` succeeds,
    reading the string as a hostname. Deferring the check to first use would turn a missing setting
    into a 30-second server-selection timeout against ``localhost``, raised from inside whichever SDK
    subsystem happened to touch storage first.
    """

    def test_a_storage_with_no_way_to_reach_the_database_raises_value_error(self) -> None:
        """Neither ``client`` nor ``connection_string``, so there is nothing to connect to."""
        with pytest.raises(ValueError, match="connection_string"):
            MongoDBStorage({"database": "appdb"})

    def test_a_memory_store_with_no_way_to_reach_the_database_raises_value_error(self) -> None:
        """Same rejection through the nested config, which is the easier of the two to get wrong."""
        with pytest.raises(ValueError, match="connection_string"):
            MongoDBMemoryStore(name="facts", config={"database": "appdb"})

    def test_a_rejected_config_leaves_nothing_registered(self) -> None:
        """A failed construction must not leave a pool behind for a later config to find."""
        with pytest.raises(ValueError):
            MongoDBStorage({"database": "appdb"})

        assert _client._REGISTRY == {}

    @pytest.mark.parametrize("surface", ["storage", "memory"])
    def test_a_config_with_no_database_is_rejected_before_a_client_is_built(self, surface: str) -> None:
        """``database`` is read first, so an incomplete config does not register a pool it then abandons.

        Read with a bare subscript, following ``BedrockKnowledgeBaseStore``'s
        ``kb_config["knowledge_base_id"]``: the key is ``Required`` in the ``TypedDict``, so mypy
        rejects a literal that omits it and the runtime failure is for a config assembled
        dynamically.

        The assertion that matters is the second one. Reading ``database`` *after* resolving would
        raise the same ``KeyError`` while leaving a registered connection pool that no object owns and
        no ``close()`` will ever reach.
        """
        with pytest.raises(KeyError, match="database"):
            if surface == "storage":
                MongoDBStorage({"connection_string": UNREACHABLE})  # type: ignore[typeddict-item]
            else:
                MongoDBMemoryStore(name="facts", config={"connection_string": UNREACHABLE})  # type: ignore[typeddict-item]

        assert _client._REGISTRY == {}

    def test_a_memory_store_without_a_name_is_rejected(self) -> None:
        """``name`` is the store config's own mandatory key, and the SDK targets stores by it.

        ``MemoryManager`` looks stores up by name and the ``search_memory`` tool takes one, so a store
        with no name is unreachable through every SDK entry point that is not a direct reference.

        The ``type: ignore`` code is worth reading: because the config arrives as keyword arguments,
        mypy reports the omission as ``Missing named argument "name"``, a plain ``call-arg`` error
        rather than a ``TypedDict`` one. The runtime check below is what catches a config assembled
        from a dict at runtime, which is the only way to reach it.
        """
        with pytest.raises(KeyError, match="name"):
            MongoDBMemoryStore(config=_storage_config())  # type: ignore[call-arg]


async def _is_closed(client: AsyncMongoClient[Any]) -> bool:
    """Whether a client has been closed, without needing a reachable server.

    ``AsyncMongoClient`` exposes no public closed flag. A closed client raises ``InvalidOperation``
    from any operation, immediately and without touching the network, while a live client against a
    dead port raises ``ServerSelectionTimeoutError`` after its timeout. Only the first is the answer
    to this question.

    Args:
        client: The client to probe.

    Returns:
        ``True`` if the client refuses to operate because it was closed.
    """
    from pymongo.errors import InvalidOperation

    try:
        await client.admin.command("ping")
    except InvalidOperation:
        return True
    except Exception:  # noqa: BLE001 - any other failure means the client was still willing to try
        return False
    return False


class TestOwnershipSurvivesTheStoreBoundary:
    """AT-010, AT-011 — who built the client decides who may close it.

    Asserted again at this level even though ``test_client.py`` covers ``_client.close``, because the
    defect this guards against, a store that calls ``self.client.close()`` unconditionally, lives in the
    store's ``close()``, not in the connection module.
    """

    async def test_a_storage_closes_the_client_it_built(self) -> None:
        """We opened the pool, so we are the only thing that can be relied on to close it."""
        storage = MongoDBStorage(_storage_config())
        client = storage._connection.client

        await storage.close()

        assert await _is_closed(client)

    async def test_a_storage_never_closes_an_injected_client(self) -> None:
        """The caller's client outlives us.

        In an agent process one client is typically shared between memory, storage and the
        application's own queries. Closing it here tears down a pool other subsystems are still
        using, and the symptom surfaces somewhere else, later.
        """
        injected: AsyncMongoClient[Any] = AsyncMongoClient(UNREACHABLE)
        try:
            storage = MongoDBStorage({"database": "appdb", "client": injected})

            await storage.close()

            assert not await _is_closed(injected)
        finally:
            await injected.close()

    async def test_a_memory_store_closes_the_client_it_built(self) -> None:
        """The same rule on the memory surface, which reads its connection from a nested key."""
        store = MongoDBMemoryStore(**_store_config())
        client = store._connection.client

        await store.close()

        assert await _is_closed(client)

    async def test_a_memory_store_never_closes_an_injected_client(self) -> None:
        """The nested config is where an injected client is easiest to lose track of."""
        injected: AsyncMongoClient[Any] = AsyncMongoClient(UNREACHABLE)
        try:
            store = MongoDBMemoryStore(name="facts", config={"database": "appdb", "client": injected})

            await store.close()

            assert not await _is_closed(injected)
        finally:
            await injected.close()

    async def test_closing_twice_is_a_no_op(self) -> None:
        """``close()`` gets called from a ``finally`` and again from an ``atexit`` in real code."""
        storage = MongoDBStorage(_storage_config())

        await storage.close()
        await storage.close()

    async def test_closing_one_instance_releases_the_connection_the_others_share(self) -> None:
        """The scope of connection sharing, asserted rather than left to be discovered.

        One registry entry per connection identity means ``close()`` is per-config, not per-instance:
        the pool the second storage was using is gone once the first one closes. That is why ``close()``
        belongs at process shutdown rather than in a per-request ``finally`` around one store of several.

        The sibling resolves its connection per operation, finds the entry gone, and opens a new pool.
        So this asserts the pool was released and the sibling still works. The new pool has no owner who
        will close it, which is a known limit of per-config ownership.
        """
        connection = _storage_config()
        first = MongoDBStorage(connection, collection="sessions")
        second = MongoDBStorage(connection, collection="transcripts")
        shared = second._connection.client

        await first.close()

        assert _client._REGISTRY == {}
        assert await _is_closed(shared)
        replacement = second._connection.client
        try:
            assert replacement is not shared
            assert not await _is_closed(replacement)
        finally:
            await second.close()


class TestTheCollectionAndOptionsTheCallerAsksFor:
    """The per-instance settings are recorded as given, with the documented defaults.

    Two collection defaults rather than one, because the two surfaces hold unrelated data in the same
    database: ``"storage"`` for the ``Storage`` contract's opaque bytes and
    ``"memory"`` for ``MemoryEntry`` documents with a vector index on them
    (``MongoDBMemoryStoreConfig.collection``). Defaulting both to one name would put session blobs and
    searchable memories in one collection, where the vector index would try to embed base64.
    """

    def test_a_storage_defaults_to_the_storage_collection(self) -> None:
        """The default collection name is ``"storage"``."""
        assert MongoDBStorage(_storage_config())._collection_name == "storage"

    def test_a_memory_store_defaults_to_the_memory_collection(self) -> None:
        """``MongoDBMemoryStoreConfig.collection``: "Defaults to ``"memory"``"."""
        assert MongoDBMemoryStore(**_store_config())._collection_name == "memory"

    def test_the_storage_options_are_recorded_as_passed(self) -> None:
        """All three, in one test, because the failure worth catching is a swap between them."""
        storage = MongoDBStorage(_storage_config(), collection="sessions", prefix="tenant-a/", ttl_seconds=3600)

        assert storage._collection_name == "sessions"
        assert storage._prefix == "tenant-a/"
        assert storage._ttl_seconds == 3600

    def test_a_storage_without_a_ttl_records_no_expiry_rather_than_zero(self) -> None:
        """Documents written without ``ttl_seconds`` never expire.

        ``None`` and ``0`` are opposites here and both are falsy. A MongoDB TTL index with
        ``expireAfterSeconds: 0`` expires documents the moment ``expires_at`` passes, so collapsing
        the default to ``0`` anywhere between here and the write path deletes data instead of keeping
        it forever.
        """
        assert MongoDBStorage(_storage_config())._ttl_seconds is None

    def test_the_database_named_in_the_config_is_the_one_recorded(self) -> None:
        """One client serves many databases, so the name has to be kept per instance.

        It is deliberately not part of the connection identity: two storages on different databases of
        one deployment share a pool, which only works if each remembers its own
        database rather than reading it back off the shared connection.
        """
        storage = MongoDBStorage(_storage_config(database="tenant_a"))
        store = MongoDBMemoryStore(**_store_config(config=_storage_config(database="tenant_b")))

        assert storage._database_name == "tenant_a"
        assert store._database_name == "tenant_b"


class TestOneConfigAcrossTwoEventLoops:
    """AT-017 — a synchronous ``Agent`` runs each invocation on a new loop.

    Two real ``asyncio.run`` calls, because one loop with two tasks does not
    reproduce the failure. Neither does an uncontended lock. ``asyncio.Lock`` has bound lazily since
    3.10, so ``acquire()`` on an unlocked lock returns without ever asking which loop it is on, and a
    test that merely crosses loops passes against an implementation that never rebinds.

    The failure this prevents is also unrecoverable rather than transient: ``RuntimeError: ... is bound
    to a different event loop`` is raised *after* the lock marks itself held, so the lock stays locked
    forever and every later invocation fails too. An agent that works once and then hangs.
    """

    def test_two_asyncio_run_calls_sharing_one_config_both_succeed(self) -> None:
        """The second invocation must not inherit a lock bound to the first one's closed loop.

        Connection sharing is bounded per event loop, because pymongo binds a client to the loop of its
        first operation and refuses it everywhere else, so the second run gets its own client. The lock
        contention is what proves the lock is not inherited from the first run's closed loop.
        """
        config = _storage_config()

        async def invocation() -> AsyncMongoClient[Any]:
            storage = MongoDBStorage(config)

            async def hold() -> None:
                async with storage._connection.get_lock():
                    await asyncio.sleep(0.01)

            await asyncio.gather(hold(), hold())
            return storage._connection.client

        first = asyncio.run(invocation())
        second = asyncio.run(invocation())

        assert first is not second, "a client bound to the first run's closed loop cannot serve the second"


class TestOneStoreInstanceAcrossTwoEventLoops:
    """AT-091c — one store across successive event loops, built once instead of once per loop.

    The class above builds a fresh ``MongoDBStorage`` inside each ``asyncio.run``, so a per-loop registry
    is enough to satisfy it. A constructor that cached the resolved connection on the instance would pass
    that class and still fail here: a store that *outlives* its first loop would never ask the registry
    again.

    Nothing here connects. ``resolve`` builds an ``AsyncMongoClient`` without I/O, and identity of the
    client object is the whole assertion, so an unreachable address is the right target: it keeps the
    test in the unit tier and makes a regression fail in milliseconds.
    ``tests/test_storage.py::TestOneStorageServesTwoSuccessiveEventLoops`` is the same claim against a
    deployment, where the wrong answer raises instead of comparing unequal.
    """

    @staticmethod
    def _client_seen_on_a_fresh_loop(read: Callable[[], AsyncMongoClient[Any]]) -> AsyncMongoClient[Any]:
        """Run ``read`` on its own loop and return the client it saw."""

        async def body() -> AsyncMongoClient[Any]:
            return read()

        return asyncio.run(body())

    def test_a_storage_reads_the_client_for_the_loop_it_is_running_on(self) -> None:
        """One instance, two loops, two clients, and the second one is live."""
        storage = MongoDBStorage(_storage_config())

        first = self._client_seen_on_a_fresh_loop(lambda: storage._connection.client)
        second = self._client_seen_on_a_fresh_loop(lambda: storage._connection.client)

        assert first is not second

    def test_a_memory_store_reads_the_client_for_the_loop_it_is_running_on(self) -> None:
        """The same on the memory surface, which resolves from the nested ``config`` key."""
        store = MongoDBMemoryStore(**_store_config())

        first = self._client_seen_on_a_fresh_loop(lambda: store._connection.client)
        second = self._client_seen_on_a_fresh_loop(lambda: store._connection.client)

        assert first is not second

    def test_the_collection_handle_follows_the_client_rather_than_outliving_it(self) -> None:
        """``_collection()`` is the only path the operations take, so it is the one that has to move.

        A ``_connection`` property with a collection still captured at construction would pass the two
        tests above and fail every write.
        """
        storage = MongoDBStorage(_storage_config())

        first = self._client_seen_on_a_fresh_loop(lambda: storage._collection().database.client)
        second = self._client_seen_on_a_fresh_loop(lambda: storage._collection().database.client)

        assert first is not second

    def test_construction_still_refuses_a_config_it_cannot_resolve(self) -> None:
        """The eager resolution ``__init__`` throws away is what keeps config rejection at construction.

        Dropping the call and letting the property do all the work would move this ``ValueError`` to the
        first ``write``, which is a worse place to learn that a config was never usable.
        """
        with pytest.raises(ValueError):
            MongoDBStorage({"database": "appdb"})  # type: ignore[typeddict-item]

    def test_one_loop_still_sees_one_client_across_many_operations(self) -> None:
        """Per operation is per *loop*, not per call. A new pool per call would be a resource leak."""
        storage = MongoDBStorage(_storage_config())

        async def read_thrice() -> set[int]:
            return {id(storage._connection.client) for _ in range(3)}

        assert len(asyncio.run(read_thrice())) == 1


class TestTheHooksThatMustStayAbsent:
    """Two optional SDK hooks where a raising stub is worse than nothing.

    Both are feature-detected: ``_resolve_namespace`` calls ``for_sandbox`` only when the attribute is
    present, and ``MemoryManager`` routes extraction through ``add_messages`` only when
    ``_has_method`` reports it. Defining either as ``raise NotImplementedError`` converts "the SDK
    picks another path" into "the SDK crashes", and for ``for_sandbox`` the path it would break is
    context offloading for every MongoDB user.

    Neither class inherits its Protocol, so ``hasattr`` is exact for both: there is no base to inherit
    a stub from. ``_has_method`` is asserted alongside it for ``add_messages`` because that is the
    probe the manager actually reads, and the two would disagree if this class ever started inheriting
    ``MemoryStore``.
    """

    def test_a_storage_does_not_define_for_sandbox(self) -> None:
        """``hasattr`` is the probe ``_resolve_namespace`` uses, so it is the one that decides this."""
        assert not hasattr(MongoDBStorage(_storage_config()), "for_sandbox")

    def test_a_memory_store_does_not_implement_add_messages(self) -> None:
        """Asserted with the SDK's own detector, against the class rather than the instance.

        ``AddMessagesContext`` carries no run-unique identifier, so no durable dedup token can be
        built from it, and extraction writes are at-least-once. Implementing this method would make
        duplicate facts a design feature. The manager's fallback is
        its own ``ModelExtractor`` plus our ``add``, which does dedupe.
        """
        store = MongoDBMemoryStore(**_store_config())

        assert not hasattr(store, "add_messages")
        assert _has_method(store, "add_messages") is False


class TestANegativeLifetimeIsRefused:
    """AT-226. A negative ``ttl_seconds`` stamps an expiry in the past, so the TTL monitor
    would delete every write within a minute. ``0`` is a valid request and stays accepted."""

    def test_a_storage_refuses_a_negative_ttl(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            MongoDBStorage(_storage_config(), ttl_seconds=-1)

    def test_a_memory_store_refuses_a_negative_ttl(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            MongoDBMemoryStore(**_store_config(ttl_seconds=-1))

    def test_a_refused_config_registers_no_connection(self) -> None:
        for build in (
            lambda: MongoDBStorage(_storage_config(), ttl_seconds=-60),
            lambda: MongoDBMemoryStore(**_store_config(ttl_seconds=-60)),
        ):
            with pytest.raises(ValueError):
                build()
        assert _client._REGISTRY == {}

    def test_zero_is_still_accepted_on_both(self) -> None:
        assert MongoDBStorage(_storage_config(), ttl_seconds=0)._ttl_seconds == 0
        assert MongoDBMemoryStore(**_store_config(ttl_seconds=0)).ttl_seconds == 0
