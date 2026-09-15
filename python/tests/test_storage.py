"""The five ``Storage`` contract methods, plus ``namespace`` and ``delete_prefix``.

``REQ-E-018`` through ``REQ-E-039``. Two tiers, split on the same rule as ``test_indexes.py``: what
this package *decides* is asserted against a fake, and what **MongoDB** does with an ``_id`` range is
asserted against a deployment.

The split matters more here than anywhere else in the suite, because the whole data model rests on one
claim about the server: that ``sort("_id", 1)`` over a bounded range returns keys in the order
Python's ``sorted()`` would. No mock can support that claim, and the SDK's snapshot history depends on
it (``AT-027``). The unit tier below therefore never asserts ordering. It asserts what filter, what
document and what preflight, and leaves the ordering to the contract tier.

Nothing here re-tests connection sharing or ownership. ``test_construction.py`` owns those, and this
file's unit tier points every store at a closed port so that a call which reaches the network fails
loudly instead of finding a stray local ``mongod``.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import sys
import time
import traceback
import typing

# Imported at runtime rather than under ``TYPE_CHECKING``: the fake's range matching uses it in an
# ``isinstance`` check, which is what tells an ``_id`` equality filter from an ``_id`` range.
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import bson
import pytest
from strands.storage import Storage
from strands.storage.storage import _NAMESPACED, _resolve_namespace
from strands.types.exceptions import StorageError

from strands_mongodb import _client, _indexes
from strands_mongodb import storage as storage_module
from strands_mongodb.storage import MongoDBStorage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator, Sequence

    from strands_mongodb.types import MongoDBConfig

# Port 1 is reserved and never listening. Same rationale as ``test_construction.py``: building an
# ``AsyncMongoClient`` performs no I/O, so this is enough to construct a store, and any call that
# actually leaves the process fails instead of finding a database to pass against.
UNREACHABLE = "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50"

# A URI whose password must not appear in any message, ``repr`` or traceback this module produces.
# ``AT-014``'s sweep across the five contract methods, which the connection layer could only leave open.
LEAKY_URI = "mongodb://appuser:SUPERSECRET@db.example.com:27017/appdb"


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    """Isolate each test from the process-global connection registry (``REQ-E-012``)."""
    _client._REGISTRY.clear()
    yield
    _client._REGISTRY.clear()


class FakeCursor:
    """What ``find()`` returns: a cursor that sorts synchronously and iterates asynchronously.

    Both halves are parity claims rather than guesses, and ``MP-001`` checks them against
    ``AsyncCursor``. ``sort`` is **not** a coroutine function on the real class and returns the cursor
    itself, so ``await collection.find(...).sort(...)`` would be a ``TypeError`` on a deployment;
    ``__iter__`` does not exist there, so a synchronous ``for`` loop over the results is the same kind
    of failure. A fake that were loose about either would let both pass here.
    """

    def __init__(self, documents: Iterable[Mapping[str, Any]], error: BaseException | None = None) -> None:
        self._documents = [dict(document) for document in documents]
        self._error = error
        self.sort_calls: list[tuple[str, int]] = []

    def sort(self, key_or_list: str, direction: int | None = None) -> FakeCursor:
        self.sort_calls.append((key_or_list, direction if direction is not None else 1))
        return self

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        for document in self._documents:
            yield document
        # After the documents, never instead of them. A connection that dies while a cursor is being
        # drained raises from the ``async for`` rather than from the ``find`` that produced it, which is
        # the shape ``REQ-E-036`` is most easily lost to: an error wrapper attached to the call alone
        # lets a bare ``PyMongoError`` out of ``list``.
        if self._error is not None:
            raise self._error


class FakeAggregationCursor:
    """What ``await aggregate()`` returns: async-iterable and nothing else.

    Deliberately **not** ``FakeCursor``. The real ``AsyncCommandCursor`` has no ``sort``, because an
    aggregation's ordering comes from a ``$sort`` stage inside the pipeline rather than from a method on
    the cursor, and a fake that offered both would let ``search`` sort the wrong way and still pass.

    It also returns whatever it was handed rather than interpreting the pipeline. Scoring is what the
    server does, and a fake that scored in Python would be checking the SDK's tokenizer against itself.
    The unit tier below asserts the pipeline this package *emits*; ``AT-032b`` through ``AT-033`` assert
    what MongoDB *does* with it.
    """

    def __init__(self, documents: Iterable[Mapping[str, Any]]) -> None:
        self._documents = [dict(document) for document in documents]

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        for document in self._documents:
            yield document


class FakeStorageDatabase:
    """The database half of the fake, which exists for one attribute: ``name``.

    ``_indexes._ensure_ordinary`` builds its namespace from ``collection.database.name`` and puts it in
    every message it raises, so a fake without this raises ``AttributeError`` from inside the module
    under test rather than the ``StorageError`` the requirement describes. Unlike ``test_indexes.py``'s
    version there is no ``create_collection`` here: that call belongs to the search path, and the
    storage collection has no search index at all.
    """

    def __init__(self, *, name: str = "appdb") -> None:
        self.name = name


class FakeStorageCollection:
    """``AsyncCollection`` narrowed to the methods ``storage.py`` calls (``MP-001``).

    Holds documents in a plain dict keyed by ``_id`` and answers a range query by comparing Python
    strings, which is close enough for *what was asked* and deliberately not offered as evidence of
    *what MongoDB returns*: the contract tier owns that. Unknown attributes raise, unlike the real
    ``AsyncCollection.__getattr__``, so a typo in the module under test fails in the unit tier.
    """

    def __init__(
        self,
        *,
        name: str = "storage",
        documents: Iterable[Mapping[str, Any]] = (),
        error: BaseException | None = None,
        iteration_error: BaseException | None = None,
        aggregate_result: Iterable[Mapping[str, Any]] = (),
        database: FakeStorageDatabase | None = None,
        indexes: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.database = database if database is not None else FakeStorageDatabase()
        self.documents: dict[str, dict[str, Any]] = {str(document["_id"]): dict(document) for document in documents}
        self._error = error
        self._iteration_error = iteration_error
        self._aggregate_result = list(aggregate_result)
        # What the server would report today. ``_id_`` is in the default because the server creates it
        # whether or not anybody asks, and a fake that started empty would let a test claim
        # ``ensure_indexes()`` creates an index the server already made.
        self.indexes: dict[str, dict[str, Any]] = (
            dict(indexes) if indexes is not None else {"_id_": {"v": 2, "key": [("_id", 1)]}}
        )
        self.pipelines: list[list[dict[str, Any]]] = []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.cursors: list[FakeCursor] = []

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((method, args, kwargs))
        if self._error is not None:
            raise self._error

    @property
    def names_called(self) -> list[str]:
        """Just the method names, for the assertions that only care whether anything was issued."""
        return [call[0] for call in self.calls]

    def matching(self, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Apply the subset of query syntax this module emits: ``_id`` equality and an ``_id`` range."""
        clause = query.get("_id")
        if clause is None:
            return sorted(self.documents.values(), key=lambda document: str(document["_id"]))
        if not isinstance(clause, Mapping):
            document = self.documents.get(str(clause))
            return [document] if document is not None else []
        lower, upper = clause.get("$gte"), clause.get("$lt")
        found = [
            document
            for key, document in self.documents.items()
            if (lower is None or key >= lower) and (upper is None or key < upper)
        ]
        return sorted(found, key=lambda document: str(document["_id"]))

    async def replace_one(self, filter: Mapping[str, Any], replacement: Mapping[str, Any], upsert: bool = False) -> Any:
        self._record("replace_one", filter, replacement, upsert=upsert)
        self.documents[str(replacement["_id"])] = dict(replacement)
        return None

    async def find_one(self, filter: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        self._record("find_one", filter)
        found = self.matching(filter or {})
        return found[0] if found else None

    async def delete_one(self, filter: Mapping[str, Any]) -> Any:
        self._record("delete_one", filter)
        for document in self.matching(filter):
            del self.documents[str(document["_id"])]
        return FakeDeleteResult(1)

    async def delete_many(self, filter: Mapping[str, Any]) -> Any:
        self._record("delete_many", filter)
        found = self.matching(filter)
        for document in found:
            del self.documents[str(document["_id"])]
        return FakeDeleteResult(len(found))

    def find(self, filter: Mapping[str, Any] | None = None, projection: Any = None) -> FakeCursor:
        self._record("find", filter, projection)
        cursor = FakeCursor(self.matching(filter or {}), self._iteration_error)
        self.cursors.append(cursor)
        return cursor

    async def aggregate(self, pipeline: Iterable[Mapping[str, Any]]) -> FakeAggregationCursor:
        # ``async def``, unlike ``find``. The driver is not uniform about this and ``MP-001`` pins both
        # directions: the cursor arrives only after an ``await`` here, and without one the caller gets
        # ``AttributeError`` on a coroutine.
        stages = [dict(stage) for stage in pipeline]
        self._record("aggregate", stages)
        self.pipelines.append(stages)
        return FakeAggregationCursor(self._aggregate_result)

    async def index_information(self) -> dict[str, dict[str, Any]]:
        self._record("index_information")
        return {name: dict(spec) for name, spec in self.indexes.items()}

    async def create_indexes(self, models: Sequence[Any]) -> list[str]:
        self._record("create_indexes", list(models))
        for model in models:
            document = dict(model.document)
            # The same two normalisations ``test_indexes.py``'s fake performs, and measured the same
            # way on 4.17.0 rather than reasoned about: ``IndexModel.document`` holds ``key`` as a
            # ``SON`` where the server reports a list of tuples, and it carries the name as a field
            # where the server uses it as the entry's key. Without them a create followed by a verify
            # reads a shape the server would never have returned, and ``AT-121``'s second call would
            # create the index again.
            name = document.pop("name")
            document["key"] = list(document["key"].items())
            self.indexes[name] = {"v": 2, **document}
        return [model.document["name"] for model in models]


class FakeDeleteResult:
    """``DeleteResult`` narrowed to the one attribute ``delete_prefix`` reads."""

    def __init__(self, deleted_count: int) -> None:
        self.deleted_count = deleted_count


def network_failure(message: str = "socket timed out") -> Exception:
    """Build a ``PyMongoError`` that carries no server error code, as a dead connection would."""
    from pymongo.errors import NetworkTimeout

    return NetworkTimeout(message)


def unit_storage(fake: FakeStorageCollection | None = None, **overrides: Any) -> MongoDBStorage:
    """Build a store whose collection is the fake, without touching the network.

    ``_collection`` is a method rather than a property precisely so a test can shadow it on the
    instance. The alternative is a constructor parameter for injecting a collection, which would be a
    public knob existing only for tests.

    Args:
        fake: The collection every method of the returned store will call. A fresh empty one by default.
        **overrides: Keyword arguments for ``MongoDBStorage``, plus ``config`` to replace connection
            fields.

    Returns:
        A store pointing at ``fake``, registered against an unreachable URI.
    """
    config: dict[str, Any] = {"database": "appdb", "connection_string": UNREACHABLE}
    config.update(overrides.pop("config", {}))
    storage = MongoDBStorage(config, **overrides)  # type: ignore[arg-type]
    collection = fake if fake is not None else FakeStorageCollection()
    # The fake is not an ``AsyncCollection`` and never will be, which is the whole reason the parity
    # class below exists. mypy is right to object; the ignore is the seam it objects at.
    storage._collection = lambda: collection  # type: ignore[assignment, return-value]
    return storage


def protocol_members(protocol: type) -> set[str]:
    """The names a ``runtime_checkable`` protocol checks for, on every interpreter the classifiers claim.

    ``AT-111`` / ``REQ-E-110``. ``__protocol_attrs__`` is a 3.12 addition: measured absent on 3.10 and 3.11
    and present on 3.12, 3.13 and 3.14. Reading it directly is what failed the first time this suite ran on
    the floor, in the two tests below.

    There is no public replacement, so this falls back to ``typing``'s private helper, which returns the same
    five names on all five versions. Deliberately not ``dir()``, which looks public and is wrong here: it
    omits members declared as a bare annotation, and a bare annotation arriving on the SDK's ``Storage`` is
    the exact upstream change these tests exist to notice. A ``dir()`` fallback would stay green through the
    failure they were written for.

    Raises rather than returning an empty set if a future ``typing`` offers neither, because both callers
    iterate the result and an empty set passes both of them vacuously.
    """
    members = getattr(protocol, "__protocol_attrs__", None)
    if members is None:
        members = getattr(typing, "_get_protocol_attrs", lambda _: None)(protocol)
    assert members, (
        f"neither __protocol_attrs__ nor typing._get_protocol_attrs yields the members of {protocol!r} "
        f"on Python {sys.version.split()[0]}; AT-034 below cannot be checked without them"
    )
    return set(members)


@pytest.mark.compat
class TestTheProtocolIsSatisfiedStructurally:
    """AT-034, AT-044c / REQ-E-034 — the protocol boundary, checked at runtime rather than by inheritance.

    ``MongoDBStorage`` does not inherit ``Storage``, matching all three backends the SDK ships, so
    nothing but this test connects the class to the protocol. ``runtime_checkable`` makes
    ``isinstance`` an attribute-presence check, which is weaker than it looks and is exactly why the
    negative case below is here too.

    Marked ``compat`` and it still runs in an ordinary session, the same arrangement the memory store's
    conformance check uses. The marker is what the weekly canary selects, and the release checklist asks for both
    of these against the newest published SDK. Only the memory one carried it at first, so the job
    watched one protocol and reported on two; ``AT-119`` in ``test_ci.py`` is what now fails if that recurs.
    """

    def test_a_storage_instance_satisfies_the_protocol(self) -> None:
        assert isinstance(unit_storage(), Storage)

    def test_the_protocol_has_five_members_and_a_four_method_class_fails(self) -> None:
        """The negative half of ``AT-034``, so the positive half fails for a reason rather than by luck.

        ``Storage``'s members are ``{write, read, delete, list, search}``. Measured: a class with the
        first four returns ``False`` and adding ``search`` returns ``True``. That is the whole reason
        ``search`` is a delegation to the SDK's own strategy instead of waiting for
        the aggregation pipeline.
        """

        class FourMethods:
            async def write(self, key: str, data: bytes) -> None: ...
            async def read(self, key: str) -> bytes | None: ...
            async def delete(self, key: str) -> None: ...
            async def list(self, query: str) -> list[str]: ...  # type: ignore[empty-body]

        assert protocol_members(Storage) == {
            "write",
            "read",
            "delete",
            "list",
            "search",
        }
        assert not isinstance(FourMethods(), Storage)

    def test_for_sandbox_is_absent_rather_than_a_stub(self) -> None:
        """AT-044c. ``ContextOffloader._storage_for_agent`` probes ``hasattr``
        (``plugin.py:308``) and skips sandbox binding when it is missing, so a stub that raised would
        break context offloading for every MongoDB user rather than degrade it."""
        assert not hasattr(unit_storage(), "for_sandbox")

    def test_every_contract_method_is_a_coroutine_function(self) -> None:
        """``isinstance`` checks presence, not that anything is awaitable. A synchronous ``read`` would
        satisfy the protocol at runtime and hand the SDK a coroutine-shaped nothing."""
        for name in sorted(protocol_members(Storage)):
            assert inspect.iscoroutinefunction(getattr(MongoDBStorage, name)), name


class TestTheProtocolMemberLookupWorksOnEveryClaimedInterpreter:
    """AT-111 / REQ-E-110 — the two tests above read a 3.12 attribute and failed on the 3.10 floor.

    A floor-audit finding. Not a defect in the package: ``src`` holds no version gate at all and compiles
    on 3.10 unchanged. This is about the suite being able to check ``AT-034`` everywhere the classifiers
    promise the package works.
    """

    def test_it_agrees_with_the_attribute_wherever_the_attribute_exists(self) -> None:
        """Both routes, compared on whichever interpreter is running, so neither can drift unnoticed."""
        attribute = getattr(Storage, "__protocol_attrs__", None)

        if attribute is None:
            assert sys.version_info < (3, 12), (
                f"__protocol_attrs__ was measured present from 3.12 and is missing on "
                f"{sys.version.split()[0]}; the fallback is now carrying every version"
            )
            # Private, so absent from the stubs. Named directly rather than through the helper's own
            # ``getattr``, because comparing the two routes is the entire assertion.
            assert protocol_members(Storage) == set(typing._get_protocol_attrs(Storage))  # type: ignore[attr-defined]
        else:
            assert sys.version_info >= (3, 12)
            assert protocol_members(Storage) == set(attribute)

    def test_it_refuses_to_return_an_empty_set(self) -> None:
        """The failure that would be silent. Both callers iterate the result, so both pass on nothing.

        A plain class has no ``__protocol_attrs__`` and ``typing``'s helper returns an empty set for it,
        measured on 3.10 and 3.14, which makes it the stand-in for a ``typing`` that stops answering.
        """

        class NotAProtocol:
            pass

        with pytest.raises(AssertionError, match="neither __protocol_attrs__ nor"):
            protocol_members(NotAProtocol)


class TestNamespaceReturnsAnInstanceMarkedAsAlreadyScoped:
    """AT-030, AT-039 / REQ-E-030, REQ-E-039 — the decision that had been missed.

    ``namespace()`` returning a real ``MongoDBStorage`` rather than the SDK's ``_NamespacedStorage``
    keeps ``list`` a single index range scan and keeps ``delete_prefix`` reachable. Both of those
    depend on the SDK not wrapping the result, and the only thing that tells it not to is the sentinel.
    """

    def test_the_scoped_instance_carries_the_sdk_sentinel(self) -> None:
        scoped = unit_storage().namespace("tenant-a")

        assert getattr(scoped, "_namespaced", None) is _NAMESPACED

    def test_an_unscoped_instance_does_not_carry_it(self) -> None:
        """Otherwise the sentinel says nothing: ``_resolve_namespace`` would skip the default prefix for
        every store, including the one the user handed straight to ``Agent``."""
        assert getattr(unit_storage(), "_namespaced", None) is None

    def test_the_sdk_leaves_a_scoped_instance_alone(self) -> None:
        """AT-039, asserted through the SDK's own resolver rather than through the attribute.

        This is the behaviour the sentinel exists for. Without it ``_resolve_namespace`` calls
        ``namespace()`` again and the memory store's keys land a level deeper than
        ``LocalFileStorage`` puts them, which breaks the one promise the protocol makes.
        """
        scoped = unit_storage().namespace("tenant-a")

        assert _resolve_namespace(scoped, "memory") is scoped

    def test_the_sdk_scopes_an_unscoped_instance(self) -> None:
        """The other half, and the reason the previous assertion is not vacuous."""
        storage = unit_storage()

        assert _resolve_namespace(storage, "memory") is not storage

    def test_the_scoped_instance_is_a_mongodbstorage_and_not_a_wrapper(self) -> None:
        """``_NamespacedStorage`` has exactly the five protocol methods, so wrapping would make
        ``delete_prefix`` unreachable (``REQ-E-031``) and turn ``list`` back into a full scan plus a
        Python-side prefix strip."""
        scoped = unit_storage().namespace("tenant-a")

        assert isinstance(scoped, MongoDBStorage)
        assert hasattr(scoped, "delete_prefix")

    def test_nesting_appends_rather_than_replaces(self) -> None:
        scoped = unit_storage().namespace("tenant-a").namespace("sessions")

        assert scoped._prefix == "tenant-a/sessions/"

    def test_the_scoped_instance_keeps_the_collection_and_the_ttl(self) -> None:
        """A namespace is a key-space scope, not a different collection or a different expiry."""
        scoped = unit_storage(collection="transcripts", ttl_seconds=60).namespace("tenant-a")

        assert scoped._collection_name == "transcripts"
        assert scoped._ttl_seconds == 60

    def test_the_scoped_instance_shares_the_connection(self) -> None:
        """``REQ-E-012``: it is built from the same config object, so it must not open a second pool."""
        storage = unit_storage()

        storage.namespace("tenant-a")

        assert len(_client._REGISTRY) == 1

    @pytest.mark.parametrize("prefix", ["tenant-a", "tenant-a/", "/tenant-a", "tenant-a//"])
    def test_the_prefix_is_normalized_to_one_trailing_slash(self, prefix: str) -> None:
        """The same convention ``_NamespacedStorage`` uses, so a prefix written either way produces one
        key layout. Four spellings of one prefix producing four key spaces is a data-loss bug that
        looks like a typo."""
        assert unit_storage().namespace(prefix)._prefix == "tenant-a/"


class TestKeysAreNormalizedBeforeTheyBecomeIds:
    """AT-018 / REQ-E-018 — adopted with the SDK's own ``_normalize_key``.

    This rejects and merges keys MongoDB would have kept: ``""`` is a legal ``_id`` and ``a//b`` and
    ``a/b`` are two documents. Normalizing anyway is the point. Every SDK-generated key is
    ``/``-separated, all three shipped backends normalize, and a key set that collides on
    ``LocalFileStorage`` while staying distinct here would break the protocol's one promise, that
    changing the backend does not change the data.
    """

    @pytest.mark.parametrize("key", ["a//b", "/a/b", "a/b/", "//a//b//"])
    async def test_equivalent_spellings_reach_the_driver_as_one_id(self, key: str) -> None:
        collection = FakeStorageCollection()
        storage = unit_storage(collection)

        await storage.write(key, b"payload")

        assert list(collection.documents) == ["a/b"]

    async def test_a_normalized_key_reads_back_whatever_spelling_wrote_it(self) -> None:
        collection = FakeStorageCollection()
        storage = unit_storage(collection)

        await storage.write("a//b", b"payload")

        assert await storage.read("/a/b/") == b"payload"

    @pytest.mark.parametrize("key", ["", "/", "//"])
    async def test_an_empty_key_raises_before_any_call_is_issued(self, key: str) -> None:
        collection = FakeStorageCollection()
        storage = unit_storage(collection)

        with pytest.raises(StorageError, match="empty"):
            await storage.write(key, b"payload")

        assert collection.calls == []

    @pytest.mark.parametrize("key", ["a/../b", "..", "../a"])
    async def test_a_dot_dot_segment_raises(self, key: str) -> None:
        """Meaningless for an ``_id`` and rejected anyway, because the shipped backends reject it and a
        key that traverses on one backend and does not on another is the same divergence again."""
        with pytest.raises(StorageError, match=r"\.\."):
            await unit_storage().write(key, b"payload")

    @pytest.mark.parametrize("method", ["read", "delete"])
    async def test_the_read_and_delete_paths_normalize_too(self, method: str) -> None:
        """Normalizing on write alone would write ``a/b`` and then miss it on ``read("a//b")``."""
        collection = FakeStorageCollection(documents=[{"_id": "a/b", "data": b"payload"}])
        storage = unit_storage(collection)

        await getattr(storage, method)("a//b")

        assert collection.calls[0][1][0] == {"_id": "a/b"}

    async def test_the_namespace_prefix_is_prepended_after_normalization(self) -> None:
        collection = FakeStorageCollection()
        storage = unit_storage(collection, prefix="tenant-a")

        await storage.write("/sessions//abc", b"payload")

        assert list(collection.documents) == ["tenant-a/sessions/abc"]


class TestListAndDeletePrefixNormalizeTheirArgument:
    """AT-019 / REQ-E-019 — ``_normalize_prefix``, which is a different function for a reason.

    A prefix is not a key: ``""`` is valid and matches everything, and a trailing slash is significant
    rather than noise. ``sessions/`` must not match ``sessions-archive/abc``, which is exactly what
    stripping the slash would do.
    """

    @pytest.mark.parametrize(("query", "expected"), [("a//b", "a/b"), ("/a/b", "a/b"), ("a/b/", "a/b/")])
    async def test_the_lower_bound_is_the_normalized_prefix(self, query: str, expected: str) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).list(query)

        assert collection.calls[0][1][0]["_id"]["$gte"] == expected

    async def test_an_empty_prefix_is_an_unbounded_query_rather_than_an_error(self) -> None:
        """``list("")`` is how the SDK's own ``KeywordSearchStrategy`` reads the whole store, so an
        empty prefix has to be a filter over everything and not a rejected argument."""
        collection = FakeStorageCollection(documents=[{"_id": "a"}, {"_id": "b"}])

        assert await unit_storage(collection).list("") == ["a", "b"]

    async def test_a_dot_dot_segment_in_a_prefix_raises(self) -> None:
        collection = FakeStorageCollection()

        with pytest.raises(StorageError, match=r"\.\."):
            await unit_storage(collection).list("a/../b")

        assert collection.calls == []

    async def test_delete_prefix_normalizes_the_same_way(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).delete_prefix("//a//")

        assert collection.calls[0][0] == "delete_many"
        assert collection.calls[0][1][0]["_id"]["$gte"] == "a/"

    async def test_a_trailing_slash_stays_significant(self) -> None:
        """The range for ``sessions/`` must exclude ``sessions-archive``, and the two differ only by the
        slash the normalizer is asked not to strip."""
        collection = FakeStorageCollection(documents=[{"_id": "sessions/abc"}, {"_id": "sessions-archive/abc"}])

        assert await unit_storage(collection).list("sessions/") == ["sessions/abc"]

    async def test_the_namespace_prefix_bounds_the_range_as_well(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection, prefix="tenant-a").list("sessions/")

        assert collection.calls[0][1][0]["_id"]["$gte"] == "tenant-a/sessions/"


class TestTheRangeBoundIsAnEncodableSuccessor:
    """AT-026b / REQ-E-026 — the two prefixes the one-line successor cannot express.

    The successor recipe on record is ``prefix[:-1] + chr(ord(prefix[-1]) + 1)``. Measured: at ``U+D7FF`` that returns a
    lone surrogate, which ``bson.encode`` refuses with ``UnicodeEncodeError``, and at ``U+10FFFF``
    ``chr`` raises ``ValueError``. Both come out of ``list``, which the protocol says returns a sorted
    list. Neither is reachable through an SDK-generated key and both are reachable through the
    protocol, which permits any string.
    """

    @pytest.mark.parametrize("prefix", ["a", "session/", "ké", "\ud7ff", "a\U0010ffff", "\U0010ffff"])
    async def test_every_bound_the_scan_emits_is_encodable(self, prefix: str) -> None:
        """The assertion is BSON's own encoder rather than a rule about surrogates, because the encoder
        is what the query has to survive."""
        collection = FakeStorageCollection()

        await unit_storage(collection).list(prefix)

        bson.encode(collection.calls[0][1][0])

    async def test_the_bound_skips_the_surrogate_block(self) -> None:
        """``U+D7FF`` increments to ``U+D800``, the first surrogate. The next scalar value is ``U+E000``,
        and nothing sorts between them, so skipping the block keeps the bound exact as well as
        encodable."""
        collection = FakeStorageCollection()

        await unit_storage(collection).list("\ud7ff")

        assert collection.calls[0][1][0]["_id"]["$lt"] == "\ue000"

    async def test_a_prefix_of_the_maximum_code_point_drops_the_upper_bound(self) -> None:
        """``$gte`` alone is already exact there: nothing sorts above a prefix of ``U+10FFFF`` except
        keys that start with it."""
        collection = FakeStorageCollection()

        await unit_storage(collection).list("\U0010ffff")

        assert "$lt" not in collection.calls[0][1][0]["_id"]

    async def test_a_carry_past_the_maximum_shortens_the_bound(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).list("a\U0010ffff")

        assert collection.calls[0][1][0]["_id"]["$lt"] == "b"


class TestTheCallsAndTheDocumentMatchPlanSection66:
    """Unit companions to `AT-020`, `AT-035` / REQ-E-020, REQ-E-025, REQ-E-035.

    What the contract tier cannot see. A deployment reports the same state after one ``replace_one``
    with ``upsert=True`` as after a delete plus an insert, and a document read back cannot distinguish
    a field that was never written from one the server has already dropped. So the outcome is asserted
    there and the call is asserted here.
    """

    async def test_write_is_one_upserting_replace(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).write("a/b", b"payload")

        assert collection.names_called == ["replace_one"]
        assert collection.calls[0][1][0] == {"_id": "a/b"}
        assert collection.calls[0][2] == {"upsert": True}

    async def test_the_document_carries_the_payload_and_a_timestamp(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).write("a/b", b"payload")

        document = collection.documents["a/b"]
        assert document["data"] == b"payload"
        assert isinstance(document["updated_at"], datetime.datetime)

    async def test_the_timestamp_is_timezone_aware_and_utc(self) -> None:
        """BSON stores a UTC instant with no zone, so a naive local ``datetime`` is written as if it
        were UTC and comes back shifted by the writer's offset. Reading ``updated_at`` is not part of
        the protocol, but an operator sorting on it is the obvious reason it is there."""
        collection = FakeStorageCollection()

        await unit_storage(collection).write("a/b", b"payload")

        assert collection.documents["a/b"]["updated_at"].tzinfo is datetime.timezone.utc

    async def test_no_expiry_field_is_written_without_a_ttl(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).write("a/b", b"payload")

        assert "expires_at" not in collection.documents["a/b"]

    async def test_a_ttl_writes_the_field_the_ttl_index_watches(self) -> None:
        """The field name is read from ``_indexes`` rather than repeated, because the two have to agree
        for a document to ever expire and nothing else checks that they do. A TTL index on
        ``expires_at`` beside documents carrying ``expiresAt`` is a store that grows forever while
        reporting every index present."""
        collection = FakeStorageCollection()

        await unit_storage(collection, ttl_seconds=60).write("a/b", b"payload")

        assert _indexes.EXPIRES_FIELD in collection.documents["a/b"]

    async def test_the_expiry_is_the_ttl_ahead_of_the_timestamp(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection, ttl_seconds=60).write("a/b", b"payload")

        document = collection.documents["a/b"]
        assert document["expires_at"] - document["updated_at"] == datetime.timedelta(seconds=60)

    async def test_read_returns_the_payload_as_bytes(self) -> None:
        """``Binary`` is a ``bytes`` subclass, so an equality assertion alone would pass for a value the
        protocol's ``bytes | None`` does not describe."""
        collection = FakeStorageCollection()
        storage = unit_storage(collection)

        await storage.write("a/b", b"payload")
        found = await storage.read("a/b")

        assert type(found) is bytes

    async def test_read_of_a_missing_key_is_one_find_one_and_no_error(self) -> None:
        collection = FakeStorageCollection()

        assert await unit_storage(collection).read("a/b") is None
        assert collection.names_called == ["find_one"]

    async def test_delete_ignores_the_deleted_count(self) -> None:
        """``REQ-E-023``: absent and deleted are the same outcome, and ``delete_one`` reports
        ``deleted_count`` 0 for the first. Reading it would turn a no-op into an error."""
        collection = FakeStorageCollection()

        await unit_storage(collection).delete("a/b")

        assert collection.names_called == ["delete_one"]

    async def test_list_projects_only_the_id_and_sorts_ascending(self) -> None:
        """The projection is what keeps a ``list`` over a store of 4 MB payloads from pulling every
        payload across the wire to return their names, and the sort is what the protocol promises."""
        collection = FakeStorageCollection()

        await unit_storage(collection).list("a/")

        assert collection.calls[0][1][1] == {"_id": 1}
        assert collection.cursors[0].sort_calls == [("_id", 1)]

    async def test_list_strips_the_namespace_prefix_and_keeps_the_query_prefix(self) -> None:
        """``REQ-E-026`` and ``REQ-E-030`` together, and they pull in opposite directions: a scoped
        store's keys come back without its namespace, and the queried prefix stays on the front. One
        strip of ``len(self._prefix)`` satisfies both; stripping the whole filter satisfies neither."""
        collection = FakeStorageCollection(documents=[{"_id": "tenant-a/sessions/abc"}])

        found = await unit_storage(collection, prefix="tenant-a").list("sessions/")

        assert found == ["sessions/abc"]

    async def test_delete_prefix_is_one_call_returning_the_count(self) -> None:
        collection = FakeStorageCollection(documents=[{"_id": "a/one"}, {"_id": "a/two"}, {"_id": "b/three"}])

        assert await unit_storage(collection).delete_prefix("a/") == 2
        assert collection.names_called == ["delete_many"]


class TestTheKeyCeiling:
    """AT-037 / REQ-E-037 — 1024 bytes UTF-8, our policy and not the server's.

    MongoDB 8.3.8 accepts a 1,000,000-byte ``_id`` and round-trips it, so nothing upstream fails when
    this check is missing or wrong. The cost lands on every ``list`` range scan instead, and on any
    deployment pinned below FCV 4.2 it lands as a write error. That is what makes the UTF-8 case below
    the one worth writing first: a ceiling checked with ``len(key)`` passes every ASCII test in this
    class and admits a key nearly four times the limit.
    """

    async def test_a_key_at_the_ceiling_is_accepted(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).write("k" * 1024, b"payload")

        assert collection.names_called == ["replace_one"]

    async def test_a_key_one_byte_over_the_ceiling_raises_before_any_call(self) -> None:
        collection = FakeStorageCollection()

        with pytest.raises(StorageError) as caught:
            await unit_storage(collection).write("k" * 1025, b"payload")

        assert collection.calls == []
        assert "1025" in str(caught.value)
        assert "1024" in str(caught.value)

    async def test_the_ceiling_counts_utf8_bytes_and_not_characters(self) -> None:
        """600 characters, 1200 bytes. ``len`` says 600 and the encoder says 1200, and only one of those
        is what MongoDB stores."""
        collection = FakeStorageCollection()

        with pytest.raises(StorageError) as caught:
            await unit_storage(collection).write("é" * 600, b"payload")

        assert collection.calls == []
        assert "1200" in str(caught.value)

    async def test_the_namespace_prefix_counts_toward_the_ceiling(self) -> None:
        """The bytes that reach the index are the prefixed ``_id``, so a scoped store measuring only the
        caller's half would write index entries the ceiling exists to prevent."""
        collection = FakeStorageCollection()

        with pytest.raises(StorageError):
            await unit_storage(collection, prefix="tenant-a").write("k" * 1020, b"payload")

        assert collection.calls == []

    async def test_the_message_shortens_the_key_instead_of_quoting_all_of_it(self) -> None:
        """Written by a mutation survivor: every assertion above matched on the two numbers, so a message
        that pasted all 1025 characters in passed them all.

        The message goes to a log, and a rejected key is unbounded in length: the protocol permits any
        string, and the reason this one was rejected is that it is long. Quoting it whole puts the entire
        key on the line that was supposed to explain the limit. The first 60 characters identify which
        key it was, and the count says how much was dropped so nobody reads the truncated form as the
        whole key.
        """
        collection = FakeStorageCollection()

        with pytest.raises(StorageError) as caught:
            await unit_storage(collection).write("k" * 1025, b"payload")

        message = str(caught.value)
        assert "k" * 1025 not in message
        assert "k" * 60 in message
        assert "[+965 more]" in message
        assert len(message) < 300

    @pytest.mark.parametrize("method", ["read", "delete"])
    async def test_the_read_and_delete_paths_check_it_too(self, method: str) -> None:
        """Not symmetry for its own sake. A key too long to have been written cannot match anything, so
        raising is both cheaper and more informative than a round trip that returns ``None``."""
        collection = FakeStorageCollection()

        with pytest.raises(StorageError):
            await getattr(unit_storage(collection), method)("k" * 1025)

        assert collection.calls == []


class TestTheSizePreflight:
    """`AT-038a` and the unit companion to `AT-029` / REQ-E-029, REQ-E-038.

    Two claims, and the second is only sound because of the first. ``write`` rejects an oversized
    payload before issuing the call, and it decides that from arithmetic rather than by encoding the
    document to measure it, which would double the peak memory of the one call most likely to be near
    a memory limit already. The arithmetic is only legitimate if it agrees with the encoder, so that is
    asserted against ``bson.encode`` over the document ``write`` actually built.
    """

    @pytest.mark.parametrize(
        ("key", "payload", "ttl_seconds"),
        [
            ("k", b"", None),
            ("k", b"x", None),
            ("ké", b"x", None),
            ("k\U0001f600", b"x", None),
            ("k" * 1024, b"y" * (4 * 1024 * 1024), None),
            ("k", b"", 60),
            ("k" * 1024, b"y" * (4 * 1024 * 1024), 60),
        ],
        ids=["empty", "one-byte", "two-byte-key", "four-byte-key", "large", "ttl", "ttl-large"],
    )
    async def test_the_arithmetic_equals_what_the_encoder_produces(
        self, key: str, payload: bytes, ttl_seconds: int | None
    ) -> None:
        """AT-038a. The document comes from the module under test rather than from a copy of the schema here,
        so a change to the shape moves both sides and the two constants fail together."""
        collection = FakeStorageCollection()

        await unit_storage(collection, ttl_seconds=ttl_seconds).write(key, payload)

        document = collection.documents[key]
        assert storage_module._encoded_size(key, payload, expires=ttl_seconds is not None) == len(bson.encode(document))

    async def test_a_document_at_the_ceiling_is_accepted(self) -> None:
        """The boundary is inclusive: ``maxBsonObjectSize`` is the largest document the server takes,
        not the first it refuses. An exclusive check would reject a 16 MiB document MongoDB stores."""
        collection = FakeStorageCollection()
        payload = b"z" * (storage_module.MAX_DOCUMENT_BYTES - storage_module._DOCUMENT_OVERHEAD_BYTES - len("a/b"))

        await unit_storage(collection).write("a/b", payload)

        assert len(bson.encode(collection.documents["a/b"])) == storage_module.MAX_DOCUMENT_BYTES

    async def test_one_byte_over_raises_naming_the_key_the_size_and_the_limit(self) -> None:
        collection = FakeStorageCollection()
        payload = b"z" * (storage_module.MAX_DOCUMENT_BYTES - storage_module._DOCUMENT_OVERHEAD_BYTES - len("a/b") + 1)

        with pytest.raises(StorageError) as caught:
            await unit_storage(collection).write("a/b", payload)

        message = str(caught.value)
        assert collection.calls == []
        assert "a/b" in message
        assert str(storage_module.MAX_DOCUMENT_BYTES + 1) in message
        assert str(storage_module.MAX_DOCUMENT_BYTES) in message

    async def test_a_ttl_lowers_the_payload_the_ceiling_allows(self) -> None:
        """``expires_at`` costs 20 bytes in the document, so the same payload that fits without a TTL can
        exceed the ceiling with one. A preflight that ignored the field would pass this call to the
        driver and surface ``DocumentTooLarge`` instead of our message."""
        collection = FakeStorageCollection()
        payload = b"z" * (storage_module.MAX_DOCUMENT_BYTES - storage_module._DOCUMENT_OVERHEAD_BYTES - len("a/b"))

        with pytest.raises(StorageError):
            await unit_storage(collection, ttl_seconds=60).write("a/b", payload)

        assert collection.calls == []

    def test_the_ceiling_constant_is_sixteen_mebibytes(self) -> None:
        """``AT-038b`` pins this to what the deployment advertises. This half pins it to the number,
        because a constant the contract tier cannot reach when ``MONGODB_URI`` is unset should still
        fail loudly if someone edits it."""
        assert storage_module.MAX_DOCUMENT_BYTES == 16 * 1024 * 1024
        assert storage_module.MAX_KEY_BYTES == 1024


class TestDriverFailuresBecomeStorageErrors:
    """AT-022, AT-036 / REQ-E-022, REQ-E-036 — the driver's error taxonomy.

    ``StorageError`` is what the protocol documents, and a caller who configured this package with a
    dict should not have to import pymongo to handle a failure. The original stays reachable through
    ``__cause__``, so nothing is hidden; what changes is which exception the caller has to name.

    Mock-driven on purpose, following ``S3Storage``: a transport failure is not something a healthy
    deployment can be asked to produce on cue, and the contract tier's job is the happy path against a
    server that works.
    """

    @pytest.mark.parametrize("call", ["write", "read", "delete", "list", "delete_prefix", "search"])
    async def test_every_entry_point_raises_storage_error_with_the_cause_chained(self, call: str) -> None:
        failure = network_failure()
        storage = unit_storage(FakeStorageCollection(error=failure))
        argument = b"payload" if call == "write" else None

        with pytest.raises(StorageError) as caught:
            if argument is None:
                await getattr(storage, call)("a/b")
            else:
                await getattr(storage, call)("a/b", argument)

        assert caught.value.__cause__ is failure

    async def test_a_failure_while_the_cursor_drains_is_wrapped_too(self) -> None:
        """The shape an error wrapper attached to the call alone lets through. ``find`` returns without
        contacting the server, so a connection that dies mid-scan raises from the ``async for``, and a
        helper that wraps one awaitable never sees it. That is why ``list`` decides the wrapping style
        for the whole module (see the module docstring)."""
        failure = network_failure()
        collection = FakeStorageCollection(documents=[{"_id": "a/one"}], iteration_error=failure)

        with pytest.raises(StorageError) as caught:
            await unit_storage(collection).list("a/")

        assert caught.value.__cause__ is failure

    async def test_the_message_names_the_operation_the_key_and_the_namespace(self) -> None:
        """Three facts, because a wrapped error that says only "failed" costs the reader the round trip
        the wrapping was supposed to save."""
        storage = unit_storage(FakeStorageCollection(error=network_failure()), collection="transcripts")

        with pytest.raises(StorageError) as caught:
            await storage.read("a/b")

        message = str(caught.value)
        assert "a/b" in message
        assert "appdb.transcripts" in message

    async def test_the_message_shape_is_the_one_the_rest_of_the_package_uses(self) -> None:
        """``_indexes._guard`` builds the same sentence for index work, and a user reading a log has no
        reason to know which module produced which line. Nothing but this assertion connects the two,
        because they are deliberately separate functions: ``_guard`` wraps one awaitable and this module
        wraps a whole method, so there is no shared helper to hold the format string.
        """

        async def already_failing() -> None:
            raise network_failure()

        with pytest.raises(StorageError) as from_indexes:
            await _indexes._guard(already_failing(), what="reading indexes on appdb.storage")

        with pytest.raises(StorageError) as from_storage:
            await unit_storage(FakeStorageCollection(error=network_failure())).read("a/b")

        for caught in (from_indexes, from_storage):
            message = str(caught.value)
            assert message.startswith("strands_mongodb: ")
            assert " failed: socket timed out" in message

    async def test_a_non_driver_error_is_not_swallowed(self) -> None:
        """The wrapper catches ``PyMongoError`` and nothing wider. A ``KeyboardInterrupt`` or a bug in
        this package reported as a storage failure is a debugging session spent on the database."""
        collection = FakeStorageCollection(error=RuntimeError("bug in this package"))

        with pytest.raises(RuntimeError, match="bug in this package"):
            await unit_storage(collection).read("a/b")


def assert_no_leak(*subjects: object) -> None:
    """Assert the password appears in none of the given renderings.

    A copy of ``test_client.py``'s helper rather than an import. Reaching across test modules ties the
    two files' collection order together for six lines of assertion, and the failure it guards against
    is one this file has to keep testing whatever happens to that one.

    Args:
        *subjects: Objects whose ``repr`` must not contain the password, plus any strings to check
            directly. Strings are checked as themselves, so an exception message or a rendered
            traceback can be passed straight in.

    Raises:
        AssertionError: Naming the leaking subject, without reproducing the password in the failure
            message, which would leak it into CI output.
    """
    for index, subject in enumerate(subjects):
        text = subject if isinstance(subject, str) else repr(subject)
        assert "SUPERSECRET" not in text, f"subject {index} ({type(subject).__name__}) leaked the password"


class TestNoStorageMethodRendersThePassword:
    """AT-014's storage half / REQ-E-014 — the credential sweep, which the connection layer left open.

    ``test_client.py`` closed this for construction and for the registry. The five contract methods are
    the surface that was still missing, and they are the ones that fail in production: a store built
    once at start-up, then a driver error on every call, each one rendered into a log line or a
    traceback that ends up in a ticket.
    """

    @pytest.mark.parametrize("call", ["write", "read", "delete", "list", "delete_prefix", "search"])
    async def test_a_wrapped_driver_failure_renders_no_password(self, call: str) -> None:
        storage = unit_storage(FakeStorageCollection(error=network_failure()), config={"connection_string": LEAKY_URI})

        with pytest.raises(StorageError) as caught:
            if call == "write":
                await storage.write("a/b", b"payload")
            else:
                await getattr(storage, call)("a/b")

        assert_no_leak(str(caught.value), caught.value, "".join(traceback.format_exception(caught.value)))

    async def test_the_store_and_its_namespaced_view_render_no_password(self) -> None:
        """``repr`` is where this leaks by accident: a ``__repr__`` added later for debugging, or a
        dataclass decorator, prints every field including ``_config``.

        The config itself is not a subject here. It holds the password by construction, which is the
        premise ``REQ-E-014`` starts from: the requirement is that the password never leaves it.
        """
        storage = unit_storage(config={"connection_string": LEAKY_URI})

        assert_no_leak(storage, storage.namespace("tenant-a"), storage._connection)

    @pytest.mark.parametrize(
        ("key", "payload"),
        [("k" * 1025, b"x"), ("a/b", b"z" * (17 * 1024 * 1024)), ("", b"x"), ("a/../b", b"x")],
        ids=["long-key", "large-payload", "empty-key", "traversal"],
    )
    async def test_a_validation_error_renders_no_password(self, key: str, payload: bytes) -> None:
        """The errors this package raises on its own, without a driver failure. They carry the key, and
        the natural way to make one useful is to name the store that rejected it."""
        storage = unit_storage(config={"connection_string": LEAKY_URI})

        with pytest.raises(StorageError) as caught:
            await storage.write(key, payload)

        assert_no_leak(str(caught.value), caught.value)


class TestSearchIsOneAggregation:
    """AT-032a, AT-032c — the pipeline this package emits, and the round trips it does not make.

    ``REQ-E-032``'s whole content is a count: **one** server-side aggregation where the SDK's default
    does ``list("")`` plus one ``read`` per key. So the primary assertion is a call count, not a result
    set. What MongoDB computes from the pipeline is the contract tier's job (``AT-032b`` onward); this
    tier asserts the four stages, their order, and the two cases where a stage is absent.
    """

    WORD = r"[\p{L}\p{N}_]+"

    @staticmethod
    def _stages(collection: FakeStorageCollection) -> list[str]:
        """The stage names in order, which is the part of the pipeline worth naming in a failure."""
        return [next(iter(stage)) for stage in collection.pipelines[0]]

    async def test_search_issues_one_aggregate_and_no_find(self) -> None:
        """``AT-032a``. The assertion that separates this from the inherited default: a store answering
        with ``find`` at all has fallen back to reading the key space into Python."""
        collection = FakeStorageCollection(aggregate_result=[{"_id": "notes/dark.md", "score": 2}])

        await unit_storage(collection).search("dark mode")

        assert collection.names_called == ["aggregate"]

    async def test_the_pipeline_is_project_match_sort_with_no_match_for_an_unnamespaced_store(self) -> None:
        """``_range_for("")`` is ``{}`` for a store with no prefix, and an empty ``$match`` is a stage the
        server still has to run. It is dropped rather than sent."""
        collection = FakeStorageCollection()

        await unit_storage(collection).search("dark")

        assert self._stages(collection) == ["$project", "$match", "$sort"]

    async def test_a_namespaced_store_bounds_the_scan_with_its_prefix_range_first(self) -> None:
        """The measured difference between a covered ``IXSCAN`` and a ``COLLSCAN``: with the range in
        front, ``explain`` reports ``IXSCAN(_id_)``; without it, ``COLLSCAN`` (``AT-032e``)."""
        collection = FakeStorageCollection()

        await unit_storage(collection, prefix="tenant-a").search("dark")

        pipeline = collection.pipelines[0]
        assert self._stages(collection) == ["$match", "$project", "$match", "$sort"]
        assert pipeline[0] == {"$match": {"_id": {"$gte": "tenant-a/", "$lt": "tenant-a0"}}}

    async def test_the_tokenizer_regex_is_the_unicode_property_form_not_backslash_w(self) -> None:
        """The finding that made this test exist. MongoDB's ``\\w`` is ASCII-only by default, so the
        obvious translation of the SDK's ``re.split(r"\\W+", ...)`` tokenizes ``café`` to ``caf`` and
        drops ``日本語`` entirely. ``AT-032d`` proves the parity against a deployment; this pins the
        pattern so a refactor cannot quietly swap it back.
        """
        collection = FakeStorageCollection()

        await unit_storage(collection).search("dark")

        # Matched against the values rather than ``repr``, which doubles every backslash and would make
        # this pass for a pattern that is not the one being sent.
        strings = set(_strings(collection.pipelines[0]))
        assert self.WORD in strings
        assert not any("\\w" in text for text in strings)

    async def test_query_tokens_are_lowercased_deduplicated_and_sorted(self) -> None:
        """Sorted for one reason only: two callers asking the same question should produce the same
        pipeline, so the server's plan cache sees one shape rather than one per token order."""
        collection = FakeStorageCollection()

        await unit_storage(collection).search("Dark DARK mode")

        assert ["dark", "mode"] in [operand for stage in collection.pipelines[0] for operand in _string_lists(stage)]

    async def test_only_keys_scoring_above_zero_survive_and_ties_break_by_key(self) -> None:
        collection = FakeStorageCollection()

        await unit_storage(collection).search("dark")

        pipeline = collection.pipelines[0]
        assert pipeline[-2] == {"$match": {"score": {"$gt": 0}}}
        assert pipeline[-1] == {"$sort": {"score": -1, "_id": 1}}

    async def test_the_projection_asks_for_no_field_that_would_end_the_covered_scan(self) -> None:
        """``AT-032e``'s unit-tier half. ``docsExamined`` is 0 only while ``_id`` is the sole field the
        pipeline reads; naming ``data`` here would make the server fetch every matching document, and no
        assertion about results would change."""
        collection = FakeStorageCollection()

        await unit_storage(collection).search("dark")

        assert "$data" not in set(_strings(collection.pipelines[0]))
        assert set(_strings(collection.pipelines[0])) & {"$_id", "$$this.match"} == {"$_id", "$$this.match"}

    @pytest.mark.parametrize("query", ["", "   ", "!!! ---", "\n\t"])
    async def test_a_query_that_tokenizes_to_nothing_costs_zero_round_trips(self, query: str) -> None:
        """``AT-032c`` (``REQ-E-032a``). The result is ``[]`` either way, which is why this needs its own
        assertion: without the short circuit the server scans the whole namespace, scores every key zero
        and the ``$match`` throws all of it away. Matches ``KeywordSearchStrategy``'s own behaviour.
        """
        collection = FakeStorageCollection(documents=[{"_id": "notes/dark.md", "data": b"x"}])

        found = await unit_storage(collection).search(query)

        assert found == []
        assert collection.names_called == []

    async def test_results_carry_the_key_without_the_instance_prefix(self) -> None:
        """The same strip ``list`` does. A caller has to be able to hand the key straight back to
        ``read``, and the prefix is this instance's business rather than theirs (``REQ-E-030``)."""
        collection = FakeStorageCollection(aggregate_result=[{"_id": "tenant-a/notes/dark.md", "score": 2}])

        found = await unit_storage(collection, prefix="tenant-a").search("dark")

        assert [result.key for result in found] == ["notes/dark.md"]

    async def test_the_score_is_a_float_and_data_is_left_unset(self) -> None:
        """``REQ-E-032b``. ``$size`` returns a BSON int32 and ``StorageSearchResult.score`` is annotated
        ``float``; the SDK's own strategy passes the ``int`` through, so honouring the annotation is a
        deliberate divergence and ``3 == 3.0`` means only a ``type`` check can see it.
        """
        collection = FakeStorageCollection(aggregate_result=[{"_id": "notes/dark.md", "score": 2}])

        found = await unit_storage(collection).search("dark")

        assert type(found[0].score) is float
        assert found[0].data is None

    async def test_a_failing_aggregation_becomes_a_storage_error(self) -> None:
        """``REQ-E-036`` covers all five contract methods, and ``search`` was the one still delegating to
        the SDK, so it had no wrapper of its own to test."""
        collection = FakeStorageCollection(error=network_failure())

        with pytest.raises(StorageError) as caught:
            await unit_storage(collection).search("dark")

        assert str(caught.value).startswith("strands_mongodb: searching 'dark' in appdb.storage failed:")
        assert isinstance(caught.value.__cause__, Exception)


def _strings(value: Any) -> Iterator[str]:
    """Every string anywhere inside a nested pipeline stage, keys included.

    Assertions here compare values rather than ``repr(pipeline)``, because ``repr`` escapes the
    backslashes in the tokenizer pattern and a substring test against it passes for the wrong regex.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)


def _string_lists(value: Any) -> Iterator[list[str]]:
    """Every list-of-strings anywhere inside a nested pipeline stage.

    The query tokens sit several levels down inside ``$setIntersection``, and a test that walked the
    exact path would break on any reshaping of the stage while telling us nothing about the tokens.
    """
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            yield [str(item) for item in value]
        for item in value:
            yield from _string_lists(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _string_lists(item)


class TestEnsureIndexesIsExplicitAndIdempotent:
    """AT-121, AT-123 / ``REQ-E-119`` — the method that creates the storage TTL index.

    ``ttl_seconds`` has stamped ``expires_at`` on every document it writes from the start and nothing in
    this package has ever created the index that acts on it, so a caller who set the argument got a
    field and no expiry. The fix is a method rather than work folded into the constructor or the first
    ``write``: ``Storage``'s round-trip count is its design and a store built by the SDK is built
    in a code path the user does not see, so an index creation hidden there is DDL nobody asked for.

    The contract half lives in ``TestEnsureIndexesAgainstARealDeployment`` below, because whether the
    server then *removes* the document is the server's behaviour and no fake can testify to it.
    """

    async def test_it_leaves_the_ttl_index_the_storage_specs_declare(self) -> None:
        """The outcome ``REQ-E-119`` states: after the call, the index is there with its option set.

        Asserted against the collection's own index state rather than against the ``create_indexes``
        argument, which is the proxy. ``expireAfterSeconds: 0`` is the option that matters; without it
        the index exists and expires nothing, which is the same user-visible failure as having no index.
        """
        collection = FakeStorageCollection()
        storage = unit_storage(collection)

        await storage.ensure_indexes()

        assert collection.indexes["strands_expires_at_ttl"]["key"] == [(_indexes.EXPIRES_FIELD, 1)]
        assert collection.indexes["strands_expires_at_ttl"]["expireAfterSeconds"] == 0

    async def test_a_second_call_creates_nothing(self) -> None:
        """``REQ-E-119``'s idempotence, asserted as the absence of the write rather than as a flag.

        No ``_initialized`` attribute is involved and none should be: the guarantee is against the
        server, so a second call reads the indexes and finds the spec already satisfied. A fake that
        did not normalise ``IndexModel.document`` into the shape ``index_information`` reports would
        pass the first assertion and fail this one, which is the point of the normalisation.
        """
        collection = FakeStorageCollection()
        storage = unit_storage(collection)

        await storage.ensure_indexes()
        first = collection.names_called.count("create_indexes")
        await storage.ensure_indexes()

        assert first == 1
        assert collection.names_called.count("create_indexes") == 1
        assert collection.names_called.count("index_information") == 2

    async def test_verify_mode_raises_with_the_command_that_would_fix_it(self) -> None:
        """AT-123 / ``REQ-E-119`` and ``REQ-E-061`` — ``ensure_indexes: False`` is verify, not skip.

        The caller this is for has no DDL rights, so the useful answer is a work order. Three claims:
        it raises rather than returning quietly, the message names the index so a ticket can be
        written against it, and it carries the ``createIndexes`` command so the operator does not have
        to reconstruct one. Nothing is created on the way out.
        """
        collection = FakeStorageCollection()
        storage = unit_storage(collection, config={"ensure_indexes": False})

        with pytest.raises(StorageError) as raised:
            await storage.ensure_indexes()

        message = str(raised.value)
        assert "strands_expires_at_ttl" in message
        assert "createIndexes" in message
        assert "create_indexes" not in collection.names_called


class TestTheIndexReadyBudgetReachesEnsure:
    """AT-126 / ``REQ-E-122`` — this class's half of the key, which it passes without ever waiting.

    ``storage_specs()`` is one ordinary index and ``_ensure_ordinary`` has no poll loop, so the number
    changes nothing here today. It is passed anyway, and asserted anyway, because the key is read from
    one config that a caller may also hand to a ``MongoDBMemoryStore`` (``REQ-E-012``): a store that
    quietly ignored it would make the same config mean two things. It also means the assertion still
    holds if the storage collection ever gains something that builds asynchronously.
    """

    @staticmethod
    def _recording(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Replace ``_indexes.ensure`` with a spy, on the module rather than on the instance.

        The module attribute is the seam the production code reaches through, so this is what a call
        that bypassed the wiring would miss.
        """
        calls: list[dict[str, Any]] = []

        async def spy(collection: Any, **kwargs: Any) -> None:
            calls.append(kwargs)

        monkeypatch.setattr(_indexes, "ensure", spy)
        return calls

    async def test_the_configured_budget_is_the_one_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._recording(monkeypatch)
        storage = unit_storage(config={"index_ready_timeout": 7})

        await storage.ensure_indexes()

        assert [call["timeout"] for call in calls] == [7]

    async def test_absence_resolves_to_the_module_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read through the constant, so this asserts the wiring rather than restating ``AT-125``."""
        calls = self._recording(monkeypatch)
        storage = unit_storage()

        await storage.ensure_indexes()

        assert [call["timeout"] for call in calls] == [_indexes.READY_TIMEOUT_SECONDS]


class TestNoOrdinaryOperationTouchesAnIndex:
    """AT-124 / ``REQ-E-121``, ``INV-007`` — the guarantee that keeps ``ensure_indexes()`` explicit.

    The requirement is a negative, and a negative is the thing a test suite silently stops covering.
    The mechanism here is the fake's own call log: every driver method it offers is recorded, so this
    walks all six ``Storage``-facing operations and asserts the log holds nothing about indexes.

    It is not merely a restatement of "we did not write that code". It is what fails if a later change
    makes ``write`` self-provisioning, which is the one plausible way somebody creates that index a
    second time and doubles every write's round-trip count in the process.
    """

    async def test_the_six_operations_neither_read_nor_create_an_index(self) -> None:
        collection = FakeStorageCollection(documents=[{"_id": "k", "data": b"v"}])
        storage = unit_storage(collection, ttl_seconds=60)

        await storage.write("k", b"v")
        await storage.read("k")
        await storage.delete("k")
        await storage.list("")
        await storage.search("v")
        await storage.delete_prefix("")

        assert "index_information" not in collection.names_called
        assert "create_indexes" not in collection.names_called
        # The positive half, so the test cannot pass by the operations having quietly done nothing.
        assert set(collection.names_called) == {
            "replace_one",
            "find_one",
            "delete_one",
            "find",
            "aggregate",
            "delete_many",
        }


class TestTheFakeMatchesTheDriver:
    """MP-001 — the parity contract for ``FakeStorageCollection`` and ``FakeCursor``.

    Same mechanical shape as ``test_indexes.py``'s, and for the same reason: a fake that returns a
    value where the driver returns a coroutine passes every unit test in this file and fails on the
    first real call. Two of the five methods here are the awkward ones. ``find`` is **not** a coroutine
    function, so awaiting it is a ``TypeError`` upstream and harmless against a loose fake, and
    ``AsyncCursor.sort`` is not one either, so the natural ``await ...sort("_id", 1)`` fails the same
    way.
    """

    # Every driver method ``storage.py`` calls, and the arguments it calls them with.
    CALLS: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...] = (
        ("replace_one", ({"_id": "k"}, {"_id": "k"}), {"upsert": True}),
        ("find_one", ({"_id": "k"},), {}),
        ("delete_one", ({"_id": "k"},), {}),
        ("delete_many", ({"_id": {"$gte": "a", "$lt": "b"}},), {}),
        # ``ensure_indexes()``'s two, reached through ``_indexes.ensure`` rather than called from
        # ``storage.py`` directly. They are here because the fake is what stands in for the driver when
        # ``AT-121`` and ``AT-123`` run, so the same coroutine-function trap applies to them.
        ("index_information", (), {}),
        ("create_indexes", ([],), {}),
    )

    @pytest.mark.parametrize(("method", "args", "kwargs"), CALLS)
    def test_the_awaited_methods_are_coroutine_functions_on_both(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        assert inspect.iscoroutinefunction(getattr(self._collection_class(), method)), method
        assert inspect.iscoroutinefunction(getattr(FakeStorageCollection, method)), method

    @pytest.mark.parametrize(("method", "args", "kwargs"), CALLS)
    def test_the_real_signature_accepts_the_arguments_we_pass(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """A fake with ``**kwargs`` accepts anything. ``upsert`` is the one that matters, because
        ``REQ-E-020``'s overwrite depends on it and a misspelling would silently insert."""
        inspect.signature(getattr(self._collection_class(), method)).bind(None, *args, **kwargs)

    def test_find_is_not_a_coroutine_function_on_either(self) -> None:
        """Measured on pymongo 4.17.0: ``find`` builds a cursor without contacting the server, so it is
        an ordinary method. ``await collection.find(...)`` raises ``TypeError`` upstream."""
        assert not inspect.iscoroutinefunction(self._collection_class().find)
        assert not inspect.iscoroutinefunction(FakeStorageCollection.find)

    def test_the_real_find_accepts_a_filter_and_a_projection_positionally(self) -> None:
        inspect.signature(self._collection_class().find).bind(None, {"_id": {"$gte": "a"}}, {"_id": 1})

    def test_sort_is_not_a_coroutine_function_and_returns_the_cursor(self) -> None:
        """The second trap. ``AsyncCursor.sort`` mutates the cursor and returns it, so the whole chain
        stays synchronous until iteration begins."""
        cursor_class = self._cursor_class()

        assert not inspect.iscoroutinefunction(cursor_class.sort)
        assert not inspect.iscoroutinefunction(FakeCursor.sort)
        assert isinstance(FakeCursor([]).sort("_id", 1), FakeCursor)

    def test_the_real_sort_accepts_a_field_and_a_direction(self) -> None:
        inspect.signature(self._cursor_class().sort).bind(None, "_id", 1)

    def test_both_cursors_are_async_iterable_and_neither_is_synchronously_iterable(self) -> None:
        """The half that matters. ``AsyncCursor`` has no ``__iter__``, so ``for key in cursor`` is a
        ``TypeError`` upstream and a silently passing test against a fake that grew one."""
        cursor_class = self._cursor_class()

        assert hasattr(cursor_class, "__aiter__")
        assert not hasattr(cursor_class, "__iter__")
        assert hasattr(FakeCursor([]), "__aiter__")
        assert not hasattr(FakeCursor([]), "__iter__")

    def test_the_fake_collection_raises_for_an_unknown_attribute(self) -> None:
        """``AsyncCollection.__getattr__`` resolves any unknown name to a sub-collection, so a typo is
        truthy and never raises. The fake is stricter on purpose."""
        with pytest.raises(AttributeError):
            FakeStorageCollection().definitely_not_a_method  # type: ignore[attr-defined]  # noqa: B018

    def test_both_collections_expose_a_database_carrying_a_name(self) -> None:
        """``_ensure_ordinary`` builds its namespace from ``collection.database.name``.

        Two hops, both of which the fake has to offer for ``AT-121`` and ``AT-123`` to mean anything.
        ``database`` is a property on the real class rather than a method, so this checks the attribute
        rather than binding a signature, and the second assertion is the one that catches a fake whose
        ``database`` is a bare string.
        """
        assert isinstance(self._collection_class().database, property)
        assert FakeStorageCollection().database.name == "appdb"

    def test_the_delete_result_carries_the_attribute_delete_prefix_reads(self) -> None:
        """``REQ-E-031`` returns a count, and ``DeleteResult.deleted_count`` is where it comes from."""
        from pymongo.results import DeleteResult

        assert "deleted_count" in dir(DeleteResult)
        assert FakeDeleteResult(2).deleted_count == 2

    # Both helpers return ``Any`` rather than ``type``. The point of every test above is to reach for
    # an attribute the driver may not have, and ``type`` resolves attribute access against ``type``
    # itself, so ``_collection_class().find`` would be a type error rather than the runtime question
    # these tests are asking. They are also imported inside the helpers rather than at module scope:
    # the driver's async classes are private-ish paths, and keeping the import here means a rename
    # upstream fails these four tests instead of stopping the whole module from importing.
    @staticmethod
    def _collection_class() -> Any:
        from pymongo.asynchronous.collection import AsyncCollection

        return AsyncCollection

    @staticmethod
    def _cursor_class() -> Any:
        from pymongo.asynchronous.cursor import AsyncCursor

        return AsyncCursor


@pytest.fixture
async def contract_storage(
    contract_uri: str, test_database_name: str, unique_collection_name: str
) -> AsyncIterator[MongoDBStorage]:
    """A store on a real deployment, in this test's own collection.

    One collection per test rather than one per session, so a range scan in one test cannot see
    another's keys. ``list("")`` is unbounded by design (``REQ-E-025``), which makes a shared
    collection the one thing this file's contract tier cannot tolerate.
    """
    config: MongoDBConfig = {"connection_string": contract_uri, "database": test_database_name}
    storage = MongoDBStorage(config, collection=unique_collection_name)
    try:
        yield storage
    finally:
        await storage.close()


@pytest.mark.contract
class TestTheFiveMethodsAgainstARealDeployment:
    """AT-020, 021, 023, 024, 025, 026, 027, 028, 029, 030, 031, 035, 038b, 081 — against a real deployment.

    The unit tier above asserts what filter and what document. This asserts what MongoDB does with
    them, and one claim here cannot be moved: that ``sort("_id", 1)`` over a bounded range returns keys
    in the order Python's ``sorted()`` would. The SDK's snapshot history reads its ordering from that
    (``AT-027``), and a fake asserting it would be asserting its own implementation.
    """

    async def test_write_overwrites_rather_than_appending(self, contract_storage: MongoDBStorage) -> None:
        """AT-020. ``upsert=True`` on a ``replace_one`` is what makes the second call a replacement, and
        a deployment is where a duplicate-key error would show up."""
        await contract_storage.write("a/b", b"first")
        await contract_storage.write("a/b", b"second")

        assert await contract_storage.read("a/b") == b"second"
        assert await contract_storage.list("") == ["a/b"]

    async def test_reading_an_absent_key_returns_none(self, contract_storage: MongoDBStorage) -> None:
        """AT-021 / REQ-E-021. The protocol's own miss signal, and the reason ``read`` cannot use a
        ``KeyError``: the SDK's session manager treats ``None`` as "no session yet"."""
        assert await contract_storage.read("never/written") is None

    async def test_deleting_an_absent_key_is_a_no_op(self, contract_storage: MongoDBStorage) -> None:
        """AT-023 / REQ-E-023. ``delete_one`` reports ``deleted_count`` 0 and no error, so this passes
        only if the count is ignored."""
        await contract_storage.delete("never/written")

        await contract_storage.write("a/b", b"payload")
        await contract_storage.delete("a/b")
        await contract_storage.delete("a/b")

        assert await contract_storage.read("a/b") is None

    @pytest.mark.parametrize(
        "payload",
        [b"", b"\x00", bytes(range(256)), bytes(range(256)) * 16384],
        ids=["empty", "nul", "every-byte", "four-mebibytes"],
    )
    async def test_bytes_round_trip_unchanged(self, contract_storage: MongoDBStorage, payload: bytes) -> None:
        """AT-024 / REQ-E-024. ``b""`` is the case that catches an ``or None`` on the way out, and the
        4 MiB case is the one that catches a payload the driver had to chunk."""
        await contract_storage.write("a/b", payload)

        assert await contract_storage.read("a/b") == payload

    async def test_a_payload_stored_under_another_binary_subtype_reads_back_as_bytes(
        self, contract_storage: MongoDBStorage
    ) -> None:
        """Why ``read`` converts instead of returning the field as stored.

        Written by a mutation survivor. Measured on pymongo 4.17.0: BSON binary **subtype 0** decodes to
        exact ``bytes``, so a collection this package wrote by itself makes the conversion a no-op and
        deleting it changes nothing. Subtype 2 and 0x80 decode to ``Binary``, and ``Binary.__eq__``
        compares the subtype, so ``value == payload`` is ``False`` for the same 7 bytes. A collection
        also written by an older driver or another language's driver is where that lands, and there the
        conversion is the difference between a correct read and a payload that compares unequal to
        itself. ``bytes(...)`` recovers the payload for every subtype.
        """
        raw = contract_storage._collection()
        await raw.replace_one({"_id": "a/b"}, {"_id": "a/b", "data": bson.Binary(b"payload", 2)}, upsert=True)

        found = await contract_storage.read("a/b")

        assert type(found) is bytes
        assert found == b"payload"

    async def test_list_with_an_empty_prefix_returns_every_key(self, contract_storage: MongoDBStorage) -> None:
        """AT-025 / REQ-E-025. The SDK's ``KeywordSearchStrategy`` calls exactly this."""
        for key in ("a/one", "b/two", "c/three"):
            await contract_storage.write(key, b"payload")

        assert await contract_storage.list("") == ["a/one", "b/two", "c/three"]

    async def test_list_returns_full_keys_rather_than_suffixes(self, contract_storage: MongoDBStorage) -> None:
        """AT-026 / REQ-E-026. ``LocalFileStorage`` returns the key the caller could pass back to
        ``read``, and a backend returning ``one`` where the key is ``a/one`` breaks every caller that
        does."""
        await contract_storage.write("a/one", b"payload")
        await contract_storage.write("a/two", b"payload")

        assert await contract_storage.list("a/") == ["a/one", "a/two"]

    async def test_list_order_equals_pythons_sorted(self, contract_storage: MongoDBStorage) -> None:
        """AT-026 and AT-027 together, on a key set chosen to separate byte order from anything else:
        digits before letters, uppercase before lowercase, and a multi-byte code point last."""
        keys = ["a/10", "a/2", "a/B", "a/a", "a/é", "a/Z"]
        for key in keys:
            await contract_storage.write(key, b"payload")

        assert await contract_storage.list("a/") == sorted(keys)

    async def test_a_uuid7_snapshot_set_lists_chronologically(self, contract_storage: MongoDBStorage) -> None:
        """AT-027's half that the SDK actually depends on.

        ``SnapshotSessionManager`` names each snapshot with a UUID7 and reads its history back in the
        order ``list`` returns. UUID7's leading 48 bits are a millisecond timestamp, so the hex form
        sorts chronologically, and MongoDB's ``_id`` sort is what has to preserve that. The ids come
        from the SDK's own generator rather than from a hand-written sequence, so the test cannot pass
        against a format the SDK does not produce.
        """
        from strands._identifier import new_uuid7

        prefix = "session/s1/scopes/agent/a1/snapshots/immutable_history/"
        identifiers = [new_uuid7() for _ in range(25)]
        for identifier in reversed(identifiers):
            await contract_storage.write(f"{prefix}snapshot_{identifier}.json", b"payload")

        found = await contract_storage.list(prefix)

        assert found == [f"{prefix}snapshot_{identifier}.json" for identifier in identifiers]

    async def test_the_snapshots_directory_round_trips_on_keys_the_sdk_builds(
        self, contract_storage: MongoDBStorage, contract_database: Any, unique_collection_name: str
    ) -> None:
        """AT-081 — the other half of the snapshot keys: the mutable ``snapshot_latest.json`` key, and both key
        shapes listed together.

        The test above covers the immutable history's ordering. This one covers the key the manager
        overwrites in place, and it spells no key itself. ``_snapshot_key`` and ``_snapshots_prefix`` are
        module-level in the SDK precisely so a second caller builds the same keys the manager reads, and
        calling them is the difference between testing the SDK's layout and testing a layout this file
        invented. The checklist writes the mutable key as ``snapshot_latest``; the SDK's constant
        is ``snapshot_latest.json``, which is why the suffix comes from the SDK here.

        ``_resolve_storage`` is included for the doubling case. The manager namespaces raw storage under
        ``session`` itself, so a caller who hands it a view this package already scoped would otherwise
        write to ``session/session/...``. Our ``namespace()`` sets the SDK's ``_NAMESPACED`` sentinel, and
        the assertion is that the manager's own resolver honours it by handing the view straight back.

        The final ``list`` covers both shapes at once. Under one agent's snapshots prefix the history
        directory sorts ahead of the mutable key on the first byte (``i`` before ``s``), so one call has
        to return the chronological history first and the latest pointer last.
        """
        from strands._identifier import new_uuid7
        from strands.session.snapshot_session_manager import _resolve_storage, _snapshot_key, _snapshots_prefix

        scoped = contract_storage.namespace("session")
        assert _resolve_storage(scoped) is scoped

        latest = _snapshot_key("s1", "a1", snapshot_id=None)
        await scoped.write(latest, b'{"messages": []}')
        assert await scoped.read(latest) == b'{"messages": []}'

        # The mutable key is overwritten on every save, so a backend that appended or refused the second
        # write would leave the manager reading a stale session.
        await scoped.write(latest, b'{"messages": [{"role": "user"}]}')
        assert await scoped.read(latest) == b'{"messages": [{"role": "user"}]}'

        # Written through the SDK's namespace, so the stored ``_id`` is the on-disk key the manager's own
        # layout comment documents: ``session/<session_id>/scopes/agent/<agent_id>/snapshots/...``.
        stored = await contract_database[unique_collection_name].find_one({"_id": f"session/{latest}"})
        assert stored is not None

        identifiers = [new_uuid7() for _ in range(3)]
        for identifier in reversed(identifiers):
            await scoped.write(_snapshot_key("s1", "a1", snapshot_id=identifier), b"checkpoint")

        found = await scoped.list(_snapshots_prefix("s1", "a1"))

        assert found == [_snapshot_key("s1", "a1", snapshot_id=i) for i in identifiers] + [latest]

    async def test_list_on_a_non_matching_prefix_returns_an_empty_list(self, contract_storage: MongoDBStorage) -> None:
        """AT-028 / REQ-E-028. ``[]`` rather than an error, and the neighbouring key is there to prove
        the range bound excludes it rather than the collection being empty."""
        await contract_storage.write("sessions-archive/abc", b"payload")

        assert await contract_storage.list("sessions/") == []

    async def test_an_oversized_payload_raises_before_the_driver_sees_it(
        self, contract_storage: MongoDBStorage
    ) -> None:
        """AT-029 / REQ-E-029. Our message rather than ``DocumentTooLarge``, and nothing written."""
        payload = b"z" * (storage_module.MAX_DOCUMENT_BYTES + 1)

        with pytest.raises(StorageError) as caught:
            await contract_storage.write("a/b", payload)

        assert "a/b" in str(caught.value)
        assert await contract_storage.list("") == []

    async def test_the_ceiling_constant_equals_what_the_deployment_advertises(self, contract_database: Any) -> None:
        """AT-038b / REQ-E-038. The constant is written out in our own module rather than imported from
        ``pymongo.common``, which is not public API, so this is what keeps the two from drifting."""
        hello = await contract_database.command("hello")

        assert storage_module.MAX_DOCUMENT_BYTES == hello["maxBsonObjectSize"]

    async def test_a_namespaced_store_prefixes_writes_and_strips_them_from_list(
        self, contract_storage: MongoDBStorage, contract_database: Any, unique_collection_name: str
    ) -> None:
        """AT-030 / REQ-E-030. Both halves, and the raw ``_id`` is read directly because a store that
        neither prefixed nor stripped would pass a test that only used its own ``list``."""
        scoped = contract_storage.namespace("tenant-a")

        await scoped.write("sessions/abc", b"payload")

        stored = await contract_database[unique_collection_name].find_one({})
        assert stored["_id"] == "tenant-a/sessions/abc"
        assert await scoped.list("") == ["sessions/abc"]
        assert await scoped.read("sessions/abc") == b"payload"

    async def test_a_namespaced_store_cannot_see_a_sibling(self, contract_storage: MongoDBStorage) -> None:
        """The isolation the prefix is for. ``tenant-a`` and ``tenant-ab`` differ by one character and
        share a prefix, so a range whose upper bound is wrong leaks one into the other."""
        await contract_storage.namespace("tenant-a").write("k", b"a")
        await contract_storage.namespace("tenant-ab").write("k", b"ab")

        assert await contract_storage.namespace("tenant-a").list("") == ["k"]
        assert await contract_storage.namespace("tenant-a").read("k") == b"a"

    async def test_delete_prefix_removes_a_range_and_returns_the_count(self, contract_storage: MongoDBStorage) -> None:
        """AT-031 / REQ-E-031. The neighbour outside the range is what makes the count meaningful."""
        for key in ("a/one", "a/two", "a/three", "b/four"):
            await contract_storage.write(key, b"payload")

        assert await contract_storage.delete_prefix("a/") == 3
        assert await contract_storage.list("") == ["b/four"]

    async def test_delete_prefix_on_a_non_matching_prefix_returns_zero(self, contract_storage: MongoDBStorage) -> None:
        await contract_storage.write("b/four", b"payload")

        assert await contract_storage.delete_prefix("a/") == 0

    async def test_a_ttl_store_writes_the_expiry_field(
        self, contract_storage: MongoDBStorage, contract_database: Any, unique_collection_name: str
    ) -> None:
        """AT-035 / REQ-E-035. Read through the driver rather than through ``read``, which returns only
        the payload, and the field is what the TTL index acts on."""
        config: MongoDBConfig = {
            "connection_string": contract_storage._config["connection_string"],
            "database": contract_storage._database_name,
        }
        expiring = MongoDBStorage(config, collection=unique_collection_name, ttl_seconds=3600)

        await expiring.write("with-ttl", b"payload")
        await contract_storage.write("without-ttl", b"payload")

        collection = contract_database[unique_collection_name]
        assert isinstance((await collection.find_one({"_id": "with-ttl"}))["expires_at"], datetime.datetime)
        assert "expires_at" not in await collection.find_one({"_id": "without-ttl"})

    async def test_the_fake_and_the_driver_agree_on_the_cursor_and_the_count(
        self, contract_database: Any, unique_collection_name: str
    ) -> None:
        """MP-001's behavioural half. The unit tier compares signatures; this runs the same three calls
        against a deployment, in the order ``list`` makes them."""
        collection = contract_database[unique_collection_name]
        await collection.replace_one({"_id": "a/one"}, {"_id": "a/one", "data": b"x"}, upsert=True)

        cursor = collection.find({"_id": {"$gte": "a/", "$lt": "a0"}}, {"_id": 1})
        found = [document["_id"] async for document in cursor.sort("_id", 1)]

        assert found == ["a/one"]
        assert (await collection.delete_many({"_id": {"$gte": "a/", "$lt": "a0"}})).deleted_count == 1


@pytest.mark.contract
class TestSearchAgainstARealDeployment:
    """AT-032b, AT-032d, AT-032e, AT-033 — the half no fake can carry.

    Everything that makes the aggregation worth writing lives on the server: whether the tokenizer
    agrees with the SDK's, whether the scan is covered, and what order the ``$sort`` actually returns.
    The unit tier asserts the stages this package emits and deliberately asserts nothing about scores.
    """

    KEYS = (
        "notes/dark-mode-toggle.md",
        "notes/toggle-dark.md",
        "notes/café-münchen.md",
        "notes/日本語.md",
        "sessions/s1/state.json",
    )

    @staticmethod
    async def _seed(storage: MongoDBStorage) -> None:
        """Payloads large enough that a document fetch would be visible in ``docsExamined``."""
        for key in TestSearchAgainstARealDeployment.KEYS:
            await storage.write(key, b"x" * 400)

    async def test_results_are_keys_and_scores_with_no_payload(self, contract_storage: MongoDBStorage) -> None:
        """``AT-032b`` (``REQ-E-032b``)."""
        await self._seed(contract_storage)

        found = await contract_storage.search("dark mode")

        assert [result.key for result in found] == ["notes/dark-mode-toggle.md", "notes/toggle-dark.md"]
        assert all(type(result.score) is float for result in found)
        assert all(result.data is None for result in found)

    @pytest.mark.parametrize(
        "query",
        ["dark mode toggle", "münchen café", "日本語", "state json", "toggle", "nothing at all here"],
    )
    async def test_the_server_scores_exactly_as_the_sdk_would(
        self, contract_storage: MongoDBStorage, query: str
    ) -> None:
        """``AT-032d``, and the reason ``[\\p{L}\\p{N}_]+`` is not ``\\w+``.

        MongoDB's ``\\w`` is ASCII-only by default while Python's is every character ``str.isalnum()``
        accepts, so ``\\w+`` tokenizes ``café`` to ``caf``, ``m``, ``nchen`` and drops ``日本語``
        altogether. Both mangled forms still return *results*, which is what makes this invisible without
        a comparison against the SDK's own scorer. The comparison is against ``token_overlap_score`` over
        the key text alone, because scoring the key space rather than the payload is the decision here.
        """
        from strands.storage.search.keyword import token_overlap_score, tokenize

        await self._seed(contract_storage)
        tokens = tokenize(query)
        expected = sorted(
            ((key, token_overlap_score(tokens, key)) for key in self.KEYS if token_overlap_score(tokens, key) > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )

        found = await contract_storage.search(query)

        assert [(result.key, int(result.score)) for result in found] == expected

    async def test_a_mixed_case_key_matches_a_lowercase_query(self, contract_storage: MongoDBStorage) -> None:
        """Written by a mutation survivor: removing ``$toLower`` from the key side changed nothing.

        Every other key in this file is already lowercase, so the stage was doing no work anywhere the
        suite could see. The SDK's ``tokenize`` lowercases whatever it is given, meaning the query side is
        always folded and the key side has to be folded to meet it. ``_normalize_key`` collapses slashes
        and rejects ``..`` and deliberately does **not** change case, so a caller writing
        ``Sessions/Agent-1/State.json`` keeps that ``_id`` and, without the stage, would never match
        ``sessions agent``.
        """
        await contract_storage.write("Sessions/Agent-1/State.json", b"x")

        found = await contract_storage.search("sessions agent")

        assert [result.key for result in found] == ["Sessions/Agent-1/State.json"]
        assert found[0].score == 2.0

    async def test_a_token_only_in_a_payload_does_not_match(self, contract_storage: MongoDBStorage) -> None:
        """The key-space trade, asserted rather than left as prose.

        The SDK's default scores ``f"{key} {content}"`` and would return this key; we score the key space
        alone, so a word that appears only inside a payload finds nothing. This is the one intended
        behavioural divergence in the method, and it is what buys the covered scan in ``AT-032e``: reading
        payload text server-side means fetching every document. The unit tier pins the mechanism (no
        ``$data`` in the projection); this pins the consequence. Replaces an earlier test that asserted
        the delegating behaviour, whose docstring already named this as the assertion that would change.
        """
        await contract_storage.write("notes/one.md", b"needle")

        assert await contract_storage.search("needle") == []
        assert [result.key for result in await contract_storage.search("notes one")] == ["notes/one.md"]

    async def test_scores_never_increase_and_ties_break_by_ascending_key(
        self, contract_storage: MongoDBStorage
    ) -> None:
        """``AT-033``. A token-overlap count is already similarity-oriented, so nothing is inverted here;
        the inversion clause in the requirement is for ``$vectorSearch``, where a ``euclidean`` metric
        does need it. What this asserts is the ordering promise and the absence of a negative score.
        """
        await self._seed(contract_storage)

        found = await contract_storage.search("notes toggle")
        pairs = [(result.score, result.key) for result in found]

        assert all(result.score > 0 for result in found)
        assert pairs == sorted(pairs, key=lambda pair: (-pair[0], pair[1]))

    async def test_the_scan_is_covered_by_the_id_index_and_fetches_no_documents(
        self, contract_storage: MongoDBStorage, contract_database: Any, unique_collection_name: str
    ) -> None:
        """``AT-032e``. The measurement the whole design rests on, asserted rather than trusted.

        A namespaced store's range gives the planner an ``IXSCAN`` on ``_id_``, and because ``_id`` is the
        only field the pipeline reads, the server never fetches a document: ``docsExamined`` is 0 even
        though every key here carries a 400-byte payload. Naming one more field in the ``$project`` turns
        that into one fetch per matching key and no other assertion in this file would change.
        """
        from strands.storage.search.keyword import tokenize

        scoped = contract_storage.namespace("tenant-a")
        await self._seed(scoped)

        plan = await contract_database.command(
            "explain",
            {
                "aggregate": unique_collection_name,
                "pipeline": scoped._search_pipeline(tokenize("dark mode")),
                "cursor": {},
            },
            verbosity="executionStats",
        )
        stats = plan["stages"][0]["$cursor"]["executionStats"]

        assert stats["totalDocsExamined"] == 0
        assert "IXSCAN" in str(plan["stages"][0]["$cursor"]["queryPlanner"]["winningPlan"])
        assert "COLLSCAN" not in str(plan["stages"][0]["$cursor"]["queryPlanner"]["winningPlan"])

    async def test_aggregate_is_awaited_where_find_is_not(
        self, contract_database: Any, unique_collection_name: str
    ) -> None:
        """MP-001's aggregation half, against the driver rather than against ``inspect``.

        ``AsyncCollection.find`` returns a cursor directly and ``aggregate`` returns a coroutine, so the
        missing ``await`` here produces ``AttributeError: 'coroutine' object has no attribute ...`` rather
        than anything that names the real problem. This is the call shape ``search`` depends on.
        """
        collection = contract_database[unique_collection_name]
        await collection.replace_one({"_id": "a/one"}, {"_id": "a/one", "data": b"x"}, upsert=True)

        cursor = await collection.aggregate([{"$match": {"_id": "a/one"}}, {"$project": {"_id": 1}}])

        assert [document async for document in cursor] == [{"_id": "a/one"}]


@pytest.mark.contract
class TestOneStorageServesTwoSuccessiveEventLoops:
    """AT-091b / REQ-E-017a — against a deployment, where the wrong answer raises.

    ``AT-091`` in ``test_client.py`` asserts which client the registry hands back, which is the mechanism.
    This asserts the outcome the requirement states: a store built once and written through twice, on two
    loops, has to store both payloads. Against a real deployment the broken version does not return a
    surprising value, it raises ``RuntimeError: Cannot use AsyncMongoClient in different event loop``.

    Every test in this file is a single ``asyncio.run`` under pytest-asyncio, which is why none of them
    caught this. A synchronous ``Agent`` is not: ``strands/_async.py`` gives each entry point a fresh loop
    in a worker thread, so a store held on the agent crosses loops on the second call. The shape below is
    the smallest thing that has that property, and it is what ``examples/unified_storage.py`` hit first.
    """

    @staticmethod
    def _write_and_read(storage: MongoDBStorage, key: str, payload: bytes) -> bytes | None:
        """Write one key and read it back, on a loop created and closed for this call alone.

        Args:
            storage: The store to use. Built before the first loop and reused after it closed.
            key: Key to write.
            payload: Bytes to write.

        Returns:
            Whatever ``read`` gives back.
        """

        async def body() -> bytes | None:
            await storage.write(key, payload)
            return await storage.read(key)

        return asyncio.run(body())

    def test_a_store_built_once_survives_the_loop_it_was_first_used_on(
        self, contract_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """The requirement's outcome, on two real loops and one real deployment.

        Not a registry assertion. The two reads are the claim: both payloads reached MongoDB and came
        back, which is the only thing a caller can observe and the thing the broken version cannot do.
        """
        storage = MongoDBStorage(
            {"database": test_database_name, "connection_string": contract_uri},
            collection=unique_collection_name,
        )

        try:
            assert self._write_and_read(storage, "loop/one", b"first") == b"first"
            assert self._write_and_read(storage, "loop/two", b"second") == b"second"

            # A third loop, to read both keys through the same store: this is where a fix that merely
            # replaced the entry each time would leave the first write unreachable.
            assert asyncio.run(storage.list("loop/")) == ["loop/one", "loop/two"]
        finally:
            asyncio.run(storage.close())


@pytest.mark.contract
class TestEnsureIndexesAgainstARealDeployment:
    """AT-121b, AT-122 / ``REQ-E-119``, ``REQ-E-120`` — the server boundary no fake can stand at.

    The unit tier above asserts that ``ensure_indexes()`` asks for the right index. That is the
    mechanism. The outcome that matters is that a document written with a TTL stops
    existing, and the thing that removes it is the server's TTL monitor: a background task in ``mongod``
    that sweeps once a minute and is reachable from no assertion this package can make about its own
    code. ``AT-122`` is therefore the only test in the suite that waits on wall-clock time, and it waits
    because the alternative is asserting the index document and calling that expiry.
    """

    # The monitor's period is 60 seconds, so a document with a one-second lifetime is removed on the
    # first sweep after it expires: under 60 seconds typically, up to 120 in the worst alignment. 180
    # is that worst case plus a sweep, which is the smallest budget that does not turn a slow but
    # correct deployment into a red build. The poll is coarse because the answer changes once.
    EXPIRY_BUDGET_SECONDS: Final[float] = 180.0
    EXPIRY_POLL_SECONDS: Final[float] = 5.0

    async def test_it_creates_the_ttl_index_and_a_second_call_leaves_it_alone(
        self, contract_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-121b. The same two claims as the unit tier, against a server that has its own opinions.

        Worth running twice here rather than only against the fake because the second call is where a
        real ``createIndexes`` would fail: the server rejects a second index over the same keys under a
        different name, so a spec whose options did not round-trip would raise on the second pass
        instead of quietly re-creating.
        """
        config: MongoDBConfig = {"connection_string": contract_uri, "database": test_database_name}
        storage = MongoDBStorage(config, collection=unique_collection_name, ttl_seconds=60)
        try:
            await storage.ensure_indexes()
            await storage.ensure_indexes()

            collection = storage._collection()
            indexes = await collection.index_information()
        finally:
            await storage.close()

        assert "strands_expires_at_ttl" in indexes, sorted(indexes)
        assert indexes["strands_expires_at_ttl"]["expireAfterSeconds"] == 0
        assert [tuple(pair) for pair in indexes["strands_expires_at_ttl"]["key"]] == [(_indexes.EXPIRES_FIELD, 1)]

    async def test_a_document_written_with_a_ttl_is_gone(
        self, contract_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-122 / ``REQ-E-120`` — the outcome, not the index.

        ``read`` returning ``None`` is the assertion, and it is deliberately the same signal a caller
        gets for a key that was never written: that is what expiry looks like from outside. The write
        is read back first so a failure at the end cannot be a write that never landed, which is the
        one other way this could go ``None``.
        """
        config: MongoDBConfig = {"connection_string": contract_uri, "database": test_database_name}
        storage = MongoDBStorage(config, collection=unique_collection_name, ttl_seconds=1)
        try:
            await storage.ensure_indexes()
            await storage.write("expiring/key", b"payload")
            assert await storage.read("expiring/key") == b"payload"

            deadline = time.monotonic() + self.EXPIRY_BUDGET_SECONDS
            while await storage.read("expiring/key") is not None:
                assert time.monotonic() < deadline, (
                    f"the document was still readable {self.EXPIRY_BUDGET_SECONDS:.0f}s after it "
                    f"expired. The TTL monitor sweeps once a minute, so this is not a slow sweep: "
                    f"either the index is absent (ensure_indexes() did not run or did nothing) or "
                    f"the deployment has ttlMonitorEnabled set to false."
                )
                await asyncio.sleep(self.EXPIRY_POLL_SECONDS)
        finally:
            await storage.close()
