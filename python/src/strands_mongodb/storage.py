"""``MongoDBStorage`` -- MongoDB behind the Strands unified ``Storage`` protocol.

Requires the ``storage`` extra::

    pip install 'strands-mongodb[storage]'

One ``Storage`` instance is consumed by five SDK subsystems -- sessions, context offloading,
memory persistence, context history and transcripts -- via the top-level ``storage`` parameter
on ``Agent``, so a single implementation covers all of them.
"""

from __future__ import annotations

import builtins
import datetime
import functools
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

from ._extras import missing_extra

# Imported for their side effect of proving the `storage` extra is installed: without it the
# failure would otherwise surface deep inside a method call as a bare ModuleNotFoundError.
try:
    import pymongo  # noqa: F401
    from strands.storage import Storage  # noqa: F401
except ImportError as error:
    missing_extra("storage", error, module=__name__)

from pymongo.errors import PyMongoError
from strands.storage.search.keyword import tokenize

# Private in the SDK, and used here for the same reason all three shipped backends use them: they are
# what makes `a//b` and `a/b` one key on every backend. Reimplementing the two would be reimplementing
# a compatibility promise, and a copy that drifted would show up as a key set that collides on
# `LocalFileStorage` and stays distinct here.
# ``StorageSearchResult`` is constructed rather than only annotated, so it moves out of the
# ``TYPE_CHECKING`` block. It is a dataclass in the same module as the protocol and pulls in nothing the
# storage extra does not already require.
from strands.storage.storage import _NAMESPACED, StorageSearchResult, _normalize_key, _normalize_prefix
from strands.types.exceptions import StorageError

from . import _client
from ._indexes import EXPIRES_FIELD

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from pymongo.asynchronous.collection import AsyncCollection

    from .types import MongoDBConfig

__all__ = ["MAX_DOCUMENT_BYTES", "MAX_KEY_BYTES", "MongoDBStorage"]

MAX_KEY_BYTES: Final[int] = 1024
"""Longest key this package writes, in UTF-8 bytes.

Our policy, not the server's. MongoDB 8.3.8 accepts a 1,000,000-byte ``_id`` and round-trips it,
because the pre-4.2 index key limit no longer applies at the default feature compatibility version.
A key that long still costs what an index entry that long costs on every ``list`` range scan, and it
fails outright on a deployment pinned below FCV 4.2. 1024 is the old limit, which makes it the number
a MongoDB reader already recognises, and the headroom is wide: the longest key shape the SDK builds is
131 bytes for a realistic session and agent id pair.
"""

MAX_DOCUMENT_BYTES: Final[int] = 16 * 1024 * 1024
"""The BSON document ceiling, ours rather than the driver's.

``pymongo.common.MAX_BSON_SIZE`` holds the same number and is not public API. ``AT-038b`` pins this
constant to the ``maxBsonObjectSize`` a real deployment advertises, so the two cannot drift.
"""

# Bytes a stored document costs before its key and payload: the 4-byte length, the `_id`
# and `data` element headers, the whole `updated_at` element, and the terminator. Shape-specific, so
# `AT-038a` pins both constants to `bson.encode` rather than to this comment.
_DOCUMENT_OVERHEAD_BYTES: Final[int] = 46
_EXPIRES_OVERHEAD_BYTES: Final[int] = 20

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _encoded_size(key: str, data: bytes, *, expires: bool) -> int:
    """Size of the document ``write`` would build, without building it.

    Arithmetic rather than ``len(bson.encode(document))`` because the call this protects is the one
    most likely to be near a memory limit already: encoding a 16 MiB payload only to measure it doubles
    its peak footprint. ``AT-038a`` asserts this agrees with the encoder exactly, over an empty
    payload, multi-byte and 4-byte code points in the key, and a 1024-byte key with a 4 MiB payload.

    Args:
        key: The ``_id`` the document will carry, prefix included.
        data: The payload.
        expires: Whether an ``expires_at`` element will be present.

    Returns:
        The number of bytes the encoded document will occupy.
    """
    return _DOCUMENT_OVERHEAD_BYTES + len(key.encode("utf-8")) + len(data) + (_EXPIRES_OVERHEAD_BYTES if expires else 0)


def _successor(text: str) -> str | None:
    """Smallest string that sorts above every key starting with ``text``.

    The upper bound of the ``_id`` range scan, and the reason it is a function rather than
    ``text[:-1] + chr(ord(text[-1]) + 1)``. That one-liner has two inputs it cannot express, both
    reachable through a protocol that permits any string as a key:

    * A prefix ending at ``U+D7FF`` increments to ``U+D800``, a lone surrogate, which no BSON encoder
      accepts. The next scalar value is ``U+E000`` and nothing sorts between them, so skipping the
      block keeps the bound exact as well as encodable.
    * A prefix ending at ``U+10FFFF`` has no successor in that position at all, and ``chr`` raises
      ``ValueError``. Incrementing the character to its left is still an exact bound, and a prefix made
      only of ``U+10FFFF`` has no upper bound to give: everything sorting above it starts with it.

    Args:
        text: The prefix the range starts at. Never empty; an empty prefix is an unfiltered query.

    Returns:
        The exclusive upper bound, or ``None`` when the range has no upper bound.
    """
    for index in range(len(text) - 1, -1, -1):
        point = ord(text[index]) + 1
        if point == 0xD800:
            point = 0xE000
        if point > 0x10FFFF:
            # Carry left. Every string starting with `text` also starts with this shorter prefix.
            continue
        return text[:index] + chr(point)
    return None


def _abbreviate(key: str) -> str:
    """Shorten a key for an error message, because a rejected key can be 1025 characters long."""
    return key if len(key) <= 60 else f"{key[:60]}[+{len(key) - 60} more]"


def _reporting(
    verb: str,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """Turn any driver error out of the wrapped method into ``StorageError``.

    A decorator over the whole method rather than a helper wrapping one awaitable, which is the shape
    ``_indexes._guard`` uses. ``list`` is why: ``find`` contacts nothing, so a connection that dies
    while the cursor drains raises from the ``async for``, and a helper that wraps the call alone never
    sees it. ``REQ-E-036`` says no method leaks a bare ``PyMongoError``, and that is the method it would
    leak from.

    The message is built here and in ``_indexes`` from the same sentence, so a reader of the log has no
    reason to know which module produced which line. There is no shared ``_errors`` module for it: two
    format strings and a test pinning them together is less machinery than a third module.

    The wrapped type is ``Coroutine`` rather than ``Awaitable``, and that is not cosmetic. ``Storage``
    declares its five methods with ``async def``, so a decorator that widened them to ``Awaitable``
    would leave ``MongoDBStorage`` satisfying the protocol at runtime and failing it under mypy, which
    is exactly the position a user typing ``Agent(storage=MongoDBStorage(config))`` would hit.

    Args:
        verb: What the method was doing, as a present participle, for the message.

    Returns:
        A decorator for an async method whose first two arguments are ``self`` and a key or prefix.
    """

    def decorate(method: Callable[_P, Coroutine[Any, Any, _R]]) -> Callable[_P, Coroutine[Any, Any, _R]]:
        @functools.wraps(method)
        async def reporting(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return await method(*args, **kwargs)
            except PyMongoError as error:
                # Bound through a local rather than indexed off `args` directly, which mypy rejects
                # for a `ParamSpec`.
                positional: tuple[Any, ...] = args
                storage: MongoDBStorage = positional[0]
                subject = positional[1] if len(positional) > 1 else ""
                raise StorageError(
                    f"strands_mongodb: {verb} {_abbreviate(str(subject))!r} in {storage._target} failed: {error}"
                ) from error

        return reporting

    return decorate


# The `Storage` protocol is satisfied structurally and this class does not inherit it, matching all
# three backends the SDK ships (`InMemoryStorage`, `LocalFileStorage`, `S3Storage`). The memory
# surface goes the other way, because both vended memory stores do.


class MongoDBStorage:
    """MongoDB-backed implementation of ``strands.storage.Storage``.

    The connection half arrives as a whole ``MongoDBConfig``, built once per process and reused; the
    per-instance half arrives as keyword arguments, because two storages in one database differ by
    collection and ``namespace()`` returns a second instance differing only by prefix. Passing the
    same config object to every instance is what makes them share one connection pool
    (``REQ-E-012``).

    Example:
        ```python
        config: MongoDBConfig = {"database": "agents", "connection_string": "mongodb+srv://..."}
        sessions = MongoDBStorage(config, collection="sessions")
        transcripts = MongoDBStorage(config, collection="transcripts")  # same pool
        ```

    One document per key, ``_id`` holding the key verbatim, so every operation is a primary-key hit or
    a range scan over the index MongoDB creates whether or not it is asked to. ``list`` is that
    range scan rather than a filtered collection scan, which is what makes it usable on a collection of
    4 MiB payloads.

    ``for_sandbox`` is absent permanently: the SDK probes it with ``hasattr`` and falls back when it is
    missing, so a stub that raised would break context offloading for every MongoDB user rather than
    degrade it.
    """

    # Declared without a value on purpose. `getattr(storage, "_namespaced", None)` is what the SDK's
    # three call sites read, so an unscoped instance has to leave it absent while `namespace()` sets it
    # (`REQ-E-039`). An assignment here would mark every instance as already scoped.
    _namespaced: object

    def __init__(
        self,
        config: MongoDBConfig,
        /,
        *,
        collection: str = "storage",
        prefix: str = "",
        ttl_seconds: int | None = None,
    ) -> None:
        """Resolve the shared connection and record this instance's placement in it.

        Args:
            config: Connection and deployment settings. Positional-only, following ``S3Storage``'s
                resource-first shape, and passed whole so it can be shared with a
                ``MongoDBMemoryStore`` through ``MongoDBMemoryStoreConfig.config``.
            collection: Collection holding this instance's documents.
            prefix: Key prefix prepended to every key, scoping this instance within the collection.
                Normalized to end in exactly one ``/``, or to be empty, so that four spellings of one
                prefix do not become four key spaces.
            ttl_seconds: Lifetime for documents this instance writes. ``None`` means they never
                expire, which is not the same as ``0``: a MongoDB TTL index with
                ``expireAfterSeconds: 0`` deletes documents as soon as their timestamp passes
                (``REQ-E-035``). Setting this stamps ``expires_at`` and nothing more. **Expiry needs
                the TTL index, and no code path in this package creates it on a storage collection.**
                ``_indexes.storage_specs()`` holds the spec an operator or a
                later release provisions it from.

        Raises:
            KeyError: If ``config`` has no ``database``. It is ``Required`` in the ``TypedDict``, so
                mypy rejects a literal that omits it; this catches a config assembled at runtime.
            ValueError: If ``config`` supplies neither ``client`` nor ``connection_string``
                (``REQ-E-015``).
            StorageError: If ``prefix`` contains a ``..`` segment.
        """
        # `database` is read before `resolve` deliberately. Resolving first would register a
        # connection pool in the process-global registry and only then raise, leaving a pool that no
        # instance owns and no `close()` can reach.
        self._database_name = config["database"]
        self._collection_name = collection
        # `_normalize_prefix` keeps a trailing slash, which is significant for matching but not for a
        # prefix that is always followed by a key, so it is stripped and re-added. `"tenant-a"`,
        # `"tenant-a/"`, `"/tenant-a"` and `"tenant-a//"` all arrive here as `"tenant-a/"`.
        normalized = _normalize_prefix(prefix).rstrip("/")
        self._prefix = f"{normalized}/" if normalized else ""
        self._ttl_seconds = ttl_seconds
        self._config = config
        # Resolved here and the result dropped on the floor, which is deliberate. The connection an
        # operation uses comes from `_connection` below, freshly, because the registry is keyed by
        # running event loop (`REQ-E-017a`) and a reference captured here would pin this instance to
        # whichever loop happened to construct it (`REQ-E-017b`). What this call buys is the failure
        # arriving at construction: a config with neither `client` nor `connection_string` raises
        # `ValueError` here (`REQ-E-015`) rather than at the first `write`.
        _client.resolve(config)

    @_reporting("writing")
    async def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, replacing whatever was there.

        One ``replace_one`` with ``upsert=True`` rather than a read followed by an insert or an update:
        the create and the overwrite are the same call, so two writers racing on one key cannot produce
        a duplicate-key error between them (``REQ-E-020``).

        Args:
            key: The key to write under. Normalized, so ``a//b`` and ``a/b`` are one key.
            data: The bytes to store, round-tripped unchanged including ``b""``.

        Raises:
            StorageError: If the key is empty, contains a ``..`` segment, exceeds
                :data:`MAX_KEY_BYTES`, or would build a document over :data:`MAX_DOCUMENT_BYTES`. Also
                if the write fails, with the driver's error chained.
        """
        document_id = self._id_for(key)
        expires = self._ttl_seconds is not None
        size = _encoded_size(document_id, data, expires=expires)
        if size > MAX_DOCUMENT_BYTES:
            # Before the call, so the driver's `DocumentTooLarge` is never the user's first signal
            # (`REQ-E-029`). The numbers are in the message because the useful next question is how
            # much to shed, and `DocumentTooLarge` does not answer it.
            raise StorageError(
                f"strands_mongodb: the document for the key {_abbreviate(document_id)!r} would encode "
                f"to {size} bytes, over the {MAX_DOCUMENT_BYTES}-byte BSON limit in {self._target}. "
                f"Store the payload elsewhere and keep a reference here, or split it across keys."
            )

        now = datetime.datetime.now(datetime.timezone.utc)
        document: dict[str, Any] = {"_id": document_id, "data": data, "updated_at": now}
        if self._ttl_seconds is not None:
            # Only when a TTL is configured. An `expires_at` of `None` would be indexed by the TTL
            # index and, as measured, a document whose field is not a date is simply never
            # expired, so the field would cost bytes and mean nothing (`REQ-E-035`).
            document[EXPIRES_FIELD] = now + datetime.timedelta(seconds=self._ttl_seconds)

        await self._collection().replace_one({"_id": document_id}, document, upsert=True)

    @_reporting("reading")
    async def read(self, key: str) -> bytes | None:
        """Return the bytes stored under ``key``, or ``None`` if there are none.

        ``None`` rather than an exception, because the SDK's session manager reads a missing key as "no
        session yet" on every cold start (``REQ-E-021``).

        Args:
            key: The key to read.

        Returns:
            The stored bytes, or ``None`` when the key is absent.

        Raises:
            StorageError: If the key is invalid or too long, or if the read fails for any reason other
                than the key being absent, with the driver's error chained (``REQ-E-022``).
        """
        document = await self._collection().find_one({"_id": self._id_for(key)})
        if document is None:
            return None
        # `bytes(...)` rather than the value as stored, and not for the type annotation's sake. Measured
        # on pymongo 4.17.0: binary subtype 0, which is what this package writes, decodes to exact
        # `bytes` and makes this a no-op. Subtypes 2 and 0x80 decode to `Binary`, and `Binary.__eq__`
        # compares the subtype, so the value a caller gets back would not equal the payload that was
        # stored. A collection an older driver also wrote is where that arrives.
        return bytes(document["data"])

    @_reporting("deleting")
    async def delete(self, key: str) -> None:
        """Remove ``key``, doing nothing if it is not there.

        ``deleted_count`` is deliberately not read. Absent and deleted are the same outcome to a caller
        cleaning up, and treating a count of 0 as a failure would turn every idempotent teardown into an
        error (``REQ-E-023``).

        Args:
            key: The key to remove.

        Raises:
            StorageError: If the key is invalid or too long, or if the delete fails, with the driver's
                error chained.
        """
        await self._collection().delete_one({"_id": self._id_for(key)})

    @_reporting("listing")
    async def list(self, query: str) -> builtins.list[str]:
        """Return every key under ``query``, in ascending order.

        A bounded range over the ``_id`` index, projected to ``_id`` alone, which is what keeps a
        listing from pulling every payload across the wire to return their names. The sort is the
        protocol's promise and MongoDB's byte order over UTF-8 matches Python's ``sorted()`` for every
        key it accepts, which ``AT-027`` checks against a deployment rather than against a fake.

        Args:
            query: A key prefix. ``""`` matches everything, which is how the SDK's keyword search
                strategy reads the whole store.

        Returns:
            Full keys rather than the part after the prefix, with this instance's namespace prefix
            removed (``REQ-E-026``, ``REQ-E-030``).

        Raises:
            StorageError: If the prefix contains a ``..`` segment, or if the scan fails at any point,
                including part way through draining the cursor.
        """
        cursor = self._collection().find(self._range_for(query), {"_id": 1})
        # Only this instance's prefix comes off, never the queried one: a caller passing `sessions/` has
        # to get keys it can hand back to `read`.
        offset = len(self._prefix)
        return [str(document["_id"])[offset:] async for document in cursor.sort("_id", 1)]

    @_reporting("searching")
    async def search(self, query: str) -> builtins.list[StorageSearchResult]:
        """Search stored content, ranked best-first, in one server-side aggregation.

        Token-overlap scoring identical to the SDK's ``KeywordSearchStrategy``, computed on the server
        over the key space instead of in Python over every payload (``REQ-E-032``). The SDK's
        default costs ``list("")`` plus one ``read`` per key; this costs one ``aggregate`` and, measured
        over 1,000 keys carrying 400-byte payloads, zero document fetches, because ``_id`` is the only
        field the pipeline reads and the scan is covered by ``_id_`` (``AT-032e``).

        The trade is deliberate: scores come from the key alone, so a query token that appears only
        inside a payload matches here where the SDK's default would match it.

        Args:
            query: A natural-language string. One that tokenizes to nothing returns without a round trip.

        Returns:
            Matched keys with relevance scores, highest first, ties in ascending key order. ``data`` is
            left ``None``, which ``StorageSearchResult`` permits and the shipped consumer handles by
            reading back at most its own ``max_results`` keys (``REQ-E-032b``).

        Raises:
            StorageError: If the aggregation fails, or fails part way through draining the cursor.
        """
        tokens = tokenize(query)
        if not tokens:
            # `KeywordSearchStrategy` short circuits here too. Without it the server would scan the whole
            # namespace to score every key zero and hand back the same empty list (`REQ-E-032a`).
            return []
        cursor = await self._collection().aggregate(self._search_pipeline(tokens))
        offset = len(self._prefix)
        return [
            StorageSearchResult(key=str(document["_id"])[offset:], score=float(document["score"]))
            async for document in cursor
        ]

    def _search_pipeline(self, tokens: set[str]) -> builtins.list[dict[str, Any]]:
        """Build the four stages ``search`` sends, exposed so ``AT-032e`` can ``explain`` the real thing.

        Args:
            tokens: The already-tokenized query. Sorted on the way in so that two callers asking the same
                question produce one plan-cache shape rather than one per token order.

        Returns:
            The pipeline, with the range ``$match`` omitted entirely when this instance has no prefix:
            ``_range_for("")`` is ``{}`` there, and an empty ``$match`` is still a stage to run.
        """
        stages: builtins.list[dict[str, Any]] = []
        # The instance prefix, not the query. `search` takes prose, and this is the bound that turns a
        # `COLLSCAN` into a covered `IXSCAN` on `_id_`, measured both ways.
        instance_range = self._range_for("")
        if instance_range:
            stages.append({"$match": instance_range})
        stages.append(
            {
                "$project": {
                    "score": {
                        "$size": {
                            "$setIntersection": [
                                {
                                    "$map": {
                                        # `[\p{L}\p{N}_]+` rather than `\w+`, which is the one substitution
                                        # that reads as equivalent and is not. MongoDB's `\w` is ASCII-only
                                        # by default; Python's is everything `str.isalnum()` accepts plus
                                        # underscore, which is exactly categories L, N and `_`. Measured:
                                        # `\w+` tokenizes `café` to `caf`, `m`, `nchen` and drops `日本語`
                                        # entirely, so non-ASCII keys silently stop matching (`AT-032d`).
                                        # It is also the positive form of the `/[^\p{L}\p{N}_]+/u` regex the
                                        # SDK cites for TypeScript parity.
                                        "input": {
                                            "$regexFindAll": {
                                                "input": {"$toLower": "$_id"},
                                                "regex": r"[\p{L}\p{N}_]+",
                                            }
                                        },
                                        "in": "$$this.match",
                                    }
                                },
                                sorted(tokens),
                            ]
                        }
                    }
                }
            }
        )
        stages.append({"$match": {"score": {"$gt": 0}}})
        # Blocking, and it cannot be otherwise: the sort key is the field the stage above computed. `_id`
        # ascending is the tie-break, which keeps the order stable and matches what the SDK's stable sort
        # produces from an already-sorted key listing.
        stages.append({"$sort": {"score": -1, "_id": 1}})
        return stages

    @_reporting("deleting the prefix")
    async def delete_prefix(self, prefix: str) -> int:
        """Remove every key under ``prefix`` in one call, returning how many went.

        Not part of the ``Storage`` protocol. It exists because deleting a session means deleting a key
        range, and the alternative is a ``list`` followed by one ``delete`` per key, which is a round
        trip per snapshot for a history that can run to hundreds (``REQ-E-031``).

        Args:
            prefix: The key prefix to remove. ``""`` empties this instance's key space.

        Returns:
            The number of documents removed.

        Raises:
            StorageError: If the prefix contains a ``..`` segment, or if the delete fails.
        """
        result = await self._collection().delete_many(self._range_for(prefix))
        return int(result.deleted_count)

    def namespace(self, prefix: str) -> MongoDBStorage:
        """Return a second instance scoped to ``prefix`` within the same collection.

        A real ``MongoDBStorage`` rather than the SDK's ``_NamespacedStorage`` wrapper, following
        ``LocalFileStorage``. The wrapper has exactly the five protocol methods, so it would put
        ``delete_prefix`` out of reach and turn ``list`` back into a full scan plus a Python-side strip.

        The sentinel is what makes that safe. Three SDK call sites read ``_namespaced`` to decide
        whether the caller already scoped the storage, and without it the memory store's keys land a
        level deeper here than they do on ``LocalFileStorage`` (``REQ-E-039``).

        Args:
            prefix: Prefix to prepend to every key, appended to any prefix already in force.

        Returns:
            A new instance sharing this one's connection pool, collection and TTL.
        """
        scoped = MongoDBStorage(
            self._config,
            collection=self._collection_name,
            # `__init__` normalizes, so a leading slash on `prefix` collapses against the trailing
            # slash on `_prefix` rather than producing an empty path segment.
            prefix=f"{self._prefix}{prefix}",
            ttl_seconds=self._ttl_seconds,
        )
        scoped._namespaced = _NAMESPACED
        return scoped

    @property
    def _target(self) -> str:
        """The MongoDB namespace this instance writes to, for error messages.

        Named ``_target`` rather than ``_namespace`` because ``namespace()`` is a method on this class
        and the two would read as the same thing while meaning a key prefix and a collection.
        """
        return f"{self._database_name}.{self._collection_name}"

    @property
    def _connection(self) -> _client._Connection:
        """The connection serving the running event loop (``REQ-E-017b``).

        A property, not an attribute set in ``__init__``. ``resolve`` returns the entry for the loop
        that is running right now, and a reference stored at construction would keep handing out the
        client bound to whichever loop built the instance. That matters because the SDK reaches this
        package from more than one loop on its own: every synchronous entry point runs through
        ``strands._async.run_async``, which calls ``asyncio.run`` on a fresh loop in a worker thread, so
        a single ``invoke_async`` can touch two loops without the caller doing anything unusual.

        Cheap enough to call per operation: a dict lookup on a hit, and a hit is the common case.
        """
        return _client.resolve(self._config)

    def _collection(self) -> AsyncCollection[dict[str, Any]]:
        """Resolve the collection this instance operates on.

        A method rather than a cached attribute, because the client behind it differs per event loop
        (``REQ-E-017a``) and a collection captured at construction would outlive the loop its client is
        bound to.
        """
        collection: AsyncCollection[dict[str, Any]] = self._connection.client[self._database_name][
            self._collection_name
        ]
        return collection

    def _id_for(self, key: str) -> str:
        """Normalize ``key``, apply this instance's prefix, and check the result against the ceiling.

        Args:
            key: The caller's key.

        Returns:
            The ``_id`` to use.

        Raises:
            StorageError: If the key is empty, contains a ``..`` segment, or encodes to more than
                :data:`MAX_KEY_BYTES` bytes.
        """
        document_id = f"{self._prefix}{_normalize_key(key)}"
        # The prefixed form, because that is what becomes an index entry. A scoped instance measuring
        # only the caller's half would write the entries the ceiling exists to prevent.
        encoded = len(document_id.encode("utf-8"))
        if encoded > MAX_KEY_BYTES:
            raise StorageError(
                f"strands_mongodb: the key {_abbreviate(document_id)!r} encodes to {encoded} bytes, "
                f"over the {MAX_KEY_BYTES}-byte limit this package applies to keys in {self._target}. "
                f"The limit is counted in UTF-8 bytes rather than characters."
            )
        return document_id

    def _range_for(self, query: str) -> dict[str, Any]:
        """Build the ``_id`` filter for a prefix, as a range rather than a regular expression.

        A range is what an index can serve. ``{"_id": {"$regex": "^prefix"}}`` reads as the same query
        and scans the whole collection unless the pattern is anchored in a way the planner recognises,
        which is a difference no unit test would show and every large collection would.

        Args:
            query: The caller's prefix, before normalization.

        Returns:
            A filter document, empty when the query and this instance's prefix are both empty.

        Raises:
            StorageError: If the prefix contains a ``..`` segment.
        """
        lower = f"{self._prefix}{_normalize_prefix(query)}"
        if not lower:
            return {}
        upper = _successor(lower)
        if upper is None:
            return {"_id": {"$gte": lower}}
        return {"_id": {"$gte": lower, "$lt": upper}}

    async def close(self) -> None:
        """Release the MongoDB connection, closing the client only if this package built it.

        Not part of the ``Storage`` protocol, which has no ``close()``. It exists because
        ``REQ-E-011`` needs a way to release a client we opened, and it does nothing at all when the
        caller injected their own (``REQ-E-010``).

        Scoped to the **config**, not to this instance: the registry holds one entry per connection
        identity, so this closes the pool every other instance built from the same config is using.
        Call it once at process shutdown rather than in a per-request ``finally``. Idempotent.

        A sibling that keeps working afterwards is not left holding the closed client. It resolves per
        operation (``REQ-E-017b``), finds no entry, and opens a new pool, which nothing will close again.
        That is the other half of the close contract and the reason the advice above is worth
        following rather than merely noting.
        """
        await _client.close(self._config)
