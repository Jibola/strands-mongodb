"""``MongoDBStorage``: MongoDB behind the Strands unified ``Storage`` protocol.

Requires the ``storage`` extra::

    pip install 'strands-mongodb[storage]'

Pass one instance to ``Agent(storage=...)`` and it serves all five SDK subsystems that use storage:
sessions, context offloading, memory persistence, context history and transcripts.
"""

from __future__ import annotations

import builtins
import datetime
import functools
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

from ._extras import missing_extra

# Imported only to prove the `storage` extra is installed, so a missing extra fails here with a clear
# message instead of as a bare ModuleNotFoundError inside a method call.
try:
    import pymongo  # noqa: F401
    from strands.storage import Storage  # noqa: F401
except ImportError as error:
    missing_extra("storage", error, module=__name__)

from pymongo.errors import PyMongoError
from strands.storage.search import KeywordSearchStrategy

# Private in the SDK, but every shipped backend uses the same two normalizers: they make `a//b` and `a/b`
# one key everywhere. A local copy that drifted would make keys collide on `LocalFileStorage` and not here.
from strands.storage.storage import _NAMESPACED, StorageSearchResult, _normalize_key, _normalize_prefix
from strands.types.exceptions import StorageError

from . import _client, _indexes, _telemetry
from ._indexes import EXPIRES_FIELD
from .vector_search import MongoDBVectorSearchStrategy

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from pymongo.asynchronous.collection import AsyncCollection
    from strands.storage.search import SearchStrategy

    from .types import MongoDBConfig

__all__ = ["MAX_DOCUMENT_BYTES", "MAX_KEY_BYTES", "MongoDBStorage"]

MAX_KEY_BYTES: Final[int] = 1024
"""Longest key this package writes, in UTF-8 bytes, including the instance prefix.

A package limit, not a server one. MongoDB 8.3.8 accepts a 1,000,000-byte ``_id`` at the default feature
compatibility version, but a long key makes every ``list`` range scan pay for a long index entry, and it
fails outright on a deployment pinned below FCV 4.2. 1024 is that old index key limit. The longest key
the SDK builds for a realistic session and agent id pair is about 131 bytes.
"""

MAX_DOCUMENT_BYTES: Final[int] = 16 * 1024 * 1024
"""The BSON document size limit (16 MiB) that ``write`` checks before sending.

Defined here because ``pymongo.common.MAX_BSON_SIZE`` is not public API. A test pins it to the
``maxBsonObjectSize`` a real deployment reports.
"""

# Bytes a stored document costs besides its key and payload: the 4-byte length, the `_id` and `data`
# element headers, the whole `updated_at` element and the terminator, plus the `expires_at` element when
# a TTL is set. A test pins both numbers to `bson.encode`.
_DOCUMENT_OVERHEAD_BYTES: Final[int] = 46
_EXPIRES_OVERHEAD_BYTES: Final[int] = 20

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _encoded_size(key: str, data: bytes, *, expires: bool) -> int:
    """Size of the document ``write`` would build, without building it.

    Computed arithmetically because encoding a payload near 16 MiB only to measure it doubles peak
    memory on the call most likely to be short of it. A test checks the result matches the encoder
    exactly, including multi-byte keys.

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

    The upper bound of the ``_id`` range scan. Simply incrementing the last character fails on two
    inputs, both valid keys:

    * ``U+D7FF`` would increment to ``U+D800``, a lone surrogate that BSON cannot encode. Nothing sorts
      between the two, so jumping to ``U+E000`` keeps the bound exact.
    * ``U+10FFFF`` has no successor and ``chr`` raises. Incrementing the character to its left is still
      an exact bound, and a prefix made only of ``U+10FFFF`` has no upper bound at all.

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
    """Turn any driver error raised by the wrapped method into ``StorageError``.

    It wraps the whole method rather than one awaitable because ``list`` raises while its cursor drains,
    after ``find`` has returned. Wrapping only the call would let that ``PyMongoError`` escape.

    The message uses the same sentence as ``_indexes``, and a test keeps the two in step.

    The wrapped type is ``Coroutine``, not ``Awaitable``: ``Storage`` declares its methods with
    ``async def``, and widening them would make ``MongoDBStorage`` fail the protocol under mypy.

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
                # Through a local because mypy rejects indexing `args` of a `ParamSpec` directly.
                positional: tuple[Any, ...] = args
                storage: MongoDBStorage = positional[0]
                subject = positional[1] if len(positional) > 1 else ""
                raise StorageError(
                    f"strands_mongodb: {verb} {_abbreviate(str(subject))!r} in {storage._target} failed: {error}"
                ) from error

        return reporting

    return decorate


# `Storage` is satisfied structurally, not inherited, like the SDK's own backends.
# `for_sandbox` is absent on purpose: the SDK probes it with `hasattr` and falls back, so a stub that
# raised would break context offloading instead of degrading it.


class MongoDBStorage:
    """A ``strands.storage.Storage`` backed by a MongoDB collection.

    Each key is stored as one document whose ``_id`` is the key (with this instance's prefix), so every
    operation is a primary-key lookup or a range scan over the ``_id`` index.

    The connection settings arrive as a ``MongoDBConfig``; build it once per process and reuse it.
    Instances given the same config object share one connection pool. Per-instance placement
    (collection, prefix, TTL) is passed as keyword arguments.

    Keys are normalized the way the SDK's own backends normalize them: runs of ``/`` collapse and leading
    and trailing ``/`` are stripped, so ``a//b`` and ``/a/b/`` are the key ``a/b``. A key that is empty
    after that, or that contains a backslash or a ``..`` segment, raises ``StorageError``.

    Every key operation (``write``, ``read``, ``delete``, ``list``, ``delete_prefix``, and the default
    keyword ``search``) and ``ensure_indexes`` raise ``StorageError`` if the collection's default
    collation is not binary. Keys must compare byte by byte, or one prefix could read and delete another's
    documents.

    Example:
        ```python
        config: MongoDBConfig = {"database": "agents", "connection_string": "mongodb+srv://..."}
        sessions = MongoDBStorage(config, collection="sessions")
        transcripts = MongoDBStorage(config, collection="transcripts")  # same pool
        ```
    """

    # Declared without a value on purpose. The SDK reads `getattr(storage, "_namespaced", None)` to tell
    # whether a storage is already scoped, so it must stay absent until `namespace()` sets it.
    _namespaced: object

    def __init__(
        self,
        config: MongoDBConfig,
        /,
        *,
        collection: str = "storage",
        prefix: str = "",
        ttl_seconds: int | None = None,
        search_strategy: SearchStrategy[MongoDBStorage] | None = None,
    ) -> None:
        """Create a storage over one collection.

        Args:
            config: Connection and deployment settings. Positional-only. Pass the same object to a
                ``MongoDBMemoryStore`` (through ``MongoDBMemoryStoreConfig.config``) to share the pool.
            collection: Collection holding this instance's documents.
            prefix: Prefix prepended to every key, scoping this instance within the collection.
                Normalized to end in exactly one ``/``, or to be empty, so ``"a"``, ``"a/"`` and ``"/a"``
                are one key space.
            ttl_seconds: Lifetime of documents this instance writes, in seconds, counted from each write.
                ``None`` means they never expire, which is not the same as ``0`` (expire at the next TTL
                sweep, which the server runs about once a minute). This only stamps ``expires_at``:
                **expiry needs the TTL index, which nothing creates implicitly**. Call
                :meth:`ensure_indexes` once at startup, or run the ``createIndexes`` command in the
                README's "Indexes and privileges" section. Documents written before the index exists
                expire once it does.
            search_strategy: What :meth:`search` runs. Defaults to the SDK's ``KeywordSearchStrategy``;
                pass a :class:`MongoDBVectorSearchStrategy` for Atlas Vector Search.

        Raises:
            KeyError: If ``config`` has no ``database``. Mypy rejects a literal that omits it; this
                catches a config assembled at runtime.
            ValueError: If ``config`` supplies neither ``client`` nor ``connection_string``.
            ValueError: If ``ttl_seconds`` is negative.
            ValueError: If the driver rejects a value in ``connection_string``, such as an out-of-range
                port.
            pymongo.errors.ConfigurationError: If the driver cannot parse ``connection_string``, such as
                one with no host or a scheme other than ``mongodb://`` or ``mongodb+srv://``.
            StorageError: If ``prefix`` contains a backslash or a ``..`` segment.
        """
        # Read before `resolve`, which registers a pool in the process-wide registry. Raising after that
        # would leave a pool no instance owns and no `close()` can reach.
        self._database_name = config["database"]
        _indexes.check_ttl(ttl_seconds)
        self._collection_name = collection
        # `_normalize_prefix` keeps a trailing slash, so strip it and add exactly one back.
        normalized = _normalize_prefix(prefix).rstrip("/")
        self._prefix = f"{normalized}/" if normalized else ""
        self._ttl_seconds = ttl_seconds
        self._search_strategy = search_strategy
        self._config = config
        # Only passed on to `_indexes.ensure`; read so a value set for a memory store is honoured here too.
        self._index_ready_timeout: float = config.get("index_ready_timeout", _indexes.READY_TIMEOUT_SECONDS)
        # Called only to fail fast on a config with no client or connection string. The result is
        # dropped: the registry is keyed by event loop, so each operation resolves its own connection.
        _client.resolve(config)

    @property
    def prefix(self) -> str:
        """The normalized key prefix (ending in ``/``), or ``""`` when none was configured."""
        return self._prefix

    @_telemetry.traced("storage", "write")
    @_reporting("writing")
    async def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, replacing whatever was there.

        A single upsert, so two writers racing on one key never hit a duplicate-key error. When the
        instance has ``ttl_seconds``, the document gets an ``expires_at`` stamp.

        Args:
            key: The key to write under. Normalized, so ``a//b`` and ``a/b`` are one key.
            data: The bytes to store, round-tripped unchanged including ``b""``.

        Raises:
            StorageError: If the key is empty, contains a backslash or a ``..`` segment, exceeds
                :data:`MAX_KEY_BYTES` with the prefix included, or would build a document over
                :data:`MAX_DOCUMENT_BYTES`. Also if the collection's default collation is not binary, if
                the write fails, with the driver's error chained, or if the search strategy fails to index
                the value (which is stored in that case).
        """
        await self._require_binary_keys()
        document_id = self._id_for(key)
        expires = self._ttl_seconds is not None
        size = _encoded_size(document_id, data, expires=expires)
        if size > MAX_DOCUMENT_BYTES:
            # Checked before sending so the caller sees the sizes, which the driver's `DocumentTooLarge`
            # does not report.
            raise StorageError(
                f"strands_mongodb: the document for the key {_abbreviate(document_id)!r} would encode "
                f"to {size} bytes, over the {MAX_DOCUMENT_BYTES}-byte BSON limit in {self._target}. "
                f"Store the payload elsewhere and keep a reference here, or split it across keys."
            )

        now = datetime.datetime.now(datetime.timezone.utc)
        document: dict[str, Any] = {"_id": document_id, "data": data, "updated_at": now}
        if self._ttl_seconds is not None:
            # Only with a TTL. The TTL index never expires a non-date value, so `None` would be dead bytes.
            document[EXPIRES_FIELD] = now + datetime.timedelta(seconds=self._ttl_seconds)

        await self._collection().replace_one({"_id": document_id}, document, upsert=True)

        # `SearchStrategy` gained `index` in strands-agents 1.56.0; a strategy from an older release has none.
        index = getattr(self._search_strategy, "index", None)
        if index is None:
            return
        try:
            await index(self, document_id[len(self._prefix) :], data)
        except Exception as error:
            raise StorageError(
                f"strands_mongodb: wrote {_abbreviate(document_id)!r} in {self._target} but indexing it for "
                f"search failed: {error}"
            ) from error

    @_telemetry.traced("storage", "read")
    @_reporting("reading")
    async def read(self, key: str) -> bytes | None:
        """Return the bytes stored under ``key``, or ``None`` if there are none.

        A missing key is not an error: the SDK's session manager reads it as "no session yet".

        Args:
            key: The key to read.

        Returns:
            The stored bytes, or ``None`` when the key is absent.

        Raises:
            StorageError: If the key is empty, contains a backslash or a ``..`` segment, or exceeds
                :data:`MAX_KEY_BYTES`. Also if the collection's default collation is not binary, or if
                the read fails, with the driver's error chained. An absent key is not a failure.
        """
        await self._require_binary_keys()
        document = await self._collection().find_one({"_id": self._id_for(key)})
        _telemetry.annotate("strands_mongodb.found", document is not None)
        if document is None:
            return None
        # pymongo decodes binary subtype 0 (what this package writes) to `bytes`, but subtypes 2 and 0x80
        # to `Binary`, whose `__eq__` also compares the subtype. Data written by another client would then
        # not equal the stored payload.
        return bytes(document["data"])

    @_telemetry.traced("storage", "delete")
    @_reporting("deleting")
    async def delete(self, key: str) -> None:
        """Remove ``key``. Deleting a key that does not exist is not an error.

        Args:
            key: The key to remove.

        Raises:
            StorageError: If the key is empty, contains a backslash or a ``..`` segment, or exceeds
                :data:`MAX_KEY_BYTES`. Also if the collection's default collation is not binary, or if
                the delete fails, with the driver's error chained.
        """
        await self._require_binary_keys()
        result = await self._collection().delete_one({"_id": self._id_for(key)})
        _telemetry.annotate("strands_mongodb.deleted_count", int(result.deleted_count))

    @_telemetry.traced("storage", "list")
    @_reporting("listing")
    async def list(self, query: str) -> builtins.list[str]:
        """Return every key under the prefix ``query``, in ascending order.

        Only keys are fetched, never payloads. MongoDB's byte order over UTF-8 matches Python's
        ``sorted()`` for every key this class accepts.

        Args:
            query: A key prefix. ``""`` matches everything, which is how the SDK's keyword search
                strategy reads the whole store.

        Returns:
            Full keys (not just the part after ``query``), without this instance's own prefix.

        Raises:
            StorageError: If the prefix contains a backslash or a ``..`` segment, if the collection's
                default collation is not binary, or if the scan fails at any point, including part way
                through draining the cursor.
        """
        await self._require_binary_keys()
        cursor = self._collection().find(self._range_for(query), {"_id": 1})
        # Strip only the instance prefix, so the keys returned can be passed straight back to `read`.
        offset = len(self._prefix)
        keys = [str(document["_id"])[offset:] async for document in cursor.sort("_id", 1)]
        _telemetry.annotate("strands_mongodb.result_count", len(keys))
        return keys

    @_telemetry.traced("storage", "search")
    @_reporting("searching")
    async def search(self, query: str) -> builtins.list[StorageSearchResult]:
        """Rank stored values against ``query`` with the configured search strategy.


        Args:
            query: A natural-language string.

        Returns:
            Matched keys relative to this instance's prefix, best first, with the stored bytes as
            ``data``.

        Raises:
            StorageError: If the collection's default collation is not binary, if a driver call fails, or
                if the search strategy raises, with its error chained.
        """
        if self._search_strategy is None:
            results = await KeywordSearchStrategy().search(self, query)
        else:
            try:
                results = await self._search_strategy.search(self, query)
            except (StorageError, PyMongoError):
                raise
            except Exception as error:
                raise StorageError(f"strands_mongodb: searching in {self._target} failed: {error}") from error
        _telemetry.annotate("strands_mongodb.result_count", len(results))
        return results

    @_telemetry.traced("storage", "delete_prefix")
    @_reporting("deleting the prefix")
    async def delete_prefix(self, prefix: str) -> int:
        """Remove every key under ``prefix`` in one call and return how many were removed.

        Not part of the ``Storage`` protocol. Use it to delete a whole session in one round trip instead
        of a ``list`` followed by one ``delete`` per key.

        Args:
            prefix: The key prefix to remove. ``""`` empties this instance's key space.

        Returns:
            The number of documents removed.

        Raises:
            StorageError: If the prefix contains a backslash or a ``..`` segment, if the collection's
                default collation is not binary, or if the delete fails, with the driver's error chained.
        """
        await self._require_binary_keys()
        result = await self._collection().delete_many(self._range_for(prefix))
        removed = int(result.deleted_count)
        _telemetry.annotate("strands_mongodb.deleted_count", removed)
        return removed

    def namespace(self, prefix: str) -> MongoDBStorage:
        """Return a new instance scoped to ``prefix`` within the same collection.

        The result is a full ``MongoDBStorage``, so ``delete_prefix`` and the indexed ``list`` keep
        working on it.

        Args:
            prefix: Prefix to prepend to every key, appended to any prefix already in force.

        Returns:
            A new instance sharing this one's connection pool, collection, TTL and search strategy.

        Raises:
            StorageError: If ``prefix`` contains a backslash or a ``..`` segment.
        """
        # A real instance rather than the SDK's `_NamespacedStorage`, which exposes only the five protocol
        # methods and would turn `list` into a full scan with a Python-side strip.
        scoped = MongoDBStorage(
            self._config,
            collection=self._collection_name,
            # `__init__` normalizes, so a leading slash here does not produce an empty path segment.
            prefix=f"{self._prefix}{prefix}",
            ttl_seconds=self._ttl_seconds,
            search_strategy=self._search_strategy,
        )
        # The SDK reads this sentinel to tell that a storage is already scoped. Without it, memory keys
        # would land one level deeper here than on `LocalFileStorage`.
        scoped._namespaced = _NAMESPACED
        return scoped

    @_telemetry.traced("storage", "ensure_indexes")
    async def ensure_indexes(self) -> None:
        """Create the TTL index that makes ``ttl_seconds`` take effect, or verify it exists.

        The index is ``strands_expires_at_ttl`` over ``expires_at`` with ``expireAfterSeconds: 0``.
        Without it, documents get an ``expires_at`` stamp but are never deleted. Also creates a
        :class:`MongoDBVectorSearchStrategy`'s vector index.

        **Nothing in this package calls this method.** An ``Agent`` builds its storage in code the
        operator does not see, and index creation there could run without the rights for it. Call it once
        at startup::

            storage = MongoDBStorage(config, ttl_seconds=3600)
            await storage.ensure_indexes()

        Idempotent and safe to call concurrently. With ``ensure_indexes: False`` in the config it only
        verifies: a missing index raises with the ``createIndexes`` command that would fix it. Documents
        written before the index exists expire once it does.

        Raises:
            StorageError: If the collection's default collation is not binary, if an index over the same
                keys exists with different options, if the index is absent and ``ensure_indexes`` is
                ``False``, if the vector index build fails or does not finish in time, or if a driver
                call fails. The server's own code and message arrive chained.
        """
        # The collation does not affect a TTL index, but a collection that cannot hold this key space is
        # not worth provisioning: it would only fail at its first write.
        await self._require_binary_keys()
        # No `@_reporting`: there is no key argument for its message, and `_indexes.ensure` already turns
        # driver errors into `StorageError`.
        strategy = self._search_strategy
        await _indexes.ensure(
            self._collection(),
            ordinary=_indexes.storage_specs(),
            search=(strategy.index_spec(),) if isinstance(strategy, MongoDBVectorSearchStrategy) else (),
            create=self._config.get("ensure_indexes", True),
            timeout=self._index_ready_timeout,
        )

    async def _require_binary_keys(self) -> None:
        """Refuse a collection whose collation would not compare this instance's keys byte by byte.

        The reason for refusing is on :func:`_indexes.require_binary_comparison`. This method decides
        when: before the first operation on a collection, not at construction (``__init__`` is
        synchronous and must not touch the network) and not in ``ensure_indexes`` (which callers may
        never call).

        No lock: the check is a side-effect-free read, so two racing first operations each issue one
        ``listCollections`` and reach the same answer.

        Only a verified collection is cached. A refusal is not cached, so a collection fixed while the
        process runs starts working. An absent collection is not cached either, because
        ``listCollections`` cannot tell it apart from one with no collation, and whoever creates it may
        give it any collation. The check repeats until this store's first write creates the collection.

        Raises:
            StorageError: If the collection's default collation is not binary, or if reading it fails.
        """
        target = (self._database_name, self._collection_name)
        connection = self._connection
        if target in connection.binary_keys:
            return
        if await _indexes.require_binary_comparison(self._collection()):
            connection.binary_keys.add(target)

    @property
    def _target(self) -> str:
        """The ``database.collection`` this instance writes to, for error messages.

        Not ``_namespace``, which would read like the ``namespace()`` key-prefix method.
        """
        return f"{self._database_name}.{self._collection_name}"

    @property
    def _connection(self) -> _client._Connection:
        """The connection serving the running event loop.

        Resolved on each access, never stored: the SDK runs synchronous entry points through
        ``asyncio.run`` on a fresh loop in a worker thread, so one instance can be used from several
        loops, and a client is bound to one. A lookup hit is a dict access.
        """
        return _client.resolve(self._config)

    def _collection(self) -> AsyncCollection[dict[str, Any]]:
        """Resolve the collection, through the running loop's client (see ``_connection``)."""
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
            StorageError: If the key is empty, contains a backslash or a ``..`` segment, or encodes to
                more than :data:`MAX_KEY_BYTES` bytes.
        """
        document_id = f"{self._prefix}{_normalize_key(key)}"
        # Measure the prefixed form, since that is what becomes the index entry.
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

        The ``_id`` index serves a range. An ``{"$regex": "^prefix"}`` filter looks equivalent but scans
        the whole collection unless the planner recognises the anchor, which no unit test would show.

        Args:
            query: The caller's prefix, before normalization.

        Returns:
            A filter document, empty when the query and this instance's prefix are both empty.

        Raises:
            StorageError: If the prefix contains a backslash or a ``..`` segment.
        """
        lower = f"{self._prefix}{_normalize_prefix(query)}"
        if not lower:
            return {}
        upper = _successor(lower)
        if upper is None:
            return {"_id": {"$gte": lower}}
        return {"_id": {"$gte": lower, "$lt": upper}}

    async def close(self) -> None:
        """Release the MongoDB connection, closing the client only if this package created it.

        Not part of the ``Storage`` protocol. Does nothing to a client you passed in through
        ``config["client"]``. Idempotent.

        **This closes the pool for every instance built from the same config**, not just this one. Call
        it once at process shutdown, not per request. An instance used after the close opens a new pool,
        which stays open until ``close()`` is called again.
        """
        await _client.close(self._config)
