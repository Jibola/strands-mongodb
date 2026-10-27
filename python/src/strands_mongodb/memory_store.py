"""``MongoDBMemoryStore``: Atlas Vector Search behind the Strands ``MemoryStore`` protocol.

Requires the ``memory`` extra::

    pip install 'strands-mongodb[memory]'

Register the store with the SDK's ``MemoryManager`` alongside any other stores. The protocol requires
only ``search``; the manager detects ``add`` and ``add_messages`` when they exist.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

from ._extras import missing_extra

# Imported to prove the `memory` extra is installed, so a missing extra fails here with a clear message
# rather than as a bare ModuleNotFoundError inside a method call.
try:
    import pymongo  # noqa: F401
    from pymongo_search_utils import __name__ as _pymongo_search_utils  # noqa: F401
    from strands.memory import MemoryStore  # noqa: F401
except ImportError as error:
    missing_extra("memory", error, module=__name__)

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import DuplicateKeyError, PyMongoError
from strands.memory.types import MemoryEntry
from strands.types.exceptions import StorageError

from . import _client, _filter, _indexes, _pipeline, _telemetry
from .types import DEFAULT_MAX_SEARCH_RESULTS, DEFAULT_OVERSAMPLING_FACTOR, MongoDBAddResult

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping

    from pymongo.asynchronous.collection import AsyncCollection
    from strands.memory.extraction.types import ExtractionConfig
    from strands.memory.types import Metadata, SearchOptions
    from strands.types.tools import AgentTool
    from typing_extensions import Unpack

    from ._indexes import SearchIndexSpec, VectorMode
    from .types import MongoDBMemoryStoreConfig, MongoDBSearchQuery, ResolvedSearchType, SearchType

__all__ = ["DEFAULT_COLLECTION", "TIER_RETRY_MAX_SECONDS", "TIER_RETRY_SECONDS", "UNSCOPED", "MongoDBMemoryStore"]

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

DEFAULT_COLLECTION = "memory"
"""Collection this store uses when the config names none.

Different from ``MongoDBStorage``'s ``"storage"`` on purpose: a shared default would put session blobs
in the collection whose vector index tries to embed them.
"""

UNSCOPED: Final[str] = "__unscoped__"
"""Scope stamped on writes and filtered on by reads when the config names none.

A sentinel rather than ``None``, ``""`` or no scope predicate at all. Dropping the predicate would let
an unconfigured store read every tenant's documents. ``None`` would match nothing, and ``""`` is falsy,
so an ``if scope:`` check would treat it as unset. The double underscores keep it from colliding with a
real scope.
"""

TIER_RETRY_SECONDS: Final[float] = 60.0
"""Delay in seconds before a search tier skipped under ``auto`` is tried again.

The retry runs inside the first store call after the delay. It reads index status without waiting for a
build, and a retry that fails keeps the tiers the store already has.
"""

TIER_RETRY_MAX_SECONDS: Final[float] = 3600.0
"""Ceiling in seconds on that delay, which doubles after each retry that still finds the tier missing."""

_NO_SEARCH_SETUP = (
    "strands_mongodb: %s is on a deployment that does not run Atlas Search, so no search index was "
    "created and no query can be answered here. The server reported error code %s when asked to list "
    "its search indexes. Ordinary indexes and writes are unaffected, so this store can hold memories "
    "and cannot read them back. Point 'connection_string' at MongoDB Atlas, or at a local Atlas "
    "deployment started with 'atlas deployments setup --type local', which runs the same search process."
)
"""Warning ``initialize()`` logs on a deployment without Atlas Search, once per connection.

It names the server's error code and what to connect to, but no command or index status: the
search-index commands do not exist on this deployment, so that advice could not work. The code is
interpolated because different deployments report different codes.
"""

_NO_SEARCH_QUERY = (
    "strands_mongodb: cannot search %s because this deployment does not run Atlas Search. Its capability "
    "probe came back with %s, and this package has no tier that answers without a search process, so "
    "there is nothing to fall back to. Writes still work, which is why "
    "initialize() only warned. Point 'connection_string' at MongoDB Atlas, or at a local Atlas "
    "deployment started with 'atlas deployments setup --type local'."
)
"""Raised by ``search`` on a deployment without Atlas Search.

Setup only warns, because writes still work. A query raises, because returning no results would look
like an empty memory.
"""

_TIER_UNAVAILABLE = (
    "strands_mongodb: the %s search tier is not available on %s and 'search_type' is 'auto', so it is "
    "skipped rather than fatal and any remaining tier answers queries instead. The '_search_type' key "
    "on every returned entry reports which one did. The server's own reason: %s"
)
"""Warning for each tier skipped under ``auto``.

The server's message is included whole because it says what to fix. On an Atlas CLI local deployment,
for example, it reports that the embedding model is not registered.
"""

_NO_TIER_PREPARED = (
    "strands_mongodb: 'search_type' is 'auto' and no search tier could be prepared on %s, so no query "
    "can be answered. The last tier's failure is chained to this one and every tier's is logged above, "
    "each with the server's own reason."
)
"""Raised under ``auto`` when no search tier could be prepared, after each failure was logged."""

_BELOW_VECTOR = (
    "strands_mongodb: 'search_type' is 'auto' and the vector tier is not available on %s, so queries "
    "resolve to the %s tier. Answers are lexical rather than semantic, which is a difference in what "
    "matches rather than only in ranking."
)
"""Warning, at most once per store, when ``auto`` resolves to a tier below vector."""

_NO_FUSION = (
    "strands_mongodb: 'search_type' is 'hybrid' but %s cannot run it. The hybrid tier is a $rankFusion "
    "over the vector and text pipelines, and that stage needs MongoDB 8.0 or later. Upgrade the "
    "deployment, or set 'search_type' to 'auto' to let this store resolve to the tiers it does have."
)
"""Raised when ``search_type`` is ``"hybrid"`` and the deployment cannot run ``$rankFusion``.

Fatal, like any named ``search_type`` that cannot be prepared: answering from another tier would change
what matches, not only the ranking. The remedy offers both an upgrade and ``auto``, because an
application author often cannot upgrade the deployment.
"""

_BOUND_DECLINED_FUSION = (
    "strands_mongodb: 'search_type' is 'auto' and %s can run the hybrid tier, but '%s' is set to %s and "
    "the two are on different scales, so queries resolve to the vector tier as they did before hybrid "
    "existed. Reciprocal rank fusion scores sum to about 0.033 at best, while cosine scores land in "
    "roughly 0 to 1. Remove the bound to get fused ranking, or set 'search_type' to 'hybrid' to keep "
    "the bound and have it applied on the fusion's scale."
)
"""Warning, at most once per store, when a score bound keeps ``auto`` off the hybrid tier.

It names the bound and its value, because the remedy depends on which bound is set and an empty result
set would not reveal it.
"""

_NO_FUSION_FOR_QUERY = (
    "strands_mongodb: this query sets both 'text' and 'vector', which only the hybrid tier reads, and "
    "the hybrid tier is not available on %s. The tiers that are: %s. Drop one of the two keys to use "
    "one of them, or see the initialization log for why the fused tier was not offered."
)
"""Raised when a query sets both ``text`` and ``vector`` and the hybrid tier is not available.

The config allowed the shape, so this is only known after ``initialize()``. Answering from one tier
would silently ignore one of the two keys.
"""

_NO_TIER_FOR_QUERY = (
    "strands_mongodb: this query cannot be answered by any search tier prepared on %s, which is: %s. "
    "The text tier needs 'text' on the query. The vector tier needs %s, because %s. Supply that key, "
    "or set 'search_type' to a tier this query fits."
)
"""Raised when tiers are prepared but none of them fits the query's shape."""

_NOT_WRITABLE = (
    "strands_mongodb: the memory store %r is not writable, so it does not accept %s. Set "
    "'writable': True in its configuration to enable writes. It defaults to False so that a store "
    "built for retrieval does not start accepting model-distilled extraction writes, or deletions, "
    "that nobody asked for."
)
"""Raised by ``add``, ``delete`` and ``delete_all`` on a store that is not writable.

It names the config key to set and the operation that was refused.
"""

_BAD_MEMORY_ID = (
    "strands_mongodb: 'memory_id' is %r, which is not a 24-character hex ObjectId. Pass the '_id' a "
    "search result carries in its metadata, which is where a deletable id comes from."
)
"""Raised by ``delete`` when ``memory_id`` is not a valid ObjectId string.

The id is rendered because it is the caller's own argument, not stored content, and repeating it back
shows the typo.
"""

_LIMIT_TOO_LOW = (
    "strands_mongodb: %r is %d, and a search limit must be at least 1. A limit below 1 cannot return "
    "anything, so it is rejected here rather than sent to a server that would reject it."
)
"""Raised when any supplied or configured result limit is below 1, naming the key it came from."""

_BOTH_SCORE_BOUNDS = (
    "strands_mongodb: 'min_score' is %r and 'max_score' is %r, and the two are mutually exclusive: a "
    "floor and a ceiling on one similarity score describe a band, and this package applies one bound "
    "rather than a range. Keep whichever one you meant and remove the other."
)
"""Raised by ``__init__`` when both ``min_score`` and ``max_score`` are set."""

_RESERVED_KEY_COLLISION = (
    "strands_mongodb: metadata key(s) %s on a document in the memory store %r collide with the keys "
    "this package sets on every result, so the stored value(s) were dropped from the returned entry "
    "and this package's own value(s) kept. Rename them in whatever wrote them; a stored '_scope' that "
    "won would make a document report a scope it was not stored in."
)
"""Warning when stored metadata uses a key this package reserves. One record per result.

It lists every colliding key and names the store, because several stores can share one collection and
the store is what locates the writer.
"""

_SCORE_BOUND_APPLIED = "strands_mongodb: %s %r on %s kept %d of %d results (applied after the query)."
"""``DEBUG`` record when a score bound drops results.

Both counts are included, so "kept 0 of 8" reads as a bound set too high rather than an empty memory.
"""


def _resolve_limit(*, query: int | None, options: int | None, instance: int | None) -> int:
    """Resolve the result limit for one search.

    Priority, highest first: ``query``, ``options``, ``instance``, then ``DEFAULT_MAX_SEARCH_RESULTS``.
    ``query`` ranks above ``options`` because a ``MongoDBSearchQuery`` only arrives on a direct call, while
    ``MemoryManager.search`` always passes ``options``. Below it, the query's ``limit`` would never apply.

    Every supplied value is checked, not only the winner, so ``limit=5`` with ``max_search_results=0``
    still raises.

    Args:
        query: ``limit`` from a ``MongoDBSearchQuery``, if one was passed.
        options: ``max_search_results`` from this call's ``SearchOptions``.
        instance: The store's own ``max_search_results``.

    Returns:
        The highest-priority value supplied, or ``DEFAULT_MAX_SEARCH_RESULTS``.

    Raises:
        ValueError: If any supplied value is below 1. The message names the key it came from.
    """
    for name, value in (("limit", query), ("max_search_results", options), ("max_search_results", instance)):
        if value is not None and value < 1:
            raise ValueError(_LIMIT_TOO_LOW % (name, value))

    if query is not None:
        return query
    if options is not None:
        return options
    if instance is not None:
        return instance
    return DEFAULT_MAX_SEARCH_RESULTS


def _object_id(memory_id: str) -> ObjectId:
    """Parse a ``memory_id`` string into the ``ObjectId`` stored in ``_id``.

    Strict, with no fallback to matching the raw string, so a malformed id raises instead of quietly
    deleting nothing.

    The type check is needed as well as the parse: ``ObjectId(None)`` generates a new id instead of failing
    (pymongo 4.17.0), which would match nothing and return ``0``, and ``ObjectId(12)`` raises ``TypeError``.

    Args:
        memory_id: The ``_id`` string a ``search`` result carries in its metadata.

    Returns:
        The parsed id.

    Raises:
        ValueError: If it is not a string, or not a 24-character hex ObjectId. ``InvalidId`` is converted
            because it is not a ``ValueError``.
    """
    if not isinstance(memory_id, str):
        raise ValueError(_BAD_MEMORY_ID % (memory_id,))
    try:
        return ObjectId(memory_id)
    except InvalidId as error:
        raise ValueError(_BAD_MEMORY_ID % (memory_id,)) from error


def _merge_metadata(
    caller: Mapping[str, Any] | None,
    reserved: Mapping[str, Any],
    *,
    store_name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Merge a document's stored metadata with this package's reserved keys, the reserved keys winning.

    A stored ``_scope`` that won would make a document report a scope it was not stored in. Collisions
    are logged in one record that lists every colliding key.

    Args:
        caller: ``metadata`` as the document carries it, or ``None`` when it carries none.
        reserved: This package's reserved keys and their values for this result.
        store_name: The store named in the log record.
        namespace: The store's ``database.collection``, attached to the log record.

    Returns:
        A new dict. Neither argument is modified.
    """
    merged: dict[str, Any] = dict(caller or {})
    collisions = sorted(key for key in reserved if key in merged)
    if collisions:
        logger.warning(
            _RESERVED_KEY_COLLISION,
            ", ".join(repr(key) for key in collisions),
            store_name,
            extra=_telemetry.event_extra("reserved_key_collision", namespace=namespace),
        )
    merged.update(reserved)
    return merged


def _reporting(
    verb: str,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """Turn any driver error raised by the wrapped method into ``StorageError``.

    Unlike ``MongoDBStorage``'s decorator of the same name, the message names only the namespace. The
    subject of a memory operation is user content, which must never reach an error, a log line or a
    traceback.

    The type is ``Coroutine`` rather than ``Awaitable`` because the protocol declares these methods
    ``async def``, and a wider type would fail the ``MemoryStore`` check under mypy.

    Args:
        verb: What the method was doing, as a present participle.

    Returns:
        A decorator for an async method whose first argument is ``self``.
    """

    def decorate(method: Callable[_P, Coroutine[Any, Any, _R]]) -> Callable[_P, Coroutine[Any, Any, _R]]:
        @functools.wraps(method)
        async def reporting(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return await method(*args, **kwargs)
            except PyMongoError as error:
                # Through a local, because mypy rejects indexing a `ParamSpec`'s `args` directly.
                positional: tuple[Any, ...] = args
                store: MongoDBMemoryStore = positional[0]
                raise StorageError(f"strands_mongodb: {verb} {store._target} failed: {error}") from error

        return reporting

    return decorate


class MongoDBMemoryStore:
    """Strands ``MemoryStore`` backed by MongoDB Atlas Vector Search and Atlas Search.

    Stores memories as text documents and answers queries from a vector, text or hybrid search tier,
    chosen by ``search_type`` and by what the deployment supports. Register it with the SDK's
    ``MemoryManager``. Beyond the protocol it offers ``delete``, ``delete_all`` and ``close``.

    Every read, write and delete is limited to this store's ``scope`` and ``name``. The package stamps the
    scope on each document and adds it to every query; it is never taken from caller input. A store with
    no ``scope`` uses :data:`UNSCOPED`, so it still filters rather than reading every scope. MongoDB has
    no server-side row filter, so this isolation is enforced by the package.

    A store accepts writes only when ``writable`` is ``True``.

    Example:
        ```python
        connection: MongoDBConfig = {"database": "agents", "connection_string": "mongodb+srv://..."}
        store = MongoDBMemoryStore(name="preferences", config=connection, scope="user-abc")
        ```
    """

    # Satisfies `MemoryStore` structurally instead of inheriting it: inheriting a `Protocol` makes every
    # member this class leaves out abstract, and mypy then refuses to instantiate it.
    #
    # `add_messages` is deliberately absent, not a stub. `AddMessagesContext` has no run-unique id to
    # deduplicate retried writes by, so the manager falls back to its own extractor, which writes through
    # `add` and is deduplicated there.

    def __init__(self, **config: Unpack[MongoDBMemoryStoreConfig]) -> None:
        """Validate the configuration and register the shared connection.

        Every check runs before the connection is registered, so a config that fails leaves no
        connection pool behind.

        Args:
            **config: See ``MongoDBMemoryStoreConfig``. The connection settings are nested under
                ``config`` so one ``MongoDBConfig`` can be shared with ``MongoDBStorage`` and with every
                other store in the process.

        Raises:
            KeyError: If ``name``, the nested ``config``, or that config's ``database`` is missing.
            ValueError: If the nested config supplies neither ``client`` nor ``connection_string``; if
                ``max_search_results`` is below 1; if both ``min_score`` and ``max_score`` are set; if the
                configured ``filter`` names an operator key, a regex value or an undeclared path; if the
                vector configuration selects two paths at once, sets only one of ``embedding_dimensions``
                and ``similarity``, or gives either a value Atlas rejects; or if ``ttl_seconds`` is
                negative. A ``ttl_seconds`` of ``0`` is accepted. Also raised by the driver for a
                connection option it rejects, such as an out-of-range port.
            pymongo.errors.ConfigurationError: If the driver cannot parse ``connection_string`` or
                rejects a ``client_args`` key.
        """
        connection = config["config"]
        self.name = config["name"]
        self.description: str | None = config.get("description")
        self.max_search_results: int | None = config.get("max_search_results")
        # `False` by default, unlike `FileMemoryStore`, because `writable` also gates automatic extraction.
        self.writable: bool = config.get("writable", False)
        self.extraction: ExtractionConfig | bool | None = config.get("extraction")
        self.scope: str = config.get("scope") or UNSCOPED
        self.filter: dict[str, Any] | None = config.get("filter")
        self.min_score: float | None = config.get("min_score")
        self.max_score: float | None = config.get("max_score")
        # `None` means never expire and `0` means expire at the next sweep, so every test of this is
        # `is not None`, never a truth test.
        self.ttl_seconds: int | None = config.get("ttl_seconds")
        _indexes.check_ttl(self.ttl_seconds)

        # Checked here, as `BedrockKnowledgeBaseStore` does, because a limit below 1 can never return anything.
        _resolve_limit(query=None, options=None, instance=self.max_search_results)
        if self.min_score is not None and self.max_score is not None:
            raise ValueError(_BOTH_SCORE_BOUNDS % (self.min_score, self.max_score))

        self._database_name = connection["database"]
        self._collection_name = config.get("collection", DEFAULT_COLLECTION)
        self._ensure_indexes: bool = connection.get("ensure_indexes", True)
        # Passed to every `_indexes.ensure` call. Only search index builds actually wait on it.
        self._index_ready_timeout: float = connection.get("index_ready_timeout", _indexes.READY_TIMEOUT_SECONDS)
        self._search_type: SearchType = connection.get("search_type") or "auto"
        # The index definition decides what is filterable, so the configured filter and every per-query
        # filter are checked against the same list.
        self._declared_fields = _indexes.declared_filter_fields(connection.get("filter_fields") or ())
        if self.filter:
            _filter.validate(self.filter, declared_fields=self._declared_fields)

        # Resolved here because both raise `ValueError` on a contradictory config, which should surface at
        # construction rather than inside the first `add` or `search`.
        self._mode: VectorMode = _indexes.vector_mode(connection)
        vector_spec, text_spec = _indexes.memory_search_specs(connection)
        specs: tuple[tuple[ResolvedSearchType, SearchIndexSpec], ...] = (
            ("vector", vector_spec),
            ("text", text_spec),
        )
        # Vector first, the order `auto` falls back through. A named `search_type` keeps only its own tier,
        # so a `text` store never builds a vector index and a `vector` store fails rather than answering
        # lexically. `hybrid` keeps both, because it is built from them.
        self._specs = (
            specs
            if self._search_type in ("auto", "hybrid")
            else tuple(pair for pair in specs if pair[0] == self._search_type)
        )
        self._index_names = {tier: spec.name for tier, spec in self._specs}
        # Whether a query may set both `text` and `vector`, decided from the config alone so the check runs
        # before any round trip. Only on the manual-vector path: with AutoEmbeddings both hybrid pipelines
        # read `text`, so `vector` has no meaning there.
        self._fusion_possible = not self._mode.auto_embed and self._search_type in ("auto", "hybrid")

        self._config = config
        self._connection_config = connection
        self._initialized = False
        # Set by `initialize()`: the tiers that can answer, in preference order. Empty after `initialize()`
        # means the deployment has no Atlas Search, which fails `search` but not `add`.
        self._tiers: tuple[ResolvedSearchType, ...] = ()
        self._warned_below_vector = False
        # When to retry a skipped tier, or `None` when none is skipped.
        self._tier_retry_at: float | None = None
        self._tier_retry_delay: float = TIER_RETRY_SECONDS
        # From the first capability probe. Retries reuse it because they may run on another event loop's
        # connection, where the probe has not run.
        self._rank_fusion = False
        # Called only to validate, so a config with neither `client` nor `connection_string` raises here.
        # Operations get their connection from `_connection`, per running event loop.
        _client.resolve(connection)

    async def initialize(self) -> None:
        """Probe the deployment and create this store's indexes.

        ``add``, ``search``, ``delete`` and ``delete_all`` call this themselves, so callers need not
        (``MemoryManager`` never does). It is idempotent and safe to call concurrently. After the first
        call it returns at once, unless a search tier skipped under ``auto`` is due for a retry.

        Under ``auto`` a tier that cannot be prepared (an index still building, say) is skipped with a
        warning. A later call retries it after :data:`TIER_RETRY_SECONDS`, and the delay doubles up to
        :data:`TIER_RETRY_MAX_SECONDS` while the tier stays missing. A retry never waits for a build, and
        a retry that fails keeps the tiers the store already has.

        On a deployment without Atlas Search this logs a warning, once per connection, and ``search``
        raises later. Writes still work there.

        Raises:
            StorageError: If a driver call fails, if the collection was created with a collation that
                does not compare strings byte by byte, if an existing index conflicts with one this store
                declares, if an index is missing and ``ensure_indexes`` is ``False``, or if ``search_type``
                names a tier the deployment cannot prepare. Under ``auto`` a search tier that cannot be
                prepared is skipped instead, and only an empty result raises.
        """
        if self._initialized and not self._tier_retry_due():
            return
        await self._initialize()

    def _tier_retry_due(self) -> bool:
        """Whether a tier skipped under ``auto`` is due to be tried again."""
        return self._tier_retry_at is not None and time.monotonic() >= self._tier_retry_at

    @_telemetry.traced("memory", "initialize")
    @_reporting("initializing")
    async def _initialize(self) -> None:
        """Do the work of :meth:`initialize` under the connection's lock, recording a span.

        Split from the unlocked fast path so calls that find the work done record no span. The flag is
        re-read under the lock, so concurrent calls on one store do the work once. The capability probe is
        cached on the connection, because search support belongs to the deployment, so stores sharing a
        connection probe once.
        """
        _telemetry.annotate("strands_mongodb.search.requested", self._search_type)
        # Bound once, so the lock, the probe and the cache write all use the same connection.
        connection = self._connection
        async with connection.get_lock():
            if self._initialized:
                if self._tier_retry_due():
                    await self._retry_search_tiers()
                return

            collection = self._collection()

            # Checked before any index is created. `_filter.compose` pins `scope` and `store_name` on every
            # read, update and delete, so a case-folding collation would let `scope="tenant-b"` match
            # documents written under `Tenant-B`: a cross-scope read. No extra lock is needed, because this
            # block already holds the connection's.
            target = (self._database_name, self._collection_name)
            verified = target in connection.binary_keys
            if not verified:
                verified = await _indexes.require_binary_comparison(collection)
                if verified:
                    connection.binary_keys.add(target)

            capabilities = connection.capabilities
            if capabilities is None:
                # Called through the module, not a direct import, so tests can count one probe per connection.
                support = await _indexes.search_support(collection)
                # Only where search exists: hybrid needs both search tiers, so without mongot the version
                # changes nothing. A plain `mongo:8` server, for example, has no search at all.
                fusion = await _indexes.rank_fusion_support(collection) if support.available else False
                capabilities = _client.Capabilities(
                    search=support.available, no_search_code=support.code, rank_fusion=fusion
                )
                connection.capabilities = capabilities
                if not capabilities.search:
                    logger.warning(
                        _NO_SEARCH_SETUP,
                        self._target,
                        capabilities.no_search_code,
                        extra=_telemetry.event_extra("no_search_setup", namespace=self._target),
                    )

            # Ordinary indexes always, search or not: they include the unique index behind `add`'s
            # deduplication.
            await _indexes.ensure(
                collection,
                ordinary=_indexes.memory_specs(),
                create=self._ensure_indexes,
                timeout=self._index_ready_timeout,
            )

            self._rank_fusion = capabilities.rank_fusion
            self._tiers = (
                await self._ensure_search_tiers(
                    collection, rank_fusion=capabilities.rank_fusion, timeout=self._index_ready_timeout
                )
                if capabilities.search
                else ()
            )
            if self._missing_a_tier(self._tiers):
                self._tier_retry_at = time.monotonic() + self._tier_retry_delay

            # The collation check above cannot verify a collection that does not exist yet. Creating the
            # indexes created it, so check again now: this block never runs again for this store, and this
            # still fails before `add` has written anything.
            if not verified and await _indexes.require_binary_comparison(collection):
                connection.binary_keys.add(target)

            # The tiers that came up, comma-separated and empty without search, so a dashboard can tell a
            # degraded `auto` store from a healthy one.
            _telemetry.annotate("strands_mongodb.search.tiers", ",".join(self._tiers))
            self._initialized = True

    async def _retry_search_tiers(self) -> None:
        """Try the tiers ``initialize()`` skipped again, reading build status without waiting.

        A retry that fails, or still finds a tier missing, keeps the tiers the store has and doubles the
        delay before the next one, up to :data:`TIER_RETRY_MAX_SECONDS`.
        """
        try:
            tiers = await self._ensure_search_tiers(self._collection(), rank_fusion=self._rank_fusion, timeout=0)
        except StorageError:
            tiers = None
        if tiers is not None:
            self._tiers = tiers
        if tiers is None or self._missing_a_tier(tiers):
            self._tier_retry_delay = min(self._tier_retry_delay * 2, TIER_RETRY_MAX_SECONDS)
            self._tier_retry_at = time.monotonic() + self._tier_retry_delay
        else:
            self._tier_retry_at = None
        _telemetry.annotate("strands_mongodb.search.tiers", ",".join(self._tiers))

    def _missing_a_tier(self, tiers: tuple[ResolvedSearchType, ...]) -> bool:
        """Whether ``auto`` skipped a configured tier, which is the only case a retry is for."""
        return self._search_type == "auto" and any(tier not in tiers for tier, _ in self._specs)

    async def _ensure_search_tiers(
        self, collection: AsyncCollection[dict[str, Any]], *, rank_fusion: bool = False, timeout: float
    ) -> tuple[ResolvedSearchType, ...]:
        """Create each configured search index and return the tiers that came up.

        One ``ensure`` call per tier, so a failing vector index does not stop the text index from being
        created. On a deployment with no registered embedding model (such as an Atlas CLI local one) the
        vector index fails with code 8, and the store can still answer lexically.

        Under ``auto`` a tier that fails is logged with the server's reason and skipped. Under a named
        ``search_type`` the failure is raised, because answering from another tier would change what
        matches.

        Args:
            collection: The collection to build indexes on.
            rank_fusion: Whether the deployment can run ``$rankFusion``.
            timeout: Seconds to wait for a build. ``0`` on a retry, which reads status and never waits.

        Returns:
            The tiers that can answer, in preference order. Hybrid first where it is on offer, then
            vector, then text.

        Raises:
            StorageError: If no tier could be prepared, if a named ``search_type``'s own tier failed, or if
                ``search_type`` is ``"hybrid"`` and the deployment cannot run it.
        """
        available: list[ResolvedSearchType] = []
        refused: StorageError | None = None
        for tier, spec in self._specs:
            try:
                await _indexes.ensure(
                    collection,
                    search=(spec,),
                    create=self._ensure_indexes,
                    timeout=timeout,
                )
            except StorageError as error:
                if self._search_type != "auto":
                    raise
                logger.warning(
                    _TIER_UNAVAILABLE,
                    tier,
                    self._target,
                    error,
                    extra=_telemetry.event_extra("tier_unavailable", namespace=self._target, tier=tier),
                )
                refused = error
                continue
            available.append(tier)

        if not available:
            # Chained to the last failure so the reason reaches callers who do not log at `WARNING`.
            raise StorageError(_NO_TIER_PREPARED % self._target) from refused

        if self._hybrid_is_available(available, rank_fusion=rank_fusion):
            # Under `auto`, hybrid goes ahead of the other two, which stay as fallbacks for queries that set
            # only one key. A store that asked for `hybrid` gets only that tier, like any named type.
            if self._search_type == "hybrid":
                return ("hybrid",)
            available.insert(0, "hybrid")
        elif self._search_type == "hybrid":
            raise StorageError(_NO_FUSION % self._target)

        # Only under `auto`: a store configured for `text` asked for exactly this.
        if self._search_type == "auto" and available[0] not in ("hybrid", "vector") and not self._warned_below_vector:
            self._warned_below_vector = True
            logger.warning(
                _BELOW_VECTOR,
                self._target,
                available[0],
                extra=_telemetry.event_extra("below_vector", namespace=self._target, tier=available[0]),
            )
        return tuple(available)

    def _hybrid_is_available(self, available: list[ResolvedSearchType], *, rank_fusion: bool) -> bool:
        """Decide whether the hybrid tier is on offer, and log when a score bound declined it.

        Hybrid needs ``$rankFusion``, both single tiers, and, under ``auto``, no score bound. The bound
        rule exists because a bound applies to whatever score the tier reports. Cosine scores land in
        roughly 0 to 1, while fused scores sum ``weight / (rank + 60)`` terms and top out near 0.033, so a
        ``min_score=0.7`` meant for the vector tier would silently empty every hybrid result. ``auto``
        keeps the vector tier instead and logs why.

        Nothing is logged when the deployment could not run hybrid anyway.

        Args:
            available: The tiers that came up, in preference order.
            rank_fusion: Whether the deployment can run ``$rankFusion``.

        Returns:
            Whether to put ``hybrid`` at the head of the list.
        """
        if not rank_fusion or not {"vector", "text"} <= set(available):
            return False

        bound = "min_score" if self.min_score is not None else "max_score" if self.max_score is not None else None
        if bound is None:
            return True
        # An explicit `hybrid` applies the bound on the fused scale, as documented, so it needs no warning.
        if self._search_type == "hybrid":
            return True
        logger.warning(
            _BOUND_DECLINED_FUSION,
            self._target,
            bound,
            getattr(self, bound),
            extra=_telemetry.event_extra("bound_declined_fusion", namespace=self._target, tier="hybrid"),
        )
        return False

    @_telemetry.traced("memory", "add")
    @_reporting("adding a memory to")
    async def add(self, content: str, metadata: Metadata | None = None) -> MongoDBAddResult:
        """Store one memory in this store's scope, at most once.

        Writes are deduplicated on scope, store name and a SHA-256 hash of ``content`` alone, so storing
        the same text again keeps one document. The SDK retries failed extraction writes, so the same
        fact can arrive more than once. A duplicate keeps its original ``created_at`` and expiry. With
        ``ttl_seconds`` set, a new document expires that many seconds after its ``created_at``.

        ``metadata`` is stored nested under the document's ``metadata`` field and returned by ``search``.
        It cannot set ``scope`` or any other top-level field.

        Args:
            content: The memory to store. Stored verbatim, and never included in an error or a log line.
            metadata: Optional caller metadata.

        Returns:
            The document id and the insert/duplicate counts. A duplicate reports the existing document's id
            with ``inserted=0`` and ``duplicates=1``.

        Raises:
            ValueError: If this store is not writable. Raised before any network call.
            StorageError: If :meth:`initialize` fails, or if a driver call fails.
        """
        # Before `initialize()`, so a read-only store rejects a write without connecting or creating an index.
        if not self.writable:
            raise ValueError(_NOT_WRITABLE % (self.name, "add()"))

        await self.initialize()

        # These fields appear only in the upsert filter. The server copies a filter's equality fields onto
        # the inserted document, and naming them in the update as well would be a conflicting update. They
        # are also the key of the unique index `strands_scope_store_content_hash_unique`.
        selector: dict[str, Any] = {
            "scope": self.scope,
            "store_name": self.name,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        # `$setOnInsert`, not `$set`: a duplicate keeps its first-seen timestamp, so
        # `strands_scope_store_created_at` sorts by recency rather than by last extraction attempt.
        created_at = datetime.datetime.now(datetime.timezone.utc)
        insert: dict[str, Any] = {
            _indexes.CONTENT_FIELD: content,
            "created_at": created_at,
        }
        # Nested, never merged into the root, so a caller's `{"scope": ...}` cannot write into another scope.
        if metadata:
            insert["metadata"] = dict(metadata)
        # Derived from `created_at`, not a second clock read, so the two timestamps are exactly
        # `ttl_seconds` apart. In `$setOnInsert` too, so a duplicate does not extend the expiry.
        if self.ttl_seconds is not None:
            insert[_indexes.EXPIRES_FIELD] = created_at + datetime.timedelta(seconds=self.ttl_seconds)

        collection = self._collection()
        try:
            result = await collection.update_one(selector, {"$setOnInsert": insert}, upsert=True)
        except DuplicateKeyError:
            # Two concurrent upserts both matched nothing and inserted, and this one lost. The memory is
            # stored, so report the winner's document rather than an error.
            return await self._existing(collection, selector)

        if result.upserted_id is None:
            return await self._existing(collection, selector)
        _telemetry.annotate("strands_mongodb.duplicate", False)
        return MongoDBAddResult(ids=[str(result.upserted_id)], inserted=1, duplicates=0)

    async def _existing(
        self, collection: AsyncCollection[dict[str, Any]], selector: Mapping[str, Any]
    ) -> MongoDBAddResult:
        """Report the document a duplicate write matched.

        Projects only ``_id``, because the caller already has the content.

        Args:
            collection: The collection the write went to.
            selector: The upsert filter, which is also the unique index's key.

        Returns:
            One id and ``duplicates=1``, or no ids if the document was deleted between the upsert and this
            read.
        """
        document = await collection.find_one(dict(selector), {"_id": 1})
        ids = [] if document is None else [str(document["_id"])]
        _telemetry.annotate("strands_mongodb.duplicate", True)
        return MongoDBAddResult(ids=ids, inserted=0, duplicates=1)

    @_telemetry.traced("memory", "search")
    @_reporting("searching")
    async def search(self, query: str | MongoDBSearchQuery, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Return the memories that best match a query, most relevant first.

        ``query`` is a string, as the protocol defines, or a ``MongoDBSearchQuery`` for a per-query
        ``filter``, ``limit``, ``oversampling_factor`` or precomputed ``vector``. The store answers from the
        first prepared tier, in preference order, that fits the query's keys.

        Results are always limited to this store's scope and name, combined with the configured and
        per-query filters; nothing in the query can change the scope. A configured ``min_score`` or
        ``max_score`` is applied after the query. Input errors are raised before any network call.

        Args:
            query: A natural-language string, or a ``MongoDBSearchQuery`` carrying any of ``text``,
                ``vector``, ``filter``, ``limit`` and ``oversampling_factor``. Never modified.
            options: The SDK's per-call options. Only ``max_search_results`` is read, and a query's
                ``limit`` takes precedence over it.

        Returns:
            Up to the resolved limit of entries, in descending relevance, on every tier including hybrid.
            Each entry's metadata holds what the caller stored plus the reserved keys ``_id``,
            ``_relevance_score``, ``_search_type`` (the tier that answered) and ``_scope``, which win over
            stored keys of the same name.

        Raises:
            ValueError: If the limit is below 1, if the query sets neither ``text`` nor ``vector`` (or
                both, where the hybrid tier cannot answer), if the filter names an undeclared path, an
                operator key or a regex value, if ``oversampling_factor`` is below 1 or ``limit`` times it
                exceeds 10000 on the vector or hybrid tier, or if no prepared tier fits the query's shape.
            StorageError: If the deployment cannot search at all, if no tier could be prepared, if
                :meth:`initialize` fails, or if a driver call fails.
        """
        # First, so a search that fails early still records what was requested.
        _telemetry.annotate("strands_mongodb.search.requested", self._search_type)
        parts: MongoDBSearchQuery = {"text": query} if isinstance(query, str) else query
        query_text = parts.get("text")
        query_vector = parts.get("vector")
        caller_filter = parts.get("filter")

        limit = _resolve_limit(
            query=parts.get("limit"),
            options=options.get("max_search_results") if options else None,
            instance=self.max_search_results,
        )
        # Only the mode-independent check. The rest runs in `_pipeline.vector_stages`: on a manual-vector
        # store a plain string is wrong for the vector tier but right for the text tier `auto` may pick.
        _pipeline.check_query_keys(
            query_text=query_text, query_vector=query_vector, fusion_possible=self._fusion_possible
        )
        if caller_filter:
            _filter.validate(caller_filter, declared_fields=self._declared_fields)
        query_filter = _filter.compose(self.filter, caller_filter, scope=self.scope, store_name=self.name)

        await self.initialize()
        tier = self._pick_tier(query_text=query_text, query_vector=query_vector)
        _telemetry.annotate("strands_mongodb.search.tier", tier)
        if tier == "hybrid":
            # `_pick_tier` returns "hybrid" only when both keys its two branches need are present.
            assert query_text is not None
            stages = _pipeline.hybrid_stages(
                mode=self._mode,
                vector_index_name=self._index_names["vector"],
                text_index_name=self._index_names["text"],
                query_text=query_text,
                limit=limit,
                query_vector=query_vector,
                query_filter=query_filter,
                oversampling_factor=parts.get("oversampling_factor", DEFAULT_OVERSAMPLING_FACTOR),
            )
        elif tier == "vector":
            # Both keys pass through: `_pick_tier` chose this tier because the mode's own key is present,
            # and `vector_stages` checks the mode itself.
            stages = _pipeline.vector_stages(
                mode=self._mode,
                index_name=self._index_names["vector"],
                limit=limit,
                query_text=query_text,
                query_vector=query_vector,
                query_filter=query_filter,
                oversampling_factor=parts.get("oversampling_factor", DEFAULT_OVERSAMPLING_FACTOR),
            )
        else:
            # `_pick_tier` returns "text" only when `query_text` is set.
            assert query_text is not None
            stages = _pipeline.text_stages(
                query_text=query_text,
                index_name=self._index_names["text"],
                limit=limit,
                query_filter=query_filter,
            )

        scored: list[tuple[float, MemoryEntry]] = []
        cursor = await self._collection().aggregate(stages)
        async for document in cursor:
            # `or 0.0`, not a `.get` default: a stage that projected nothing yields `None`, which would
            # break the bound comparison.
            score = float(document.get(_pipeline.SCORE_FIELD) or 0.0)
            # `_scope` comes from this store, never from the document, so a document cannot report a
            # scope other than the one it was read from.
            entry = MemoryEntry(
                content=str(document.get(_indexes.CONTENT_FIELD, "")),
                store_name=self.name,
                metadata=_merge_metadata(
                    document.get("metadata"),
                    {
                        "_id": str(document["_id"]),
                        "_relevance_score": score,
                        "_search_type": tier,
                        "_scope": self.scope,
                    },
                    store_name=self.name,
                    namespace=self._target,
                ),
            )
            scored.append((score, entry))

        results = self._within_bounds(scored)
        _telemetry.annotate("strands_mongodb.result_count", len(results))
        return results

    def _pick_tier(self, *, query_text: str | None, query_vector: list[float] | None) -> ResolvedSearchType:
        """Choose the first prepared tier, in preference order, that fits the query's keys.

        The choice follows the query's shape, not only ``search_type``, which is why each result reports
        the tier that actually ran.

        Args:
            query_text: What the caller put in ``text``, if anything.
            query_vector: What the caller put in ``vector``, if anything.

        Returns:
            The tier to build stages for.

        Raises:
            StorageError: If no tier is prepared. The message differs between a deployment without Atlas
                Search and one where index creation failed, because the remedies differ.
            ValueError: If the query sets both ``text`` and ``vector`` and the hybrid tier is not
                prepared, or if tiers exist but none fits this query's shape. The message names the key
                to add.
        """
        if query_text is not None and query_vector is not None and "hybrid" not in self._tiers:
            # Raised rather than answered from one tier, which would silently ignore one of the two keys.
            raise ValueError(_NO_FUSION_FOR_QUERY % (self._target, ", ".join(self._tiers) or "none"))

        if not self._tiers:
            capabilities = self._connection.capabilities
            if capabilities is None or not capabilities.search:
                # A phrase rather than a bare code, because with no cached probe there is no code to name.
                code = capabilities.no_search_code if capabilities is not None else None
                evidence = f"server error code {code}" if code is not None else "no probe result cached"
                raise StorageError(_NO_SEARCH_QUERY % (self._target, evidence))
            raise StorageError(_NO_TIER_PREPARED % self._target)

        for tier in self._tiers:
            # Hybrid needs `text` for `$search` and the vector path's own key, which with AutoEmbeddings is
            # `text` again, so there a plain string fits the hybrid tier.
            if tier == "hybrid" and query_text is not None and self._vector_key(query_text, query_vector) is not None:
                return "hybrid"
            if tier == "text" and query_text is not None:
                return "text"
            if tier == "vector" and self._vector_key(query_text, query_vector) is not None:
                return "vector"

        needed, because = (
            ("'text'", "the deployment embeds it on the Atlas AutoEmbeddings path")
            if self._mode.auto_embed
            else ("'vector'", "'embedding_dimensions' and 'similarity' select the manual-vector path")
        )
        raise ValueError(_NO_TIER_FOR_QUERY % (self._target, ", ".join(self._tiers), needed, because))

    def _vector_key(self, query_text: str | None, query_vector: list[float] | None) -> Any:
        """Return whichever query key the vector tier reads on this store's path, or ``None``.

        AutoEmbeddings embeds ``text`` on the deployment, and the manual path searches an embedding the
        caller computed. The vector and hybrid tiers share this so the rule has one definition.

        Args:
            query_text: What the caller put in ``text``, if anything.
            query_vector: What the caller put in ``vector``, if anything.

        Returns:
            The key this store's vector path needs, or ``None`` if the query has not supplied it.
        """
        return query_text if self._mode.auto_embed else query_vector

    def _within_bounds(self, scored: list[tuple[float, MemoryEntry]]) -> list[MemoryEntry]:
        """Apply the configured score bound, in Python, after the query.

        Client-side rather than a ``$match`` on ``$meta``, because the tiers' scores are on different
        scales and a server-side bound would need a different expression per tier. BM25's ``searchScore``
        is unbounded and depends on the corpus; cosine ``vectorSearchScore`` is in ``[0, 1]``.

        Args:
            scored: Every mapped entry with its score, in the order the search returned them.

        Returns:
            The entries that passed, in the same order. Every entry when no bound is set.
        """
        minimum = self.min_score
        maximum = self.max_score
        if minimum is not None:
            name = "min_score"
            bound = minimum
            kept = [entry for score, entry in scored if score >= minimum]
        elif maximum is not None:
            name = "max_score"
            bound = maximum
            kept = [entry for score, entry in scored if score <= maximum]
        else:
            return [entry for _, entry in scored]

        # Logged only when the bound dropped something, so polling a corpus mongot has not indexed yet
        # does not fill the log.
        if len(kept) != len(scored):
            logger.debug(
                _SCORE_BOUND_APPLIED,
                name,
                bound,
                self._target,
                len(kept),
                len(scored),
                extra=_telemetry.event_extra("score_bound_applied", namespace=self._target),
            )
        return kept

    @_telemetry.traced("memory", "delete")
    @_reporting("deleting a memory from")
    async def delete(self, memory_id: str) -> int:
        """Delete one memory, named by the ``_id`` a ``search`` result reported.

        Not part of the ``MemoryStore`` protocol. The delete matches on the same scope, store name and
        configured ``filter`` as ``search``, plus the id, so a store can delete only what it can read.

        Three cases return ``0`` and are deliberately indistinguishable: the memory was already deleted,
        it belongs to another scope, or it belongs to another store in this scope. Telling them apart
        would reveal whether an id exists in a scope the caller cannot read.

        Args:
            memory_id: The ``_id`` string from a search result's metadata.

        Returns:
            ``1`` if a document was deleted, ``0`` if none matched, as counted by the server.

        Raises:
            ValueError: If this store is not writable, or if ``memory_id`` is not a 24-character hex
                ObjectId string. Both are raised before any network call. A document whose ``_id`` is not
                an ``ObjectId`` cannot be deleted this way; this package never writes one.
            StorageError: If :meth:`initialize` fails, or if a driver call fails.
        """
        # Writability first, before the id is parsed or anything connects, so a read-only store says
        # "not writable" rather than complaining about the id.
        if not self.writable:
            raise ValueError(_NOT_WRITABLE % (self.name, "delete()"))
        object_id = _object_id(memory_id)

        await self.initialize()
        predicate = _filter.compose(self.filter, scope=self.scope, store_name=self.name)
        result = await self._collection().delete_one({**predicate, "_id": object_id})
        _telemetry.annotate("strands_mongodb.deleted_count", result.deleted_count)
        return result.deleted_count

    @_telemetry.traced("memory", "delete_all")
    @_reporting("deleting every memory in")
    async def delete_all(self) -> int:
        """Delete every memory this store can read.

        Matches what :meth:`search` reads: this store's scope and name, narrowed by any configured
        ``filter``. Other stores sharing the collection and scope keep their documents, so clearing a
        whole scope takes one call per store.

        Returns:
            How many documents the server deleted. A ``0`` on a scope believed to hold memories usually
            means the configured scope or store name differs from the one that was written to.

        Raises:
            ValueError: If this store is not writable. Raised before any network call.
            StorageError: If :meth:`initialize` fails, or if a driver call fails.
        """
        if not self.writable:
            raise ValueError(_NOT_WRITABLE % (self.name, "delete_all()"))

        await self.initialize()
        predicate = _filter.compose(self.filter, scope=self.scope, store_name=self.name)
        result = await self._collection().delete_many(predicate)
        _telemetry.annotate("strands_mongodb.deleted_count", result.deleted_count)
        return result.deleted_count

    def get_tools(self) -> list[AgentTool]:
        """Return the agent tools this store contributes, which is none.

        ``MemoryManager`` extends its tool list with the result, so an empty list adds nothing.

        Returns:
            An empty ``list``.
        """
        return []

    @property
    def _target(self) -> str:
        """The ``database.collection`` namespace, the only subject errors and log messages name.

        Messages never name a memory, because that would put user content into errors, logs and
        tracebacks.

        Returns:
            ``database.collection``.
        """
        return f"{self._database_name}.{self._collection_name}"

    @property
    def _connection(self) -> _client._Connection:
        """The connection serving the running event loop.

        A property, not an attribute captured in ``__init__``, because connections are per event loop and
        a store is often used from more than one: the SDK's synchronous entry points run on a fresh loop
        in a worker thread. Capabilities are cached per connection, so a new loop probes again, but a
        store that has already initialized skips the probe.
        """
        return _client.resolve(self._connection_config)

    def _collection(self) -> AsyncCollection[dict[str, Any]]:
        """Resolve this store's collection from the running event loop's connection.

        Not cached, because a collection captured earlier would hold a client bound to another loop that
        may already be closed.

        Returns:
            The collection, resolved fresh on every call.
        """
        return self._connection.client[self._database_name][self._collection_name]

    async def close(self) -> None:
        """Release the MongoDB connection, closing the client only if this package created it.

        The connection is shared per config, so this affects every store built from the same config.
        Idempotent, and an injected client is never closed. Not part of the ``MemoryStore`` protocol, so
        ``MemoryManager`` never calls it.

        A store that keeps working afterwards opens a new pool, and nothing closes that pool again, so call
        this once, at process shutdown.
        """
        await _client.close(self._connection_config)
