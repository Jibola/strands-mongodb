"""``MongoDBMemoryStore`` -- Atlas Vector Search behind the Strands ``MemoryStore`` protocol.

Requires the ``memory`` extra, whose name does not match this module's::

    pip install 'strands-mongodb[memory]'

Registered with the SDK's ``MemoryManager`` alongside any other stores. The protocol requires
only ``search``; ``add`` and ``add_messages`` are optional and feature-detected by the manager.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

from ._extras import missing_extra

# Imported for their side effect of proving the `memory` extra is installed: without it the
# failure would otherwise surface deep inside a method call as a bare ModuleNotFoundError.
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

from . import _client, _filter, _indexes, _pipeline
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

__all__ = ["DEFAULT_COLLECTION", "UNSCOPED", "MongoDBMemoryStore"]

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

DEFAULT_COLLECTION = "memory"
"""Collection this store uses when the config names none.

Distinct from ``MongoDBStorage``'s ``"storage"`` on purpose. The two surfaces hold unrelated shapes:
opaque bytes under an ``_id`` on one side, ``MemoryEntry`` text with a vector index over it on the
other. One default for both would put session blobs in the collection whose index tries to embed
them.
"""

UNSCOPED: Final[str] = "__unscoped__"
"""Scope stamped and filtered on when the config names none (``REQ-E-048``).

A sentinel rather than ``None`` or ``""``, and rather than dropping the predicate. Dropping it would
make the *unconfigured* store the one that reads every tenant's documents, which is the failure that
looks like a relevance quirk instead of a leak. ``None`` fails the other way and matches nothing,
because no document carries it. ``""`` is falsy, so every later ``if scope:`` would read it as unset.

Double-underscored on both ends so it cannot collide with a scope somebody meant: a real tenant id
that spells this is a tenant id chosen to collide.
"""

_NO_SEARCH_SETUP = (
    "strands_mongodb: %s is on a deployment that does not run Atlas Search, so no search index was "
    "created and no query can be answered here. The server reported error code %s when asked to list "
    "its search indexes. Ordinary indexes and writes are unaffected, so this store can hold memories "
    "and cannot read them back. Point 'connection_string' at MongoDB Atlas, or at a local Atlas "
    "deployment started with 'atlas deployments setup --type local', which runs the same search process."
)
"""Warning ``initialize()`` degrades with (``REQ-E-053a``). Emitted once per connection, not per store.

Names the code and what to connect to, and deliberately names neither a command to run nor an index
status: on this deployment the search-index commands do not exist, so ``REQ-E-061``'s remedy block
would be an afternoon spent on advice that cannot work (``AT-053a``).

The code is interpolated rather than written in, which is the later half of this message. It carried the
literal ``31082`` until then, and three of the four deployments since measured report something else, so
on those it named a code absent from the reader's own log.
"""

_NO_SEARCH_QUERY = (
    "strands_mongodb: cannot search %s because this deployment does not run Atlas Search. Its capability "
    "probe came back with %s, and v0.1.0 ships no tier that answers without a search process, "
    "so there is nothing to fall back to. Writes still work, which is why "
    "initialize() only warned. Point 'connection_string' at MongoDB Atlas, or at a local Atlas "
    "deployment started with 'atlas deployments setup --type local'."
)
"""What ``search`` raises on the same deployment (``REQ-E-053a``, ``AT-053c``).

The split is deliberate: a capability probe may degrade, a query path may not.
Returning zero
results here would be an empty memory that reads as a store with nothing in it.
"""

_TIER_UNAVAILABLE = (
    "strands_mongodb: the %s search tier is not available on %s and 'search_type' is 'auto', so it is "
    "skipped rather than fatal and any remaining tier answers queries instead. The '_search_type' key "
    "on every returned entry reports which one did. The server's own reason: %s"
)
"""Per-tier warning under ``auto`` (``REQ-E-053b``).

The server's message is carried whole rather than paraphrased, because the reason is the one piece of
information that says what to fix: on an Atlas CLI local deployment it is the list of registered
embedding models, which is empty.
"""

_NO_TIER_PREPARED = (
    "strands_mongodb: 'search_type' is 'auto' and no search tier could be prepared on %s, so no query "
    "can be answered. Each tier's failure is logged above with the server's own reason."
)
"""``REQ-E-053b``'s floor: warn per tier, then raise once nothing is left."""

_BELOW_VECTOR = (
    "strands_mongodb: 'search_type' is 'auto' and the vector tier is not available on %s, so queries "
    "resolve to the %s tier. Answers are lexical rather than semantic, which is a difference in what "
    "matches rather than only in ranking."
)
"""``REQ-E-053``'s "warn when it lands below vector", at most once per store."""

_NO_TIER_FOR_QUERY = (
    "strands_mongodb: this query cannot be answered by any search tier prepared on %s, which is: %s. "
    "The text tier needs 'text' on the query. The vector tier needs %s, because %s. Supply that key, "
    "or set 'search_type' to a tier this query fits."
)
"""Raised when tiers exist and none of them fits the query's shape (``REQ-E-074a``)."""

_NOT_WRITABLE = (
    "strands_mongodb: the memory store %r is not writable, so it does not accept %s. Set "
    "'writable': True in its configuration to enable writes. It defaults to False so that a store "
    "built for retrieval does not start accepting model-distilled extraction writes, or deletions, "
    "that nobody asked for."
)
"""``REQ-E-058``, and ``REQ-E-079`` for the two deletes.

Names the key, because a message saying only "not writable" sends a caller to source, and names the
operation, because one hardcoding ``add()`` on a ``delete()`` call sends them to the wrong line.
"""

_BAD_MEMORY_ID = (
    "strands_mongodb: 'memory_id' is %r, which is not a 24-character hex ObjectId. Pass the '_id' a "
    "search result carries in its metadata, which is where a deletable id comes from."
)
"""``REQ-E-076``'s rejection, naming the argument rather than the document it failed to name.

The id itself is rendered, and it is the one subject :func:`_reporting` would refuse: it is the caller's
own argument rather than stored content, and a message repeating a mistyped id back is how the caller
sees the typo.
"""

_LIMIT_TOO_LOW = (
    "strands_mongodb: %r is %d, and a search limit must be at least 1. A limit below 1 cannot return "
    "anything, so it is rejected here rather than sent to a server that would reject it as code "
    "7912700."
)
"""``REQ-E-041``'s floor, applied to any resolved or supplied value and naming which one was wrong."""

_BOTH_SCORE_BOUNDS = (
    "strands_mongodb: 'min_score' is %r and 'max_score' is %r, and the two are mutually exclusive: a "
    "floor and a ceiling on one similarity score describe a band, and this package applies one bound "
    "rather than a range. Keep whichever one you meant and remove the other."
)
"""``REQ-E-059``'s pair, rejected in ``__init__`` rather than deferred to a query (``AT-059a``)."""

_RESERVED_KEY_COLLISION = (
    "strands_mongodb: metadata key(s) %s on a document in the memory store %r collide with the keys "
    "this package sets on every result, so the stored value(s) were dropped from the returned entry "
    "and this package's own value(s) kept. Rename them in whatever wrote them; a stored '_scope' that "
    "won would make a document report a scope it was not stored in."
)
"""``REQ-E-057``. One record per merge, listing every colliding key, and naming the store.

A dropped value with no log line is indistinguishable from a value that never existed, and one process
holds several stores over one collection, so the store name is what locates the writer.
"""

_SCORE_BOUND_APPLIED = "strands_mongodb: %s %r on %s kept %d of %d results (applied after the query)."
"""``REQ-E-059``'s effect, at ``DEBUG`` with both counts (``AT-059b``).

Both counts, because a message saying only that results were filtered tells the reader something
happened without telling them how much, and on a bound set slightly too high that is the difference
between a tuning problem and an empty memory.
"""


def _resolve_limit(*, query: int | None, options: int | None, instance: int | None) -> int:
    """Resolve the result ceiling from the four places it can come from (``REQ-E-041``).

    ``query["limit"]`` sits above ``options`` because a ``MongoDBSearchQuery`` can only arrive on a
    direct call: ``MemoryStore.search`` is declared ``query: str`` and both SDK call sites pass one, while
    ``MemoryManager.search`` always passes ``options`` with ``max_search_results`` already resolved.
    Below ``options`` the key would be unreachable rather than lower priority.

    Every supplied value is checked, not only the one that wins, which is ``REQ-E-041``'s wording rather
    than an extra. A caller who passes ``limit=5`` alongside ``max_search_results=0`` wrote one of those
    two by mistake, and resolving to 5 silently keeps the mistake in their code.

    Args:
        query: ``limit`` from a ``MongoDBSearchQuery``, if one was passed.
        options: ``max_search_results`` from this call's ``SearchOptions``.
        instance: The store's own ``max_search_results``.

    Returns:
        The first value that was supplied, or ``DEFAULT_MAX_SEARCH_RESULTS``.

    Raises:
        ValueError: If any supplied value is below 1, naming the key it came from. ``limit`` and
            ``max_search_results`` are different keys in different dicts, and a message naming the wrong
            one sends the caller to the wrong line.
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
    """Parse the id ``search`` reported back into the type the ``_id`` field holds (``REQ-E-076``).

    Strict, with no fallback to matching the raw string when the parse fails. A delete that also tried
    ``{"_id": memory_id}`` would return ``0`` for a typo, ``0`` for an already deleted memory and ``0``
    for a caller who passed the wrong field entirely, and one answer for three causes is the signal a
    wiring bug needs to be found by.

    The type check is not the format check written twice. Measured on pymongo 4.17.0, ``ObjectId(None)``
    returns a freshly generated id rather than failing, so a ``None`` reaching the driver composes into a
    well-formed predicate that matches a document which has never existed, and the call reports a
    truthful, useless ``0``. ``ObjectId(12)`` raises ``TypeError``, which is neither the driver error
    :func:`_reporting` wraps nor the ``ValueError`` a caller mistake should arrive as.

    Args:
        memory_id: The ``_id`` string a ``search`` result carries in its metadata. Annotated ``str``
            because that is what the read path emits, and checked anyway because an annotation is not
            enforcement.

    Returns:
        The parsed id.

    Raises:
        ValueError: If it is not a string, or not a 24-character hex ObjectId. ``InvalidId`` is a
            ``BSONError`` and not a ``ValueError``, so allowing it through would hand a caller mistake to
            them in the driver's exception hierarchy.
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
) -> dict[str, Any]:
    """Merge caller metadata under this package's reserved keys (``REQ-E-040``, ``REQ-E-057``).

    A function rather than an inline ``{**stored, **reserved}`` so the unit tier can assert the merge
    without a deployment, which is the whole of ``AT-057``. The collision it defends against is not
    hypothetical for one key in particular: a stored ``_scope`` that won would make a document report a
    scope it was not stored in, and that reads as a leak rather than as a metadata quirk, which is the
    failure this function exists to prevent.

    One log record per merge, listing every colliding key. A record per key would turn a document with
    four bad keys into four lines saying the same thing.

    Args:
        caller: ``metadata`` as the document carries it, or ``None`` when it carries none.
        reserved: This package's four keys and their values for this result.
        store_name: Named in the log record, because one process holds several stores over one
            collection and the store is what locates whatever wrote the document.

    Returns:
        A new dict. Neither argument is modified.
    """
    merged: dict[str, Any] = dict(caller or {})
    collisions = sorted(key for key in reserved if key in merged)
    if collisions:
        logger.warning(_RESERVED_KEY_COLLISION, ", ".join(repr(key) for key in collisions), store_name)
    merged.update(reserved)
    return merged


def _reporting(
    verb: str,
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """Turn any driver error out of the wrapped method into ``StorageError`` (``REQ-E-036``).

    ``MongoDBStorage``'s decorator of this name with one deliberate difference: it renders the key it was
    operating on, and this one renders nothing but the namespace. A memory store's subject *is* the
    memory, so a message naming it would put user content into an error string, a log line and every
    traceback that reaches a ticket. The namespace is enough to locate the store, and ``REQ-E-014``'s
    reasoning about connection strings applies with more force to what a user told an agent.

    ``Coroutine`` rather than ``Awaitable``, for the same reason as in ``storage.py``: the protocol
    declares these with ``async def``, and widening them would leave the class satisfying ``MemoryStore``
    at runtime and failing it under mypy.

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
                # Bound through a local rather than indexed off `args` directly, which mypy rejects for
                # a `ParamSpec`.
                positional: tuple[Any, ...] = args
                store: MongoDBMemoryStore = positional[0]
                raise StorageError(f"strands_mongodb: {verb} {store._target} failed: {error}") from error

        return reporting

    return decorate


class MongoDBMemoryStore:
    """Atlas Vector Search implementation of ``strands.memory.MemoryStore``.

    Satisfies the Protocol structurally and does not inherit it, unlike ``FileMemoryStore`` and
    ``BedrockKnowledgeBaseStore``, which both do. Inheriting a ``Protocol`` makes every member the
    subclass leaves out abstract, and mypy then refuses to instantiate the subclass at all: checking
    ``FileMemoryStore(name="x")`` reports *Cannot instantiate abstract class "FileMemoryStore" with
    abstract attributes "add_messages", "get_tools" and "initialize"*. Following the convention here
    would hand every type-checked consumer an error on the constructor call with no way to fix it.

    ``add_messages`` is absent, and must stay absent rather than become a stub that raises:
    ``AddMessagesContext`` carries no run-unique identifier, so no dedup token that survives a retry
    can be built from it, while extraction writes are at-least-once. The manager feature-detects it
    with ``_has_method`` and falls back to its own ``ModelExtractor`` feeding ``add``, which dedupes.

    Example:
        ```python
        connection: MongoDBConfig = {"database": "agents", "connection_string": "mongodb+srv://..."}
        store = MongoDBMemoryStore(name="preferences", config=connection, scope="user-abc")
        ```

    Every read is scoped by this package rather than by the caller (``REQ-E-046``), and an unconfigured
    scope resolves to :data:`UNSCOPED` rather than to no predicate at all. Isolation here is
    application-enforced: MongoDB applies no server-side row filter, so a document lands in the right
    scope only because ``add`` put it there and comes back only because ``search`` asked for it.
    """

    def __init__(self, **config: Unpack[MongoDBMemoryStoreConfig]) -> None:
        """Resolve the shared connection and record this store's identity, placement and bounds.

        Every check runs before ``_client.resolve``, so a config that fails validation leaves no
        registered connection pool behind that nothing owns and no ``close()`` can reach.

        Args:
            **config: See ``MongoDBMemoryStoreConfig``. The connection half is nested under
                ``config``, following ``BedrockKnowledgeBaseStore``, so one ``MongoDBConfig`` can be
                shared with a ``MongoDBStorage`` and with every other store in the process.

        Raises:
            KeyError: If ``name`` or the nested ``config`` is absent, or if that config names no
                ``database``. All three are ``Required`` keys, so mypy rejects a literal that omits
                one.
            ValueError: If the nested config supplies neither ``client`` nor ``connection_string``
                (``REQ-E-015``); if ``max_search_results`` is below 1 (``REQ-E-041``); if both
                ``min_score`` and ``max_score`` are set (``REQ-E-059``); if the configured ``filter``
                names an operator key or an undeclared path (``REQ-E-049``, ``REQ-E-050``); or if the
                vector configuration selects two paths at once (``REQ-E-067``).
        """
        connection = config["config"]
        self.name = config["name"]
        self.description: str | None = config.get("description")
        self.max_search_results: int | None = config.get("max_search_results")
        # `False` by default, unlike `FileMemoryStore`. `writable` is what gates automatic extraction, so
        # the cost of the wrong default is a retrieval store that starts accepting model-distilled writes
        # (`REQ-E-058`).
        self.writable: bool = config.get("writable", False)
        self.extraction: ExtractionConfig | bool | None = config.get("extraction")
        self.scope: str = config.get("scope") or UNSCOPED
        self.filter: dict[str, Any] | None = config.get("filter")
        self.min_score: float | None = config.get("min_score")
        self.max_score: float | None = config.get("max_score")

        # Rejected here rather than on the first query. A store whose ceiling is zero can never answer
        # anything, and `BedrockKnowledgeBaseStore` raises in the same place for the same value.
        _resolve_limit(query=None, options=None, instance=self.max_search_results)
        if self.min_score is not None and self.max_score is not None:
            raise ValueError(_BOTH_SCORE_BOUNDS % (self.min_score, self.max_score))

        self._database_name = connection["database"]
        self._collection_name = config.get("collection", DEFAULT_COLLECTION)
        self._ensure_indexes: bool = connection.get("ensure_indexes", True)
        self._search_type: SearchType = connection.get("search_type") or "auto"
        # The index definition is the single authority for what is filterable, so both the
        # configured filter and every per-query one are checked against the same list.
        self._declared_fields = _indexes.declared_filter_fields(connection.get("filter_fields") or ())
        if self.filter:
            _filter.validate(self.filter, declared_fields=self._declared_fields)

        # Resolved once, in the constructor, because both raise `ValueError` on a contradictory config and
        # a constructor is where a caller can still fix the dict. Deferring them to `initialize()` would
        # surface a config mistake from inside whichever `add` or `search` happened to run first.
        self._mode: VectorMode = _indexes.vector_mode(connection)
        vector_spec, text_spec = _indexes.memory_search_specs(connection)
        specs: tuple[tuple[ResolvedSearchType, SearchIndexSpec], ...] = (
            ("vector", vector_spec),
            ("text", text_spec),
        )
        # Vector first, which is the preference order `auto` resolves down from. A named `search_type`
        # narrows the list to one, so a store asking for `text` never creates a vector index it will not
        # query, and one asking for `vector` fails rather than quietly answering lexically.
        self._specs = (
            specs if self._search_type == "auto" else tuple(pair for pair in specs if pair[0] == self._search_type)
        )
        self._index_names = {tier: spec.name for tier, spec in self._specs}

        self._config = config
        self._connection_config = connection
        self._initialized = False
        # Resolved by `initialize()`, in preference order, holding only the tiers that can actually
        # answer. Empty means no tier could be prepared, which `search` reports and `add` does not care
        # about.
        self._tiers: tuple[ResolvedSearchType, ...] = ()
        self._warned_below_vector = False
        # Resolved and discarded, exactly as in `MongoDBStorage.__init__`: the connection an operation
        # uses comes from the `_connection` property, per running event loop (`REQ-E-017b`). This call is
        # here so a config supplying neither `client` nor `connection_string` raises at construction
        # (`REQ-E-015`) instead of at the first `add`.
        _client.resolve(connection)

    @_reporting("initializing")
    async def initialize(self) -> None:
        """Probe the deployment and create this store's indexes, once (``REQ-E-052``, ``REQ-E-065``).

        Idempotent and safe to call concurrently. ``search`` and ``add`` both call it rather than
        requiring the caller to, because ``MemoryManager`` does not call it and a store that only works
        after an undocumented setup step is a store that mostly does not work (``AT-045b``).

        The fast path is the unlocked ``_initialized`` check, which is what keeps this cheap enough to
        call on every operation. The lock is taken only on the first call, and the flag is re-read inside
        it, so N stores racing on one connection do the work once rather than N times.

        The capability probe is cached on the *connection*, not on the store: Atlas Search is a property
        of the deployment, so ten stores on one database probe once between them (``AT-065b``). The
        warning lives inside that branch for the same reason. A deployment with no search is worth one
        line in the log, not ten.

        A deployment that cannot search is a warning here and an error at query time (``REQ-E-066``).
        Writes work on it, so failing construction or setup would take away the half that functions.

        Raises:
            StorageError: If a driver call fails, or if ``search_type`` names a tier the deployment
                cannot prepare. Under ``auto`` a tier that cannot be prepared is skipped instead, and
                only an empty result raises.
        """
        if self._initialized:
            return

        # Bound once, not re-read through the property at each step. `_connection` resolves per running
        # event loop and this whole block runs on one, so the three reads would return the same object
        # anyway; binding says so, and keeps the lock, the probe and the cache write on one connection
        # even if that ever stops being true.
        connection = self._connection
        async with connection.get_lock():
            if self._initialized:
                return

            collection = self._collection()
            capabilities = connection.capabilities
            if capabilities is None:
                # Through the module rather than through a direct import, so this stays one observable
                # call per connection that `AT-065b`'s counting spy can see.
                support = await _indexes.search_support(collection)
                capabilities = _client.Capabilities(search=support.available, no_search_code=support.code)
                connection.capabilities = capabilities
                if not capabilities.search:
                    logger.warning(_NO_SEARCH_SETUP, self._target, capabilities.no_search_code)

            # Ordinary indexes first, and unconditionally. The uniqueness index behind `add`'s
            # deduplication is one of them, and it is needed whether or not this deployment can search.
            await _indexes.ensure(collection, ordinary=_indexes.memory_specs(), create=self._ensure_indexes)

            self._tiers = await self._ensure_search_tiers(collection) if capabilities.search else ()
            self._initialized = True

    async def _ensure_search_tiers(self, collection: AsyncCollection[dict[str, Any]]) -> tuple[ResolvedSearchType, ...]:
        """Create each configured search index and return the tiers that came up (``REQ-E-067``).

        One ``ensure`` call per tier, deliberately, rather than one call carrying both specs. The vector
        index is created first, and on a deployment with no registered embedding model its failure is
        code 8 (measured on Atlas CLI local: *voyage-4-lite not registered yet*). In a single
        combined call that failure would stop the text index from ever being created, so a store that
        could have answered lexically would answer nothing.

        Under ``auto`` a tier that cannot be prepared is logged and skipped, and the server's own reason
        travels into the message: it is the only part of the log line that says *why*, and it is the part
        a reader can act on. Under a named ``search_type`` the same failure is fatal, because there is no
        second tier to fall through to and answering from one the caller did not ask for would change
        what matches rather than only how it ranks.

        Args:
            collection: This store's collection, already resolved.

        Returns:
            The tiers that can answer, in preference order. Vector first when both came up.

        Raises:
            StorageError: If no tier could be prepared, or if a named ``search_type``'s own tier failed.
        """
        available: list[ResolvedSearchType] = []
        for tier, spec in self._specs:
            try:
                await _indexes.ensure(collection, search=(spec,), create=self._ensure_indexes)
            except StorageError as error:
                if self._search_type != "auto":
                    raise
                logger.warning(_TIER_UNAVAILABLE, tier, self._target, error)
                continue
            available.append(tier)

        if not available:
            raise StorageError(_NO_TIER_PREPARED % self._target)

        # Gated on `auto` as well as on the order, because the message says `auto` and a store configured
        # `search_type='text'` has exactly one tier by its own request. Warning there would describe the
        # caller's own configuration back to them as a degradation.
        if self._search_type == "auto" and available[0] != "vector" and not self._warned_below_vector:
            self._warned_below_vector = True
            logger.warning(_BELOW_VECTOR, self._target, available[0])
        return tuple(available)

    @_reporting("adding a memory to")
    async def add(self, content: str, metadata: Metadata | None = None) -> MongoDBAddResult:
        """Store one memory, at most once per scope (``REQ-E-042``, ``REQ-E-043``).

        Deduplicated on ``(scope, store_name, content_hash)`` by the unique partial index
        ``strands_scope_store_content_hash_unique``, because extraction writes are at-least-once: the
        SDK re-runs an extraction whose write failed, so the same fact arrives repeatedly by design and
        an ordinary insert would grow the collection once per retry. The growth reads as the agent
        remembering the same thing twice.

        The hash covers content and nothing else. Including ``created_at`` would make every regenerated
        entry a fresh document, which is the same defect wearing a hash.

        ``scope``, ``store_name`` and ``content_hash`` are named in the upsert *filter only*, and that is
        the whole trick rather than a shortcut: the server copies a filter's equality fields onto the
        document it inserts, so the stored shape comes out identical while no field is named in both halves
        of the call, which is where a conflicting-update error would come from.

        Caller metadata is nested under ``metadata`` rather than merged into the document root. A merge
        would make ``{"scope": "tenant-b"}`` a one-line cross-scope write, and the
        nested copy stays readable, so a legitimate key that merely shares a name with ours is not
        silently dropped.

        Args:
            content: The memory to store. Stored verbatim, and never rendered into an error or a log
                line by this module.
            metadata: Arbitrary caller metadata, stored nested and returned by ``search``.

        Returns:
            The resolved document ids and the insert/duplicate split. A duplicate reports the existing
            document's id, so a caller cannot distinguish a lost race from their own second write, and
            ``duplicates`` is reported rather than hidden so a fact count below a write count has a
            signal to read.

        Raises:
            ValueError: If this store is not writable (``REQ-E-058``). Raised before any network call,
                and left as ``ValueError`` rather than wrapped, because it is a configuration mistake
                rather than a storage failure.
            StorageError: If a driver call fails.
        """
        # First, and before `initialize()`. A read-only store must reject a write without having created
        # an index or opened a connection to do it.
        if not self.writable:
            raise ValueError(_NOT_WRITABLE % (self.name, "add()"))

        await self.initialize()

        selector: dict[str, Any] = {
            "scope": self.scope,
            "store_name": self.name,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        # `$setOnInsert`, not `$set`: a duplicate must leave the first-seen timestamp alone, or
        # `strands_scope_store_created_at` sorts by "last extraction attempt" rather than by recency.
        insert: dict[str, Any] = {
            _indexes.CONTENT_FIELD: content,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        }
        if metadata:
            insert["metadata"] = dict(metadata)

        collection = self._collection()
        try:
            result = await collection.update_one(selector, {"$setOnInsert": insert}, upsert=True)
        except DuplicateKeyError:
            # The narrow race `REQ-E-043` names: two upserts match nothing, both insert, and the loser
            # lands here. The winner's document is the right answer, since an error would tell the caller
            # their write was rejected when it was in fact already stored.
            return await self._existing(collection, selector)

        if result.upserted_id is None:
            return await self._existing(collection, selector)
        return MongoDBAddResult(ids=[str(result.upserted_id)], inserted=1, duplicates=0)

    async def _existing(
        self, collection: AsyncCollection[dict[str, Any]], selector: Mapping[str, Any]
    ) -> MongoDBAddResult:
        """Report the document a duplicate write resolved to.

        Projected down to ``_id``, because the content is already in the caller's hand and reading it
        back would pull a memory across the wire for no reason.

        Args:
            collection: The collection the write went to.
            selector: The upsert filter, which is also the unique index's key.

        Returns:
            One id and ``duplicates=1``. An empty ``ids`` is possible only if the document was deleted
            between the upsert and this read, which is reported as a duplicate with nothing to point at
            rather than as an error.
        """
        document = await collection.find_one(dict(selector), {"_id": 1})
        ids = [] if document is None else [str(document["_id"])]
        return MongoDBAddResult(ids=ids, inserted=0, duplicates=1)

    @_reporting("searching")
    async def search(self, query: str | MongoDBSearchQuery, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Answer a query from whichever prepared tier fits it (``REQ-E-040``, ``REQ-E-041``).

        ``query`` is widened past the protocol's ``str``, which is safe because a method parameter is
        contravariant: every caller that could pass a ``str`` still can. Both SDK call sites do pass a
        ``str``, so the string form is the production path and the ``MongoDBSearchQuery`` form is the
        extension a direct caller reaches for when they need a per-query filter, limit or embedding.

        Everything the caller can get wrong is checked before ``initialize()`` runs, so a bad limit or a
        bad filter is a ``ValueError`` rather than a connection timeout against a deployment that was
        never going to be asked anything. That ordering is asserted rather than incidental: two unit
        tests point a store at a closed port and require ``ValueError``.

        Scope is applied by :func:`_filter.compose`, not by the caller, and the returned ``_scope`` key
        is read off this store rather than off the document. A document that could name its own scope
        could name one it was not stored in.

        Args:
            query: A natural-language string, or a ``MongoDBSearchQuery`` carrying any of ``text``,
                ``vector``, ``filter``, ``limit`` and ``oversampling_factor``. Never modified.
            options: The SDK's per-call options. Only ``max_search_results`` is read, and it sits below
                a query-level ``limit`` because a ``MongoDBSearchQuery`` can only arrive on a direct
                call while ``MemoryManager.search`` always supplies ``options``.

        Returns:
            Entries in descending relevance, each carrying the four reserved metadata keys beside
            whatever the caller stored. Ordering comes from the search stage; the score bound only
            drops.

        Raises:
            ValueError: If the limit is below 1, if the query sets both or neither of ``text`` and
                ``vector``, if the filter names an undeclared path or an operator key, or if no prepared
                tier fits the query's shape.
            StorageError: If the deployment cannot search at all, if no tier could be prepared, or if a
                driver call fails.
        """
        parts: MongoDBSearchQuery = {"text": query} if isinstance(query, str) else query
        query_text = parts.get("text")
        query_vector = parts.get("vector")
        caller_filter = parts.get("filter")

        limit = _resolve_limit(
            query=parts.get("limit"),
            options=options.get("max_search_results") if options else None,
            instance=self.max_search_results,
        )
        # The mode-independent half of the shape check only. The other half lives inside
        # `_pipeline.vector_stages` and cannot run here: on a manual-vector store a plain string is wrong
        # for the vector tier and exactly right for the text one, so rejecting it now would reject the
        # query `auto` is about to answer.
        _pipeline.check_query_keys(query_text=query_text, query_vector=query_vector)
        if caller_filter:
            _filter.validate(caller_filter, declared_fields=self._declared_fields)
        query_filter = _filter.compose(self.filter, caller_filter, scope=self.scope, store_name=self.name)

        await self.initialize()
        tier = self._pick_tier(query_text=query_text, query_vector=query_vector)
        if tier == "vector":
            # Both keys pass through. `check_query_keys` has already rejected both-set, and `_pick_tier`
            # chose this tier only because the mode's own key is present, so the mode check inside
            # `vector_stages` is satisfied either way and a conditional here would only duplicate it.
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
            # Not None on this branch: `_pick_tier` returns "text" only where `query_text` is set.
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
            # `or 0.0` rather than a default, because a stage that projected nothing yields `None` and a
            # `None` score would break the bound comparison rather than sort last.
            score = float(document.get(_pipeline.SCORE_FIELD) or 0.0)
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
                ),
            )
            scored.append((score, entry))

        return self._within_bounds(scored)

    def _pick_tier(self, *, query_text: str | None, query_vector: list[float] | None) -> ResolvedSearchType:
        """Choose the prepared tier that can answer this query (``REQ-E-054``, ``REQ-E-066``).

        Preference order comes from ``_tiers``, which ``initialize()`` built vector-first. The choice is
        the query's *shape* rather than the caller's stated preference, which is why ``_search_type``
        travels back in the result metadata: a store that reported the requested tier instead of the one
        that ran would satisfy every assertion except the one that matters.

        Args:
            query_text: What the caller put in ``text``, if anything.
            query_vector: What the caller put in ``vector``, if anything.

        Returns:
            The tier to build stages for.

        Raises:
            StorageError: If no tier is prepared at all. Two conditions with two remedies, so two
                messages: a deployment that cannot search is not the same as one where index creation
                failed, and advice naming ``createSearchIndexes`` is unusable on the first.
            ValueError: If tiers exist but none fits this query's shape. A caller error, and the message
                names the key to add.
        """
        if not self._tiers:
            capabilities = self._connection.capabilities
            if capabilities is None or not capabilities.search:
                # A phrase rather than the bare code, because the `capabilities is None` branch has no
                # code to name and "server error code None" would read as one the server sent.
                code = capabilities.no_search_code if capabilities is not None else None
                evidence = f"server error code {code}" if code is not None else "no probe result cached"
                raise StorageError(_NO_SEARCH_QUERY % (self._target, evidence))
            raise StorageError(_NO_TIER_PREPARED % self._target)

        for tier in self._tiers:
            if tier == "text" and query_text is not None:
                return "text"
            # The vector tier's own key follows the configured path: AutoEmbeddings embeds `text` on the
            # deployment, and the manual path indexes an embedding the caller computed.
            if tier == "vector" and (query_text if self._mode.auto_embed else query_vector) is not None:
                return "vector"

        needed, because = (
            ("'text'", "the deployment embeds it on the Atlas AutoEmbeddings path")
            if self._mode.auto_embed
            else ("'vector'", "'embedding_dimensions' and 'similarity' select the manual-vector path")
        )
        raise ValueError(_NO_TIER_FOR_QUERY % (self._target, ", ".join(self._tiers), needed, because))

    def _within_bounds(self, scored: list[tuple[float, MemoryEntry]]) -> list[MemoryEntry]:
        """Apply the configured score bound, in Python, after the query (``REQ-E-059``).

        The score travels alongside the entry rather than being read back out of
        ``metadata["_relevance_score"]``, so the comparison uses the float the mapping computed instead
        of re-parsing this package's own output.

        In Python rather than as a ``$match`` on ``$meta``: the documents are already retrieved and paid
        for either way, and a server-side bound would need a different expression per tier for scores
        that are not on comparable scales. BM25's ``searchScore`` is unbounded and depends on corpus
        statistics; cosine ``vectorSearchScore`` is normalised to ``[0, 1]``.

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

        # Both counts, and only when the bound actually dropped something. A bound set slightly too high
        # returns an empty list, and "kept 0 of 8" is the difference between reading that as a tuning
        # problem and reading it as an empty memory. Silent on a no-op, so polling for a corpus that has
        # not reached mongot yet does not fill the log.
        if len(kept) != len(scored):
            logger.debug(_SCORE_BOUND_APPLIED, name, bound, self._target, len(kept), len(scored))
        return kept

    @_reporting("deleting a memory from")
    async def delete(self, memory_id: str) -> int:
        """Remove one memory this store can see, named by the id ``search`` reported (``REQ-E-076``).

        Not part of the SDK's ``MemoryStore`` protocol, which has no deletion member at all: this is the
        package's own surface, added because a store that can accept a fact and never drop one cannot
        answer a deletion request against the data it holds.

        The predicate is :func:`_filter.compose`'s output with ``_id`` added, which is the read path's
        call unchanged rather than a second predicate that happens to agree with it. That is the whole of
        the isolation claim here (``REQ-E-078``): a store can delete only what it can read, so a
        well-formed id belonging to another scope or another store name matches nothing.

        Three cases return ``0`` and are deliberately indistinguishable: the memory was already deleted,
        it belongs to another scope, or it belongs to another store in this scope. Telling them apart
        would answer "does this id exist somewhere I cannot read?", which is a cross-scope existence
        oracle built out of a delete.

        Args:
            memory_id: The ``_id`` string from a search result's metadata. The read path projects
                ``str(document["_id"])``, so the id arrives here as a string and is parsed back.

        Returns:
            ``1`` if a document was removed, ``0`` if none matched. The server's count rather than a
            constant, so a caller writing an erasure record has something that came from the deployment.

        Raises:
            ValueError: If this store is not writable (``REQ-E-079``), or if ``memory_id`` is not a
                24-character hex ObjectId (``REQ-E-076``). Both are raised before any network call, and a
                document whose ``_id`` is not an ``ObjectId`` is therefore unreachable through this
                method, which is a limit rather than an oversight: this package writes no such document.
            StorageError: If a driver call fails.
        """
        # Before the parse and before `initialize()`, in that order. A read-only store must turn a delete
        # down without having opened a connection or created an index to do it, and it should say
        # "not writable" rather than complain about an id it was never going to use.
        if not self.writable:
            raise ValueError(_NOT_WRITABLE % (self.name, "delete()"))
        object_id = _object_id(memory_id)

        await self.initialize()
        predicate = _filter.compose(self.filter, scope=self.scope, store_name=self.name)
        result = await self._collection().delete_one({**predicate, "_id": object_id})
        return result.deleted_count

    @_reporting("deleting every memory in")
    async def delete_all(self) -> int:
        """Remove every memory this store can see (``REQ-E-077``).

        The same predicate :meth:`delete` and :meth:`search` build, with nothing added, so what this
        empties is exactly what this store can read: one scope's documents under *this* store's name,
        narrowed further by any configured ``filter``. Two stores over one collection in one scope are two
        calls, and that is an open question rather than a settled design: a scope-wide erasure that took
        one call would have to reach past ``store_name``, and reaching past the isolation predicate is the
        one thing this method will not do without a requirement saying so.

        Returns:
            How many documents the server removed. Worth reading rather than discarding: it is the only
            number an erasure record can cite, and a ``0`` against a scope believed to hold facts says the
            scope or the store name in the caller's configuration is not the one that was written to.

        Raises:
            ValueError: If this store is not writable (``REQ-E-079``), raised before any network call.
            StorageError: If a driver call fails.
        """
        if not self.writable:
            raise ValueError(_NOT_WRITABLE % (self.name, "delete_all()"))

        await self.initialize()
        predicate = _filter.compose(self.filter, scope=self.scope, store_name=self.name)
        result = await self._collection().delete_many(predicate)
        return result.deleted_count

    def get_tools(self) -> list[AgentTool]:
        """Return this store's own agent tools, of which v0.1.0 has none (``REQ-E-052a``).

        Present and empty rather than absent. ``MemoryManager._build_tools`` reads
        ``if _has_method(store, "get_tools"): tools.extend(store.get_tools())``, so a present method
        returning ``[]`` extends the list by nothing and is indistinguishable from absence at runtime.
        That is the opposite of ``add_messages``, whose presence decides which extractor runs, and it is
        what lets v0.2.0 add a tool here without a compatibility event.

        Returns:
            An empty ``list``. A ``list`` specifically, since the manager calls ``extend`` on it.
        """
        return []

    @property
    def _target(self) -> str:
        """The namespace this store operates on, for messages that must not name a memory.

        Returns:
            ``database.collection``. The only subject :func:`_reporting` renders, because the subject of
            a memory operation is the memory itself and naming it would put user content into an error
            string, a log line and every traceback that reaches a ticket.
        """
        return f"{self._database_name}.{self._collection_name}"

    @property
    def _connection(self) -> _client._Connection:
        """The connection serving the running event loop (``REQ-E-017b``).

        A property rather than an attribute, matching ``MongoDBStorage._connection`` and for the same
        reason: ``resolve`` answers per loop (``REQ-E-017a``), and a reference captured in ``__init__``
        would pin this store to the loop that constructed it. A store built once and used from two loops
        is the ordinary case, not an exotic one, because the SDK's synchronous entry points run their
        work on a fresh loop in a worker thread.

        One consequence worth naming: ``capabilities`` is cached on the connection, and a second loop has
        a second connection, so the first store to call ``initialize`` on it probes the deployment once
        more. A store that already initialized on an earlier loop does not: ``_initialized`` is per store
        instance and short-circuits before the probe, which is right, because indexes and search support
        are properties of the deployment and do not change with the loop.
        """
        return _client.resolve(self._connection_config)

    def _collection(self) -> AsyncCollection[dict[str, Any]]:
        """Resolve this store's collection from the shared connection.

        A method rather than a cached attribute, matching ``MongoDBStorage``. The client behind it
        differs per event loop (``REQ-E-017a``), and a collection captured at construction would keep a
        handle on a client bound to a loop that may already be closed.

        Returns:
            The collection, resolved fresh on every call.
        """
        return self._connection.client[self._database_name][self._collection_name]

    async def close(self) -> None:
        """Release the MongoDB connection, closing the client only if this package built it.

        Not part of the ``MemoryStore`` protocol, so ``MemoryManager`` never calls it. Same scope as
        ``MongoDBStorage.close``: per config rather than per store, because one registry entry serves
        every store built from that config. Idempotent, and a no-op for an injected client
        (``REQ-E-010``).

        A sibling store that keeps working afterwards opens a new pool rather than raising off the closed
        client, because resolution is per operation (``REQ-E-017b``). Nothing closes that pool again, so
        call this once at process shutdown.
        """
        await _client.close(self._connection_config)
