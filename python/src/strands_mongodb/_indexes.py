"""Index specifications, idempotent ensure and verify, and search-index readiness polling.

Written against ``AsyncCollection`` rather than calling ``pymongo_search_utils``' index helpers, which are
synchronous and poll with a blocking ``time.sleep``.

Each kind of index has its own idempotency test:

* **Ordinary** indexes (TTL included) match on the **key list**, then every option this package asked for
  is compared. Not on the name: the server rejects a second index over the same keys (code 85) whatever
  it is called. Not on the keys alone either: the dedup index exists for its ``unique`` flag, so a
  non-unique index over the same fields must not count.
* **Atlas Search** indexes match by name, are compared on ``latestDefinition``, then polled. A build is
  asynchronous, and a search over an unfinished index returns zero documents rather than an error.

Nothing here drops or alters an existing index. A conflict is reported with the commands that resolve it,
because the index may be serving queries this package knows nothing about.

Private module, but the specs are a promise to operators: the README's index commands are rendered
from them, and a test keeps the two identical.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

from strands_mongodb import _telemetry
from strands_mongodb._extras import missing_extra
from strands_mongodb._filter import BAKED_IN_FILTER_FIELDS

# Both surfaces reach this module and everything it needs is in the narrower `storage` extra.
try:
    from pymongo import IndexModel
    from pymongo.errors import OperationFailure, PyMongoError
    from pymongo.operations import SearchIndexModel
    from strands.types.exceptions import StorageError
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    missing_extra("storage", error, module=__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opentelemetry.util.types import AttributeValue
    from pymongo.asynchronous.collection import AsyncCollection

    from strands_mongodb.types import MongoDBConfig, TextEmbedder

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "IndexSpec",
    "MAX_NUM_CANDIDATES",
    "POLL_INTERVAL_SECONDS",
    "READY_TIMEOUT_SECONDS",
    "SEARCH_UNSUPPORTED_CODES",
    "SearchIndexSpec",
    "SearchSupport",
    "VectorMode",
    "check_ttl",
    "declared_filter_fields",
    "drop_index_command",
    "embed",
    "ensure",
    "memory_search_specs",
    "memory_specs",
    "RANK_FUSION_MIN_MAJOR",
    "rank_fusion_support",
    "readiness",
    "search_support",
    "storage_specs",
    "text_index_definition",
    "vector_index_definition",
    "vector_mode",
]

logger = logging.getLogger(__name__)

CONTENT_FIELD: Final[str] = "content"
"""Document field holding the text a memory is about, indexed for vector and full-text search."""

VECTOR_FIELD: Final[str] = "embedding"
"""Document field holding a caller-supplied embedding, on the manual-vector path only."""

EXPIRES_FIELD: Final[str] = "expires_at"
"""Document field the TTL indexes watch. Absent on documents with no expiry, for which the index is inert."""


def check_ttl(ttl_seconds: int | None) -> None:
    """Reject a negative ``ttl_seconds``.

    A negative lifetime stamps an ``expires_at`` in the past, so the TTL monitor deletes every write on its
    next sweep. ``0`` is allowed: it asks for expiry at the next sweep.

    Args:
        ttl_seconds: The configured lifetime, or ``None`` for no expiry.

    Raises:
        ValueError: If ``ttl_seconds`` is below ``0``.
    """
    if ttl_seconds is not None and ttl_seconds < 0:
        raise ValueError(
            f"strands_mongodb: 'ttl_seconds' is {ttl_seconds}, and a lifetime cannot be negative. A negative "
            f"value stamps an expiry in the past, so every write would be deleted on the next TTL sweep. Use "
            f"0 to expire at the next sweep, or None for no expiry."
        )


DEFAULT_EMBEDDING_MODEL: Final[str] = "voyage-4-lite"
"""Atlas embedding model used when the config names none. Matches the community ``strands-mongodb-store``."""

DEFAULT_VECTOR_INDEX_NAME: Final[str] = "vector_index"
"""Default Atlas Vector Search index name."""

DEFAULT_TEXT_INDEX_NAME: Final[str] = "text_index"
"""Default Atlas Search index name."""

SIMILARITIES: Final[tuple[str, ...]] = ("cosine", "dotProduct", "euclidean")
"""Distance metrics an Atlas vector index accepts. Mirrors ``types.VectorSimilarity`` at runtime."""

MAX_DIMENSIONS: Final[int] = 8192
"""Upper bound Atlas enforces on ``numDimensions``. Measured: outside ``[1..8192]`` is code 8."""

MAX_NUM_CANDIDATES: Final[int] = 10000
"""Upper bound the server enforces on ``numCandidates``.

Outside ``[1..10000]`` the server returns code 8, *"numCandidates" must be within bounds [1..10000]*
(measured on mongot 8.3.8). An integration test re-reads it from a real deployment.
"""

READY_TIMEOUT_SECONDS: Final[float] = 300.0
"""Default seconds to wait for search-index builds to report ``READY``.

Builds on cloud Atlas have taken about 105 seconds idle and over 120 seconds on a busy cluster. A longer
timeout delays only a failure. Override it with ``MongoDBConfig["index_ready_timeout"]``.
"""

POLL_INTERVAL_SECONDS: Final[float] = 0.5
"""Delay between readiness reads, in seconds."""

_READY_STATUS: Final[str] = "READY"
_FAILED_STATUS: Final[str] = "FAILED"
_NAMESPACE_EXISTS: Final[int] = 48

RANK_FUSION_MIN_MAJOR: Final[int] = 8
"""Major version of ``mongod`` from which ``$rankFusion`` (with ``$vectorSearch`` inputs) exists: 8.0, not 8.1."""

SEARCH_UNSUPPORTED_CODES: Final[frozenset[int]] = frozenset({31082, 115, 6047401})
"""Server error codes from ``list_search_indexes()`` that mean "this deployment has no Atlas Search".

* **31082** on ``mongo:8`` (8.3.8).
* **115** (``CommandNotSupported``) on enterprise ``mongod`` 7.0.12 with no mongot configured.
* **6047401** on ``mongo:6`` (6.0.28) and ``mongo:7`` (7.0.40).

The number is the stable part: the message differs between deployments, and so does ``codeName`` (31082 is
``SearchNotEnabled`` on 8.3.8 and ``Location31082`` on 7.0.12).

Code 59 (``CommandNotFound``) means the same at ``create_search_index`` on 6.0 and 7.0, but is left out on
purpose: that call only runs after :func:`search_support` has cleared the deployment, so it is unreachable.
"""

_COLLECTION_ABSENT: Final[int] = 26
"""Server error code for a namespace that does not exist. Not an error when reading indexes."""

SIMPLE_LOCALE: Final[str] = "simple"
"""The one collation locale that compares strings byte by byte. See :func:`require_binary_comparison`."""

Readiness = Literal["ready", "building", "failed"]
"""The three states of a search index build, each with a different remedy."""


class SearchSupport(NamedTuple):
    """What :func:`search_support` found.

    Attributes:
        available: Whether the deployment runs Atlas Search.
        code: The server error code that said it does not, so messages can name the code this server
            sent. Set only when ``available`` is ``False``, otherwise ``None``.
    """

    available: bool
    code: int | None = None


def readiness(status: str) -> Readiness:
    """Map a search index's reported ``status`` onto the state whose remedy the caller needs.

    Any other status (a new full-text index reports ``PENDING``, for one) reads as building, which costs
    a poll. Reading it as ready would search an unfinished index, which returns nothing and no error.

    Args:
        status: The ``status`` field as the server reported it.

    Returns:
        ``"ready"`` only for ``READY``, ``"failed"`` only for ``FAILED``, ``"building"`` otherwise.
    """
    if status == _READY_STATUS:
        return "ready"
    if status == _FAILED_STATUS:
        return "failed"
    return "building"


@dataclass(frozen=True)
class VectorMode:
    """Which of the two vector paths a config selected, resolved once and passed around.

    Attributes:
        model: Atlas embedding model, on the AutoEmbeddings path. ``None`` on the manual path.
        dimensions: Vector length, on the manual path. ``None`` on the AutoEmbeddings path.
        similarity: Distance metric, on the manual path. ``None`` on the AutoEmbeddings path.
    """

    model: str | None = None
    dimensions: int | None = None
    similarity: str | None = None

    @property
    def auto_embed(self) -> bool:
        """Whether the deployment embeds the text, rather than the caller.

        Returns:
            ``True`` when a model is set.
        """
        return self.model is not None


def vector_mode(config: MongoDBConfig) -> VectorMode:
    """Resolve the vector path from the config, rejecting the combinations that mean two things.

    Manual values are checked here because Atlas rejects them only at index creation, with a code 8.

    Args:
        config: The connection configuration. Only the vector-related keys are read.

    Returns:
        The resolved mode, with exactly one path populated. An empty config resolves to AutoEmbeddings
        with :data:`DEFAULT_EMBEDDING_MODEL`.

    Raises:
        ValueError: If both paths are configured, if only one of the manual pair is set, or if a manual
            value is outside what Atlas accepts. The message names the conflicting keys.
    """
    model = config.get("embedding_model")
    dimensions = config.get("embedding_dimensions")
    similarity = config.get("similarity")

    if model is not None and (dimensions is not None or similarity is not None):
        also = ", ".join(
            name
            for name, value in (
                ("embedding_dimensions", dimensions),
                ("similarity", similarity),
            )
            if value is not None
        )
        raise ValueError(
            f"strands_mongodb: 'embedding_model' selects the Atlas AutoEmbeddings path, where the "
            f"deployment embeds the text, and {also} selects the manual path, where you supply the "
            f"vectors. They index different fields with different contents, so set one or the other. "
            f"Remove 'embedding_model' to embed client-side, or remove {also} to let Atlas do it."
        )

    if dimensions is None and similarity is None:
        return VectorMode(model=model or DEFAULT_EMBEDDING_MODEL)

    if dimensions is None:
        raise ValueError(
            "strands_mongodb: 'similarity' is set but 'embedding_dimensions' is not, and a vector "
            "index cannot be created without the vector length. Set 'embedding_dimensions' to the "
            "length your embedding model produces, or remove 'similarity' to use Atlas "
            "AutoEmbeddings."
        )
    if similarity is None:
        raise ValueError(
            "strands_mongodb: 'embedding_dimensions' is set but 'similarity' is not. There is no "
            "default: the right metric is a property of your embeddings, and the wrong one returns "
            f"plausible neighbours that are the wrong neighbours. Set one of {', '.join(SIMILARITIES)}."
        )
    if not 1 <= dimensions <= MAX_DIMENSIONS:
        raise ValueError(
            f"strands_mongodb: 'embedding_dimensions' is {dimensions}, and Atlas requires a value "
            f"within bounds [1..{MAX_DIMENSIONS}]. Set it to the length your embedding model produces."
        )
    if similarity not in SIMILARITIES:
        raise ValueError(
            f"strands_mongodb: 'similarity' is {similarity!r}, which Atlas does not accept. Choose "
            f"one of {', '.join(SIMILARITIES)}."
        )
    return VectorMode(dimensions=dimensions, similarity=similarity)


def declared_filter_fields(extra: Iterable[str]) -> tuple[str, ...]:
    """Return every path the vector index declares filterable, in index order.

    Shared by the index definition and ``_filter``'s check of filter paths, so the two cannot drift.

    Args:
        extra: The config's ``filter_fields``. Duplicates of the baked-in fields are dropped, since Atlas
            rejects a definition with two ``filter`` entries on one path.

    Returns:
        ``scope`` and ``store_name`` first, then the caller's fields in the order given.
    """
    fields = list(BAKED_IN_FILTER_FIELDS)
    for path in extra:
        if path not in fields:
            fields.append(path)
    return tuple(fields)


def vector_index_definition(
    mode: VectorMode,
    *,
    text_field: str = CONTENT_FIELD,
    vector_field: str = VECTOR_FIELD,
    filter_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the Atlas Vector Search index definition for the ``memory`` collection.

    AutoEmbeddings uses ``type: "autoEmbed"`` over the text field (``numDimensions: -1`` is code 8).

    Args:
        mode: The resolved path, from :func:`vector_mode`.
        text_field: Field the deployment embeds, on the AutoEmbeddings path.
        vector_field: Field holding the caller's vectors, on the manual path.
        filter_fields: Extra paths to declare filterable, beyond ``scope`` and ``store_name``.

    Returns:
        A definition ready for ``SearchIndexModel``, with the vector field first and every filter path
        after it.
    """
    if mode.auto_embed:
        first: dict[str, Any] = {
            "type": "autoEmbed",
            "path": text_field,
            "model": mode.model,
            "modality": "text",
        }
    else:
        first = {
            "type": "vector",
            "path": vector_field,
            "numDimensions": mode.dimensions,
            "similarity": mode.similarity,
        }
    return {
        "fields": [
            first,
            *({"type": "filter", "path": path} for path in declared_filter_fields(filter_fields)),
        ]
    }


def text_index_definition(*, text_field: str = CONTENT_FIELD) -> dict[str, Any]:
    """Build the Atlas Search full-text index definition for the ``memory`` collection.

    ``dynamic: False`` indexes only the text field, not metadata no query reads.

    Args:
        text_field: The field to map as a string.

    Returns:
        A definition ready for ``SearchIndexModel``.
    """
    return {"mappings": {"dynamic": False, "fields": {text_field: [{"type": "string"}]}}}


@dataclass(frozen=True)
class IndexSpec:
    """One ordinary index: what to create, and what makes an existing index count as it.

    Attributes:
        name: Name used when this package creates the index. Never used to find one, since the server
            does not treat the name as an identity.
        keys: The key list, in order, used to match existing indexes. It can match several: the
            server's identity for an index is its keys and its collation, and every match is examined.
        options: Every option this package asks for, and so every option compared against an existing
            index. Options set by someone else (``background``, ``hidden``) are not a mismatch.
        date_only: Whether every key names a field that holds only BSON dates. A collation cannot change
            how such an index compares, so the collation checks skip it.
    """

    name: str
    keys: tuple[tuple[str, int], ...]
    options: Mapping[str, Any] = field(default_factory=dict)
    date_only: bool = False

    def to_model(self) -> IndexModel:
        """Build the driver object that creates this index.

        Returns:
            An ``IndexModel`` carrying the keys, the name and every option.
        """
        return IndexModel(list(self.keys), name=self.name, **dict(self.options))

    def creation_command(self, collection_name: str) -> str:
        """Render the raw ``createIndexes`` command (what pymongo sends) for an operator to run.

        Args:
            collection_name: Collection the index belongs on.

        Returns:
            A single-line ``db.runCommand({...})`` whose argument is valid JSON.
        """
        command = {
            "createIndexes": collection_name,
            "indexes": [{"key": dict(self.keys), "name": self.name, **dict(self.options)}],
        }
        return f"db.runCommand({json.dumps(command)})"


def drop_index_command(collection_name: str, index_name: str) -> str:
    """Render the command that removes a conflicting ordinary index, for a mismatch message.

    Rendered, never run: the index may be serving queries this package knows nothing about.

    Args:
        collection_name: Collection the index sits on.
        index_name: Name the deployment reported for it.

    Returns:
        A single-line ``db.runCommand({...})`` whose argument is valid JSON.
    """
    # Encoded, not interpolated: the name is whatever the deployment reports, and an operator pastes this.
    return f"db.runCommand({json.dumps({'dropIndexes': collection_name, 'index': index_name})})"


@dataclass(frozen=True)
class SearchIndexSpec:
    """One Atlas Search index: ``vectorSearch`` or ``search``.

    Attributes:
        name: The index name, which for a search index is its identity. ``createSearchIndexes`` rejects a
            different definition under an existing name with code 68.
        definition: The definition. An existing index must contain it (see ``_definition_gap``).
        type: ``"vectorSearch"`` or ``"search"``.
    """

    name: str
    definition: Mapping[str, Any]
    type: str

    def to_model(self) -> SearchIndexModel:
        """Build the driver object that creates this index.

        Returns:
            A ``SearchIndexModel`` carrying the definition, the name and the type.
        """
        return SearchIndexModel(definition=dict(self.definition), name=self.name, type=self.type)

    def creation_command(self, collection_name: str) -> str:
        """Render the ``createSearchIndexes`` command that creates this index, for an operator to run.

        Args:
            collection_name: Collection the index belongs on.

        Returns:
            A single-line ``db.runCommand({...})`` whose argument is valid JSON.
        """
        command = {
            "createSearchIndexes": collection_name,
            "indexes": [{"name": self.name, "type": self.type, "definition": dict(self.definition)}],
        }
        return f"db.runCommand({json.dumps(command)})"

    def drop_command(self, collection_name: str) -> str:
        """Render the command that removes this search index, for a mismatch or failure message.

        Args:
            collection_name: Collection the index sits on.

        Returns:
            A single-line ``db.runCommand({...})`` whose argument is valid JSON.
        """
        return f"db.runCommand({json.dumps({'dropSearchIndex': collection_name, 'name': self.name})})"


def storage_specs() -> tuple[IndexSpec, ...]:
    """Return the ordinary indexes the ``storage`` collection needs.

    Keys are stored in ``_id``, whose index the server creates, so only the TTL index remains.

    Returns:
        One spec, the same for every store.
    """
    return (
        IndexSpec(
            name="strands_expires_at_ttl",
            keys=((EXPIRES_FIELD, 1),),
            options={"expireAfterSeconds": 0},
            # This package only ever writes `expires_at` as a BSON date, so a collation cannot change what
            # this index compares.
            date_only=True,
        ),
    )


def memory_specs() -> tuple[IndexSpec, ...]:
    """Return the ordinary indexes the ``memory`` collection needs.

    The TTL index exists whether or not ``ttl_seconds`` is set, so a store that sets one later needs no index
    change: its memories expire from the first write after that. Memories written before carry no
    ``expires_at`` and never expire, because the index is inert for documents without the field.

    Returns:
        Three specs: the unique partial dedup index, the recency index, and the TTL index.
    """
    return (
        IndexSpec(
            name="strands_scope_store_content_hash_unique",
            keys=(("scope", 1), ("store_name", 1), ("content_hash", 1)),
            options={
                "unique": True,
                # A unique index treats a missing field as `null`, so documents without a hash would collide.
                "partialFilterExpression": {"content_hash": {"$exists": True}},
            },
        ),
        IndexSpec(
            name="strands_scope_store_created_at",
            keys=(("scope", 1), ("store_name", 1), ("created_at", -1)),
        ),
        IndexSpec(
            name="strands_expires_at_ttl",
            keys=((EXPIRES_FIELD, 1),),
            options={"expireAfterSeconds": 0},
            # This package only ever writes `expires_at` as a BSON date, so a collation cannot change what
            # this index compares.
            date_only=True,
        ),
    )


def memory_search_specs(config: MongoDBConfig) -> tuple[SearchIndexSpec, ...]:
    """Return the Atlas Search indexes the ``memory`` collection needs.

    Args:
        config: The connection configuration.

    Returns:
        The vector index first, then the full-text index.

    Raises:
        ValueError: From :func:`vector_mode`, if the vector configuration means two things.
    """
    mode = vector_mode(config)
    return (
        SearchIndexSpec(
            name=config.get("vector_index_name") or DEFAULT_VECTOR_INDEX_NAME,
            definition=vector_index_definition(mode, filter_fields=config.get("filter_fields") or ()),
            type="vectorSearch",
        ),
        SearchIndexSpec(
            name=config.get("text_index_name") or DEFAULT_TEXT_INDEX_NAME,
            definition=text_index_definition(),
            type="search",
        ),
    )


async def search_support(collection: AsyncCollection[Any]) -> SearchSupport:
    """Report whether this deployment can run Atlas Search at all, and on what evidence.

    Called before :func:`ensure` gets search specs, so a deployment without mongot gets its ordinary
    indexes and a warning. Probing first keeps errors from ``createSearchIndexes`` on a deployment that
    has mongot from being mistaken for "no search".

    The probe calls ``list_search_indexes`` because ``buildInfo`` cannot tell: ``mongo:8`` and an Atlas
    CLI local deployment both report empty ``modules`` and only the second has search, while enterprise
    7.0.12 reports ``modules: ['enterprise']`` and has none. The result is always truthy; test
    ``.available``.

    Args:
        collection: The collection whose deployment to probe. Need not exist.

    Returns:
        :class:`SearchSupport`. ``available`` is ``True`` if the deployment answers the call, ``False``
        if it answers with one of :data:`SEARCH_UNSUPPORTED_CODES`, in which case ``code`` carries the
        one it used.

    Raises:
        StorageError: If the call fails for any other reason, including authentication. A bad password
            must not read as "no search".
    """
    with _telemetry.span("strands_mongodb.indexes.probe", _probe_attributes(collection, "search")):
        return await _search_support(collection)


async def _search_support(collection: AsyncCollection[Any]) -> SearchSupport:
    """The body of :func:`search_support`, inside its probe span."""
    try:
        cursor = await collection.list_search_indexes()
        async for _ in cursor:
            break
    except OperationFailure as error:
        if error.code in SEARCH_UNSUPPORTED_CODES:
            return SearchSupport(available=False, code=error.code)
        # Code 26 (absent collection) still means the call worked, so search is available.
        if error.code != _COLLECTION_ABSENT:
            raise StorageError(
                f"strands_mongodb: probing Atlas Search support on "
                f"{collection.database.name}.{collection.name} failed: {error}"
            ) from error
    except PyMongoError as error:
        raise StorageError(
            f"strands_mongodb: probing Atlas Search support on "
            f"{collection.database.name}.{collection.name} failed: {error}"
        ) from error
    return SearchSupport(available=True)


class CollationReport(NamedTuple):
    """What ``listCollections`` says about one collection's string comparison.

    An absent collection and an uncollated one both have no collation, but only the second is verified.

    Attributes:
        exists: Whether the server reported the collection at all.
        collation: The collation document it reported, or ``None`` if there was none (or no collection).
    """

    exists: bool
    collation: dict[str, Any] | None


async def read_collation(collection: AsyncCollection[Any]) -> CollationReport:
    """Report whether the collection exists and what default collation it carries.

    Uses ``listCollections`` filtered to this name, so a shared database's other collections are not
    listed. The server (8.0.32) reports a collation under ``options.collation`` and does not store a
    simple one: a collection created with ``{locale: "simple"}`` reports empty ``options``.

    Args:
        collection: The collection to ask about. Need not exist.

    Returns:
        A :class:`CollationReport`: ``(False, None)`` for an absent collection, ``(True, None)`` for one
        without a collation.

    Raises:
        StorageError: If the call fails. An absent collection is not a failure.
    """
    with _telemetry.span("strands_mongodb.indexes.probe", _probe_attributes(collection, "collation")):
        return await _read_collation(collection)


async def _read_collation(collection: AsyncCollection[Any]) -> CollationReport:
    """The body of :func:`read_collation`, inside its probe span."""
    try:
        cursor = await collection.database.list_collections(filter={"name": collection.name})
        async for document in cursor:
            options = document.get("options") or {}
            collation = options.get("collation")
            return CollationReport(exists=True, collation=dict(collation) if collation is not None else None)
    except PyMongoError as error:
        raise StorageError(
            f"strands_mongodb: reading the collation of {collection.database.name}.{collection.name} failed: {error}"
        ) from error
    return CollationReport(exists=False, collation=None)


async def require_binary_comparison(collection: AsyncCollection[Any]) -> bool:
    """Refuse a collection whose default collation is not binary, and say whether it verified one.

    An isolation check. Keys live in ``_id``, and key equality, prefix ranges and ``_id`` uniqueness all
    use the collection's default collation. Under a case-folding one, a namespace can read and delete
    another's documents, and keys differing only in case collide. A per-query ``{"locale": "simple"}``
    is no fix: such queries cannot use the ``_id`` index, and ``_id`` uniqueness would stay folded.

    An absent collection passes, since this package creates collections implicitly without a collation,
    but it is not verified: someone may create it a moment later with any collation.

    Args:
        collection: The collection about to be operated on.

    Returns:
        ``True`` if the collection exists and compares strings byte by byte, an answer worth caching.
        ``False`` if it does not exist yet, which must not be cached.

    Raises:
        StorageError: If the collection has a collation whose locale is not :data:`SIMPLE_LOCALE`,
            naming the locale found and the remedy. Also if reading the collation fails, with the driver's
            error chained.
    """
    report = await read_collation(collection)
    collation = report.collation
    if collation is None or collation.get("locale") == SIMPLE_LOCALE:
        return report.exists
    namespace = f"{collection.database.name}.{collection.name}"
    raise StorageError(
        f"strands_mongodb: the collection {namespace} was created with the collation "
        f"{collation.get('locale')!r}, and this package cannot store keys in a collection whose string "
        f"comparison is not binary. Keys live in '_id': a read is an equality match on it, a prefix "
        f"listing is a range over it, and a namespace is part of it, so a collation that folds case or "
        f"accents lets one namespace read and delete another's documents, and makes two keys differing "
        f"only in case collide on the unique '_id' index. A collation cannot be changed in place. Point "
        f"this store at a collection created without one -- the 'collection' argument -- or recreate "
        f"this one with db.createCollection({collection.name!r}) and no collation, or with "
        f"{{collation: {{locale: {SIMPLE_LOCALE!r}}}}}, which is binary comparison stated explicitly. "
        f"Reported: {collation!r}."
    )


async def rank_fusion_support(collection: AsyncCollection[Any]) -> bool:
    """Report whether this deployment can run ``$rankFusion``.

    Unlike :func:`search_support`, this reads ``buildInfo``: ``$rankFusion`` is a ``mongod`` stage, so
    ``mongod``'s version decides it. ``versionArray`` avoids parsing suffixes such as ``-rc1``.

    Every failure reads as ``False``. The answer only picks between two working search tiers, so a wrong
    ``False`` costs ranking quality while a wrong ``True`` makes every search fail.

    Args:
        collection: Any collection on the deployment to probe. Need not exist, since ``buildInfo`` is a
            database command.

    Returns:
        ``True`` if the server reports major version :data:`RANK_FUSION_MIN_MAJOR` or later.
    """
    with _telemetry.span("strands_mongodb.indexes.probe", _probe_attributes(collection, "rank_fusion")):
        return await _rank_fusion_support(collection)


async def _rank_fusion_support(collection: AsyncCollection[Any]) -> bool:
    """The body of :func:`rank_fusion_support`, inside its probe span."""
    try:
        build_info = await collection.database.command("buildInfo")
    except PyMongoError:
        return False
    version = build_info.get("versionArray") if isinstance(build_info, Mapping) else None
    if not isinstance(version, Sequence) or isinstance(version, str) or not version:
        return False
    major = version[0]
    # `bool` is a subclass of `int`; a boolean here is not a known version.
    if not isinstance(major, int) or isinstance(major, bool):
        return False
    return major >= RANK_FUSION_MIN_MAJOR


async def ensure(
    collection: AsyncCollection[Any],
    *,
    ordinary: Sequence[IndexSpec] = (),
    search: Sequence[SearchIndexSpec] = (),
    create: bool = True,
    timeout: float = READY_TIMEOUT_SECONDS,
    interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """Bring a collection's indexes to the declared state, or explain what is stopping that.

    Safe to call on every ``initialize()``: when everything is in place it costs one read for the
    ordinary indexes plus one per search index, and no writes. ``create=False`` verifies without
    creating (for credentials with no DDL rights) and still waits for a build someone else started.
    Nothing here drops or alters an existing index; each conflict is reported with the commands that
    resolve it.

    Args:
        collection: The collection to index. Its ``database`` is used to create the collection before a
            search index is created.
        ordinary: Ordinary and TTL indexes, from :func:`storage_specs` or :func:`memory_specs`.
        search: Atlas Search indexes, from :func:`memory_search_specs`.
        create: Whether missing indexes may be created.
        timeout: Seconds to wait for search builds, in total across all of them.
        interval: Seconds between readiness reads.

    Raises:
        StorageError: If an existing index conflicts with a declared one, if a search build has failed,
            if a build does not finish inside ``timeout``, if something is missing and ``create`` is
            ``False``, or if a driver call fails. Driver errors are chained, so the server's code and
            message survive.
    """
    attributes = _namespace_attributes(collection)
    attributes["strands_mongodb.indexes.mode"] = "ensure" if create else "verify"
    # A list the helpers append to, so the span still gets the count when one of them raises part way.
    created: list[int] = []
    with _telemetry.span("strands_mongodb.indexes.ensure", attributes):
        try:
            if ordinary:
                await _ensure_ordinary(collection, tuple(ordinary), create=create, created=created)
            if search:
                await _ensure_search(
                    collection, tuple(search), create=create, timeout=timeout, interval=interval, created=created
                )
        finally:
            _telemetry.annotate("strands_mongodb.indexes.created", sum(created))


def _namespace_attributes(collection: AsyncCollection[Any]) -> dict[_telemetry.Attribute, AttributeValue]:
    """The database and collection, which every index span carries."""
    return {"strands_mongodb.database": collection.database.name, "strands_mongodb.collection": collection.name}


def _probe_attributes(
    collection: AsyncCollection[Any], probe: Literal["search", "rank_fusion", "collation"]
) -> dict[_telemetry.Attribute, AttributeValue]:
    """A probe span's attributes: the namespace and what it probed."""
    attributes = _namespace_attributes(collection)
    attributes["strands_mongodb.indexes.probe"] = probe
    return attributes


async def _ensure_ordinary(
    collection: AsyncCollection[Any], specs: tuple[IndexSpec, ...], *, create: bool, created: list[int]
) -> None:
    """Create or verify the ordinary indexes, matching on the key list."""
    namespace = f"{collection.database.name}.{collection.name}"
    present = await _guard(collection.index_information(), what=f"reading indexes on {namespace}")
    # Several indexes can share a key list (identity is keys plus collation), so keep all of them.
    by_keys: dict[tuple[tuple[str, int], ...], list[tuple[str, Mapping[str, Any]]]] = {}
    for name, found in present.items():
        by_keys.setdefault(tuple(tuple(pair) for pair in found["key"]), []).append((name, found))

    missing: list[IndexSpec] = []
    for spec in specs:
        matches = by_keys.get(spec.keys)
        if not matches:
            missing.append(spec)
            continue
        _check_matches(collection.name, namespace, spec, matches)

    if not missing:
        return
    if not create:
        raise StorageError(_missing_ordinary_message(collection.name, namespace, missing))

    logger.info(
        "strands_mongodb: creating %d index(es) on %s: %s",
        len(missing),
        namespace,
        ", ".join(spec.name for spec in missing),
        extra=_telemetry.event_extra("creating_indexes", namespace=namespace),
    )
    # One call for all of them. The server tolerates concurrent identical creates (10 at once: no
    # failures, one index).
    await _guard(
        collection.create_indexes([spec.to_model() for spec in missing]),
        what=f"creating indexes on {namespace}",
    )
    created.append(len(missing))


def _folding_locale(found: Mapping[str, Any]) -> str | None:
    """Return the locale of a reported collation that does not compare strings byte for byte.

    Args:
        found: The entry ``index_information`` reported for an index.

    Returns:
        The locale, or ``None`` when the index reports no collation or ``simple``. The server does not
        store a simple collation, so an index created with one reports nothing.
    """
    collation = found.get("collation")
    if not isinstance(collation, Mapping):
        return None
    locale = collation.get("locale")
    if locale == SIMPLE_LOCALE:
        return None
    return str(locale)


def _check_matches(
    collection_name: str,
    namespace: str,
    spec: IndexSpec,
    matches: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    """Check every existing index over a declared index's keys, not just one.

    Two checks. First, a unique index under a folding collation is refused wherever it appears: the
    server enforces its folded uniqueness even beside a correct index, so records from different scopes
    can collide. Second, at least one match must be usable (the right options and a binary collation).
    Once one is, the others are someone else's business, so a non-unique folding index beside a usable
    one is accepted. ``matches`` is every existing index over the spec's keys, in server order.

    Raises:
        StorageError: If any match enforces uniqueness under a folding collation, or if no match is
            usable. In the second case the first match's failure is reported.
    """
    if not spec.date_only:
        for found_name, found in matches:
            locale = _folding_locale(found)
            if locale is not None and found.get("unique"):
                raise StorageError(_folded_constraint_message(collection_name, namespace, spec, found_name, locale))

    first: StorageError | None = None
    for found_name, found in matches:
        try:
            _check_collation(collection_name, namespace, spec, found_name, found)
            _check_options(collection_name, namespace, spec, found_name, found)
        except StorageError as refusal:
            first = first or refusal
            continue
        return
    # The caller never passes an empty `matches`, so reaching here means every match failed.
    assert first is not None
    raise first


def _folded_constraint_message(
    collection_name: str,
    namespace: str,
    spec: IndexSpec,
    found_name: str,
    locale: str,
) -> str:
    """Build the refusal for a unique index that enforces uniqueness under a folding collation.

    Separate from ``_check_collation``'s message because here the declared index may already be present
    and correct, so creating it does not help. The message ends with the drop command and the command
    that creates the declared index.
    """
    return (
        f"strands_mongodb: the index {found_name!r} on {namespace} covers the keys {list(spec.keys)} "
        f"that {spec.name!r} needs, it is unique, and it was created with the collation {locale!r}. The "
        f"server enforces that uniqueness on the folded values, so two records differing only in the case "
        f"of those keys collide as one -- and it goes on enforcing it whether or not a correctly collated "
        f"index sits beside it, because an index's identity here is its keys and its collation together "
        f"and the server keeps both indexes. This package queries and deduplicates with byte-for-byte "
        f"comparison, so it cannot see that rule and cannot report it. Dropping this index is not this "
        f"package's decision to make: it may be serving queries nothing here knows about. To resolve it, "
        f"drop it, and create the declared index if it is not already there:\n"
        f"{drop_index_command(collection_name, found_name)}\n"
        f"{spec.creation_command(collection_name)}\n"
    )


def _check_collation(
    collection_name: str,
    namespace: str,
    spec: IndexSpec,
    found_name: str,
    found: Mapping[str, Any],
) -> None:
    """Refuse a matched index whose collation does not compare strings byte for byte.

    An index carries its own collation, so a collection that passes ``require_binary_comparison`` can
    still hold a folding unique index. The dedup rule would then be the folded one: records whose keys
    differ only in case collide, so a memory can be reported as a duplicate and never stored.

    Not an :attr:`IndexSpec.options` entry: the server does not store a simple collation, so a wanted
    ``{"locale": "simple"}`` would be absent on every correct index.

    Raises:
        StorageError: If the index reports a collation whose locale is not ``simple``. The message holds
            the drop and recreate commands; nothing is dropped here.
    """
    if spec.date_only:
        return
    locale = _folding_locale(found)
    if locale is None:
        return
    raise StorageError(
        f"strands_mongodb: the index {found_name!r} on {namespace} covers the keys {list(spec.keys)} "
        f"that {spec.name!r} needs, but it was created with the collation {locale!r} and this package "
        f"needs one that compares strings byte for byte. An index that folds case makes the rule it "
        f"enforces the folded rule, which for {spec.name!r} means two records differing only in case "
        f"collide as one, and a query that compares exactly cannot use it at all. The server would let a "
        f"second index be created over these keys under a different collation, but that does not help: "
        f"both would exist and both would be enforced. Dropping this one is not this package's decision "
        f"to make -- it may be serving queries nothing here knows about. To resolve it, drop the existing "
        f"index and let the next call recreate it:\n"
        f"{drop_index_command(collection_name, found_name)}\n"
        f"{spec.creation_command(collection_name)}\n"
    )


def _check_options(
    collection_name: str,
    namespace: str,
    spec: IndexSpec,
    found_name: str,
    found: Mapping[str, Any],
) -> None:
    """Compare an existing index against the options a spec asked for, raising on the first difference.

    Only options this package sets are compared. ``v``, ``background``, ``hidden`` and anything else the
    server or another tool added are ignored.
    """
    for option, wanted in spec.options.items():
        if option in found and found[option] == wanted:
            continue
        actual = f"{found[option]!r}" if option in found else "absent"
        raise StorageError(
            f"strands_mongodb: the index {found_name!r} on {namespace} covers the keys "
            f"{list(spec.keys)} that {spec.name!r} needs, but its {option} is {actual} and this "
            f"package needs {wanted!r}. Creating a second index over the same keys under the same "
            f"collation is rejected by the server whatever it is named, and dropping this one is not "
            f"this package's decision to make: it may be serving queries nothing here knows about. To "
            f"resolve it, drop the "
            f"existing index and let the next call recreate it:\n"
            f"{drop_index_command(collection_name, found_name)}\n"
            f"{spec.creation_command(collection_name)}\n"
        )


def _missing_ordinary_message(collection_name: str, namespace: str, missing: Sequence[IndexSpec]) -> str:
    """Build the verify-only message, listing every missing index's creation command on its own line."""
    commands = "\n".join(spec.creation_command(collection_name) for spec in missing)
    return (
        f"strands_mongodb: {len(missing)} required index(es) are absent from {namespace} and "
        f"ensure_indexes is False, so nothing was created. Run these against the database as somebody "
        f"holding the rights to, then call initialize() again:\n"
        f"{commands}\n"
        f"Absent: {', '.join(spec.name for spec in missing)}."
    )


class SearchIndexNotReady(StorageError):
    """A search index exists but did not report ``READY`` within the timeout."""


async def _ensure_search(
    collection: AsyncCollection[Any],
    specs: tuple[SearchIndexSpec, ...],
    *,
    create: bool,
    timeout: float,
    interval: float,
    created: list[int],
) -> None:
    """Create or verify the Atlas Search indexes, then wait for anything still building."""
    namespace = f"{collection.database.name}.{collection.name}"
    found = {spec.name: await _read_search_index(collection, spec.name, namespace) for spec in specs}

    missing: list[SearchIndexSpec] = []
    waiting: list[SearchIndexSpec] = []
    for spec in specs:
        document = found[spec.name]
        if document is None:
            missing.append(spec)
            continue
        state = readiness(str(document.get("status", "")))
        if state == "failed":
            raise StorageError(_failed_message(collection.name, namespace, spec, document))
        _check_definition(collection.name, namespace, spec, document)
        if state != "ready":
            waiting.append(spec)

    if missing:
        if not create:
            raise StorageError(_missing_search_message(collection.name, namespace, missing))
        # `createSearchIndexes` on a missing collection fails with code 26, unlike `createIndexes`, which
        # creates it implicitly.
        await _create_collection(collection, namespace)
        for spec in missing:
            logger.info(
                "strands_mongodb: creating %s index %r on %s",
                spec.type,
                spec.name,
                namespace,
                extra=_telemetry.event_extra("creating_search_index", namespace=namespace),
            )
            await _guard(
                collection.create_search_index(spec.to_model()),
                what=f"creating search index {spec.name!r} on {namespace}",
            )
            created.append(1)
        waiting.extend(missing)

    if not waiting:
        return
    # Raised after the wait span closes: a timeout is the wait's outcome (`timed_out`, status unset), and
    # the error belongs to the `ensure` span.
    with _telemetry.span("strands_mongodb.indexes.wait", _namespace_attributes(collection)):
        stuck = await _wait_until_ready(collection, waiting, namespace, timeout=timeout, interval=interval)
        _telemetry.annotate("strands_mongodb.indexes.outcome", "ready" if stuck is None else "timed_out")
    if stuck is not None:
        raise SearchIndexNotReady(
            f"strands_mongodb: waited {timeout}s for {len(stuck)} search index(es) on "
            f"{namespace} to report READY and they did not: {', '.join(stuck)}. BUILDING at this point means "
            f"a build slower than the timeout. A store with a named search_type can call initialize() "
            f"again to keep waiting; under 'auto' the store answers from its other tiers and tries this one "
            f"again later. Any other status is worth reading directly with "
            f"db.{collection.name}.getSearchIndexes()."
        )


async def _read_search_index(collection: AsyncCollection[Any], name: str, namespace: str) -> dict[str, Any] | None:
    """Read one search index by name, or ``None`` if it is not there.

    ``list_search_indexes`` is a coroutine that returns an async cursor, so it is awaited before it is
    iterated.
    """
    try:
        cursor = await collection.list_search_indexes(name)
        async for document in cursor:
            return dict(document)
    except OperationFailure as error:
        # Code 26: an absent collection has no indexes. `$listSearchIndexes` is not documented to hide
        # this the way `list_indexes` does, so it is handled here.
        if error.code != _COLLECTION_ABSENT:
            raise StorageError(
                f"strands_mongodb: reading search index {name!r} on {namespace} failed: {error}"
            ) from error
    except PyMongoError as error:
        raise StorageError(f"strands_mongodb: reading search index {name!r} on {namespace} failed: {error}") from error
    return None


def _check_definition(collection_name: str, namespace: str, spec: SearchIndexSpec, document: Mapping[str, Any]) -> None:
    """Check that an existing search index's definition covers the declared one.

    The server would reject a different definition under the same name with code 68, but only on
    creation. This catches the mismatch on the warm path too, where nothing is created.
    """
    definition = document.get("latestDefinition")
    if definition is None:
        return
    gap = _definition_gap(spec.definition, definition)
    if gap is None:
        return
    raise StorageError(
        f"strands_mongodb: the {spec.type} index {spec.name!r} on {namespace} already exists and its "
        f"definition does not cover the one this package needs: {gap}. Reported: {definition!r}. A "
        f"search index cannot be redefined in place, and dropping it discards a build somebody may be "
        f"querying, so the remedy is left here rather than taken:\n"
        f"{spec.drop_command(collection_name)}\n"
        f"{spec.creation_command(collection_name)}\n"
    )


def _definition_gap(wanted: Any, found: Any, path: str = "definition") -> str | None:
    """Describe the first place ``found`` fails to cover ``wanted``, or ``None`` if it covers it.

    Containment, not equality, because mongot rewrites a ``search`` index's ``latestDefinition``: it
    collapses the one-element list in ``mappings.fields.<field>`` into the object it held and fills in
    ``indexOptions``, ``store`` and ``norms``. Equality would fail on a text index this package just
    created. (A ``vectorSearch`` definition comes back unchanged.)

    The rules:

    * A declared object is covered when every key it declares is present and covered. Added keys are
      ignored.
    * A declared one-element list is covered by an object reported in its place.
    * A declared list is covered when each of its entries is covered by some reported entry, in any order.

    Args:
        wanted: The declared definition, or a fragment of one.
        found: What the server reported in the same position.
        path: Dotted position within the definition, used only to say where a difference is.

    Returns:
        A sentence naming the position and both values, or ``None`` when nothing is missing.
    """
    if isinstance(wanted, Mapping):
        if not isinstance(found, Mapping):
            return f"{path} is {found!r} where this package declared the object {dict(wanted)!r}"
        for key, value in wanted.items():
            if key not in found:
                return f"{path}.{key} is absent, and this package declared {value!r}"
            gap = _definition_gap(value, found[key], f"{path}.{key}")
            if gap is not None:
                return gap
        return None

    if isinstance(wanted, list):
        if len(wanted) == 1 and isinstance(found, Mapping):
            return _definition_gap(wanted[0], found, path)
        if not isinstance(found, list):
            return f"{path} is {found!r} where this package declared the list {wanted!r}"
        for entry in wanted:
            if not any(_definition_gap(entry, candidate, path) is None for candidate in found):
                return f"{path} declares {entry!r} and nothing reported at {path} covers it"
        return None

    if wanted != found:
        return f"{path} is {found!r} where this package declared {wanted!r}"
    return None


def _failed_message(collection_name: str, namespace: str, spec: SearchIndexSpec, document: Mapping[str, Any]) -> str:
    """Build the message for a search index build that failed.

    Reported rather than recreated: issuing the same definition over a failed build is accepted but does
    not restart it. ``statusDetail`` is rendered whole, since its shape varies by host and index.
    """
    detail = json.dumps(document.get("statusDetail", []), default=str)
    return (
        f"strands_mongodb: the {spec.type} index {spec.name!r} on {namespace} reports status FAILED, "
        f"so searches against it return nothing rather than an error. The server's detail: {detail}. "
        f"A failed build does not restart when the same definition is issued again, so it has to be "
        f"dropped and recreated:\n"
        f"{spec.drop_command(collection_name)}\n"
        f"{spec.creation_command(collection_name)}\n"
    )


def _missing_search_message(collection_name: str, namespace: str, missing: Sequence[SearchIndexSpec]) -> str:
    """Build the verify-only message for absent search indexes."""
    commands = "\n".join(spec.creation_command(collection_name) for spec in missing)
    return (
        f"strands_mongodb: {len(missing)} required search index(es) are absent from {namespace} and "
        f"ensure_indexes is False, so nothing was created. A search against an absent index returns "
        f"no results rather than an error, so this is reported instead of ignored. Run these, wait for "
        f"each to report status READY, then call initialize() again:\n"
        f"{commands}\n"
        f"Absent: {', '.join(spec.name for spec in missing)}."
    )


async def _create_collection(collection: AsyncCollection[Any], namespace: str) -> None:
    """Create the collection if it is not already there, without checking first.

    ``check_exists=False`` avoids a check-then-act race and a round trip, and means pymongo never raises
    ``CollectionInvalid``. A lost race arrives as code 48 (namespace exists) instead, which is ignored.
    8.3.8 accepts the second create outright; the handler is for other server versions.
    """
    try:
        await collection.database.create_collection(collection.name, check_exists=False)
    except OperationFailure as error:
        if error.code != _NAMESPACE_EXISTS:
            raise StorageError(f"strands_mongodb: creating the collection {namespace} failed: {error}") from error
    except PyMongoError as error:
        raise StorageError(f"strands_mongodb: creating the collection {namespace} failed: {error}") from error


async def _wait_until_ready(
    collection: AsyncCollection[Any],
    specs: Sequence[SearchIndexSpec],
    namespace: str,
    *,
    timeout: float,
    interval: float,
) -> list[str] | None:
    """Poll until every index reports READY, or until ``timeout``.

    Sleeps with ``asyncio.sleep`` so other coroutines keep running during a build. Status changes are
    logged, so a long wait does not look like a hang. An index leaves ``pending`` only when it reports
    READY and its reported definition still covers the declared one.

    Returns:
        ``None`` once every index is ready, or ``"<name> is <status>"`` for each one still pending at the
        deadline, for the caller to raise on.

    Raises:
        StorageError: If a build reports FAILED, if a ready index's definition does not cover the declared
            one, or if a read fails.
    """
    deadline = time.monotonic() + timeout
    pending = {spec.name: spec for spec in specs}
    last: dict[str, str] = {}

    while True:
        for name in list(pending):
            document = await _read_search_index(collection, name, namespace)
            # A vanished index reads as building. Someone else dropped it, and recreating it would race
            # whoever did.
            status = "absent" if document is None else str(document.get("status", "unreported"))
            if status != last.get(name):
                logger.info(
                    "strands_mongodb: search index %r on %s is %s",
                    name,
                    namespace,
                    status,
                    extra=_telemetry.event_extra("search_index_status", namespace=namespace),
                )
                last[name] = status
            state = readiness(status)
            if state == "failed":
                # A missing document never reports FAILED, so `or {}` is only for the type checker.
                raise StorageError(_failed_message(collection.name, namespace, pending[name], document or {}))
            if state == "ready":
                # Recheck the definition: another actor may have dropped and recreated the index under
                # the same name during the wait, and an index created by this call was never checked.
                # Accepting a wrong definition makes searches return nothing while every index is READY.
                _check_definition(collection.name, namespace, pending[name], document or {})
                del pending[name]

        if not pending:
            return None
        if time.monotonic() >= deadline:
            return [f"{name} is {last[name]}" for name in pending]
        await asyncio.sleep(interval)


async def _guard(awaitable: Any, *, what: str) -> Any:
    """Await something and turn a driver error into ``StorageError``.

    ``StorageError`` is what the ``Storage`` protocol documents. The driver error is chained, so the
    server's code and message stay reachable through ``__cause__``.
    """
    try:
        return await awaitable
    except PyMongoError as error:
        raise StorageError(f"strands_mongodb: {what} failed: {error}") from error


async def embed(embedder: TextEmbedder, text: str, *, dimensions: int, setting: str) -> list[float]:
    """Embed ``text`` and check the vector's length against the index definition.

    The server rejects a wrong length on a query but silently leaves a written one out of the index.

    Args:
        embedder: What embeds the text.
        text: The text to embed.
        dimensions: The length the index declares.
        setting: The name of the option that set ``dimensions``, for the error message.

    Returns:
        The vector, as floats.

    Raises:
        ValueError: If the vector's length is not ``dimensions``.
    """
    vector = [float(value) for value in await embedder.embed(text)]
    if len(vector) != dimensions:
        raise ValueError(
            f"strands_mongodb: the embedder returned a {len(vector)}-dimension vector and {setting!r} is "
            f"{dimensions}. Set {setting!r} to the length the embedding model produces."
        )
    return vector
