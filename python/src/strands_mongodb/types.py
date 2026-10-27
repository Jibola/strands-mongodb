"""Configuration, query and result types for the MongoDB backends.

The shapes follow the SDK's ``strands.vended_memory_stores.bedrock_knowledge_base.types``: ``TypedDict``
config with mandatory keys marked ``Required``, the connection settings nested under ``config``, the
client injected through the config dict, and an ``add`` result as a ``@dataclass``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from typing_extensions import Required, TypedDict

from strands_mongodb._extras import missing_extra

# Both extras install `strands-agents`; the message names the smaller one and also suggests `[all]`.
try:
    from strands.memory import MemoryStoreConfig
except ImportError as error:
    missing_extra("storage", error, module=__name__)

if TYPE_CHECKING:
    # Type-only, so importing this module never imports the driver. This depends on
    # `from __future__ import annotations`: without it `TypedDict` evaluates annotations eagerly and the
    # class bodies below raise `NameError` at import.
    from pymongo import AsyncMongoClient

__all__ = [
    "DEFAULT_MAX_SEARCH_RESULTS",
    "DEFAULT_OVERSAMPLING_FACTOR",
    "MongoDBAddResult",
    "MongoDBConfig",
    "MongoDBMemoryStoreConfig",
    "MongoDBSearchQuery",
    "RESERVED_METADATA_KEYS",
    "ResolvedSearchType",
    "SearchType",
    "VectorSimilarity",
]

SearchType = Literal["auto", "vector", "text", "hybrid"]
"""Search tiers a caller may ask for.

``"auto"`` picks the best tier the deployment supports. ``"hybrid"`` runs one ``$rankFusion`` over the
vector and text pipelines and needs both tiers plus MongoDB 8.0 or later. Under ``"auto"``, hybrid is
preferred where available, except when ``min_score`` or ``max_score`` is set, because fused scores are on
a different scale (see those keys).
"""

VectorSimilarity = Literal["cosine", "dotProduct", "euclidean"]
"""Distance metrics an Atlas vector index accepts, for the manual-vector path.

The value is also checked at runtime, for configs built from environment variables or settings files.
"""

ResolvedSearchType = Literal["vector", "text", "hybrid"]
"""Search tier that answered a query, reported in the ``_search_type`` metadata key.

Separate from :data:`SearchType` because ``"auto"`` can be requested but is never the answer.
"""

DEFAULT_MAX_SEARCH_RESULTS: Final[int] = 10
"""Result ceiling when neither the call nor the store supplies one.

Matches ``FileMemoryStore`` and ``BedrockKnowledgeBaseStore``, so swapping stores keeps the result count.
"""

DEFAULT_OVERSAMPLING_FACTOR: Final[int] = 10
"""Multiplier turning a result limit into a ``$vectorSearch`` candidate count (``numCandidates``).

Matches the ``pymongo_search_utils`` default.
"""

RESERVED_METADATA_KEYS: Final[frozenset[str]] = frozenset({"_relevance_score", "_id", "_search_type", "_scope"})
"""Metadata keys this package sets on every returned ``MemoryEntry``.

``MemoryEntry`` has no score field, so relevance, document id, answering tier and scope travel in
``metadata``. On a collision with caller metadata the reserved value wins and the collision is logged,
following the ``BedrockKnowledgeBaseStore`` convention.
"""


class MongoDBConfig(TypedDict, total=False):
    """Connection and deployment settings, built once per process and shared across stores.

    Each config builds one ``AsyncMongoClient`` and its connection pool. Pass the same config to every
    store, rather than one per store, so they share a pool.

    Attributes:
        database: Database holding this deployment's collections. The only required key.
        client: An existing client to use. This package never closes it, not even in ``close()``.
        connection_string: Used only when ``client`` is absent. Supplying neither raises.
        client_args: Extra keyword arguments for the client this package constructs. Ignored when
            ``client`` is supplied.
        ensure_indexes: Whether ``initialize()`` creates missing indexes. Defaults to ``True``.
            Verification always runs, so ``False`` means verify only.
        index_ready_timeout: Seconds to wait for a search index to report ``READY``, timed separately for
            each search tier, so an ``auto`` store can wait this long once per tier. Defaults to
            ``_indexes.READY_TIMEOUT_SECONDS`` (300). Raise it on a shared deployment where index builds
            queue behind others.
        search_type: Tier to search with. Defaults to ``"auto"``, which resolves to the best tier the
            deployment supports and warns when it lands below vector. Under ``"auto"`` a tier skipped at
            ``initialize()`` is retried later (see ``memory_store.TIER_RETRY_SECONDS``). A named tier that
            cannot be prepared raises ``StorageError`` instead.
        embedding_model: Atlas embedding model for the AutoEmbeddings path, used in the index definition
            and the query stage. Mutually exclusive with the two manual-vector keys below. Setting none
            of them selects AutoEmbeddings with ``DEFAULT_EMBEDDING_MODEL``.
        embedding_dimensions: Length of the vectors stored on the documents, for the manual path where
            the caller embeds. Selects that path and requires ``similarity``.
        similarity: Distance metric for the manual path: ``"cosine"``, ``"dotProduct"`` or
            ``"euclidean"``. No default: the right metric depends on your embeddings, and the wrong one
            returns plausible but wrong neighbours.
        filter_fields: Extra document fields to declare filterable in the vector index. The server rejects
            a filter on an undeclared path (code 8), and adding a field later means recreating the index.
        vector_index_name: Name of the Atlas Vector Search index.
        text_index_name: Name of the Atlas Search full-text index.
    """

    database: Required[str]
    client: AsyncMongoClient[Any]
    connection_string: str
    client_args: dict[str, Any]
    ensure_indexes: bool
    index_ready_timeout: float
    search_type: SearchType
    embedding_model: str
    embedding_dimensions: int
    similarity: VectorSimilarity
    filter_fields: list[str]
    vector_index_name: str
    text_index_name: str


class MongoDBMemoryStoreConfig(MemoryStoreConfig, total=False):
    """Full configuration for a ``MongoDBMemoryStore``, passed as its constructor keyword arguments.

    Extends the SDK's ``MemoryStoreConfig``, so ``name``, ``description``, ``max_search_results``,
    ``writable`` and ``extraction`` are all accepted here and ``name`` stays mandatory. ``writable``
    defaults to ``False``, so the store accepts no writes, deletes or extraction until you set it.

    Attributes:
        config: Connection to the deployment. Reuse one across stores that differ only by ``scope``.
        collection: Collection holding this store's documents. Defaults to ``"memory"``.
        scope: Namespace isolating this store's documents. The store stamps it on every write and filters
            every search by it; callers never set it on documents. If absent, a distinct sentinel scope
            is used, never "unfiltered".
        filter: Extra predicate, ANDed with the scope predicate on ``search``, ``delete`` and
            ``delete_all``, so it can only narrow what the store reads or deletes. Validated at
            construction like a per-query filter.
        min_score: Floor a result must meet. The useful bound for vector and text scores, which are
            similarity-oriented. **Fused scores use a different scale:** the hybrid tier scores by
            reciprocal rank fusion, ``sum(weight / (RANK_FUSION_PENALTY + rank))`` with the penalty in
            ``_pipeline.RANK_FUSION_PENALTY``, so a document ranked first by both tiers scores about 0.033
            and one ranked first by a single tier about 0.016. A cosine floor of 0.7 would empty every
            hybrid result. So under ``"auto"``, setting either bound keeps the store off the hybrid tier.
            With ``search_type="hybrid"`` the bound applies to the fused score.
        max_score: Ceiling a result must not exceed. Mutually exclusive with ``min_score``. The same scale
            note applies: a ceiling meant for cosine scores admits every fused result. On a ``dotProduct``
            index the score is ``(1 + dot) / 2`` over unnormalized vectors, so a corpus vector longer than
            the query can score above 1 (2.0 for one five times its length). A ceiling of 1.0 there
            silently drops the strongest matches.
        ttl_seconds: How long a memory lives, in seconds from its own ``created_at``. Absent means no
            expiry and no expiry field on the document. ``0`` means "expire at the next TTL sweep", not
            "disabled". A negative value raises ``ValueError`` when the store is constructed. Set on insert
            only, so re-adding a fact the store already holds does not extend its expiry. ``initialize()``
            creates the ``strands_expires_at_ttl`` index, so no separate ``ensure_indexes()`` call is
            needed, unlike ``MongoDBStorage``.
    """

    config: Required[MongoDBConfig]
    collection: str
    scope: str
    filter: dict[str, Any]
    min_score: float
    max_score: float
    ttl_seconds: int


class MongoDBSearchQuery(TypedDict, total=False):
    """A MongoDB-specific search query, for ``MongoDBMemoryStore.search`` in place of a plain string.

    It widens the ``Storage`` protocol's ``SearchQuery`` type parameter, which the SDK documents as
    widenable to richer query objects. SDK-internal callers keep passing a plain string.

    Every key is optional because a ``TypedDict`` cannot express the valid combinations. Set ``text`` or
    ``vector``. Set both only on a manual-vector store that can use the hybrid tier, where each feeds one
    half of the fusion. Anything else raises ``ValueError`` at search time.

    Attributes:
        text: Natural-language query. Embedded server-side on the AutoEmbeddings path.
        vector: A pre-computed embedding, used directly. Needs no embedding model registered on the
            deployment.
        filter: MQL predicate over declared filterable fields, validated before it reaches the server.
        limit: Result ceiling for this call, overriding the store's own.
        oversampling_factor: Multiplier turning ``limit`` into a vector candidate count. Raising it trades
            query cost for recall. On the vector and hybrid tiers, a ``limit * oversampling_factor``
            above 10000 raises ``ValueError``.
    """

    text: str
    vector: list[float]
    filter: dict[str, Any]
    limit: int
    oversampling_factor: int


@dataclass
class MongoDBAddResult:
    """Result returned by ``MongoDBMemoryStore.add``.

    Attributes:
        ids: Stringified ``_id`` of every document the call resolved to, whether inserted now or
            already present.
        inserted: How many documents the call created.
        duplicates: How many were already present and therefore not created. Extraction writes are
            at-least-once, so this explains a fact count lower than the write count.
    """

    ids: list[str]
    inserted: int
    duplicates: int
