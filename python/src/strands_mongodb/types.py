"""Configuration, query and result types for the MongoDB backends.

The shapes follow ``strands.vended_memory_stores.bedrock_knowledge_base.types`` rather than being
designed here, because that store is the SDK's own model for a memory store backed by a remote
service: ``TypedDict`` config with mandatory keys marked ``Required``, the connection half nested
under a key literally named ``config``, the client injected through the config dict, and an ``add``
result as a ``@dataclass``.

Both extras install ``strands-agents``, so this module needs no extra of its own to be correct --
but it is the one module whose dependency neither extra owns exclusively, so the guard below names
the smaller of the two and leans on the ``[all]`` line the message already carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from typing_extensions import Required, TypedDict

from strands_mongodb._extras import missing_extra

try:
    from strands.memory import MemoryStoreConfig
except ImportError as error:
    missing_extra("storage", error, module=__name__)

if TYPE_CHECKING:
    # Deferred, so annotating a config never constructs a driver. `from __future__ import
    # annotations` is what makes this work and is not optional here: `TypedDict` evaluates its
    # annotations eagerly to compute `__required_keys__`, so without it the class body below raises
    # `NameError` at import. Pinned by AT-056c.
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

Deliberately narrower than the tiers the package can describe. A ``Literal`` in a config is a promise
the caller can typecheck against, so it holds only values that actually run: ``"regex"`` is planned for
a later release and is absent here until it lands, rather than being a documented feature that raises.

``"hybrid"`` was absent on those grounds and now runs (``REQ-E-141``). Widening a ``Literal`` is
backwards compatible, which was the whole bet when this was written narrow, and it paid: every config
that typechecked before still does.

``"hybrid"`` resolves to one ``$rankFusion`` over the vector and text pipelines and needs both tiers
available plus a server at major version 8 or later. Under ``"auto"`` it is preferred where it is
available, **except** when ``min_score`` or ``max_score`` is set -- see those keys for the scale, which
is the reason for the exception (``REQ-E-145``).
"""

VectorSimilarity = Literal["cosine", "dotProduct", "euclidean"]
"""Distance metrics an Atlas vector index accepts, for the manual-vector path.

A ``Literal`` for the same reason :data:`SearchType` is one: the caller can typecheck against it, and
these three are fixed by Atlas rather than by us. The runtime check in ``_indexes.vector_mode`` stays,
because a config assembled from environment variables or a settings file never passes through a type
checker at all.
"""

ResolvedSearchType = Literal["vector", "text", "hybrid"]
"""Search tiers that can answer a query, reported in the ``_search_type`` metadata key.

Separate from :data:`SearchType` because asking for a tier and reporting one are different domains:
``"auto"`` is requestable and never an answer, and the answer set widens on a different schedule as
tiers land. Both stay narrow for the same reason -- a caller matching on this key should not have to
handle a branch that cannot occur.

The two widened by different amounts when ``"hybrid"`` landed, three-to-four here and two-to-three
there, which is what a single union covering both directions could not have done.
"""

DEFAULT_MAX_SEARCH_RESULTS: Final[int] = 10
"""Result ceiling when neither the call nor the store supplies one.

Matches ``FileMemoryStore`` and ``BedrockKnowledgeBaseStore``, so a caller swapping stores sees the
same result count from the same query.
"""

DEFAULT_OVERSAMPLING_FACTOR: Final[int] = 10
"""Multiplier turning a result limit into a ``$vectorSearch`` candidate count.

Matches ``pymongo_search_utils``' own default, which computes ``numCandidates = top_k *
oversampling_factor``. Deriving the count rather than passing a constant is ``REQ-E-072``; matching
their default means a caller who sets nothing gets the candidate count that library would have used.
"""

RESERVED_METADATA_KEYS: Final[frozenset[str]] = frozenset({"_relevance_score", "_id", "_search_type", "_scope"})
"""Metadata keys this package owns on every returned ``MemoryEntry``.

``MemoryEntry`` has no score field, so relevance, document id, answering tier and scope all travel in
``metadata``. Caller metadata merges underneath these: on a collision the reserved value wins and the
collision is logged, following the ``BedrockKnowledgeBaseStore`` convention rather than
``FileMemoryStore``'s unprefixed keys.
"""


class MongoDBConfig(TypedDict, total=False):
    """Connection and deployment settings, built once per process and shared across stores.

    Constructing an ``AsyncMongoClient`` constructs a connection pool, so a multi-tenant application
    that builds one store per user per request must not build one client per store. Passing the same
    config to every store is what makes them share a pool.

    Attributes:
        database: Database holding this deployment's collections. The one mandatory field.
        client: An existing client to use. Never closed by this package, at any point, including on
            its own ``close()`` -- ownership follows whoever constructed it.
        connection_string: Used only when ``client`` is absent. Supplying neither raises.
        client_args: Extra keyword arguments for the client this package constructs. Ignored when
            ``client`` is supplied, since that client is already configured.
        ensure_indexes: Whether to create missing indexes on ``initialize()``. Defaults to ``True``.
            Verification always runs, so ``False`` means verify-only rather than skip.
        index_ready_timeout: Seconds to wait for a search index to report ``READY``, timed separately
            for each search tier, so an ``auto`` store can wait this long once per tier. Defaults to
            ``_indexes.READY_TIMEOUT_SECONDS``, which is a
            measurement against a development cluster and cannot cover the case this key exists for: a
            shared deployment where a build queues behind somebody else's. Raising it changes
            when a failure arrives, not when a success does.
        search_type: Tier to search with. Defaults to ``"auto"``, which resolves to the best tier the
            deployment supports and warns when it lands below vector.
        embedding_model: Atlas embedding model name for the AutoEmbeddings path, both in the index
            definition and in the query stage. Mutually exclusive with the two manual-vector fields
            below; setting neither selects AutoEmbeddings with ``DEFAULT_EMBEDDING_MODEL``.
        embedding_dimensions: Length of the vectors stored on the documents, for the manual path where
            the caller embeds. Selects that path, and requires ``similarity`` with it. The path CI can
            verify, since an ``autoEmbed`` index is rejected at creation on a deployment with no
            registered model.
        similarity: Distance metric for the manual path: ``"cosine"``, ``"dotProduct"`` or
            ``"euclidean"``. No default, because the right metric is a property of the caller's
            embeddings and the wrong one returns plausible neighbours that are the wrong neighbours.
        filter_fields: Extra document fields to declare filterable in the vector index. A query
            filtering on an undeclared path is rejected by the server, measured as code 8 on mongot
            8.3.8, so these have to be known at index-creation time and adding one later means
            recreating the index.
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
    ``writable`` and ``extraction`` are all accepted here and ``name`` stays mandatory.

    Attributes:
        config: Connection to the deployment. Reuse one across stores that differ only by ``scope``.
        collection: Collection holding this store's documents. Defaults to ``"memory"``.
        scope: Namespace isolating this store's documents. Applied as a search filter and stamped on
            every write by this package, never by the caller. Absence resolves to a distinct sentinel
            scope rather than to "unfiltered".
        filter: Additional search predicate, composed as a narrowing conjunction with the scope
            predicate. Affects ``search`` only, and cannot widen what the store can read.
        min_score: Floor a result must meet. Atlas vector and text scores are similarity-oriented, so
            this is the useful bound of the two. **Not on the same scale as a fused score:** the hybrid
            tier ranks by reciprocal rank fusion, ``sum(weight / (RANK_FUSION_PENALTY + rank))`` with the
            penalty at ``_pipeline.RANK_FUSION_PENALTY``, so a document first in both tiers scores about
            0.033 and one first in a single tier about 0.016. A cosine floor of 0.7 empties the result
            set. Under ``"auto"`` that is why setting either bound keeps the store on the vector tier
            (``REQ-E-145``); asking for ``search_type="hybrid"`` by name applies the bound to the fused
            score instead, because the tier was the caller's own choice (``REQ-E-146``).
        max_score: Ceiling a result must not exceed. Mutually exclusive with ``min_score``. The same
            scale note applies, and bites harder: a ceiling meant for cosine scores admits everything.
            A ``dotProduct`` index inverts the danger instead of blunting it. That metric's score is
            ``(1 + dot) / 2`` with nothing normalising the vectors first, so a corpus vector longer than
            the query scores above 1 -- 2.0, measured, for one five times its length -- and a ceiling of
            1.0 there drops the strongest matches rather than the weakest, with no error to read
            (``REQ-E-183``).
        ttl_seconds: How long a memory lives, as an offset from its own ``created_at``. Absent means no
            expiry and stamps no field; ``0`` means "expire at the next sweep" and is a legitimate
            value, not a disabled one (``REQ-E-148``). Set on insert only, so re-extracting a fact this
            store already holds does not push its expiry out. Unlike ``MongoDBStorage``, this surface
            needs no separate ``ensure_indexes()`` call to make expiry effective: ``initialize()``
            already creates ``strands_expires_at_ttl`` (``REQ-E-149``).
    """

    config: Required[MongoDBConfig]
    collection: str
    scope: str
    filter: dict[str, Any]
    min_score: float
    max_score: float
    ttl_seconds: int


class MongoDBSearchQuery(TypedDict, total=False):
    """A MongoDB-specific query object, widening the ``Storage`` protocol's ``SearchQuery``.

    ``Storage.search`` takes a single argument and its ``SearchQuery`` type parameter is documented as
    widenable to "richer query objects (e.g. a pre-computed embedding vector with metadata filters)",
    so this is a sanctioned extension rather than a departure. SDK-internal callers keep passing a
    plain string.

    Every field is optional because the valid combinations are not expressible: a query needs ``text``
    or ``vector`` and not both, and marking either ``Required`` would forbid the other path. The
    search implementation checks that instead.

    Attributes:
        text: Natural-language query. Embedded server-side on the AutoEmbeddings path.
        vector: A pre-computed embedding, used directly. The path that needs no embedding model
            registered on the deployment, and the one CI can verify.
        filter: MQL predicate over declared filterable fields, validated before it reaches the
            server.
        limit: Result ceiling for this call, overriding the store's own.
        oversampling_factor: Multiplier turning ``limit`` into a candidate count. Raising it trades
            query cost for recall; it is the main way a query becomes expensive by accident, so the
            product is validated against Atlas's bounds.
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
        duplicates: How many were already present and therefore not created. Reported rather than
            hidden: extraction writes are at-least-once, so a silent collapse leaves someone asking
            why their fact count is lower than their write count with no signal to read.
    """

    ids: list[str]
    inserted: int
    duplicates: int
