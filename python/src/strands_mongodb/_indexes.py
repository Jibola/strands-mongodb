"""Index specifications, idempotent ensure and verify, and search-index readiness polling.

Reimplemented against ``AsyncCollection`` rather than calling ``pymongo_search_utils``' index helpers
(``REQ-E-066``). Those helpers are synchronous, poll with a blocking ``time.sleep``, and their
``vector_search_index_definition`` is not even exported, so there is nothing here to reuse.

Three kinds of index and three different idempotency tests, which is the whole reason this is a
module rather than a list of ``create_indexes`` calls:

* **Ordinary** indexes match on the **key list**, then compare every option this package asked for. Not
  on the name, because the server does not treat the name as an identity: creating a second index over
  keys that already carry one is ``OperationFailure`` code 85 whatever it is called. And not on the key
  list alone, because the dedup index exists for its ``unique`` flag rather than for its keys, and
  accepting an ordinary index over the same three fields ships a missing uniqueness guard with every index
  reported present.
* **TTL** indexes are ordinary indexes whose ``expireAfterSeconds`` is one of those options. It stopped
  being a special case once the comparison covered everything.
* **Atlas Search** indexes are listed by name and compared on ``latestDefinition``, then polled, because
  a build is asynchronous: created and queryable are two different states, and a search over an
  unfinished index returns zero documents rather than an error.

Every behaviour above was measured against MongoDB 8.3.8 before it was written, and two of them
contradicted what was first drafted for them. This module is what follows from the measurements, not from
that draft.

Private, like ``_client`` and ``_filter``. The specs are a promise to the operator who provisions them,
not an API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

from strands_mongodb._extras import missing_extra
from strands_mongodb._filter import BAKED_IN_FILTER_FIELDS

# `storage` rather than `memory`, for the same reason `_client` names it: both surfaces reach this
# module and everything it needs is in the narrower extra. `StorageError` is in `strands`, which both
# extras install, so one guard covers the pair.
try:
    from pymongo import IndexModel
    from pymongo.errors import OperationFailure, PyMongoError
    from pymongo.operations import SearchIndexModel
    from strands.types.exceptions import StorageError
except ImportError as error:  # pragma: no cover - exercised in a subprocess by AT-008b
    missing_extra("storage", error, module=__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymongo.asynchronous.collection import AsyncCollection

    from strands_mongodb.types import MongoDBConfig

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "IndexSpec",
    "POLL_INTERVAL_SECONDS",
    "READY_TIMEOUT_SECONDS",
    "SEARCH_UNSUPPORTED_CODES",
    "SearchIndexSpec",
    "SearchSupport",
    "VectorMode",
    "declared_filter_fields",
    "drop_index_command",
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
"""Document field the TTL indexes watch. Absent on documents with no expiry, which is why the TTL
index can be created unconditionally: it is inert for a document that does not carry this field."""

DEFAULT_EMBEDDING_MODEL: Final[str] = "voyage-4-lite"
"""Atlas embedding model used when the config names none.

Taken from the community ``strands-mongodb-store`` 0.2.0, and confirmed against it, rather than
chosen here, so a caller moving between the two packages gets the same vectors from the same text.
"""

DEFAULT_VECTOR_INDEX_NAME: Final[str] = "vector_index"
"""Default Atlas Vector Search index name, from langchain-mongodb ``vectorstores.py:219``."""

DEFAULT_TEXT_INDEX_NAME: Final[str] = "text_index"
"""Default Atlas Search index name, from langchain-mongodb ``retrievers/parent_document.py:218``."""

SIMILARITIES: Final[tuple[str, ...]] = ("cosine", "dotProduct", "euclidean")
"""Distance metrics an Atlas vector index accepts. Mirrors ``types.VectorSimilarity`` at runtime."""

MAX_DIMENSIONS: Final[int] = 8192
"""Upper bound Atlas enforces on ``numDimensions``. Measured: outside ``[1..8192]`` is code 8."""

READY_TIMEOUT_SECONDS: Final[float] = 300.0
"""How long a search-index build is waited on, before a caller overrides it (``REQ-E-123``).

Was ``pymongo_search_utils``' own ``TIMEOUT`` of 120, on the reasoning that
reimplementing their polling against the async driver (``REQ-E-066``) is no reason to change how long a
caller waits. Two measurements against cloud Atlas retired that: two index builds in a new database on
an idle ``strands-dev`` took **104.7s** on 2026-09-02, and a single build on the same cluster with the
three-tier suite's collections beside it ran past **120s** on 2026-09-15, failing the README's own
AutoEmbeddings quickstart. A budget inside the observed range is wrong in both directions, so the number
is now a measurement with room rather than an inheritance.

Raising it changes when a *failure* arrives and not when a success does: the poll interval decides the
latter and is untouched. ``MongoDBConfig["index_ready_timeout"]`` is the per-deployment override, for
the case no default covers -- a shared cluster where a build queues behind somebody else's.
"""

POLL_INTERVAL_SECONDS: Final[float] = 0.5
"""Delay between readiness reads. ``pymongo_search_utils``' own ``INTERVAL``."""

_READY_STATUS: Final[str] = "READY"
_FAILED_STATUS: Final[str] = "FAILED"
_NAMESPACE_EXISTS: Final[int] = 48

RANK_FUSION_MIN_MAJOR: Final[int] = 8
"""Major version of ``mongod`` from which ``$rankFusion`` exists (``REQ-E-143``).

Eight, not 8.1. The MongoDB 8.0 manual carries the stage with its own compatibility note and a worked
example putting ``$vectorSearch`` in an input pipeline; 8.2 is where ``$scoreFusion`` arrives and 8.3
where ``$rerank`` does. Atlas hands out 8.0.23 by default, so an off-by-one-minor floor here would have
switched hybrid search off for most of the deployments it was written for.

A major-version integer and not a full version tuple because the stage has not changed since it landed:
there is no 8.0.x patch to exclude, and a finer floor would be a claim this package cannot support.
"""

SEARCH_UNSUPPORTED_CODES: Final[frozenset[int]] = frozenset({31082, 115, 6047401})
"""Server error codes meaning "this deployment has no Atlas Search", as ``list_search_indexes()`` reports.

A set rather than one code, because four deployments answer this probe with three different numbers
(``REQ-E-053a`` holds the table). Measured 2026-09-02, on the ``list_search_indexes()`` surface:

* **31082** on ``mongo:8``, reported version 8.3.8 with ``modules: []``. Every search surface on that
  deployment answers with this one, which is why it was long the only member here.
* **115**, ``CommandNotSupported``, on ``mongod`` 7.0.12 with ``modules: ['enterprise']``. That build
  carries mongot support and has none configured, so its query stages answer 31082 while its
  search-index commands answer this.
* **6047401** on ``mongo:6`` 6.0.28 and ``mongo:7`` 7.0.40, both ``modules: []``, *"$listSearchIndexes
  stage is only allowed on MongoDB Atlas"*.

Ordinary index creation and every CRUD path work normally on all four. Two things the readings settle.
The message is not a candidate key, since it differs between the four deployments and between surfaces
on one of them, and neither is ``codeName``: the same numeric 31082 is ``SearchNotEnabled`` on 8.3.8 and
``Location31082`` on 7.0.12. The number is the stable part.

Code 59, ``CommandNotFound``, means the same thing at ``create_search_index`` on 6.0.28 and 7.0.40, and
is deliberately absent. That call runs only after :func:`search_support` has cleared the deployment, so
a member matching it would be unreachable. ``AT-053d`` asserts the exclusion so it stays a decision.
"""

_COLLECTION_ABSENT: Final[int] = 26
"""Server error code for a namespace that does not exist. Not an error when reading indexes."""

SIMPLE_LOCALE: Final[str] = "simple"
"""The one collation locale that compares strings byte by byte, which is what a key space needs.

Every other locale is a language-aware comparison, and every language-aware comparison is a comparison
this package's ``_id`` key model cannot be built on: see :func:`require_binary_comparison`. Public
because ``AT-176`` asserts against it and the refusal message names it.
"""

Readiness = Literal["ready", "building", "failed"]
"""The three states ``REQ-E-062`` distinguishes, because the remedy differs between them."""


class SearchSupport(NamedTuple):
    """What :func:`search_support` found, which is an answer and the evidence for it.

    Two fields rather than a ``bool`` because ``REQ-E-053a`` asks the warning and the later query failure
    to name *the server's own code*, and the only place that code exists is inside the probe. Returning
    it means an operator on 7.0.40 reads 6047401 in the message and can match it against their own log,
    where a fixed 31082 sent them looking for an error their server never emitted.

    Attributes:
        available: Whether the deployment runs Atlas Search.
        code: The server error code that said it does not, or ``None``. Set only when ``available`` is
            ``False``, which is the invariant every reader of this record may assume.
    """

    available: bool
    code: int | None = None


def readiness(status: str) -> Readiness:
    """Map a search index's reported ``status`` onto the state whose remedy the caller needs.

    Written as a default rather than as a lookup, and that is the finding rather than a style
    preference. Immediately after creating one vector and one full-text index on 8.3.8, the vector
    index reported ``BUILDING`` and the full-text index reported ``PENDING``. ``PENDING`` was not in the
    design's list of statuses, so a lookup table built from that list would have had no entry for the
    state half the indexes were actually in.

    An unrecognised status read as *building* costs one poll interval. Read as *ready* it ships
    a real failure: the search runs against an unfinished index, returns zero documents, and the
    user reads it as the agent having forgotten. The server documents no full status vocabulary.

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
        """Whether the deployment embeds, rather than the caller.

        Returns:
            ``True`` when a model is set, which is the only way the server-side path is selected.
        """
        return self.model is not None


def vector_mode(config: MongoDBConfig) -> VectorMode:
    """Resolve the vector path from the config, rejecting the combinations that mean two things.

    ``REQ-E-067``. The exactly-one-mode rule mirrors ``pymongo_search_utils._check_param_config``
    without importing it, since that function lives in the ``[memory]`` extra and reads its own
    parameter names.

    Both manual fields are checked here rather than left to Atlas, even though Atlas rejects a bad
    metric and an out-of-range dimension count too. Its rejection arrives at index creation, after the
    collection and the other five indexes exist, as a code 8 naming a JSON path in a command the caller
    never wrote. Rejecting the same values while they are still a dict costs two comparisons.

    Args:
        config: The connection configuration. Only the four vector-related keys are read.

    Returns:
        The resolved mode, with exactly one path populated. An empty config resolves to AutoEmbeddings
        with :data:`DEFAULT_EMBEDDING_MODEL`, which keeps the zero-configuration path the Atlas one.

    Raises:
        ValueError: If both paths are configured, if only one of the manual pair is set, or if a manual
            value is outside what Atlas accepts. Names the fields in conflict, because the caller wrote
            a dict and needs to know which key to remove.
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

    ``REQ-E-071``, and the reason it is readable rather than only writable: ``_filter.validate`` has to
    reject a filter over an undeclared path, and the index definition is the single authority
    for what is declared. Two modules computing the same list separately is how they drift apart.

    Args:
        extra: The config's ``filter_fields``. Duplicates of the baked-in pair are dropped, since Atlas
            rejects a definition with two ``filter`` entries on one path and a caller who lists
            ``scope`` there is asking for something already true.

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
    """Build the Atlas Vector Search definition the ``memory`` collection needs.

    The AutoEmbeddings variant uses ``type: "autoEmbed"`` over the text field, not ``type: "vector"``
    with ``numDimensions: -1``. The latter is langchain-mongodb's constructor
    validation rather than a wire format, and the server answers ``-1`` with code 8.

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
    """Build the Atlas Search full-text definition the ``memory`` collection needs.

    ``dynamic: False``, so only the text field is indexed. Dynamic mapping would index every field of
    every document, including whatever metadata a caller attached to a memory, which is storage and
    build time spent on fields no query in this package reads.

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
        name: Name used when this package creates the index. Never used to *find* one, since the server
            does not treat the name as an identity.
        keys: The key list, in order. This is the match key.
        options: Every option this package asks for, and therefore every option compared against an
            existing index. Options nobody here set are nobody here's business: a ``background`` or
            ``hidden`` flag another tool added is not a mismatch.
    """

    name: str
    keys: tuple[tuple[str, int], ...]
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_model(self) -> IndexModel:
        """Build the driver object that creates this index.

        Returns:
            An ``IndexModel`` carrying the keys, the name and every option.
        """
        return IndexModel(list(self.keys), name=self.name, **dict(self.options))

    def creation_command(self, collection_name: str) -> str:
        """Render the server command that creates this index (``REQ-E-061``).

        The raw ``createIndexes`` command rather than a shell helper, because that is the command
        pymongo itself sends and so it can be checked from this repository. A helper's argument order is
        a detail this package has no way to verify, and getting it wrong hands an operator something
        that fails in their hands.

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

    Takes the name found on the deployment rather than reading one off a spec, because the index in the
    way is the one somebody else created and it is named whatever they named it.

    Rendered rather than run. Dropping an index this package did not create is destructive, it is not
    recoverable from here, and the index may be serving queries this package knows nothing about.

    Args:
        collection_name: Collection the index sits on.
        index_name: Name the deployment reported for it.

    Returns:
        A single-line ``db.runCommand({...})``.
    """
    return f'db.runCommand({{"dropIndexes": "{collection_name}", "index": "{index_name}"}})'


@dataclass(frozen=True)
class SearchIndexSpec:
    """One Atlas Search index: ``vectorSearch`` or ``search``.

    Attributes:
        name: The index name, which here *is* the identity. ``list_search_indexes`` takes a name and
            ``createSearchIndexes`` rejects a second definition under an existing one with code 68.
        definition: The definition, compared verbatim against ``latestDefinition``.
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
        """Render the ``createSearchIndexes`` command that creates this index (``REQ-E-061``).

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
            A single-line ``db.runCommand({...})``.
        """
        return f'db.runCommand({{"dropSearchIndex": "{collection_name}", "name": "{self.name}"}})'


def storage_specs() -> tuple[IndexSpec, ...]:
    """Return the ordinary indexes the ``storage`` collection needs.

    Row 1 is ``_id_``, which the server creates and which is the reason key-as-``_id`` was chosen in the
    first place. That leaves the TTL index.

    Returns:
        One spec. Takes no arguments, so every store asks for the same list and the command block
        ``REQ-E-061`` hands an operator is the block that satisfies the next verify.
    """
    return (
        IndexSpec(
            name="strands_expires_at_ttl",
            keys=((EXPIRES_FIELD, 1),),
            options={"expireAfterSeconds": 0},
        ),
    )


def memory_specs() -> tuple[IndexSpec, ...]:
    """Return the ordinary indexes the ``memory`` collection needs.

    The expiry index is optional and is created anyway. Making it conditional on a
    configured TTL is the obvious implementation and it fails silently: a store that gains an expiry
    after its first ``initialize()`` stamps ``expires_at`` on every document and nothing ever deletes
    them, with no error to read and no way to notice except watching the collection grow. Unconditional
    fails loudly instead, as one more index in the block an operator provisions, and it costs nothing at
    rest because a TTL index is inert for documents that carry no ``expires_at``.

    **That argument described a situation that could not arise until this package added**
    ``ttl_seconds`` to ``MongoDBMemoryStoreConfig``: before it, nothing on this surface wrote
    ``expires_at`` at all, and this index was inert on every document in every collection this package
    had ever created. It is now load-bearing, and the unconditional choice is what makes ``REQ-E-149``
    true -- a store that gains a ``ttl_seconds`` expires its existing documents on the next sweep, with
    no migration, because the index was there the whole time.

    Returns:
        Three specs: the unique partial dedup index, the recency index, and the TTL index.
    """
    return (
        IndexSpec(
            name="strands_scope_store_content_hash_unique",
            keys=(("scope", 1), ("store_name", 1), ("content_hash", 1)),
            options={
                "unique": True,
                # Partial, because a unique index treats a missing field as `null`: without this, the
                # second document with no `content_hash` collides with the first.
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
        ),
    )


def memory_search_specs(config: MongoDBConfig) -> tuple[SearchIndexSpec, ...]:
    """Return the Atlas Search indexes the ``memory`` collection needs.

    Config-driven where the ordinary specs are not, because both names and the whole vector definition
    come from it. This is also the path ``initialize()`` takes, so the mode conflict in ``REQ-E-067``
    has to be rejected from here and not only when :func:`vector_mode` is called directly.

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

    ``REQ-E-053a``. Asked before :func:`ensure` is given a ``search`` set, so a deployment without
    mongot gets its ordinary indexes and a warning instead of a failure. The alternative, letting
    ``createSearchIndexes`` fail and treating a member of :data:`SEARCH_UNSUPPORTED_CODES` as benign
    wherever it turns up, would also swallow the code on a deployment that has mongot and is
    misconfigured, which is ``REQ-E-061``'s condition and has a different remedy.

    ``list_search_indexes`` is the probe rather than a ``buildInfo`` read of ``modules``, because what
    the caller needs to know is whether the *call* works. ``mongo:8`` reports version 8.3.8 with an
    empty ``modules`` list and no search, an Atlas CLI local deployment reports the same empty list and
    has search, so the version document cannot distinguish them. The reading that settles it is
    ``mongod`` 7.0.12, which reports ``modules: ['enterprise']`` and has no search: on the four measured
    deployments the module list does not merely fail to separate them, it points the wrong way.

    Named for what it returns rather than ``search_available``, which it replaced. The old
    name invited ``if await search_available(...)``, and a two-field record is always truthy, so keeping
    it would have turned every caller into a silent always-has-search bug.

    Args:
        collection: The collection whose deployment to probe. Need not exist.

    Returns:
        :class:`SearchSupport`. ``available`` is ``True`` if the deployment answers the call, ``False``
        if it answers with one of :data:`SEARCH_UNSUPPORTED_CODES`, in which case ``code`` carries the
        one it used.

    Raises:
        StorageError: If the call fails for any other reason, including authentication. A probe that
            reported ``False`` on a bad password would send the caller looking for the wrong problem.
    """
    try:
        cursor = await collection.list_search_indexes()
        async for _ in cursor:
            break
    except OperationFailure as error:
        if error.code in SEARCH_UNSUPPORTED_CODES:
            return SearchSupport(available=False, code=error.code)
        # An absent collection has no indexes, which answers the question asked here: the call worked.
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


async def default_collation(collection: AsyncCollection[Any]) -> dict[str, Any] | None:
    """Report the collection's default collation, or ``None`` when it has none.

    ``listCollections`` filtered to one name, which is the only way to ask: a collation is fixed at
    creation and lives in the collection's options, and nothing on ``AsyncCollection`` carries it. The
    filter is not an optimisation -- an unfiltered call reads every collection in the database to answer
    a question about one, and on a database an application shares that is a listing it has no business
    producing.

    Args:
        collection: The collection to ask about. Need not exist.

    Returns:
        The collation document the server reports, or ``None`` if the collection has no collation or does
        not exist yet. The two are the same answer here on purpose: a collection this package creates
        implicitly is created without one.

        Measured on 8.0.32, because the shape is the whole contract of this function and it is not
        derivable from the documentation: a collection created with ``{locale: "en", strength: 2}`` is
        reported with the resolved ten-field document under ``options.collation``, and a collection
        created with ``{locale: "simple"}`` is reported with ``options`` **empty**. The server does not
        store a simple collation, so on that version the only collation this ever returns is one worth
        refusing. :func:`require_binary_comparison` still admits ``simple`` explicitly, for a server that
        reports what it was asked for.

    Raises:
        StorageError: If the call fails. Unlike :func:`search_support` there is no code to forgive: an
            absent collection is an empty result rather than an error, so anything that arrives here is
            a real failure and reporting the collection as uncollated would be reporting a guess.
    """
    try:
        cursor = await collection.database.list_collections(filter={"name": collection.name})
        async for document in cursor:
            options = document.get("options") or {}
            collation = options.get("collation")
            return dict(collation) if collation is not None else None
    except PyMongoError as error:
        raise StorageError(
            f"strands_mongodb: reading the collation of {collection.database.name}.{collection.name} failed: {error}"
        ) from error
    return None


async def require_binary_comparison(collection: AsyncCollection[Any]) -> None:
    """Refuse a collection whose default collation is not binary (``REQ-E-170``).

    A collation is a property of the collection, and every query against that collection inherits it
    unless it names another. This package's key space is one document per key with the key in ``_id``, so
    equality on a key, the range bounds behind a prefix scan, and the uniqueness of ``_id_`` are all
    string comparisons it does not choose. Under a case-folding collation a namespaced read matches
    another namespace's document, a prefix delete removes it, and two keys that differ only in case stop
    being two keys -- so the refusal here is about isolation and about the key model, not about ordering.

    Overriding per query was the alternative and it is worse. ``collation={"locale": "simple"}`` on each
    call restores binary matching and makes the ``_id`` index unusable for it, turning the two operations
    the key layout exists to keep on that index -- ``list`` and ``delete_prefix`` -- into collection
    scans; and ``_id_`` is the one index whose collation cannot be chosen, so the uniqueness would stay
    folded and a legitimate second key would arrive as a duplicate. A collection like this cannot host
    this key space, and saying so once is more use than working incorrectly in a way that reads as
    working.

    Args:
        collection: The collection about to be operated on.

    Raises:
        StorageError: If the collection has a collation whose locale is not
            :data:`SIMPLE_LOCALE`, naming the locale found and the remedy.
    """
    collation = await default_collation(collection)
    if collation is None or collation.get("locale") == SIMPLE_LOCALE:
        return
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
    """Report whether this deployment can run ``$rankFusion`` (``REQ-E-143``).

    A version read, which is the probe :func:`search_support` spends a paragraph refusing. The two are
    asking about different processes and the difference is the whole justification. Atlas Search is
    **mongot**, attached or not attached independently of ``mongod``'s version: ``mongo:8`` reports
    8.3.8 with no search, an Atlas CLI local deployment reports the same empty ``modules`` list and has
    search, and ``mongod`` 7.0.12 reports ``modules: ['enterprise']`` and has none -- on the four
    measured deployments the version document points the wrong way. ``$rankFusion`` is an aggregation
    stage inside ``mongod`` itself, so that same document is authoritative for it.

    The floor is major version 8. Not 8.1: the MongoDB 8.0 manual gives 8.0 in the stage's own
    compatibility note and lists ``$vectorSearch`` among the stages an input pipeline may contain, with
    a worked example. ``$scoreFusion`` is the one that needs 8.2, and ``$rerank`` needs 8.3 plus Native
    Reranking enabled in Atlas project settings plus a Voyage AI key; neither is used here.

    ``versionArray`` rather than the ``version`` string, because the server computes it as integers and
    the string carries suffixes like ``-rc1`` whose parsing this package would otherwise own.

    Every failure reads as ``False``, which is the opposite of :func:`search_support`'s closed-set raise
    and is the right asymmetry. That probe's answer decides whether search runs at all, so a wrong
    password must not read as a missing cluster. This one only decides which of two working tiers is
    preferred: degrading costs the caller the better ranking, while guessing ``True`` costs them a query
    that fails on every search. Only one of the two is recoverable, and a deployment that will not
    answer ``buildInfo`` has a problem the next call surfaces anyway.

    Args:
        collection: Any collection on the deployment to probe. Need not exist -- ``buildInfo`` is a
            database command and does not read it.

    Returns:
        ``True`` if the server reports major version :data:`RANK_FUSION_MIN_MAJOR` or later.
    """
    try:
        build_info = await collection.database.command("buildInfo")
    except PyMongoError:
        return False
    version = build_info.get("versionArray") if isinstance(build_info, Mapping) else None
    if not isinstance(version, Sequence) or isinstance(version, str) or not version:
        return False
    major = version[0]
    # `bool` is a subclass of `int` and `True >= 8` is `False`, so it would answer correctly by
    # accident. Excluded anyway: a server sending a boolean here is not a server whose version is known.
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

    ``REQ-E-060`` through ``REQ-E-064``, ``REQ-E-066`` and ``REQ-E-068``. Safe to call on every
    ``initialize()``: the warm path is one read per ordinary set plus one per search index, and no
    writes.

    ``create=False`` is verify-only rather than skip (``REQ-E-061``). A caller whose credentials have no
    DDL rights still needs to know whether the indexes are there, because a missing vector index does
    not fail a search, it returns nothing. Verification still waits out a build somebody else started,
    since *building* is a state whose remedy is patience rather than a command.

    Nothing here drops or alters an existing index. Every conflict is reported with the command that
    resolves it, and the decision stays with whoever knows what else that index serves.

    Args:
        collection: The collection to index. Its ``database`` is used for the one collection creation
            the search path needs.
        ordinary: Ordinary and TTL indexes, from :func:`storage_specs` or :func:`memory_specs`.
        search: Atlas Search indexes, from :func:`memory_search_specs`.
        create: Whether missing indexes may be created.
        timeout: Seconds to wait for search builds, in total across all of them.
        interval: Seconds between readiness reads.

    Raises:
        StorageError: If an existing index conflicts with a declared one, if a search build has failed,
            if a build does not finish inside ``timeout``, if something is missing and ``create`` is
            ``False``, or if the driver refuses a call. Driver errors arrive chained, so the server's
            own code and message survive.
    """
    if ordinary:
        await _ensure_ordinary(collection, tuple(ordinary), create=create)
    if search:
        await _ensure_search(collection, tuple(search), create=create, timeout=timeout, interval=interval)


async def _ensure_ordinary(collection: AsyncCollection[Any], specs: tuple[IndexSpec, ...], *, create: bool) -> None:
    """Create or verify the ordinary indexes, matching on the key list (``REQ-E-063``)."""
    namespace = f"{collection.database.name}.{collection.name}"
    present = await _guard(collection.index_information(), what=f"reading indexes on {namespace}")
    by_keys = {tuple(tuple(pair) for pair in found["key"]): (name, found) for name, found in present.items()}

    missing: list[IndexSpec] = []
    for spec in specs:
        match = by_keys.get(spec.keys)
        if match is None:
            missing.append(spec)
            continue
        _check_options(collection.name, namespace, spec, *match)

    if not missing:
        return
    if not create:
        raise StorageError(_missing_ordinary_message(collection.name, namespace, missing))

    logger.info(
        "strands_mongodb: creating %d index(es) on %s: %s",
        len(missing),
        namespace,
        ", ".join(spec.name for spec in missing),
    )
    # One call for all of them. Three round trips where one would do is three chances for a partial
    # result to leave the collection half indexed, and the server tolerates a concurrent identical
    # create anyway (measured: 10 at once, zero failures, one index).
    await _guard(
        collection.create_indexes([spec.to_model() for spec in missing]),
        what=f"creating indexes on {namespace}",
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
    server or another tool added are not ours to have an opinion about, and treating them as mismatches
    would make every index on a real deployment look wrong.
    """
    for option, wanted in spec.options.items():
        if option in found and found[option] == wanted:
            continue
        actual = f"{found[option]!r}" if option in found else "absent"
        raise StorageError(
            f"strands_mongodb: the index {found_name!r} on {namespace} covers the keys "
            f"{list(spec.keys)} that {spec.name!r} needs, but its {option} is {actual} and this "
            f"package needs {wanted!r}. Creating a second index over the same keys is rejected by the "
            f"server whatever it is named, and dropping this one is not this package's decision to "
            f"make: it may be serving queries nothing here knows about. To resolve it, drop the "
            f"existing index and let the next call recreate it:\n"
            f"{drop_index_command(collection_name, found_name)}\n"
            f"{spec.creation_command(collection_name)}\n"
        )


def _missing_ordinary_message(collection_name: str, namespace: str, missing: Sequence[IndexSpec]) -> str:
    """Build the verify-only message, which is a work order rather than a complaint (``REQ-E-061``).

    Every command goes on its own line, all of them in one message. A caller who has to raise one ticket
    per index raises three tickets, and the whole list is computable in a single pass.
    """
    commands = "\n".join(spec.creation_command(collection_name) for spec in missing)
    return (
        f"strands_mongodb: {len(missing)} required index(es) are absent from {namespace} and "
        f"ensure_indexes is False, so nothing was created. Run these against the database as somebody "
        f"holding the rights to, then call initialize() again:\n"
        f"{commands}\n"
        f"Absent: {', '.join(spec.name for spec in missing)}."
    )


async def _ensure_search(
    collection: AsyncCollection[Any],
    specs: tuple[SearchIndexSpec, ...],
    *,
    create: bool,
    timeout: float,
    interval: float,
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
        # Before the index, not after. `createSearchIndexes` against a missing namespace is code 26,
        # while `createIndexes` in the same position creates the collection implicitly (REQ-E-068).
        await _create_collection(collection, namespace)
        for spec in missing:
            logger.info("strands_mongodb: creating %s index %r on %s", spec.type, spec.name, namespace)
            await _guard(
                collection.create_search_index(spec.to_model()),
                what=f"creating search index {spec.name!r} on {namespace}",
            )
        waiting.extend(missing)

    if waiting:
        await _wait_until_ready(collection, waiting, namespace, timeout=timeout, interval=interval)


async def _read_search_index(collection: AsyncCollection[Any], name: str, namespace: str) -> dict[str, Any] | None:
    """Read one search index by name, or ``None`` if it is not there.

    ``list_search_indexes`` is itself a coroutine returning an async cursor, so the cursor arrives from
    the ``await`` rather than from iterating the call. Pinned by ``MP-002``, because getting it the other
    way round is a ``TypeError`` on a deployment and nothing at all against a loose fake.
    """
    try:
        cursor = await collection.list_search_indexes(name)
        async for document in cursor:
            return dict(document)
    except OperationFailure as error:
        # A collection that does not exist has no indexes, which is the answer rather than an error.
        # `list_indexes` already normalises this for ordinary indexes; the aggregate behind
        # `$listSearchIndexes` is not documented to, so it is normalised here.
        if error.code != _COLLECTION_ABSENT:
            raise StorageError(
                f"strands_mongodb: reading search index {name!r} on {namespace} failed: {error}"
            ) from error
    except PyMongoError as error:
        raise StorageError(f"strands_mongodb: reading search index {name!r} on {namespace} failed: {error}") from error
    return None


def _check_definition(collection_name: str, namespace: str, spec: SearchIndexSpec, document: Mapping[str, Any]) -> None:
    """Check that an existing search index's definition covers the declared one (``REQ-E-069``).

    Re-issuing a *different* definition under an existing name is code 68, *"already exists with a
    different definition. Drop it first if needed."*, so this reports the same thing before the server
    has to, and reports it on the warm path where nothing is re-issued at all.
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

    Containment rather than equality, because equality is not available (``REQ-E-069``).
    ``latestDefinition`` round-trips byte for byte for a ``vectorSearch`` index and is rewritten for a
    ``search`` one: mongot 8.3.8 collapses the one-element list in ``mappings.fields.<field>`` into the
    object it held and fills in ``indexOptions``, ``store`` and ``norms``. Comparing for equality
    therefore passes on the vector index and fails on the text index that this package created itself
    one call earlier.

    Three rules, in the order they are applied:

    * A declared object is covered when every key it declares is present and covered. Keys the server or
      another tool added are ignored, which is the rule ``REQ-E-063`` already applies to ordinary index
      options and the only rule under which an index somebody else wrote can be judged at all.
    * A declared one-element list is covered by the object the server reports in its place, which is the
      rewrite above.
    * A declared list is covered when **something** in the reported list covers each of its entries,
      order-insensitively. Order is in fact preserved for the vector definition, but requiring it would
      report a mismatch over an index that is functionally identical, and extra fields are somebody
      else's business.

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
    """Build the message for a build that failed (``REQ-E-062``).

    Reported rather than recreated. An identical definition issued over a failed build is accepted and
    does not restart it, so a create here would burn the whole timeout to learn what the status already
    said.

    ``statusDetail`` is rendered whole rather than picked apart. Its shape is per-host and per-index and
    is not something this package can claim to know; the reader needs the server's words.
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
    """Build the verify-only message for absent search indexes (``REQ-E-061``)."""
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

    ``check_exists=False`` removes a check-then-act race and a round trip. It also removes
    ``CollectionInvalid`` from the picture: pymongo raises that only from the client-side branch this
    argument disables, so a genuine race arrives as code 48 instead. On 8.3.8 the second create is
    accepted outright and 48 never arrives, so the handler below is defense for other server versions.
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
) -> None:
    """Poll until every index reports READY, or until ``timeout`` (``REQ-E-062``, ``REQ-E-066``).

    ``asyncio.sleep`` rather than ``time.sleep``. That single substitution is most of what
    ``REQ-E-066`` is about: the reference helpers block, and blocking here stalls every other coroutine
    in an agent process for as long as the build takes.

    Transitions are logged as they happen. A two-minute wait with no output reads as a hang, and the
    status is the only thing that distinguishes a slow build from a stuck one.

    Readiness is a claim about two things, so both are checked before an index leaves ``pending``: the
    status the server reports, and the definition it reports alongside it (``REQ-E-175``). See the
    comment on the ``ready`` branch for what drifts and how.
    """
    deadline = time.monotonic() + timeout
    pending = {spec.name: spec for spec in specs}
    last: dict[str, str] = {}

    while True:
        for name in list(pending):
            document = await _read_search_index(collection, name, namespace)
            # An index that has vanished reads as *building*, through the same default that catches an
            # unrecognised status. It was there on the previous read, so something else dropped it, and
            # waiting is the only move left: reporting ready would be false, and recreating it would
            # race whoever is holding the other end of that drop.
            status = "absent" if document is None else str(document.get("status", "unreported"))
            if status != last.get(name):
                logger.info("strands_mongodb: search index %r on %s is %s", name, namespace, status)
                last[name] = status
            state = readiness(status)
            if state == "failed":
                # Reachable only for the literal FAILED status, which cannot come from a missing
                # document, so the fallback below is for the type checker rather than for a caller.
                raise StorageError(_failed_message(collection.name, namespace, pending[name], document or {}))
            if state == "ready":
                # Rechecked here, and not only on the cold read before the poll began
                # (``REQ-E-175``). The status is not the whole answer: a definition that was right when
                # the wait started can be wrong when it ends, because a search index cannot be redefined
                # in place but it can be dropped and recreated, and mongot will happily build the
                # replacement under the same name while this loop watches the name rather than the
                # definition. Two ways in, both measured against the fake:
                #
                # * an index found BUILDING with a covering definition, dropped and recreated by another
                #   actor mid-build, reporting READY with a definition that no longer covers this
                #   package's;
                # * an index this package created itself one call earlier, whose build was never checked
                #   against what was asked for at all -- the cold read found nothing, so
                #   ``_ensure_search`` had no document to check and never got another.
                #
                # Accepting either ships premortem **P-2** with the loudest symptom removed: searches run,
                # return nothing or the wrong thing, and every index reports READY. One extra containment
                # check over a document already in hand is the whole cost; there is no additional read.
                _check_definition(collection.name, namespace, pending[name], document or {})
                del pending[name]

        if not pending:
            return
        if time.monotonic() >= deadline:
            stuck = ", ".join(f"{name} is {last[name]}" for name in pending)
            raise StorageError(
                f"strands_mongodb: waited {timeout}s for {len(pending)} search index(es) on "
                f"{namespace} to report READY and they did not: {stuck}. BUILDING at this point means "
                f"a build slower than the timeout, and initialize() can simply be called again to keep "
                f"waiting. Any other status is worth reading directly with "
                f"db.{collection.name}.getSearchIndexes()."
            )
        await asyncio.sleep(interval)


async def _guard(awaitable: Any, *, what: str) -> Any:
    """Await something and turn a driver error into the exception a caller catches.

    ``StorageError`` is what the ``Storage`` protocol documents, and a caller should not have to import
    pymongo to handle a failure from a package they configured with a dict. The original is chained, so
    the server's code and message stay reachable through ``__cause__``.
    """
    try:
        return await awaitable
    except PyMongoError as error:
        raise StorageError(f"strands_mongodb: {what} failed: {error}") from error
