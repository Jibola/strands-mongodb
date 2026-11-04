"""Aggregation stages for the vector, text and hybrid search tiers.

Pure functions over plain dicts: no client and no I/O, so every shape is testable without a deployment.
That matters because the mistakes this module prevents are shape mistakes a live query can answer with
wrong results rather than an error.

Every search stage comes from ``pymongo_search_utils``' builders. This module adds what they leave to the
caller:

* **One return type.** Every function returns ``list[dict]``, so the tiers compose the same way.
* **Which builder, and which path.** Both follow ``VectorMode``. The ``autoEmbed`` index declares the
  text field as its vector field, so the AutoEmbeddings query paths at ``content`` and the manual query
  paths at ``embedding``. Crossing them fails with code 8, but only after a round trip.
* **The score projection.** The vector builders emit no score. The three ``$meta`` names (vector, text,
  fused) are not interchangeable.
* **The hybrid tier.** :func:`hybrid_stages` wraps the vector and text pipelines in one server-side
  ``$rankFusion`` (MongoDB 8.0+). The library's client-side fusion builders are not used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from strands_mongodb._extras import missing_extra
from strands_mongodb._indexes import CONTENT_FIELD, MAX_NUM_CANDIDATES, VECTOR_FIELD
from strands_mongodb.types import DEFAULT_OVERSAMPLING_FACTOR

try:
    from pymongo_search_utils import (
        autoembedding_vector_search_stage,
        text_search_stage,
        vector_search_stage,
    )
except ImportError as error:  # pragma: no cover - exercised in the extras tier
    missing_extra("memory", error, module=__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from strands_mongodb._indexes import VectorMode

__all__ = [
    "FUSION_SCORE_META",
    "MAX_NUM_CANDIDATES",
    "RANK_FUSION_PENALTY",
    "candidate_count",
    "check_candidate_count",
    "check_query_keys",
    "hybrid_stages",
    "text_stages",
    "vector_stages",
]

RANK_FUSION_PENALTY: Final[int] = 60
"""Rank constant the server applies inside ``$rankFusion``. This package never sends it.

It sets the scale of every hybrid score. A fused score is the sum, over the tiers that ranked a document,
of ``weight / (rank + 60)``. A document first in one tier scores about ``1/61`` (0.016) and one first in
both about 0.033, while cosine similarity and BM25 land roughly between 0 and 1. A ``min_score`` chosen
for those scales empties every hybrid result. The config docstring and the README quote this constant.
"""

VECTOR_SCORE_META: Final[str] = "vectorSearchScore"
"""``$meta`` name for a ``$vectorSearch`` score. Not interchangeable with the other two."""

FUSION_SCORE_META: Final[str] = "score"
"""``$meta`` name for a ``$rankFusion`` score. Not interchangeable with the other two.

``{"$meta": "score"}`` is metadata, not the ``"$score"`` field. ``$rankFusion`` returns documents with no
score field, so without the ``$set`` appended after it every hybrid result would score ``0.0`` and any
``min_score`` would silently empty the result list. Available from MongoDB 8.0, the same floor as the
stage (:data:`_indexes.RANK_FUSION_MIN_MAJOR`).
"""

TEXT_SCORE_META: Final[str] = "searchScore"
"""``$meta`` name for a ``$search`` score. Not interchangeable with the other two.

Crossing it with the vector name is not reliably an error. On mongot 8.3.8, asking for it on a
``$vectorSearch`` pipeline fails with *"query requires $search score metadata, but it is not available"*.
On Atlas 8.0.30 the same pipeline succeeds and returns no score field, which ``memory_store`` reads as
``0.0``. So these constants are the guard, not the server.

The tiers score on different scales, which is why ``_search_type`` travels in the result metadata and
``min_score`` is not comparable across tiers.
"""

SCORE_FIELD: Final[str] = "score"
"""Field every tier projects its score into, so a reader does not branch on the tier."""


def candidate_count(limit: int, oversampling_factor: int = DEFAULT_OVERSAMPLING_FACTOR) -> int:
    """Derive ``numCandidates`` from a resolved limit and validate it.

    Derived from the limit so the candidate pool grows with it. ``pymongo_search_utils`` computes the
    same product but does not validate it.

    Args:
        limit: Resolved result ceiling for this query. Must be positive.
        oversampling_factor: Multiplier turning the limit into a candidate pool.

    Returns:
        ``limit * oversampling_factor``, guaranteed within ``[1..MAX_NUM_CANDIDATES]``.

    Raises:
        ValueError: If ``limit`` is not positive, if the factor is not positive, or if the product
            exceeds :data:`MAX_NUM_CANDIDATES`. The message names both numbers, since either can be
            lowered.
    """
    if limit < 1:
        # The server rejects this from a different layer than the `numCandidates` checks: code 7912700
        # on MongoDB 8.0 and 65137 on 9.0.
        raise ValueError(
            f"strands_mongodb: 'limit' is {limit} and a search limit must be at least 1. The server "
            f"rejects a non-positive limit outright."
        )
    if oversampling_factor < 1:
        raise ValueError(
            f"strands_mongodb: 'oversampling_factor' is {oversampling_factor} and must be at least 1. "
            f"A factor below 1 asks the server to consider fewer candidates than the results it must "
            f"return, which is rejected as numCandidates below the limit."
        )

    candidates = limit * oversampling_factor
    if candidates > MAX_NUM_CANDIDATES:
        raise ValueError(
            f"strands_mongodb: numCandidates would be {candidates}, and the server requires a value "
            f"within bounds [1..{MAX_NUM_CANDIDATES}]. It is derived as limit ({limit}) times "
            f"'oversampling_factor' ({oversampling_factor}), so lower whichever of the two you can "
            f"afford to: the limit costs you results, the factor costs you recall."
        )
    return candidates


def check_candidate_count(*, limit: int, num_candidates: int) -> None:
    """Reject a limit larger than the candidate pool.

    The server checks this separately from the ``numCandidates`` bounds: ``limit=5`` with
    ``numCandidates=4`` is code 8, *limit should be less than or equal to numCandidates*, and ``limit=5``
    with ``numCandidates=5`` succeeds. :func:`candidate_count` cannot produce a violation today. The check
    keeps the bound enforced if that derivation changes.

    Args:
        limit: Resolved result ceiling.
        num_candidates: Candidate pool, normally from :func:`candidate_count`.

    Raises:
        ValueError: If ``limit`` exceeds ``num_candidates``.
    """
    if limit > num_candidates:
        raise ValueError(
            f"strands_mongodb: limit ({limit}) is greater than numCandidates ({num_candidates}), and "
            f"the server requires limit <= numCandidates. Raise 'oversampling_factor' or lower the "
            f"limit."
        )


def vector_stages(
    *,
    mode: VectorMode,
    index_name: str,
    limit: int,
    query_text: str | None = None,
    query_vector: list[float] | None = None,
    query_filter: Mapping[str, Any] | None = None,
    oversampling_factor: int = DEFAULT_OVERSAMPLING_FACTOR,
    score: bool = True,
) -> list[dict[str, Any]]:
    """Build the vector tier: one ``$vectorSearch`` plus the score projection it lacks.

    ``mode`` picks the builder and the ``path``. The ``autoEmbed`` index declares the text field as its
    vector field, because the deployment embeds it. The manual index declares the field the caller writes.

    There is no ``$limit`` stage. ``$vectorSearch`` carries its own ``limit``, and its ``filter`` is a
    pre-filter, so nothing downstream can shrink the result set below the requested count.

    Args:
        mode: Resolved vector path, from ``_indexes.vector_mode``.
        index_name: Atlas Vector Search index to query.
        limit: Resolved result ceiling.
        query_text: Natural-language query, on the AutoEmbeddings path only.
        query_vector: Pre-computed embedding, on the manual path only.
        query_filter: Validated MQL pre-filter over declared filterable paths. Omitted from the stage
            when empty.
        oversampling_factor: Multiplier behind ``numCandidates``.
        score: Whether to append the score projection. :func:`hybrid_stages` passes ``False`` because
            ``$rankFusion`` input pipelines may hold only selection stages.

    Returns:
        The search stage, and the ``$set`` projection when ``score``.

    Raises:
        ValueError: If the query's shape contradicts ``mode``, if neither or both query fields are
            supplied, or if the derived candidate count is out of bounds.
    """
    _check_query_shape(mode, query_text=query_text, query_vector=query_vector)

    candidates = candidate_count(limit, oversampling_factor)
    check_candidate_count(limit=limit, num_candidates=candidates)
    # The builders drop a falsy `filter`, so an empty one needs no normalizing here.
    shared: dict[str, Any] = {
        "search_field": CONTENT_FIELD if mode.auto_embed else VECTOR_FIELD,
        "index_name": index_name,
        "top_k": limit,
        "filter": dict(query_filter) if query_filter else None,
        "oversampling_factor": oversampling_factor,
    }

    if mode.auto_embed:
        # `_check_query_shape` guarantees `query_text`, and `auto_embed` guarantees `mode.model`.
        assert query_text is not None and mode.model is not None
        stage = autoembedding_vector_search_stage(query=query_text, model=mode.model, **shared)
    else:
        assert query_vector is not None
        stage = vector_search_stage(query_vector=query_vector, **shared)

    return [stage, _score_stage(VECTOR_SCORE_META)] if score else [stage]


def text_stages(
    *,
    query_text: str,
    index_name: str,
    limit: int,
    query_filter: Mapping[str, Any] | None = None,
    score: bool = True,
) -> list[dict[str, Any]]:
    """Build the full-text tier, ordered so a filtered query still returns its full count.

    Every stage comes from ``text_search_stage``, including the ``$limit``.

    ``$search`` filtering is a post-filter, so the filter ``$match`` must come before ``$limit``. Over 200
    matching documents with 10 in scope, asking for 5, this order returns 5 and an early ``$limit: 5``
    returns 0. No fixed over-fetch ahead of the ``$match`` is safe either: when the filter anti-correlates
    with relevance, every multiplier up to 38x returned nothing, and only the collection size always fills
    the limit.

    Leaving the stream unbounded until the ``$match`` does not fetch the whole corpus. The aggregation pulls
    from mongot in batches of 101, growing by 1.5x, and stops once ``$limit`` is satisfied, so cost tracks
    how many documents survive the filter. A filter that anti-correlates with relevance can still read to
    the end, the accepted worst case.

    Args:
        query_text: Natural-language query, scored with BM25 by Lucene's standard analyzer.
        index_name: Atlas Search index to query.
        limit: Resolved result ceiling, applied last.
        query_filter: Validated MQL predicate, applied as a ``$match`` before the limit. Dropped when
            empty.
        score: Whether to include the score projection (the builder's ``include_scores``).
            :func:`hybrid_stages` passes ``False``. The projection sits before ``$limit``, so it is
            dropped by keyword rather than by slicing the list.

    Returns:
        ``$search``, ``$match`` when there is a filter, ``$set`` when ``score``, then ``$limit``.
    """
    return list(
        text_search_stage(
            query=query_text,
            search_field=CONTENT_FIELD,
            index_name=index_name,
            limit=limit,
            filter=dict(query_filter) if query_filter else None,
            include_scores=score,
        )
    )


def hybrid_stages(
    *,
    mode: VectorMode,
    vector_index_name: str,
    text_index_name: str,
    query_text: str,
    limit: int,
    query_vector: list[float] | None = None,
    query_filter: Mapping[str, Any] | None = None,
    oversampling_factor: int = DEFAULT_OVERSAMPLING_FACTOR,
) -> list[dict[str, Any]]:
    """Build the hybrid tier: one ``$rankFusion`` over the vector and text pipelines.

    The server's ``$rankFusion`` (MongoDB 8.0+) owns the weights, the rank constant, tie-breaking and the
    merge of documents both tiers return, so this package does not implement reciprocal rank fusion itself.

    Both input pipelines are built with ``score=False``. ``$rankFusion`` ranks by position within each
    input pipeline, and an input pipeline may hold only selection stages, so the server rejects a score
    projection there. No weights are emitted (both default to 1) and no ``scoreDetails``.

    ``$rankFusion`` returns documents with no score field, so a ``$set`` after it projects the fused score.
    See :data:`RANK_FUSION_PENALTY` for its scale. A final ``$limit`` follows, because the fused list is the
    union of both branches and can hold up to twice ``limit`` documents.

    Args:
        mode: Resolved vector path, passed through to :func:`vector_stages`.
        vector_index_name: Atlas Vector Search index for the vector branch.
        text_index_name: Atlas Search index for the text branch.
        query_text: Natural-language query. Required, because the text branch has nothing else to search
            with.
        limit: Resolved result ceiling, applied within each branch and again to the fused list.
        query_vector: Pre-computed embedding, on the manual path only.
        query_filter: Validated MQL predicate. Each branch places it where its tier needs it: inside the
            stage for vector, in a ``$match`` for text.
        oversampling_factor: Multiplier behind ``numCandidates`` on the vector branch.

    Returns:
        Three stages: the ``$rankFusion``, the ``$set`` projecting its fused score, and the ``$limit``.

    Raises:
        ValueError: Whatever :func:`vector_stages` or :func:`text_stages` raises, unchanged.
    """
    vector = vector_stages(
        mode=mode,
        index_name=vector_index_name,
        limit=limit,
        query_text=query_text if mode.auto_embed else None,
        query_vector=query_vector,
        query_filter=query_filter,
        oversampling_factor=oversampling_factor,
        score=False,
    )
    text = text_stages(
        query_text=query_text, index_name=text_index_name, limit=limit, query_filter=query_filter, score=False
    )
    fusion = {"$rankFusion": {"input": {"pipelines": {"vector": vector, "text": text}}}}
    # The fusion returns the union of both branches, up to twice `limit`.
    return [fusion, _score_stage(FUSION_SCORE_META), {"$limit": limit}]


def _score_stage(meta: str) -> dict[str, Any]:
    """Return the score projection the vector and hybrid tiers need.

    Args:
        meta: ``$meta`` name for the tier. The names are not interchangeable.

    Returns:
        A ``$set`` stage writing the score into :data:`SCORE_FIELD`.
    """
    return {"$set": {SCORE_FIELD: {"$meta": meta}}}


def check_query_keys(
    *, query_text: str | None, query_vector: list[float] | None, fusion_possible: bool = False
) -> None:
    """Reject the two query shapes no tier can answer. Needs no vector mode.

    Separate from :func:`_check_query_shape` so the memory store can run it before choosing a tier. The
    mode-dependent check cannot run that early: on a manual-vector store, a ``text`` query is wrong for the
    vector tier but right for the text tier.

    Args:
        query_text: What the caller put in ``text``, if anything.
        query_vector: What the caller put in ``vector``, if anything.
        fusion_possible: Whether the store's configuration could use both keys: the manual-vector path
            with ``search_type`` at ``auto`` or ``hybrid``, where the fused tier reads ``text`` in one
            input pipeline and ``vector`` in the other. It depends on the config only, so this check runs
            before any round trip. Whether the server can fuse is decided later, by the tier list.

    Raises:
        ValueError: If both query fields are set where nothing could use both, or if neither is.
    """
    if query_text is not None and query_vector is not None and not fusion_possible:
        raise ValueError(
            "strands_mongodb: the query sets both 'text' and 'vector', and this store has no "
            "interpretation that uses both. Set 'text' to have the deployment embed it, or 'vector' to "
            "supply the embedding yourself. Both together are read only by the hybrid tier on a "
            "manual-vector store, where each key feeds one half of the fusion."
        )
    if query_text is None and query_vector is None:
        raise ValueError(
            "strands_mongodb: the query sets neither 'text' nor 'vector'. MongoDBSearchQuery marks "
            "every field optional because the valid combinations are not expressible as a TypedDict, "
            "so exactly one of the two has to be present."
        )


def _check_query_shape(mode: VectorMode, *, query_text: str | None, query_vector: list[float] | None) -> None:
    """Reject a query whose shape contradicts the configured vector path.

    An AutoEmbeddings-shaped stage against a manual index fails with code 8, ``queryVector must be
    present``. The reverse is untested, because an ``autoEmbed`` index needs a registered embedding model.
    Both are rejected here, where the message can name the config key that chose the path.

    Args:
        mode: Resolved vector path.
        query_text: What the caller put in ``text``, if anything.
        query_vector: What the caller put in ``vector``, if anything.

    Raises:
        ValueError: If both query fields are set, if neither is, or if the one that is set belongs to
            the other path.
    """
    check_query_keys(query_text=query_text, query_vector=query_vector)

    if mode.auto_embed and query_vector is not None:
        raise ValueError(
            "strands_mongodb: the query supplies 'vector', but this store is on the Atlas "
            "AutoEmbeddings path, whose index declares the text field as its vector field and holds no "
            "caller-supplied embeddings. That path is selected by 'embedding_model', and also by "
            "setting none of the vector keys at all, which is why this can surprise a caller who set "
            "nothing. Query with 'text' instead, or set 'embedding_dimensions' and 'similarity' in "
            "MongoDBConfig to select the manual path. Switching paths means recreating the vector "
            "index, since the two index different fields."
        )
    if not mode.auto_embed and query_text is not None:
        raise ValueError(
            "strands_mongodb: the query supplies 'text', but 'embedding_dimensions' and 'similarity' "
            "in MongoDBConfig select the manual-vector path, where the deployment does not embed and "
            "this package does not either. Supply 'vector' with an embedding you computed, or remove "
            "both config keys to use Atlas AutoEmbeddings."
        )
