"""Search stage construction over ``pymongo_search_utils``' pure builders (``REQ-E-070``).

Pure functions over plain dicts, like ``_filter``: no client, no I/O, nothing that needs a deployment.
Everything here is checkable in the unit tier, which matters because the two mistakes this module
exists to prevent are both shape mistakes that a live query answers with results rather than an error.

The library's own builders are the source of every stage. What this module adds is the three things
they leave to a caller, all of which are decisions rather than plumbing:

* **One return type.** ``vector_search_stage`` returns a single dict and ``text_search_stage`` returns
  a list of up to four stages. Every function here returns ``list[dict]`` so the two tiers compose the
  same way.
* **Which builder, and which path.** Both follow ``VectorMode``: the ``autoEmbed`` index declares the
  *text* field as its vector field, so the AutoEmbeddings query paths at ``content`` while the manual
  query paths at ``embedding``. Crossing them is code 8 rather than a wrong answer, measured, but only
  after a round trip.
* **The score projection.** The vector builders emit no score, because hybrid ranks by reciprocal rank
  instead. ``$meta`` names differ per tier and are not interchangeable: measured, each errors on the
  other tier.

Hybrid is deliberately absent. ``combine_pipelines``, ``reciprocal_rank_stage`` and
``final_hybrid_stage`` are the three exports v0.1.0 does not call: they are phase-2 surface.

Private, like ``_client`` and ``_filter``. The emitted stages are a promise pinned by ``AT-070``; these
signatures are not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from strands_mongodb._extras import missing_extra
from strands_mongodb._indexes import CONTENT_FIELD, VECTOR_FIELD
from strands_mongodb.types import DEFAULT_OVERSAMPLING_FACTOR

try:
    from pymongo_search_utils import (
        autoembedding_vector_search_stage,
        text_search_stage,
        vector_search_stage,
    )
except ImportError as error:  # pragma: no cover - exercised by AT-006b in the extras tier
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
"""Rank constant the server adds inside ``$rankFusion``, fixed and not configurable here.

The reason this is a named constant in a package that never sends it: it sets the *scale* of every
score the hybrid tier returns, and that scale is nothing like the other two tiers'. A fused score is
``sum over the tiers that ranked it of weight x 1/(rank + 60)``, so a document first in one tier scores
about ``1/61``, or ``0.016``, and one first in both about ``0.033``, while ``cosine``
similarity and BM25 both land in roughly 0 to 1. A ``min_score`` chosen against either of those and
applied to a fused score filters out every result and looks exactly like an empty collection
(``REQ-E-145``).

So the number is here to be *quoted*: the config docstring and the README operator reference both
resolve their stated range against this constant rather than against a remembered 60, which is what
``AT-151`` asserts.
"""

MAX_NUM_CANDIDATES: Final[int] = 10000
"""Upper bound the server enforces on ``numCandidates``.

Measured on mongot 8.3.8: outside ``[1..10000]`` is code 8, *"numCandidates" must be within bounds
[1..10000]*. Read off a server and written down here, which is the same kind of claim
``_indexes.MAX_DIMENSIONS`` makes and goes stale the same way, so ``AT-072b`` re-reads it from a real
deployment rather than trusting this line.
"""

VECTOR_SCORE_META: Final[str] = "vectorSearchScore"
"""``$meta`` name carrying a ``$vectorSearch`` score. Not interchangeable with the text one."""

FUSION_SCORE_META: Final[str] = "score"
"""``$meta`` name carrying a ``$rankFusion`` score, and the third name that is not interchangeable.

The plainest of the three, and the one most likely to be mistaken for a field rather than metadata:
``{"$meta": "score"}`` reads almost like ``"$score"`` and means something else entirely. It is also the
only one of the three whose stage does *not* project it for you -- ``$rankFusion`` emits the fused
documents with no score field at all, so without the ``$set`` this module appends outside the stage,
every hybrid result would arrive scoring ``0.0`` and any ``min_score`` would empty the result list.
That failure is silent and looks exactly like an empty collection, with nothing to say the score was at fault.

Available since 8.0, the same floor as the stage itself (:data:`_indexes.RANK_FUSION_MIN_MAJOR`).
"""

TEXT_SCORE_META: Final[str] = "searchScore"
"""``$meta`` name carrying a ``$search`` score.

Not interchangeable with the vector one, and crossing them is not reliably loud. Measured on mongot
8.3.8, asking for this one on a ``$vectorSearch`` pipeline is *"query requires $search score metadata,
but it is not available"*. Measured on Atlas 8.0.30, the same pipeline is accepted and every document
comes back with no score field, which the caller in ``memory_store`` reads as ``0.0``. So these two
constants are the guard rather than the server, and ``AT-070c`` asserts the invariant that holds on
either deployment.

Both names yield a ``float`` where they belong, but on different scales, which is why ``_search_type``
travels in the result metadata and ``min_score`` is not comparable across tiers.
"""

SCORE_FIELD: Final[str] = "score"
"""Field the score is projected into, on both tiers, so a reader does not branch on the tier."""


def candidate_count(limit: int, oversampling_factor: int = DEFAULT_OVERSAMPLING_FACTOR) -> int:
    """Derive ``numCandidates`` from a resolved limit and validate it (``REQ-E-072``).

    Derived rather than passed as a constant, so a caller who raises their limit gets a candidate pool
    that grew with it. ``MongoDBSearchQuery`` carries ``oversampling_factor`` rather than
    ``numCandidates`` for the same reason: the dial that matters is the ratio, and exposing the product
    directly is how a query ends up asking for 10,000 candidates to return 5 results.

    ``pymongo_search_utils`` computes the same product inside its builders. It does not validate it,
    and the two bounds the server enforces are separate checks with separate codes.

    Args:
        limit: Resolved result ceiling for this query. Must be positive.
        oversampling_factor: Multiplier turning the limit into a candidate pool.

    Returns:
        ``limit * oversampling_factor``, guaranteed within ``[1..MAX_NUM_CANDIDATES]``.

    Raises:
        ValueError: If ``limit`` is not positive, if the factor is not positive, or if the product
            exceeds :data:`MAX_NUM_CANDIDATES`. Both numbers are named, because either one is the one
            the caller can lower and only they know which.
    """
    if limit < 1:
        # Measured as code 7912700, `Expected limit to be positive`, from a different layer than the
        # `numCandidates` checks below. Distinct enough to be worth its own message.
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
    """Reject a limit larger than the candidate pool (``REQ-E-072a``).

    Separate from :func:`candidate_count` because the server enforces it separately: measured,
    ``limit=5`` with ``numCandidates=4`` is code 8, *limit should be less than or equal to
    numCandidates*, while ``limit=5`` with ``numCandidates=5`` succeeds. The bound is ``<=``.

    Unreachable through this module's own arithmetic, where the product of a positive limit and a
    factor of 1 or more is never below the limit. Checked anyway, so the derivation can change without
    the bound quietly going unchecked, and so a caller assembling a stage by hand hits it here.

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
    """Build the vector tier: one ``$vectorSearch`` and the score projection it does not include.

    ``REQ-E-070``, ``REQ-E-072``, ``REQ-E-074``. Which builder runs follows ``mode``, and so does
    ``path``: the ``autoEmbed`` index definition declares the *text* field as the vector field, because the
    deployment embeds it, while the manual definition declares the vector field the caller writes.

    No ``$limit`` stage. ``$vectorSearch`` carries its own ``limit`` and its ``filter`` is a pre-filter
    applied during the search, so unlike the text tier there is nothing downstream that could shrink
    the result set below the requested count.

    Args:
        mode: Resolved vector path, from ``_indexes.vector_mode``.
        index_name: Atlas Vector Search index to query.
        limit: Resolved result ceiling.
        query_text: Natural-language query, on the AutoEmbeddings path only.
        query_vector: Pre-computed embedding, on the manual path only (``REQ-E-074``).
        query_filter: Validated MQL pre-filter over declared filterable paths. Omitted from the stage
            when empty, since ``{}`` and absent mean the same thing and look different to the planner.
        oversampling_factor: Multiplier behind ``numCandidates``.
        score: Whether to append the score projection. ``False`` only from :func:`hybrid_stages`, where
            the stage would be both dead weight and illegal: ``$rankFusion`` ranks by position inside
            each input pipeline and accepts only selection stages there (``REQ-E-142``).

    Returns:
        The search stage, and the ``$set`` projection when ``score``.

    Raises:
        ValueError: If the query's shape contradicts ``mode``, if neither or both query fields are
            supplied (``REQ-E-074a``), or if the derived candidate count is out of bounds.
    """
    _check_query_shape(mode, query_text=query_text, query_vector=query_vector)

    candidates = candidate_count(limit, oversampling_factor)
    check_candidate_count(limit=limit, num_candidates=candidates)
    # `filter=None` and `filter={}` are both falsy to the builders, which drop the key. Passing the
    # dict through rather than normalizing first keeps that decision in one place: theirs.
    shared: dict[str, Any] = {
        "search_field": CONTENT_FIELD if mode.auto_embed else VECTOR_FIELD,
        "index_name": index_name,
        "top_k": limit,
        "filter": dict(query_filter) if query_filter else None,
        "oversampling_factor": oversampling_factor,
    }

    if mode.auto_embed:
        # `query_text` is not None here: `_check_query_shape` rejects the alternative, and `mode.model`
        # is likewise non-None whenever `auto_embed` is True.
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
    """Build the full-text tier, in the order that makes a filtered query return its full count.

    ``REQ-E-070``, ``REQ-E-073``, ``REQ-E-073a``. Everything here comes from ``text_search_stage``,
    including the ``$limit``, and the value of this function is that it adds nothing.

    ``$search`` filtering is a post-filter, so the ordering is load bearing. Measured over 200 documents
    all matching the query with 10 of them in scope, asking for 5: the emitted order returns 5 and an
    early ``$limit: 5`` returns 0. An over-fetch ahead of the ``$match``, which is the ordinary reading
    of "over-fetch" and what ``REQ-E-073`` originally prescribed, returns between 0 and 5 depending on
    the corpus, and on one where the filter anti-correlates with relevance every multiplier up to 38x
    returns nothing. No fixed multiplier is safe: the smallest bound that always fills the limit is the
    collection size.

    Leaving the result set unbounded until after the filter is what works, and it is not a request for
    the whole corpus. The aggregation pulls from mongot in batches of 101 growing by 1.5x and stops once
    ``$limit`` is satisfied, so cost tracks how many documents survive the ``$match`` rather than the
    limit or the collection size. A filter that anti-correlates with relevance still escalates to the
    end, which ``REQ-E-073a`` accepts as the worst case.

    Args:
        query_text: Natural-language query, scored by Lucene's standard BM25 analyzer.
        index_name: Atlas Search index to query.
        limit: Resolved result ceiling, applied last.
        query_filter: Validated MQL predicate, applied as a ``$match`` before the limit. Dropped when
            empty rather than emitted as ``{"$match": {}}``.
        score: Whether to include the score projection, as ``include_scores`` on the builder. ``False``
            only from :func:`hybrid_stages`. Note the projection is not last here -- ``$limit`` follows
            it -- so dropping it is a keyword rather than a slice of the returned list.

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
    """Build the hybrid tier: one ``$rankFusion`` over the two pipelines this module already builds.

    ``REQ-E-142``. The whole function is a wrapper, and that is the point of it. Reciprocal rank fusion
    is not hard to write -- ``pymongo_search_utils.combine_pipelines`` writes it, over ``$unionWith``
    and a ``$group``, and ``langchain-mongodb`` ships it -- but writing it makes this package the owner
    of the weights, the penalties, the tie-breaking and the merge of documents that both tiers return.
    The server owns all four in a stage that has existed since 8.0.

    Both input pipelines are built with ``score=False``. ``$rankFusion`` ranks documents by their
    **position** within each input pipeline, so a ``$meta`` projection inside one is read by nothing --
    and an input pipeline may contain only selection stages, so the server rejects it rather than
    ignoring it.

    No weights are emitted: both pipelines carry the server's default of 1, and ``{"vector": 1, "text":
    1}`` is the same instruction spelled longer. No ``scoreDetails`` either, which is the per-tier
    breakdown and not the score.

    The score itself is a second stage, outside the fusion. ``$rankFusion`` returns fused documents
    carrying no score field, so the ``$set`` reading ``{"$meta": "score"}`` is what makes ``min_score``
    and ``_relevance_score`` mean anything on this tier -- see :data:`FUSION_SCORE_META` for what its
    absence would look like, and :data:`RANK_FUSION_PENALTY` for the scale it lands on.

    Args:
        mode: Resolved vector path, passed through to :func:`vector_stages`.
        vector_index_name: Atlas Vector Search index for the vector branch.
        text_index_name: Atlas Search index for the text branch. Separate arguments because the two
            tiers are separate indexes and a single ``index_name`` would silently query one twice.
        query_text: Natural-language query. Required here, unlike on the vector path, because the text
            branch has nothing else to search with -- which is why ``REQ-E-144`` makes a vector-only
            query ineligible for this tier rather than degrading it.
        limit: Resolved result ceiling, applied within each branch. The fusion returns at most the
            union of the two, which is at most ``2 * limit`` before the caller's own limit applies.
        query_vector: Pre-computed embedding, on the manual path only.
        query_filter: Validated MQL predicate. Goes to both branches, where each builder puts it where
            its own tier needs it -- inside the stage for vector, in a ``$match`` for text.
        oversampling_factor: Multiplier behind ``numCandidates`` on the vector branch.

    Returns:
        Two stages: the ``$rankFusion`` and the ``$set`` projecting its fused score.

    Raises:
        ValueError: Whatever the two builders raise, unchanged. Validation is theirs, and reusing them
            means reusing the rejections and not only the happy path.
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
    return [fusion, _score_stage(FUSION_SCORE_META)]


def _score_stage(meta: str) -> dict[str, Any]:
    """The projection both tiers need and only the text builder provides.

    Args:
        meta: ``$meta`` name for the tier. The two are not interchangeable and each errors on the
            other tier, measured, so this is a name to get right rather than a preference.

    Returns:
        A ``$set`` stage writing the score into :data:`SCORE_FIELD`.
    """
    return {"$set": {SCORE_FIELD: {"$meta": meta}}}


def check_query_keys(
    *, query_text: str | None, query_vector: list[float] | None, fusion_possible: bool = False
) -> None:
    """Reject the two query shapes no tier can answer, without needing a mode (``REQ-E-074a``).

    Split out of :func:`_check_query_shape` so the memory store can run it before it knows which tier
    will answer. The mode-dependent half below cannot run that early: ``embedding_dimensions`` selects
    the manual vector path, and a query carrying ``text`` is wrong for the *vector* tier there while
    still being exactly right for the text tier on the same store.

    Args:
        query_text: What the caller put in ``text``, if anything.
        query_vector: What the caller put in ``vector``, if anything.
        fusion_possible: Whether this store's *configuration* could put both keys to work, which is the
            manual-vector path with ``search_type`` left at ``auto`` or set to ``hybrid``
            (``REQ-E-151``). Both-set stops being nonsense there: the fused tier reads ``text`` in one
            input pipeline and ``vector`` in the other. Deliberately a fact about the config and not
            about the deployment, so this stays the check that runs before the first round trip --
            whether the server can actually fuse is settled later, by the tier list.

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
    """Reject a query whose shape contradicts the configured vector path (``REQ-E-074a``).

    One direction is measured loud: an AutoEmbeddings-shaped stage against a manual index is code 8,
    ``queryVector must be present``. The other cannot be measured on any deployment CI reaches, because
    an ``autoEmbed`` index needs a registered embedding model. Assuming it fails the same way
    would be inventing behaviour, so both are rejected here, where the message can name the config key
    that chose the path.

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
