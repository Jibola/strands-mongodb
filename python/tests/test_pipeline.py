"""Search stage construction: ``REQ-E-070`` through ``REQ-E-074a``.

Three tiers again, and the split is sharper here than in ``test_indexes.py`` because the unit tier
can only ever check that a dict has the shape somebody claimed. `AT-070` asserts the AutoEmbeddings
query shape exactly, and passing it settles nothing about whether Atlas accepts that shape: the question is
open precisely because two published packages emit different ones. `AT-070i` is the test that would
settle it and it skips everywhere CI can reach.

What the contract and integration tiers add is the arithmetic and the ordering. `MAX_NUM_CANDIDATES`
is a number copied from a server, so `AT-072b` re-reads it from that server. `REQ-E-073`'s ordering is
the case where pipelines containing identical stages return the requested count or nothing depending
only on where the ``$limit`` sits, so `AT-073b` runs the wrong ones too rather than asserting the good
one in isolation.

What was measured is recorded in this suite, including the requirement whose prescribed
mechanism the measurement contradicted.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Final

import pytest
from conftest import skip_if_search_is_unreachable
from pymongo import AsyncMongoClient
from pymongo.errors import OperationFailure, PyMongoError

from strands_mongodb import _indexes, _pipeline

if TYPE_CHECKING:
    from strands_mongodb.types import MongoDBConfig

AUTO_EMBED_MODE = _indexes.vector_mode({"database": "d"})
"""The zero-configuration mode: no vector keys set at all resolves to AutoEmbeddings."""

MANUAL_MODE = _indexes.vector_mode({"database": "d", "embedding_dimensions": 4, "similarity": "cosine"})
"""The mode CI can actually create an index for."""

VECTOR = [1.0, 0.0, 0.0, 0.0]

MANUAL_VECTOR_CONFIG: MongoDBConfig = {
    "database": "unused",
    "embedding_dimensions": 4,
    "similarity": "cosine",
}
"""Matches ``test_indexes.MANUAL_VECTOR_CONFIG``. Duplicated rather than imported across test modules,
which is the convention the suite already follows for fixture data."""


def _names(stages: list[dict[str, Any]]) -> list[str]:
    """Stage names in order, which is the property `REQ-E-073a` is actually about."""
    return [next(iter(stage)) for stage in stages]


class TestTheAutoEmbeddingsQueryShape:
    """AT-070 / REQ-E-070, and the shape half of that question.

    ``pymongo_search_utils`` 0.3.0 emits ``{"query": {"text": q}}`` with a top-level ``model``; the
    community ``strands-mongodb-store`` 0.2.0 emits a bare ``"query": q`` and no ``model``. The two
    cannot both be right and no deployment CI reaches can adjudicate, so this pins the one we send and
    the exactness is the point: a test asserting only that ``query`` is present passes for both.
    """

    def test_the_stage_matches_the_builder_exactly(self) -> None:
        """AT-070, as an equality against the whole dict rather than a subset check."""
        stages = _pipeline.vector_stages(
            mode=AUTO_EMBED_MODE,
            index_name="vector_index",
            limit=5,
            query_text="dark mode",
        )

        assert stages[0] == {
            "$vectorSearch": {
                "index": "vector_index",
                "path": _indexes.CONTENT_FIELD,
                "query": {"text": "dark mode"},
                "numCandidates": 50,
                "limit": 5,
                "model": _indexes.DEFAULT_EMBEDDING_MODEL,
            }
        }

    def test_the_query_is_a_nested_document_and_not_a_bare_string(self) -> None:
        """The one difference between the two candidate shapes, asserted on its own.

        Named separately from the equality above because this is the assertion that would fail if
        the question resolves the other way, and a reader coming back to this file after that happens needs
        to find the line to change rather than a dict to diff.
        """
        stage = _pipeline.vector_stages(mode=AUTO_EMBED_MODE, index_name="i", limit=1, query_text="q")[0][
            "$vectorSearch"
        ]

        assert stage["query"] == {"text": "q"}
        assert "queryVector" not in stage
        assert stage["model"] == _indexes.DEFAULT_EMBEDDING_MODEL

    def test_a_configured_model_reaches_the_stage(self) -> None:
        """The model travels from the config through ``VectorMode`` into the query, not only into the
        index definition. Both have to name the same model or the query is scored against vectors a
        different model produced."""
        mode = _indexes.vector_mode({"database": "d", "embedding_model": "voyage-3-large"})

        stage = _pipeline.vector_stages(mode=mode, index_name="i", limit=1, query_text="q")[0]

        assert stage["$vectorSearch"]["model"] == "voyage-3-large"


class TestTheBuilderAndPathFollowTheMode:
    """AT-070a / REQ-E-070, REQ-E-067.

    Two mistakes are possible here and neither is a type error: calling the wrong builder, and giving
    the right builder the wrong ``path``. The ``autoEmbed`` definition indexes the **text** field,
    so the AutoEmbeddings query paths at ``content`` while the manual query paths at ``embedding``.
    """

    def test_the_manual_path_sends_a_query_vector_over_the_vector_field(self) -> None:
        stages = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="vector_index", limit=5, query_vector=VECTOR)

        assert stages[0] == {
            "$vectorSearch": {
                "index": "vector_index",
                "path": _indexes.VECTOR_FIELD,
                "queryVector": VECTOR,
                "numCandidates": 50,
                "limit": 5,
            }
        }

    def test_the_manual_stage_carries_no_model_and_no_text_query(self) -> None:
        """The mirror of ``test_the_query_is_a_nested_document_and_not_a_bare_string``. Measured: the
        server dispatches on the index, so an autoembed-shaped stage against a manual index is code 8,
        ``queryVector must be present``. Loud, but only after a round trip."""
        stage = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=1, query_vector=VECTOR)[0][
            "$vectorSearch"
        ]

        assert "model" not in stage
        assert "query" not in stage

    @pytest.mark.parametrize(
        ("mode", "kwargs", "expected_path"),
        [
            (AUTO_EMBED_MODE, {"query_text": "q"}, _indexes.CONTENT_FIELD),
            (MANUAL_MODE, {"query_vector": VECTOR}, _indexes.VECTOR_FIELD),
        ],
        ids=["autoembed-paths-at-the-text-field", "manual-paths-at-the-vector-field"],
    )
    def test_the_path_is_the_field_the_index_declared(
        self, mode: Any, kwargs: dict[str, Any], expected_path: str
    ) -> None:
        """The path in the query has to be the path in the index definition, which is the authority
        for it. Read out of ``vector_index_definition`` rather than restated, so the two cannot
        drift apart without this failing."""
        declared = _indexes.vector_index_definition(mode)["fields"][0]["path"]
        stage = _pipeline.vector_stages(mode=mode, index_name="i", limit=1, **kwargs)[0]

        assert stage["$vectorSearch"]["path"] == expected_path == declared


class TestEveryFunctionReturnsAListOfStages:
    """AT-070b / REQ-E-070 — the asymmetry this module exists to absorb.

    ``vector_search_stage`` returns one dict and ``text_search_stage`` returns a list of four. A
    caller that has to know which is which is a caller writing the branch this module exists to hold.
    """

    def test_the_vector_tier_returns_a_list(self) -> None:
        stages = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=1, query_vector=VECTOR)

        assert isinstance(stages, list)
        assert all(isinstance(stage, dict) for stage in stages)

    def test_the_text_tier_returns_a_list(self) -> None:
        stages = _pipeline.text_stages(query_text="q", index_name="i", limit=1)

        assert isinstance(stages, list)
        assert all(isinstance(stage, dict) for stage in stages)

    def test_mutating_one_result_does_not_affect_the_next(self) -> None:
        """Appending a stage to a returned pipeline is an ordinary thing for a caller to do, and it
        must not reach into anything the next call reuses."""
        first = _pipeline.text_stages(query_text="q", index_name="i", limit=1)
        first.append({"$count": "n"})

        assert _names(_pipeline.text_stages(query_text="q", index_name="i", limit=1)) == [
            "$search",
            "$set",
            "$limit",
        ]

    def test_the_text_tier_copies_the_builder_list_rather_than_handing_it_over(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``list(...)`` around ``text_search_stage``, asserted as identity rather than as equality.

        ``pymongo_search_utils`` 0.3.0 builds a fresh list per call, so the test above passes whether or
        not we copy: deleting the copy left the suite green. That makes
        the copy a guard against a library that starts sharing, and the only way to observe it is to be
        the library that shares. Patching the builder rather than trusting it is also the only assertion
        here that would survive the library changing under us.
        """
        shared: list[dict[str, Any]] = [{"$search": {}}]
        monkeypatch.setattr(_pipeline, "text_search_stage", lambda **_: shared)

        returned = _pipeline.text_stages(query_text="q", index_name="i", limit=1)
        returned.append({"$count": "n"})

        assert returned is not shared
        assert shared == [{"$search": {}}]


class TestTheScoreProjection:
    """AT-070c's unit half / REQ-E-070.

    The vector builders emit no score, by design: hybrid search ranks by reciprocal rank instead. The
    text builder emits ``searchScore``. Measured, the two ``$meta`` names are not interchangeable and
    each errors on the other tier, so this is a correctness assertion rather than a naming preference.
    """

    def test_the_vector_tier_projects_the_vector_score(self) -> None:
        stages = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=1, query_vector=VECTOR)

        assert stages[1] == {"$set": {"score": {"$meta": "vectorSearchScore"}}}

    def test_the_text_tier_projects_the_search_score(self) -> None:
        stages = _pipeline.text_stages(query_text="q", index_name="i", limit=1)

        assert {"$set": {"score": {"$meta": "searchScore"}}} in stages

    def test_neither_tier_projects_the_other_tiers_score_name(self) -> None:
        """The crossed name is the mistake this pins, and it is one keystroke from correct."""
        vector = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=1, query_vector=VECTOR)
        text = _pipeline.text_stages(query_text="q", index_name="i", limit=1)

        assert {"$set": {"score": {"$meta": "searchScore"}}} not in vector
        assert {"$set": {"score": {"$meta": "vectorSearchScore"}}} not in text


class TestTheCandidateCount:
    """AT-072, AT-072a / REQ-E-072, REQ-E-072a.

    Two bounds, measured separately because the server enforces them separately: the range
    ``[1..10000]`` at code 8, and ``limit <= numCandidates`` at code 8 with a different message.
    """

    @pytest.mark.parametrize(
        ("limit", "factor", "expected"),
        [(1, 10, 10), (5, 10, 50), (10, 10, 100), (10, 1, 10), (100, 100, 10000)],
    )
    def test_the_count_is_the_product(self, limit: int, factor: int, expected: int) -> None:
        """AT-072. ``pymongo_search_utils`` computes this inside the builder; asserting it here is what
        makes the derivation ours to validate rather than theirs to change."""
        assert _pipeline.candidate_count(limit, factor) == expected

    def test_the_count_scales_with_the_limit(self) -> None:
        """A fixed constant would pass every equality above if the factor happened to be 1, so the
        scaling is asserted as a relation rather than as a table row."""
        assert _pipeline.candidate_count(20, 10) == 2 * _pipeline.candidate_count(10, 10)

    def test_the_default_factor_matches_the_library(self) -> None:
        from strands_mongodb.types import DEFAULT_OVERSAMPLING_FACTOR

        stage = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=7, query_vector=VECTOR)[0][
            "$vectorSearch"
        ]

        assert stage["numCandidates"] == 7 * DEFAULT_OVERSAMPLING_FACTOR

    @pytest.mark.parametrize(
        ("limit", "factor"),
        [(1001, 10), (10001, 1), (100, 101), (1, 10001)],
        ids=["10010", "10001", "10100", "10001-by-factor"],
    )
    def test_a_product_above_the_ceiling_is_rejected(self, limit: int, factor: int) -> None:
        """AT-072. Measured: ``numCandidates`` of 10001 is code 8. Rejected here so the caller learns
        which of the two numbers to lower."""
        with pytest.raises(ValueError, match=r"numCandidates"):
            _pipeline.candidate_count(limit, factor)

    def test_the_ceiling_itself_is_accepted(self) -> None:
        """Measured: 10000 succeeds. An off-by-one here would reject a query the server answers."""
        assert _pipeline.candidate_count(1000, 10) == _pipeline.MAX_NUM_CANDIDATES

    @pytest.mark.parametrize("factor", [0, -1])
    def test_a_non_positive_factor_is_rejected(self, factor: int) -> None:
        """``factor`` of 0 makes ``numCandidates`` zero, which is below the measured floor of 1."""
        with pytest.raises(ValueError):
            _pipeline.candidate_count(5, factor)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_limit_is_rejected(self, limit: int) -> None:
        """Measured as code **7912700** on 8.0 and **65137** on 9.0, from a different layer than
        the ``numCandidates`` checks. The message says so rather than folding both into one."""
        with pytest.raises(ValueError, match=r"limit"):
            _pipeline.candidate_count(limit, 10)

    def test_a_limit_above_the_candidate_count_is_rejected(self) -> None:
        """AT-072a / REQ-E-072a. Unreachable through ``candidate_count``'s own arithmetic for any
        factor of 1 or more, which is why it is checked where the two numbers meet instead."""
        with pytest.raises(ValueError, match=r"numCandidates"):
            _pipeline.check_candidate_count(limit=5, num_candidates=4)

    def test_a_limit_equal_to_the_candidate_count_is_accepted(self) -> None:
        """Measured: ``limit=5`` with ``numCandidates=5`` succeeds, so the bound is ``<=`` and this is
        the boundary case a ``<`` would get wrong."""
        _pipeline.check_candidate_count(limit=5, num_candidates=5)

    def test_the_relationship_is_checked_at_the_point_the_stage_is_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`REQ-E-072a` says *before sending the stage*, and that is a claim about wiring rather than
        about arithmetic.

        The two tests above prove ``check_candidate_count`` rejects the right numbers. Neither proves
        ``vector_stages`` calls it, and it cannot be proved by feeding in bad numbers, because the
        derivation cannot produce them: a positive limit times a factor of at least 1 is never below the
        limit. Deleting the call left the whole suite green. So the derivation is
        replaced with one that produces a bad pair, which is the only input that distinguishes a wired
        guard from an absent one.
        """
        monkeypatch.setattr(_pipeline, "candidate_count", lambda *_, **__: 1)

        with pytest.raises(ValueError, match="greater than numCandidates"):
            _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=5, query_vector=VECTOR)


class TestTheFullTextStageOrder:
    """AT-073a / REQ-E-073, REQ-E-073a — the assertion that has to be about order.

    Every wrong ordering here contains exactly the same four stages. Asserting presence passes for all
    of them; asserting the sequence is the only thing that separates the pipeline returning 5 results
    from the one returning 0.
    """

    def test_the_stages_are_search_match_set_limit_in_that_order(self) -> None:
        stages = _pipeline.text_stages(
            query_text="dark mode", index_name="text_index", limit=5, query_filter={"scope": "s"}
        )

        assert _names(stages) == ["$search", "$match", "$set", "$limit"]

    def test_the_limit_is_the_last_stage(self) -> None:
        """Stated on its own because it is the whole of `REQ-E-073a`: a ``$limit`` anywhere before the
        ``$match`` returns 0 of 5, measured."""
        stages = _pipeline.text_stages(query_text="q", index_name="i", limit=5, query_filter={"scope": "s"})

        assert _names(stages).index("$limit") == len(stages) - 1
        assert _names(stages).index("$limit") > _names(stages).index("$match")

    def test_no_stage_bounds_the_result_set_before_the_match(self) -> None:
        """The negative form, which catches an over-fetch somebody adds later in good faith. A fixed
        20x over-fetch measurably returns 2 of 5 requested."""
        stages = _pipeline.text_stages(query_text="q", index_name="i", limit=5, query_filter={"scope": "s"})
        before_match = _names(stages)[: _names(stages).index("$match")]

        assert before_match == ["$search"]

    def test_a_query_with_no_filter_omits_the_match(self) -> None:
        """``text_search_stage`` drops the stage rather than emitting ``{"$match": {}}``, and an empty
        ``$match`` would be a stage the plan cache sees for a different shape of query."""
        stages = _pipeline.text_stages(query_text="q", index_name="i", limit=5)

        assert _names(stages) == ["$search", "$set", "$limit"]

    def test_the_search_stage_names_the_index_and_the_content_field(self) -> None:
        stages = _pipeline.text_stages(query_text="dark mode", index_name="text_index", limit=5)

        assert stages[0] == {
            "$search": {
                "index": "text_index",
                "text": {"query": "dark mode", "path": _indexes.CONTENT_FIELD},
            }
        }


class TestTheFilterReachesTheRightPlace:
    """REQ-E-071's query half. The vector filter is a pre-filter inside the stage; the text filter is
    a post-filter in its own ``$match``. Same caller input, two different destinations."""

    def test_the_vector_filter_goes_inside_the_stage(self) -> None:
        stages = _pipeline.vector_stages(
            mode=MANUAL_MODE, index_name="i", limit=5, query_vector=VECTOR, query_filter={"scope": "s"}
        )

        assert stages[0]["$vectorSearch"]["filter"] == {"scope": "s"}

    def test_the_text_filter_goes_in_a_match_stage(self) -> None:
        stages = _pipeline.text_stages(query_text="q", index_name="i", limit=5, query_filter={"scope": "s"})

        assert stages[1] == {"$match": {"scope": "s"}}
        assert "filter" not in stages[0]["$search"]

    def test_an_empty_filter_is_omitted_from_the_vector_stage(self) -> None:
        """``{}`` means no predicate, and ``$vectorSearch`` with an empty ``filter`` is a different
        query shape for the same intent."""
        stages = _pipeline.vector_stages(
            mode=MANUAL_MODE, index_name="i", limit=5, query_vector=VECTOR, query_filter={}
        )

        assert "filter" not in stages[0]["$vectorSearch"]


class TestTheHybridTierIsOneNativeStage:
    """AT-144 / ``REQ-E-142`` — the assertion that gates a pull request on this feature.

    The construction test that would prove the most, ``AT-143``, cannot run here: the input pipelines
    contain ``$vectorSearch``, which needs mongot, so the CI container fails on the input rather than
    on the stage and proves nothing about either. What runs on every change is the emitted shape, and
    the two properties it pins are the two that a reasonable rewrite would break.
    """

    def _hybrid(self, **overrides: Any) -> list[dict[str, Any]]:
        """Build a hybrid pipeline with both tiers answerable, overriding one keyword at a time."""
        kwargs: dict[str, Any] = {
            "mode": MANUAL_MODE,
            "vector_index_name": "vi",
            "text_index_name": "ti",
            "query_text": "q",
            "query_vector": VECTOR,
            "limit": 5,
        }
        kwargs.update(overrides)
        return _pipeline.hybrid_stages(**kwargs)

    def test_the_fusion_is_one_stage_plus_the_score_the_server_does_not_project(self) -> None:
        """Two stages, and the second is not optional. ``$rankFusion`` returns fused documents with no
        score field, so without this ``$set`` every hybrid result would map to ``0.0`` and a configured
        ``min_score`` would return nothing while the collection was full."""
        stages = self._hybrid()

        assert _names(stages) == ["$rankFusion", "$set"]
        assert stages[1] == {"$set": {"score": {"$meta": "score"}}}

    def test_the_outer_projection_uses_neither_tiers_score_name(self) -> None:
        """The fused score has a third ``$meta`` name of its own, and asking for either tier's name here
        is the mistake the single-tier builders already guard against on their own paths."""
        serialized = json.dumps(self._hybrid())

        assert "vectorSearchScore" not in serialized
        assert "searchScore" not in serialized

    def test_the_two_tiers_arrive_as_named_input_pipelines(self) -> None:
        """Named, because ``$rankFusion`` addresses weights and ``scoreDetails`` by pipeline name and a
        positional list would leave nothing to address."""
        fusion = self._hybrid()[0]["$rankFusion"]

        assert set(fusion["input"]["pipelines"]) == {"vector", "text"}

    def test_the_score_projection_is_outside_the_fusion_not_inside_it(self) -> None:
        """The same stage type in the two places it can go, one of which the server rejects. Asserting
        both halves together is what makes this a placement test rather than two absence tests."""
        stages = self._hybrid()
        pipelines = stages[0]["$rankFusion"]["input"]["pipelines"]

        assert _names(stages)[-1] == "$set"
        assert not any("$set" in _names(branch) for branch in pipelines.values())

    def test_no_union_reaches_the_server_on_this_path(self) -> None:
        """``REQ-E-142``'s prohibition, asserted on the serialized pipeline rather than on the top-level
        stage names, because a ``$unionWith`` reintroduced by a rewrite would appear *inside* an input
        pipeline and the outer stage list would still read as one ``$rankFusion``."""
        assert "$unionWith" not in json.dumps(self._hybrid())

    def test_neither_input_pipeline_carries_a_score_projection(self) -> None:
        """The one subtlety in the construction. ``$rankFusion`` ranks by position within each input
        pipeline, so a ``$set`` computing ``$meta`` there is read by nothing and lands in no output
        document — and the server rejects the non-selection stage outright."""
        pipelines = self._hybrid()[0]["$rankFusion"]["input"]["pipelines"]

        for name, stages in pipelines.items():
            assert "$set" not in _names(stages), name

    def test_the_input_pipelines_are_the_builders_this_package_already_has(self) -> None:
        """Composition asserted as equality against the two builders called with ``score=False``, which
        is what makes the filter placement and the ``numCandidates`` arithmetic the same arithmetic on
        this path as on the single-tier paths."""
        pipelines = self._hybrid(query_filter={"scope": "s"})[0]["$rankFusion"]["input"]["pipelines"]

        assert pipelines["vector"] == _pipeline.vector_stages(
            mode=MANUAL_MODE,
            index_name="vi",
            limit=5,
            query_vector=VECTOR,
            query_filter={"scope": "s"},
            score=False,
        )
        assert pipelines["text"] == _pipeline.text_stages(
            query_text="q", index_name="ti", limit=5, query_filter={"scope": "s"}, score=False
        )

    def test_the_vector_tier_still_validates_its_candidate_count(self) -> None:
        """Reuse has to include the rejections, not only the happy path: a hybrid builder that assembled
        the stage dicts itself would silently accept a limit the single-tier path refuses."""
        with pytest.raises(ValueError, match="numCandidates"):
            self._hybrid(limit=_pipeline.MAX_NUM_CANDIDATES)

    def test_a_query_contradicting_the_mode_is_rejected_here_too(self) -> None:
        with pytest.raises(ValueError):
            self._hybrid(mode=AUTO_EMBED_MODE, query_vector=VECTOR)


class TestTheScoreKeywordIsTheOnlyDifference:
    """AT-144's other half — the keyword ``REQ-E-142`` adds to the two existing builders.

    Default ``True``, so every existing caller emits what it emitted before; the hybrid builder is the
    only thing that passes ``False``. A default flipped the other way would have been a silent behaviour
    change at two call sites for the benefit of one new one.
    """

    def test_the_vector_tier_keeps_its_score_stage_by_default(self) -> None:
        stages = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=1, query_vector=VECTOR)

        assert stages[-1] == {"$set": {"score": {"$meta": "vectorSearchScore"}}}

    def test_the_vector_tier_drops_only_the_score_stage(self) -> None:
        with_score = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=1, query_vector=VECTOR)
        without = _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=1, query_vector=VECTOR, score=False)

        assert without == with_score[:-1]

    def test_the_text_tier_drops_only_the_score_stage(self) -> None:
        """The text tier's ``$limit`` comes after its score stage, so "drop the last one" is wrong here
        and a slice would have passed on the vector tier and corrupted this one."""
        with_score = _pipeline.text_stages(query_text="q", index_name="i", limit=1)
        without = _pipeline.text_stages(query_text="q", index_name="i", limit=1, score=False)

        assert _names(with_score) == ["$search", "$set", "$limit"]
        assert _names(without) == ["$search", "$limit"]
        assert without == [stage for stage in with_score if "$set" not in stage]

    def test_the_text_tier_keeps_its_filter_when_the_score_goes(self) -> None:
        stages = _pipeline.text_stages(
            query_text="q", index_name="i", limit=1, query_filter={"scope": "s"}, score=False
        )

        assert _names(stages) == ["$search", "$match", "$limit"]


class TestAQueryContradictingTheModeIsRejected:
    """AT-074a / REQ-E-074a.

    One of the two directions is measured loud (``queryVector must be present``, code 8) and the other
    cannot be measured anywhere CI reaches, because an ``autoEmbed`` index needs a registered model.
    Both are rejected before the round trip rather than one guessed to behave like the other.
    """

    def test_a_text_query_on_the_manual_path_raises(self) -> None:
        with pytest.raises(ValueError) as caught:
            _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=5, query_text="q")

        message = str(caught.value)
        assert "embedding_dimensions" in message, "the message has to name the key that chose the path"
        assert "'text'" in message, "and the query field the caller actually supplied"

    def test_a_vector_query_on_the_autoembeddings_path_raises(self) -> None:
        with pytest.raises(ValueError) as caught:
            _pipeline.vector_stages(mode=AUTO_EMBED_MODE, index_name="i", limit=5, query_vector=VECTOR)

        message = str(caught.value)
        assert "embedding_model" in message
        assert "'vector'" in message

    def test_neither_a_text_nor_a_vector_query_raises(self) -> None:
        """``MongoDBSearchQuery`` marks every field optional because the valid combinations are not
        expressible in a ``TypedDict``, so an empty query reaches here as a type-correct value."""
        with pytest.raises(ValueError):
            _pipeline.vector_stages(mode=MANUAL_MODE, index_name="i", limit=5)

    @pytest.mark.parametrize("mode", [AUTO_EMBED_MODE, MANUAL_MODE], ids=["autoembed", "manual"])
    def test_both_at_once_raises_for_being_both_and_not_for_the_mode(self, mode: Any) -> None:
        """Supplying both is not a shape the server has an interpretation for, and picking one for the
        caller means silently ignoring the other.

        The message is asserted, and both modes are, because the first version of this test was passing
        for the wrong reason. It used the manual mode and checked only that ``ValueError`` came out.
        Deleting the both-set branch entirely left it green: with both fields set, the mode check fires
        instead and raises about the wrong one of the two. Mutation testing is what
        surfaced that.
        """
        with pytest.raises(ValueError, match="both 'text' and 'vector'"):
            _pipeline.vector_stages(mode=mode, index_name="i", limit=5, query_text="q", query_vector=VECTOR)


@pytest.mark.integration
class TestTheBoundsThisPackageCopiesFromTheServer:
    """AT-072b / REQ-E-072 — the server boundary, for a constant rather than for a call.

    ``MAX_NUM_CANDIDATES`` is a number read off a server and written into this package. That is the
    same kind of claim ``MAX_DIMENSIONS`` makes, and the same kind that goes stale silently.

    **Re-marked from ``contract`` to ``integration``.** It needs no search *index*, which is
    why both tests below name one that does not exist, and the original marker followed from that. It
    does need search *support*. Measured against ``mongo:8``, reported version 8.3.8 with
    ``modules: []``: both cases come back as ``OperationFailure`` code 31082, *Using $search and
    $vectorSearch aggregation stages requires additional configuration*, rather than as the bound and
    the code asserted here. ``conftest.py``'s tier table has always said ``contract`` means any real
    MongoDB including a plain ``mongod``, so this class was relying on ``MONGODB_URI`` happening to
    point at a deployment with mongot.
    """

    async def test_the_ceiling_is_the_ceiling_the_server_enforces(
        self, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """Sent without a vector index on purpose. The range check fires before the index lookup, so the
        failure mode distinguishes *out of bounds* from *no such index*, and the value is what we assert
        on.

        Both checks live in the search process, though, which is why the guard below is not decoration.
        Measured against a deployment whose ``mongotHost`` pointed at a closed port: the answer is
        ``HostUnreachable`` naming that port, because there is nothing left to enforce the bound with
        (`AT-168` / `REQ-E-162`). A test that catches its own failure has to ask -- the phase hooks in
        ``conftest.py`` only see exceptions that escape.
        """
        collection = atlas_database[unique_collection_name]
        await collection.insert_one({"_id": "seed"})

        def stage(num_candidates: int) -> list[dict[str, Any]]:
            return [
                {
                    "$vectorSearch": {
                        "index": "absent_index",
                        "path": _indexes.VECTOR_FIELD,
                        "queryVector": VECTOR,
                        "numCandidates": num_candidates,
                        "limit": 1,
                    }
                }
            ]

        with pytest.raises(OperationFailure) as caught:
            await (await collection.aggregate(stage(_pipeline.MAX_NUM_CANDIDATES + 1))).to_list()

        skip_if_search_is_unreachable(caught.value)
        assert f"[1..{_pipeline.MAX_NUM_CANDIDATES}]" in str(caught.value)

    async def test_a_non_positive_limit_is_the_error_the_message_claims(
        self, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """The comment in ``candidate_count`` names the server's code for this, and a code in a comment
        is a claim like any other. The code changed between majors: 7912700 on 8.0, 65137 on 9.0.

        Guarded like the test above (`AT-168`), though on the same crippled deployment this one still passes:
        ``mongod`` rejects a non-positive ``limit`` before it dispatches anything, so the outage never
        reaches it. That ordering is the server's business and not a thing this package asserts, which
        is the whole reason the guard is here rather than an assumption.
        """
        collection = atlas_database[unique_collection_name]
        await collection.insert_one({"_id": "seed"})

        with pytest.raises(OperationFailure) as caught:
            await (
                await collection.aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": "absent_index",
                                "path": _indexes.VECTOR_FIELD,
                                "queryVector": VECTOR,
                                "numCandidates": 10,
                                "limit": 0,
                            }
                        }
                    ]
                )
            ).to_list()

        skip_if_search_is_unreachable(caught.value)
        assert caught.value.code in (7912700, 65137)


QUERY_TEXT = "dark mode toggle"
"""One query for every integration test here, so ranking differences come from the corpus."""


async def _seed_and_index(
    database: Any, name: str, *, count: int, in_scope_every: int, rank_in_scope_last: bool = False
) -> Any:
    """Build the corpus both search tiers query, with both indexes ready.

    Every document matches the text query, and one in ``in_scope_every`` is in scope, which is the
    arrangement that makes the post-filter ordering observable: a filter that removes nothing proves
    nothing about `REQ-E-073`.

    ``rank_in_scope_last`` is what makes `AT-073b` deterministic. With uniform text every document
    scores within rounding distance of every other, so which documents land in a truncated prefix is
    not stable across index builds: the same 20x over-fetch measured 5, 5 and 3 results on three
    corpora built by identical code. Repeating the query terms in the out-of-scope documents and
    padding the in-scope ones instead makes BM25 rank every in-scope document below every out-of-scope
    one, which is the worst case a fixed over-fetch has to survive and is stable to assert on.

    Args:
        database: The ``atlas_database`` fixture.
        name: Collection name, from ``unique_collection_name``.
        count: How many documents to write.
        in_scope_every: Every nth document is in scope ``"keep"``; the rest are ``"drop"``.
        rank_in_scope_last: Score the in-scope documents below every out-of-scope one.

    Returns:
        The collection, with the vector and full-text indexes both READY.
    """

    def content(index: int, *, in_scope: bool) -> str:
        if not rank_in_scope_last:
            return f"{QUERY_TEXT} note number {index}"
        if in_scope:
            # Terms once, then padding: BM25's length normalisation ranks these last.
            return QUERY_TEXT + " " + " ".join(f"padding{n}" for n in range(60))
        return " ".join([QUERY_TEXT] * 5)

    collection = database[name]
    await collection.insert_many(
        [
            {
                "_id": index,
                _indexes.CONTENT_FIELD: content(index, in_scope=index % in_scope_every == 0),
                "scope": "keep" if index % in_scope_every == 0 else "drop",
                # Aligned with `VECTOR` on the first axis and separated on the second, so cosine
                # similarity orders them by `index` and the expected ranking is computable.
                _indexes.VECTOR_FIELD: [1.0, float(index) / count, 0.0, 0.0],
            }
            for index in range(count)
        ]
    )
    await _indexes.ensure(collection, search=_indexes.memory_search_specs(MANUAL_VECTOR_CONFIG), create=True)
    return collection


SEARCHABLE_BUDGET: Final[float] = 90.0
"""How long a document is waited on before it counts as never having been indexed.

The same figure ``AT-086`` and ``test_memory_store.py`` use, for the same reason. The corpus
here is seeded before the index is created, so what it waits out is the initial sync rather than
`REQ-E-091`'s embedding lag, and on every deployment measured so far that finishes as the index reports
``READY``. The budget is generous anyway: it costs nothing when the premise holds, since every caller
returns on the first non-empty result, and the 10-second version of this constant in
``test_memory_store.py`` is what made six tests fail on 2026-09-02 the first time the suite met a cloud
cluster. Two helpers waiting out one boundary should not disagree about how long it takes.
"""


async def _until_answered(
    collection: Any, pipeline: list[dict[str, Any]], *, budget: float = SEARCHABLE_BUDGET
) -> list[Any]:
    """Run a search pipeline until mongot has the corpus, or give up.

    READY means the index exists, not that every document has reached it, so a search issued
    immediately after ``ensure`` can legitimately return nothing. Polling here rather than sleeping a
    fixed interval, because the wait is a property of the deployment and a fixed sleep is either flaky
    or slow.

    Args:
        collection: Collection to query.
        pipeline: Stages to run.
        budget: Seconds to keep asking before giving up, polling twice a second.

    Returns:
        The first non-empty result, or the last empty one if the index never caught up.
    """
    results: list[Any] = []
    deadline = time.monotonic() + budget
    while True:
        try:
            results = await (await collection.aggregate(pipeline)).to_list()
        except PyMongoError as error:
            # `REQ-E-124`. This helper drives the driver directly, so the code is on the error itself
            # rather than on a cause; the guard reads both, which is why one helper serves both files.
            skip_if_search_is_unreachable(error)
            raise
        if results or time.monotonic() >= deadline:
            return results
        await asyncio.sleep(0.5)


@pytest.mark.integration
class TestTheOrderingAgainstMongot:
    """AT-073, AT-073b, AT-073c / REQ-E-073, REQ-E-073a — ordering against mongot, and the numbers that
    changed a requirement.

    The three pipelines here contain the same stages and differ only in their order. One returns the
    requested count; the other two return fewer, and one of them returns nothing at all. Nothing in the
    unit tier can tell them apart, because the difference is what mongot does with a post-filter.
    """

    async def test_the_emitted_order_returns_the_requested_count(
        self, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """AT-073. 200 documents match the text, 10 are in scope, and 5 are asked for."""
        collection = await _seed_and_index(atlas_database, unique_collection_name, count=200, in_scope_every=20)
        stages = _pipeline.text_stages(
            query_text=QUERY_TEXT,
            index_name=_indexes.DEFAULT_TEXT_INDEX_NAME,
            limit=5,
            query_filter={"scope": "keep"},
        )

        results = await _until_answered(collection, stages)

        assert len(results) == 5
        assert {document["scope"] for document in results} == {"keep"}

    async def test_no_pre_filter_bound_below_the_corpus_size_fills_the_limit(
        self, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """AT-073b. Asserting what the wrong implementations do, because `REQ-E-073` prescribed one of
        them: an over-fetch ahead of the ``$match``.

        Six multipliers from 1x to 38x, on a corpus where BM25 ranks all 10 in-scope documents below all
        190 out-of-scope ones. Every one of them returns nothing. The smallest bound that fills the limit
        is 200, the corpus size, which is not a number available when the stage is built and stops being
        a bound at all once it is. That is the finding `REQ-E-073a` records, and it is stronger than the
        one the requirement was written against.
        """
        collection = await _seed_and_index(
            atlas_database, unique_collection_name, count=200, in_scope_every=20, rank_in_scope_last=True
        )
        emitted = _pipeline.text_stages(
            query_text=QUERY_TEXT,
            index_name=_indexes.DEFAULT_TEXT_INDEX_NAME,
            limit=5,
            query_filter={"scope": "keep"},
        )
        search = emitted[0]
        keep = {"$match": {"scope": "keep"}}

        async def bounded_at(size: int) -> int:
            stages = [search, {"$limit": size}, keep, {"$limit": 5}]
            return len(await (await collection.aggregate(stages)).to_list())

        assert len(await _until_answered(collection, emitted)) == 5
        # 1x is the naive ordering and 38x is a 190-document over-fetch to return 5 results.
        assert {multiplier: await bounded_at(5 * multiplier) for multiplier in (1, 2, 5, 10, 20, 38)} == {
            multiplier: 0 for multiplier in (1, 2, 5, 10, 20, 38)
        }
        assert await bounded_at(200) == 5

    async def test_the_cost_of_not_bounding_early_tracks_the_filter_and_not_the_limit(
        self, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """AT-073c / REQ-E-073a's cost claim, which is the objection to having no early ``$limit``.

        ``internalMongotBatchSizeHistory`` is what the aggregation actually requested from mongot rather
        than what it returned, so this asserts the pull protocol rather than trusting it. Measured: the
        first request is always 101 documents, whatever the limit and whatever the corpus size, and each
        further request is 1.5x the last (101, 152, 228, 342). It stops once ``$limit`` is satisfied.

        Asserted against two limits 50x apart because their histories are identical, which is the
        precise form of the claim: the escalation is driven by how many documents survive the ``$match``,
        not by how many were asked for. On this corpus, where nothing survives until rank 190, a limit of
        1 costs exactly what a limit of 50 costs.

        The favourable direction is not asserted here because it is not reproducible. With uniform text
        a 5-result query measured ``[101]`` on some corpora and ``[101, 152]`` on others built by
        identical code, for the tie-ordering reason in ``_seed_and_index``. The bound that holds is the
        one above.

        Two explain shapes report it. 8.0's ``$_internalSearchMongotRemote`` lists every request size.
        9.0 runs search as an extension, under ``$_internalDocumentResultsAndMetadata``, and reports the
        first request's size and the number of requests, the second only at ``executionStats``. Measured on
        9.0.4: a first request of 108 and two requests, for both limits. Skips where neither is reported:
        Atlas 8.0.30's remote stage carries ``explain``, ``mongotQuery`` and ``requiresSearchMetaCursor``
        and nothing else, so the pull protocol is unobservable there rather than different.
        """
        collection = await _seed_and_index(
            atlas_database, unique_collection_name, count=200, in_scope_every=20, rank_in_scope_last=True
        )

        async def history(limit: int) -> dict[str, Any]:
            stages = _pipeline.text_stages(
                query_text=QUERY_TEXT,
                index_name=_indexes.DEFAULT_TEXT_INDEX_NAME,
                limit=limit,
                query_filter={"scope": "keep"},
            )
            await _until_answered(collection, stages)
            command = {"aggregate": unique_collection_name, "pipeline": stages, "cursor": {}}
            explained = await atlas_database.command("explain", command, verbosity="executionStats")
            for stage in explained["stages"]:
                if "$_internalSearchMongotRemote" in stage:
                    remote = stage["$_internalSearchMongotRemote"]
                    if "internalMongotBatchSizeHistory" in remote:
                        return {"requests": list(remote["internalMongotBatchSizeHistory"])}
                    pytest.skip(
                        "this deployment's explain omits internalMongotBatchSizeHistory: Atlas 8.0.30 reports "
                        f"only {sorted(remote)}, so the pull protocol is unobservable here"
                    )
                if "$_internalDocumentResultsAndMetadata" in stage:
                    search = stage["$_internalDocumentResultsAndMetadata"]["source"]["$search"]["explain"]
                    return {
                        "first": search["metadata"]["cursorOptions"]["batchSize"],
                        "requests": search["resourceUsage"]["numBatches"],
                    }
            pytest.skip(f"no search stage this test can read in {[next(iter(s)) for s in explained['stages']]}")

        for_one = await history(1)
        for_fifty = await history(50)

        assert for_one == for_fifty, "the limit changed what was pulled from mongot"
        assert for_one in ({"requests": [101, 152]}, {"first": 108, "requests": 2})


@pytest.mark.integration
class TestBothTiersAgainstMongot:
    """AT-074, AT-070c / REQ-E-074 — mongot's ordering for the vector half.

    Manual-vector mode throughout, for the reason ``test_indexes.TestSearchIndexesAgainstAtlas`` gives:
    an ``autoEmbed`` definition fails at creation on a deployment with no registered model, so the
    default path is unreachable here and ``AT-070i`` carries it.
    """

    async def test_a_precomputed_embedding_returns_the_same_shape_as_the_text_path(
        self, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """AT-074 / REQ-E-074, asserted as *the same shape* rather than as "it works".

        The requirement is about parity between the two paths, so the assertion compares them: same
        keys on the documents, same score field, same type. A test that only checked the vector path
        returned something would pass with a score field the text path does not have.
        """
        collection = await _seed_and_index(atlas_database, unique_collection_name, count=50, in_scope_every=5)
        vector = _pipeline.vector_stages(
            mode=MANUAL_MODE,
            index_name=_indexes.DEFAULT_VECTOR_INDEX_NAME,
            limit=5,
            query_vector=VECTOR,
            query_filter={"scope": "keep"},
        )
        text = _pipeline.text_stages(
            query_text=QUERY_TEXT,
            index_name=_indexes.DEFAULT_TEXT_INDEX_NAME,
            limit=5,
            query_filter={"scope": "keep"},
        )

        from_vector = await _until_answered(collection, vector)
        from_text = await _until_answered(collection, text)

        assert len(from_vector) == len(from_text) == 5
        assert {frozenset(document) for document in from_vector} == {frozenset(document) for document in from_text}
        assert all(isinstance(document[_pipeline.SCORE_FIELD], float) for document in from_vector)
        assert all(isinstance(document[_pipeline.SCORE_FIELD], float) for document in from_text)

    async def test_the_vector_filter_is_a_pre_filter_and_still_fills_the_limit(
        self, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """The asymmetry behind there being no ``$limit`` stage on the vector tier.

        ``$vectorSearch``'s ``filter`` is applied during the search rather than after it, so the stage's
        own ``limit`` is a count of documents that already passed the filter. Same corpus proportions
        that leave the text tier's post-filter needing the ordering, and here 5 is simply 5.
        """
        collection = await _seed_and_index(atlas_database, unique_collection_name, count=200, in_scope_every=20)
        stages = _pipeline.vector_stages(
            mode=MANUAL_MODE,
            index_name=_indexes.DEFAULT_VECTOR_INDEX_NAME,
            limit=5,
            query_vector=VECTOR,
            query_filter={"scope": "keep"},
        )

        results = await _until_answered(collection, stages)

        assert len(results) == 5
        assert {document["scope"] for document in results} == {"keep"}
        assert _names(stages) == ["$vectorSearch", "$set"]

    @pytest.mark.parametrize(
        ("tier", "crossed_meta"),
        [("vector", _pipeline.TEXT_SCORE_META), ("text", _pipeline.VECTOR_SCORE_META)],
    )
    async def test_the_crossed_score_name_never_yields_a_usable_score(
        self, atlas_database: Any, unique_collection_name: str, tier: str, crossed_meta: str
    ) -> None:
        """AT-070c. The failure mode of getting ``$meta`` wrong, asserted in both directions.

        Worth a test because the plausible worry is that a wrong metadata name yields nothing and ships a
        store whose scores are all missing. **Both deployments were measured and they disagree**, so the
        worry is right on one of them:

        * mongot 8.3.8 refuses the pipeline, naming the unavailable metadata field.
        * Atlas 8.0.30 accepts it and returns documents with no ``score`` field at all. Composed with
          ``memory_store``'s deliberate ``float(document.get(SCORE_FIELD) or 0.0)``, that is a uniform
          zero relevance score on every result and no error anywhere.

        The invariant both satisfy is the one asserted: a crossed name never produces a score a caller
        could mistake for a real one. What keeps the mistake out of the package is not the server but that
        both names are constants, pinned to their literal values by ``TestTheScoreProjection``.
        """
        collection = await _seed_and_index(atlas_database, unique_collection_name, count=20, in_scope_every=2)
        if tier == "vector":
            good = _pipeline.vector_stages(
                mode=MANUAL_MODE, index_name=_indexes.DEFAULT_VECTOR_INDEX_NAME, limit=3, query_vector=VECTOR
            )
        else:
            good = _pipeline.text_stages(query_text=QUERY_TEXT, index_name=_indexes.DEFAULT_TEXT_INDEX_NAME, limit=3)
        answered = await _until_answered(collection, good)
        assert all(isinstance(document[_pipeline.SCORE_FIELD], float) for document in answered)

        crossed = [good[0], {"$set": {_pipeline.SCORE_FIELD: {"$meta": crossed_meta}}}]
        try:
            crossed_results = await (await collection.aggregate(crossed)).to_list()
        except OperationFailure as refused:
            assert "score metadata, but it is not available" in str(refused)
        else:
            assert crossed_results, "the crossed pipeline answered nothing, so it proves nothing"
            assert all(_pipeline.SCORE_FIELD not in document for document in crossed_results)


@pytest.mark.integration
@pytest.mark.autoembed
class TestTheAutoEmbeddingsQueryAgainstADeploymentThatCanAnswerIt:
    """AT-070i — the test that settled the query shape, and it took a cloud cluster to run it.

    Skips wherever ``MONGODB_AUTOEMBED_URI`` is unset, which was everywhere for the whole build: mongot
    8.3.8 local reports zero registered models, so the ``autoEmbed`` index cannot be created and the
    query shape could not be adjudicated. `AT-070` pins what we emit; only this test can say whether the
    server agrees.

    It first ran on 2026-09-01 against Atlas 8.0.30 and passed in 76.59s, most of which is the index
    build: ``PENDING`` at t+0, ``BUILDING`` at t+60s, queryable at t+80s. The question is closed in favour of
    the shape this package already emitted.
    """

    async def test_the_server_accepts_the_stage_we_emit(
        self, autoembed_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """If this fails with a query-shape error, the question reopens against ``pymongo_search_utils`` and the
        fix is the ``query`` key in ``_pipeline.vector_stages``, plus its `AT-070` assertion."""
        client: AsyncMongoClient[Any] = AsyncMongoClient(autoembed_uri)
        try:
            collection = client[test_database_name][unique_collection_name]
            await collection.insert_many(
                [{"_id": n, _indexes.CONTENT_FIELD: f"{QUERY_TEXT} note {n}", "scope": "keep"} for n in range(20)]
            )
            await _indexes.ensure(
                collection, search=_indexes.memory_search_specs({"database": test_database_name}), create=True
            )
            stages = _pipeline.vector_stages(
                mode=AUTO_EMBED_MODE,
                index_name=_indexes.DEFAULT_VECTOR_INDEX_NAME,
                limit=5,
                query_text="dark mode",
                query_filter={"scope": "keep"},
            )

            results = await _until_answered(collection, stages)

            assert len(results) == 5
            assert all(isinstance(document[_pipeline.SCORE_FIELD], float) for document in results)
        finally:
            await client.drop_database(test_database_name)
            await client.close()
