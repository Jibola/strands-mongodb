"""Acceptance tests for the public configuration, query and result types.

These types are the configuration surface a caller writes against: the store config, the connection
config, the search query and the result of ``add``.

Most of what ``types.py`` promises is unobservable at runtime. A ``Required`` key and a narrow ``Literal``
are instructions to a type checker; the interpreter ignores both. So the tests split by who enforces the
property: ``dataclasses.fields`` and ``__annotations__`` for what the interpreter knows, and a real
``mypy`` run for what only it knows.

The split matters most for required keys. Under ``from __future__ import annotations``, which this package
and the SDK both use everywhere, ``TypedDict`` cannot see ``Required`` at all: on ``strands-agents``
1.54.0, ``MemoryStoreConfig.__required_keys__`` is an **empty** frozenset, with ``name`` in
``__optional_keys__`` despite being declared ``Required[str]``. Asserting ``"name" in __required_keys__``
would therefore fail against a correct implementation. Dropping the future import from this package's
module would fix its own keys, leave the inherited ``name`` unseen, and break the module on import (see
``TestTheModuleImportsWithoutADriver``). Mypy is what enforces required keys.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, get_args

import pytest
from strands.memory import MemoryStoreConfig

from strands_mongodb.types import (
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_OVERSAMPLING_FACTOR,
    RESERVED_METADATA_KEYS,
    MongoDBAddResult,
    MongoDBConfig,
    MongoDBMemoryStoreConfig,
    MongoDBSearchQuery,
    ResolvedSearchType,
    SearchType,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestAddResultReportsWhatTheWriteDid:
    """AT-056 — ``add`` returns a dataclass, not a bare count or an id list."""

    def test_it_is_a_dataclass_carrying_the_three_declared_fields_in_order(self) -> None:
        """Field order is part of the shape, because a dataclass is positionally constructible.

        ``MongoDBAddResult(["a"], 1, 0)`` is a call a caller can write, so reordering the fields is a
        silent breaking change that no keyword-only test would catch.
        """
        assert dataclasses.is_dataclass(MongoDBAddResult)
        assert [f.name for f in dataclasses.fields(MongoDBAddResult)] == ["ids", "inserted", "duplicates"]

    def test_it_round_trips_the_values_it_was_given(self) -> None:
        """The values a caller passes are the values the result reports."""
        result = MongoDBAddResult(ids=["6512", "6513"], inserted=1, duplicates=0)

        assert result.ids == ["6512", "6513"]
        assert result.inserted == 1
        assert result.duplicates == 0

    def test_duplicates_has_no_default_so_a_caller_cannot_be_told_zero_by_omission(self) -> None:
        """``duplicates`` is how ``add`` reports facts it skipped because the store already held them.

        A default of ``0`` would let a code path that never counted duplicates construct a result
        claiming there were none. Every construction site has to answer the question.
        """
        by_name = {f.name: f for f in dataclasses.fields(MongoDBAddResult)}

        assert by_name["duplicates"].default is dataclasses.MISSING
        assert by_name["duplicates"].default_factory is dataclasses.MISSING


class TestMemoryStoreConfigExtendsTheSdkConfig:
    """AT-056a — the store config is the SDK's config plus MongoDB's fields."""

    def test_it_carries_every_key_the_sdk_config_declares(self) -> None:
        """``MemoryManager`` reads ``name``, ``description``, ``max_search_results``, ``writable``
        and ``extraction``. A config that dropped one would leave the store unroutable.

        Asserted through ``__annotations__``, which ``TypedDict`` merges across bases, because a
        ``TypedDict`` subclass does not keep its base in ``__mro__`` and ``issubclass`` is therefore
        not available to ask this question.
        """
        assert set(MemoryStoreConfig.__annotations__) <= set(MongoDBMemoryStoreConfig.__annotations__)

    def test_it_adds_the_documented_mongodb_fields(self) -> None:
        """The eight fields beyond the SDK's, named so a rename is deliberate.

        Widening this set is the expected way for the config to grow, and the assertion is written as
        equality anyway, so a new field is a line in a diff rather than something a reviewer has to notice
        the absence of.
        """
        added = set(MongoDBMemoryStoreConfig.__annotations__) - set(MemoryStoreConfig.__annotations__)

        assert added == {
            "config",
            "collection",
            "scope",
            "filter",
            "min_score",
            "max_score",
            "ttl_seconds",
            "embedder",
        }

    def test_mypy_requires_both_name_and_config(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """The half of ``AT-056a`` the interpreter cannot answer.

        ``name`` is inherited as ``Required[str]`` from the SDK and ``config`` is declared
        ``Required`` here, so a literal missing either is an error. Both are asserted in one run
        because a checker that had lost sight of the base class entirely would still flag ``config``.
        """
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBMemoryStoreConfig

            no_name: MongoDBMemoryStoreConfig = {"config": {"database": "agents"}}
            no_config: MongoDBMemoryStoreConfig = {"name": "facts"}
            """
        )

        assert result.returncode == 1, result.stdout
        assert 'Missing key "name" for TypedDict "MongoDBMemoryStoreConfig"' in result.stdout, result.stdout
        assert 'Missing key "config" for TypedDict "MongoDBMemoryStoreConfig"' in result.stdout, result.stdout

    def test_mypy_accepts_a_complete_literal(self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]) -> None:
        """The positive case, without which the test above passes against a config that rejects
        everything."""
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBMemoryStoreConfig

            config: MongoDBMemoryStoreConfig = {
                "name": "facts",
                "max_search_results": 5,
                "config": {"database": "agents", "connection_string": "mongodb://localhost"},
                "scope": "user:1",
            }
            """
        )

        assert result.returncode == 0, result.stdout


class TestDatabaseIsTheOneMandatoryConnectionField:
    """AT-056b — ``database`` is the one connection field a config cannot omit."""

    def test_the_config_declares_the_documented_fields(self) -> None:
        """The connection config has exactly these thirteen keys, so any change to the set is deliberate.

        A new key is cheap to add and impossible to remove once a caller sets it, so the list is spelled out
        here and each key beyond the ten core connection and index settings has its reason written down.

        ``embedding_dimensions`` and ``similarity`` are what a manual vector index needs at creation time.
        An ``autoEmbed`` index is created only on deployments that run automated embedding, so without
        these two keys the package could build no vector index anywhere else.

        ``index_ready_timeout`` is how long a caller waits for a search index to become queryable. It is a
        key rather than a constant because a first build on a shared cloud cluster can take longer than
        any fixed default: a wait long enough for one deployment is a failure on another, and only the
        caller knows which they have. A further key needs its reason added here.
        """
        assert set(MongoDBConfig.__annotations__) == {
            "database",
            "client",
            "connection_string",
            "client_args",
            "ensure_indexes",
            "search_type",
            "embedding_model",
            "filter_fields",
            "vector_index_name",
            "text_index_name",
            "embedding_dimensions",
            "similarity",
            "index_ready_timeout",
        }

    def test_the_two_added_fields_are_optional_and_typed(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """The zero-configuration path stays the Atlas one: a caller who sets neither field gets the
        ``autoEmbed`` mode and never learns these exist. ``embedding_dimensions`` is an ``int``, and a
        string there is the mistake worth catching, since the server's own answer to it arrives after
        the collection and five other indexes have been created.
        """
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBConfig

            default: MongoDBConfig = {"database": "agents"}
            manual: MongoDBConfig = {
                "database": "agents",
                "embedding_dimensions": 1024,
                "similarity": "cosine",
            }
            wrong: MongoDBConfig = {"database": "agents", "embedding_dimensions": "1024"}
            """
        )

        assert result.returncode == 1, result.stdout
        assert result.stdout.count("error:") == 1, result.stdout
        assert "embedding_dimensions" in result.stdout, result.stdout

    def test_mypy_rejects_a_config_literal_without_a_database(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """A config with a connection string and no database names a server, not a place to write.

        Runtime detection is not available: ``__required_keys__`` is empty under postponed
        annotations, so nothing raises when the key is absent. The failure would surface as a
        ``KeyError`` inside the first query instead.
        """
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBConfig

            config: MongoDBConfig = {"connection_string": "mongodb://localhost"}
            """
        )

        assert result.returncode == 1, result.stdout
        assert 'Missing key "database" for TypedDict "MongoDBConfig"' in result.stdout, result.stdout

    def test_mypy_rejects_a_misspelled_optional_key(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """``total=False`` makes every other key optional, which is not the same as unchecked.

        This is the property that makes the config worth typing at all: a typo in an optional key is
        otherwise a setting that silently does nothing.
        """
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBConfig

            config: MongoDBConfig = {"database": "agents", "ensure_index": True}
            """
        )

        assert result.returncode == 1, result.stdout
        assert "ensure_index" in result.stdout, result.stdout


class TestTheModuleImportsWithoutADriver:
    """AT-056c — annotating a config must not construct one.

    Blocking ``pymongo`` outright is a stronger statement than checking ``sys.modules``: it fails at
    the point of import rather than reporting on collection order, and it is the environment a
    ``strands-agents``-only resolution would produce if either extra ever stopped pinning the driver.
    """

    def test_importing_the_types_module_with_pymongo_blocked_succeeds(self) -> None:
        """Two failure modes, one assertion.

        Moving ``AsyncMongoClient`` out of ``TYPE_CHECKING`` makes the guarded import fire and raises
        the ``missing_extra`` message. Dropping ``from __future__ import annotations`` raises
        ``NameError`` from the class body instead, because ``TypedDict`` evaluates its annotations
        eagerly to compute ``__required_keys__``. That holds on Python 3.14 too, where PEP 649 otherwise
        defers annotations.
        """
        source = """
            import sys


            class Blocker:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "pymongo" or fullname.startswith("pymongo."):
                        raise ImportError(f"No module named {fullname!r}", name=fullname)
                    return None


            sys.meta_path.insert(0, Blocker())

            from strands_mongodb.types import MongoDBConfig

            print(MongoDBConfig.__name__)
            print("pymongo" in sys.modules)
            """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["MongoDBConfig", "False"], result.stdout


class TestTheSearchTierVocabulary:
    """AT-056d — two literals, because asking for a tier and reporting one are different domains."""

    def test_the_requestable_tiers_are_the_four_that_run(self) -> None:
        """AT-141 — a ``Literal`` in the config is a promise the caller can typecheck against, so it
        holds only values that actually run. ``regex`` does not run, so it is absent.

        Adding a tier widens this alias without removing a value, so every config that typechecked before
        still does.

        Asserted against the named alias rather than the field annotation, which is a string under
        postponed annotations and would need resolving before ``get_args`` saw anything.
        """
        assert get_args(SearchType) == ("auto", "vector", "text", "hybrid")

    def test_the_reportable_tiers_are_the_three_that_can_answer_a_query(self) -> None:
        """AT-141 — ``_search_type`` carries the answering tier, so a fallback is visible in the
        returned data.

        The two literals differ by ``auto`` alone, which is why they are separate: ``auto`` is a request
        and can never be an answer. A single union would let a result claim it was answered by ``auto``.
        """
        assert get_args(ResolvedSearchType) == ("vector", "text", "hybrid")

    def test_the_two_vocabularies_differ_by_auto_alone(self) -> None:
        """AT-141 — the property the split exists for, asserted directly rather than inferred from
        the two tuples above, so that a widening that adds a value to one and forgets the other fails
        here and names it.
        """
        assert set(get_args(SearchType)) - set(get_args(ResolvedSearchType)) == {"auto"}
        assert not set(get_args(ResolvedSearchType)) - set(get_args(SearchType))

    def test_mypy_accepts_the_hybrid_tier_at_the_config_site(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """AT-142 — ``hybrid`` runs, so mypy accepts it as a config's ``search_type``.

        The config rejects tiers that cannot answer a query, and the next test checks that half.
        """
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBConfig

            config: MongoDBConfig = {"database": "agents", "search_type": "hybrid"}
            """
        )

        assert result.returncode == 0, result.stdout

    def test_mypy_rejects_a_tier_that_still_does_not_run(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """AT-142 — a tier that does not run is rejected by mypy at the config site, not at query
        time. ``regex`` is that tier.
        """
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBConfig

            config: MongoDBConfig = {"database": "agents", "search_type": "regex"}
            """
        )

        assert result.returncode == 1, result.stdout
        assert "regex" in result.stdout, result.stdout


class TestTheSearchQueryWidensTheProtocolQuery:
    """AT-056e — the search query type carries a pre-computed vector as well as text."""

    def test_it_declares_the_fields_the_widened_query_carries(self) -> None:
        """`Storage.search` takes one argument, so anything a caller wants to say about a query has
        to travel inside it: the text, a pre-computed embedding, a filter, and the two dials.
        """
        assert set(MongoDBSearchQuery.__annotations__) == {
            "text",
            "vector",
            "filter",
            "limit",
            "oversampling_factor",
        }

    def test_every_field_is_optional_because_the_valid_combinations_are_not_expressible(self) -> None:
        """A query needs ``text`` or ``vector``, and both only on the hybrid tier, which no ``TypedDict``
        can say.

        Marking either ``Required`` would forbid the other path. The constraint is checked at runtime
        by the search implementation instead, and this pins that the type deliberately does not try.
        """
        assert MongoDBSearchQuery.__required_keys__ == frozenset()

    def test_mypy_accepts_a_vector_only_query(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """A query carrying only a pre-computed vector typechecks."""
        result = run_mypy(
            """
            from strands_mongodb.types import MongoDBSearchQuery

            query: MongoDBSearchQuery = {"vector": [0.1, 0.2], "limit": 5}
            """
        )

        assert result.returncode == 0, result.stdout


class TestTheSharedConstants:
    """AT-056f — values other modules must not respell."""

    def test_the_four_reserved_metadata_keys_are_the_ones_the_readme_documents(self) -> None:
        """Search sets these keys on every returned entry and resolves metadata collisions against
        them, so both behaviours have to agree on one set.
        """
        assert RESERVED_METADATA_KEYS == frozenset({"_relevance_score", "_id", "_search_type", "_scope"})

    def test_the_result_default_matches_both_shipped_stores(self) -> None:
        """``FileMemoryStore`` and ``BedrockKnowledgeBaseStore`` both default to 10 results.

        Behavioural parity across stores is the reason for the number, so it is asserted rather than
        left as a literal in ``search``.
        """
        assert DEFAULT_MAX_SEARCH_RESULTS == 10

    def test_the_oversampling_default_matches_the_stage_builder_it_feeds(self) -> None:
        """``pymongo_search_utils.vector_search_stage`` defaults ``oversampling_factor`` to 10 and
        computes ``numCandidates = top_k * oversampling_factor``.

        This package derives the candidate count the same way rather than passing a constant. Matching the
        default means the derived count equals that library's for a caller who sets nothing.
        """
        assert DEFAULT_OVERSAMPLING_FACTOR == 10


class TestTheModuleDeclaresItsSurface:
    """Every name the tests above import is public, and nothing else is."""

    def test_all_lists_exactly_the_exported_names(self) -> None:
        """A type module with no ``__all__`` exports its imports too, so ``MemoryStoreConfig`` and
        ``TypedDict`` would read as part of this package's surface.
        """
        import strands_mongodb.types as types_module

        assert types_module.__all__ == [
            "DEFAULT_MAX_SEARCH_RESULTS",
            "DEFAULT_OVERSAMPLING_FACTOR",
            "MongoDBAddResult",
            "MongoDBConfig",
            "MongoDBMemoryStoreConfig",
            "MongoDBSearchQuery",
            "RESERVED_METADATA_KEYS",
            "ResolvedSearchType",
            "SearchType",
            "TextEmbedder",
            "VectorSimilarity",
        ]

    @pytest.mark.parametrize("name", ["MongoDBConfig", "MongoDBAddResult", "SearchType"])
    def test_each_exported_name_resolves(self, name: str) -> None:
        """``__all__`` entries that do not exist fail only on ``import *``, which nothing here does."""
        import strands_mongodb.types as types_module

        assert hasattr(types_module, name)
