"""``MongoDBVectorSearchStrategy``: the unit tier against the storage fake, and Atlas Vector Search for real.

The unit tier asserts what the strategy sends: the ``$set`` after a write, the ``$vectorSearch`` stage,
the index definition. The integration tier asserts what a deployment with mongot does with it, which is
the only place the prefix pre-filter and the score can be observed.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import pytest
from strands.storage.search import StorageSearchResult
from strands.types.exceptions import StorageError
from test_storage import FakeStorageCollection, unit_storage

from strands_mongodb import MongoDBVectorSearchStrategy, _indexes
from strands_mongodb.storage import MongoDBStorage
from strands_mongodb.vector_search import DEFAULT_STORAGE_VECTOR_INDEX_NAME, _ancestors

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from strands_mongodb.types import MongoDBConfig

VOCABULARY = ("dark", "mode", "coffee", "tea")


class BagOfWordsEmbedder:
    """A deterministic embedder: one dimension per word in ``VOCABULARY``, plus a floor so no vector is zero."""

    def __init__(self, dimensions: int = len(VOCABULARY)) -> None:
        self.dimensions = dimensions
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        words = text.lower().split()
        vector = [0.01 + words.count(word) for word in VOCABULARY]
        return (vector + [0.01] * self.dimensions)[: self.dimensions]


def strategy(embedder: BagOfWordsEmbedder | None = None, **overrides: Any) -> MongoDBVectorSearchStrategy:
    options: dict[str, Any] = {"dimensions": len(VOCABULARY), "similarity": "cosine"}
    options.update(overrides)
    return MongoDBVectorSearchStrategy(embedder=embedder or BagOfWordsEmbedder(), **options)


class TestConstruction:
    """AT-237 — every bound the server would enforce later is enforced at construction."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"dimensions": 0}, "'dimensions' is 0"),
            ({"dimensions": _indexes.MAX_DIMENSIONS + 1}, "'dimensions'"),
            ({"similarity": "manhattan"}, "'similarity' is 'manhattan'"),
            ({"max_results": 0}, "'max_results'"),
            ({"oversampling_factor": 0}, "'oversampling_factor'"),
            ({"max_results": 101, "oversampling_factor": 100}, "numCandidates would be 10100"),
        ],
    )
    def test_an_out_of_range_option_raises(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            strategy(**overrides)

    def test_the_index_spec_declares_the_vector_and_the_prefix_filter(self) -> None:
        spec = strategy(dimensions=4, similarity="euclidean", index_name="v").index_spec()

        assert spec.name == "v"
        assert spec.type == "vectorSearch"
        assert spec.definition == {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 4, "similarity": "euclidean"},
                {"type": "filter", "path": "prefixes"},
            ]
        }

    def test_the_default_index_name_differs_from_the_memory_stores(self) -> None:
        assert strategy().index_spec().name == DEFAULT_STORAGE_VECTOR_INDEX_NAME
        assert DEFAULT_STORAGE_VECTOR_INDEX_NAME != _indexes.DEFAULT_VECTOR_INDEX_NAME


class TestIndex:
    """AT-238 — ``write`` stores the vector on the document it just replaced."""

    async def test_a_write_sets_the_vector_and_every_ancestor_prefix(self) -> None:
        collection = FakeStorageCollection()
        storage = unit_storage(collection, prefix="tenant-a", search_strategy=strategy())

        await storage.write("notes/dark.md", b"dark mode")

        assert collection.names_called == ["replace_one", "update_one"]
        _, (filter_, update), _ = collection.calls[-1]
        assert filter_ == {"_id": "tenant-a/notes/dark.md", "data": b"dark mode"}
        assert update == {"$set": {"embedding": [1.01, 1.01, 0.01, 0.01], "prefixes": ["tenant-a/", "tenant-a/notes/"]}}

    async def test_a_slower_write_does_not_put_its_vector_on_a_newer_value(self) -> None:
        release = asyncio.Event()

        class SlowForDark(BagOfWordsEmbedder):
            async def embed(self, text: str) -> list[float]:
                if text == "dark":
                    await release.wait()
                return await super().embed(text)

        collection = FakeStorageCollection()
        storage = unit_storage(collection, search_strategy=strategy(SlowForDark()))

        slow = asyncio.create_task(storage.write("k", b"dark"))
        await asyncio.sleep(0)
        await storage.write("k", b"tea")
        release.set()
        await slow

        assert collection.documents["k"]["data"] == b"tea"
        assert collection.documents["k"]["embedding"] == await BagOfWordsEmbedder().embed("tea")

    @pytest.mark.parametrize("data", [b"", b"\xff\xfe"])
    async def test_an_empty_or_binary_value_is_stored_but_not_embedded(self, data: bytes) -> None:
        embedder = BagOfWordsEmbedder()
        collection = FakeStorageCollection()

        await unit_storage(collection, search_strategy=strategy(embedder)).write("k", data)

        assert collection.names_called == ["replace_one"]
        assert embedder.calls == []
        assert collection.documents["k"]["data"] == data

    async def test_a_vector_of_the_wrong_length_fails_the_write_after_storing_the_value(self) -> None:
        collection = FakeStorageCollection()
        storage = unit_storage(collection, search_strategy=strategy(BagOfWordsEmbedder(dimensions=3)))

        with pytest.raises(StorageError, match="3-dimension vector"):
            await storage.write("k", b"dark")

        assert collection.documents["k"]["data"] == b"dark"
        assert "update_one" not in collection.names_called

    @pytest.mark.parametrize(
        ("document_id", "expected"),
        [("k", []), ("a/k", ["a/"]), ("a/b/c", ["a/", "a/b/"])],
    )
    def test_ancestors_are_every_slash_terminated_prefix(self, document_id: str, expected: list[str]) -> None:
        assert _ancestors(document_id) == expected


class TestSearch:
    """AT-239 — one ``$vectorSearch`` aggregation, pre-filtered to the storage's prefix."""

    async def test_an_unprefixed_store_sends_no_filter(self) -> None:
        collection = FakeStorageCollection(aggregate_result=[{"_id": "notes/a", "data": b"dark", "score": 0.9}])
        storage = unit_storage(collection, search_strategy=strategy(max_results=3, oversampling_factor=5))

        found = await storage.search("dark")

        assert collection.pipelines == [
            [
                {
                    "$vectorSearch": {
                        "index": DEFAULT_STORAGE_VECTOR_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": [1.01, 0.01, 0.01, 0.01],
                        "numCandidates": 15,
                        "limit": 3,
                    }
                },
                {"$project": {"data": 1, "score": {"$meta": "vectorSearchScore"}}},
            ]
        ]
        assert found == [StorageSearchResult(key="notes/a", score=0.9, data=b"dark")]

    async def test_a_prefixed_store_filters_on_its_prefix_and_strips_it(self) -> None:
        collection = FakeStorageCollection(aggregate_result=[{"_id": "tenant-a/notes/a", "data": b"x", "score": 1}])
        storage = unit_storage(collection, prefix="tenant-a", search_strategy=strategy())

        found = await storage.search("dark")

        assert collection.pipelines[0][0]["$vectorSearch"]["filter"] == {"prefixes": {"$eq": "tenant-a/"}}
        assert [(result.key, type(result.score)) for result in found] == [("notes/a", float)]

    @pytest.mark.parametrize("query", ["", "   "])
    async def test_a_blank_query_embeds_nothing_and_sends_nothing(self, query: str) -> None:
        embedder = BagOfWordsEmbedder()
        collection = FakeStorageCollection()

        assert await unit_storage(collection, search_strategy=strategy(embedder)).search(query) == []
        assert embedder.calls == []
        assert collection.names_called == []


class TestEnsureIndexes:
    """AT-240 — ``ensure_indexes`` provisions the strategy's vector index alongside the ordinary ones."""

    async def test_the_vector_index_is_passed_to_ensure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[Sequence[_indexes.SearchIndexSpec]] = []

        async def ensure(collection: Any, **kwargs: Any) -> None:
            seen.append(kwargs["search"])

        monkeypatch.setattr(_indexes, "ensure", ensure)
        configured = strategy(index_name="v")

        await unit_storage(search_strategy=configured).ensure_indexes()
        await unit_storage().ensure_indexes()

        assert seen == [(configured.index_spec(),), ()]

    async def test_any_strategy_with_an_index_spec_gets_its_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[Sequence[_indexes.SearchIndexSpec]] = []

        async def ensure(collection: Any, **kwargs: Any) -> None:
            seen.append(kwargs["search"])

        spec = _indexes.SearchIndexSpec(name="text", definition={"mappings": {"dynamic": True}}, type="search")

        class TextStrategy:
            def index_spec(self) -> _indexes.SearchIndexSpec:
                return spec

            async def search(self, storage: Any, query: str, **kwargs: Any) -> list[Any]:
                return []

        monkeypatch.setattr(_indexes, "ensure", ensure)

        await unit_storage(search_strategy=TextStrategy()).ensure_indexes()  # type: ignore[arg-type]

        assert seen == [(spec,)]


@pytest.fixture
async def vector_storage(
    atlas_uri: str, test_database_name: str, unique_collection_name: str
) -> AsyncIterator[MongoDBStorage]:
    """A store with a vector strategy in this test's own collection, its index built and queryable."""
    config: MongoDBConfig = {"connection_string": atlas_uri, "database": test_database_name}
    storage = MongoDBStorage(config, collection=unique_collection_name, search_strategy=strategy())
    try:
        await storage.write("seed", b"tea")
        await storage.ensure_indexes()
        await storage.delete("seed")
        yield storage
    finally:
        await storage.close()


async def eventually(
    search: Callable[[], Any], until: Callable[[list[StorageSearchResult]], bool], timeout: float = 60
) -> list[StorageSearchResult]:
    """Poll ``search`` until ``until`` holds. mongot indexes a write asynchronously, so one read can miss it."""
    deadline = time.monotonic() + timeout
    while True:
        found: list[StorageSearchResult] = await search()
        if until(found) or time.monotonic() > deadline:
            return found
        await asyncio.sleep(0.5)


@pytest.mark.integration
class TestVectorSearchAgainstAtlas:
    """AT-241 — nearest first, scoped to the prefix, and gone once deleted."""

    async def test_the_nearest_value_ranks_first_with_its_payload(self, vector_storage: MongoDBStorage) -> None:
        await vector_storage.write("notes/ui.md", b"dark mode")
        await vector_storage.write("notes/drinks.md", b"coffee tea")

        found = await eventually(lambda: vector_storage.search("dark"), lambda found: len(found) == 2)

        assert [result.key for result in found] == ["notes/ui.md", "notes/drinks.md"]
        assert found[0].data == b"dark mode"
        assert found[0].score > found[1].score > 0

    async def test_a_namespace_sees_only_its_own_keys(self, vector_storage: MongoDBStorage) -> None:
        tenant_a = vector_storage.namespace("tenant-a")
        await tenant_a.write("notes/ui.md", b"dark mode")
        await vector_storage.namespace("tenant-b").write("notes/ui.md", b"dark mode")
        await eventually(lambda: vector_storage.search("dark"), lambda found: len(found) == 2)

        found = await tenant_a.search("dark")

        assert [result.key for result in found] == ["notes/ui.md"]

    async def test_a_deleted_or_emptied_value_stops_matching(self, vector_storage: MongoDBStorage) -> None:
        await vector_storage.write("gone", b"dark")
        await vector_storage.write("emptied", b"dark")
        await eventually(lambda: vector_storage.search("dark"), lambda found: len(found) == 2)

        await vector_storage.delete("gone")
        await vector_storage.write("emptied", b"")

        found = await eventually(lambda: vector_storage.search("dark"), lambda found: found == [])
        assert found == []
