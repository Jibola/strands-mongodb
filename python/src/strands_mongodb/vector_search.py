"""``MongoDBVectorSearchStrategy``: Atlas Vector Search behind ``MongoDBStorage.search``.

Requires the ``storage`` extra and a deployment with Atlas Vector Search (Atlas, or an Atlas CLI local
deployment)::

    pip install 'strands-mongodb[storage]'
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any, Final

from ._extras import missing_extra

try:
    from strands.storage.search import StorageSearchResult
except ImportError as error:  # pragma: no cover - exercised in the extras tier
    missing_extra("storage", error, module=__name__)

from . import _indexes
from .types import DEFAULT_MAX_SEARCH_RESULTS, DEFAULT_OVERSAMPLING_FACTOR

if TYPE_CHECKING:
    from .storage import MongoDBStorage
    from .types import TextEmbedder, VectorSimilarity

__all__ = ["DEFAULT_STORAGE_VECTOR_INDEX_NAME", "MongoDBVectorSearchStrategy"]

DEFAULT_STORAGE_VECTOR_INDEX_NAME: Final[str] = "storage_vector_index"
"""Vector index name when none is given. Distinct from the memory store's ``vector_index``."""

PREFIXES_FIELD: Final[str] = "prefixes"
"""Every ``/``-terminated ancestor of a document's ``_id``, the path ``search`` pre-filters on.

``$vectorSearch`` cannot filter on an ``_id`` range, so a prefix becomes an equality match on this array.
"""


# The server bound on `numCandidates`. `_pipeline.MAX_NUM_CANDIDATES` holds the same number, but that
# module needs the `memory` extra.
_MAX_NUM_CANDIDATES: Final[int] = 10000


class MongoDBVectorSearchStrategy:
    """Vector search for :class:`~strands_mongodb.MongoDBStorage`, backed by Atlas Vector Search.

    The vector lives on the key's own document, so ``write`` and ``delete`` keep the index in step. Call
    ``MongoDBStorage.ensure_indexes()`` once to create the index. Empty and non-UTF-8 values aren't indexed.

    Example:
        ```python
        storage = MongoDBStorage(
            config,
            search_strategy=MongoDBVectorSearchStrategy(
                embedder=VoyageEmbedder(), dimensions=1024, similarity="cosine"
            ),
        )
        await storage.ensure_indexes()
        ```
    """

    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        dimensions: int,
        similarity: VectorSimilarity,
        index_name: str = DEFAULT_STORAGE_VECTOR_INDEX_NAME,
        max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
        oversampling_factor: int = DEFAULT_OVERSAMPLING_FACTOR,
    ) -> None:
        """Configure the strategy. Nothing connects until the first write or search.

        Args:
            embedder: Any SDK ``Embedder``. Embeds stored values and queries.
            dimensions: Length of the vectors ``embedder`` returns.
            similarity: ``"cosine"``, ``"dotProduct"`` or ``"euclidean"``.
            index_name: Name of the Atlas Vector Search index on the storage's collection.
            max_results: Most results one search returns.
            oversampling_factor: Multiplier turning ``max_results`` into ``numCandidates``.

        Raises:
            ValueError: If an argument is out of range.
        """
        if not 1 <= dimensions <= _indexes.MAX_DIMENSIONS:
            raise ValueError(
                f"strands_mongodb: 'dimensions' is {dimensions} and must be within 1..{_indexes.MAX_DIMENSIONS}."
            )
        if similarity not in _indexes.SIMILARITIES:
            raise ValueError(
                f"strands_mongodb: 'similarity' is {similarity!r} and must be one of "
                f"{', '.join(_indexes.SIMILARITIES)}."
            )
        if max_results < 1 or oversampling_factor < 1:
            raise ValueError(
                f"strands_mongodb: 'max_results' ({max_results}) and 'oversampling_factor' "
                f"({oversampling_factor}) must both be at least 1."
            )
        if max_results * oversampling_factor > _MAX_NUM_CANDIDATES:
            raise ValueError(
                f"strands_mongodb: numCandidates would be {max_results * oversampling_factor}, over the "
                f"server's bound of {_MAX_NUM_CANDIDATES}. Lower 'max_results' or 'oversampling_factor'."
            )
        self._embedder = embedder
        self._dimensions = dimensions
        self._similarity = similarity
        self._index_name = index_name
        self._max_results = max_results
        self._num_candidates = max_results * oversampling_factor

    def index_spec(self) -> _indexes.SearchIndexSpec:
        """Return the vector index this strategy searches, for ``MongoDBStorage.ensure_indexes``.

        Returns:
            A ``vectorSearch`` index over ``embedding`` with ``prefixes`` declared filterable.
        """
        return _indexes.SearchIndexSpec(
            name=self._index_name,
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": _indexes.VECTOR_FIELD,
                        "numDimensions": self._dimensions,
                        "similarity": self._similarity,
                    },
                    {"type": "filter", "path": PREFIXES_FIELD},
                ]
            },
            type="vectorSearch",
        )

    async def index(self, storage: MongoDBStorage, key: str, data: bytes, **kwargs: Any) -> None:
        """Embed ``data`` and store the vector on the document ``storage.write`` just replaced.

        The update matches only while the document still holds ``data``, so a slower write's vector never
        lands on a newer value.

        Args:
            storage: The storage the value was written to.
            key: The normalized key, relative to ``storage``.
            data: The bytes written.
            **kwargs: Unused; accepted for forward compatibility.

        Raises:
            ValueError: If the embedder returns a vector whose length is not ``dimensions``.
        """
        text = _decode_text(data)
        if not text:
            return
        vector = await self._embed(text)
        document_id = f"{storage.prefix}{key}"
        await storage._collection().update_one(
            {"_id": document_id, "data": data},
            {"$set": {_indexes.VECTOR_FIELD: vector, PREFIXES_FIELD: _ancestors(document_id)}},
        )

    async def search(self, storage: MongoDBStorage, query: str, **kwargs: Any) -> builtins.list[StorageSearchResult]:
        """Return the stored values nearest the query, under ``storage``'s prefix only.

        Args:
            storage: The storage to search.
            query: A natural-language string. A blank one returns ``[]`` without embedding.
            **kwargs: Unused; accepted for forward compatibility.

        Returns:
            Up to ``max_results`` results, best first, with the stored bytes as ``data``.

        Raises:
            ValueError: If the embedder returns a vector whose length is not ``dimensions``.
        """
        if not query.strip():
            return []
        stage: dict[str, Any] = {
            "index": self._index_name,
            "path": _indexes.VECTOR_FIELD,
            "queryVector": await self._embed(query),
            "numCandidates": self._num_candidates,
            "limit": self._max_results,
        }
        prefix = storage.prefix
        if prefix:
            stage["filter"] = {PREFIXES_FIELD: {"$eq": prefix}}
        pipeline = [
            {"$vectorSearch": stage},
            {"$project": {"data": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]
        cursor = await storage._collection().aggregate(pipeline)
        return [
            StorageSearchResult(
                key=str(document["_id"])[len(prefix) :],
                score=float(document.get("score") or 0.0),
                data=bytes(document["data"]),
            )
            async for document in cursor
        ]

    async def _embed(self, text: str) -> builtins.list[float]:
        """Embed ``text`` and check the length against the index definition.

        The server would reject a wrong length on search, but silently skip indexing it on write.
        """
        vector = [float(value) for value in await self._embedder.embed(text)]
        if len(vector) != self._dimensions:
            raise ValueError(
                f"strands_mongodb: the embedder returned a {len(vector)}-dimension vector and the index "
                f"{self._index_name!r} declares {self._dimensions}."
            )
        return vector


def _ancestors(document_id: str) -> builtins.list[str]:
    """Every ``/``-terminated prefix of ``document_id``, shortest first: ``a/b/c`` gives ``a/``, ``a/b/``."""
    parts = document_id.split("/")[:-1]
    return ["/".join(parts[: depth + 1]) + "/" for depth in range(len(parts))]


def _decode_text(data: bytes) -> str | None:
    """Decode ``data`` as UTF-8, or return ``None`` for empty or non-UTF-8 input."""
    if not data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None
