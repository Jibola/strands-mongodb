"""The phase-2 ``SearchStrategy`` surface, declared and not implemented (``REQ-E-075``).

``MongoDBVectorSearchStrategy`` makes Atlas Vector Search available to callers whose primary storage is
something else, because a strategy is *passed* a ``Storage`` rather than owning one. That is phase 2
work. It is declared here rather than omitted because the published API surface carries the name, and a
published name that does not exist is indistinguishable from a broken install.

Nothing is imported at runtime. ``MongoDBConfig``, ``Storage`` and ``StorageSearchResult`` are all
``TYPE_CHECKING``-only, so this module executes on any install, including a metadata-only one, and the
first thing a caller learns is which phase the class belongs to rather than which dependency is absent.
The two modules that do need ``pymongo`` guard their imports with ``missing_extra`` instead.

The absence here is deliberately unlike ``add_messages`` on ``MongoDBMemoryStore`` (``REQ-E-044``),
which has to be genuinely missing because the SDK feature-detects it and would route calls into a stub
that raises. Nothing in ``strands`` detects a ``SearchStrategy``: every shipped backend calls
``KeywordSearchStrategy()`` inline for its own default, and since 1.56.0 three of them also accept one
through a ``search_strategy`` parameter. Neither route reaches this class, because
``__init__`` raises before there is an instance to pass. So a class that raises misleads nobody except
the caller who constructed it, and that caller is told why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.storage import Storage

    # Not re-exported by `strands.storage`, whose `__all__` names four things and not this one, so the
    # module path is the only import that resolves. `storage.py` reaches it the same way, for the same
    # reason. The SDK's own `SearchStrategy` protocol annotates with it, so a return type in the public
    # contract is spelled with a name the public package does not export.
    from strands.storage.storage import StorageSearchResult

    from strands_mongodb.types import MongoDBConfig

__all__ = ["MongoDBVectorSearchStrategy"]

_PHASE_2 = (
    "MongoDBVectorSearchStrategy is a phase 2 surface and is not implemented in this release. "
    "It is planned for v0.2.0. For Atlas Vector Search today, use MongoDBMemoryStore, whose "
    "search() runs the vector tier."
)


class MongoDBVectorSearchStrategy:
    """Atlas Vector Search as a strategy over any ``Storage``. Phase 2; constructing it raises.

    Satisfies the SDK's ``SearchStrategy`` protocol structurally, which is what the ``search``
    signature below is for. Nothing else here is load-bearing until the tier is written.

    Example:
        ```python
        # Phase 2. In v0.1.0 this raises NotImplementedError.
        strategy = MongoDBVectorSearchStrategy({"database": "agents"})
        results = await strategy.search(storage, "dark mode toggle")
        ```

        The call form is direct, not registered: the SDK has no seam that accepts a strategy, so a
        caller holds one and awaits it, exactly as ``KeywordSearchStrategy``'s own example does.
    """

    def __init__(self, config: MongoDBConfig, /, *, collection: str = "storage") -> None:
        """Raise, naming the phase.

        The parameters are the documented ones and are declared so the documented call shape reaches this
        message rather than a signature error. They are not read.

        Args:
            config: Connection and deployment settings, shared with the two implemented surfaces.
            collection: Collection the strategy would search.

        Raises:
            NotImplementedError: Always, in v0.1.0.
        """
        raise NotImplementedError(_PHASE_2)

    async def index(self, storage: Storage, key: str, data: bytes, **kwargs: Any) -> None:
        """Index a single entry for future searches.

        Unreachable for the same reason ``search`` is, and declared for a different one. ``index`` was
        added to ``SearchStrategy`` by ``strands-agents`` 1.56.0, after this class was written, for
        strategies that maintain an index incrementally rather than searching on the fly. A placeholder
        that satisfies the protocol it was published against but not the one a user installs is a
        published name that lies about what it is, which is the failure ``REQ-E-075`` exists to avoid
        (``REQ-E-130``).

        Args:
            storage: The storage backend the entry belongs to.
            key: The storage key being written.
            data: The raw bytes being stored.
            **kwargs: Strategy-specific options for forward compatibility.

        Raises:
            NotImplementedError: Always, in v0.1.0.
        """
        raise NotImplementedError(_PHASE_2)

    async def search(self, storage: Storage, query: str, **kwargs: Any) -> list[StorageSearchResult]:
        """Search content in storage matching query.

        Unreachable: no instance of this class can exist. The signature is copied from the SDK's
        ``SearchStrategy`` protocol so a type checker pins the shape phase 2 has to fill, and so an SDK
        release that changes it fails here rather than during that work.

        Args:
            storage: The storage to search over.
            query: A natural-language string query.
            **kwargs: Strategy-specific options for forward compatibility.

        Returns:
            Matched keys with relevance scores, ranked best-first.

        Raises:
            NotImplementedError: Always, in v0.1.0.
        """
        raise NotImplementedError(_PHASE_2)
