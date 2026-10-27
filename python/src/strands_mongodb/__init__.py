"""MongoDB storage and memory backends for the Strands Agents SDK.

Two surfaces, each behind its own install extra:

- :mod:`strands_mongodb.storage`: ``MongoDBStorage``, implementing the SDK's unified
  ``strands.storage.Storage`` protocol. Install with ``pip install 'strands-mongodb[storage]'``.
- :mod:`strands_mongodb.memory_store`: ``MongoDBMemoryStore``, implementing the SDK's
  ``strands.memory.MemoryStore`` protocol. Install with ``pip install 'strands-mongodb[memory]'``.

Both classes are importable from here and are loaded on first access, not at import time. That way a
``[storage]``-only install can use ``MongoDBStorage`` without importing the memory module, which would
fail on ``pymongo_search_utils``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    # Type checkers cannot see through `__getattr__`, so without these the lazy names would type as `object`.
    from strands_mongodb.memory_store import MongoDBMemoryStore
    from strands_mongodb.storage import MongoDBStorage

try:
    __version__ = version("strands-mongodb")
except PackageNotFoundError:
    # Imported from a source tree that was never installed.
    __version__ = "0.0.0+unknown"

__all__ = ["MongoDBMemoryStore", "MongoDBStorage", "__version__"]

_LAZY: Final[dict[str, str]] = {
    "MongoDBStorage": "strands_mongodb.storage",
    "MongoDBMemoryStore": "strands_mongodb.memory_store",
}


def __getattr__(name: str) -> object:
    """Resolve a backend class on first access, importing only its own module.

    A module-level ``__getattr__`` (PEP 562) makes a type checker accept any attribute name on this
    module. Returning ``object`` rather than ``Any`` means a misspelled name is still rejected at its
    first real use.

    Args:
        name: The attribute being looked up.

    Returns:
        The class named by ``name``.

    Raises:
        AttributeError: If ``name`` is not a lazily exported class. ``hasattr``, ``inspect``, ``copy``
            and ``pickle`` rely on this exception type when probing for attributes.
        ImportError: If the class's install extra is missing. The message names the extra to install.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """List the lazily exported names alongside the eager ones.

    Without this, ``dir(strands_mongodb)`` and tab completion omit both classes until they are imported.

    Returns:
        The module's attribute names, sorted.
    """
    return sorted({*globals(), *_LAZY})
