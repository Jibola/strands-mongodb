"""Acceptance tests for the phase-2 strategy placeholder.

Covers ``AT-075`` (``REQ-E-075``). The class under test does nothing, so what these tests assert is
everything around the doing: that the name exists, that touching it says which phase it belongs to,
that the signature phase 2 has to fill is the SDK's own, and that the one module in this package
allowed to raise ``NotImplementedError`` is this one.

The last of those is the reason this file is not three lines. ``AT-044d`` bans that raise across the
whole package, because a stub the SDK feature-detects turns "the SDK picks another path" into "the SDK
crashes" (``REQ-E-044``). ``REQ-E-075`` needs exactly that raise in one place. The exemption is
therefore asserted from both ends: ``AT-044d`` excludes this module by name and fails if it stops
raising, and ``test_every_not_implemented_raise_lives_in_the_placeholder_class`` below fails if the
raise spreads anywhere else.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from strands.storage.search.types import SearchStrategy

import strands_mongodb
from strands_mongodb.search import MongoDBVectorSearchStrategy

if TYPE_CHECKING:
    from collections.abc import Callable

    from strands_mongodb.types import MongoDBConfig

SRC = Path(strands_mongodb.__file__).parent

CONFIG: MongoDBConfig = {"database": "phase2", "connection_string": "mongodb://127.0.0.1:27017"}


class TestTouchingThePlaceholderNamesThePhase:
    """AT-075 — the class is present, and constructing it is where the story ends."""

    def test_constructing_it_raises_not_implemented_error(self) -> None:
        """The raise is in ``__init__``, not in ``search``, and that placement is the requirement.

        A stub that constructs successfully survives dependency injection and configuration wiring,
        so the first sign of trouble arrives in whatever code path finally searched, several awaits
        away from the line that named the class. Failing at construction puts it on the line the
        caller wrote.
        """
        with pytest.raises(NotImplementedError):
            MongoDBVectorSearchStrategy(CONFIG, collection="storage")

    def test_the_message_names_the_class_the_phase_and_what_works_today(self) -> None:
        """Three things, because a placeholder message that carries fewer sends the reader hunting.

        The class name, since the traceback of a lazily resolved attribute does not always show it.
        The phase, so the reader learns this is scheduled rather than broken. And the surface that
        does run Atlas Vector Search in this release, which is ``MongoDBMemoryStore`` and not
        ``MongoDBStorage``: storage search is lexical over keys (``REQ-E-032``), so naming it here
        would be advice that does not work.
        """
        with pytest.raises(NotImplementedError) as caught:
            MongoDBVectorSearchStrategy(CONFIG)

        message = str(caught.value)
        assert "MongoDBVectorSearchStrategy" in message
        assert "phase 2" in message
        assert "MongoDBMemoryStore" in message

    def test_both_constructor_parameters_exist(self) -> None:
        """``NotImplementedError`` rather than ``TypeError`` is the whole assertion.

        A placeholder that raised before declaring the documented parameters would accept no call shape,
        so every caller who wrote the documented one would get a signature error and learn nothing
        about the phase. Passing ``collection`` by keyword and the config positionally is that shape.
        """
        with pytest.raises(NotImplementedError):
            MongoDBVectorSearchStrategy(CONFIG, collection="memory")


class TestTheSignaturePhaseTwoHasToFill:
    """AT-075 — the unreachable method's only job is to be type checked."""

    def test_search_matches_the_sdk_protocol_signature(self) -> None:
        """Compared against ``SearchStrategy.search`` itself, so SDK drift lands here.

        Nothing can hold an instance of this class, so this method cannot be called and cannot be
        tested by calling it. What it can do is record the shape phase 2 has to fill, and the way to
        keep that record honest is to read the Protocol rather than to copy it into an assertion.
        A release that renames ``storage`` or widens ``query`` fails this test on the day it is
        installed, which is cheaper than discovering it while implementing the tier.
        """
        ours = inspect.signature(MongoDBVectorSearchStrategy.search)
        theirs = inspect.signature(SearchStrategy.search)

        assert [(p.name, p.kind) for p in ours.parameters.values()] == [
            (p.name, p.kind) for p in theirs.parameters.values()
        ]
        assert [p.annotation for p in ours.parameters.values()] == [p.annotation for p in theirs.parameters.values()]
        assert ours.return_annotation == theirs.return_annotation

    def test_search_is_a_coroutine_function(self) -> None:
        """Async is part of the contract, and a signature comparison does not see it.

        ``inspect.signature`` reports the same parameters for ``def`` and ``async def``. Every method
        on both SDK contracts is a coroutine, so a placeholder declared ``def`` would pin
        the wrong shape while passing the test above.
        """
        assert inspect.iscoroutinefunction(MongoDBVectorSearchStrategy.search)

    def test_an_instance_type_checks_as_a_search_strategy(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """Structural conformance, asserted the only way this class permits.

        ``SearchStrategy`` is a Protocol and is not ``runtime_checkable``, so there is no
        ``isinstance`` to call, and there is no instance to call it on either. mypy never runs the
        constructor, so the assignment below type checks a class that would raise if executed. That
        is the same trade ``AT-052`` takes for ``MemoryStore``.
        """
        result = run_mypy(
            """
            from strands.storage.search.types import SearchStrategy
            from strands_mongodb import MongoDBVectorSearchStrategy
            from strands_mongodb.types import MongoDBConfig

            config: MongoDBConfig = {"database": "phase2"}
            strategy: SearchStrategy = MongoDBVectorSearchStrategy(config)
            """
        )

        assert result.returncode == 0, result.stdout


class TestTheOneAllowedRaise:
    """AT-075 / AT-044d — the exemption's boundary, asserted rather than assumed."""

    def test_every_not_implemented_raise_lives_in_the_placeholder_class(self) -> None:
        """Read from the AST, so a docstring discussing the raise is not a hit.

        ``AT-044d`` excludes ``search.py`` from its package-wide scan. An exclusion by file name is
        only as narrow as something makes it: without this test, a ``raise NotImplementedError`` added
        to a helper function in this module, or to a second class in it, would pass both tests. What
        the pair actually pins is that this raise appears in this package exactly where
        ``REQ-E-075`` puts it.
        """
        found = {
            (path.name, owner.name)
            for path in sorted(SRC.rglob("*.py"))
            for owner in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(owner, ast.ClassDef) and _raises_not_implemented(owner)
        }
        loose = {
            path.name
            for path in sorted(SRC.rglob("*.py"))
            if _raises_not_implemented(_without_classes(ast.parse(path.read_text(encoding="utf-8"))))
        }

        assert found == {("search.py", "MongoDBVectorSearchStrategy")}
        assert loose == set()


class TestThePlaceholderNeedsNothingInstalled:
    """AT-075 — a module that imports nothing cannot report a missing dependency."""

    def test_importing_it_pulls_no_third_party_module_and_it_still_raises(self) -> None:
        """One subprocess, two claims, and the second is why the first matters.

        ``storage.py`` and ``memory_store.py`` guard their imports with ``missing_extra`` (``AT-006f``),
        which is right for modules that need ``pymongo``. This one needs nothing: ``MongoDBConfig``,
        ``Storage`` and ``StorageSearchResult`` are all ``TYPE_CHECKING``-only. So a caller who
        installed the package with no extra reaches the phase-2 message rather than an ``ImportError``
        naming a dependency the module never uses.

        This venv has every extra installed, so what is measured here is that the module imports
        nothing, not that a bare install works. The two together are the claim; a real
        ``[storage]``-only environment is demonstrated by hand instead.
        """
        source = """
            import sys

            import strands_mongodb.search as search

            watched = ("pymongo", "pymongo_search_utils", "strands")
            print(",".join(name for name in watched if name in sys.modules))
            try:
                search.MongoDBVectorSearchStrategy({"database": "phase2"})
            except NotImplementedError as error:
                print(type(error).__name__)
            """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.splitlines() == ["", "NotImplementedError"], result.stdout


def _raises_not_implemented(node: ast.AST) -> bool:
    """Report whether a node's body raises ``NotImplementedError`` anywhere inside it.

    Args:
        node: Any AST node.

    Returns:
        ``True`` if a ``raise`` statement inside it names ``NotImplementedError``.
    """
    return any(
        isinstance(inner, ast.Raise) and _raised_name(inner) == "NotImplementedError" for inner in ast.walk(node)
    )


def _raised_name(node: ast.Raise) -> str | None:
    """Return the bare exception name a ``raise`` statement names.

    Args:
        node: A ``raise`` statement.

    Returns:
        The exception's name, or ``None`` for a bare re-raise.
    """
    exception: Any = node.exc
    if isinstance(exception, ast.Call):
        exception = exception.func
    if isinstance(exception, ast.Attribute):
        return exception.attr
    return exception.id if isinstance(exception, ast.Name) else None


def _without_classes(module: ast.Module) -> ast.Module:
    """Return the module with its class definitions removed.

    Used to ask where a raise is **not**: everything left is module level, a plain function, or a
    nested block, none of which ``REQ-E-075`` permits.

    Args:
        module: A parsed module.

    Returns:
        A module node holding only the statements that are not class definitions.
    """
    return ast.Module(body=[node for node in module.body if not isinstance(node, ast.ClassDef)], type_ignores=[])
