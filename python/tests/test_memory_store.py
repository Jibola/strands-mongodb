"""``MongoDBMemoryStore``'s ``MemoryStore`` surface: protocol shape, scope, dedup, search tiers.

The memory store's acceptance set (``REQ-E-040``-``REQ-E-048``, ``REQ-E-052``-``REQ-E-059``), split across three
tiers by what each claim has to be observed against:

* **unit** for what is decided before any round trip: the protocol shape, the limit chain, the
  reserved-key merge, the ``writable`` gate, the score-bound pair, and a source-level tripwire on the
  regex tier.
* **contract** for writes, ordinary indexes, the dedup constraint and the no-search-support path. A
  plain ``mongod`` runs all of those, and answers no query at all (see ``conftest.py``), so nothing in
  this tier searches.
* **integration** for every read, because v0.1.0 ships no search tier that runs without mongot
 and one of ``REQ-E-053a``'s three server error codes is what a query gets instead.

**Most of what this file claims needs a server, unlike ``test_storage.py`` and ``test_indexes.py``.**
Both of those hand-write a fake collection because their claims are about the pipeline or the command
this package emits, which a server would answer identically whether or not the emission was right. Most
claims here are about what the server then *does*: one document instead of two, a scope predicate that
excludes, a tier that answers. A fake asserting those would be asserting its own implementation.

**The unit tier holds one fake collection anyway, and it is the ``test_storage.py``
case rather than an exception to it.** What it records is the ``$match`` this package composes and the
``MemoryEntry`` it builds back out of a document. Both are this package's own code, and a deployment
makes neither more real: mongot answers a predicate that lost the store's configured filter with
*results* rather than with an error, which is how that term went untested until a mutant removed it and
nothing failed. The fake is fixed documents in, recorded stages out. It resolves nothing, matches
nothing and ranks nothing, so every claim about matching or ranking is still on a tier below.

``AT-044a`` and ``AT-044b`` overlap ``test_construction.py``'s ``TestTheHooksThatMustStayAbsent``,
which already pins ``not hasattr(store, "add_messages")`` and ``_has_method(...) is False``. The pair
is completed here because its second half, ``_has_method(store, "add") is True``, is what makes the
first half mean anything: a store missing *both* methods passes the absence check and routes nowhere.
Same precedent as ``AT-055`` sitting beside ``AT-056d``.
"""

from __future__ import annotations

import ast
import asyncio
import datetime
import hashlib
import inspect
import logging
import re
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest
from bson import ObjectId
from conftest import skip_if_search_is_unreachable
from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure
from strands.memory import MemoryManager
from strands.memory.types import MemoryEntry, _has_method
from strands.types.exceptions import StorageError

import strands_mongodb
from strands_mongodb import _client, _indexes
from strands_mongodb.memory_store import UNSCOPED, MongoDBMemoryStore, _merge_metadata, _resolve_limit
from strands_mongodb.storage import MongoDBStorage
from strands_mongodb.types import DEFAULT_MAX_SEARCH_RESULTS, RESERVED_METADATA_KEYS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from strands_mongodb.types import MongoDBConfig, MongoDBMemoryStoreConfig

# Same reserved port and the same reason as in ``test_construction.py``: a unit-tier test that reaches
# the network fails on the timeout instead of finding a stray local mongod and passing for the wrong
# reason. The 50 ms selection timeout keeps that failure fast.
UNREACHABLE = "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=50"

# A URI whose inline password must not appear anywhere, for AT-014's memory half. Deliberately not a
# plausible-looking secret, so a grep over a failure report finds it.
LEAKY_URI = "mongodb://u:SUPERSECRET@127.0.0.1:1/?serverSelectionTimeoutMS=50"
SECRET = "SUPERSECRET"


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    """Isolate each test from the process-global connection registry.

    Same justification as the fixture of this name in ``test_client.py`` and
    ``test_construction.py``: the registry is module-level by design (``REQ-E-012``), so a test that
    constructs a store otherwise leaves a connection behind for the next one.
    """
    _client._REGISTRY.clear()
    yield
    _client._REGISTRY.clear()


def _connection_config(**overrides: Any) -> MongoDBConfig:
    """Build a minimal valid connection config pointing at nothing.

    Args:
        **overrides: Fields to set or replace.

    Returns:
        A config with ``database`` and ``connection_string`` set.
    """
    config: dict[str, Any] = {"database": "appdb", "connection_string": UNREACHABLE}
    config.update(overrides)
    return config  # type: ignore[return-value]


def _store_config(**overrides: Any) -> MongoDBMemoryStoreConfig:
    """Build a minimal valid memory-store config, nesting the connection half under ``config``.

    Args:
        **overrides: Top-level store fields to set or replace. Pass ``config=`` to replace the
            connection half.

    Returns:
        A config with ``name`` and a valid nested ``config``.
    """
    store: dict[str, Any] = {"name": "facts", "config": _connection_config()}
    store.update(overrides)
    return store  # type: ignore[return-value]


def _store(**overrides: Any) -> MongoDBMemoryStore:
    """Build a store on an unreachable deployment: constructs fully, answers nothing.

    Args:
        **overrides: Top-level store fields, as for :func:`_store_config`.

    Returns:
        The store.
    """
    return MongoDBMemoryStore(**_store_config(**overrides))


def _real_store(uri: str, database: str, collection: str, **overrides: Any) -> MongoDBMemoryStore:
    """Build a store on a real deployment, in a collection the caller owns.

    Takes a URI rather than a fixture's client, following ``test_storage.py``'s ``contract_storage``:
    the package builds and owns the client, so ``close()`` is exercised on the path a user takes
    rather than short-circuited by ``REQ-E-010``'s injected-client rule.

    Args:
        uri: Connection string for the deployment.
        database: Database to use, normally this session's throwaway one.
        collection: Collection to use, normally this test's own.
        **overrides: Top-level store fields. Pass ``connection=`` to add connection-level keys such
            as ``search_type`` or ``filter_fields``.

    Returns:
        The store. The caller closes it.
    """
    connection: dict[str, Any] = {"connection_string": uri, "database": database}
    connection.update(overrides.pop("connection", {}))
    store: dict[str, Any] = {"name": "facts", "config": connection, "collection": collection}
    store.update(overrides)
    return MongoDBMemoryStore(**store)


def _package_sources() -> dict[str, str]:
    """Read every module in the installed package as text.

    Returns:
        Module file name to source, for the source-level tripwires.
    """
    root = Path(strands_mongodb.__file__).parent
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.py"))}


def _string_literals(source: str) -> list[str]:
    """Return every string literal in a module except its docstrings.

    Docstrings are excluded because two of the tripwires below scan for an operator name, and the place
    an operator name most legitimately appears is a docstring explaining why the module does not use
    it. ``storage.py`` has exactly that: ``{"_id": {"$regex": "^prefix"}}`` in prose, arguing for a
    range query instead. A raw ``in source`` scan reads that as the operator being emitted, which would
    make ``AT-055`` fail on a comment and pass on a regression the day someone deleted the comment.

    Args:
        source: Module source.

    Returns:
        Every non-docstring string constant, in source order.
    """
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def _raised_names(source: str) -> set[str]:
    """Return the exception names a module raises.

    Read from ``ast.Raise`` nodes rather than from the text, so prose describing an exception this
    package deliberately does not raise cannot trip ``AT-044d``. ``test_construction.py``'s own
    docstring contains that exact sentence about ``NotImplementedError``.

    Args:
        source: Module source.

    Returns:
        The bare names raised, with any module path stripped.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


SUBSET_PROTOCOL_SNIPPET = '''
"""The nine ``MemoryStore`` members the SDK actually calls, declared here rather than imported."""

from __future__ import annotations

from typing import Any, Protocol

from strands.memory.extraction.types import ExtractionConfig
from strands.memory.types import MemoryEntry, Metadata, SearchOptions
from strands.types.tools import AgentTool

from strands_mongodb.memory_store import MongoDBMemoryStore


class MemoryStoreSubset(Protocol):
    name: str
    description: str | None
    max_search_results: int | None
    writable: bool
    extraction: ExtractionConfig | bool | None

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]: ...

    async def add(self, content: str, metadata: Metadata | None = None) -> Any: ...

    async def initialize(self) -> None: ...

    def get_tools(self) -> list[AgentTool]: ...


store: MemoryStoreSubset = MongoDBMemoryStore(
    name="facts", config={"database": "appdb", "connection_string": "mongodb://127.0.0.1:1"}
)
'''


@pytest.mark.compat
class TestTheProtocolMembersTheSDKCalls:
    """AT-052 / REQ-E-052 — asserted by assignment because ``isinstance`` is not available.

    ``MemoryStore`` is a plain ``Protocol``, not ``runtime_checkable``, so ``isinstance`` against it
    raises ``TypeError`` rather than returning ``False``; and it declares ten members while
    ``REQ-E-044`` requires one of them absent, so no SDK-authored check can pass. ``REQ-E-052``'s amendment
    records why the original ``isinstance`` wording was replaced. What is left is the strongest check
    available: a nine-member Protocol declared in this file, satisfied by assignment under mypy, plus
    the SDK's own runtime detector on the four methods the manager dispatches through.

    Marked ``compat`` and it still runs in an ordinary session. The marker selects it for the CI cron
    that resolves the latest published ``strands-agents`` instead of the pinned one. That leaves this as the
    only mitigation: a tenth member the manager starts calling shows up as a mypy failure here rather
    than in a user's traceback. The gate in ``conftest.py`` covers the three tiers that need a backend,
    and this needs none, so nothing skips it locally.
    """

    def test_the_store_satisfies_a_nine_member_subset_protocol_under_mypy(self, run_mypy: Any) -> None:
        """Assignment to the Protocol is the assertion; mypy reports the gap if a member is missing.

        A missing attribute, a wrong parameter name or a narrowed return type all surface as an error
        on the assignment line, which is the whole point of writing it as an assignment rather than as
        a set of ``hasattr`` checks.
        """
        result = run_mypy(SUBSET_PROTOCOL_SNIPPET)

        assert result.returncode == 0, result.stdout
        assert "error:" not in result.stdout, result.stdout

    def test_the_subset_omits_exactly_one_member_of_the_sdk_protocol(self) -> None:
        """The subset is nine of ten, and the tenth is ``add_messages`` (``REQ-E-044``).

        Derived from the SDK's own annotations rather than hard-coded, so a member added upstream makes
        this fail on the pinned SDK too, not only on the compat cron. Without it the subset could drift
        into asserting a protocol the SDK no longer has.
        """
        from strands.memory.types import MemoryStore

        attributes = {name for name in MemoryStore.__annotations__ if not name.startswith("_")}
        methods = {name for name, value in vars(MemoryStore).items() if not name.startswith("_") and callable(value)}

        assert attributes | methods == {
            "name",
            "description",
            "max_search_results",
            "writable",
            "extraction",
            "search",
            "add",
            "add_messages",
            "initialize",
            "get_tools",
        }
        assert "add_messages" not in SUBSET_PROTOCOL_SNIPPET

    @pytest.mark.parametrize("method", ["search", "add", "initialize", "get_tools"])
    def test_the_sdk_detector_finds_each_method_the_manager_dispatches_through(self, method: str) -> None:
        """``_has_method`` reads ``getattr(type(store), name)``, so this is a claim about the class."""
        assert _has_method(_store(), method) is True


class TestTheHookThatReturnsNothing:
    """AT-052a / REQ-E-052a — ``get_tools`` is present and empty, and presence changes nothing.

    None of the three vended stores defines ``get_tools``, so there is no precedent for the return
    value, while ``REQ-E-052`` requires the detector above to report it present. Both hold at once
    because ``MemoryManager._build_tools`` reads
    ``if _has_method(store, "get_tools"): tools.extend(store.get_tools())``
    (``memory_manager.py:252-254``): a present method returning ``[]`` extends the list by nothing.

    That is the asymmetry with ``add_messages``, whose presence decides *which extractor runs*. A
    method whose presence changes routing is a hazard; one whose presence changes nothing is where
    v0.2.0's tools go without a compatibility event.
    """

    def test_get_tools_returns_an_empty_list(self) -> None:
        """An empty ``list``, not ``None`` and not a tuple: the manager calls ``extend`` on it."""
        tools = _store().get_tools()

        assert tools == []
        assert isinstance(tools, list)

    def test_a_manager_holding_the_store_registers_only_its_own_two_tools(self) -> None:
        """The observable form of "identical to absence": the tool list the agent would see.

        ``add_tool_config=True`` and a writable store, because the manager disables the add tool by
        default and the interesting count is the one where both of its own tools are live. A store
        contributing anything would show up as a third entry.
        """
        store = _store(writable=True)

        # The store deliberately does not satisfy the ten-member `MemoryStore`, so the SDK's own
        # annotation rejects it while `_has_method` accepts it at runtime. The ignore is `REQ-E-044`
        # being visible at the call site rather than a wart to remove.
        manager = MemoryManager(stores=[store], add_tool_config=True)  # type: ignore[list-item]

        assert [tool.tool_name for tool in manager.tools] == ["search_memory", "add_memory"]


class TestAddMessagesStaysAbsentAndAddDoesNot:
    """AT-044a, AT-044b / REQ-E-044 — the pair rather than the half.

    ``test_construction.py`` already asserts the absence. What it cannot assert on its own is that the
    absence routes anywhere: ``_has_method(store, "add_messages") is False`` also holds for a store with
    no write path at all, and the manager's fallback is ``ModelExtractor`` feeding ``add``. So the two
    probes belong together, and the second one is new here.
    """

    def test_add_messages_is_absent_by_both_probes(self) -> None:
        """``hasattr`` and the SDK's detector agree, because this class inherits no Protocol.

        Both are asserted because they would disagree the day this class started inheriting
        ``MemoryStore``: the inherited stub satisfies ``hasattr`` while ``_has_method`` still reports it
        absent, and that divergence is what would silently reintroduce the raising-stub hazard.
        """
        store = _store()

        assert not hasattr(store, "add_messages")
        assert _has_method(store, "add_messages") is False

    def test_add_is_present_by_the_same_detector(self) -> None:
        """The half that makes the absence mean something rather than mean nothing."""
        assert _has_method(_store(), "add") is True

    def test_a_writable_store_satisfies_the_managers_write_sink_check(self) -> None:
        """The manager's own reading of the pair, which is where a wrong answer would surface.

        ``MemoryManager.__init__`` raises *"is writable but has no add or add_messages method"* for a
        writable store it cannot write to (``memory_manager.py:130-133``). Constructing the manager is
        the assertion; it raises ``ValueError`` if either probe above is wrong in the direction that
        matters.
        """
        manager = MemoryManager(stores=[_store(writable=True)])  # type: ignore[list-item]

        assert manager.tools[0].tool_name == "search_memory"


class TestNoMethodStandsInForAbsence:
    """AT-044d / REQ-E-044 — a stub that raises is present, and presence is what the SDK reads.

    ``_has_method`` and ``hasattr`` both report a ``raise NotImplementedError`` body as a member the
    caller may use, so the SDK routes to it and every call fails. That converts "the SDK picks another
    path" into "the SDK crashes", for ``add_messages`` on this class and for ``for_sandbox`` on
    ``MongoDBStorage`` (``AT-044c``). Asserted over both public classes and then over the whole package,
    because the hazard is the pattern rather than the two names.
    """

    @pytest.mark.parametrize("cls", [MongoDBMemoryStore, MongoDBStorage])
    def test_neither_public_class_raises_not_implemented_error(self, cls: type) -> None:
        """Read from the class body's ``raise`` statements, so a docstring mentioning it is fine."""
        assert "NotImplementedError" not in _raised_names(inspect.getsource(cls))

    def test_no_module_in_the_package_raises_not_implemented_error(self) -> None:
        """The tripwire, one level wider: an internal module could hold the same stub.

        ``_client.resolve`` and ``_indexes.ensure`` are both places where "not supported here" is a real
        condition, and both already answer it with ``ValueError`` or ``StorageError`` carrying the
        server's own message (``REQ-E-053a``). This fails the day one of them takes the shortcut.

        One module is exempt, and the exemption is written to expire. ``search.py`` is ``REQ-E-075``'s
        phase-2 placeholder: the class v0.2.0 publishes is declared and raises when constructed,
        which is the opposite decision from ``add_messages`` and safe for the recorded reason,
        that nothing in the SDK detects a ``SearchStrategy``. The second assertion below fails the day
        that module stops raising, which is the day the exemption should be deleted rather than
        inherited. How narrow the exemption is stays pinned in ``test_search.py``, which asserts the
        raise appears in that class and nowhere else in the package.
        """
        sources = _package_sources()
        offenders = {
            name: sorted(_raised_names(source))
            for name, source in sources.items()
            if name != "search.py" and "NotImplementedError" in _raised_names(source)
        }

        assert offenders == {}
        assert "NotImplementedError" in _raised_names(sources["search.py"])


class TestTheResultLimit:
    """AT-041 / REQ-E-041 — a four-link chain and a floor, both asserted where they are decided.

    The resolver is tested directly rather than through a result count, and the reason is a measurement
    rather than convenience: the observable outcome of a limit is *how many documents come back*, which
    needs a search-capable deployment, and on the only one CI reaches the count is also shaped by the
    corpus and by the score bounds. So the chain is asserted here as arithmetic and the count is
    asserted on the integration tier (``AT-040``, ``AT-059b``), which is the split the two tiers exist
    for.

    ``query["limit"]`` sits above ``options`` because a ``MongoDBSearchQuery`` can only arrive on a
    direct call: ``MemoryStore.search`` is declared ``query: str``, and both SDK call sites pass a
    ``str``. Below ``options`` the key would be unreachable rather than lower priority, since
    ``MemoryManager.search`` always passes ``options`` with ``max_search_results`` already resolved.
    ``REQ-E-041``'s amendment carries the reasoning.
    """

    def test_a_query_limit_wins_over_everything_below_it(self) -> None:
        """All four links supplied, and the topmost one is the answer."""
        assert _resolve_limit(query=3, options=7, instance=11) == 3

    def test_options_win_over_the_instance_default(self) -> None:
        """What the manager-driven path resolves to: no query limit, options carrying its own chain."""
        assert _resolve_limit(query=None, options=7, instance=11) == 7

    def test_the_instance_value_wins_over_the_package_default(self) -> None:
        """The store's own ``max_search_results``, which is where a per-store ceiling belongs."""
        assert _resolve_limit(query=None, options=None, instance=11) == 11

    def test_nothing_supplied_resolves_to_the_package_default(self) -> None:
        """Ten, from ``types.DEFAULT_MAX_SEARCH_RESULTS``, matching both shipped stores."""
        assert _resolve_limit(query=None, options=None, instance=None) == DEFAULT_MAX_SEARCH_RESULTS
        assert DEFAULT_MAX_SEARCH_RESULTS == 10

    def test_a_limit_below_one_is_rejected_at_construction(self) -> None:
        """``__init__``, because a store whose ceiling is zero can never answer anything.

        ``BedrockKnowledgeBaseStore`` raises in the same place for the same value, so the shape is
        precedented rather than invented.
        """
        with pytest.raises(ValueError, match="max_search_results"):
            _store(max_search_results=0)

    async def test_a_limit_below_one_in_options_is_rejected_before_any_round_trip(self) -> None:
        """The store points at a closed port, so a resolver running after the query fails differently.

        That is the assertion, not an accident of the fixture: ``ValueError`` here means the check ran
        before the network, and a connection error would mean it ran after.
        """
        with pytest.raises(ValueError, match="max_search_results"):
            await _store().search("anything", {"max_search_results": 0})

    async def test_a_limit_below_one_in_a_query_is_rejected_before_any_round_trip(self) -> None:
        """The fourth raise site, on the key a later amendment added to the chain."""
        with pytest.raises(ValueError, match="limit"):
            await _store().search({"text": "anything", "limit": 0})


class TestReservedMetadataKeysWin:
    """AT-057 / REQ-E-040, REQ-E-057 — the caller's key is dropped, and the drop is logged.

    Four reserved keys travel out with every result: ``_relevance_score``, ``_id``, ``_search_type`` and
    ``_scope``. A stored document whose ``metadata`` holds one of those names would otherwise decide the
    value of a field the caller is meant to trust, and ``_scope`` is the one where that matters
    most: a document claiming a scope it was not stored in reads as a leak.

    Merged by a helper rather than inline, so this tier can assert the merge without a deployment. The
    integration tier then asserts the four keys are populated at all (``AT-040``).
    """

    def test_a_colliding_caller_key_keeps_the_reserved_value(self) -> None:
        """The reserved value survives and the caller's is gone, not renamed and not nested."""
        merged = _merge_metadata(
            {"_scope": "somebody-elses-scope", "topic": "cats"},
            {"_scope": "tenant-a", "_id": "abc", "_search_type": "vector", "_relevance_score": 0.5},
            store_name="facts",
        )

        assert merged["_scope"] == "tenant-a"
        assert merged["topic"] == "cats"
        assert list(merged.values()).count("somebody-elses-scope") == 0

    def test_non_colliding_caller_keys_survive_untouched(self) -> None:
        """The merge is not a filter. Caller metadata is why the field exists."""
        merged = _merge_metadata({"topic": "cats", "source": "chat"}, {"_id": "abc"}, store_name="facts")

        assert merged == {"topic": "cats", "source": "chat", "_id": "abc"}

    def test_the_collision_is_logged_with_the_key_and_the_store(self, caplog: pytest.LogCaptureFixture) -> None:
        """A dropped value with no log record is indistinguishable from a value that never existed.

        ``warning`` rather than ``debug``: a collision is a caller mistake with a fix, and the fix
        cannot be found from an absent field. The store name is in the record because one process holds
        several stores over one collection.
        """
        with caplog.at_level(logging.WARNING, logger="strands_mongodb.memory_store"):
            _merge_metadata({"_id": "spoofed"}, {"_id": "real"}, store_name="facts")

        assert len(caplog.records) == 1
        assert "_id" in caplog.records[0].message
        assert "facts" in caplog.records[0].message

    def test_every_reserved_key_is_defended(self) -> None:
        """Parametrized over the constant rather than a hand-written list, so the two cannot drift."""
        for key in RESERVED_METADATA_KEYS:
            merged = _merge_metadata({key: "caller"}, {key: "reserved"}, store_name="facts")

            assert merged[key] == "reserved", key


class TestNoRegexTierIsReachable:
    """AT-055 / REQ-E-055 — a tripwire on a requirement that cannot be satisfied yet.

    The requirement says ``re.escape`` applies *while the regex tier is active*, and v0.1.0 runs no such
    tier: ``SearchType`` accepts ``auto``, ``vector`` and ``text``, and ``regex`` is scheduled for
    v0.2.0. So the clause is vacuously true, which is not the same as satisfied. A contract
    test of the escaping cannot exist without the tier, and a skipped placeholder asserts nothing while
    looking like coverage.

    What is testable is the vacuity. ``AT-056d`` pins that neither ``SearchType`` nor
    ``ResolvedSearchType`` admits ``regex``, so no config value and no result value selects the tier.
    These two add that no module emits the operator either, so there is no unselectable back door.
    ``REQ-E-050`` already stops a caller from smuggling one in through ``filter``, at any depth.

    Both fail the day the regex tier lands, which is the day ``re.escape`` has to be there.
    """

    def test_no_module_emits_a_regex_match_operator(self) -> None:
        """Every ``$regex``-family token in the package, from string literals only.

        ``$regexFindAll`` is allowed and is not this tier. It is an aggregation *expression* over ``_id``
        in ``MongoDBStorage``'s keyword search, tokenizing a stored key rather than matching a caller's
        query, and it takes no caller input at all. The next test pins where it lives so the allowance
        cannot quietly widen.
        """
        found = {
            token
            for source in _package_sources().values()
            for literal in _string_literals(source)
            for token in re.findall(r"\$regex[A-Za-z]*", literal)
        }

        assert found <= {"$regexFindAll"}, found

    def test_the_one_allowed_regex_expression_stays_where_it_is(self) -> None:
        """``$regexFindAll`` in ``storage.py`` and nowhere else, least of all on a memory query path."""
        modules = {
            name
            for name, source in _package_sources().items()
            if any("$regexFindAll" in literal for literal in _string_literals(source))
        }

        assert modules == {"storage.py"}

    def test_no_module_emits_a_text_search_operator(self) -> None:
        """``$text`` is the other half of the tier v0.1.0 defers, so it is the other half of the wire.

        Not a stylistic ban. A ``$text`` index would be a fourth index nobody provisioned and a fifth
        scoring scale ``_search_type`` has no value for.
        """
        found = {
            name
            for name, source in _package_sources().items()
            for literal in _string_literals(source)
            if "$text" in literal
        }

        assert found == set()


class TestWritable:
    """AT-058 / REQ-E-058 — defaults to ``False``, and ``add`` on a read-only store says which key.

    Two precedents disagree: ``FileMemoryStore`` defaults it to ``True``,
    ``BedrockKnowledgeBaseStore`` and the official extension template both to ``False``. Two of three,
    including the template this package is modelled on, decides it. The cost is asymmetric in the same
    direction: ``writable`` gates automatic extraction, so ``True`` means a store built for retrieval
    starts accepting model-distilled writes nobody asked for.
    """

    def test_the_default_is_read_only(self) -> None:
        """Absent from the config means ``False``, not ``None`` and not truthy-by-omission."""
        assert _store().writable is False

    def test_configuring_it_makes_the_store_writable(self) -> None:
        """The opt-in has to actually arrive, or the default is the only behavior."""
        assert _store(writable=True).writable is True

    async def test_add_on_a_read_only_store_names_the_key_that_enables_it(self) -> None:
        """``ValueError`` naming ``writable``, before any round trip.

        A message that says only "not writable" leaves the caller reading source to find the key. The
        error arrives before the network for the same reason as the limit checks: this is caller
        configuration, and a connection error would hide it.
        """
        with pytest.raises(ValueError, match="writable"):
            await _store().add("a fact nobody asked to store")


class TestScoreBounds:
    """AT-059a / REQ-E-059 — the mutually exclusive pair, rejected at construction.

    ``MongoDBMemoryStoreConfig``'s ``max_score`` docstring already promises *"Mutually exclusive with
    ``min_score``"*, which is a claim shipped in a public type with nothing behind it.
    ``BedrockKnowledgeBaseStore`` raises for the same pair in ``__init__``. ``__init__`` rather than
    ``search``, because the pair is store configuration and a constructor that accepts a contradiction
    only defers the error to a query.

    What the bounds then *do* to a result set is ``AT-059b``, on the integration tier, against real
    scores.
    """

    def test_both_bounds_together_are_rejected_naming_both_keys(self) -> None:
        """Both names in the message, because either one is the one the caller meant to keep."""
        with pytest.raises(ValueError) as error:
            _store(min_score=0.1, max_score=0.9)

        assert "min_score" in str(error.value)
        assert "max_score" in str(error.value)

    @pytest.mark.parametrize("bound", ["min_score", "max_score"])
    def test_either_bound_alone_constructs(self, bound: str) -> None:
        """The rejection is of the pair, not of the feature."""
        store = _store(**{bound: 0.5})

        assert getattr(store, bound) == 0.5


class TestEveryCallerInputIsCheckedBeforeTheNetwork:
    """AT-076 / REQ-E-049, REQ-E-050, REQ-E-074a — the query and filter guards, on a closed port.

    Same premise and the same reserved port as ``TestTheLimitChain``'s last two tests: a check that
    reached the network would time out instead of raising, so the ``ValueError`` is evidence of the
    ordering as well as of the check.

    Which is the whole reason these are here rather than left where the guards live.
    ``test_pipeline.py`` and ``test_filter.py`` already close both over their own functions. What was
    unasserted is that the memory store *calls* them, and calls them first: a mutation pass
    deleted the ``check_query_keys`` call from ``search``, the per-query ``_filter.validate``
    call and the configured-filter one from ``__init__``, and the suite stayed green
    for all three. A store with no guard on the wire path is not a hypothetical failure mode; it is
    ``REQ-E-049``'s cross-scope read with the guard still passing its own unit tests.

    The positive control for all four rejections is ``AT-077``'s first test, which puts a declared path in
    both filter positions and asserts the predicate that comes out.
    """

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ({"text": "dark mode", "vector": [1.0, 0.0, 0.0, 0.0]}, "both"),
            ({"limit": 5}, "neither"),
        ],
        ids=["both-keys", "neither-key"],
    )
    async def test_a_query_setting_both_or_neither_vector_key_is_rejected(self, query: Any, expected: str) -> None:
        """``REQ-E-074a`` at this surface, which is the only surface a caller reaches.

        ``limit`` is set on the neither-key case so the limit chain is not what raises, and the message
        substrings are the two words that distinguish the pair.
        """
        with pytest.raises(ValueError, match=expected):
            await _store().search(query)

    @pytest.mark.parametrize(
        ("bad_filter", "expected"),
        [({"metadata.tier": "gold"}, "filter_fields"), ({"$where": "sleep(1000)"}, r"\$where")],
        ids=["undeclared-path", "operator-key"],
    )
    async def test_a_per_query_filter_is_validated_before_the_query_runs(
        self, bad_filter: dict[str, Any], expected: str
    ) -> None:
        """``REQ-E-049`` and ``REQ-E-050``, on the filter that arrives with a single query."""
        with pytest.raises(ValueError, match=expected):
            await _store().search({"text": "dark mode", "filter": bad_filter})

    @pytest.mark.parametrize(
        ("bad_filter", "expected"),
        [({"metadata.tier": "gold"}, "filter_fields"), ({"$where": "sleep(1000)"}, r"\$where")],
        ids=["undeclared-path", "operator-key"],
    )
    def test_a_configured_filter_is_validated_at_construction(self, bad_filter: dict[str, Any], expected: str) -> None:
        """The same two guards, one query earlier.

        A store filter applies to every query this store will ever run, so a constructor that accepts one
        the index cannot serve defers the error to whichever search happens first and reports it as that
        search's problem.
        """
        with pytest.raises(ValueError, match=expected):
            _store(filter=bad_filter)


class _RecordingCollection:
    """A collection that records what it was handed and answers with documents the test wrote.

    The one fake collection in this file, and the module docstring says why it is here. It resolves
    nothing: ``documents`` come back in the order given, whatever the stages ask for. That makes it
    usable for the two claims it serves, which are the stages this package emits and the entries it
    builds out of a document, and useless for anything about matching, ranking or counting.

    Attributes:
        documents: What ``aggregate`` answers with.
        failure: A driver error to raise instead of answering, for ``AT-080``.
        pipelines: Every pipeline handed to ``aggregate``, in call order.
        upserts: Every ``(selector, update)`` pair handed to ``update_one``.
        deletes: Every filter handed to ``delete_one`` or ``delete_many``, in call order.
        deleted: The count the two delete methods report back, which a test sets to a number no real
            deletion of its documents could produce.
    """

    def __init__(self, documents: list[dict[str, Any]], *, failure: Exception | None = None) -> None:
        self.documents = documents
        self.failure = failure
        self.pipelines: list[list[dict[str, Any]]] = []
        self.upserts: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.deletes: list[dict[str, Any]] = []
        self.deleted = 0

    async def aggregate(self, pipeline: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the stages, then answer or fail.

        Args:
            pipeline: The stages the store composed.
            *args: Ignored, so a signature change downstream does not fail here for the wrong reason.
            **kwargs: Ignored, as above.

        Returns:
            An async iterator over :attr:`documents`, which is what the driver returns.

        Raises:
            Exception: :attr:`failure`, if one was set.
        """
        self.pipelines.append([dict(stage) for stage in pipeline])
        if self.failure is not None:
            raise self.failure
        return _cursor(self.documents)

    async def update_one(self, selector: Any, update: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the upsert, then fail.

        There is no success path here on purpose. What ``add`` does with an ``UpdateResult`` is a document
        count, which is ``AT-042`` and ``AT-043`` against a deployment; a hand-written result would be
        this fake asserting its own return value.

        Args:
            selector: The upsert filter.
            update: The update document.
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            Exception: :attr:`failure`, if one was set.
            NotImplementedError: Otherwise.
        """
        self.upserts.append((dict(selector), dict(update)))
        if self.failure is not None:
            raise self.failure
        raise NotImplementedError("no test needs a faked UpdateResult; the write path's claims are counts")

    async def delete_one(self, selector: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the filter, then report :attr:`deleted`.

        A fabricated count here, where ``update_one`` above refuses to fabricate one, and the difference
        is what the count is being read for. ``add`` turns an ``UpdateResult`` into a document count, so a
        faked result would be this fake asserting its own arithmetic. ``delete`` passes the driver's
        number straight through, so a fake returning a number no real deletion could produce is exactly
        what catches a method that returns ``1`` on its own authority.

        Args:
            selector: The delete filter.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            An object exposing ``deleted_count``, which is the one field of the driver's ``DeleteResult``
            this package reads.

        Raises:
            Exception: :attr:`failure`, if one was set.
        """
        self.deletes.append(dict(selector))
        if self.failure is not None:
            raise self.failure
        return _DeleteResult(self.deleted)

    async def delete_many(self, selector: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the filter, then report :attr:`deleted`, as :meth:`delete_one` does.

        Args:
            selector: The delete filter.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            An object exposing ``deleted_count``.

        Raises:
            Exception: :attr:`failure`, if one was set.
        """
        self.deletes.append(dict(selector))
        if self.failure is not None:
            raise self.failure
        return _DeleteResult(self.deleted)

    @property
    def stages(self) -> list[dict[str, Any]]:
        """The one recorded pipeline.

        Returns:
            The stages, failing the test if ``aggregate`` was not called exactly once.
        """
        assert len(self.pipelines) == 1, f"expected one aggregate call, got {len(self.pipelines)}"
        return self.pipelines[0]


class _DeleteResult:
    """The one field of ``pymongo.results.DeleteResult`` this package reads.

    Attributes:
        deleted_count: How many documents the server removed.
    """

    def __init__(self, deleted_count: int) -> None:
        self.deleted_count = deleted_count


async def _cursor(documents: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Yield each document, as the driver's cursor does.

    An async generator rather than a list, because ``search`` consumes the result with ``async for`` and a
    fake that returned a list would let a synchronous iteration pass here and fail on a deployment. Copied
    per document, so a test can assert on the dict it wrote afterwards.

    Args:
        documents: What to yield.

    Yields:
        Each document, shallow-copied.
    """
    for document in documents:
        yield dict(document)


def _stage(stages: list[dict[str, Any]], operator: str) -> Any:
    """Return the one stage using this operator.

    Args:
        stages: A recorded pipeline.
        operator: The stage operator to find, such as ``$match``.

    Returns:
        That stage's value, failing the test if there is not exactly one.
    """
    found = [stage[operator] for stage in stages if operator in stage]
    assert len(found) == 1, f"expected one {operator} stage, got {len(found)} in {[tuple(s) for s in stages]}"
    return found[0]


def _prepared(
    store: MongoDBMemoryStore,
    *,
    documents: list[dict[str, Any]] | None = None,
    tiers: tuple[Any, ...] = ("text",),
    failure: Exception | None = None,
) -> _RecordingCollection:
    """Put a store in the state ``initialize()`` would leave it in, without a deployment.

    Three assignments, and each one stands for a round trip: ``_initialized`` for the probe and the index
    calls, ``_tiers`` for what came up, ``_collection`` for the collection itself. Reaching into private
    state rather than patching ``_indexes``, because what these tests are about begins after setup
    finishes. ``AT-078`` is the one that patches setup instead and asserts what it did.

    Args:
        store: The store to prepare. Modified in place.
        documents: What ``aggregate`` answers with. Defaults to one scoreless document, since half of
            these tests do not care what comes back.
        tiers: The prepared tiers, in preference order. ``()`` is the state ``AT-078``'s query test needs
            and the one ``initialize()`` refuses to leave a store in.
        failure: A driver error for the collection to raise instead of answering.

    Returns:
        The recording collection the store now resolves to.
    """
    default = [{"_id": "abc", "content": "the user prefers dark mode"}]
    collection = _RecordingCollection(default if documents is None else documents, failure=failure)
    store._initialized = True
    store._tiers = tiers
    store._collection = lambda: collection  # type: ignore[assignment, return-value]
    return collection


class TestWhatTheStoreSendsAndWhatItMapsBack:
    """AT-077 / REQ-E-040, REQ-E-046, REQ-E-051, REQ-E-057 — predicate out, entry back.

    ``AT-046`` and ``AT-051`` close the scope predicate against a real deployment by observing what a
    second scope's documents do, which is the stronger claim and the reason they live on the integration
    tier. What they cannot see is a term that narrows nothing they seeded: two of the four terms here are
    the caller's own, and a query that silently dropped one still returns the documents the test wrote.
    A mutant that dropped the store's configured filter failed nothing.

    The mapping half is the same shape of gap. A document is whatever the collection holds, so the
    reserved keys are decided here and a document that could influence them could report a scope it was
    not stored in.
    """

    async def test_the_predicate_carries_the_scope_the_store_and_both_caller_filters(self) -> None:
        """Four terms from four places, in the order ``_filter.compose`` documents.

        Also the positive control for ``AT-076``'s four rejections: a declared path in both filter
        positions, accepted, and this is where each one went.
        """
        store = _store(
            config=_connection_config(filter_fields=["metadata.tier", "metadata.region"]),
            scope="tenant-a",
            filter={"metadata.tier": "gold"},
        )
        collection = _prepared(store)

        await store.search({"text": "dark mode", "filter": {"metadata.region": "emea"}})

        assert _stage(collection.stages, "$match") == {
            "$and": [
                {"scope": {"$eq": "tenant-a"}},
                {"store_name": {"$eq": "facts"}},
                {"metadata.tier": "gold"},
                {"metadata.region": "emea"},
            ]
        }

    async def test_a_stored_scope_key_cannot_make_a_document_report_another_scope(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The scope-leak read half: ``_scope`` is read off the store, whatever the document holds.

        ``AT-057`` closes the same rule over ``_merge_metadata`` directly, where the reserved dict is the
        test's own argument. This is the store deciding what goes into that dict, which is the decision a
        document could reach.
        """
        store = _store(scope="tenant-a")
        _prepared(
            store,
            documents=[
                {
                    "_id": "abc",
                    "content": "the user prefers dark mode",
                    "score": 0.9,
                    "metadata": {"_scope": "tenant-b", "tier": "gold"},
                }
            ],
        )

        with caplog.at_level(logging.WARNING):
            entries = await store.search("dark mode")

        assert _metadata(entries[0])["_scope"] == "tenant-a"
        assert _metadata(entries[0])["tier"] == "gold"
        assert [record for record in caplog.records if "'_scope'" in record.getMessage()] != []

    @pytest.mark.parametrize(
        "document",
        [
            {"_id": "abc", "content": "the user prefers dark mode"},
            {"_id": "abc", "content": "the user prefers dark mode", "score": None},
        ],
        ids=["no-score-field", "score-projected-null"],
    )
    async def test_an_unscored_document_maps_to_zero_rather_than_failing(self, document: dict[str, Any]) -> None:
        """``REQ-E-059`` says no result arrives unscored, and the fallback is for when one does.

        Both shapes, because they fail differently and neither is produced by any pipeline this package
        emits: an absent ``score`` is a ``KeyError`` without the fallback, a projected ``null`` is a
        ``TypeError``. Two unreachable lines elsewhere died the same way, by handing the collaborator
        something a deployment does not produce, and the alternative is a
        defense-in-depth line that nothing but a mutant ever reaches.
        """
        store = _store()
        _prepared(store, documents=[document])

        entries = await store.search("dark mode")

        assert _metadata(entries[0])["_relevance_score"] == 0.0

    async def test_only_the_documents_own_metadata_reaches_the_entry(self) -> None:
        """``REQ-E-040``'s shape: the caller's nested keys, this package's four, and nothing else.

        The document root holds every field ``add`` writes plus what the search stages add. Mapping from
        the root instead of from ``metadata`` hands the caller ``content_hash`` and ``score`` as
        if they had stored them, and hands them a ``scope`` key that sits beside the ``_scope`` this
        package reports and disagrees with it whenever a document was moved.
        """
        store = _store(scope="tenant-a")
        _prepared(
            store,
            documents=[
                {
                    "_id": "abc",
                    "content": "the user prefers dark mode",
                    "scope": "tenant-a",
                    "store_name": "facts",
                    "content_hash": "f" * 64,
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "score": 1.5,
                    "metadata": {"tier": "gold"},
                }
            ],
        )

        entries = await store.search("dark mode")

        assert entries[0].content == "the user prefers dark mode"
        assert entries[0].store_name == "facts"
        assert _metadata(entries[0]) == {
            "tier": "gold",
            "_id": "abc",
            "_relevance_score": 1.5,
            "_search_type": "text",
            "_scope": "tenant-a",
        }


class TestDeletingOneMemory:
    """AT-082 / REQ-E-076, REQ-E-078 — one filter out, and the ids that never reach the wire.

    The filter is asserted whole rather than probed term by term, for the reason ``AT-077`` gives: a
    dropped term narrows nothing a test seeded and so is invisible to a count. On this tier there is
    nothing seeded at all, since the fake resolves no matching, so the emitted dict is the entire claim
    here and ``AT-085`` is what watches another store's document survive.
    """

    async def test_the_filter_is_the_read_paths_predicate_with_the_id_added(self) -> None:
        """`_filter.compose` output, plus ``_id`` as a sibling of ``$and``.

        Sibling rather than a fourth conjunct, which is the same predicate: two top-level keys are an
        implicit conjunction. Asserted in the shape the code emits, so this test fails if the code starts
        building the predicate itself rather than borrowing the read path's.
        """
        store = _store(
            config=_connection_config(filter_fields=["metadata.tier"]),
            scope="tenant-a",
            filter={"metadata.tier": "gold"},
            writable=True,
        )
        collection = _prepared(store)
        collection.deleted = 1
        memory_id = ObjectId()

        deleted = await store.delete(str(memory_id))

        assert deleted == 1
        assert collection.deletes == [
            {
                "$and": [
                    {"scope": {"$eq": "tenant-a"}},
                    {"store_name": {"$eq": "facts"}},
                    {"metadata.tier": "gold"},
                ],
                "_id": memory_id,
            }
        ]

    async def test_the_count_comes_from_the_server_rather_than_from_this_method(self) -> None:
        """Seven, which no single-document delete can return, so a hardcoded ``1`` fails here."""
        store = _store(writable=True)
        collection = _prepared(store)
        collection.deleted = 7

        assert await store.delete(str(ObjectId())) == 7

    @pytest.mark.parametrize(
        "memory_id",
        ["", "nope", "zzzzzzzzzzzzzzzzzzzzzzzz", "6a96a03ca471f4a00187022", 12, None],
        ids=["empty", "short", "non-hex", "23-chars", "int", "none"],
    )
    async def test_an_id_that_is_not_an_object_id_is_rejected_before_the_wire(self, memory_id: Any) -> None:
        """``ValueError`` naming the argument, with nothing sent.

        ``None`` is the case worth having: measured on pymongo 4.17.0, ``ObjectId(None)`` returns a freshly
        generated id rather than failing, so an unchecked ``None`` composes into a well-formed filter that
        matches nothing and returns ``0``, reporting "already gone" for a call that named nothing at all
        (``REQ-E-076``). ``12`` is the other shape the annotation does not enforce: ``ObjectId(12)`` raises
        ``TypeError``, which is neither the driver error ``_reporting`` wraps nor the ``ValueError`` a
        caller mistake should be.
        """
        store = _store(writable=True)
        collection = _prepared(store)

        with pytest.raises(ValueError, match="memory_id"):
            await store.delete(memory_id)

        assert collection.deletes == []


class TestDeletingEveryMemoryThisStoreCanSee:
    """AT-083 / REQ-E-077, REQ-E-078 — the read path's predicate, with nothing added.

    Which is also the scope of the method, and why it stays an open question: the predicate carries
    ``store_name`` and the store's configured filter, so this empties one store's view rather than one
    scope. Two stores in a scope are two calls, and the README says so.
    """

    async def test_the_filter_is_exactly_the_read_paths_predicate(self) -> None:
        """No ``_id``, no extra term, and the configured filter still in it.

        The configured filter is the term a mutation can drop for free: `AT-083`'s own count comes from the
        fake, and on a deployment a store with no configured filter deletes the same documents either way.
        """
        store = _store(
            config=_connection_config(filter_fields=["metadata.tier"]),
            scope="tenant-a",
            filter={"metadata.tier": "gold"},
            writable=True,
        )
        collection = _prepared(store)

        await store.delete_all()

        assert collection.deletes == [
            {
                "$and": [
                    {"scope": {"$eq": "tenant-a"}},
                    {"store_name": {"$eq": "facts"}},
                    {"metadata.tier": "gold"},
                ]
            }
        ]

    async def test_the_count_comes_from_the_server(self) -> None:
        """Passed through, which is what makes the return worth reading in an erasure log."""
        store = _store(writable=True)
        collection = _prepared(store)
        collection.deleted = 12

        assert await store.delete_all() == 12


class TestDeletionIsAWrite:
    """AT-084 / REQ-E-079 — ``writable`` gates both deletes, and it gates them before setup runs.

    The error half of ``AT-084`` is in ``AT-080``'s parametrize below, since a wrapped driver error is the
    same claim for four methods and the class that owns it already builds a store on ``LEAKY_URI``.
    """

    @pytest.mark.parametrize("call", ["delete", "delete_all"])
    async def test_a_read_only_store_refuses_before_it_initializes(self, call: str) -> None:
        """``ValueError`` naming the key, from a store that has opened nothing.

        The store points at a reserved port with a 50 ms selection timeout, so a refusal that arrived after
        ``initialize()`` would be a ``StorageError`` rather than this. ``_initialized`` is asserted as well,
        because that is the claim rather than a symptom of it: a read-only store must turn a delete down
        without having created an index to do it, exactly as ``add`` does (``REQ-E-058``).
        """
        store = _store()

        with pytest.raises(ValueError, match="writable"):
            if call == "delete":
                await store.delete(str(ObjectId()))
            else:
                await store.delete_all()

        assert store._initialized is False


class TestTheScoreBoundOnScoresThisTestChose:
    """AT-079 / REQ-E-059 — the inclusive boundary and the ceiling, which real scores cannot reach.

    ``AT-059b`` closes the floor on the integration tier against real BM25 scores, and it sets its bound to
    the midpoint between two of them, which is the only bound a test can pick when it does not know what
    the scores will be. Two things follow from that, and both were unasserted until a mutation pass
    said so. A result scoring *exactly* the bound is never on either side of a midpoint, and it
    is the ordinary case rather than a corner: a caller who read a bound off an earlier result's
    ``_relevance_score`` lands on it every time. And ``max_score`` had no test applying it to a result set
    at all, so the ceiling branch was three unreached lines.

    Fixed scores cost nothing the live tier was providing. The bound is applied in Python over a list, so a
    deployment contributes the numbers and nothing else.
    """

    @staticmethod
    def _documents() -> list[dict[str, Any]]:
        """Three documents scoring 0.9, 0.5 and 0.1.

        Returns:
            The documents, in the order a search would return them.
        """
        return [
            {"_id": str(index), "content": f"fact {index}", "score": score}
            for index, score in enumerate((0.9, 0.5, 0.1))
        ]

    @pytest.mark.parametrize(
        ("bound", "kept"),
        [
            ({"min_score": 0.5}, ["fact 0", "fact 1"]),
            ({"max_score": 0.5}, ["fact 1", "fact 2"]),
            ({"min_score": 0.1}, ["fact 0", "fact 1", "fact 2"]),
        ],
        ids=["floor", "ceiling", "floor-at-the-lowest-score"],
    )
    async def test_a_bound_equal_to_a_results_score_keeps_that_result(
        self, bound: dict[str, float], kept: list[str]
    ) -> None:
        """Both bounds are inclusive, and each one drops only its own side.

        One parametrization per bound rather than three tests, because this is one rule read three times
        and the interesting result is the same document in the first two rows: ``fact 1`` scores the bound
        exactly and survives either way. The third row is the same boundary at the end of the list, where a
        bound that dropped what it equals empties a result set a caller was told to expect in full.
        """
        store = _store(**bound)
        _prepared(store, documents=self._documents())

        entries = await store.search("dark mode")

        assert [entry.content for entry in entries] == kept


_PRETEND_REASON = "no embedding model is registered (a stand-in for the real failure, not a server message)"
"""Reason a refused index build carries, worded so a failure report cannot be read as a real server error."""


class _SetupRecorder:
    """A stand-in for the two ``_indexes`` calls ``initialize()`` makes, which can refuse a named index.

    Patched over the module attributes rather than driven from a collection, because what ``initialize()``
    decides is which specs it passes and in which order, and every one of those is settled before a command
    is built. ``AT-060`` through ``AT-064`` are what test the real ``ensure`` against a deployment.

    Refusing by index name rather than by tier, so the refusal is expressed the way the deployment expresses
    it: one index will not build, and which tier that removes is what the store decides.

    Attributes:
        probes: The collection name for each capability probe.
        ordinary: One entry per ordinary ``ensure`` call, holding the spec names it carried.
        search: The name of each search index that was created, in creation order.
        timeouts: The ``timeout`` keyword of every call, ordinary and search alike, in call order.
            ``None`` for a call that passed none, which is the failure ``AT-126`` exists to catch.
        failing: Index names to refuse.
        available: What the capability probe answers.
    """

    def __init__(self, *, available: bool = True, failing: frozenset[str] = frozenset()) -> None:
        self.probes: list[str] = []
        self.ordinary: list[tuple[str, ...]] = []
        self.search: list[str] = []
        self.timeouts: list[float | None] = []
        self.failing = failing
        self.available = available

    async def probe(self, collection: Any) -> _indexes.SearchSupport:
        """Stand in for ``_indexes.search_support``.

        Yields to the event loop once, which is the one behaviour of the real function that matters to a
        caller here: it is a round trip, so it suspends, and a fake that never suspends would make every
        concurrency claim in this class pass by running sequentially.

        Returns the real ``SearchSupport`` rather than a two-field fake of its own, so the invariant the
        production code relies on holds here too: a code is present exactly when search is absent. The
        code is 6047401, one of the three ``REQ-E-053a`` measured, chosen because it is *not* the one this
        package spent 18 tasks treating as the only one.

        Args:
            collection: The collection being probed.

        Returns:
            :attr:`available`, wrapped, with a measured code when it is ``False``.
        """
        self.probes.append(collection.name)
        await asyncio.sleep(0)
        return _indexes.SearchSupport(available=self.available, code=None if self.available else 6047401)

    async def ensure(self, collection: Any, *, ordinary: Any = (), search: Any = (), **kwargs: Any) -> None:
        """Stand in for ``_indexes.ensure``, recording what it was asked for.

        Args:
            collection: Ignored; the specs are the claim.
            ordinary: Ordinary index specs.
            search: Search index specs, one per call from ``_ensure_search_tiers``.
            **kwargs: ``create``, ``timeout`` and ``interval``. ``timeout`` is recorded for
                ``AT-126``; the other two are not asserted here.

        Raises:
            StorageError: If a search spec's name is in :attr:`failing`, which is how the real function
                reports a build the server refused.
        """
        self.timeouts.append(kwargs.get("timeout"))
        if ordinary:
            self.ordinary.append(tuple(spec.name for spec in ordinary))
        for spec in search:
            if spec.name in self.failing:
                raise StorageError(f"strands_mongodb: creating {spec.name} failed: {_PRETEND_REASON}")
            self.search.append(spec.name)


def _patch_setup(
    monkeypatch: pytest.MonkeyPatch, *, available: bool = True, failing: frozenset[str] = frozenset()
) -> _SetupRecorder:
    """Replace both calls ``initialize()`` makes, so it runs to completion with no deployment.

    Patched on the module rather than on the store, which is what makes the count meaningful:
    ``initialize()`` reaches both through ``_indexes`` deliberately, so one store doing the work twice and
    ten stores doing it once each are both visible here (``AT-065``'s precedent).

    Args:
        monkeypatch: The fixture, so both patches are undone with the test.
        available: What the capability probe answers.
        failing: Search index names to refuse.

    Returns:
        The recorder, holding what was asked for once the test has run.
    """
    recorder = _SetupRecorder(available=available, failing=failing)
    monkeypatch.setattr(_indexes, "search_support", recorder.probe)
    monkeypatch.setattr(_indexes, "ensure", recorder.ensure)
    return recorder


class TestTheIndexReadyBudgetComesFromTheConfig:
    """AT-126 / ``REQ-E-122`` — the memory store's half of one config key, at both call sites.

    ``initialize()`` calls ``_indexes.ensure`` once for the ordinary specs and once per search tier, and
    only the search calls wait on anything. That asymmetry is the trap: a change that threads the key
    through the call that obviously needs it leaves the other one on the module default, and nothing
    observable differs until somebody raises the budget and finds half of it ignored. Both calls are
    asserted here, and ``test_indexes.py`` asserts that the set of call sites is the set checked.

    The default case matters as much as the override. ``REQ-E-123`` raised the module constant to 300
    seconds, and a store that read the key with a hard-coded fallback of its own would keep the old
    number alive in a second place.
    """

    async def test_every_call_carries_the_configured_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _patch_setup(monkeypatch)
        store = _store(config=_connection_config(index_ready_timeout=7))

        await store.initialize()

        # One ordinary call plus one per search tier, all of them carrying the caller's number rather
        # than the module's. `set` rather than a positional check: which tiers run is `AT-078`'s claim.
        assert len(recorder.timeouts) == 1 + len(recorder.search)
        assert set(recorder.timeouts) == {7}

    async def test_absence_resolves_to_the_module_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read through ``_indexes.READY_TIMEOUT_SECONDS`` rather than compared to a literal, so this
        keeps testing the wiring rather than restating ``AT-125``."""
        recorder = _patch_setup(monkeypatch)
        store = _store()

        await store.initialize()

        assert set(recorder.timeouts) == {_indexes.READY_TIMEOUT_SECONDS}


class TestWhatInitializeDecides:
    """AT-078 / REQ-E-052, REQ-E-053, REQ-E-053a, REQ-E-053b, REQ-E-067 — which indexes, ordered, fatal when.

    What a deployment shows is the outcome: which indexes exist afterwards, and that a warning happened.
    The claims here are the decisions upstream of it. Which spec was asked for, in what order, which
    refusal was re-raised instead of logged, and whether the lock was taken at all are all invisible in
    that outcome, because ``_indexes.ensure`` is idempotent and a store that redoes its setup leaves the
    same indexes behind as one that does not. Every claim below was carried by the code alone until Task
    8's mutation pass changed it and watched the suite stay green.

    One claim has no deployment at all. "Vector is preferred when both tiers are available" only decides an
    answer on the AutoEmbeddings path, where a plain string fits both tiers, and that is the index an Atlas
    CLI local deployment cannot build. The manual path does build both indexes there, measured in
    a demo run, but on it the query key picks the tier and the order never gets a vote.

    Patching ``_indexes`` is what makes the AutoEmbeddings both-available case exist, and the same patch is
    what lets one named index be refused on demand. Everything asserted here is a decision this package
    makes about what to send. What the server does with it is ``test_indexes.py``'s tier.
    """

    async def test_auto_creates_both_tiers_vector_first_and_prefers_the_vector_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The preference order, and why it is an order rather than a fallback.

        The tier that answers is the observable half. On the AutoEmbeddings path a plain string fits both
        tiers, so reversing the pair leaves every query answered lexically by a store whose
        vector index exists, was built, and is never queried. Nothing about the result set says so: the
        entries come back scored and ranked, on a different scale, from a different index.
        """
        recorder = _patch_setup(monkeypatch)
        store = _store()

        await store.initialize()

        assert recorder.search == ["vector_index", "text_index"]
        assert store._tiers == ("vector", "text")
        assert store._pick_tier(query_text="dark mode", query_vector=None) == "vector"

    async def test_a_named_search_type_creates_only_its_own_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``REQ-E-053``'s narrowing: a text store never creates a vector index it will not query.

        A cost claim as much as a correctness one, and on the AutoEmbeddings path the cost is per write:
        an unqueried vector index still embeds every document that lands in the collection.
        """
        recorder = _patch_setup(monkeypatch)
        store = _store(config=_connection_config(search_type="text"))

        await store.initialize()

        assert recorder.search == ["text_index"]
        assert store._tiers == ("text",)

    async def test_a_named_search_types_own_tier_failing_is_fatal_and_says_what_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``REQ-E-053``'s other half: one tier by request means nothing to fall through to.

        The assertion is on the message and not only on the type, because skipping the failure
        falls through to the empty-tier floor two lines below, which raises ``StorageError`` as well. Same
        class, different information: one names the index the server refused and carries its reason, the
        other names ``auto``, which this store is not.
        """
        _patch_setup(monkeypatch, failing=frozenset({"vector_index"}))
        store = _store(config=_connection_config(search_type="vector"))

        with pytest.raises(StorageError) as caught:
            await store.initialize()

        assert "vector_index" in str(caught.value)
        assert _PRETEND_REASON in str(caught.value)
        assert "'search_type' is 'auto'" not in str(caught.value)

    async def test_auto_with_every_tier_refused_warns_per_tier_then_raises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``REQ-E-053b``'s floor, and *where* it is raised is the claim.

        Returning an empty tier tuple instead leaves ``initialize()`` succeeding on a store that
        can never answer anything, and moves the first symptom to whichever query runs next, where the
        reason each tier failed is already several log lines behind.
        """
        _patch_setup(monkeypatch, failing=frozenset({"vector_index", "text_index"}))
        store = _store()

        with caplog.at_level(logging.WARNING), pytest.raises(StorageError) as caught:
            await store.initialize()

        assert "no search tier could be prepared" in str(caught.value)
        assert len([record for record in caplog.records if "search tier is not available" in record.getMessage()]) == 2
        assert store._initialized is False

    async def test_auto_landing_on_text_warns_that_answers_became_lexical(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``REQ-E-053``'s "warn when it lands below vector", which is a second warning and not the first.

        ``AT-053b`` asserts the per-tier line on a real deployment where the vector index cannot build,
        and that line reports what failed. This one reports the consequence in the caller's
        terms: a store still answering, semantically degraded, which is the half a reader acts on. Removing
        it left the suite green, because the other warning was still there to be found.
        """
        _patch_setup(monkeypatch, failing=frozenset({"vector_index"}))
        store = _store()

        with caplog.at_level(logging.WARNING):
            await store.initialize()

        messages = [record.getMessage() for record in caplog.records]
        assert store._tiers == ("text",)
        assert [message for message in messages if "vector search tier is not available" in message] != []
        assert [message for message in messages if "resolve to the text tier" in message] != []

    async def test_a_deployment_that_cannot_search_warns_once_for_ten_stores(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``REQ-E-053a``'s warning is per connection, which is the half ``AT-053a`` does not count.

        Ten stores built from one config share one ``_Connection``, so the probe runs once and the line
        reporting its result belongs inside that branch. Ten copies of a six-sentence warning on
        every process start is the difference between a log a reader scans and one they filter out, and the
        filter they write is the one that hides it on the eleventh store too.
        """
        recorder = _patch_setup(monkeypatch, available=False)
        stores = [_store(name=f"facts{index}") for index in range(10)]

        with caplog.at_level(logging.WARNING):
            await asyncio.gather(*(store.initialize() for store in stores))

        assert recorder.probes == ["memory"]
        assert len([record for record in caplog.records if "does not run Atlas Search" in record.getMessage()]) == 1
        assert recorder.search == []
        assert all(store._tiers == () for store in stores)

    async def test_the_second_initialize_neither_takes_the_lock_nor_redoes_the_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``REQ-E-052``'s idempotency, counted rather than inferred from a collection's indexes.

        An index count cannot see this. ``_indexes.ensure`` is idempotent by design, so a store that redid
        the whole setup on every call leaves a collection indistinguishable from one that did it
        once, and a store that only stopped taking the fast path leaves no trace at all. What is
        observable is the work: two calls, one probe, one ordinary ``ensure``, two search ``ensure``s, and
        the lock taken once. The lock is the reason the fast path is unlocked, and ``add`` and ``search``
        both call ``initialize()`` on every agent turn, so it is a lock every operation would queue on.
        """
        recorder = _patch_setup(monkeypatch)
        store = _store()
        real_get_lock = store._connection.get_lock
        locks = 0

        def counting() -> Any:
            nonlocal locks
            locks += 1
            return real_get_lock()

        store._connection.get_lock = counting  # type: ignore[method-assign]

        await store.initialize()
        await store.initialize()

        assert locks == 1
        assert recorder.probes == ["memory"]
        assert len(recorder.ordinary) == 1
        assert recorder.search == ["vector_index", "text_index"]

    async def test_one_store_initializing_twice_at_once_does_the_work_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flag re-read under the lock, which is the half of ``REQ-E-052`` ten stores cannot reach.

        ``AT-065``'s ten stores contend on the connection lock and then each does its own index work,
        because ``_initialized`` is per store. One store is the case that arrives on every agent turn:
        ``add`` and ``search`` both call ``initialize()``, so two operations issued together are two
        coroutines inside one flag. The second re-reads it after waiting, and without that read it repeats
        setup the first has already finished, on a store that is already usable.
        """
        recorder = _patch_setup(monkeypatch)
        store = _store()

        await asyncio.gather(store.initialize(), store.initialize())

        assert recorder.probes == ["memory"]
        assert len(recorder.ordinary) == 1
        assert recorder.search == ["vector_index", "text_index"]

    async def test_a_query_with_no_prepared_tier_names_index_creation_not_a_missing_module(self) -> None:
        """``_pick_tier``'s two messages for two remedies (``REQ-E-053a``, ``REQ-E-053b``).

        This deployment can search, so the other message's advice about pointing ``connection_string`` at
        Atlas would send a reader to replace something that is not broken. Reached by preparing nothing
        rather than by refusing a tier, because ``initialize()`` refuses to leave a store in this state and
        the guard covers the state anyway: ``_tiers`` is also empty on a store nobody initialized.
        """
        store = _store()
        _prepared(store, tiers=())
        store._connection.capabilities = _client.Capabilities(search=True)

        with pytest.raises(StorageError) as caught:
            await store.search("dark mode")

        assert "no search tier could be prepared" in str(caught.value)
        # The other message's own words rather than the server code it used to hard-code. Now
        # that code is interpolated, so `"31082" not in ...` would pass on a message built entirely from
        # the wrong template and prove nothing.
        assert "does not run Atlas Search" not in str(caught.value)

    async def test_a_query_that_no_prepared_tier_fits_names_the_key_to_add(self) -> None:
        """``REQ-E-074a`` where a manual-vector store meets ``MemoryManager`` (``AT-078``'s third message).

        The shipped call site always sends a plain string, and a manual-vector store asking for
        ``search_type='vector'`` has nothing that can answer one: the deployment does not embed on that path
        and this package does not either. ``ValueError`` rather than ``StorageError``, because the remedy is
        a key on the query rather than anything about the deployment, and the message carries both that key
        and the two config keys that chose the path the caller is fighting.
        """
        store = _store(config=_connection_config(search_type="vector", embedding_dimensions=4, similarity="cosine"))
        _prepared(store, tiers=("vector",))

        with pytest.raises(ValueError) as caught:
            await store.search("dark mode")

        assert "'vector'" in str(caught.value)
        assert "embedding_dimensions" in str(caught.value)


class TestADriverErrorArrivesNamingTheNamespaceAndNothingElse:
    """AT-080 / REQ-E-036a — the two driver calls ``AT-014``'s memory half never reaches.

    ``AT-014`` asserts ``StorageError`` out of ``initialize``, ``search`` and ``add`` against a real
    authentication failure, and it passed from its first run for a reason that hid a gap. An auth failure
    surfaces inside ``_indexes.search_support`` or ``_indexes.ensure``, and both wrap their own driver
    errors, so ``_reporting`` never sees a ``PyMongoError`` at all. The two calls that reach the driver
    unwrapped are ``update_one`` on the write path and ``aggregate`` on the read path. Both were
    unexercised: rendering the whole connection config into the message left the suite green,
    and ``REQ-E-036a`` exists because the behavior was real, needed, and asserted by nothing.

    Which is why the store is built on ``LEAKY_URI`` and the driver error carries no password. A real one
    does not, so the config this store is holding is the only route the password has into the message, and
    closing that route is the whole point of a decorator that renders a namespace and nothing else.
    ``_assert_no_leak`` is defined with ``AT-014``'s own tests below and shared rather than repeated.
    """

    @pytest.mark.parametrize("call", ["add", "search", "delete", "delete_all"])
    async def test_an_unwrapped_driver_error_names_the_operation_and_the_namespace(self, call: str) -> None:
        """The verb, the namespace, the chained cause, and neither the memory nor the password.

        Code 8 rather than a plausible code, because which code it is changes nothing here: ``_reporting``
        wraps every ``PyMongoError``, and a decorator that inspected the code would be the thing that
        decides some failures are not worth reporting.

        The two delete calls are ``AT-084``'s error half, added here rather than given a class of
        their own: ``delete_one`` and ``delete_many`` are the third and fourth driver calls this store
        makes unwrapped, so they belong to the gap this class exists to close.
        """
        verbs = {
            "add": "adding a memory to",
            "search": "searching",
            "delete": "deleting a memory from",
            "delete_all": "deleting every memory in",
        }
        content = "the user prefers dark mode"
        failure = OperationFailure("pretend failure, carrying no credentials", code=8)
        store = _store(config=_connection_config(connection_string=LEAKY_URI), writable=True)
        _prepared(store, failure=failure)

        with pytest.raises(StorageError) as caught:
            if call == "add":
                await store.add(content)
            elif call == "search":
                await store.search(content)
            elif call == "delete":
                await store.delete(str(ObjectId()))
            else:
                await store.delete_all()

        message = str(caught.value)
        assert verbs[call] in message
        assert store._target in message
        assert caught.value.__cause__ is failure
        assert content not in message
        _assert_no_leak(message, caught.value, "".join(traceback.format_exception(caught.value)))


# --- contract tier: a real deployment, no query ------------------------------------------------


@pytest.fixture
async def contract_store(contract_uri: str, test_database_name: str, unique_collection_name: str) -> Any:
    """A writable store scoped to ``tenant-a``, in this test's own collection, on a real deployment.

    One collection per test rather than one per session, because most of what this tier asserts is a
    document count and a shared collection makes every count another test's business.
    """
    store = _real_store(contract_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a")
    try:
        yield store
    finally:
        await store.close()


async def _require_no_search_support(store: MongoDBMemoryStore) -> int:
    """Skip unless the deployment really has no Atlas Search, naming why, and report the code it used.

    The tier marker says "any real MongoDB", which cannot express "one without mongot", and the three
    claims below are about the deployment that lacks it. CI points ``MONGODB_URI`` at ``mongo:8``, where
    the probe reports ``False``; a developer who points it at Atlas gets a skip with a reason rather
    than a failure they would learn to ignore.

    Returns the code so the callers can assert that the message names *this* deployment's code rather
    than a literal. Three codes mean this and they differ by server version and build, so a test that
    hard-coded one passed on the deployment it was written against and nowhere else. That is the defect
    this helper avoids, and asserting against a literal here would have hidden it.

    Args:
        store: The store whose collection to probe.

    Returns:
        The server error code this deployment answers the probe with.
    """
    support = await _indexes.search_support(store._collection())
    if support.available:
        pytest.skip("MONGODB_URI names a deployment with Atlas Search; REQ-E-053a is about one without")
    assert support.code is not None, "the probe reported no search support and no code, which it must not"
    return support.code


@pytest.mark.contract
class TestDedupAgainstTheUniqueIndex:
    """AT-042, AT-043 / REQ-E-042, REQ-E-043 — the reason the index is unique.

    Extraction writes are at-least-once and the coordinator re-drives the **whole batch** on any single
    failure, so the same fact arrives repeatedly by design. Without the unique partial index over
    ``(scope, store_name, content_hash)`` the collection grows once per retry, and the growth looks like
    the agent remembering the same thing twice.

    The hash covers content only, deliberately: a regenerated ``created_at`` must not be able to defeat
    dedup. ``$setOnInsert`` is what keeps the first-seen timestamp, which is asserted here because the
    obvious ``$set`` also passes a count test.
    """

    async def test_the_same_content_twice_stores_one_document(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """One document, and a result that says so rather than reporting two writes."""
        await contract_store.initialize()

        first = await contract_store.add("the user prefers dark mode")
        second = await contract_store.add("the user prefers dark mode")

        assert first.inserted == 1
        assert first.duplicates == 0
        assert second.inserted == 0
        assert second.duplicates == 1
        assert second.ids == first.ids
        assert await contract_database[unique_collection_name].count_documents({}) == 1

    async def test_a_duplicate_leaves_the_first_seen_timestamp_alone(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """``$setOnInsert``, not ``$set``: the second write must not move ``created_at`` forward.

        A store whose timestamps refresh on every retry has a recency index that sorts by "last
        extraction attempt", which is not what ``strands_scope_store_created_at`` is for.
        """
        await contract_store.initialize()
        await contract_store.add("the user prefers dark mode")
        collection = contract_database[unique_collection_name]
        before = (await collection.find_one({}))["created_at"]

        await contract_store.add("the user prefers dark mode")

        assert (await collection.find_one({}))["created_at"] == before

    async def test_a_concurrent_duplicate_key_error_is_reported_as_success(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """``REQ-E-043``, injected rather than raced, because the race is not reproducible on demand.

        The window is real and narrow: two upserts both match nothing, both insert, and the loser gets
        ``DuplicateKeyError``. Injecting it is the only way to assert *which* answer the loser gets, and
        the answer has to be the winner's id, since a caller that sees an error cannot tell a lost race
        from a rejected write. The natural race is asserted for its outcome in the next test.
        """
        await contract_store.initialize()
        first = await contract_store.add("the user prefers dark mode")
        real = contract_store._collection()

        class RaisingOnce:
            """The real collection, with ``DuplicateKeyError`` out of the first ``update_one``."""

            def __init__(self) -> None:
                self.raised = False

            def __getattr__(self, name: str) -> Any:
                return getattr(real, name)

            async def update_one(self, *args: Any, **kwargs: Any) -> Any:
                if not self.raised:
                    self.raised = True
                    raise DuplicateKeyError("E11000 duplicate key error collection: injected", 11000)
                return await real.update_one(*args, **kwargs)

        injected = RaisingOnce()
        contract_store._collection = lambda: injected  # type: ignore[assignment, return-value]

        second = await contract_store.add("the user prefers dark mode")

        assert injected.raised is True
        assert second.duplicates == 1
        assert second.inserted == 0
        assert second.ids == first.ids
        assert await contract_database[unique_collection_name].count_documents({}) == 1

    async def test_ten_concurrent_adds_of_one_fact_store_one_document(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """The race as it actually arrives, asserted on the outcome instead of on the exception.

        Whether any of the ten hits the ``DuplicateKeyError`` window depends on the server's timing, so
        asserting that one did would be flaky. What cannot vary is the outcome: one document, one
        insert, nine duplicates, no exception.
        """
        await contract_store.initialize()

        results = await asyncio.gather(
            *(contract_store.add("the user prefers dark mode") for _ in range(10)),
            return_exceptions=True,
        )

        assert [result for result in results if isinstance(result, BaseException)] == []
        # Narrowed by a second comprehension rather than suppressed. `return_exceptions=True` types the
        # list as `MongoDBAddResult | BaseException`, and mypy does not carry the assertion above into the
        # two lines below, so reading `.inserted` off the union is an error it is right to report.
        added = [result for result in results if not isinstance(result, BaseException)]
        assert sum(result.inserted for result in added) == 1
        assert sum(result.duplicates for result in added) == 9
        assert await contract_database[unique_collection_name].count_documents({}) == 1


@pytest.mark.contract
class TestInitializeIsIdempotent:
    """AT-045a / REQ-E-045 — twice is the same as once, on the ordinary index set.

    Called on every ``search`` and every ``add`` (``REQ-E-045``), so the warm path has to be cheap and
    must not accumulate anything. ``_indexes.ensure``'s own idempotency is asserted elsewhere; what this adds
    is that ``initialize()`` is the caller that keeps it, rather than passing a different spec set the
    second time and creating a second index under another name.
    """

    async def test_initializing_twice_creates_one_index_set(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """The three memory specs plus ``_id_``, after two calls."""
        await contract_store.initialize()
        await contract_store.initialize()

        information = await contract_database[unique_collection_name].index_information()

        assert set(information) == {
            "_id_",
            "strands_scope_store_content_hash_unique",
            "strands_scope_store_created_at",
            "strands_expires_at_ttl",
        }


@pytest.mark.contract
class TestTheStoreStampsTheScopeItself:
    """AT-046, AT-048a / REQ-E-046, REQ-E-048 — the scope stamp, on the write path.

    Isolation here is application-enforced, not database-enforced: there is no server-side
    row filter, so a document lands in the right scope only because this package put it there. Every
    assertion reads the stored document rather than the returned result, because the result is this
    package's own report and the document is what the next query will match.

    ``REQ-E-048``'s sentinel is the other half. An absent ``scope`` resolving to "no predicate" would make
    the *unconfigured* store the one that reads everything, which is the failure mode that looks like a
    relevance quirk instead of a leak.
    """

    async def test_the_stored_document_carries_the_stores_scope_and_name(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """The four fields the dedup and recency indexes are built over, plus the content."""
        await contract_store.initialize()

        await contract_store.add("the user prefers dark mode")

        document = await contract_database[unique_collection_name].find_one({})
        assert document["scope"] == "tenant-a"
        assert document["store_name"] == "facts"
        assert document["content"] == "the user prefers dark mode"
        assert document["created_at"]
        # Over the content and nothing else, which is what makes dedup survive a retry: a hash covering
        # `created_at` would make every regenerated entry a new document.
        assert document["content_hash"] == hashlib.sha256(b"the user prefers dark mode").hexdigest()

    async def test_caller_metadata_cannot_set_the_write_scope(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """Caller metadata is nested, so a key named ``scope`` inside it is inert rather than rejected.

        Asserted rather than assumed, because the tempting implementation is to merge caller metadata
        into the document root, where ``{"scope": "tenant-b"}`` would be a one-line cross-scope write.
        The nested copy stays visible: dropping it would be silent data loss on a legitimate key that
        merely shares a name with ours.
        """
        await contract_store.initialize()

        await contract_store.add("a fact", {"scope": "tenant-b", "_scope": "tenant-b"})

        document = await contract_database[unique_collection_name].find_one({})
        assert document["scope"] == "tenant-a"
        assert document["metadata"]["scope"] == "tenant-b"

    async def test_a_configured_filter_cannot_set_the_write_scope(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        contract_database: Any,
    ) -> None:
        """``filter`` is read-path narrowing and the write path does not consult it at all.

        ``scope`` is one of the two baked-in filterable fields (``_indexes.declared_filter_fields``), so
        this filter passes ``REQ-E-049``'s declared-field check and would be accepted by a query. That is
        what makes it worth asserting on the write path: the value is valid, it just has no authority
        here.
        """
        store = _real_store(
            contract_uri,
            test_database_name,
            unique_collection_name,
            writable=True,
            scope="tenant-a",
            filter={"scope": "tenant-b"},
        )
        try:
            await store.initialize()
            await store.add("a fact")
        finally:
            await store.close()

        assert await contract_database[unique_collection_name].count_documents({"scope": "tenant-a"}) == 1
        assert await contract_database[unique_collection_name].count_documents({"scope": "tenant-b"}) == 0

    async def test_a_store_with_no_scope_stamps_the_sentinel(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        contract_database: Any,
    ) -> None:
        """``AT-048a``. The field is present and equal to the sentinel, never missing and never empty.

        Missing is the dangerous one: the dedup index is partial on ``content_hash`` and the scope
        predicate would have to become ``{"scope": {"$exists": False}}`` to match, which no other query in
        this package writes. Empty is the quiet one, because ``""`` is falsy and every ``if scope:`` in a
        future edit would treat it as unset.

        ``store.scope`` reports the resolved value rather than the caller's ``None``, which departs from
        ``BedrockKnowledgeBaseStore``: reading ``None`` there while the documents carry a sentinel would
        let a reader conclude the store is unscoped when it is scoped to ``__unscoped__``.
        """
        store = _real_store(contract_uri, test_database_name, unique_collection_name, writable=True)
        try:
            assert store.scope == UNSCOPED
            assert UNSCOPED == "__unscoped__"

            await store.initialize()
            await store.add("a fact from a store nobody scoped")
        finally:
            await store.close()

        document = await contract_database[unique_collection_name].find_one({})
        assert document["scope"] == UNSCOPED


@pytest.mark.contract
class TestWritesStayInTheirScope:
    """AT-047b / REQ-E-047 — two stores, one collection, and dedup that does not cross the boundary.

    The read half is ``AT-047a`` on the integration tier, for the reason in the module docstring. This
    half is the one a plain deployment can prove, and it carries a claim the read half cannot: the unique
    index leads with ``scope``, so the *same* fact in two scopes is two documents. An index over
    ``content_hash`` alone would collapse them into one and hand tenant B a document written by tenant A,
    which is a cross-scope leak arriving through the dedup path rather than through a query.
    """

    async def test_the_same_fact_in_two_scopes_is_two_documents(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        contract_database: Any,
    ) -> None:
        """Both writes insert, and each lands under its own scope."""
        first = _real_store(contract_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a")
        second = _real_store(contract_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-b")
        try:
            await first.initialize()

            a = await first.add("the user prefers dark mode")
            b = await second.add("the user prefers dark mode")
        finally:
            await first.close()
            await second.close()

        assert a.inserted == 1
        assert b.inserted == 1
        assert a.ids != b.ids
        collection = contract_database[unique_collection_name]
        assert await collection.count_documents({"scope": "tenant-a"}) == 1
        assert await collection.count_documents({"scope": "tenant-b"}) == 1


@pytest.mark.contract
class TestDeletionStaysInsideTheStoreThatIssuedIt:
    """AT-085 / REQ-E-076–REQ-E-079 — real documents, no query.

    Contract rather than integration, which is the tier rule rather than a shortcut: the
    memory read path needs mongot and nothing here reads through ``search``. The id comes from
    ``MongoDBAddResult.ids[0]``, the same string ``search`` projects into ``_id`` metadata, so the round
    trip a caller makes is the round trip under test.

    This is what the unit tier cannot claim. There a fake records the filter it was handed; here a server
    decides what that filter matches, which is the difference between asserting a predicate and asserting
    an outcome.
    """

    async def test_a_memory_deletes_once_and_the_second_attempt_reports_nothing(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """One of two documents goes, the other stays, and deleting the same id again returns ``0``.

        The second call is ``REQ-E-078``'s "matched nothing is not an error" on the path a retry takes: an
        erasure job that runs twice must not fail the second time.
        """
        first = await contract_store.add("the user prefers dark mode")
        await contract_store.add("the user works in emea")
        collection = contract_database[unique_collection_name]

        assert await contract_store.delete(first.ids[0]) == 1
        assert await collection.count_documents({}) == 1
        assert await contract_store.delete(first.ids[0]) == 0
        assert await collection.count_documents({}) == 1

    async def test_a_sibling_stores_id_deletes_nothing_and_reads_the_same_as_an_absent_one(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        contract_database: Any,
    ) -> None:
        """``REQ-E-078``'s oracle argument, as two return values that have to be equal.

        A well-formed id belonging to another store in the same scope, and an id belonging to nothing at
        all, both answer ``0``. If the first answered anything else, a loop over ids would tell a caller
        which documents exist outside their own store, one call at a time, without ever deleting one.
        """
        mine = _real_store(contract_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a")
        sibling = _real_store(
            contract_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a", name="notes"
        )
        try:
            theirs = await sibling.add("the user prefers dark mode")

            absent = await mine.delete(str(ObjectId()))
            borrowed = await mine.delete(theirs.ids[0])
        finally:
            await mine.close()
            await sibling.close()

        assert borrowed == absent == 0
        collection = contract_database[unique_collection_name]
        assert await collection.count_documents({"store_name": "notes"}) == 1

    async def test_delete_all_empties_its_own_store_and_leaves_the_sibling(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        contract_database: Any,
    ) -> None:
        """The open question as an assertion rather than a paragraph.

        Two memories in one store, one in a sibling, all three in one scope and one collection. The count
        returned is exactly the two, and the sibling's document is still there afterwards, which is the
        behavior a customer answering a deletion request has to know about: emptying a scope served by two
        stores is two calls.
        """
        mine = _real_store(contract_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a")
        sibling = _real_store(
            contract_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a", name="notes"
        )
        try:
            await mine.add("the user prefers dark mode")
            await mine.add("the user works in emea")
            await sibling.add("the user prefers dark mode")

            deleted = await mine.delete_all()
        finally:
            await mine.close()
            await sibling.close()

        assert deleted == 2
        collection = contract_database[unique_collection_name]
        assert await collection.count_documents({"store_name": "facts"}) == 0
        assert await collection.count_documents({"store_name": "notes"}) == 1


@pytest.mark.contract
class TestADeploymentWithNoSearchSupport:
    """AT-053a, AT-053c / REQ-E-053, REQ-E-053a — measured rather than assumed.

    Measured 2026-09-01 against ``mongo:8``, reported version 8.3.8 with ``modules: []``: every search
    surface fails identically, ``OperationFailure`` code **31082**, while ordinary index creation and
    every CRUD path work normally. So this deployment can hold memories and cannot read them back.

    Measured again 2026-09-02 against three more deployments, which changed how these tests assert. Code
    31082 is what ``mongo:8`` answers and not what ``mongo:6``, ``mongo:7`` or an enterprise ``mongod``
    7.0.12 answer, so every assertion below reads the code from the deployment under test instead of
    naming one. `REQ-E-053a` has the table. A suite pinned to one deployment's code passed here and
    hid a probe that raised on the other three.

    ``initialize()`` degrades and ``search()`` raises, which is the line this package draws from the vended
    precedent: capability probing may legitimately degrade, query paths log and re-raise. Failing
    ``initialize()`` would take away the half that works; returning zero results from ``search()`` would
    be an empty memory that reads as empty, since v0.1.0 has no tier that runs without mongot and there
    is nothing for ``auto`` to resolve to.

    The message is not the missing-index one. A missing or ``NOT READY`` index on a deployment that *does*
    support search is ``REQ-E-061``/``REQ-E-062``, and those quote ``createSearchIndexes``. Any of the
    three codes means that command does not exist here, so quoting it would be advice the caller cannot
    act on. Two conditions, two messages, asserted apart below.
    """

    async def test_initialize_completes_with_the_ordinary_indexes_and_one_warning(
        self,
        contract_store: MongoDBMemoryStore,
        contract_database: Any,
        unique_collection_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The degrade path: three ordinary indexes present, search indexes skipped, one warning."""
        code = await _require_no_search_support(contract_store)

        with caplog.at_level(logging.WARNING, logger="strands_mongodb.memory_store"):
            await contract_store.initialize()

        warnings = [
            record
            for record in caplog.records
            if record.name == "strands_mongodb.memory_store" and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1, [record.getMessage() for record in warnings]
        assert str(code) in warnings[0].getMessage()
        assert "Atlas" in warnings[0].getMessage()
        assert set(await contract_database[unique_collection_name].index_information()) == {
            "_id_",
            "strands_scope_store_content_hash_unique",
            "strands_scope_store_created_at",
            "strands_expires_at_ttl",
        }

    async def test_a_write_still_works_on_a_deployment_that_cannot_read(
        self, contract_store: MongoDBMemoryStore, contract_database: Any, unique_collection_name: str
    ) -> None:
        """The half that keeps working is why ``initialize()`` degrades instead of raising."""
        await _require_no_search_support(contract_store)
        await contract_store.initialize()

        result = await contract_store.add("the user prefers dark mode")

        assert result.inserted == 1
        assert await contract_database[unique_collection_name].count_documents({}) == 1

    async def test_search_on_auto_raises_naming_the_code_and_what_to_connect_to(
        self, contract_store: MongoDBMemoryStore
    ) -> None:
        """``AT-053c``'s second half. ``auto`` is the shipped default, so this is the unconfigured path."""
        code = await _require_no_search_support(contract_store)
        await contract_store.initialize()

        with pytest.raises(StorageError) as caught:
            await contract_store.search("dark mode")

        assert str(code) in str(caught.value)
        assert "Atlas" in str(caught.value)

    async def test_search_type_vector_raises_about_the_deployment_not_about_an_index(
        self, contract_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """``AT-053a``. ``REQ-E-053`` already says a named tier raises; this pins *which* message.

        The absent strings are the assertion. A caller told to run ``createSearchIndexes`` on a
        deployment where that command is unavailable spends the afternoon on it.
        """
        store = _real_store(
            contract_uri,
            test_database_name,
            unique_collection_name,
            writable=True,
            connection={"search_type": "vector"},
        )
        try:
            code = await _require_no_search_support(store)
            await store.initialize()

            with pytest.raises(StorageError) as caught:
                await store.search("dark mode")
        finally:
            await store.close()

        assert str(code) in str(caught.value)
        assert "createSearchIndexes" not in str(caught.value)
        assert "NOT READY" not in str(caught.value)


def _with_password(uri: str, secret: str = SECRET) -> str:
    """Return ``uri`` with ``u:<secret>@`` as its userinfo, replacing whatever was there.

    Args:
        uri: A connection string.
        secret: The password to embed.

    Returns:
        The rewritten URI. Credentials that do not authenticate are the point: the failing path is
        where a driver error is most likely to echo them back.
    """
    scheme, _, rest = uri.partition("://")
    host = rest.partition("@")[2] or rest
    return f"{scheme}://u:{secret}@{host}"


def _assert_no_leak(*subjects: Any) -> None:
    """Assert no subject renders the password.

    The same helper as ``test_storage.py``'s ``assert_no_leak``, duplicated rather than shared for the
    same reason each test module defines its own ``UNREACHABLE``: a failure here should point at this
    file. The failure message names the subject's position and type and never the password, since a
    CI log is exactly where it must not appear.

    Args:
        *subjects: Strings, or objects to render with ``repr``.
    """
    for index, subject in enumerate(subjects):
        text = subject if isinstance(subject, str) else repr(subject)
        assert SECRET not in text, f"subject {index} ({type(subject).__name__}) leaked the password"


@pytest.mark.contract
class TestNoMemoryMethodRendersThePassword:
    """AT-014's memory half / REQ-E-014 — the credential sweep, completed in two parts.

    ``test_client.py`` closed construction and the registry; ``test_storage.py`` closed the five storage
    methods. The four memory methods are what was left, and they are the ones that fail in production: a
    store built once at start-up, then a driver error on every call, each one rendered into a log line
    or a traceback that reaches a ticket.

    Both paths run against the real deployment rather than against a closed port, which is what makes
    the marker ``contract``. The failing path uses credentials the server rejects, measured as
    ``OperationFailure`` code 18: an authentication failure is the error most likely to echo what it
    authenticated with. The succeeding path supplies an injected client alongside a password-bearing
    ``connection_string``, so every call reaches a live deployment while the config still holds a
    secret. That combination is constructed rather than typical, and it is the shape ``REQ-E-014``
    exists for: ``_Connection`` stores no URI precisely because a dataclass ``__repr__`` would print it.

    The config is not a subject. It holds the password by construction, which is the premise; the
    requirement is that nothing this package writes copies it out.
    """

    @pytest.mark.parametrize("call", ["initialize", "search", "add"])
    async def test_a_wrapped_driver_failure_renders_no_password(
        self, call: str, contract_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """Each of the three methods that reach the network, on a real authentication failure."""
        store = _real_store(
            _with_password(contract_uri),
            test_database_name,
            unique_collection_name,
            writable=True,
        )
        try:
            with pytest.raises(StorageError) as caught:
                if call == "initialize":
                    await store.initialize()
                elif call == "search":
                    await store.search("dark mode")
                else:
                    await store.add("a fact")
        finally:
            await store.close()

        _assert_no_leak(
            str(caught.value),
            caught.value,
            "".join(traceback.format_exception(caught.value)),
        )

    async def test_the_succeeding_path_renders_no_password_at_debug(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``DEBUG``, because that is the level a user turns on when something is wrong.

        ``get_tools`` is in here despite touching nothing: it is one of the four members the SDK calls,
        and the cheapest way for it to start leaking is a debug line naming the store it belongs to.
        """
        client: AsyncMongoClient[Any] = AsyncMongoClient(contract_uri)
        config: dict[str, Any] = {
            "client": client,
            "connection_string": _with_password(contract_uri),
            "database": test_database_name,
        }
        store = MongoDBMemoryStore(
            name="facts",
            # A plain dict rather than a `MongoDBConfig`, because holding a password alongside an
            # injected client is the shape this test constructs on purpose and no `TypedDict` narrows to
            # it without repeating the whole literal inline.
            config=config,  # type: ignore[arg-type]
            collection=unique_collection_name,
            writable=True,
        )
        try:
            with caplog.at_level(logging.DEBUG):
                await store.initialize()
                await store.add("the user prefers dark mode")
                store.get_tools()
        finally:
            await store.close()
            # Closed here rather than by `store.close()`, which never closes an injected client
            # (`REQ-E-010`). That asymmetry is the reason this path can hold a secret at all.
            await client.close()

        _assert_no_leak(*[record.getMessage() for record in caplog.records], store, store._connection)

    async def test_neither_the_store_nor_its_connection_renders_the_password(self) -> None:
        """``repr`` is where this leaks by accident: one added later for debugging, or a decorator.

        Needs no deployment and is here anyway, beside the two halves it belongs with. ``_Connection``
        is a dataclass, so it generates a ``__repr__`` over every field, and the field it deliberately
        does not have is the connection string.
        """
        store = _store(config=_connection_config(connection_string=LEAKY_URI))

        _assert_no_leak(store, store._connection, store._connection.client)


@pytest.mark.contract
class TestTenStoresShareOneSetup:
    """AT-065's ``initialize()`` half / REQ-E-012, REQ-E-045 — the client, the probe and the indexes.

    ``test_indexes.py`` closes the index half against ``_indexes.ensure`` directly, with no lock, because ten
    concurrent identical ``create_indexes`` calls were measured to produce zero failures and one index.
    What ``initialize()`` adds is two more counts that only exist once there is a store: one client for
    ten stores built from one config (``REQ-E-012``), and one capability probe rather than ten.

    The probe count is a spy, and it is the one assertion here that cannot be an outcome. The result of
    a probe is a boolean every store would compute identically, so a per-store probe is invisible in
    behavior and visible only as ten round trips on every process start.
    """

    async def test_ten_stores_initializing_concurrently_share_one_client_probe_and_index_set(
        self,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
        contract_database: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ten distinct stores, one collection, one ``asyncio.gather``."""
        probed: list[str] = []
        real_probe = _indexes.search_support

        async def counting(collection: Any) -> _indexes.SearchSupport:
            probed.append(collection.name)
            return await real_probe(collection)

        monkeypatch.setattr(_indexes, "search_support", counting)
        stores = [
            _real_store(contract_uri, test_database_name, unique_collection_name, name=f"facts{index}")
            for index in range(10)
        ]
        try:
            results = await asyncio.gather(*(store.initialize() for store in stores), return_exceptions=True)

            assert [result for result in results if isinstance(result, BaseException)] == []
            assert len({id(store._connection.client) for store in stores}) == 1
            assert len(_client._REGISTRY) == 1
            assert probed == [unique_collection_name]
            assert len(await contract_database[unique_collection_name].index_information()) == 4
        finally:
            await stores[0].close()


# --- integration tier: mongot, because no v0.1.0 search tier runs without it -----------------------

MANUAL_VECTOR: dict[str, Any] = {"embedding_dimensions": 4, "similarity": "cosine"}
"""Connection keys selecting the manual-vector path, matching ``test_pipeline.py``'s corpus width.

Four dimensions rather than a realistic 1024, because nothing here measures embedding quality and a
hand-written vector is the only kind whose expected neighbour is computable by reading it.
"""

QUERY_VECTOR: list[float] = [1.0, 0.0, 0.0, 0.0]
"""The query for every vector-tier test, aligned with the seeded vectors on the first axis."""

QUERY_TEXT = "dark mode"
"""The query for every text-tier test. Present in every seeded document, so ranking is not the claim."""


def _metadata(entry: MemoryEntry) -> dict[str, Any]:
    """Return an entry's metadata, failing the test if it has none.

    ``MemoryEntry.metadata`` is ``Metadata | None`` because the SDK allows a store to omit it, and this
    package never does (`REQ-E-040`). Asserting that once here keeps every caller reading as the claim
    it is making rather than as a chain of optional-narrowing, and mypy checks these tests
    (``pyproject.toml`` ``files``), so the narrowing is not optional.

    Args:
        entry: The entry to read.

    Returns:
        The entry's metadata.
    """
    assert entry.metadata is not None, f"REQ-E-040 requires metadata on every entry; {entry.content!r} has none"
    return dict(entry.metadata)


_SEARCHABLE_BUDGET: Final[float] = 180.0
"""How long a write is waited on before it counts as never having arrived (`REQ-E-091`).

Measured on `strands-dev`: a document written after both search indexes read ``READY`` first answered a
query **31 seconds later**, and a later run measured 15.5 seconds for the same thing on the same cluster.

It was 10 seconds until 2026-09-02, calibrated against an Atlas CLI local deployment, where the
shipped default cannot build its vector index, degrades to full text, and has a document
searchable in under a second. Six integration tests failed on it the first time the suite met a cluster
with a registered embedding model, every one with an empty result rather than a wrong one. It was then 90
seconds, ``AT-086``'s figure for the same boundary, until the all-tiers run of 01:45:57 on 2026-09-02 spent
the whole 90 twice and came back empty: ``AT-045b`` and ``AT-059b``, both of which then passed against the
same cluster minutes later with nothing else running, in **109.5 seconds each**, nearly all of it the two
index builds inside their first ``add()`` and **under 5 seconds** of it this wait. The lag is not a constant.
It stretches while a run is driving several dozen search indexes through creation and initial sync on one
cluster, and the failing runs never measured how far, because a budget that is exceeded stops timing.

A budget is not a delay: every caller below polls twice a second and returns as soon as its premise is
satisfied, so a quiet cluster runs exactly as fast at 180 as it did at 10.
"""


async def _answered(
    store: MongoDBMemoryStore, query: Any, *, expected: int = 1, budget: float = _SEARCHABLE_BUDGET
) -> list[Any]:
    """Search until mongot has the corpus, or give up and return the last short result.

    Same reason as ``test_pipeline.py``'s ``_until_answered``: ``READY`` means the index exists, not
    that every document has reached it, so a query issued straight after ``initialize()`` can
    legitimately return nothing. On the AutoEmbeddings path there is a second wait behind the first,
    because Atlas generates the embedding after the write acknowledges (`REQ-E-091`). Polling rather
    than sleeping a fixed interval, because the wait is a property of the deployment.

    ``expected`` is what makes this wait for *the corpus* rather than for the first document to arrive.
    Documents reach mongot one at a time, so a caller whose premise is two documents and who waits only
    for non-empty gets the first one and a failure that reads as a wrong result rather than as a race.
    Measured: this is the flake ``AT-059b`` hit, where the corpus assertion saw one of two.

    Returning the short list rather than failing here keeps the failure at the assertion that cares,
    where the message names what the test expected instead of naming this helper.

    Args:
        store: The store to query.
        query: A string or a ``MongoDBSearchQuery``.
        expected: How many results the caller's premise needs. Never an upper bound: a test asserting
            that a scope predicate *excludes* something has to keep this at 1 and assert the count
            itself, or the wait would hide the extra document it is looking for.
        budget: Seconds to keep asking before giving up, polling twice a second.

    Returns:
        The first result list holding at least ``expected`` entries, or the last shorter one.
    """
    results: list[Any] = []
    deadline = time.monotonic() + budget
    while True:
        try:
            results = list(await store.search(query))
        except StorageError as error:
            # `REQ-E-124`. Around the search call alone, and only for the one code that means mongot
            # never answered. Every caller below reaches search through this helper, so this is the
            # whole of the memory store's exposure; anything that fails elsewhere still fails.
            skip_if_search_is_unreachable(error)
            raise
        if len(results) >= expected or time.monotonic() >= deadline:
            return results
        await asyncio.sleep(0.5)


async def _seed_vectors(collection: Any, *, scope: str, store_name: str, facts: dict[str, list[float]]) -> None:
    """Write vector-bearing documents directly, bypassing ``add``.

    Out of band on purpose: ``add`` takes content and metadata and no
    vector, the manual path indexes an ``embedding`` field, and client-side embedding is a non-goal,
    so no ``MemoryStore`` method can put a vector in a document. The read path under test
    here is `REQ-E-074`'s, which ``test_pipeline.py`` already exercises one layer down. Writing these with
    ``insert_many`` follows ``test_pipeline.py``'s integration corpus rather than inventing a write API
    to make the test look tidier.

    The document shape is the stored one, including the fields the scope pre-filter reads. A missing ``scope``
    or ``store_name`` here would make `AT-047a` pass by returning nothing.

    Args:
        collection: Collection to write to.
        scope: Value for the ``scope`` field.
        store_name: Value for the ``store_name`` field.
        facts: Content mapped to the embedding to store with it.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    await collection.insert_many(
        [
            {
                _indexes.CONTENT_FIELD: content,
                _indexes.VECTOR_FIELD: vector,
                "scope": scope,
                "store_name": store_name,
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "created_at": now,
            }
            for content, vector in facts.items()
        ]
    )


@pytest.fixture
async def atlas_store(atlas_uri: str, test_database_name: str, unique_collection_name: str) -> Any:
    """A writable store scoped to ``tenant-a`` on a deployment with mongot, on the shipped default.

    Default means `REQ-E-067`'s AutoEmbeddings path and ``search_type="auto"``, which is what a user
    who configures nothing gets. On an Atlas CLI local deployment that path cannot build its index,
    so this fixture is also the degraded-default fixture, and `AT-053b` is the test that
    says so out loud.
    """
    store = _real_store(atlas_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a")
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.integration
class TestWhatASearchHandsBack:
    """AT-040 / REQ-E-040 — the retrieval boundary, closed end to end.

    The whole round trip on one surface: ``add`` writes, mongot indexes, ``search`` reads, and what
    comes back is what the SDK's ``MemoryManager`` and ``search_memory`` tool will format. Every layer
    below this has its own test and none of them can see a ``MemoryEntry``.

    Queried with a plain string rather than a ``MongoDBSearchQuery``, because both SDK call sites pass a
    ``str`` (`REQ-E-041`'s amendment measured this), so the string form is the production path and the
    widened form is the extension.
    """

    async def test_a_search_returns_memory_entries_carrying_the_four_reserved_keys(
        self, atlas_store: MongoDBMemoryStore
    ) -> None:
        """AT-040. Caller metadata survives beside the reserved keys, under them on collision."""
        await atlas_store.add("the user prefers dark mode", metadata={"source": "chat"})
        results = await _answered(atlas_store, QUERY_TEXT)

        assert results, f"the corpus never became searchable within {_SEARCHABLE_BUDGET:.0f} seconds"
        entry = results[0]
        assert isinstance(entry, MemoryEntry)
        assert entry.content == "the user prefers dark mode"
        assert entry.store_name == "facts"
        metadata = _metadata(entry)
        # Named individually rather than as a set comparison, so a failure says which key is missing
        # instead of printing two sets for the reader to diff.
        assert metadata["_id"]
        assert metadata["_scope"] == "tenant-a"
        assert metadata["_search_type"] in {"vector", "text"}
        assert isinstance(metadata["_relevance_score"], float)
        assert metadata["source"] == "chat"

    async def test_every_reserved_key_is_present_on_every_entry(self, atlas_store: MongoDBMemoryStore) -> None:
        """AT-040, over more than one result.

        The test above reads ``results[0]``, which an implementation stamping the keys onto the first
        entry alone would satisfy. Three facts, all matching, all checked.
        """
        for fact in ("dark mode is preferred", "dark mode at night", "dark mode everywhere"):
            await atlas_store.add(fact)
        results = await _answered(atlas_store, QUERY_TEXT, expected=3)

        assert len(results) == 3, f"expected all three facts back, got {[entry.content for entry in results]}"
        for entry in results:
            assert RESERVED_METADATA_KEYS <= set(_metadata(entry))


@pytest.mark.integration
class TestTheFirstSearchSetsItselfUp:
    """AT-045b / REQ-E-045 — the read half of "``initialize()`` is not the caller's responsibility".

    ``AT-047b`` covers the write half on the contract tier, incidentally rather than by design: its
    second store is never initialized and its ``add`` lands anyway. ``AT-045a`` is the idempotency claim
    and calls ``initialize()`` explicitly, so it is not that half. This half is the one that needs
    mongot, because a search index that does not exist is a query failure rather than a slow write.

    Worth its own class because the lazy path and the eager path differ in exactly one way that
    matters: on the lazy path the index is created *inside* the call the user is awaiting, so if
    ``initialize`` did not wait for ``READY`` the first search would return nothing and look like an
    empty memory rather than a race.
    """

    async def test_searching_a_store_nobody_initialized_returns_results(
        self, atlas_store: MongoDBMemoryStore, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """AT-045b. No ``initialize()`` call anywhere in this test, and no ``add`` before the write."""
        await atlas_store.add("the user prefers dark mode")
        results = await _answered(atlas_store, QUERY_TEXT)

        assert [entry.content for entry in results] == ["the user prefers dark mode"]
        names = [index["name"] async for index in await atlas_database[unique_collection_name].list_search_indexes()]
        assert names, "a search answered, so a search index has to exist"


@pytest.mark.integration
class TestNoSearchCrossesAScope:
    """AT-047a / REQ-E-047 — scope isolation, run once per search tier.

    The two tiers enforce scope by different mechanisms, and the tier is chosen by config rather than by
    test infrastructure, so both halves live here. Text uses a ``$match`` after ``$search``, which sees
    every document mongot returned and drops the wrong ones. Vector uses a ``filter`` inside
    ``$vectorSearch``, which is evaluated by mongot against the indexed filter fields before scoring.
    Only the second is what the shipped default resolves to on a cluster with a registered model, so
    testing one and inferring the other would leave production's mechanism unexercised.

    Both halves are adversarial in the same way: the other scope's document matches the query at least
    as well as this scope's. A corpus where the excluded document would not have been returned anyway
    tests nothing.
    """

    async def test_the_text_tier_never_returns_another_scopes_document(
        self, atlas_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-047a, text tier. Two stores, one collection, the same fact in both, one result each.

        The tier is pinned rather than left at ``auto``, and that is the whole point of this half. On a
        deployment with no registered model ``auto`` degrades to text and this ran the mechanism it
        names; on a cluster that has one it resolves to ``vector``, and the run of 2026-09-02 was
        exercising the pre-filter twice while the class docstring claimed both mechanisms were covered.
        """
        stores = [
            _real_store(
                atlas_uri,
                test_database_name,
                unique_collection_name,
                writable=True,
                scope=scope,
                connection={"search_type": "text"},
            )
            for scope in ("tenant-a", "tenant-b")
        ]
        try:
            await stores[0].add("dark mode for tenant a")
            await stores[1].add("dark mode for tenant b")

            for store, expected in ((stores[0], "dark mode for tenant a"), (stores[1], "dark mode for tenant b")):
                results = await _answered(store, QUERY_TEXT)
                assert [entry.content for entry in results] == [expected]
                assert {_metadata(entry)["_scope"] for entry in results} == {store.scope}
        finally:
            for store in stores:
                await store.close()

    async def test_the_vector_tier_never_returns_another_scopes_document(
        self, atlas_uri: str, atlas_database: Any, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-047a, vector tier, where the predicate is a pre-filter rather than a later ``$match``.

        The vectors are seeded rather than added, for the reason in :func:`_seed_vectors`.
        ``tenant-b``'s vector is *closer* to the query than ``tenant-a``'s, so relevance alone would
        rank it first and a broken pre-filter shows up as the wrong document rather than as an empty
        result.
        """
        collection = atlas_database[unique_collection_name]
        await _seed_vectors(
            collection, scope="tenant-a", store_name="facts", facts={"dark mode for tenant a": [0.9, 0.1, 0.0, 0.0]}
        )
        await _seed_vectors(
            collection, scope="tenant-b", store_name="facts", facts={"dark mode for tenant b": QUERY_VECTOR}
        )
        store = _real_store(
            atlas_uri,
            test_database_name,
            unique_collection_name,
            scope="tenant-a",
            connection={**MANUAL_VECTOR, "search_type": "vector"},
        )
        try:
            results = await _answered(store, {"vector": QUERY_VECTOR, "limit": 10})
            assert [entry.content for entry in results] == ["dark mode for tenant a"]
            assert _metadata(results[0])["_search_type"] == "vector"
        finally:
            await store.close()


@pytest.mark.integration
class TestTheSentinelScopeIsAScopeLikeAnyOther:
    """AT-048b / REQ-E-048 — the read half of the unscoped store's isolation.

    ``AT-048a`` proves the sentinel is *written*. This proves it is also *filtered on*, which is the
    half that matters: a store with no ``scope`` configured is the shape a single-tenant application
    has, and the two failures available to it are opposite. Filtering on nothing returns every tenant's
    documents. Filtering on ``None`` returns nothing, because no document carries ``None``. The sentinel
    exists so that neither is reachable, and only a query can tell them apart.
    """

    async def test_an_unscoped_store_reads_the_sentinel_scope_and_nothing_else(
        self, atlas_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-048b. One unscoped store, one scoped store, one collection, one result each."""
        unscoped = _real_store(atlas_uri, test_database_name, unique_collection_name, writable=True)
        scoped = _real_store(atlas_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-b")
        try:
            await unscoped.add("dark mode with no tenant")
            await scoped.add("dark mode for tenant b")

            results = await _answered(unscoped, QUERY_TEXT)
            assert [entry.content for entry in results] == ["dark mode with no tenant"]
            assert _metadata(results[0])["_scope"] == UNSCOPED
            assert [entry.content for entry in await _answered(scoped, QUERY_TEXT)] == ["dark mode for tenant b"]
        finally:
            await unscoped.close()
            await scoped.close()


@pytest.mark.integration
class TestTheShippedDefaultAgainstADeploymentWithNoModel:
    """AT-053b / REQ-E-053b — the degradation the package's own default runs into, on purpose.

    Not a contrived config. ``embedding_model`` unset means AutoEmbeddings with the default model, and
    an Atlas CLI local deployment registers no models at all, so ``createSearchIndexes`` for the vector
    half fails with code 8 while the full-text half builds normally. That is the configuration every
    developer running this suite locally has, which is what makes the behavior worth pinning: if
    ``initialize()`` raised here, the package would be unusable on the only Atlas-like deployment that
    runs on a laptop, and if it stayed silent the first sign of trouble would be results that are
    lexical when the caller expected semantic.

    So it warns, and the warning carries the server's own message rather than a paraphrase, because the
    list of registered models is the one piece of information that tells the reader what to fix. This
    also skips rather than fails on a deployment that *does* have a model, for the same reason
    :func:`_require_no_search_support` does on the contract tier: the marker cannot express it.
    """

    async def test_initialize_warns_with_the_servers_message_and_does_not_raise(
        self, atlas_store: MongoDBMemoryStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AT-053b, the write-side half: a partial index set is a warning, not a failure."""
        with caplog.at_level(logging.WARNING, logger="strands_mongodb"):
            await atlas_store.initialize()

        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        if not warnings:
            pytest.skip("MONGODB_ATLAS_URI names a deployment with a registered embedding model; nothing is degraded")
        message = " ".join(record.getMessage() for record in warnings)
        assert "vector" in message.lower(), "the warning has to say which half of the index set is missing"
        # The server's phrasing, not ours. Measured on mongot 8.3.8 as `voyage-4-lite is not registered`.
        assert "regist" in message.lower(), f"the server's own reason has to survive into the log: {message!r}"

    async def test_search_answers_from_the_text_tier_when_the_vector_index_is_absent(
        self, atlas_store: MongoDBMemoryStore
    ) -> None:
        """AT-053b, the read-side half. ``auto`` means "whichever tier can answer", and one can."""
        await atlas_store.add("the user prefers dark mode")
        results = await _answered(atlas_store, QUERY_TEXT)

        assert [entry.content for entry in results] == ["the user prefers dark mode"]
        if _metadata(results[0])["_search_type"] == "vector":
            pytest.skip("MONGODB_ATLAS_URI names a deployment with a registered embedding model; nothing is degraded")
        assert _metadata(results[0])["_search_type"] == "text"


@pytest.mark.integration
class TestSearchTypeReportsTheTierThatRan:
    """AT-054 / REQ-E-054 — the key that makes `REQ-E-053b`'s degradation visible instead of silent.

    ``_search_type`` is the only way a caller can tell a semantic answer from a lexical one, and the two
    look identical otherwise: same shape, same score field, same ordering by descending relevance. A
    store that reported the *requested* tier rather than the tier that ran would satisfy every other
    test in this file, which is why the two halves below use configs that resolve differently and assert
    the difference rather than asserting a constant.
    """

    async def test_the_text_tier_reports_text(self, atlas_store: MongoDBMemoryStore) -> None:
        """AT-054, text half. Asked for ``auto`` against a deployment with no model, so text ran."""
        await atlas_store.add("the user prefers dark mode")
        results = await _answered(atlas_store, QUERY_TEXT)

        assert results
        if _metadata(results[0])["_search_type"] == "vector":
            pytest.skip("MONGODB_ATLAS_URI names a deployment with a registered embedding model; nothing is degraded")
        assert _metadata(results[0])["_search_type"] == "text"

    async def test_the_vector_tier_reports_vector(
        self, atlas_uri: str, atlas_database: Any, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-054, vector half, on the manual path with seeded vectors.

        ``search_type`` is left at ``auto`` here rather than pinned to ``vector``, so the assertion is
        about resolution and not about echoing the config back.
        """
        await _seed_vectors(
            atlas_database[unique_collection_name],
            scope="tenant-a",
            store_name="facts",
            facts={"the user prefers dark mode": QUERY_VECTOR},
        )
        store = _real_store(
            atlas_uri, test_database_name, unique_collection_name, scope="tenant-a", connection=MANUAL_VECTOR
        )
        try:
            results = await _answered(store, {"vector": QUERY_VECTOR})
            assert [entry.content for entry in results] == ["the user prefers dark mode"]
            assert _metadata(results[0])["_search_type"] == "vector"
        finally:
            await store.close()


_SCORE_DRIFT: Final[float] = 1e-3
"""How far the same document's score may move between two identical queries (`REQ-E-059`).

Measured on `strands-dev` on 2026-09-02: the test below read 0.8866585493087769 for its top document and
got **0.8867274522781372** back for the same document about a minute later, on the ``vector`` tier. A drift
of 6.9e-5. Nothing in the package touches the number, which arrives in ``$meta`` and is copied into
``_relevance_score`` unchanged; a query given as text is embedded again for the second call, and an
approximate nearest-neighbour search need not traverse the same nodes twice.

1e-3 is about 14 times that drift and 100 times smaller than the 0.115 gap the test asserts between its two
documents before it bounds anything, so the tolerance cannot let one of them pass for the other. The
identity claim is carried by content anyway; this constant only keeps the score assertion from being a
promise Atlas never made.
"""


@pytest.mark.integration
class TestTheScoreBoundDropsResults:
    """AT-059b / REQ-E-059 — the bound applied after the query, and the comparison this test refuses.

    `REQ-E-059` puts the bound in Python rather than in a `$match` on `$meta`, so what it filters is what
    mongot already returned and paid for. The test is calibrated against a first unbounded query instead
    of against a literal, because a fixed threshold is a different claim on each tier: BM25's
    ``searchScore`` is unbounded and depends on corpus statistics, while cosine ``vectorSearchScore`` is
    normalised to ``[0, 1]``. **The two are not comparable, and this test does not compare them.** A
    single number that "works on both" would be a number that happens to sit below every score in this
    particular corpus, which is a fact about the corpus and not about the bound.

    The second comparison it refuses was found the hard way, on 2026-09-02, by making it: the surviving
    result's score is checked to :data:`_SCORE_DRIFT` and not for equality, because the same document on the
    same query text scored 6.9e-5 differently a minute apart. Which document survived is asserted by its
    content, which is the claim the requirement actually makes.
    """

    async def test_a_bound_above_the_lowest_score_drops_exactly_that_result(
        self, atlas_uri: str, test_database_name: str, unique_collection_name: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AT-059b. Measure first, then bound just above the measurement's floor."""
        writer = _real_store(atlas_uri, test_database_name, unique_collection_name, writable=True, scope="tenant-a")
        try:
            # Repeated terms outrank a single mention, which is what puts a measurable gap between the
            # highest and lowest score. Uniform text scores everything within rounding distance.
            await writer.add("dark mode dark mode dark mode")
            await writer.add("dark mode once, then padding " + " ".join(f"word{n}" for n in range(60)))
            unbounded = await _answered(writer, QUERY_TEXT, expected=2)
            assert len(unbounded) == 2, f"the corpus is the premise here: {[e.content for e in unbounded]}"
        finally:
            await writer.close()

        # Sorted as pairs rather than as bare scores: the survivor is identified by its content below, so
        # which of the two texts mongot ranks higher is measured here rather than assumed.
        measured = sorted((_metadata(entry)["_relevance_score"], entry.content) for entry in unbounded)
        (lowest, _), (highest, top_content) = measured
        assert lowest < highest, "without a score gap this test cannot distinguish a bound from a no-op"
        bound = (lowest + highest) / 2

        bounded = _real_store(atlas_uri, test_database_name, unique_collection_name, scope="tenant-a", min_score=bound)
        try:
            with caplog.at_level(logging.DEBUG, logger="strands_mongodb.memory_store"):
                results = await _answered(bounded, QUERY_TEXT)

            assert [entry.content for entry in results] == [top_content], (
                f"the bound sat between the two scores, so exactly the top document survives it: "
                f"{[(_metadata(entry)['_relevance_score'], entry.content) for entry in results]}"
            )
            kept = _metadata(results[0])["_relevance_score"]
            assert kept >= bound, f"a result the bound kept has to clear it: {kept} against {bound}"
            assert kept == pytest.approx(highest, abs=_SCORE_DRIFT), (
                f"the same document on the same query moved further than {_SCORE_DRIFT} between two calls: "
                f"{highest} then {kept}"
            )
            debug = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
            # Both counts, and the key that caused the drop. A message saying only "filtered results"
            # tells the reader something happened without telling them how much, which on a bound set
            # slightly too high is the difference between a tuning problem and an empty memory.
            assert any("min_score" in message and "2" in message and "1" in message for message in debug), (
                f"the effect has to be visible at debug, naming the bound and both counts: {debug!r}"
            )
        finally:
            await bounded.close()
