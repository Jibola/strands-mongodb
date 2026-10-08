"""Spans the package records for both stores, and what never reaches one.

Every assertion reads what a real ``InMemorySpanExporter`` received behind the global tracer provider,
through conftest's ``exported_spans``, rather than patching the tracer. A value reaches these scans the
same way whatever route put it on the span.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest
from bson import ObjectId
from conftest import rendered
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode
from pymongo import AsyncMongoClient
from pymongo.errors import OperationFailure
from strands.types.exceptions import StorageError
from test_memory_store import _connection_config, _metadata, _patch_setup, _prepared, _real_store, _store
from test_storage import UNREACHABLE, FakeStorageCollection, network_failure, unit_storage

import strands_mongodb
from strands_mongodb import MongoDBStorage, _client, _pipeline, _telemetry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

ACQUIRED: Final = "strands_mongodb.client.acquired"


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    """Isolate each test from the process-global connection registry, as every store test file does."""
    _client._REGISTRY.clear()
    yield
    _client._REGISTRY.clear()


def _named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    """The finished spans called ``name``, in the order they ended."""
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _one(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    """The single finished span called ``name``, failing with every span name if there is not exactly one."""
    spans = _named(exporter, name)
    assert len(spans) == 1, f"expected one {name!r}, got {[span.name for span in exporter.get_finished_spans()]}"
    return spans[0]


class _Probe:
    """The three attributes ``traced`` reads, on an object that is not a store.

    So the decorator's own claims are tested apart from any store, and a failure here points at
    ``_telemetry`` rather than at a call site.
    """

    _database_name = "appdb"
    _collection_name = "things"
    name = "facts"

    @_telemetry.traced("memory", "probe")
    async def succeed(self, secret: str) -> str:
        _telemetry.annotate("strands_mongodb.result_count", 3)
        return secret

    @_telemetry.traced("memory", "probe")
    async def fail(self, secret: str) -> None:
        raise StorageError(f"strands_mongodb: searching {secret} failed")

    @_telemetry.traced("memory", "probe")
    async def hang(self) -> None:
        await asyncio.sleep(10)


class TestTheTracerBelongsToTheApplication:
    """AT-201 / REQ-E-196 — looked up on every call, and nothing configured by the package."""

    async def test_the_tracer_is_looked_up_on_every_call_under_the_package_name(
        self, monkeypatch: pytest.MonkeyPatch, exported_spans: InMemorySpanExporter
    ) -> None:
        """A tracer captured at import would miss a provider set afterwards, which is the usual order."""
        calls: list[tuple[Any, ...]] = []
        real = trace.get_tracer

        def spy(*args: Any, **kwargs: Any) -> trace.Tracer:
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(trace, "get_tracer", spy)
        probe = _Probe()

        await probe.succeed("x")
        await probe.succeed("x")

        assert calls == [("strands_mongodb", strands_mongodb.__version__)] * 2
        assert len(_named(exported_spans, "strands_mongodb.memory.probe")) == 2

    def test_the_package_configures_no_provider_exporter_or_sampler(self) -> None:
        """Read off the source, because configuring one is a single line in any module."""
        package = Path(strands_mongodb.__file__).parent
        source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py")))

        for forbidden in ("set_tracer_provider", "opentelemetry.sdk", "SpanProcessor", "SpanExporter", "Sampler"):
            assert forbidden not in source, forbidden


class TestTheSpanAndItsBaseAttributes:
    """AT-205 / REQ-E-200 — kind, namespace attributes, and no ``db.*`` key, on the decorator alone."""

    async def test_a_span_is_internal_and_carries_the_namespace_and_store(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        assert await _Probe().succeed("x") == "x"

        span = _one(exported_spans, "strands_mongodb.memory.probe")
        assert span.kind is SpanKind.INTERNAL
        assert dict(span.attributes or {}) == {
            "strands_mongodb.database": "appdb",
            "strands_mongodb.collection": "things",
            "strands_mongodb.store_name": "facts",
            "strands_mongodb.result_count": 3,
        }
        assert span.status.status_code is StatusCode.UNSET

    def test_the_span_name_is_readable_off_the_wrapper(self) -> None:
        """What the README test enumerates, without a registry a test-only method would pollute."""
        assert _Probe.succeed.__dict__["strands_mongodb_span"] == "strands_mongodb.memory.probe"


class TestAFailureMarksTheSpanAndCopiesNothing:
    """AT-207 / REQ-E-202 — status and type, never the message, and the raise unchanged."""

    async def test_a_raising_operation_is_an_error_named_by_class_alone(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        with pytest.raises(StorageError) as caught:
            await _Probe().fail("MARKER-KEY")

        span = _one(exported_spans, "strands_mongodb.memory.probe")
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description == "StorageError"
        assert (span.attributes or {})["error.type"] == "strands.types.exceptions.StorageError"
        assert list(span.events) == []
        assert "MARKER-KEY" not in rendered(span)
        assert str(caught.value) == "strands_mongodb: searching MARKER-KEY failed"

    async def test_the_raised_message_is_the_undecorated_one(self) -> None:
        """No provider is the default for most users, and the decorator must not change what they see."""
        with pytest.raises(StorageError) as decorated:
            await _Probe().fail("k")
        with pytest.raises(StorageError) as undecorated:
            await _Probe.fail.__wrapped__(_Probe(), "k")  # type: ignore[attr-defined]

        assert str(decorated.value) == str(undecorated.value)

    def test_a_real_store_with_no_provider_installed_behaves_as_if_undecorated(self) -> None:
        """The default, no provider at all, in a fresh interpreter: this suite cannot test it in-process.

        ``exported_spans`` installs an SDK provider the first time any test asks for it, and OpenTelemetry
        will not take one back. From then on every test in the run, the comparison above included, runs with
        a provider that records. So the default, nothing installed, is checked here instead: the provider is
        OpenTelemetry's proxy, and a store's results and its raised message are what the undecorated method
        gives.
        """
        script = textwrap.dedent(
            """
            import asyncio, sys
            sys.path.insert(0, {tests!r})
            from opentelemetry import trace
            from opentelemetry.trace import ProxyTracerProvider
            from test_storage import FakeStorageCollection, network_failure, unit_storage
            from strands_mongodb import MongoDBStorage

            assert type(trace.get_tracer_provider()) is ProxyTracerProvider, trace.get_tracer_provider()
            undecorated = MongoDBStorage.read.__wrapped__

            async def failure(read, storage):
                try:
                    await read(storage, "k")
                except Exception as error:
                    return type(error).__name__, str(error)

            async def main():
                storage = unit_storage(FakeStorageCollection())
                await storage.write("k", b"v")
                assert [await storage.read("k"), await storage.read("absent")] == [b"v", None]
                assert [await undecorated(storage, "k"), await undecorated(storage, "absent")] == [b"v", None]
                broken = unit_storage(FakeStorageCollection(error=network_failure()))
                decorated, plain = await failure(MongoDBStorage.read, broken), await failure(undecorated, broken)
                assert decorated == plain and decorated[0] == "StorageError", (decorated, plain)

            asyncio.run(main())
            print("ok")
            """
        ).format(tests=str(Path(__file__).parent))

        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

        assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr

    async def test_a_cancelled_operation_ends_its_span_without_an_error(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        """Cancellation is the caller's decision, not a store failure, so the status stays ``UNSET``."""
        task = asyncio.ensure_future(_Probe().hang())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        span = _one(exported_spans, "strands_mongodb.memory.probe")
        assert span.end_time is not None
        assert span.status.status_code is StatusCode.UNSET


class TestTheAllowList:
    """AT-214 / REQ-E-209 — enforced by type, and dropped rather than raised at runtime."""

    async def test_an_unknown_name_is_dropped_and_logged_at_debug(
        self, exported_spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="strands_mongodb._telemetry"):
            with trace.get_tracer("host").start_as_current_span("host"):
                _telemetry.annotate("strands_mongodb.query", "x")  # type: ignore[arg-type]
                _telemetry.event(ACQUIRED, {"strands_mongodb.scope": "x"})  # type: ignore[dict-item]

        host = _one(exported_spans, "host")
        assert "strands_mongodb.query" not in (host.attributes or {})
        assert [event.name for event in host.events] == [ACQUIRED]
        assert dict(host.events[0].attributes or {}) == {}
        dropped = [record for record in caplog.records if record.name == "strands_mongodb._telemetry"]
        assert [record.levelno for record in dropped] == [logging.DEBUG, logging.DEBUG]

    def test_an_unknown_name_fails_the_type_checker(
        self, run_mypy: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        result = run_mypy('from strands_mongodb import _telemetry\n_telemetry.annotate("strands_mongodb.query", 1)\n')

        assert result.returncode != 0, result.stdout
        assert "strands_mongodb.query" in result.stdout


STORAGE_OPERATIONS: Final = ("write", "read", "delete", "list", "search", "delete_prefix", "ensure_indexes")


class TestEveryStorageOperationRecordsOneSpan:
    """AT-202, AT-205 / REQ-E-197, REQ-E-200 — the storage half, against the unit fake."""

    async def test_each_operation_records_one_internal_span_on_its_namespace(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        storage = unit_storage(
            FakeStorageCollection(aggregate_result=[{"_id": "notes/a", "score": 2}]), collection="sessions"
        )

        await storage.write("notes/a", b"x")
        await storage.read("notes/a")
        await storage.list("notes/")
        await storage.search("dark mode")
        await storage.delete("notes/a")
        await storage.delete_prefix("notes/")
        await storage.ensure_indexes()

        spans = [span for span in exported_spans.get_finished_spans() if span.name.startswith("strands_mongodb.")]
        called = ("write", "read", "list", "search", "delete", "delete_prefix", "ensure_indexes")
        assert sorted(called) == sorted(STORAGE_OPERATIONS)
        assert [span.name for span in spans] == [f"strands_mongodb.storage.{op}" for op in called]
        for span in spans:
            attributes = dict(span.attributes or {})
            assert span.kind is SpanKind.INTERNAL
            assert attributes["strands_mongodb.database"] == "appdb"
            assert attributes["strands_mongodb.collection"] == "sessions"
            assert [key for key in attributes if key.startswith("db.")] == []

    async def test_each_outcome_is_a_count_or_a_flag(self, exported_spans: InMemorySpanExporter) -> None:
        fake = FakeStorageCollection(aggregate_result=[{"_id": "notes/a", "score": 2}])
        storage = unit_storage(fake)
        await storage.write("notes/a", b"x")
        await storage.write("notes/b", b"y")

        await storage.read("notes/a")
        await storage.read("notes/missing")
        await storage.list("notes/")
        await storage.search("dark mode")
        await storage.delete("notes/a")
        await storage.delete_prefix("notes/")

        def attribute(name: str, key: str) -> list[Any]:
            return [(span.attributes or {}).get(key) for span in _named(exported_spans, name)]

        assert attribute("strands_mongodb.storage.read", "strands_mongodb.found") == [True, False]
        assert attribute("strands_mongodb.storage.list", "strands_mongodb.result_count") == [2]
        assert attribute("strands_mongodb.storage.search", "strands_mongodb.result_count") == [1]
        assert attribute("strands_mongodb.storage.delete", "strands_mongodb.deleted_count") == [1]
        assert attribute("strands_mongodb.storage.delete_prefix", "strands_mongodb.deleted_count") == [1]

    async def test_a_driver_failure_is_recorded_as_the_error_the_caller_sees(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        """The span is outside ``_reporting``, so it records ``StorageError`` and not the driver's type."""
        storage = unit_storage(FakeStorageCollection(error=network_failure()))

        with pytest.raises(StorageError):
            await storage.read("notes/a")

        span = _one(exported_spans, "strands_mongodb.storage.read")
        assert span.status.description == "StorageError"
        assert "notes/a" not in rendered(span)


class TestEveryMemoryOperationRecordsOneSpan:
    """AT-202, AT-205, AT-206 / REQ-E-197, REQ-E-200, REQ-E-201 — the memory half, against the unit fakes."""

    async def test_initialize_records_one_span_and_the_tiers_it_prepared(
        self, monkeypatch: pytest.MonkeyPatch, exported_spans: InMemorySpanExporter
    ) -> None:
        """An ``auto`` store whose vector index will not build: ``initialize`` says so, once."""
        _patch_setup(monkeypatch, failing=frozenset({"vector_index"}))
        store = _store()

        await store.initialize()
        await store.initialize()

        span = _one(exported_spans, "strands_mongodb.memory.initialize")
        attributes = dict(span.attributes or {})
        assert attributes["strands_mongodb.search.requested"] == "auto"
        assert attributes["strands_mongodb.search.tiers"] == "text"
        assert attributes["strands_mongodb.store_name"] == "facts"
        assert attributes["strands_mongodb.database"] == "appdb"
        assert attributes["strands_mongodb.collection"] == store._collection_name

    async def test_the_fast_path_records_nothing(self, exported_spans: InMemorySpanExporter) -> None:
        """``search`` and ``add`` call ``initialize()`` every time; only the first may cost a span."""
        store = _store()
        _prepared(store)

        await store.initialize()
        await store.search("dark mode")

        assert _named(exported_spans, "strands_mongodb.memory.initialize") == []
        assert len(_named(exported_spans, "strands_mongodb.memory.search")) == 1

    async def test_search_reports_the_tier_that_answered_and_how_many(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        store = _store()
        _prepared(store, documents=[{"_id": "abc", "content": "dark mode", _pipeline.SCORE_FIELD: 1.0}])

        results = await store.search("dark mode")

        attributes = dict(_one(exported_spans, "strands_mongodb.memory.search").attributes or {})
        assert attributes["strands_mongodb.search.requested"] == "auto"
        assert attributes["strands_mongodb.search.tier"] == "text"
        assert attributes["strands_mongodb.search.tier"] == _metadata(results[0])["_search_type"]
        assert attributes["strands_mongodb.result_count"] == len(results) == 1

    async def test_both_deletes_report_the_drivers_count(self, exported_spans: InMemorySpanExporter) -> None:
        """Seven, which no real deletion of one id could produce, so a constant ``1`` would fail."""
        store = _store(writable=True)
        collection = _prepared(store)
        collection.deleted = 7

        await store.delete(str(ObjectId()))
        await store.delete_all()

        for name in ("strands_mongodb.memory.delete", "strands_mongodb.memory.delete_all"):
            assert (_one(exported_spans, name).attributes or {})["strands_mongodb.deleted_count"] == 7

    async def test_a_failed_add_is_an_error_span_that_carries_no_content(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        store = _store(writable=True)
        _prepared(store, failure=OperationFailure("refused", code=8))

        with pytest.raises(StorageError):
            await store.add("MARKER-CONTENT")

        span = _one(exported_spans, "strands_mongodb.memory.add")
        assert span.status.description == "StorageError"
        assert "MARKER-CONTENT" not in rendered(span)

    async def test_a_setup_failure_reached_through_search_marks_both_spans(
        self, monkeypatch: pytest.MonkeyPatch, exported_spans: InMemorySpanExporter
    ) -> None:
        """Every tier refused: ``initialize`` raises inside ``search``, and both spans say so."""
        _patch_setup(monkeypatch, failing=frozenset({"vector_index", "text_index"}))
        store = _store()

        with pytest.raises(StorageError):
            await store.search("dark mode")

        search = _one(exported_spans, "strands_mongodb.memory.search")
        initialize = _one(exported_spans, "strands_mongodb.memory.initialize")
        assert initialize.parent is not None
        assert search.context is not None
        assert initialize.parent.span_id == search.context.span_id
        assert search.status.status_code is initialize.status.status_code is StatusCode.ERROR
        assert search.status.description == initialize.status.description == "StorageError"


def _acquired(span: ReadableSpan) -> list[bool]:
    """The ``client.injected`` flag of every ``client.acquired`` event on ``span``, in order."""
    return [bool((e.attributes or {})["strands_mongodb.client.injected"]) for e in span.events if e.name == ACQUIRED]


class TestANewPoolIsAnEventOnTheSpanThatCausedIt:
    """AT-203 / REQ-E-198 — one event per registry entry created, none per cache hit or adoption."""

    def test_a_store_built_inside_a_traced_block_marks_that_block(self, exported_spans: InMemorySpanExporter) -> None:
        tracer = trace.get_tracer("the-application")
        with tracer.start_as_current_span("host"):
            MongoDBStorage({"database": "appdb", "connection_string": UNREACHABLE})
            MongoDBStorage({"database": "appdb", "connection_string": UNREACHABLE})

        assert _acquired(_one(exported_spans, "host")) == [False]

    async def test_an_injected_client_says_so(self, exported_spans: InMemorySpanExporter) -> None:
        client: AsyncMongoClient[Any] = AsyncMongoClient(UNREACHABLE)
        try:
            with trace.get_tracer("the-application").start_as_current_span("host"):
                MongoDBStorage({"database": "appdb", "client": client})
        finally:
            await client.close()

        assert _acquired(_one(exported_spans, "host")) == [True]

    def test_a_store_outliving_its_loop_marks_the_operation_that_needed_a_new_pool(
        self, exported_spans: InMemorySpanExporter
    ) -> None:
        """A synchronous ``Agent``: built at import time, then driven on a fresh loop per call.

        Construction records nowhere, because there is no span to record on. The first call adopts the
        pool that construction built, which acquires nothing. The second call's loop needs a pool of its
        own, and that one is an event on the ``read`` span that waited for it.
        """
        fake = FakeStorageCollection()
        storage = unit_storage(fake)
        # `unit_storage` shadows `_collection` so nothing touches the network. Touching `_connection` first
        # puts back the one step that shadowing skipped, which is the per-loop resolve this test is about.
        storage._collection = lambda: (storage._connection, fake)[1]  # type: ignore[assignment, return-value]

        asyncio.run(storage.read("k"))
        asyncio.run(storage.read("k"))

        reads = _named(exported_spans, "strands_mongodb.storage.read")
        assert [_acquired(span) for span in reads] == [[], [False]]


#: One distinct marker per value REQ-E-203 forbids, each one unlikely to occur anywhere by chance.
KEY: Final = "mk-key-7f3a"
PREFIX: Final = "mk-prefix-1c9e"
PAYLOAD: Final = "mk-payload-55d0"
CONTENT: Final = "mk-content-41ab"
METADATA_KEY: Final = "mk_metakey_3e8f"
METADATA_VALUE: Final = "mk-metavalue-90cd"
QUERY: Final = "mk-query-6b2e"
FILTER_VALUE: Final = "mk-filter-d71a"
SCOPE: Final = "mk-scope-28f4"
MEMORY_ID: Final = "feedfacecafe0123456789ab"
USERNAME: Final = "mkuser4c1d"
PASSWORD: Final = "mkpass9e7b"
CONNECTION_STRING: Final = f"mongodb://{USERNAME}:{PASSWORD}@127.0.0.1:1/?serverSelectionTimeoutMS=50"
MARKERS: Final = (
    KEY,
    PREFIX,
    PAYLOAD,
    CONTENT,
    METADATA_KEY,
    METADATA_VALUE,
    QUERY,
    FILTER_VALUE,
    SCOPE,
    MEMORY_ID,
    USERNAME,
    PASSWORD,
    CONNECTION_STRING,
)

STORAGE_SPANS: Final = {f"strands_mongodb.storage.{op}" for op in STORAGE_OPERATIONS}
MEMORY_SPANS: Final = {f"strands_mongodb.memory.{op}" for op in ("initialize", "add", "search", "delete", "delete_all")}


def _leaks(exporter: InMemorySpanExporter, records: list[logging.LogRecord]) -> list[str]:
    """Every ``(surface, marker)`` pair where a forbidden value appeared, so a failure names them all."""
    surfaces = [(f"span {span.name}", rendered(span)) for span in exporter.get_finished_spans()]
    surfaces += [
        (f"log {record.name}", f"{record.getMessage()} {vars(record)!r}")
        for record in records
        if record.name.startswith("strands_mongodb")
    ]
    return [f"{where}: {marker}" for where, text in surfaces for marker in MARKERS if marker in text]


class TestNothingTheCallerSuppliedReachesASpanOrALog:
    """AT-208 / REQ-E-203 — a marker in every forbidden value, every operation driven, every surface read.

    The unit tier, against the fakes, so the credentials can be real-looking and the run needs nothing.
    The contract class below repeats it against a deployment for the values a fake cannot carry back.
    """

    async def test_every_operation_of_both_stores(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        exported_spans: InMemorySpanExporter,
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="strands_mongodb")

        fake = FakeStorageCollection(aggregate_result=[{"_id": f"{PREFIX}/{KEY}", "score": 2}])
        storage = unit_storage(fake, prefix=PREFIX, config={"connection_string": CONNECTION_STRING})
        await storage.write(KEY, PAYLOAD.encode())
        await storage.read(KEY)
        await storage.list(KEY)
        await storage.search(QUERY)
        await storage.delete(KEY)
        await storage.delete_prefix(KEY)
        await storage.ensure_indexes()

        _patch_setup(monkeypatch, failing=frozenset({"vector_index"}))
        store = _store(
            writable=True,
            scope=SCOPE,
            filter={"metadata.region": FILTER_VALUE},
            config=_connection_config(connection_string=CONNECTION_STRING, filter_fields=["metadata.region"]),
        )
        await store.initialize()
        collection = _prepared(
            store,
            documents=[
                {
                    "_id": ObjectId(MEMORY_ID),
                    "content": CONTENT,
                    "metadata": {METADATA_KEY: METADATA_VALUE},
                    _pipeline.SCORE_FIELD: 1.0,
                }
            ],
        )
        await store.search({"text": QUERY, "filter": {"metadata.region": FILTER_VALUE}})
        collection.failure = OperationFailure(f"refused {CONTENT}", code=8)
        with pytest.raises(StorageError):
            await store.add(CONTENT, {METADATA_KEY: METADATA_VALUE})
        collection.failure = None
        await store.delete(MEMORY_ID)
        await store.delete_all()

        names = {span.name for span in exported_spans.get_finished_spans()}
        assert names >= STORAGE_SPANS | MEMORY_SPANS, sorted((STORAGE_SPANS | MEMORY_SPANS) - names)
        assert _leaks(exported_spans, caplog.records) == []


@pytest.mark.contract
class TestTheSpansAgainstARealDeployment:
    """AT-202, AT-206, AT-208 — the outcomes a fake writes for itself, read off a real mongod instead.

    A plain deployment, so no search: ``search`` is the integration tier's, and its tier attribute is
    asserted against the unit fakes above. What this class owns is the driver's own answers. A second
    ``add`` of the same fact is absorbed by the unique index, which only a server enforces. A delete of a
    key that was never written reports zero, which ``FakeStorageCollection.delete_one`` cannot.
    """

    async def test_both_stores_against_mongod(
        self,
        caplog: pytest.LogCaptureFixture,
        exported_spans: InMemorySpanExporter,
        contract_uri: str,
        test_database_name: str,
        unique_collection_name: str,
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="strands_mongodb")
        storage = MongoDBStorage(
            {"connection_string": contract_uri, "database": test_database_name},
            collection=unique_collection_name,
            prefix=PREFIX,
        )
        store = _real_store(
            contract_uri, test_database_name, f"{unique_collection_name}_memory", writable=True, scope=SCOPE
        )
        try:
            await storage.write(KEY, PAYLOAD.encode())
            await storage.read(KEY)
            await storage.delete(f"{KEY}-never-written")

            await store.add(CONTENT, {METADATA_KEY: METADATA_VALUE})
            await store.add(CONTENT, {METADATA_KEY: METADATA_VALUE})
            await store.delete(MEMORY_ID)
            await store.delete_all()
        finally:
            await storage.close()
            await store.close()

        assert (_one(exported_spans, "strands_mongodb.storage.read").attributes or {})["strands_mongodb.found"] is True
        missing = _one(exported_spans, "strands_mongodb.storage.delete")
        assert (missing.attributes or {})["strands_mongodb.deleted_count"] == 0
        adds = _named(exported_spans, "strands_mongodb.memory.add")
        assert [(span.attributes or {})["strands_mongodb.duplicate"] for span in adds] == [False, True]
        assert (_one(exported_spans, "strands_mongodb.memory.delete").attributes or {})[
            "strands_mongodb.deleted_count"
        ] == 0
        assert (_one(exported_spans, "strands_mongodb.memory.delete_all").attributes or {})[
            "strands_mongodb.deleted_count"
        ] == 1
        assert _leaks(exported_spans, caplog.records) == []
