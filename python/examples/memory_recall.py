"""An agent learns a fact in one process and recalls it in the next, from a query with no words in common.

Run the two steps as separate commands:

    python examples/memory_recall.py write
    python examples/memory_recall.py recall

The write step runs one agent turn. When the turn ends, an extraction trigger fires, an extractor pulls
one fact out of the turn, and the fact is stored in MongoDB. The recall step, in a new process, builds
the same two stores and searches with a query that shares no word with the stored fact:

    stored:  the guest prefers a window table for two on thursday evenings
    query:   where should we seat them

A keyword search finds nothing for that query. A vector search over embeddings that Atlas generates
server-side finds the fact. That is this package's default, and it needs no embedding code, API key or
model of your own.

Both stores live in one collection and are kept apart by scope. The assistant writes to the guest's
notes and reads a curated policy store it cannot write to. Each result shows which store it came from.

Needs MONGODB_URI pointing at an Atlas cluster with Atlas Vector Search and a registered embedding
model. The turn uses a scripted model bundled with the examples, so no model provider is needed; pass
--model bedrock to use a real one. Pass --dry-run to see what a step would do without connecting.
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any, cast

from strands import Agent
from strands.memory import (
    ExtractionConfig,
    ExtractionResult,
    ExtractorContext,
    InvocationTrigger,
    MemoryManager,
    MemoryStore,
)
from strands.types.content import Message

from _support import (
    ExampleError,
    carried_options,
    connection,
    emit_summary,
    model_banner,
    mongodb_config,
    next_command,
    redact,
    resolve_model,
    say,
    step_parser,
)
from strands_mongodb import MongoDBMemoryStore
from strands_mongodb.types import MongoDBMemoryStoreConfig

SCRIPT = "memory_recall.py"
STEPS = ("write", "recall")

#: What the user says. Only the text after the marker is a fact worth storing; the rest is small talk.
MARKER = "Note for next time: "
TURN = f"Table 7 went well tonight. {MARKER}the guest prefers a window table for two on Thursday evenings."

#: Shares no word with the fact. :func:`require_disjoint` checks this at runtime.
QUERY = "where should we seat them?"

WRITABLE_STORE = "guest_notes"
CURATED_STORE = "house_policies"
CURATED_ENTRY = "Window tables are held for parties who booked ahead."

#: One text reply is enough: the turn only gives ``InvocationTrigger`` an invocation to fire after.
REPLIES: list[dict[str, Any]] = [
    {"role": "assistant", "content": [{"text": "Noted. I will keep that in mind for the next booking."}]}
]

_NOT_A_WORD = re.compile(r"[^a-z0-9]+")


def words(text: str) -> set[str]:
    """Return the distinct lowercase words in ``text``."""
    return {word for word in _NOT_A_WORD.split(text.lower()) if word}


def require_disjoint(fact: str, query: str) -> None:
    """Fail unless the query and the fact share no word.

    The example shows that recall is semantic. A shared word would let a keyword match answer the
    query, so an edit to either string that adds one stops the example.

    Args:
        fact: The text that was or will be stored.
        query: The text the recall step searches with.

    Raises:
        ExampleError: If the two share a word.
    """
    overlap = sorted(words(fact) & words(query))
    if overlap:
        raise ExampleError(
            f"the query and the stored fact share {overlap}, so recall would prove nothing beyond a "
            f"keyword match. Edit QUERY or TURN in {SCRIPT} until the two are disjoint."
        )


def fact_from(turn: str) -> str:
    """Return the durable part of a turn.

    Args:
        turn: What the user said.

    Returns:
        Everything after :data:`MARKER`, stripped, or an empty string when the marker is absent.
    """
    _, _, remainder = turn.partition(MARKER)
    return remainder.strip().rstrip(".")


class MarkerExtractor:
    """Turns a conversation into facts by looking for a marker, with no model call.

    An extractor implements ``strands.memory.Extractor``, a single ``extract`` method whose results
    the coordinator passes to the store's ``add``. This one keeps the text after :data:`MARKER` in each
    user message, so the write path costs no model calls and gives the same output on every run.

    In a real application, pass ``extraction=True`` instead. That selects the SDK's ``ModelExtractor``,
    which uses the agent's own model to extract facts, at one model call per turn.
    """

    def __init__(self) -> None:
        """Start with an empty record of what was extracted."""
        self.extracted: list[str] = []

    async def extract(self, messages: list[Message], context: ExtractorContext | None = None) -> list[ExtractionResult]:
        """Return one entry per user message carrying the marker.

        Args:
            messages: The settled turn, tool traffic already filtered out by the coordinator.
            context: Carries the agent's model, which this extractor does not use.

        Returns:
            One :class:`ExtractionResult` per marked fact, tagged so the recall step can print which
            extractor produced what it found.
        """
        results = []
        for message in messages:
            if message.get("role") != "user":
                continue
            for block in message.get("content", []):
                fact = fact_from(block.get("text", "")) if "text" in block else ""
                if fact:
                    self.extracted.append(fact)
                    results.append(ExtractionResult(content=fact, metadata={"extractor": type(self).__name__}))
        return results


def build_stores(arguments: Any, uri: str, extraction: ExtractionConfig | None) -> list[MongoDBMemoryStore]:
    """Build the two stores both steps use: one collection, two scopes.

    The writable store holds what the assistant learns about this guest. The curated store holds house
    policy and has ``writable=False``, so the agent can search it but cannot write to it.

    Args:
        arguments: Parsed command line.
        uri: Connection string.
        extraction: Extraction config for the writable store, or ``None`` in the step that only reads.

    Returns:
        The writable store first, the curated store second.
    """
    config = mongodb_config(uri, arguments.database)
    # Built as a dict because the SDK types `extraction` as `ExtractionConfig | bool`, not optional:
    # the read-only step must omit the key rather than pass `None`.
    writable_config: MongoDBMemoryStoreConfig = {
        "name": WRITABLE_STORE,
        "description": "what the assistant has learned about this guest",
        "config": config,
        "collection": arguments.collection,
        "scope": arguments.scope,
        "writable": True,
    }
    if extraction is not None:
        writable_config["extraction"] = extraction
    writable = MongoDBMemoryStore(**writable_config)
    curated = MongoDBMemoryStore(
        name=CURATED_STORE,
        description="house policy, curated by the restaurant and read-only to the assistant",
        config=config,
        collection=arguments.collection,
        scope=f"{arguments.scope}-house",
        writable=False,
    )
    return [writable, curated]


def managed(stores: list[MongoDBMemoryStore]) -> list[MemoryStore]:
    """Type this package's stores as the ``list[MemoryStore]`` the manager takes.

    Your own code will need the same cast. The SDK documents ``add``, ``add_messages`` and ``get_tools``
    as optional, but its ``Protocol`` declares all of them, so a type checker rejects a store that omits
    one. ``MongoDBMemoryStore`` omits ``add_messages`` on purpose: the manager detects that and feeds its
    extractor's results to ``add``, which deduplicates. The cast is safe because at runtime the manager
    calls only the methods it detected.

    Args:
        stores: The stores to hand over. Not mutated. The caller keeps its own list to call ``close()``,
            which is not part of the protocol.

    Returns:
        The same objects, typed as the manager's parameter.
    """
    return cast("list[MemoryStore]", stores)


async def write(arguments: Any, uri: str) -> dict[str, Any]:
    """Run one turn and let the extraction trigger store what it extracted.

    Nothing here calls ``add`` on the writable store. The store carries an ``ExtractionConfig``, the
    trigger fires on ``AfterInvocationEvent``, and the coordinator passes each extracted entry to ``add``
    in a background task. ``flush()`` waits for that write, so the process does not exit with it still
    in flight.

    Args:
        arguments: Parsed command line.
        uri: Connection string.

    Returns:
        The summary payload.
    """
    extractor = MarkerExtractor()
    stores = build_stores(arguments, uri, ExtractionConfig(trigger=[InvocationTrigger()], extractor=extractor))
    writable, curated = stores

    # The curated store is read-only to the agent, so a writable instance outside the manager stands in
    # for whoever curates it. Seeding it is setup.
    curator = MongoDBMemoryStore(
        name=CURATED_STORE,
        config=mongodb_config(uri, arguments.database),
        collection=arguments.collection,
        scope=f"{arguments.scope}-house",
        writable=True,
    )

    manager = MemoryManager(stores=managed(stores))
    agent = Agent(model=resolve_model(arguments.model, REPLIES), memory_manager=manager)

    try:
        await curator.add(CURATED_ENTRY)
        say(f"curated {CURATED_ENTRY!r} into scope {curator.scope!r} of store {CURATED_STORE!r}")

        say(f"the user says: {TURN!r}")
        await agent.invoke_async(TURN)
        say(f"the turn settled with {len(agent.messages)} messages in context")

        # With `--model bedrock` the reply often says the assistant cannot save anything. That is true:
        # `MemoryManager` builds an `add_memory` tool only with `add_tool_config=True`, which this script
        # does not pass. The extractor does the writing.
        say("the assistant has no tool for writing memories, only search. What follows writes them:")

        await manager.flush()
        say(f"{type(extractor).__name__} produced {len(extractor.extracted)} fact(s), written by extraction")
        for fact in extractor.extracted:
            say(f"  stored in {WRITABLE_STORE!r}, scope {arguments.scope!r}: {fact!r}")
        if not extractor.extracted:
            raise ExampleError(
                "extraction produced nothing, so there is nothing for the recall step to find. The turn "
                f"has to contain {MARKER!r} for this extractor to see a fact in it."
            )
    finally:
        await asyncio.gather(curator.close(), writable.close(), curated.close())

    return {
        "step": "write",
        "stored": [{"content": fact, "store_name": WRITABLE_STORE} for fact in extractor.extracted],
        "curated": [{"content": CURATED_ENTRY, "store_name": CURATED_STORE}],
        "writable_store": WRITABLE_STORE,
    }


async def recall(arguments: Any, uri: str) -> dict[str, Any]:
    """Search from a fresh process, with a query that shares no word with the answer.

    The process builds the same two stores from the same connection string and calls
    ``MemoryManager.search``, which needs no agent and no model.

    Args:
        arguments: Parsed command line.
        uri: Connection string.

    Returns:
        The summary payload.
    """
    stores = build_stores(arguments, uri, None)
    manager = MemoryManager(stores=managed(stores))

    try:
        say(f"searching {len(stores)} store(s) in one collection for {QUERY!r}")
        results = await manager.search(QUERY)
        say(f"{len(results)} hit(s), each attributed to the store that answered")
        hits = []
        for entry in results:
            score = float(entry.metadata.get("_relevance_score", 0.0)) if entry.metadata else 0.0
            scope = entry.metadata.get("_scope") if entry.metadata else None
            say(f"  [{entry.store_name}] score {score:.4f}, scope {scope!r}: {entry.content!r}")
            hits.append({"content": entry.content, "score": score, "store_name": entry.store_name, "scope": scope})
        if not hits:
            # Atlas embeds a new document after the write is acknowledged (about 15 seconds on a test
            # cluster), and a search before then returns nothing rather than failing.
            # `MemoryManager.search` also turns a store that raised into a logged warning and returns the
            # other stores' results, so a failed search looks the same as an empty scope.
            say("nothing came back. Three causes, in the order worth checking:")
            say("  the write step has not run against this database and collection;")
            say("  Atlas has not finished embedding it yet, which takes a few seconds after the write")
            say("    returns, so if the write step just printed its summary, wait and run this again;")
            say("  or a store's search failed. Check for a 'store search failed' warning above: the")
            say("    manager logs those and returns the hits it did get, so a failure looks like this.")
    finally:
        await asyncio.gather(*(store.close() for store in stores))

    return {"step": "recall", "query": QUERY, "hits": hits, "writable_store": WRITABLE_STORE}


def describe(step: str) -> list[str]:
    """Return what a step would do, for ``--dry-run``.

    Args:
        step: The step named on the command line.

    Returns:
        One line per action, in order.
    """
    if step == "write":
        return [
            f"seed {CURATED_STORE!r} with one curated entry, through a writable instance of it",
            f"build {WRITABLE_STORE!r} (writable, extraction on) and {CURATED_STORE!r} (read-only)",
            "run one agent turn, so InvocationTrigger has an invocation to fire after",
            "await MemoryManager.flush(), so the background extraction write completes before exit",
        ]
    return [
        "build the same two stores in this process, which has seen none of the above",
        f"search both for {QUERY!r} and print each hit with its score and store",
    ]


async def run(arguments: Any, parser: Any) -> int:
    """Dispatch one step and print the command for the next.

    Args:
        arguments: Parsed command line.
        parser: The parser, read for its defaults when building the next command.

    Returns:
        A process exit code.
    """
    require_disjoint(fact_from(TURN), QUERY)
    options = carried_options(parser, arguments, ["model", "database", "collection", "scope"])
    following = STEPS[1] if arguments.step == STEPS[0] else None

    if arguments.dry_run:
        say(f"dry run of the {arguments.step!r} step. Nothing connects, nothing is written.")
        say(model_banner(arguments.model))
        for line in describe(arguments.step):
            say(f"  would {line}")
        if following:
            say(f"then run: {next_command(SCRIPT, following, options)}")
        emit_summary({"step": arguments.step, "dry_run": True}, enabled=arguments.summary_json)
        return 0

    uri = connection()
    say(f"connecting to {redact(uri)}, database {arguments.database!r}, collection {arguments.collection!r}")
    say(model_banner(arguments.model))

    summary = await (write(arguments, uri) if arguments.step == "write" else recall(arguments, uri))
    if following:
        say(f"now run, in a new process: {next_command(SCRIPT, following, options)}")
    emit_summary(summary, enabled=arguments.summary_json)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the step, and return an exit code.

    Returning a code instead of raising ``SystemExit`` lets this be called from a REPL.

    Args:
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` for a condition the reader can fix.
    """
    parser = step_parser(__doc__ or "", STEPS)
    parser.add_argument("--scope", default="bistro-demo", help="scope that isolates this tenant's memories")
    arguments = parser.parse_args(argv)

    try:
        return asyncio.run(run(arguments, parser))
    except ExampleError as error:
        print(f"{SCRIPT}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
