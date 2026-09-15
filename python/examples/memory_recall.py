"""An agent learns a fact in one process and recalls it in the next, from a query with no words in common.

Two commands, two processes:

    python examples/memory_recall.py write
    python examples/memory_recall.py recall

The first runs one agent turn. An extraction trigger fires when the turn settles, an extractor turns the
turn into one fact, and the fact goes into MongoDB. The process then exits, taking every Python object
and the connection pool with it. The second process starts cold, builds the same two stores, and asks a
question that shares no word with the stored fact:

    stored:  the guest prefers a window table for two on thursday evenings
    query:   where should we seat them

A keyword scan answers that with nothing. What answers it is a vector search over embeddings the Atlas
cluster generated server-side, which is this package's default and needs no embedding code, no API key
and no model of your own.

Two stores share the collection and differ by ``scope``: the assistant writes to the guest's own notes
and reads a curated policy store it is not allowed to write to. Every result prints the store that
produced it.

Requires ``MONGODB_URI`` and nothing else. The turn runs against a scripted model bundled with these
examples so a reader with no Bedrock access can still run all of it; pass ``--model bedrock`` for a real
one. Pass ``--dry-run`` to see what a step would do before pointing it at your own cluster.
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

#: What the user says. Everything after the marker is the durable fact; the rest is small talk that
#: should not be remembered, which is the whole reason an extractor exists.
MARKER = "Note for next time: "
TURN = f"Table 7 went well tonight. {MARKER}the guest prefers a window table for two on Thursday evenings."

#: Deliberately disjoint from the fact, word for word. Checked at runtime by :func:`require_disjoint`.
QUERY = "where should we seat them?"

WRITABLE_STORE = "guest_notes"
CURATED_STORE = "house_policies"
CURATED_ENTRY = "Window tables are held for parties who booked ahead."

#: One text reply is enough. The turn exists so ``InvocationTrigger`` has an invocation to fire after.
REPLIES: list[dict[str, Any]] = [
    {"role": "assistant", "content": [{"text": "Noted. I will keep that in mind for the next booking."}]}
]

_NOT_A_WORD = re.compile(r"[^a-z0-9]+")


def words(text: str) -> set[str]:
    """Return the distinct lowercase words in ``text``."""
    return {word for word in _NOT_A_WORD.split(text.lower()) if word}


def require_disjoint(fact: str, query: str) -> None:
    """Fail unless the query and the fact share no word.

    The claim this example makes is that recall is semantic. A query sharing even one content word with
    the stored fact could be answered by a keyword scan, so the demonstration would prove nothing, and
    the way that happens is by someone editing one of the two strings. Checking here means the example
    stops rather than quietly weakening.

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

    An ``Extractor`` is one method (``strands.memory.Extractor``), and the coordinator hands whatever it
    returns straight to the store's ``add``. So the cheapest honest extractor is a rule: keep the part of
    a user turn that follows a marker, drop the rest. That keeps this example's cost at zero model calls
    for the write path and makes its output identical on every run.

    A real deployment usually wants ``extraction=True`` instead, which selects the SDK's
    ``ModelExtractor`` and distills facts with the agent's own model, at one model call per turn.
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
    """Build the two stores both steps use.

    One connection config, one collection, two scopes. The writable store holds what the assistant
    learns about this guest; the curated store holds house policy and is read-only to the agent, which
    is what ``writable=False`` buys: the agent can be told about it and cannot overwrite it.

    Args:
        arguments: Parsed command line.
        uri: Connection string.
        extraction: Extraction config for the writable store, or ``None`` in the step that only reads.

    Returns:
        The writable store first, the curated store second.
    """
    config = mongodb_config(uri, arguments.database)
    # Assembled as a `MongoDBMemoryStoreConfig` rather than passed inline, because `extraction` is
    # `ExtractionConfig | bool` in the SDK's own TypedDict and not optional: the step that only reads has
    # to omit the key, not pass `None` for it.
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
    """Present a list of this package's stores as the list of protocol objects the manager takes.

    A cast, and it is worth explaining rather than hiding, because a reader writing
    ``list[MemoryStore]`` in their own code will meet it too.

    The SDK's ``MemoryStore`` docstring says a store implements ``search``, "plus optionally ``add``,
    ``add_messages``, and ``get_tools``". The ``Protocol`` declares all four as members, so a type checker
    rejects any store that omits one of the three optional ones. ``MongoDBMemoryStore`` omits
    ``add_messages`` deliberately: the manager feature-detects it with ``_has_method`` and falls back to
    its own extractor feeding ``add``, which dedupes, and extraction writes are at-least-once. A stub that
    raised would defeat that detection and turn "the SDK picks another path" into "the SDK crashes".

    So the docstring and the type disagree, in the SDK. At runtime the manager calls
    only what it detected, which is why this cast is safe rather than merely quiet.

    Args:
        stores: The stores to hand over. Not mutated, and the caller keeps its own concrete list for
            ``close()``, which is this package's method and not part of the protocol.

    Returns:
        The same objects, typed as the manager's parameter.
    """
    return cast("list[MemoryStore]", stores)


async def write(arguments: Any, uri: str) -> dict[str, Any]:
    """Run one turn and let the extraction trigger persist what it produced.

    Nothing here calls ``add`` on the writable store. The write happens because the store carries an
    ``ExtractionConfig``, the trigger fires on ``AfterInvocationEvent``, and the coordinator passes each
    extracted entry to ``add`` in a background task. ``flush()`` is what makes that safe to exit after:
    without it the process can end while the write is still in flight, and the example fails
    intermittently, which is the worst way for an example to be wrong.

    Args:
        arguments: Parsed command line.
        uri: Connection string.

    Returns:
        The summary payload.
    """
    extractor = MarkerExtractor()
    stores = build_stores(arguments, uri, ExtractionConfig(trigger=[InvocationTrigger()], extractor=extractor))
    writable, curated = stores

    # The curated store is read-only to the agent, so a second instance of it, writable and outside the
    # manager, stands in for whoever curates it. Seeding it is setup, not the path this example is about.
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

        # Said before the extractor runs, because with `--model bedrock` the reply above tends to say
        # the assistant cannot save anything, and it is right: `MemoryManager` only builds an
        # `add_memory` tool when constructed with `add_tool_config=True`, which this script does not
        # pass. The write comes from the extractor instead, and without this line the next two lines
        # read as contradicting the model.
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
    """Search from a cold start, with a query that shares no word with the answer.

    This process has never seen the fact, the agent that learned it, or the objects it was written
    through. It builds the same two stores from the same connection string and asks
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
            # Three causes, and this list is in the order a reader should check them. The lag one
            # (`REQ-E-091`) is likeliest if the write step just printed its summary: Atlas embeds a new
            # document after the write acknowledges, about 15 seconds on the cluster this was developed
            # against, and a search before that finishes returns nothing rather than failing.
            #
            # The third is the one worth reading the whole list for. `MemoryManager.search` gathers with
            # `return_exceptions=True` and turns a store that raised into a `logging.warning` plus a
            # `store_failure_count` on its span, then returns the results it did get. So a search that
            # failed outright and a scope that is genuinely empty are the same empty list here, and the
            # only difference is a log line above this one. Seen in practice: the Atlas
            # search process was briefly unreachable and this printed zero hits with no error.
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

    Returning a code rather than raising ``SystemExit`` is what lets the acceptance tests read a status
    from a subprocess and lets a reader import this file in a REPL without it killing the session.

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
