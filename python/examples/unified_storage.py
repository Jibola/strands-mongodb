"""Three SDK subsystems, one MongoDB collection, and a session that survives the process that made it.

Run the two steps as separate commands:

    python examples/unified_storage.py run
    python examples/unified_storage.py resume

Agent(storage=...) takes one backend and hands it to every subsystem that does not bring its own, so one
MongoDBStorage serves three writers, each under its own key prefix:

    session/    conversation snapshots, from SnapshotSessionManager
    context/    the turn's messages, stashed by the context manager
    memory/     a knowledge file, from FileMemoryStore(storage=...)

The run step runs a turn that writes all three, then prints a census of the collection read back from
MongoDB: each top-level prefix with its document count and byte total. The resume step, in a new
process, restores the session from session/ and prints how many messages came back.

The memory/ documents are not vector memory. Storage holds opaque bytes under a string key, and on this
backend FileMemoryStore recalls by token overlap over the keys alone. MongoDBMemoryStore, shown in
memory_recall.py, uses a separate collection with typed fields and a vector index.

Needs MONGODB_URI. Any MongoDB deployment works; Atlas Search is not needed. Pass --dry-run to see what
a step would do without connecting.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from pymongo import AsyncMongoClient
from strands import Agent, tool
from strands.session import SnapshotSessionManager
from strands.vended_memory_stores.file_memory_store import FileMemoryStore

from _support import (
    ExampleError,
    carried_options,
    connection,
    emit_summary,
    model_banner,
    mongodb_config,
    next_command,
    prefix_census,
    redact,
    resolve_model,
    say,
    step_parser,
)
from strands_mongodb import MongoDBStorage

SCRIPT = "unified_storage.py"
STEPS = ("run", "resume")

#: Enough rows that the tool result is over ``context_manager="auto"``'s offload threshold.
BOOKINGS = 200

MEMORY_STORE = "kitchen_notes"
KNOWLEDGE = (
    "Closing checklist: wipe the pass, log the waste sheet, and leave the walk-in door propped open "
    "for ten minutes before setting the alarm."
)

#: The first reply calls the tool, giving the context manager an oversized result to move. The second
#: ends the turn.
REPLIES: list[dict[str, Any]] = [
    {
        "role": "assistant",
        "content": [
            {"toolUse": {"name": "reservation_book", "toolUseId": "tooluse-1", "input": {"service": "dinner"}}}
        ],
    },
    {"role": "assistant", "content": [{"text": "Dinner is fully committed. I have filed the closing notes."}]},
]

PROMPT = "Pull up tonight's reservation book and file the closing checklist."


@tool
def reservation_book(service: str) -> str:
    """Return every booking for one service, which is more text than a turn should carry.

    The :data:`BOOKINGS` lines are long enough that the context manager moves the result out of context.

    Args:
        service: Which service to list. Only used in the heading.

    Returns:
        One line per booking.
    """
    rows = [
        f"table {number:>2} | {19 + number % 4}:{(number * 15) % 60:02d} | party of {2 + number % 6} | "
        f"guest {number:03d} | {'window' if number % 3 == 0 else 'main room'}"
        for number in range(1, BOOKINGS + 1)
    ]
    return f"{service} service, {len(rows)} bookings\n" + "\n".join(rows)


def build_storage(arguments: Any, uri: str) -> MongoDBStorage:
    """Build the one storage instance every subsystem in this example shares.

    No ``prefix`` is set, so the census shows the SDK's own keys. A real application usually sets one
    per tenant.

    Args:
        arguments: Parsed command line.
        uri: Connection string.

    Returns:
        A ``MongoDBStorage`` on the requested database and collection.
    """
    return MongoDBStorage(mongodb_config(uri, arguments.database), collection=arguments.collection)


async def census_lines(arguments: Any, uri: str) -> list[dict[str, Any]]:
    """Read the prefix census with a separate driver client, and print each line.

    A plain client shows that these are ordinary documents that any MongoDB tool, ``mongosh`` included,
    can read.

    Args:
        arguments: Parsed command line.
        uri: Connection string.

    Returns:
        The census, one entry per top-level prefix.
    """
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(uri)
    try:
        census = await prefix_census(client[arguments.database][arguments.collection])
    finally:
        await client.close()

    say(f"census of {arguments.database}.{arguments.collection}, read back with a plain driver:")
    if not census:
        say("  the collection is empty. Run the 'run' step against this database and collection first.")
    for entry in census:
        say(f"  {entry['prefix']:<12} {entry['documents']:>3} document(s)  {entry['bytes']:>7} bytes")
    return census


def build_agent(arguments: Any, storage: MongoDBStorage) -> tuple[Agent, SnapshotSessionManager]:
    """Assemble the agent and the session manager that shares its storage.

    Neither the session manager nor the context manager's stash gets a backend of its own. Both use
    ``agent.storage``: one connection pool, one collection, and the subsystems kept apart by key prefix.

    Args:
        arguments: Parsed command line.
        storage: The shared backend.

    Returns:
        The agent and the session manager, which ``run`` uses again to save a snapshot.
    """
    sessions = SnapshotSessionManager(session_id=arguments.session_id)
    agent = Agent(
        model=resolve_model(arguments.model, REPLIES),
        tools=[reservation_book],
        context_manager="auto",
        session_manager=sessions,
        storage=storage,
    )
    return agent, sessions


async def run(arguments: Any, uri: str) -> dict[str, Any]:
    """Produce all three prefixes in one turn, then report what MongoDB holds.

    Args:
        arguments: Parsed command line.
        uri: Connection string.

    Returns:
        The summary payload.
    """
    storage = build_storage(arguments, uri)
    # `FileMemoryStore` subclasses the SDK's `MemoryStore` Protocol without implementing `add_messages`,
    # `get_tools` or `initialize`, so mypy calls it abstract. At runtime it works, because the SDK
    # detects those methods rather than requiring them. Any type-checked caller needs this ignore.
    knowledge = FileMemoryStore(name=MEMORY_STORE, storage=storage)  # type: ignore[abstract]

    try:
        # Constructing the Agent restores any existing session from a synchronous hook, which the SDK runs
        # on a new event loop in a worker thread. `MongoDBStorage` keeps one client per event loop for this.
        agent, sessions = build_agent(arguments, storage)
        say(f"session {arguments.session_id!r} opened with {len(agent.messages)} message(s) in context")

        say(f"the user says: {PROMPT!r}")
        await agent.invoke_async(PROMPT)
        say(f"the turn settled with {len(agent.messages)} message(s) in context")

        entry_id = await knowledge.add(KNOWLEDGE)
        say(f"FileMemoryStore({MEMORY_STORE!r}) wrote {entry_id!r} through the same backend")

        snapshot_id = await sessions.save_snapshot(agent, is_latest=False)
        say(f"snapshot_latest was written when the turn ended; immutable checkpoint {snapshot_id!r} added")

        census = await census_lines(arguments, uri)

        prefixes = [entry["prefix"] for entry in census]
        missing = [expected for expected in ("session/", "context/", "memory/") if expected not in prefixes]
        if missing:
            raise ExampleError(
                f"nothing was written under {missing}, so this run does not show what it claims. The "
                f"census found {prefixes or 'nothing'}."
            )
        say(f"three subsystems, one collection: {', '.join(prefixes)}")

        return {
            "step": "run",
            "session_id": arguments.session_id,
            "messages": len(agent.messages),
            "snapshot_id": snapshot_id,
            "memory_entry_id": entry_id,
            "census": census,
        }
    finally:
        await storage.close()


async def resume(arguments: Any, uri: str) -> dict[str, Any]:
    """Restore the session in a fresh process that has run no turn.

    Args:
        arguments: Parsed command line.
        uri: Connection string.

    Returns:
        The summary payload.
    """
    storage = build_storage(arguments, uri)

    try:
        agent, _ = build_agent(arguments, storage)
        restored = len(agent.messages)
        say(f"restored {restored} message(s) from session {arguments.session_id!r}, before any turn ran")
        if restored:
            first = agent.messages[0]
            blocks = [block for block in first.get("content", []) if "text" in block]
            if blocks:
                say(f"  the first restored message ({first.get('role')}): {blocks[0]['text'][:90]!r}")
        else:
            say("  nothing came back. Run the 'run' step first, with this same --session-id.")

        census = await census_lines(arguments, uri)
        say(
            "on the memory/ prefix: Storage moves opaque bytes under a string key, and FileMemoryStore "
            "recall is token overlap over that key space, so it matches file names and "
            "not the text inside them. MongoDBMemoryStore is a different collection with typed fields "
            "and a vector index. See memory_recall.py for that one."
        )

        return {
            "step": "resume",
            "session_id": arguments.session_id,
            "restored_messages": restored,
            "census": census,
        }
    finally:
        await storage.close()


def describe(step: str) -> list[str]:
    """Return what a step would do, for ``--dry-run``.

    Args:
        step: The step named on the command line.

    Returns:
        One line per action, in order.
    """
    if step == "run":
        return [
            "build one MongoDBStorage and pass it as Agent(storage=...)",
            'add a SnapshotSessionManager and context_manager="auto", neither carrying storage of its own',
            f"run one turn in which the model calls {reservation_book.tool_name!r}, whose result is far "
            "over the offload threshold; the context manager stashes the turn under context/",
            f"write one knowledge file through FileMemoryStore({MEMORY_STORE!r}, storage=...)",
            "save an immutable snapshot, then read the prefix census back with a plain driver",
        ]
    return [
        "rebuild the same storage and the same session manager in this process",
        "let Agent construction restore snapshot_latest, and print how many messages came back",
        "read the census again, and say what the memory/ prefix is and is not",
    ]


async def dispatch(arguments: Any, parser: Any) -> int:
    """Run one step and print the command for the next.

    Args:
        arguments: Parsed command line.
        parser: The parser, read for its defaults when building the next command.

    Returns:
        A process exit code.
    """
    options = carried_options(parser, arguments, ["model", "database", "collection", "session_id"])
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

    summary = await (run(arguments, uri) if arguments.step == "run" else resume(arguments, uri))
    if following:
        say(f"now run, in a new process: {next_command(SCRIPT, following, options)}")
    emit_summary(summary, enabled=arguments.summary_json)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the step, and return an exit code.

    Args:
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` for a condition the reader can fix.
    """
    parser = step_parser(__doc__ or "", STEPS)
    parser.add_argument("--session-id", default="dinner-service", help="session whose snapshots both steps share")
    arguments = parser.parse_args(argv)

    try:
        return asyncio.run(dispatch(arguments, parser))
    except ExampleError as error:
        print(f"{SCRIPT}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
