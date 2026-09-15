"""What both examples need and neither should own.

Not part of the installed package. It lives here so the two scripts can share a scripted model, a
connection lookup and a key-space census without either one growing a second job, and so a reader can
see every piece of scaffolding in one file instead of hunting it through both examples.

Nothing here is API. The names are stable only for the two scripts beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from collections.abc import AsyncIterator, Sequence
from itertools import count
from typing import Any

from strands.models import Model
from strands.types.streaming import StopReason, StreamEvent

from strands_mongodb.types import MongoDBConfig

__all__ = [
    "URI_VARIABLE",
    "ExampleError",
    "ScriptedModel",
    "carried_options",
    "connection",
    "emit_summary",
    "model_banner",
    "mongodb_config",
    "next_command",
    "prefix_census",
    "redact",
    "resolve_model",
    "say",
    "step_parser",
]

#: The one variable either example requires. Named in the failure message rather than described, so a
#: reader can copy it out of the error into an export.
URI_VARIABLE = "MONGODB_URI"

#: Everything between the scheme and the host, which is where a password sits.
_CREDENTIALS = re.compile(r"://[^@/]+@")

_line = count(1)


class ExampleError(RuntimeError):
    """A condition the reader can fix, reported without a traceback.

    Anything else is left to propagate. A stack trace is the right output for a bug in the example,
    and the wrong output for a missing environment variable.
    """


def say(message: str) -> None:
    """Print one numbered line of prose.

    Numbering, rather than plain prints, because both examples are read as a transcript after the
    fact: a numbered line can be pointed at, and a step that prints nothing between line 4 and line 5
    is visible.

    Args:
        message: One line, already worded for a reader.
    """
    print(f"{next(_line):>2}. {message}", flush=True)


def redact(uri: str) -> str:
    """Return ``uri`` with any credentials replaced.

    Args:
        uri: A MongoDB connection string.

    Returns:
        The same string with everything between ``://`` and ``@`` replaced by ``<redacted>``. A URI
        with no credentials comes back unchanged.
    """
    return _CREDENTIALS.sub("://<redacted>@", uri)


def connection() -> str:
    """Read the connection string, or explain what to set.

    Returns:
        The value of ``MONGODB_URI``.

    Raises:
        ExampleError: If the variable is unset or empty. The message names the variable and shows the
            shape of the value without ever printing a real one.
    """
    uri = os.environ.get(URI_VARIABLE, "")
    if not uri:
        raise ExampleError(
            f"{URI_VARIABLE} is not set. Point it at a deployment this example may write to, for "
            f"example with a shell that reads it from a file you keep out of version control:\n"
            f"    set -a; . .env; set +a\n"
            f"Pass --dry-run to see what a step would do without connecting to anything."
        )
    return uri


def model_banner(name: str) -> str:
    """Say which model is running the turn and what picking it changed.

    Naming the model is not enough on its own. A reader who sees a plausible assistant reply and no
    note about where it came from can read a canned string as evidence of model behavior, so the line
    says what is scripted and what is not (``REQ-E-085``).

    Args:
        name: The value of ``--model``.

    Returns:
        One line, ready to hand to :func:`say`.
    """
    if name == "scripted":
        return (
            "model: scripted. The assistant's replies come from a list in _support.py, so the turn "
            "costs nothing and reads the same on every run. Every MongoDB call below is real. Pass "
            "--model bedrock to run the same script against the SDK's default provider."
        )
    return (
        f"model: {name}. Replies come from the provider, so the wording below will differ from run to "
        f"run and the turn costs whatever the provider charges. Pass --model scripted to remove that."
    )


def mongodb_config(uri: str, database: str) -> MongoDBConfig:
    """Build the connection half of a ``MongoDBConfig``.

    Both examples pass a connection string rather than a client, which is the shape a reader is most
    likely to start from and the one that leaves lifecycle to the package.

    Typed as ``MongoDBConfig`` and not ``dict[str, Any]``. The package exports the ``TypedDict``, so an
    example that declares it gets the same key checking a reader's own code would, and a returned
    ``dict[str, Any]`` would silently accept a misspelled ``connection_sting``.

    Args:
        uri: A MongoDB connection string.
        database: Database to use.

    Returns:
        A mapping accepted as the ``config`` of either surface.
    """
    return {"connection_string": uri, "database": database}


def step_parser(description: str, steps: Sequence[str]) -> argparse.ArgumentParser:
    """Build the parser both examples share.

    Every default is a fixed string rather than a generated one, and that is what makes the command
    one step prints runnable as the next step: a random default would be gone by the time the reader
    ran it.

    Args:
        description: Shown by ``--help``.
        steps: Step names, in the order they are meant to be run.

    Returns:
        A parser carrying the step argument and the flags common to both examples.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(steps), help="which step to run")
    parser.add_argument("--model", choices=("scripted", "bedrock"), default="scripted", help="model to run the turn")
    parser.add_argument("--database", default="strands_examples", help="database to read and write")
    parser.add_argument("--collection", default="example", help="collection to read and write")
    parser.add_argument(
        "--dry-run", action="store_true", help="describe the step and the next command, connect to nothing"
    )
    parser.add_argument("--summary-json", action="store_true", help="print one machine-readable line at the end")
    return parser


def next_command(script: str, step: str, options: Sequence[tuple[str, str]]) -> str:
    """Build the verbatim command for the next step.

    Args:
        script: File name of the running example.
        step: The step to run next.
        options: Flag and value pairs to carry over, already filtered to the ones that differ from
            their defaults. Dropping a flag here sends the reader to a namespace the previous step
            never wrote to, which is why the acceptance test asserts on this string.

    Returns:
        A command line, quoted so it survives a copy into a shell.
    """
    parts = ["python", f"examples/{script}", step]
    for flag, value in options:
        parts += [flag, shlex.quote(value)]
    return " ".join(parts)


def carried_options(
    parser: argparse.ArgumentParser, parsed: argparse.Namespace, names: Sequence[str]
) -> list[tuple[str, str]]:
    """Return the options worth repeating in the next command.

    An option holding its default is left out, because a command line of every flag at its default
    value is noise a reader has to read past to find the one thing that matters.

    Args:
        parser: The parser that produced ``parsed``, read for its defaults.
        parsed: The parsed arguments.
        names: Destination names to consider, in the order they should appear.

    Returns:
        Flag and value pairs, defaults omitted.
    """
    options: list[tuple[str, str]] = []
    for name in names:
        value = getattr(parsed, name)
        if value != parser.get_default(name):
            options.append((f"--{name.replace('_', '-')}", str(value)))
    return options


def emit_summary(payload: dict[str, Any], *, enabled: bool) -> None:
    """Print the one machine-readable line, when it was asked for.

    Off by default so a reader sees prose and nothing else. The acceptance tests turn it on rather
    than parsing the prose, which is what lets the prose be rewritten without breaking them.

    Args:
        payload: Anything JSON-serializable.
        enabled: Whether ``--summary-json`` was passed.
    """
    if enabled:
        print(json.dumps(payload), flush=True)


async def prefix_census(collection: Any) -> list[dict[str, Any]]:
    """Count documents and bytes per top-level key prefix.

    Read from the collection rather than assembled from what the script believes it wrote. The
    prefixes are the SDK's, not this package's: ``session/`` comes from ``SnapshotSessionManager``,
    ``offloader/`` from ``ContextOffloader`` and ``memory/`` from ``FileMemoryStore``. A census that
    reports what it found survives the SDK renaming one of them; a hard-coded list does not.

    Args:
        collection: An ``AsyncCollection`` holding storage documents.

    Returns:
        One entry per prefix, sorted by prefix, each with ``prefix``, ``documents`` and ``bytes``.
        A key with no ``/`` reports under its whole self.
    """
    pipeline: list[dict[str, Any]] = [
        {
            "$project": {
                "prefix": {"$arrayElemAt": [{"$split": [{"$toString": "$_id"}, "/"]}, 0]},
                "size": {"$bsonSize": "$$ROOT"},
            }
        },
        {"$group": {"_id": "$prefix", "documents": {"$sum": 1}, "bytes": {"$sum": "$size"}}},
        {"$sort": {"_id": 1}},
    ]
    census = []
    async for group in await collection.aggregate(pipeline):
        census.append({"prefix": f"{group['_id']}/", "documents": group["documents"], "bytes": group["bytes"]})
    return census


class ScriptedModel(Model):
    """A ``Model`` that reads its replies from a list instead of a provider.

    Here so both examples run with ``MONGODB_URI`` as their only configuration. The parts of these
    examples worth reading are the memory store, the storage backend and what MongoDB holds afterwards;
    the model's contribution is to produce one turn so the extraction trigger and the offloader's
    after-tool hook fire. A scripted reply does that for free, and a reader with no Bedrock access can
    still run the whole thing. Pass ``--model bedrock`` to swap in a real one.

    The event sequence is the SDK's own, taken from ``tests/fixtures/mocked_model_provider.py``:
    ``messageStart``, then a ``contentBlockStart`` / ``contentBlockDelta`` / ``contentBlockStop`` triple
    per content block, then ``messageStop`` carrying the stop reason. A block with a ``toolUse`` sets the
    reason to ``tool_use`` and serializes its input, which is how the event loop knows to run the tool
    and come back for another turn.
    """

    def __init__(self, replies: Sequence[dict[str, Any]]) -> None:
        """Script the replies this model will give, in order.

        Args:
            replies: Assistant messages, each ``{"role": "assistant", "content": [...]}``. A block
                carrying ``text`` streams as text; a block carrying ``toolUse`` streams as a tool call.
        """
        self._replies = list(replies)
        self._index = 0

    def update_config(self, **model_config: Any) -> None:
        """Accept and ignore configuration, because there is none to change."""

    def get_config(self) -> dict[str, Any]:
        """Return an empty configuration.

        Returns:
            An empty mapping, which is what makes ``Model.context_window_limit`` resolve to ``None``.
        """
        return {}

    async def structured_output(self, output_model: Any, prompt: Any, system_prompt: str | None = None, **kwargs: Any):
        """Refuse, rather than return nothing.

        Args:
            output_model: Ignored.
            prompt: Ignored.
            system_prompt: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always. A scripted model has no structured output, and a silent
                ``None`` from a structured call is an afternoon of debugging for whoever meets it.
        """
        raise NotImplementedError("ScriptedModel has no structured output; pass --model bedrock")
        yield {}  # pragma: no cover - unreachable, present so the method is an async generator

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Stream the next scripted reply as ``StreamEvent`` dictionaries.

        Every argument is ignored, including the messages. A model that answered the conversation would
        make the example's output depend on the model, and the point of these two scripts is that the
        output depends on MongoDB.

        ``StreamEvent`` rather than ``dict[str, Any]`` in the annotation, because the supertype promises
        it and mypy checks each yielded literal against it. That is worth having here: an event with a
        misspelled key streams as a no-op and produces a turn with no content, which reads like a model
        problem rather than a typo.

        Yields:
            One reply's worth of stream events.

        Raises:
            RuntimeError: If the script has run out. Raising names the shortfall; falling off the end
                would yield an empty stream, which the event loop reads as a reply with no content.
        """
        if self._index >= len(self._replies):
            raise RuntimeError(
                f"ScriptedModel ran out of replies after {len(self._replies)}; the turn asked for another"
            )
        reply = self._replies[self._index]
        self._index += 1

        yield {"messageStart": {"role": "assistant"}}
        stop_reason: StopReason = "end_turn"
        for block in reply["content"]:
            if "text" in block:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
                yield {"contentBlockStop": {}}
            if "toolUse" in block:
                stop_reason = "tool_use"
                tool_use = block["toolUse"]
                yield {
                    "contentBlockStart": {
                        "start": {"toolUse": {"name": tool_use["name"], "toolUseId": tool_use["toolUseId"]}}
                    }
                }
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_use["input"])}}}}
                yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": stop_reason}}


def resolve_model(name: str, replies: Sequence[dict[str, Any]]) -> Model:
    """Return the model the ``--model`` flag asked for.

    Args:
        name: ``scripted`` or ``bedrock``.
        replies: The script, used only by ``scripted``.

    Returns:
        A ``Model``. The Bedrock branch takes no arguments, so it uses whatever region and credentials
        the environment already provides and whichever model id the SDK defaults to.

    Raises:
        ExampleError: If ``name`` is neither.
    """
    if name == "scripted":
        return ScriptedModel(replies)
    if name == "bedrock":
        from strands.models import BedrockModel

        return BedrockModel()
    raise ExampleError(f"unknown model {name!r}")
