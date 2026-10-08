"""OpenTelemetry spans for both stores, and the one module in the package that imports OpenTelemetry.

The package configures nothing: no provider, no exporter, no sampling. The tracer is looked up on every
call, so an application that installs its provider after importing this package still gets spans, and one
that installs none gets OpenTelemetry's no-op tracer and pays for a function call.

What reaches a span is decided here and nowhere else. ``traced`` reads the instance and never the call's
arguments, which is where keys, content, queries and filters live. Everything else arrives through
``span``, ``annotate`` and ``event``, and their names are restricted to ``Attribute``: a ``Literal``, so ``mypy``
rejects a name outside it, and a runtime check that drops one rather than raising, because a telemetry
typo must never become a failed write.

Exceptions are recorded as a status and a type name and never as a message. The messages this package
raises name the key and the namespace on purpose, and a span is exported to a third party where a raised
exception is not.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import TYPE_CHECKING, Any, Final, Literal, ParamSpec, TypeVar, get_args

from strands_mongodb import __version__
from strands_mongodb._extras import missing_extra

# Guarded like every other module that needs an extra. Both runtime extras bring `opentelemetry-api`, so
# this fires only on a bare install, and it should name the extra that fixes it rather than a module name.
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
except ImportError as error:  # pragma: no cover - exercised in a subprocess by AT-008b
    missing_extra("storage", error, module=__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterator, Mapping

    from opentelemetry.util.types import AttributeValue

__all__ = ["Attribute", "EventName", "LogEvent", "SpanName", "annotate", "event", "event_extra", "span", "traced"]

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

#: Every attribute name the package may set, one per row of the README's attribute table.
Attribute = Literal[
    "strands_mongodb.store_name",
    "strands_mongodb.search.requested",
    "strands_mongodb.search.tier",
    "strands_mongodb.search.tiers",
    "strands_mongodb.result_count",
    "strands_mongodb.deleted_count",
    "strands_mongodb.found",
    "strands_mongodb.duplicate",
    "strands_mongodb.database",
    "strands_mongodb.collection",
    "strands_mongodb.client.injected",
    "strands_mongodb.indexes.mode",
    "strands_mongodb.indexes.created",
    "strands_mongodb.indexes.outcome",
    "strands_mongodb.indexes.probe",
]

#: Every span opened by ``span`` rather than by ``traced``: index work, as a child of the operation that needed it.
SpanName = Literal[
    "strands_mongodb.indexes.ensure",
    "strands_mongodb.indexes.probe",
    "strands_mongodb.indexes.wait",
]

#: The ``strands_mongodb_event`` value of every log record the package emits, one per message.
LogEvent = Literal[
    "retry_writes_disabled",
    "attribution_not_visible",
    "closed_loop_client_dropped",
    "reserved_key_collision",
    "no_search_setup",
    "tier_unavailable",
    "below_vector",
    "bound_declined_fusion",
    "score_bound_applied",
    "creating_indexes",
    "creating_search_index",
    "search_index_status",
    "attribute_dropped",
]

#: Every span event the package may add.
EventName = Literal["strands_mongodb.client.acquired"]

_ALLOWED: Final[frozenset[str]] = frozenset(get_args(Attribute))


def traced(
    surface: Literal["storage", "memory"], operation: str
) -> Callable[[Callable[_P, Coroutine[Any, Any, _R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """Record one ``INTERNAL`` span named ``strands_mongodb.<surface>.<operation>`` around a store method.

    Outermost on the method, above ``_reporting``, so the span sees the ``StorageError`` the caller sees
    rather than the driver error underneath. ``Coroutine`` rather than ``Awaitable`` for the reason
    ``_reporting`` gives: the protocols declare these methods ``async def``.

    Args:
        surface: ``storage`` or ``memory``. A memory span also records the store's name.
        operation: The public method's name.

    Returns:
        A decorator for an async method whose ``self`` has ``_database_name`` and ``_collection_name``.

    Raises:
        Exception: Whatever the decorated method raises, re-raised as the same object after the span is
            marked. The decorator raises nothing of its own.
    """
    name = f"strands_mongodb.{surface}.{operation}"

    def decorate(method: Callable[_P, Coroutine[Any, Any, _R]]) -> Callable[_P, Coroutine[Any, Any, _R]]:
        @functools.wraps(method)
        async def tracing(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # Bound through a local rather than indexed off `args` directly, which mypy rejects for a
            # `ParamSpec`. Only `self` is read: the arguments are what this module must never see.
            positional: tuple[Any, ...] = args
            tracer = trace.get_tracer("strands_mongodb", __version__)
            with tracer.start_as_current_span(
                name,
                kind=SpanKind.INTERNAL,
                attributes=_base(positional[0], surface),
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                try:
                    return await method(*args, **kwargs)
                except Exception as error:
                    # `Exception`, not `BaseException`: a cancelled call ends its span with the status
                    # unset, because cancelling is the caller's decision and not a store failure.
                    _mark(span, error)
                    raise

        # Stored on the wrapper rather than in a module-level registry, so the README test can list the
        # spans a store class emits without seeing ones a test decorated for itself.
        tracing.__dict__["strands_mongodb_span"] = name
        return tracing

    return decorate


@contextlib.contextmanager
def span(name: SpanName, attributes: Mapping[Attribute, AttributeValue]) -> Iterator[None]:
    """Open an ``INTERNAL`` child span under the current one, for work a store operation causes.

    The same rules as ``traced``: the tracer looked up on entry, attributes filtered through the allow-list,
    and an exception recorded as a status and a type name and re-raised unchanged. Inside it, ``annotate``
    sets attributes on this span rather than on the operation's.

    Args:
        name: An index span name.
        attributes: Its starting attributes, usually the namespace and the mode or probe.

    Yields:
        Nothing. The span is current for the body of the ``with``.

    Raises:
        Exception: Whatever the body raises, re-raised as the same object after the span is marked. The
            context manager raises nothing of its own.
    """
    kept: dict[str, AttributeValue] = {key: value for key, value in attributes.items() if _allowed(key)}
    tracer = trace.get_tracer("strands_mongodb", __version__)
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.INTERNAL,
        attributes=kept,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield
        except Exception as error:
            _mark(current, error)
            raise


def event_extra(event: LogEvent, *, namespace: str | None = None, tier: str | None = None) -> dict[str, str]:
    """The ``extra`` mapping for one log record, so a JSON formatter can index it without parsing the message.

    Prefixed ``strands_mongodb_`` because ``logging`` raises ``KeyError`` for an ``extra`` key that collides
    with a ``LogRecord`` attribute such as ``name`` or ``message``. A field that does not apply is omitted
    rather than set to ``None``, so a formatter's output has no keys that never carry anything.

    Args:
        event: The record's stable name.
        namespace: ``<database>.<collection>``, where the record concerns one.
        tier: The search tier, where the record concerns one.

    Returns:
        The mapping to pass as ``extra``.
    """
    extra: dict[str, str] = {"strands_mongodb_event": event}
    if namespace is not None:
        extra["strands_mongodb_namespace"] = namespace
    if tier is not None:
        extra["strands_mongodb_tier"] = tier
    return extra


def annotate(name: Attribute, value: AttributeValue) -> None:
    """Set an outcome attribute on the current span, which inside a store method is the operation's own.

    A no-op against the no-op tracer, so call sites carry no conditionals.

    Args:
        name: An allow-listed attribute name.
        value: A count, flag, tier name or configuration value. Never anything the caller supplied.
    """
    if _allowed(name):
        trace.get_current_span().set_attribute(name, value)


def event(name: EventName, attributes: Mapping[Attribute, AttributeValue]) -> None:
    """Add an event to the current span, keeping only allow-listed attributes.

    Args:
        name: The event name.
        attributes: The event's attributes.
    """
    kept: dict[str, AttributeValue] = {key: value for key, value in attributes.items() if _allowed(key)}
    trace.get_current_span().add_event(name, kept)


def _allowed(name: str) -> bool:
    """Whether ``name`` is on the allow-list, logging at ``DEBUG`` when it is not."""
    if name in _ALLOWED:
        return True
    logger.debug(
        "strands_mongodb: dropped span attribute %r, which is not on the allow-list",
        name,
        extra=event_extra("attribute_dropped"),
    )
    return False


def _mark(current: trace.Span, error: Exception) -> None:
    """Record a failure as ``ERROR`` with the class name, and ``error.type``, never the message."""
    kind = type(error)
    current.set_status(Status(StatusCode.ERROR, kind.__name__))
    current.set_attribute("error.type", f"{kind.__module__}.{kind.__qualname__}")


def _base(instance: Any, surface: str) -> dict[str, AttributeValue]:
    """The attributes every span carries, read off the store and never off the call."""
    attributes: dict[str, AttributeValue] = {
        "strands_mongodb.database": instance._database_name,
        "strands_mongodb.collection": instance._collection_name,
    }
    if surface == "memory":
        attributes["strands_mongodb.store_name"] = instance.name
    return attributes
