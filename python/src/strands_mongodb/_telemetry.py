"""OpenTelemetry spans and structured log fields for both stores. The only module that imports OpenTelemetry.

The package configures no provider, exporter or sampling. The tracer is looked up on every call, so a
provider installed after import still gets spans, and with none installed the no-op tracer costs a
function call.

Only allow-listed attributes are recorded. No span carries caller data: keys, content, metadata, query
text, filter values, scope or memory ids. ``traced`` reads the store instance, never the
call's arguments. ``span``, ``annotate`` and ``event`` accept only names in ``Attribute``, checked by
``mypy`` and again at runtime, where an unknown name is dropped rather than raised so a telemetry typo
never fails a write.

Exceptions are recorded as a status and a type name, never a message. Raised messages name the key and
the namespace, and spans are exported to third parties.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import TYPE_CHECKING, Any, Final, Literal, ParamSpec, TypeVar, get_args

from strands_mongodb import __version__
from strands_mongodb._extras import missing_extra

# Both runtime extras bring `opentelemetry-api`, so this fires only on a bare install.
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
except ImportError as error:  # pragma: no cover - exercised in a subprocess
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
    "tier_building",
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

    Apply it outermost, above ``_reporting``, so the span sees the ``StorageError`` the caller sees rather
    than the driver error underneath.

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
            # A local because mypy rejects indexing `ParamSpec` args. Only `self` is read: the call's
            # arguments carry caller data.
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
                    # Not `BaseException`: a cancelled call is not a store failure, so its status stays unset.
                    _mark(span, error)
                    raise

        # On the wrapper, not in a registry, so a test can list a store class's spans without its own.
        tracing.__dict__["strands_mongodb_span"] = name
        return tracing

    return decorate


@contextlib.contextmanager
def span(name: SpanName, attributes: Mapping[Attribute, AttributeValue]) -> Iterator[None]:
    """Open an ``INTERNAL`` child span under the current one, for work a store operation causes.

    Same rules as ``traced``: the tracer is looked up on entry, attributes pass through the allow-list, and
    an exception is recorded as a status and a type name, then re-raised unchanged. Inside it, ``annotate``
    sets attributes on this span rather than the operation's.

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
    """Build the ``extra`` mapping for one log record, so a JSON formatter can index it without parsing.

    Keys are prefixed ``strands_mongodb_`` because ``logging`` raises ``KeyError`` for an ``extra`` key
    that collides with a ``LogRecord`` attribute such as ``name``. A field that does not apply is omitted
    rather than set to ``None``.

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

    A no-op under the no-op tracer, so call sites need no conditionals.

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
