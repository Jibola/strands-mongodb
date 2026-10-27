"""Connection resolution, ownership and per-loop locking, shared by both public surfaces.

``MongoDBConfig`` is a ``TypedDict``, so it is a plain ``dict`` and cannot hold state. The client built
from it, the lock guarding its one-time setup and its capability probe therefore live in a module-level
registry keyed by connection identity. That is what lets any number of stores configured from the same
deployment share one connection pool per event loop.

Private: the registry, the identity scheme and ``_Connection`` may change in any release.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import urllib.parse
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Final

from strands_mongodb._extras import missing_extra

# `storage`, because both surfaces reach this module and `[memory]` is a superset of `[storage]`.
try:
    from pymongo import AsyncMongoClient
    from pymongo.driver_info import DriverInfo
except ImportError as error:  # pragma: no cover
    missing_extra("storage", error, module=__name__)

from strands_mongodb import _telemetry

if TYPE_CHECKING:
    from strands_mongodb.types import MongoDBConfig

__all__ = ["Capabilities", "close", "resolve"]

logger = logging.getLogger(__name__)

# The name the server logs and Atlas telemetry group by. Changing it orphans historical usage data.
_DRIVER_NAME = "Strands"

# The handshake `appName` shown in server logs, `$currentOp` and the profiler. Renaming it also orphans
# historical usage data.
_APP_NAME = "strands-mongodb"
_APP_NAME_SUFFIX = "-" + _APP_NAME
# pymongo's limit, in UTF-8 bytes. Over it, `AsyncMongoClient` raises `ValueError`.
_APP_NAME_MAX_BYTES = 128

_MISSING_CONNECTION = (
    "strands_mongodb: a MongoDB connection is required, but neither 'client' nor "
    "'connection_string' was set in the configuration. Set 'connection_string' to a MongoDB URI, or "
    "set 'client' to an AsyncMongoClient you already have."
)

_RETRY_WRITES_DISABLED = (
    "retryWrites is disabled on this MongoDB connection. strands_mongodb relies on the driver's own "
    "retry to replay its idempotent upserts across a primary stepdown, and with retryWrites=false "
    "that replay does not happen: a write interrupted by an election surfaces to the caller instead. "
    "Nothing is corrupted either way. Remove retryWrites=false to restore the behavior this package "
    "was designed around."
)


@dataclass
class Capabilities:
    """What a deployment can do, probed once and cached on its connection.

    ``_Connection.capabilities`` is ``None`` until the probe has run, so "not probed" never shares a value
    with "no search".

    Attributes:
        search: Whether the deployment runs Atlas Search. This is a property of the deployment, not of a
            collection, so one probe serves every store on the connection.
        no_search_code: The server error code that reported no search, or ``None``. Set only when
            ``search`` is ``False``. Cached because ``search()`` names it in every later error, and the
            code differs by server version and build, so nothing else on hand can derive it.
        rank_fusion: Whether the deployment's ``mongod`` is new enough for ``$rankFusion``, the hybrid
            tier. It is probed only where search exists, because hybrid also needs both search tiers.
            The default is ``False`` so a deployment that was never probed is never sent a stage it
            cannot run.

            Unlike ``search``, this is read from the server version. Atlas Search is a separate process
            whose presence no version predicts, while ``$rankFusion`` is a stage inside the process that
            ``buildInfo`` describes.
    """

    search: bool
    no_search_code: int | None = None
    rank_fusion: bool = False


@dataclass
class _Connection:
    """One MongoDB connection and the per-connection state a ``TypedDict`` cannot hold.

    No field holds the connection string. A dataclass ``__repr__`` covers every field, so a stored URI
    would print its inline password wherever this object is rendered.

    Attributes:
        client: The shared client, built here or injected by the caller.
        owned: Whether this module built the client. ``False`` for an injected client, which is never
            closed.
        lock: Guards one-time setup, such as index creation, against concurrent stores. Valid only for
            ``loop``; read it through :meth:`get_lock`.
        loop: The event loop ``lock`` is usable on, or ``None`` before its first use.
        capabilities: Result of the server capability probe, or ``None`` until it has run.
        binary_keys: ``(database, collection)`` pairs whose default collation was read and found binary.
            A namespace is added only when a collection existed to read it from, because
            ``listCollections`` reports an absent collection and an uncollated one identically. Cached on
            the connection because every store over one collection, including each namespaced store,
            shares the answer. A collation cannot be altered in place, so the cache goes stale only if the
            collection is dropped and recreated under the same name while the process runs.
        recheck_attribution: Whether a later ``resolve`` of an injected client should check its driver
            metadata again and re-apply the attribution if it is gone. Cleared for good when an append
            does not become visible, which bounds the re-applying on a driver that ignores it.
    """

    # `repr=False` because `AsyncMongoClient.__repr__` renders every non-default option. pymongo redacts
    # the URI password but not `authMechanismProperties` (which can carry `AWS_SESSION_TOKEN`) or
    # `tlsCertificateKeyFilePassword`, and this repr is reached through `repr(_REGISTRY)` as well.
    client: AsyncMongoClient[Any] = field(repr=False)
    owned: bool
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    loop: asyncio.AbstractEventLoop | None = None
    capabilities: Capabilities | None = None
    binary_keys: set[tuple[str, str]] = field(default_factory=set)
    recheck_attribution: bool = True

    def get_lock(self) -> asyncio.Lock:
        """Return a lock usable on the running event loop, replacing it if the loop changed.

        A synchronous ``Agent`` runs each invocation on a fresh loop, and an ``asyncio.Lock`` that created
        a waiter future on one loop raises ``RuntimeError`` when awaited on another. The binding happens
        only under contention, so an uncontended lock crosses loops silently until load arrives, and from
        then on every contended acquire on a later loop raises.

        Replacing the lock is safe because it only guards work scoped to one loop. Any waiters on the old
        lock belong to a loop that has already closed.

        Returns:
            A lock bound to the running loop, stable for the lifetime of that loop.

        Raises:
            RuntimeError: If called with no running event loop.
        """
        running = asyncio.get_running_loop()
        if self.loop is not running:
            self.lock = asyncio.Lock()
            self.loop = running
        return self.lock


#: A connection identity paired with the event loop that connection serves. ``None`` means no loop: a
#: client built by synchronous code that no loop has adopted yet, or an injected client, which is never
#: split per loop.
_RegistryKey = tuple[str, asyncio.AbstractEventLoop | None]

# Keyed by the digest from `_identity`, never by a connection string, so the keys cannot leak a password.
# Process-global: every store built from one config shares one pool per event loop.
_REGISTRY: dict[_RegistryKey, _Connection] = {}

_REGISTRY_LOCK: Final = threading.RLock()
"""Guards every read, mutation and iteration of :data:`_REGISTRY`.

A thread lock, not an ``asyncio`` one: ``resolve`` is synchronous (the store constructors call it), and
threads that each run their own loop, as synchronous ``Agent`` instances do, reach the registry with no
loop in common.

Reentrant because ``resolve`` holds it while calling :func:`_register`, which takes it too. Splitting the
two to avoid nesting would put a check and its act on either side of a release.

Never held across an ``await``. Because it is reentrant, another coroutine on the same thread would
acquire it while the holder is suspended, so it would exclude nothing on that loop. Meanwhile every other
thread that reaches the registry would block for the whole wait.
"""


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """Return the running event loop, or ``None`` when there is none.

    ``resolve`` is called both from coroutines and from ``MongoDBStorage.__init__`` at module level, where
    no loop is running.

    Returns:
        The running loop, or ``None`` outside one.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _reap_closed_loops() -> None:
    """Drop every registry entry whose event loop has closed.

    Called each time a client is built. It sweeps every identity because such an entry is unusable by
    anyone: pymongo refuses the client from any other loop. Keeping it would leak one entry per loop for the
    life of the process.

    The client is dropped, not closed, because ``close()`` sends ``endSessions`` on the loop that is gone.
    Its sockets are reachable only from that loop's selector, and their file descriptors are released when
    the object is collected.

    The snapshot and the removals run under :data:`_REGISTRY_LOCK` because another thread may be writing
    the dict.
    """
    with _REGISTRY_LOCK:
        for key in [key for key, _ in _REGISTRY.items() if key[1] is not None and key[1].is_closed()]:
            _REGISTRY.pop(key, None)


def _package_version() -> str:
    """Return this package's installed version for driver attribution.

    Returns:
        The installed version, or ``"unknown"`` when the package is not installed as a distribution,
        which happens when it is run straight from a source checkout.
    """
    try:
        return version("strands-mongodb")
    except PackageNotFoundError:  # pragma: no cover - requires an uninstalled source checkout
        return "unknown"


def _driver_info() -> DriverInfo:
    """Build the attribution PyMongo appends to its own driver metadata.

    Returns:
        A ``DriverInfo`` naming Strands and the installed package version.
    """
    return DriverInfo(name=_DRIVER_NAME, version=_package_version())


def _app_name(connection_string: str, client_args: dict[str, Any]) -> str:
    """Derive the ``appName`` a client this package builds sends: the caller's, suffixed, or the package's own.

    The caller's value is the one pymongo would have used. A ``client_args`` key in any letter case wins,
    as pymongo's keyword options do. Otherwise it is the last ``appName`` in the query string. Only the
    text after the first ``?`` is read, so the part of the URI that holds credentials is never parsed,
    and an ``srv`` URI is never resolved: its TXT record cannot carry ``appName`` anyway.

    Args:
        connection_string: The configured URI.
        client_args: The configured client arguments.

    Returns:
        ``strands-mongodb`` when no value is configured. The value unchanged when it already ends in the
        suffix. Otherwise ``<value>-strands-mongodb``, with the value shortened on a character boundary
        when the whole would exceed pymongo's 128-byte limit, so the suffix always survives.
    """
    configured: str | None = None
    for key, value in client_args.items():
        if key.lower() == "appname":
            configured = None if value is None else str(value)
    if configured is None:
        _, _, query = connection_string.partition("?")
        for pair in query.replace(";", "&").split("&"):
            key, _, value = pair.partition("=")
            if key.lower() == "appname":
                configured = urllib.parse.unquote_plus(value)

    if not configured:
        return _APP_NAME
    if configured == _APP_NAME or configured.endswith(_APP_NAME_SUFFIX):
        return configured
    room = _APP_NAME_MAX_BYTES - len(_APP_NAME_SUFFIX.encode())
    # `errors="ignore"` drops a multi-byte character the cut split in half, rather than replacing it.
    kept = configured.encode()[:room].decode(errors="ignore")
    return kept + _APP_NAME_SUFFIX


def _identity(config: MongoDBConfig) -> str:
    """Compute the registry key for a config.

    Two configs share a client when they would produce the same one. ``database`` is excluded, since one
    client serves any number of databases. ``client_args`` is included, because a client built with
    different timeouts is a different client and a cache hit would otherwise discard the arguments.

    An injected client keys on its object identity. Two clients built from the same URI can differ in
    credentials, read preference or timeouts, so handing one caller another's client would be wrong.

    The result is a digest, never the connection string itself. A dict keyed on URIs leaks its keys
    through ``repr``, a ``KeyError`` and any debugger that walks module globals, and a digest cannot be
    reversed into a password.

    Args:
        config: The configuration to key.

    Returns:
        An opaque key, stable for the lifetime of the process.

    Raises:
        ValueError: If neither ``client`` nor ``connection_string`` is set.
    """
    injected = config.get("client")
    if injected is not None:
        return f"client:{id(injected):x}"

    connection_string = config.get("connection_string")
    if not connection_string:
        raise ValueError(_MISSING_CONNECTION)

    # `default=repr` so an unserializable argument (a `ServerApi`, an SSLContext) still produces a key.
    # Distinct objects then key distinctly, which errs toward a second client over sharing the wrong one.
    args = json.dumps(config.get("client_args") or {}, sort_keys=True, default=repr)
    digest = hashlib.sha256(f"{connection_string}\x00{args}".encode()).hexdigest()
    return f"uri:{digest}"


def _warn_if_retries_disabled(client: AsyncMongoClient[Any]) -> None:
    """Warn when the resolved client will not retry writes.

    Reads ``client.options`` rather than parsing the URI, so it catches the setting whether it came from
    the URI, from ``client_args`` or from an injected client.

    Args:
        client: The client about to be registered.
    """
    if client.options.retry_writes is False:
        logger.warning(_RETRY_WRITES_DISABLED, extra=_telemetry.event_extra("retry_writes_disabled"))


def _already_attributed(client: AsyncMongoClient[Any]) -> bool:
    """Whether this client's driver metadata already names Strands.

    ``driver.name`` is a pipe-delimited chain such as ``PyMongo|c|async|Strands``. Entries are compared
    whole and case-insensitively, as pymongo 4.14.1 and later do internally, so the answer does not depend
    on the installed pymongo. A substring match would wrongly count an entry such as ``StrandsToolkit``.

    Args:
        client: The client to inspect, read through its public ``options``.

    Returns:
        Whether an entry equal to this package's driver name is already in the chain.
    """
    chain = client.options.pool_options.metadata["driver"]["name"]
    return _DRIVER_NAME.casefold() in (entry.casefold() for entry in chain.split("|"))


def _attribute_injected_client(client: AsyncMongoClient[Any]) -> None:
    """Attach Strands attribution to a client the caller injected, at most once.

    This inlines ``pymongo_search_utils.append_client_metadata``, which lives in the ``[memory]`` extra,
    while this module must also work under a ``[storage]``-only install.

    ``append_metadata`` is called without probing for it because the pymongo floor, 4.14.0, introduced it.
    On older versions a ``hasattr`` probe would pass anyway, since the client's ``__getattr__`` returns a
    ``Database`` of that name.

    Args:
        client: The injected client to attribute in place.
    """
    # At most once: closing a store and resolving the same client again reaches this call again. pymongo
    # 4.14.0 does not dedupe the append, so each cycle would grow the handshake until it passes the
    # server's 512-byte client metadata limit and every new connection fails (22 cycles on mongod 8).
    if _already_attributed(client):
        return

    client.append_metadata(_driver_info())


def _reattribute_if_it_was_discarded(connection: _Connection) -> None:
    """Re-apply an injected client's attribution if the driver discarded it.

    ``append_metadata`` writes into the client's ``ClientOptions``, and resolving a ``mongodb+srv://`` URI
    replaces them on the client's first connection. For an injected SRV client the append lands first and
    is lost (pymongo 4.17.0: ``PyMongo|c|async|Strands`` before, ``PyMongo|c|async`` after). This restores
    the attribution for every pool built afterwards, not for handshakes already sent.

    Args:
        connection: The registered connection for an injected client.
    """
    if not connection.recheck_attribution:
        return
    client = connection.client
    if _already_attributed(client):
        # The flag stays set: the loss happens on the first connection, which can come after any resolve.
        return

    client.append_metadata(_driver_info())
    if _already_attributed(client):
        return

    # The append did not become visible, so retrying would grow the handshake once per resolve on a driver
    # that appends without deduping. Stop after one attempt.
    connection.recheck_attribution = False
    logger.debug(
        "strands_mongodb: append_metadata did not become visible on the injected client, so the Strands "
        "attribution is left as it is",
        extra=_telemetry.event_extra("attribution_not_visible"),
    )


def resolve(config: MongoDBConfig) -> _Connection:
    """Return the shared connection for a config and the running loop, creating it on first use.

    There is one pool per connection identity per event loop. ``AsyncMongoClient`` binds to the loop of
    its first operation, so one client cannot serve a synchronous ``Agent``, which runs every entry point
    on a loop of its own.

    Args:
        config: The configuration to resolve. Only the connection fields are read.

    Returns:
        The registered connection. The same object for every config with the same connection identity
        resolved on the same loop, so callers must not mutate it.

    Raises:
        ValueError: If neither ``client`` nor ``connection_string`` is set. Raised before any client is
            built, and nothing is registered. pymongo would otherwise read a schemeless string as a
            hostname and fail later with a server-selection timeout against ``localhost``.
        ValueError: If ``AsyncMongoClient`` rejects a malformed connection string or option value. It is
            not wrapped, and nothing is registered.
        pymongo.errors.ConfigurationError: If ``AsyncMongoClient`` rejects an option, such as an unknown
            ``client_args`` key. It is not wrapped, and nothing is registered.
    """
    identity = _identity(config)

    # One lock over the whole body: every branch is a check followed by an act on process-global state.
    # Nothing here does I/O, since `AsyncMongoClient()` connects lazily. `_identity` runs above the lock so
    # an unusable config is rejected without touching the registry.
    with _REGISTRY_LOCK:
        injected = config.get("client")
        if injected is not None:
            # One slot, keyed on no loop: there is no second copy to build, and an injected client is never
            # closed. Driving it from two loops gets pymongo's own RuntimeError.
            return _register(identity, None, injected, owned=False, attribute=True)

        running = _running_loop()
        existing = _REGISTRY.get((identity, running))
        if existing is not None:
            # Returning here keeps the retryWrites warning to one line per connection, not one per store.
            return existing

        if running is not None:
            # Adoption: a client built with no loop running has not bound to a loop, so the first loop
            # that asks takes it over. Building another would leave the first open and never closed in the
            # common case of a store constructed at import time and used on one loop.
            adopted = _REGISTRY.pop((identity, None), None)
            if adopted is not None:
                _REGISTRY[(identity, running)] = adopted
                return adopted

        _reap_closed_loops()

        # `driver=` and `appname=` reach pymongo through `**kwargs`, so nothing type checks them; the tests
        # assert the resulting pool metadata. Every case variant of `appname` is removed so the client
        # receives exactly one.
        client_args = dict(config.get("client_args") or {})
        app_name = _app_name(config["connection_string"], client_args)
        client_args = {key: value for key, value in client_args.items() if key.lower() != "appname"}
        client: AsyncMongoClient[Any] = AsyncMongoClient(
            config["connection_string"],
            driver=_driver_info(),
            appname=app_name,
            **client_args,
        )
        return _register(identity, running, client, owned=True, attribute=False)


def _register(
    identity: str,
    loop: asyncio.AbstractEventLoop | None,
    client: AsyncMongoClient[Any],
    *,
    owned: bool,
    attribute: bool,
) -> _Connection:
    """Put a client in the registry under one identity and loop, warning at most once about it.

    Args:
        identity: The connection identity from :func:`_identity`.
        loop: The loop this client serves, or ``None`` for the loop-independent slot.
        client: The client to register.
        owned: Whether this module built it, which decides whether :func:`close` may close it.
        attribute: Whether to attach Strands attribution, which only an injected client needs.

    Returns:
        The registered connection, existing or new.
    """
    # Takes the lock itself so the check and the insert are one step across threads, which is what keeps
    # the warning to at most once. The lock is reentrant, so nesting under `resolve` is free.
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get((identity, loop))
        if existing is not None:
            if attribute:
                _reattribute_if_it_was_discarded(existing)
            return existing
        if attribute:
            _attribute_injected_client(client)
        _warn_if_retries_disabled(client)
        connection = _Connection(client=client, owned=owned)
        _REGISTRY[(identity, loop)] = connection
    # Runs once per new registry entry (a cache hit returns above and adoption skips this function), so a
    # pool built is an event and a pool reused is not. Safe under the caller's lock: it does no I/O.
    _telemetry.event("strands_mongodb.client.acquired", {"strands_mongodb.client.injected": not owned})
    return connection


async def close(config: MongoDBConfig) -> None:
    """Release the connection for a config, closing the client only if this module built it.

    An injected client is never closed. It is often shared with other subsystems and the application's
    own queries, and closing it would break them somewhere else, later.

    Idempotent, and safe on a config that was never resolved, since callers run it from ``finally`` blocks
    and ``atexit`` handlers. Every loop's slot for the identity is released, not only the running loop's, because
    nothing in the public API names a loop.

    Args:
        config: The configuration whose connection to release.
    """
    try:
        identity = _identity(config)
    except ValueError:
        # Nothing was ever registered for this config, and a teardown path should not raise.
        return

    running = asyncio.get_running_loop()
    with _REGISTRY_LOCK:
        # Snapshot under the lock, iterate outside it: the loop body awaits, and the lock is never held
        # across an `await`.
        keys = [key for key in _REGISTRY if key[0] == identity]

    for key in keys:
        # Popped with a default: during the `await` below, a concurrent `close()` for the same config or a
        # reap in `resolve()` may already have removed the slot.
        with _REGISTRY_LOCK:
            connection = _REGISTRY.pop(key, None)
        if connection is None or not connection.owned:
            continue
        loop = key[1]
        if loop is None or loop is running:
            await connection.client.close()
        elif not loop.is_closed():
            # Another live loop's client must close on that loop, since `endSessions` uses its sockets.
            # Scheduled, not awaited: this loop may be the one that loop is waiting for.
            pending = connection.client.close()
            try:
                asyncio.run_coroutine_threadsafe(pending, loop)
            except RuntimeError:
                # That loop closed between the check and the call. Swallowed so the remaining slots are
                # still released, and the coroutine is closed so it never warns as un-awaited. The client
                # is dropped unclosed, like any closed loop's client.
                pending.close()
                logger.debug(
                    "strands_mongodb: an event loop closed while its client was being scheduled for "
                    "close; the slot is released and the client is dropped unclosed",
                    extra=_telemetry.event_extra("closed_loop_client_dropped"),
                )
        # A closed loop's client is dropped, not closed: `close()` needs the loop its sockets belong to,
        # and pymongo raises a `RuntimeError` there, which is not a `PyMongoError`.
