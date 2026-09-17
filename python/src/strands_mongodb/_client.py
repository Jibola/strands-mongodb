"""Connection resolution, ownership and per-loop locking, shared by both public surfaces.

``MongoDBConfig`` is a ``TypedDict``, so it is a plain ``dict`` and cannot hold state. The client
built from it, the lock guarding its one-time setup and its capability probe therefore live in a
module-level registry keyed by connection identity, which is what lets N stores configured from the
same database share one connection pool (``REQ-E-012``).

Private because none of it is a promise. The registry, the identity scheme and ``_Connection`` are all
free to change within 0.1.x.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Final

from strands_mongodb._extras import missing_extra

# `storage` rather than `memory`: this module is reachable from both surfaces, and everything it
# needs is in the narrower extra. A `memory`-only install does not exist, since `[memory]` is a
# superset of `[storage]` (REQ-E-002).
try:
    from pymongo import AsyncMongoClient
    from pymongo.driver_info import DriverInfo
except ImportError as error:  # pragma: no cover - exercised in a subprocess by AT-008b
    missing_extra("storage", error, module=__name__)

if TYPE_CHECKING:
    from strands_mongodb.types import MongoDBConfig

__all__ = ["Capabilities", "close", "resolve"]

logger = logging.getLogger(__name__)

# The name the server logs and Atlas telemetry group by. Changing it orphans historical usage data.
_DRIVER_NAME = "Strands"

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

    A record rather than a bare ``bool`` on ``_Connection``, for two reasons that both come from the
    field it replaces being ``Any | None``. ``None`` has to keep meaning *not probed yet*, and
    ``bool | None`` makes that a three-state flag whose ``if capabilities:`` reads as "has search" while
    also being false for "unknown". And search support is the first of these, not the last:
    ``rank_fusion`` sits beside it, exactly as this docstring predicted, and ``$rerank`` is the next
    one — 8.3, plus Native Reranking enabled in the Atlas project, plus a Voyage AI key, so it is three
    conditions rather than one and it is not in v0.1.0.

    Attributes:
        search: Whether the deployment runs Atlas Search. A property of the deployment rather than of a
            collection, which is why one probe serves every store on the connection
            (``REQ-E-065``).
        no_search_code: The server error code that said so, or ``None``. Set only when ``search`` is
            ``False``, and cached here for the same reason ``search`` is: the code is seen once, by the
            probe, and ``search()`` needs it on every later call to name it in what it raises. Three
            codes mean this and they differ by server version and build, so the one this deployment used
            is not derivable from anything else on hand (``REQ-E-053a``).
        rank_fusion: Whether the deployment's ``mongod`` is new enough for ``$rankFusion``, which is the
            hybrid tier (``REQ-E-143``). Defaults to ``False`` so that the value a
            store reads when the probe was skipped is the one that costs nothing: hybrid also needs both
            search tiers, so this is never probed on a deployment without search and the default is what
            that deployment's stores see. ``False`` there means "resolve as you did before hybrid
            existed", while a default of ``True`` would mean "send a stage this server has never heard
            of".

            Unlike ``search``, this one is a version read. The two probes disagree about that on
            purpose: Atlas Search is a separate process whose presence no version predicts, and
            ``$rankFusion`` is a stage inside the process ``buildInfo`` describes.
    """

    search: bool
    no_search_code: int | None = None
    rank_fusion: bool = False


@dataclass
class _Connection:
    """One MongoDB connection and the per-connection state a ``TypedDict`` cannot hold.

    No field holds the connection string. That is not incidental: a dataclass generates a ``__repr__``
    covering every field, so a URI stored here would print its inline password wherever this object is
    rendered (``REQ-E-014``).

    Attributes:
        client: The shared client. Built by us or handed to us.
        owned: Whether we built it. ``False`` for an injected client, which is never closed
            (``REQ-E-010``).
        lock: Guards one-time setup, such as index creation, against N stores racing. Valid only for
            ``loop``; read it through :meth:`get_lock`.
        loop: The event loop ``lock`` is usable on, or ``None`` before its first use.
        capabilities: Result of the server capability probe, or ``None`` until it has run.
        binary_keys: ``(database, collection)`` pairs whose default collation has been read and found
            binary (``REQ-E-170``). Cached here rather than on the store for the reason ``capabilities``
            is: N stores over one collection share the answer, and a namespaced store is a second store
            over the same collection (``REQ-E-039``), so a per-instance cache would be a round trip per
            namespace. A collation cannot be altered in place, so the only way this cache goes stale is
            somebody dropping the collection and recreating it under the same name while the process
            runs, which is not a case worth a round trip per operation to catch.
    """

    # `repr=False`, and this is the field that needs it (``REQ-E-171``). A dataclass renders every field,
    # and `AsyncMongoClient.__repr__` renders every non-default option it was built with. pymongo redacts
    # the password in the URI it prints, which is what makes this look safe and is not the whole story:
    # `authMechanismProperties` carries `AWS_SESSION_TOKEN` and `tlsCertificateKeyFilePassword` is a
    # password by name, and neither is redacted. Both reached `repr(connection)` and, through the values
    # of the registry dict, `repr(_REGISTRY)` -- so a debugger frame, a logged local, or an exception
    # rendering its arguments published them. Reported by a production-readiness review; `AT-177` is the
    # gate. The client is reachable as an attribute, which is what the rest of this module uses; what it
    # is not any more is part of anything that renders itself by default.
    client: AsyncMongoClient[Any] = field(repr=False)
    owned: bool
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    loop: asyncio.AbstractEventLoop | None = None
    capabilities: Capabilities | None = None
    binary_keys: set[tuple[str, str]] = field(default_factory=set)

    def get_lock(self) -> asyncio.Lock:
        """Return a lock usable on the running event loop, replacing it if the loop changed.

        ``REQ-E-017``. A synchronous ``Agent`` runs each invocation on a fresh loop, and an
        ``asyncio.Lock`` raises ``RuntimeError: ... is bound to a different event loop`` once it has
        created a waiter future on one loop and is awaited on another.

        Two details make this worth a method rather than a comment. The binding happens **lazily**, on
        contention only, because ``acquire()`` on an unlocked lock returns without touching the loop:
        an uncontended lock crosses loops silently and hides the problem until load arrives. And the
        ``RuntimeError`` is raised *after* the lock marks itself held, so a lock that fails this way
        stays locked forever and every later invocation fails too.

        Replacing the lock is safe here because it only ever guards work scoped to a single loop. A
        lock discarded with waiters on it would strand them, but those waiters live on a loop that has
        already been closed.

        Returns:
            A lock bound to the running loop, stable for the lifetime of that loop.

        Raises:
            RuntimeError: If called with no running event loop, which is a caller bug: every user of
                this lock is a coroutine.
        """
        running = asyncio.get_running_loop()
        if self.loop is not running:
            self.lock = asyncio.Lock()
            self.loop = running
        return self.lock


#: A connection identity paired with the event loop that connection serves (``REQ-E-017a``). ``None``
#: means no loop: either a client built by synchronous code that has not bound itself yet, or an
#: injected one, which is never split per loop.
_RegistryKey = tuple[str, asyncio.AbstractEventLoop | None]

# Keyed by the digest from `_identity`, never by a connection string (REQ-E-014). Process-global, so
# every store built from one config shares one pool per event loop.
_REGISTRY: dict[_RegistryKey, _Connection] = {}

_REGISTRY_LOCK: Final = threading.RLock()
"""Guards every read, every mutation and every iteration of :data:`_REGISTRY` (``REQ-E-172``).

A **thread** lock rather than an ``asyncio`` one, because the hazard is not concurrency on one loop.
``resolve`` is synchronous and process-global state is per process, so two threads each running their own
loop -- which is what a synchronous ``Agent`` does, and what a thread pool of them does more of -- reach
this dict at the same time with no loop in common to serialise them. An ``asyncio.Lock`` would serialise
neither, and a coroutine cannot be required here anyway: ``resolve`` is called from
``MongoDBStorage.__init__``, which is not one.

Reentrant, because ``resolve`` holds it across a call to :func:`_register`, which needs it too. A plain
``Lock`` would deadlock that single thread against itself, and splitting the two functions to avoid the
nesting would put the check and the act on either side of a release, which is the defect this exists to
close.

**Never held across an ``await``.** :func:`close` takes it for the snapshot and again for each removal,
and releases it before waiting on ``client.close()``. Holding it across the suspension would block the
thread the loop runs on, so the coroutine that must release it could not be resumed by the loop that is
waiting for it -- a deadlock rather than a slow path, and in a teardown.

Reported by a production-readiness review, which reproduced the ``KeyError`` with overlapping event-loop
resolutions across threads. ``AT-178`` is the gate.
"""


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """Return the running event loop, or ``None`` when there is none.

    ``resolve`` is synchronous and is called from both worlds: from a coroutine, and from
    ``MongoDBStorage.__init__`` at module level. Only the first has a loop to key on.

    Returns:
        The running loop, or ``None`` outside one.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _reap_closed_loops() -> None:
    """Drop every registry entry whose event loop has closed (``REQ-E-017a``).

    Called before building a client, which is the only moment the registry can grow. Swept across the
    whole registry rather than one identity, because such an entry is unusable by anyone: pymongo
    refuses the client from any other loop, and we cannot close it either, since ``close()`` sends
    ``endSessions`` on the loop that is gone. Keeping it would leak an entry per loop for the life of
    the process.

    The client itself is dropped rather than closed, deliberately. It holds sockets that only its dead
    loop's selector can reach, so the file descriptors are released when the object is collected.

    Under :data:`_REGISTRY_LOCK`, and removing with ``pop(key, None)`` rather than ``del`` (``REQ-E-172``).
    Two things went wrong without it, and they need different halves of the fix. The snapshot iterated a
    dict another thread could be writing, which is a ``RuntimeError`` from CPython and needs the lock; and
    the removal named a key another thread's reap could already have taken, which is a ``KeyError`` and
    needs the default. Both surfaced in ``resolve``, on the caller's first use of a connection, having
    nothing to do with what that caller asked for. ``close()`` reached the second conclusion first, for
    ``REQ-E-017c``, from one loop rather than from two threads.
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


def _identity(config: MongoDBConfig) -> str:
    """Compute the registry key for a config.

    Two configs share a client when they would produce the same one. ``database`` is excluded, since
    one client serves any number of databases, which is the whole point of sharing it.

    An injected client keys on its own object identity rather than on anything about it. Two clients
    built from the same URI are not interchangeable: they can differ in credentials, read preference,
    or timeouts, and nothing here inspects those. Handing one caller the other's client would be worse
    than building two.

    ``client_args`` is part of the identity, because a client built with different timeouts is a
    different client. Without it, the second config's arguments would be silently discarded on the
    cache hit.

    The result is a digest, not the string it summarizes. ``REQ-E-014`` forbids emitting a connection
    string into any ``repr``, and a dict keyed on URIs leaks its keys through ``repr(registry)``, a
    ``KeyError``, and any debugger that walks module globals. A digest preserves the identity
    semantics ``REQ-E-012`` needs and is not reversible into a password.

    Args:
        config: The configuration to key.

    Returns:
        An opaque key, stable for the lifetime of the process.

    Raises:
        ValueError: If neither ``client`` nor ``connection_string`` is set (``REQ-E-015``).
    """
    injected = config.get("client")
    if injected is not None:
        return f"client:{id(injected):x}"

    connection_string = config.get("connection_string")
    if not connection_string:
        raise ValueError(_MISSING_CONNECTION)

    # `default=repr` so an unserializable argument (a `ServerApi`, an SSLContext) still produces a
    # stable key instead of raising. Distinct objects then key distinctly, which errs toward building
    # a second client rather than sharing the wrong one.
    args = json.dumps(config.get("client_args") or {}, sort_keys=True, default=repr)
    digest = hashlib.sha256(f"{connection_string}\x00{args}".encode()).hexdigest()
    return f"uri:{digest}"


def _warn_if_retries_disabled(client: AsyncMongoClient[Any]) -> None:
    """Warn when the resolved client will not retry writes (``REQ-E-016``).

    Read off ``client.options`` rather than parsed out of the connection string. The resolved option is
    the thing that matters, and reading it catches the setting whether it arrived through the URI,
    through ``client_args``, or on a client the caller injected. It also means this package ships no
    URI parser.

    Args:
        client: The client about to be registered.
    """
    if client.options.retry_writes is False:
        logger.warning(_RETRY_WRITES_DISABLED)


def _already_attributed(client: AsyncMongoClient[Any]) -> bool:
    """Whether this client's driver metadata already names Strands (``REQ-E-185``).

    The metadata's ``driver.name`` is a pipe-delimited chain -- ``PyMongo|c|async|Strands`` -- and
    ``append_metadata`` appends to it. Splitting on the delimiter and comparing whole entries
    case-insensitively is what pymongo's own ``PoolOptions._update_metadata`` does from 4.14.1 onward,
    and doing it here is what makes the answer independent of which pymongo is installed.

    Whole entries rather than a substring search, because a caller whose own attribution is
    ``StrandsToolkit`` has not been attributed as Strands and must not be treated as though they had.

    Args:
        client: The client to inspect. Read through ``options``, which is public and is what
            ``AT-013a`` already asserts against.

    Returns:
        Whether an entry equal to this package's driver name is already in the chain.
    """
    chain = client.options.pool_options.metadata["driver"]["name"]
    return _DRIVER_NAME.casefold() in (entry.casefold() for entry in chain.split("|"))


def _attribute_injected_client(client: AsyncMongoClient[Any]) -> None:
    """Attach Strands attribution to a client we did not build (``REQ-E-013``), at most once.

    This is ``pymongo_search_utils.append_client_metadata`` inlined. That function is three lines and
    would be the obvious thing to call, but it lives in the ``[memory]`` extra, and this module is
    reachable from ``storage.py`` under a ``[storage]``-only install.

    The call is made without probing for the method, because the declared pymongo floor is 4.14.0 and
    ``AsyncMongoClient.append_metadata`` arrived in that release. Below it the attribute still resolves
    -- ``__getattr__`` hands back a ``Database`` of that name -- so a probe would have to test
    callability rather than presence, and the branch behind it could only log. Attribution is
    bookkeeping this package does for MongoDB's own usage counting; a caller did not ask for it and has
    nothing to do about it, so the floor carries the requirement instead of a message at runtime.

    Args:
        client: The injected client to attribute in place.
    """
    # At most once per client, and this guard is the whole of that (``REQ-E-185``). A caller who closes
    # a store and resolves the same client again reaches this function again, because `close` releases
    # the registry slot the second `resolve` then misses. On pymongo 4.14.1 and later the second append
    # is a no-op inside the driver; on 4.14.0, which is this package's declared floor, it is not, and
    # the chain grows by one entry per cycle until the handshake exceeds the server's 512-byte limit on
    # client metadata and every new connection fails with an `OperationFailure` naming a document size.
    # Measured against mongod 8: dead at 22 cycles.
    if _already_attributed(client):
        return

    client.append_metadata(_driver_info())


def resolve(config: MongoDBConfig) -> _Connection:
    """Return the shared connection for a config and the running loop, creating it on first use.

    One pool per connection identity is ``REQ-E-012``, and ``REQ-E-017a`` bounds it per event loop:
    ``AsyncMongoClient`` binds to the loop of its first operation, so a single client cannot serve a
    synchronous ``Agent``, which runs every entry point on a loop of its own.

    Args:
        config: The configuration to resolve. Only the connection fields are read.

    Returns:
        The registered connection. The same object for every config with the same connection identity
        resolved on the same loop, so callers must not mutate it.

    Raises:
        ValueError: If neither ``client`` nor ``connection_string`` is set. Raised before any client
            is built, and nothing is registered (``REQ-E-015``). Nothing downstream would raise it for
            us: pymongo reads a schemeless string as a hostname, so the failure would otherwise be a
            server-selection timeout against ``localhost``.
    """
    identity = _identity(config)

    # One lock over the whole body, because every branch below is a check followed by an act on
    # process-global state and the two must not be separable by another thread (``REQ-E-172``). Nothing
    # in here does I/O: `AsyncMongoClient()` builds a pool lazily and connects on first use, and
    # `_identity` -- the one call that can raise -- is deliberately above the lock, so a config with no
    # way to reach a database is rejected without touching the registry at all (``REQ-E-015``).
    with _REGISTRY_LOCK:
        injected = config.get("client")
        if injected is not None:
            # REQ-E-017a's carve-out: one slot, keyed on no loop at all. We did not build this client, so
            # there is no second copy to build, and REQ-E-010 forbids closing the one we have. A caller
            # who drives an injected client from two loops gets pymongo's own RuntimeError, which names
            # the object they own.
            return _register(identity, None, injected, owned=False, attribute=True)

        running = _running_loop()
        existing = _REGISTRY.get((identity, running))
        if existing is not None:
            # Returning here is what keeps REQ-E-016's warning to one line per connection rather than one
            # per store, and a Storage is resolved once per SDK subsystem that consumes it. Reapplying the
            # attribution would be harmless too, but for our own reason rather than pymongo's: this comment
            # used to credit `_update_metadata`'s dedupe, which the declared floor 4.14.0 does not have.
            # `_already_attributed` is what makes it harmless on every supported version (``REQ-E-185``).
            return existing

        if running is not None:
            # Adoption (REQ-E-017a). A client built with no loop running has not bound itself to anything,
            # so the first loop that asks for it takes it over. Building a second here instead would leave
            # the first open, unused, and never closed, in the ordinary case of constructing a store at
            # import time and using it on one loop.
            adopted = _REGISTRY.pop((identity, None), None)
            if adopted is not None:
                _REGISTRY[(identity, running)] = adopted
                return adopted

        _reap_closed_loops()

        # `driver=` reaches pymongo through `**kwargs` rather than a named parameter, so nothing type
        # checks it. AT-013a asserts the resulting pool metadata instead of the call.
        client: AsyncMongoClient[Any] = AsyncMongoClient(
            config["connection_string"],
            driver=_driver_info(),
            **(config.get("client_args") or {}),
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
        owned: Whether we built it, which decides whether :func:`close` may close it.
        attribute: Whether to attach Strands attribution, which only an injected client needs.

    Returns:
        The registered connection, existing or new.
    """
    # Its own acquisition rather than a documented precondition, because this is also the function that
    # decides whether the warning happens: the check and the insert have to be one step for the "at most
    # once" in this docstring to hold across threads (``REQ-E-172``). :data:`_REGISTRY_LOCK` is reentrant,
    # so the nesting under ``resolve``, which is the only caller, costs nothing.
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get((identity, loop))
        if existing is not None:
            return existing
        if attribute:
            _attribute_injected_client(client)
        _warn_if_retries_disabled(client)
        _REGISTRY[(identity, loop)] = _Connection(client=client, owned=owned)
        return _REGISTRY[(identity, loop)]


async def close(config: MongoDBConfig) -> None:
    """Release the connection for a config, closing the client only if we built it.

    ``REQ-E-010`` and ``REQ-E-011``. In an agent process one ``AsyncMongoClient`` is typically shared
    across memory, storage and the application's own queries, so closing an injected one tears down a
    pool that unrelated subsystems are still using. The symptom appears somewhere else, later, which
    is what makes the reference implementation's unconditional ``self.client.close()`` costly.

    Idempotent, and safe on a config that was never resolved. This gets called from a ``finally`` and
    again from an ``atexit``.

    Every slot for the identity is released, not just the running loop's (``REQ-E-017a``). Nothing in
    the public API names a loop, so a caller who closes a config once and finds another loop's pool
    still registered has a leak no further call could reach.

    Args:
        config: The configuration whose connection to release.
    """
    try:
        identity = _identity(config)
    except ValueError:
        # Nothing was ever registered for a config this broken, so there is nothing to release and no
        # reason to raise from a teardown path.
        return

    running = asyncio.get_running_loop()
    with _REGISTRY_LOCK:
        # The snapshot is taken under the lock and the iteration below runs outside it, which is the only
        # arrangement available here (``REQ-E-172``): there is an `await` in the loop body, and holding a
        # thread lock across it would block the thread this loop runs on, so whoever has to release the
        # lock could never be resumed. A teardown path must not be able to deadlock.
        keys = [key for key in _REGISTRY if key[0] == identity]

    for key in keys:
        # Popped with a default because the snapshot above can go stale before this line reaches the end
        # of it (`REQ-E-017c`). There is an `await` inside this loop, so anything that removes a slot
        # gets a turn: a second concurrent `close()` for the same config, which is what three stores
        # closing under one `gather` do, or the reap in `resolve()`. Either way the slot is gone because
        # somebody released it, which is the outcome this call wanted.
        with _REGISTRY_LOCK:
            connection = _REGISTRY.pop(key, None)
        if connection is None or not connection.owned:
            continue
        loop = key[1]
        if loop is None or loop is running:
            await connection.client.close()
        elif not loop.is_closed():
            # Another live loop's client. Its `close()` sends `endSessions` over sockets that belong to
            # that loop, so it has to run there. Scheduled and not waited on: this loop may be the one
            # that loop is waiting for, and a teardown path must not be able to deadlock.
            asyncio.run_coroutine_threadsafe(connection.client.close(), loop)
        # A closed loop's client is dropped and not closed. Nothing can close it: `close()` needs the
        # loop the sockets belong to, and the `RuntimeError` pymongo raises instead is not a
        # `PyMongoError`, so it escapes the driver's own handling.
