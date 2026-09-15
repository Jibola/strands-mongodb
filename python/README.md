# strands-mongodb

MongoDB storage and memory backends for the [Strands Agents](https://strandsagents.com) SDK.

This is the Python half of the [strands-mongodb](../README.md) repository. One MongoDB deployment
sits behind both of the SDK's persistence extension points:

| Surface | Class | Module | Extra | SDK protocol | Serves |
|---|---|---|---|---|---|
| Storage | `MongoDBStorage` | `strands_mongodb.storage` | `[storage]` | `strands.storage.Storage` | Sessions, context offloading, memory persistence, context history, transcripts |
| Memory | `MongoDBMemoryStore` | `strands_mongodb.memory_store` | `[memory]` | `strands.memory` `MemoryStore` | Semantic recall over Atlas Vector Search, extraction writes |

Note that the memory module and its extra are spelled differently: `strands_mongodb.memory_store` is
installed by `strands-mongodb[memory]`.

Either class can be imported from the package root or from its own module:

<!-- run: unit -->
```python
from strands_mongodb import MongoDBStorage          # resolved on first access
from strands_mongodb.storage import MongoDBStorage  # identical class
```

The root import resolves the class lazily, so it imports only the module you name. On a
`[storage]`-only install, `import strands_mongodb` followed by `MongoDBStorage` never touches the
memory module and never needs `pymongo-search-utils`. The one exception is
`from strands_mongodb import *`, which walks `__all__` and therefore loads both.

The `Storage` protocol is the leverage point: the SDK accepts a single top-level `storage` on
`Agent`, and five subsystems inherit it. One correct `MongoDBStorage` therefore delivers session
management and context offloading without a separate session-manager class.

**Already using another MongoDB backend for Strands?** Three exist: the archived `mongodb_memory` tool
in `strands-agents-tools`, and the community `strands-mongodb-store` and `strands-mongodb-storage`
packages. [MIGRATION.md](MIGRATION.md) maps each one's settings and document fields onto this
package's, says which of the three need data copied and which do not, and covers the index-name
collision all three run into.

## Install

Pick the surface you need. Each extra pulls only its own dependencies.

<!-- skip: these reach PyPI for a package that is not published yet -->
```bash
pip install 'strands-mongodb[storage]'   # MongoDBStorage
pip install 'strands-mongodb[memory]'    # MongoDBMemoryStore
pip install 'strands-mongodb[all]'       # both
```

> **Quote the brackets.** In `zsh` (the default macOS shell) and `fish`, an unquoted
> `pip install strands-mongodb[storage]` fails with `no matches found` because the shell tries to
> glob the brackets. `bash` is unaffected. The quoted form above works in every shell.

A bare `pip install strands-mongodb` installs no runtime dependencies and gives you no working
backend. It is metadata only. The extra you choose is what selects the dependency set, so
`[storage]` never drags in vector-search machinery you are not using.

## Quickstart

Both snippets below are run by the test suite from this file, so they are the versions that work rather
than the versions that were written. Each reads its connection string from `MONGODB_URI` and its
database from `MONGODB_DATABASE`, defaulting to `agents`.

### Storage

<!-- run: contract -->
```python
import asyncio
import os

from strands_mongodb import MongoDBStorage
from strands_mongodb.types import MongoDBConfig

connection: MongoDBConfig = {
    "connection_string": os.environ["MONGODB_URI"],
    "database": os.environ.get("MONGODB_DATABASE", "agents"),
}


async def main() -> None:
    storage = MongoDBStorage(connection, collection="storage")
    try:
        await storage.write("notes/deployment", b"the cluster is in us-east-1")
        print(await storage.read("notes/deployment"))
        print(await storage.list("notes/"))
        await storage.delete("notes/deployment")
    finally:
        await storage.close()


asyncio.run(main())
```

Nothing has to be provisioned first. Storage needs no index the server does not already create, which
is the reason keys are stored as `_id`.

### Memory

This one needs an Atlas cluster with a registered embedding model, because it takes the default path
where the deployment embeds the text and you never handle a vector.

<!-- run: autoembed -->
```python
import asyncio
import os

from strands_mongodb import MongoDBMemoryStore
from strands_mongodb.types import MongoDBConfig

connection: MongoDBConfig = {
    "connection_string": os.environ["MONGODB_URI"],
    "database": os.environ.get("MONGODB_DATABASE", "agents"),
}


async def main() -> None:
    store = MongoDBMemoryStore(name="preferences", config=connection, scope="user-abc", writable=True)
    try:
        await store.initialize()  # creates the indexes; add() and search() call it too
        await store.add("Prefers metric units and a 24-hour clock.")

        # Atlas embeds a new document after the write is acknowledged, around 15 seconds on the cluster
        # this was measured against, so a search that runs immediately finds nothing rather than failing.
        for _ in range(30):
            found = await store.search("how should times be displayed for this person?")
            if found:
                break
            await asyncio.sleep(2)
        else:
            raise SystemExit("Atlas has not embedded the memory yet. Run this again.")

        for entry in found:
            print(entry.metadata["_relevance_score"], entry.content)
    finally:
        await store.close()


asyncio.run(main())
```

The query shares no word with the stored fact. That is the point of the vector tier, and it is also why
the first run of this snippet is slower than the rest: `initialize()` waits for a new Atlas Search index
to finish building before it returns.

To give an agent both surfaces, pass one config to both and let the SDK wire them:

<!-- skip: needs a model provider and its credentials, which no test tier configures -->
```python
from strands import Agent
from strands.memory import MemoryManager

store = MongoDBMemoryStore(name="preferences", config=connection, scope="user-abc")
agent = Agent(
    model=your_model,
    storage=MongoDBStorage(connection),
    memory_manager=MemoryManager(stores=[store]),
)
```

One `storage` on `Agent` serves five subsystems, so this is also what turns on session persistence and
context offloading.

A type checker will reject `stores=[store]`, and the reason is the SDK's rather than yours: its
`MemoryStore` protocol declares `add_messages` as a member while its own docstring calls it optional, so
any store that omits it fails the assignment. `MongoDBMemoryStore` omits it on purpose, because
`MemoryManager` feature-detects the method and falls back to its own extractor, which dedupes.
`examples/memory_recall.py` carries the one-line `cast` and the full explanation.

## Connecting to a deployment

`MongoDBConfig` takes either a client you built or a connection string, and it needs one of the two. In
preference order:

| How | Key | What it costs you |
|---|---|---|
| Inject a client | `client` | Nothing. The credential stays inside an object you own, and you get every driver option, including ones this package never exposes. |
| A connection string with a passwordless mechanism | `connection_string` | Your deployment has to be set up for X.509, AWS IAM, OIDC or Kerberos. |
| A connection string with a password | `connection_string` | The password is now a value in a plain dict. |

That order is a security order, not a style one. `MongoDBConfig` is a `TypedDict`, so it is a dict at
runtime, and a dict ends up in tracebacks, log records and debugger frames. An injected client keeps the
credential out of it:

<!-- run: contract -->
```python
import asyncio
import os

from pymongo import AsyncMongoClient

from strands_mongodb import MongoDBStorage
from strands_mongodb.types import MongoDBConfig

client = AsyncMongoClient(os.environ["MONGODB_URI"])
connection: MongoDBConfig = {
    "client": client,
    "database": os.environ.get("MONGODB_DATABASE", "agents"),
}


async def main() -> None:
    storage = MongoDBStorage(connection)
    try:
        await storage.write("notes/injected", b"the credential never entered the config dict")
        await storage.delete("notes/injected")
    finally:
        # `storage.close()` deliberately does nothing to an injected client. It is yours, and something
        # else in the process may still be using it.
        await client.close()


asyncio.run(main())
```

An injected client also carries its own event loop affinity, which matters if you build one per request.
`client_args` covers the middle ground: this package passes it to the `AsyncMongoClient` it builds from
your connection string, so TLS settings, timeouts and read preferences all reach the driver without an
injected client.

### Authentication mechanisms

The driver decides what is available here, not this package. `pymongo` 4.17.0 declares `MONGODB-AWS`,
`MONGODB-OIDC`, `MONGODB-X509`, `GSSAPI`, `PLAIN`, `SCRAM-SHA-1` and `SCRAM-SHA-256`, and any of them
works through either route above: as `authMechanism` in the connection string, or as a keyword to the
client you inject. Whether your deployment accepts one is a question for the deployment.

One of them needs a package neither of this package's extras installs. `MONGODB-AWS` requires
`pymongo-auth-aws`, which arrives with `pip install 'pymongo[aws]'`, and the driver does not tell you
until it tries to authenticate: the failure is a `ConfigurationError` on the first connection, not an
`ImportError` at startup, so a misconfigured process starts cleanly and fails on its first operation.

## Configuration reference

Three `TypedDict`s, and every key of all three is listed here. A test compares these tables against the
declarations, so a key added to the package and not to this file fails the build.

### `MongoDBConfig`

The connection half. Build one per process and share it: two stores holding the same config object share
one connection pool.

<!-- keys: MongoDBConfig -->
| Key | Type | Required | Default | What it does |
|---|---|---|---|---|
| `database` | `str` | yes | | Database holding every collection this config reaches. |
| `client` | `AsyncMongoClient` | no | | A client you built. Preferred over `connection_string`, and this package never closes it. |
| `connection_string` | `str` | no | | A MongoDB URI. Required unless `client` is set; supplying neither raises. |
| `client_args` | `dict[str, Any]` | no | `{}` | Extra keyword arguments for the client this package builds. Ignored when `client` is set. |
| `ensure_indexes` | `bool` | no | `True` | Whether to create missing indexes. `False` verifies and reports instead. See below. |
| `index_ready_timeout` | `float` | no | `300` | Seconds to wait for a search index to become queryable, in total across every index one call builds. Raise it on a shared deployment where a build queues behind somebody else's. |
| `search_type` | `"auto" \| "vector" \| "text"` | no | `"auto"` | Which memory search tiers to prepare and query. |
| `embedding_model` | `str` | no | `voyage-4-lite` | Atlas embedding model, on the AutoEmbeddings path where the deployment embeds your text. |
| `embedding_dimensions` | `int` | no | | Vector length, on the manual path where you supply the vectors. Mutually exclusive with `embedding_model`, and requires `similarity`. |
| `similarity` | `"cosine" \| "dotProduct" \| "euclidean"` | no | | Distance metric, on the manual path. Requires `embedding_dimensions`. No default: the right metric is a property of your embeddings. |
| `filter_fields` | `list[str]` | no | `[]` | Extra document paths a caller may filter on. `scope` and `store_name` are always declared. |
| `vector_index_name` | `str` | no | `vector_index` | Name of the Atlas Vector Search index. |
| `text_index_name` | `str` | no | `text_index` | Name of the Atlas Search full-text index. |

### `MongoDBMemoryStoreConfig`

The keyword arguments of `MongoDBMemoryStore`. The first five come from the SDK's `MemoryStoreConfig`
and behave as they do for any store, with one exception noted below.

<!-- keys: MongoDBMemoryStoreConfig -->
| Key | Type | Required | Default | What it does |
|---|---|---|---|---|
| `name` | `str` | yes | | Identifies the store. Stamped on every document as `store_name` and part of every read predicate. |
| `description` | `str` | no | | Passed to the SDK, which shows it to the model when it chooses a store. |
| `max_search_results` | `int` | no | `10` | Result ceiling. A per-call `limit` overrides it. Below 1 raises. |
| `writable` | `bool` | no | `False` | Gates `add`, `delete` and `delete_all`. `False` here, unlike `FileMemoryStore`: the cost of the wrong default is a retrieval store that starts accepting model-written memories. |
| `extraction` | `ExtractionConfig \| bool` | no | | Passed to the SDK's extractor. This package never extracts anything itself. |
| `config` | `MongoDBConfig` | yes | | The connection, nested. Share one across stores. |
| `collection` | `str` | no | `memory` | Collection holding this store's documents. |
| `scope` | `str` | no | a sentinel | Tenant boundary. Stamped on writes and applied to reads by this package, never by the caller. Absence is a distinct scope, never "everything". |
| `filter` | `dict[str, Any]` | no | | Extra search predicate, composed as a narrowing conjunction. It cannot widen what this store can read. Paths must appear in `filter_fields`. |
| `min_score` | `float` | no | | Relevance floor. Applied after the search, so it drops results without changing their order. |
| `max_score` | `float` | no | | Relevance ceiling. Mutually exclusive with `min_score`. |

### `MongoDBSearchQuery`

Passed to `search()` instead of a plain string, when a per-query filter, limit or embedding is needed.
Every key is optional, but exactly one of `text` and `vector` has to be set.

<!-- keys: MongoDBSearchQuery -->
| Key | Type | Required | Default | What it does |
|---|---|---|---|---|
| `text` | `str` | no | | The query text. What `search("...")` fills in. |
| `vector` | `list[float]` | no | | A query embedding, on the manual-vector path. Mutually exclusive with `text`. |
| `filter` | `dict[str, Any]` | no | | Narrows this query only. Same rules as the store's `filter`. |
| `limit` | `int` | no | `10` | Result ceiling for this query. Outranks both `max_search_results` settings. |
| `oversampling_factor` | `int` | no | `10` | Multiplier turning the limit into a `$vectorSearch` candidate count. |

## Search: what answers a query

`search()` picks a tier from the query's **shape**, not from what you asked for, and reports which one
ran in the `_search_type` metadata key. Two tiers exist in `0.1.0`:

| Tier | Runs | Needs |
|---|---|---|
| `vector` | `$vectorSearch` against `vector_index` | Atlas, or an Atlas CLI local deployment, plus a registered embedding model on the AutoEmbeddings path |
| `text` | `$search` against `text_index` | Atlas Search, and no embedding model |

`search_type` decides which of them `initialize()` prepares. `"auto"` prepares both and prefers `vector`;
naming one prepares only that one, so a `text` store never builds a vector index it will never query, and
a `vector` store fails rather than quietly answering a query lexically.

Preparation can come up short without failing. A deployment with no search process at all gets its
ordinary indexes and a warning, and the search call is what raises. A cluster with Atlas Search but no
registered embedding model rejects the `autoEmbed` index at creation, and a store configured for `"auto"`
falls back to the text tier and says so in a warning. Both of those are visible only in your logs, which
is worth knowing before you conclude that recall is bad.

`hybrid` and `regex` are not here. `SearchType` is a `Literal` of the values that actually run, so a type
checker rejects them today rather than a runtime raising on them later.

### Supplying your own vectors

Setting `embedding_dimensions` and `similarity` selects the manual path, where the vector index covers an
`embedding` field on each document instead of asking Atlas to embed `content`.

<!-- manual-vectors -->
On that path this package indexes the `embedding` field and queries it, and **never produces a vector to
put in it**. Client-side embedding is a non-goal, so nothing here calls a model. `add()` takes content and
metadata and has no vector parameter, and `MemoryManager` would not pass one if it had: a memory written
through `MongoDBMemoryStore` carries no `embedding`, and the store's own vector index cannot retrieve it.
Writing that field is yours, through the driver, against the same collection:

<!-- skip: it needs a collection, an embedding model and a document that the reader supplies -->
```python
await collection.update_one(
    {"_id": entry_id},
    {"$set": {"embedding": await your_embedding_model.embed(text)}},
)
```

So in `0.1.0` the manual path is a `MongoDBStorage` and pipeline capability rather than a supported
`MongoDBMemoryStore` configuration: the read half works and the write half is the caller's. The
AutoEmbeddings default is unaffected, and is the supported memory configuration — there mongot embeds the
indexed `content` field and the document carries no vector by design.

### Reserved metadata keys

Every returned entry carries four keys this package owns in its `metadata`, because the SDK's
`MemoryEntry` has nowhere else to put them:

| Key | Holds |
|---|---|
| `_relevance_score` | The score from whichever tier answered |
| `_id` | The document id, which is also the `memory_id` that `delete()` takes |
| `_search_type` | `vector` or `text`, whichever ran |
| `_scope` | The scope of the store that answered, read from the store and never from the document |

Your own metadata merges underneath these. On a name collision the reserved value wins and the collision
is logged, so a key of yours is never silently dropped without a record.

## Tenancy

Three boundaries are available, and they are not equivalent. The difference that matters is who enforces
them:

| Boundary | Configure | Enforced by |
|---|---|---|
| Database per tenant | `database` | MongoDB, if the tenant's user is scoped to that database |
| Collection per tenant | `collection`, or `prefix` on storage | MongoDB, if privileges are per collection |
| Scope inside one collection | `scope` on memory, `namespace()` on storage | This package |

The third one is the cheapest and the one to understand before choosing it. MongoDB applies no
server-side row filter, so a memory lands in the right scope because `add` put it there, and comes back
because `search` asked for it. The package holds that line in four places: it stamps `scope` itself and a
caller cannot set it through `filter` or `metadata`; an absent scope resolves to a distinct sentinel
rather than to no predicate; a caller `filter` composes as a narrowing conjunction and cannot widen the
read; and `delete()` reaches exactly as far as `search()` does. Those are `REQ-E-046` through
`REQ-E-051`, and the tests for them run two stores in one collection and check both directions.

If a tenant boundary is a compliance boundary, use one of the first two rows.

## Erasing memories

`MongoDBMemoryStore` deletes as well as writes, which is what an erasure request against stored facts
needs from a memory backend. The SDK's `MemoryStore` protocol has no deletion member, so these are the
package's own surface rather than a protocol method.

| Call | Removes | Returns |
|---|---|---|
| `await store.delete(memory_id)` | The one memory with that `_id`, if this store can see it | `1` or `0` |
| `await store.delete_all()` | Every memory this store can see | The count |

`memory_id` is the `_id` string a search result carries in its `metadata`, so the id you delete by is
the id you were handed. Anything that is not a 24-character hex ObjectId is rejected before a network
call.

Both calls require `writable: True`, and both reach exactly as far as `search()` does: one scope, this
store's name, narrowed by any configured `filter`. A store cannot delete what it cannot read. Two
consequences are worth stating rather than leaving to be discovered:

- **Erasing a scope takes one call per store in it.** `delete_all()` empties one store's view, not the
  whole scope, because `store_name` is part of the predicate. Whether a scope-wide erasure should be a
  single call is still open.
- **`0` does not mean "no such memory."** A well-formed id gets `0` whether it names a memory already
  deleted, one in another scope, or one in another store in this scope. Telling those apart would
  answer whether an id exists somewhere the caller cannot read.

## Indexes and privileges

`initialize()` creates what it needs by default, which is convenient in development and often the wrong
thing in production, where index creation is a DBA's call and the application's credential is not
supposed to hold `createIndex`. Set `ensure_indexes: False` and this section is the handoff.

### What gets created

Six indexes across two collections. Names are prefixed `strands_` so an operator can tell ours from
theirs, and they are stable: renaming one would be a breaking change.

| Collection | Index | Keys | Why |
|---|---|---|---|
| `storage` | `strands_expires_at_ttl` | `expires_at` | Expires `ttl_seconds` documents. `MongoDBStorage` never creates it as a side effect of a write: call `ensure_indexes()`. |
| `memory` | `strands_scope_store_content_hash_unique` | `scope`, `store_name`, `content_hash` | Makes a duplicate `add` a no-op instead of a second copy. Partial, so documents with no hash are unaffected. |
| `memory` | `strands_scope_store_created_at` | `scope`, `store_name`, `created_at` desc | Every non-search read is scope plus store plus recency. |
| `memory` | `strands_expires_at_ttl` | `expires_at` | Expires memories with a TTL. |
| `memory` | `vector_index` | `content` as `autoEmbed`, `scope` and `store_name` as filters | The vector tier. Atlas only. Name from `vector_index_name`. |
| `memory` | `text_index` | `content` as a string field | The text tier. Atlas Search only. Name from `text_index_name`. |

The two search indexes shown are the zero-configuration shape, where Atlas embeds your text. Setting
`embedding_dimensions` and `similarity` instead swaps the `autoEmbed` field for a `vector` field of that
width, and adding `filter_fields` appends one filter entry per path.

### Creating them by hand

Paste into `mongosh`, connected to the database in your `MongoDBConfig`. This block is generated from
the same code that runs at `initialize()` and a test asserts they are identical line for line, so it is
worth pasting rather than retyping:

<!-- run: mongosh -->
```javascript
db.runCommand({"createIndexes": "storage", "indexes": [{"key": {"expires_at": 1}, "name": "strands_expires_at_ttl", "expireAfterSeconds": 0}]})
db.runCommand({"createIndexes": "memory", "indexes": [{"key": {"scope": 1, "store_name": 1, "content_hash": 1}, "name": "strands_scope_store_content_hash_unique", "unique": true, "partialFilterExpression": {"content_hash": {"$exists": true}}}]})
db.runCommand({"createIndexes": "memory", "indexes": [{"key": {"scope": 1, "store_name": 1, "created_at": -1}, "name": "strands_scope_store_created_at"}]})
db.runCommand({"createIndexes": "memory", "indexes": [{"key": {"expires_at": 1}, "name": "strands_expires_at_ttl", "expireAfterSeconds": 0}]})
db.runCommand({"createSearchIndexes": "memory", "indexes": [{"name": "vector_index", "type": "vectorSearch", "definition": {"fields": [{"type": "autoEmbed", "path": "content", "model": "voyage-4-lite", "modality": "text"}, {"type": "filter", "path": "scope"}, {"type": "filter", "path": "store_name"}]}}]})
db.runCommand({"createSearchIndexes": "memory", "indexes": [{"name": "text_index", "type": "search", "definition": {"mappings": {"dynamic": false, "fields": {"content": [{"type": "string"}]}}}}]})
```

Substitute your collection names if you set `collection`. The first four are ordinary indexes and any
MongoDB accepts them; the last two need Atlas Search, and the vector one additionally needs an embedding
model registered on the cluster.

### Ensure versus verify

`ensure_indexes: False` is verify-only rather than skip. A credential with no DDL rights still needs to
know whether the indexes are there, and what happens when one is absent depends on which index it is:

<!-- verify -->
| Index kind | `True` (default) does | `False`, and it is missing |
|---|---|---|
| Ordinary and TTL | Creates what is missing, one `createIndexes` call per collection | Raises. The message is the list of `createIndexes` commands that fix it, one line each |
| Search, under `search_type: "auto"` | Creates it, then polls until it is queryable | A warning naming the index, and that tier is dropped. The store starts and answers from the tier that is left; if neither is left it raises |
| Search, under `search_type: "vector"` or `"text"` | Creates it, then polls until it is queryable | Raises. There is no second tier to fall back to, and answering from one you did not ask for changes what matches |

So verify mode is not a quiet mode. A missing ordinary index stops startup with a work order, because the
alternative is worse than a late start: the uniqueness index behind `add`'s deduplication is one of them,
and an application that runs without it writes duplicates silently and surfaces the problem in the data
weeks later. The one case that continues is a search index under `auto`, where the package has somewhere
to fall back to and a store answering lexically is worth more than a store that will not start. That one
is visible only in your logs, so read the warnings.

`MongoDBStorage` follows the same rule through `ensure_indexes()`, which is explicit rather than part of
any operation: its collection has one ordinary index and no search index, so `ensure_indexes: False`
there raises with the command for `strands_expires_at_ttl` and creates nothing.

Waiting is bounded. `initialize()` polls a new search index every 0.5 s for up to 300 s, and past that it
warns and returns rather than blocking your startup forever.

### Privileges

Measured on Atlas 8.0.30 against the built-in roles. It was assumed that a `readWrite` user cannot create
indexes; that was wrong on this server, and a test rechecks the table against whatever deployment you
point the contract tier at, because role definitions belong to the server and can change.

<!-- privileges -->
| Action | This package needs it for | `read` | `readWrite` |
|---|---|---|---|
| `find` | Every read, on both surfaces | yes | yes |
| `insert` | `write`, `add` | no | yes |
| `update` | `write` overwriting an existing key | no | yes |
| `remove` | `delete`, `delete_all` | no | yes |
| `listIndexes` | Verify mode, ordinary indexes | yes | yes |
| `listSearchIndexes` | Verify mode, search indexes | yes | yes |
| `createIndex` | Ensure mode, ordinary indexes | no | yes |
| `createCollection` | Ensure mode, the namespace a search index needs | no | yes |
| `createSearchIndexes` | Ensure mode, search indexes | no | yes |

So `readWrite` covers everything including index creation, and `read` is enough for a search-only store
in verify mode. The interesting case is the one in between: an application that writes memories but must
not create indexes. No built-in role expresses that, so it takes a custom role granting the four data
actions and the two `list` ones, and `ensure_indexes: False`. Keep `insert` in that role even though this
package never sends an `insert` command: every write it makes is an upsert, and a role holding `update`
without `insert` is refused one.

Self-managed MongoDB reads the same. Measured on 6.0.28, 7.0.40 and 8.3.8 started with `--auth`, none of
them running a search process: all eight rows above are identical to Atlas, `createSearchIndexes` included.
A role says what the server would authorize, not what it can do, so `readWrite` carries that action on a
deployment with no search at all.

Two notes for running as such a user. `rolesInfo`, which the test above uses to read the table back, needs
`viewRole`, and no data-access role grants it, so that one test skips under a least-privilege credential
instead of running. And `dropDatabase` is in neither role, so the test suite cannot drop the throwaway
databases it creates; it warns and leaves them. No code path in the package drops a database, and erasure
uses `remove`, which `readWrite` grants.

## Limits

Everything a caller runs into that this package does not remove. Each row says what you can do about it,
and the last column names the requirement or the id where the detail lives.

<!-- limits -->
| Limit | Where it bites, and what you can do | Source |
|---|---|---|
| One client per event loop | A new loop means a new connection and a new handshake. A synchronous `Agent` calling `invoke()` in a loop pays that per call. Drive the package from one loop, through the SDK's `invoke_async`. | `REQ-E-017a` |
| The connection resolves per operation | An operation after `close()` opens a fresh pool rather than raising, so a mistaken `close()` costs a connection instead of an error. Call `close()` once, at process shutdown. | `REQ-E-017b` |
| `close()` is scoped to the config | Closing one store's connection closes the pool that every store built from the same config is sharing. Siblings with an operation in flight are the ones that notice. | `REQ-E-017c` |
| A memory is not searchable when `add()` returns | Atlas embeds and indexes asynchronously, about 15 s on the cluster this was developed against. Poll for the memory rather than reading straight back. | `REQ-E-091` |
| An empty search result is ambiguous | `MemoryManager` turns a failed store into a warning and returns what survived, so zero hits can mean zero matches or a search that failed. Watch for `store search failed` in your logs. | `store search failed` |
| Document size | 16 MB, the server's `BSON` ceiling. Checked before the write, so you get a `ValueError` naming the size instead of a driver error. Chunk, or store a pointer. | `MAX_DOCUMENT_BYTES` |
| Storage key length | 1024 bytes, which keeps a key inside an index entry. Same early `ValueError`. | `MAX_KEY_BYTES` |
| `list()` is unpaginated | The SDK's protocol takes a prefix and no cursor, so a prefix matching a million keys returns a million keys. Use narrower prefixes, or `namespace()`. | `Storage.list()` |
| `ttl_seconds` needs an index nothing creates implicitly | `write` stamps `expires_at`, and the documents stay until the storage collection has its TTL index. No ordinary operation creates it, so the documents stay forever if nothing does. Call `await storage.ensure_indexes()` once at startup, or run the command above. | `MongoDBStorage.ensure_indexes()` |
| Vector dimensions | 8192 on the manual vector path, rejected at configuration time rather than by the server. | `MAX_DIMENSIONS` |
| Results per search | 10 by default. Raise `max_search_results` on the store, or `limit` on a single query. | `DEFAULT_MAX_SEARCH_RESULTS` |
| Vector candidates considered | 10 per result requested. Raise `oversampling_factor` for recall, lower it for latency. | `DEFAULT_OVERSAMPLING_FACTOR` |
| Waiting for a search index | 300 s, then `initialize()` warns and returns rather than blocking your startup. The index keeps building; the first queries are the ones that miss. Set `index_ready_timeout` to change it. | `READY_TIMEOUT_SECONDS` |

## Examples

Two runnable scripts in [`examples/`](examples/README.md), each split into two commands you run as two
processes, so the only thing crossing between them is MongoDB:

<!-- skip: the examples are run by their own tests, which supply the second command and the backend -->
```bash
python examples/memory_recall.py write     # then: recall
python examples/unified_storage.py run     # then: resume
```

`memory_recall.py` stores a fact in one process and recalls it in the next from a query that shares no
word with it, over server-side embeddings. `unified_storage.py` has three SDK subsystems writing to one
collection under three key prefixes, then restores the session in a fresh interpreter.

`MONGODB_URI` is the only variable either one needs, and `--dry-run` prints what a step would do without
connecting to anything.

## Requirements

| | |
|---|---|
| Python | >= 3.10 |
| `strands-agents` | >= 1.48.0, the release that introduced the unified `Storage` interface |
| MongoDB | Any deployment serves storage. The text tier needs Atlas Search; the vector tier needs Atlas or an Atlas CLI local deployment. The default embedding path also needs a model registered on the cluster, measured present on Atlas and absent on a local `mongot` 8.3.8. |

## Development

<!-- skip: running this would install into whatever interpreter the test suite happens to be using -->
```bash
pip install -e '.[all,dev]'

hatch run test          # unit tier, no MongoDB needed
hatch run lint
hatch run typecheck
hatch run prepare       # format, lint, typecheck, test
```

The test suite is tiered by what infrastructure it needs. Tiers whose backend is absent are
skipped with a reason, never silently passed:

| Tier | Marker | Backend | Gate |
|---|---|---|---|
| Unit | none | Mocked `AsyncMongoClient` | Always runs |
| Contract | `contract` | Any real MongoDB | `MONGODB_URI` |
| Integration | `integration` | Atlas or Atlas CLI local, with `mongot` | `MONGODB_ATLAS_URI` |
| AutoEmbeddings | `autoembed` | Cluster with a registered Atlas embedding model | `MONGODB_AUTOEMBED_URI` |
| Compatibility | `compat` | Latest published `strands-agents`, not the pinned one | CI cron only |

Two markers mean two requirements, not either one: an `autoembed` test also carries `integration`
and needs both variables set.

Those three variables are also read from a `.env` file, in `python/` first and then the repository
root, whichever exists. Only those three keys are read, and a value already set in the environment is
never replaced, so CI behaves as if the file were absent. Both paths are `.gitignore`d.

<!-- skip: each line is a whole test run, and one of them is the run reading this line -->
```bash
hatch run test                                  # unit only
MONGODB_URI=mongodb://localhost:27017 hatch run test
hatch run cov                                   # release gate: 90% aggregate, 70% per module

# Opt out of a configured backend: empty, not unset. `env -u` is refilled from `.env`.
MONGODB_URI= MONGODB_ATLAS_URI= MONGODB_AUTOEMBED_URI= hatch run test
```

Pointing all three variables at one Atlas cluster runs every tier and covers less than three
deployments do. Some tests assert what happens where a capability is *missing*, and they skip with a
reason on a cluster that has it. For the widest coverage:

| Variable | Deployment | Covers only there |
|---|---|---|
| `MONGODB_URI` | plain `mongod`, no `mongot` | A search query against a server with no search process |
| `MONGODB_ATLAS_URI` | Atlas CLI local, `mongot`, no registered model | An `autoEmbed` index rejected at creation, and the degraded default that follows |
| `MONGODB_AUTOEMBED_URI` | Atlas cluster with a registered model | The default embedding mode, end to end |

Each test session writes to a database named `strands_mongodb_test_<random>` and drops it at the
end, so pointing `MONGODB_URI` at a deployment you care about is survivable but still not advised.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
