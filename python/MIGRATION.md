# Migrating to `strands-mongodb`

Three packages already put MongoDB behind the Strands Agents SDK. This guide covers all three: what each
setting becomes, what happens to documents you have already written, and the two places where the change
is not a configuration edit.

Start with the table. One of these migrations costs a configuration change and nothing else. The other two
copy documents, because the read predicate this package uses names fields those documents do not carry.

<!-- verdict -->
| Coming from | Does migrating move data? | What you do |
|---|---|---|
| `strands-mongodb-storage` | **No.** Both packages write the same `_id` and the same `data` field, so this package reads documents that package wrote | Point `MongoDBConfig` at the database and collection you already use |
| `strands-tools` `mongodb_memory` | **Yes**, and the agent's wiring changes shape too. A tool the model called becomes a store the SDK drives | Rewire the agent, then copy memories with a transformation |
| `strands-mongodb-store` | **Yes.** Documents need three fields added and one retyped | Copy memories with a transformation |

## Which package you have

<!-- sources -->
| Package | What it gave you | This guide read it from | On |
|---|---|---|---|
| `strands-tools` `mongodb_memory` | A `@tool` the model calls with an `action` string, on Atlas Vector Search with client-side Bedrock embeddings | `tools/src/strands_tools/mongodb_memory.py` | Read from the repository, which is the record |
| `strands-mongodb-store` 0.2.0 | A `MemoryStore` on Atlas AutoEmbeddings, with `search` and `add` | `https://raw.githubusercontent.com/skamalj/strands-agents-session/main/memory/mongodb/src/strands_mongodb_store/__init__.py` | 2026-09-01 |
| `strands-mongodb-storage` 0.1.0 | A `Storage` with `write`, `read`, `delete` and `list` | `https://raw.githubusercontent.com/skamalj/strands-agents-session/main/storage/mongodb/src/strands_mongodb_storage/__init__.py` | 2026-09-01 |

The first of those lives in the same workspace as this package, so the tests read the module itself and
fail when a setting is added to it. The other two are **not vendored** into this repository and are **not
fetched** when the tests run, so what you see below is a transcript taken on the date in the last column.
[What this guide cannot check](#what-this-guide-cannot-check) says what follows from that.

## From `strands-mongodb-storage`

**Nothing happens to your documents.** That package writes `{"_id": <prefix><normalized key>, "data":
Binary(...)}` and nothing else. This package writes the same `_id` and the same `data`, plus `updated_at`,
which their reader takes `doc["data"]` and never looks at. Both packages normalize a key the same way:
collapse runs of `/`, strip the ends, reject an empty key, reject a `..` segment. This package imports the
SDK's `_normalize_key` and `_normalize_prefix` (`storage.py:39`); theirs inlines a character-for-character
copy of both (`__init__.py:23` and `:32`). Both build the prefix as
`_normalize_prefix(prefix).rstrip("/")` with one `/` re-added, so `tenant-a`, `tenant-a/`, `/tenant-a` and
`tenant-a//` all reach the same key space in either. The same `write("sessions/abc/state.json")` therefore
produces the same `_id`.

The copy is worth knowing about rather than glossing over: it means the two agree today because their copy
matches the SDK release it was taken from, not because anything keeps them in step. If you are cutting over
gradually with both packages writing, check that their copy still matches the SDK you have installed.

So the migration is a configuration change. This runs against a real deployment on every CI pass, and it
writes the old package's document shape with the raw driver before reading it back through this one:

<!-- run: contract -->
```python
import asyncio
import os

from bson.binary import Binary
from pymongo import AsyncMongoClient

from strands_mongodb import MongoDBStorage
from strands_mongodb.types import MongoDBConfig

DATABASE = os.environ.get("MONGODB_DATABASE", "agents")
KEY = "sessions/abc/state.json"

connection: MongoDBConfig = {
    "connection_string": os.environ["MONGODB_URI"],
    "database": DATABASE,
}


async def main() -> None:
    # Exactly what strands-mongodb-storage 0.1.0 writes: the normalized key as `_id`, the payload as
    # `data`, no other field.
    client = AsyncMongoClient(os.environ["MONGODB_URI"])
    try:
        await client[DATABASE]["storage"].replace_one(
            {"_id": KEY}, {"_id": KEY, "data": Binary(b'{"turns": 3}')}, upsert=True
        )
    finally:
        await client.close()

    storage = MongoDBStorage(connection, collection="storage")
    try:
        print(await storage.read(KEY))
        print(await storage.list("sessions/"))
        await storage.delete(KEY)
    finally:
        await storage.close()


asyncio.run(main())
```

### Settings

<!-- settings: community-storage -->
| Theirs | Here | Note |
|---|---|---|
| `connection_string` (default `mongodb://127.0.0.1:27017/`) | `MongoDBConfig["connection_string"]` | No default here. A localhost default is convenient once and confusing afterwards |
| `database_name` (default `strands_storage`) | `MongoDBConfig["database"]`, **required** | The one setting that changes from a default to a value you write down. If you never set theirs, the answer is `strands_storage` |
| `collection_name` (default `storage`) | `MongoDBStorage(..., collection="storage")` | Same default, so nothing to change |
| `client` | `MongoDBConfig["client"]` | Same idea, and preferred here for the same reason: a config dict prints in a traceback and a connection string carries a password. Theirs takes a synchronous `MongoClient`; this package takes an `AsyncMongoClient` |
| `prefix` | `MongoDBStorage(..., prefix=...)` | Identical normalization, so an existing prefix keeps addressing the same documents |

### Documents

<!-- fields: community-storage -->
| Their field | What happens to it |
|---|---|
| `_id` | Read as is. Same key convention, same normalization, same prefix handling |
| `data` | Read as is. `read()` does `bytes(document["data"])`, so a `Binary` and a raw `bytes` both work |

Two things this package adds to documents it writes itself, neither of which affects reading theirs:
`updated_at`, a BSON date stamped on every write, and `expires_at`, present only when the instance was
built with `ttl_seconds`. Their reader ignores both.

Three differences arrive with the move, and they change what the old documents can do rather than what they
are. `search()` exists here and does not exist there: the `Storage` protocol has five methods and that
package implements four, so an SDK strategy that searches storage has nothing to call. `list()` here is a
bounded range over the `_id` index, sorted by the server; theirs is `{"_id": {"$regex": "^..."}}` over the
collection, sorted in Python afterwards. Both return every matching key. And a write here is checked
against a 1024-byte key ceiling and MongoDB's 16 MB document limit before it is issued, so an oversized
value fails with a message naming the limit rather than as a driver error.

## From `strands-tools` `mongodb_memory`

### The wiring changes before any data does

This is the part a field mapping will not tell you. The archived plugin is a single `@tool` that the model
calls with an `action` string, one of `record`, `retrieve`, `list`, `get` or `delete`. The model decided when
to store something and when to recall it, and your agent's job was to hand the tool over.

`MongoDBMemoryStore` is not a tool. It is a store that the SDK's `MemoryManager` reads and writes: the
manager decides when to extract facts from a conversation and when to search, and it exposes its own tools to
the model. So you stop passing a tool and start passing a store, and the tools the model sees change name
and shape with it. Map the fields first if you like, but nothing will call them until the wiring moves.

If you want the model to keep deciding what to store, as it did with `record`, construct the manager with
`MemoryManager(stores=[store], add_tool_config=True)` and the store with `writable=True`. Both default to
`False`. Without `add_tool_config` the model gets a `search_memory` tool and no `add_memory` tool, so
it can recall but never save. Without a writable store, `add_tool_config=True` raises `ValueError`
when the manager is constructed.

One consequence of the shift is worth stating plainly: the plugin let the model pass a `namespace` on some
paths, and this package does not. `scope` is configured once, stamped by the package on every write, and
cannot be set, changed or removed through a caller `filter` or through `metadata`. A prompt cannot reach it.

### Settings

<!-- settings: tools-memory -->
| Theirs | Here | Note |
|---|---|---|
| `MONGODB_ATLAS_CLUSTER_URI` (no default) | `MongoDBConfig["connection_string"]` | Or `MongoDBConfig["client"]`, which is preferred: a config dict prints in a traceback |
| `MONGODB_DATABASE_NAME` (default `strands_memory`) | `MongoDBConfig["database"]`, **required** | If you never set theirs, the database you have is `strands_memory` |
| `MONGODB_COLLECTION_NAME` (default `memories`) | `MongoDBMemoryStoreConfig["collection"]`, default `memory` | **The defaults differ.** Set `collection` explicitly or this package will look in an empty collection and find nothing |
| `MONGODB_EMBEDDING_MODEL` (default `amazon.titan-embed-text-v2:0`) | `MongoDBConfig["embedding_model"]` for the Atlas AutoEmbeddings path, default `voyage-4-lite`. Or `embedding_dimensions` plus `similarity` for the manual path | A Bedrock model name is not an Atlas model name. See [Keeping your existing vectors](#keeping-your-existing-vectors) |
| `MONGODB_NAMESPACE` (default `default`) | `MongoDBMemoryStoreConfig["scope"]` | Their namespace is validated against `^[A-Za-z0-9_-]{1,64}$`; `scope` has no such restriction, so any existing namespace is a legal scope |
| `AWS_REGION` (default `us-west-2`) | Nothing | The default path has Atlas embed the text, so there is no Bedrock call, no `boto3` and no region. On the manual path you embed before calling `add()`, so the region is your code's business |
| `vector_index_name` (default `vector_index`) | `MongoDBConfig["vector_index_name"]`, same default | Same name, same default, **different definition**. See [The index name is the same and the shape is not](#the-index-name-is-the-same-and-the-shape-is-not) |

No environment variable configures this package. Everything above is a key in a `TypedDict` you pass in,
which is why there is no table column for one.

### Documents

<!-- fields: tools-memory -->
| Their field | Here | What to do |
|---|---|---|
| `memory_id`, a string like `mem_1756713600000_a1b2c3d4` | `_id`, a server-generated `ObjectId` | Do not carry it over. `delete(memory_id)` here requires a 24-character hex ObjectId and rejects their format, so a copied id would be an id nothing can delete |
| `content` | `content` | Copy unchanged. This is the field the vector index embeds |
| `embedding`, 1024 Titan floats | `embedding`, only on the manual path | Drop it for AutoEmbeddings, keep it to reuse your vectors |
| `namespace` | `scope` | Rename. This is what makes the copy a copy: a search here filters on `scope`, so a document with only `namespace` is invisible |
| `timestamp`, an ISO-8601 **string** | `created_at`, a BSON **date** | Rename and retype. A string cannot carry a TTL index and sorts lexically |
| `metadata` | `metadata` | Copy unchanged, except that four keys are reserved here and are overwritten on read rather than stored: `_relevance_score`, `_id`, `_search_type` and `_scope` |
| absent | `store_name` | Add it, set to the `name` you give the store. Every read predicate here names it |
| absent | `content_hash` | Add it, or accept that migrated memories will not deduplicate. See [the hazards](#behaviour-differences-your-code-has-to-survive) |

### Keeping your existing vectors

The plugin embedded text itself, through Bedrock, into 1024-dimensional Titan v2 vectors with cosine
similarity. You have two options and the cheap one is not always the right one.

**Re-embed with Atlas.** Copy `content` and let this package's default path build an `autoEmbed` index, so
the deployment embeds at write time and at query time and you never handle a vector again. The AWS
dependency disappears. The cost is one pass over your content and a new index build.

**Keep the Titan vectors.** Both packages store the caller's vector in a field named `embedding`, so
`{"embedding_dimensions": 1024, "similarity": "cosine"}` makes your existing documents queryable as they
are. The cost is that you keep computing Titan vectors yourself, in your own code, for every `add()`, and
`embedding_dimensions` and `embedding_model` are mutually exclusive: setting both raises a `ValueError`
naming the conflict, because they index different fields with different contents.

## From `strands-mongodb-store`

This one is already a `MemoryStore` on the AutoEmbeddings path, so the shape of the integration does not
change and neither does your embedding story. Two things do: the store gains capabilities it did not have,
and its documents are missing the fields this package filters on.

The capabilities first, because they are the reason to move. That package has `search` and `add`. It has no
`delete`, no `delete_all`, no `initialize` and no `get_tools`, so today you have no way to erase a stored
memory through the store, which is a problem if anybody asks you to. This package has all four.

### Settings

Every setting there is a keyword-only constructor argument, and there are no environment variables on
either side.

<!-- settings: community-memory -->
| Theirs | Here | Note |
|---|---|---|
| `name` | `MongoDBMemoryStoreConfig["name"]` | Unchanged. It is the SDK's own key, and here it is also stamped on documents as `store_name` |
| `connection_string` | `MongoDBConfig["connection_string"]` | Unchanged in meaning. Theirs builds a synchronous `MongoClient`; this package uses `AsyncMongoClient` |
| `database_name` (default `strands_memory`) | `MongoDBConfig["database"]`, **required** | If you never set theirs, the database you have is `strands_memory` |
| `collection_name` (default `memory`) | `MongoDBMemoryStoreConfig["collection"]`, default `memory` | Same default, so nothing to change, and that is exactly why the index name collides |
| `client` | `MongoDBConfig["client"]` | Same idea. An injected client is not closed by `close()` in either package |
| `description` | `MongoDBMemoryStoreConfig["description"]` | Unchanged. SDK key |
| `max_search_results` | `MongoDBMemoryStoreConfig["max_search_results"]` | Unchanged. SDK key, default 10 in both |
| `writable` | `MongoDBMemoryStoreConfig["writable"]` | Unchanged. SDK key. Note `delete()` and `delete_all()` here also require it |
| `extraction` | `MongoDBMemoryStoreConfig["extraction"]` | Unchanged. SDK key |
| `model` (default `voyage-4-lite`) | `MongoDBConfig["embedding_model"]`, same default | Same default, and both declare it inside the index definition rather than calling an embedder |
| `index_name` (default `vector_index`) | `MongoDBConfig["vector_index_name"]`, same default | Same name, **different definition**. This is the collision that stops an in-place move dead |
| `num_candidates` (default `max(limit * 10, 100)`) | `MongoDBSearchQuery["oversampling_factor"]`, default 10 | A multiplier here rather than an absolute, so it scales with the limit instead of flooring at 100 |
| `create_index` (default `True`) | `MongoDBConfig["ensure_indexes"]`, default `True` | Same switch, one level up: it sits on the connection config, so a storage and a memory store sharing one config share the answer. Set it `False` and `initialize()` verifies the indexes and reports what is missing instead of creating anything, which is what a least-privilege deployment needs |

### Documents

<!-- fields: community-memory -->
| Their field | Here | What to do |
|---|---|---|
| `_id`, a `uuid4` string | `_id`, a server-generated `ObjectId` | Do not carry it over. `delete()` here requires a 24-character hex ObjectId, so a copied uuid would be an id nothing can delete |
| `content` | `content` | Copy unchanged. Same field name, same purpose, and the field both `autoEmbed` definitions embed |
| `metadata` | `metadata` | Copy, but default it to `{}`. Theirs inserts the caller's value as given, so a document written without metadata carries `metadata: null` rather than an empty object |
| `createdAt`, an ISO-8601 **string** | `created_at`, a BSON **date** | Rename and retype |
| absent | `scope` | Add it. That package has no tenancy field at all, so every document it wrote is in one undivided space and you choose what scope they land in |
| absent | `store_name` | Add it, set to the `name` you give the store |
| absent | `content_hash` | Add it, or accept that migrated memories will not deduplicate |

## Copying memories

One script covers either source. It reads the old documents, adds the three fields this package filters on,
retypes the timestamp, and writes them to the new collection. Set `SOURCE` to `"tools"` or `"community"` and
it picks the right field names.

The `content_hash` is computed here, in Python, and that is not a stylistic choice. MongoDB aggregation has
no SHA-256 operator, so a `$merge` or `$out` pipeline cannot fill the field in, and a server-side
`$function` needs JavaScript enabled on the deployment. Copying through a client is the portable answer.

<!-- run: contract -->
```python
import asyncio
import datetime
import hashlib
import os

from pymongo import AsyncMongoClient

DATABASE = os.environ.get("MONGODB_DATABASE", "agents")

# "tools" for the archived strands-tools plugin, "community" for strands-mongodb-store.
SOURCE = "tools"
OLD_COLLECTION = "memories" if SOURCE == "tools" else "memory"
TIMESTAMP_FIELD = "timestamp" if SOURCE == "tools" else "createdAt"

# What the new store is configured with. `SCOPE` is the one value the old data cannot supply for the
# community package, which has no tenancy field; for the archived plugin it is the old `namespace`.
NEW_COLLECTION = "memory"
STORE_NAME = "user-memories"
SCOPE = "acme"

# Keep the old vectors, or drop them and let Atlas embed the content. Dropping is the default here
# because it is what `MongoDBConfig["embedding_model"]` expects.
KEEP_VECTORS = False


def parse_timestamp(value: object) -> datetime.datetime:
    """Turn either package's ISO-8601 string into the BSON date this package sorts and expires on."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.datetime.now(datetime.timezone.utc)


def transform(old: dict) -> dict:
    """Map one old document onto this package's shape."""
    content = old["content"]
    new = {
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "scope": old.get("namespace") or SCOPE,
        "store_name": STORE_NAME,
        "created_at": parse_timestamp(old.get(TIMESTAMP_FIELD)),
        "metadata": dict(old.get("metadata") or {}),
    }
    if KEEP_VECTORS and old.get("embedding"):
        new["embedding"] = old["embedding"]
    return new


async def main() -> None:
    client = AsyncMongoClient(os.environ["MONGODB_URI"])
    database = client[DATABASE]
    try:
        # Seeded here so this runs as written. Against your own deployment, delete this block.
        await database[OLD_COLLECTION].insert_one(
            {
                "_id": "mem_1756713600000_a1b2c3d4",
                "content": "the user prefers dark mode",
                "namespace": "acme",
                TIMESTAMP_FIELD: "2026-08-24T09:15:00.000Z",
                "metadata": None,
            }
        )

        copied = 0
        async for old in database[OLD_COLLECTION].find({}):
            new = transform(old)
            # Upsert on the same three fields the unique index covers, so re-running the copy after an
            # interruption does not duplicate what already landed.
            await database[NEW_COLLECTION].update_one(
                {"scope": new["scope"], "store_name": new["store_name"], "content_hash": new["content_hash"]},
                {"$setOnInsert": new},
                upsert=True,
            )
            copied += 1
        print(f"copied {copied} memories into {DATABASE}.{NEW_COLLECTION}")
    finally:
        await client.close()


asyncio.run(main())
```

Run it against a copy first. It reads the old collection and never writes to it, so the old store keeps
working throughout, which means you can cut over when the copy is verified rather than during it.

## Behaviour differences your code has to survive

Everything above is about data. These are about code that reads the results.

<!-- hazards -->
| What changes | Coming from | Here | What to do |
|---|---|---|---|
| Where the relevance score lives | `score`, a top-level key on each memory dict in the archived plugin's `retrieve` payload (`mongodb_memory.py:575`); `metadata["_score"]` in `strands-mongodb-store` (`RELEVANCE_SCORE_KEY`, `__init__.py:30`) | `metadata["_relevance_score"]` | Rename the read. `_relevance_score` is one of four reserved metadata keys here, with `_id`, `_search_type` and `_scope`, and all four are written onto the result at read time rather than stored |
| Where the memory's id lives | `memory_id`, top-level on each memory dict (`mongodb_memory.py:571`); `strands-mongodb-store` returns `MemoryEntry(content=..., metadata=...)` and no id at all (`__init__.py:152`) | `metadata["_id"]`, the ObjectId as a string | Coming from the archived plugin, move the read. Coming from `strands-mongodb-store`, this is new: it is what makes `delete(memory_id)` reachable from a search result, which that package has no way to do |
| Deduplication | Neither source deduplicates. Both insert on every call | A unique partial index on `(scope, store_name, content_hash)`, and `add()` reports `duplicates` | Nothing, for new writes. For copied documents see the `content_hash` row below |
| `content_hash` is missing on copied documents | absent in both | Required for deduplication | The unique index is partial on `content_hash` existing, so documents without it do not violate it and nothing breaks. But a later `add()` of identical content will insert a second document, because there is nothing to collide with. The copy script above fills the field in, which is the reason it is a script and not a `$merge` pipeline |
| `store_name` is missing on copied documents | absent in both | Named in every read predicate | Documents without it are invisible to `search`, `delete_all` and `delete`. Not an error, just no results, which is the failure mode worth knowing about before you conclude the migration worked |
| The timestamp's type | `timestamp` in the archived plugin, `createdAt` in `strands-mongodb-store`, both ISO-8601 strings | `created_at`, a BSON date | Retype it during the copy. A string sorts lexically and cannot carry a TTL index, so leaving it a string quietly costs you both expiry and recency ordering |
| Pagination | The archived plugin pages `list` and `retrieve` with a `next_token` cursor | Not present. `search` takes a `limit` | If you paged through results, that loop becomes a single call with a higher limit |
| Response size caps | The archived plugin truncates at 70,000 characters total, 12,000 per memory, 5 memories per response, because its output went to a model's context | No caps. Results are returned to your code, not rendered into a prompt | If you relied on the truncation to bound a prompt, bound it yourself |

## The index name is the same and the shape is not

All three source packages default their vector index to `vector_index`. So does this one. Two of them also
default their collection to a name this package defaults to: `strands-mongodb-store` uses `memory` and
`strands-mongodb-storage` uses `storage`. So the most likely first attempt at a migration is pointing this
package at a collection that already has a `vector_index`. None of the three definitions is the one this
package needs, and they miss by different distances:

| Package | Index type | Definition | Read from |
|---|---|---|---|
| `strands-tools` `mongodb_memory` | `search`, the legacy form | `mappings.fields.embedding` as a `knnVector`, 1024 dimensions, cosine, plus `namespace` as a string | `mongodb_memory.py:207-222` |
| `strands-mongodb-store` | `vectorSearch` | One `autoEmbed` field on `content` and nothing else. No filter fields | `__init__.py:105-117`, 2026-09-01 |
| This package | `vectorSearch` | `autoEmbed` on `content`, plus `filter` entries for `scope` and `store_name` | `_indexes.py`, `memory_search_specs()` |

The second row is the near miss, and it is the one worth being careful about: its single field is exactly
one of the three this package declares, so an index that looks right is missing precisely the two entries
that make scope and store isolation enforceable server-side.

**`initialize()` changes nothing, and under a named `search_type` it raises.** It reads the existing index by name, compares the reported
definition against the one it needs by containment, and when something it declared is absent it stops. It
does not drop the index, and it does not issue a create for a name that is already taken. The message names
the gap and leaves both remedies for you to run:

```text
strands_mongodb: the vectorSearch index 'vector_index' on agents.memory already exists and its definition
does not cover the one this package needs: definition.fields is absent, and this package declared
[{'type': 'autoEmbed', 'path': 'content', 'model': 'voyage-4-lite', 'modality': 'text'}, {'type': 'filter',
'path': 'scope'}, {'type': 'filter', 'path': 'store_name'}]. Reported: {'mappings': {'dynamic': False,
'fields': {'embedding': {'type': 'knnVector', 'dimensions': 1024, 'similarity': 'cosine'}, 'namespace':
{'type': 'string'}}}}. A search index cannot be redefined in place, and dropping it discards a build
somebody may be querying, so the remedy is left here rather than taken:
db.runCommand({"dropSearchIndex": "memory", "name": "vector_index"})
db.runCommand({"createSearchIndexes": "memory", "indexes": [...]})
```

Under `search_type: "auto"`, the default, the same message arrives as a warning instead. The vector tier
is dropped and the store answers from the text tier, so a migrated store that starts and returns lexical
matches may be this case. Check `_search_type` on a result, or the warning, before concluding that it is
working.

Measured against Atlas 8.0.30 on 2026-09-01, with the database name replaced and the create command's
definition elided for width. Both foreign definitions were created on that deployment first, and Atlas
accepted both, including the legacy `knnVector` form, which it reports as `type: "search"` with the
definition echoed back unchanged.

Coming from `strands-mongodb-store` the message differs in one clause. `definition.fields` is present there,
so rather than *"is absent"* it names the first field nothing in the existing index covers. On the default
server-side embedding path that field is a `'type': 'filter'` entry, because the `autoEmbed` field does
match: same path, same model. That is the useful version of the message, since it tells you the index is
one line short rather than wrong. If you configure `embedding_dimensions` and supply your own vectors, the
field it names first is the `vector` one on `embedding` instead.

The clause about dropping is not boilerplate: a search index build takes minutes, and something may be
querying the old one, so the choice of when to drop it belongs to you.

Two ways to avoid the collision entirely. Copy into a **new collection**, which is what the script above
does, and let this package build its own indexes there. Or, if you want both packages on one collection
during a cutover, give this one a different index name with `MongoDBConfig["vector_index_name"]`.

## What this guide cannot check

The archived `strands-tools` plugin is in the same workspace as this package, so its settings and document
fields are extracted from `tools/src/strands_tools/mongodb_memory.py` by the test suite and compared against
the tables above. Add a setting to that plugin and a test here fails.

The two community packages are **not vendored** into this repository and are **not fetched** when the tests
run. Vendoring them would add third-party source with its own notice obligations; fetching them would make
this suite depend on network and on somebody else's default branch. So the tables above are a transcript
taken on **2026-09-01**, and the tests check that the guide matches the transcript, not that the transcript
matches upstream.

What that means for you: the URLs in [Which package you have](#which-package-you-have) are the source of
truth, the date is when it was last read, and a claim here that contradicts what you see in your own
installed version is a claim that has aged. Check the version you actually have. `strands-mongodb-store`
was read at 0.2.0 and `strands-mongodb-storage` at 0.1.0.
