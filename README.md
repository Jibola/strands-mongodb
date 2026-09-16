# strands-mongodb

MongoDB storage and memory backends for the [Strands Agents](https://strandsagents.com) SDK.

One MongoDB deployment behind the SDK's two extension points:

| Surface | Class | SDK protocol | Serves |
|---|---|---|---|
| Storage | `MongoDBStorage` | `strands.storage.Storage` | Sessions, context offloading, memory persistence, context history, transcripts |
| Memory | `MongoDBMemoryStore` | `strands.memory` `MemoryStore` | Semantic recall over Atlas Vector Search, extraction writes |

The `Storage` protocol is the leverage point: the SDK accepts a single top-level `storage` on
`Agent`, and **five** subsystems inherit it. One correct `MongoDBStorage` therefore delivers
session management and context offloading without a separate session-manager class.

## Status

> **Under construction. Both halves have their operations; neither is released.** `MongoDBStorage`
> implements all five `Storage` contract methods plus `namespace()` and `delete_prefix()`, so an
> `Agent` given one gets working sessions, context offloading and context history.
> `MongoDBMemoryStore` has `initialize()`, `add()`, `search()` across a vector tier, a text tier and a
> `hybrid` tier fusing the two with `$rankFusion`,
> `delete()`, `delete_all()` and `get_tools()`. `add_messages()` is absent on purpose: the SDK's
> `MemoryManager` feature-detects it, and a present one would take extraction away from the SDK's own
> extractor and hand it to one this package would have to write.
>
> | Version | Contains |
> |---|---|
> | `0.0.1` | Package metadata, extras, import paths, public names |
> | `0.1.0` | `MongoDBStorage` + `MongoDBMemoryStore`, the three search tiers, record expiry, docs, tests |
>
> Storage `search()` ranks by token overlap in a single aggregation over the key space. Measured over 1,000
> keys carrying 400-byte payloads: 2 wire commands and 9 ms, against 1,041 commands and 210 ms for the
> SDK's default, for the same ranking. Scores come from keys rather than payload text, so a word that
> appears only inside a stored value will not match.
>
> Record expiry is armed differently on each surface. `MongoDBMemoryStore.initialize()` provisions the
> TTL index itself, so a store configured with `ttl_seconds` expires its records without further work.
> `MongoDBStorage` leaves it to an explicit `await storage.ensure_indexes()` at startup, because an
> `Agent` builds its storage in a code path the operator never sees and DDL should not be issued from
> one. Stamping the field is not the same as provisioning the index that acts on it, and only one of
> the two surfaces can safely do both on its own.
>
> Verified against two deployments rather than one: MongoDB 8.3.8 with a local mongot, and cloud
> Atlas 8.0.32. The two report search index state differently in six places, and the tests read both
> spellings, because a suite pinned to one deployment passes everywhere it was written and nowhere
> else.

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

## Documentation

[`python/README.md`](python/README.md) is the reference: quickstarts for both surfaces, every
configuration key of all three `TypedDict`s, how a search picks its tier, the tenancy boundaries and who
enforces each one, how to erase stored memories, the index and privilege handoff for a DBA, and the
limits this package does not remove. This page stays short on purpose. Where the two would say the same
thing, only one of them does.

[`python/MIGRATION.md`](python/MIGRATION.md) is for readers arriving from one of the three MongoDB
backends that already exist for Strands: the archived `mongodb_memory` tool in `strands-agents-tools`,
and the community `strands-mongodb-store` and `strands-mongodb-storage` packages. One of the three
migrates by pointing this package at the same collection. The other two need data copied, and the guide
gives the script.

## Examples

Two runnable scripts in [`python/examples/`](python/examples/README.md). Each splits into two commands
you run as separate processes, so the only thing that carries state between them is MongoDB.
`MONGODB_URI` is all either one needs, and `--dry-run` shows what a step would do before it connects.

## Requirements

| | |
|---|---|
| Python | >= 3.10 |
| `strands-agents` | >= 1.48.0, the release that introduced the unified `Storage` interface and `StorageError` |
| MongoDB | Any deployment serves storage. The text tier needs Atlas Search; the vector tier needs Atlas or an Atlas CLI local deployment. The default embedding path also needs a model registered on the cluster, measured present on Atlas and absent on a local `mongot` 8.3.8. |

## Erasing memories

`MongoDBMemoryStore` has `delete(memory_id)` and `delete_all()`, so an erasure request against stored
facts has something to call. Both need `writable: True`, and both reach exactly as far as `search()`
does, which is one scope and one store. [`python/README.md`](python/README.md#erasing-memories) has the
details, including the two ways this surprises people.

## Repository layout

| Directory | Package | Publishes to |
|---|---|---|
| `python/` | `strands-mongodb` | PyPI |

A `typescript/` sibling publishing the same name to npm is deferred pending a parity spike;
the layout above reserves the shape so it can be added without moving files.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) is the process: how to get an environment, how to report a bug, what a
pull request has to pass. [AGENTS.md](AGENTS.md) is what a change has to look like once you are writing
one, and it is the shorter read of the two.

Most of the test suite needs no MongoDB deployment, so a first run on a laptop with no cluster is green.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
