# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), with one deviation the
format does not cover: **no version below carries a date, because none of them has been released.**

The repository's landing page publishes a version table saying what `0.0.1` and `0.1.0` contain, and
both `README.md` files carry a marker above their install blocks saying the package is not on PyPI
yet. A test keeps the three in step: the versions here must be exactly the versions the landing page
publishes, and nothing here may carry a date while either README still says the package is
unpublished. Removing those markers without dating a release fails the suite.

Versions are `MAJOR.MINOR.PATCH` per [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and
a release is cut by pushing a `python-v` tag. [RELEASING.md](RELEASING.md) is the procedure.

## [Unreleased]

Work that has landed on `main` and is not yet inside a tagged version. Entries move down into a
version heading when that version is tagged.

### Added

- `MongoDBStorage(search_strategy=...)`, as on the SDK's `S3Storage`.
- `MongoDBVectorSearchStrategy`: Atlas Vector Search for `MongoDBStorage`, using any SDK `Embedder`.
- `embedder` on `MongoDBMemoryStore`'s manual-vector path, so `add` and string `search` embed.

### Changed

- `MongoDBStorage.search` defaults to the SDK's `KeywordSearchStrategy`, which also scores payloads, so
  `FileMemoryStore` recalls by body text.
- Under `search_type="auto"`, `initialize()` no longer waits for index builds; the first `search` does.
- Examples use `context_manager="auto"` instead of `ContextOffloader`.

## [0.1.0]

The first version with working code in it. Planned, not released.

### Added

- `MongoDBStorage`, implementing all five `Storage` contract methods plus `namespace()` and
  `delete_prefix()`, so an `Agent` given one gets sessions, context offloading and context history.
- `MongoDBMemoryStore`, with `initialize()`, `add()`, `search()`, `delete()`, `delete_all()` and
  `get_tools()`.
- Three search tiers behind `MongoDBMemoryStore.search()`: `vector`, `text`, and `hybrid` fusing the
  two with `$rankFusion`. `auto` picks among them. Every tier returns at most the requested limit.
- Under `auto`, a search tier that could not be prepared at startup is tried again on a later
  operation, first after 60 seconds and then with the delay doubling up to an hour. A retry does not
  wait for an index build.
- Record expiry, armed differently on each surface: `MongoDBMemoryStore.initialize()` provisions the
  TTL index itself, while `MongoDBStorage` leaves it to an explicit `await storage.ensure_indexes()`
  at startup, because an `Agent` builds its storage where an operator never sees it and DDL should
  not be issued from there. A negative `ttl_seconds` is rejected at construction.
- Driver attribution in the connection handshake: every client carries `Strands-MongoDB` and the package
  version in its driver metadata, as MongoDB's handshake specification asks of a wrapping library.
  `appName` is left to the application.
- OpenTelemetry spans for every public operation of both stores, and for the index work they cause,
  on your application's tracer provider. The package configures no provider. Keys, content, queries,
  scopes and credentials are never recorded. Every log record also carries a stable
  `strands_mongodb_event` field for JSON log pipelines.
- [MIGRATION.md](MIGRATION.md), mapping three existing MongoDB memory packages onto this one.

### Not included

- `MongoDBMemoryStore.add_messages()`, absent on purpose. The SDK's `MemoryManager` feature-detects
  it, and a present one would take extraction away from the SDK's own extractor and hand it to one
  this package would have to write.

## [0.0.1]

Package metadata only: extras, import paths and public names, with no backend behind them. Planned,
not released.

### Added

- The `storage`, `memory` and `all` extras, each pulling only its own dependencies. A bare
  `pip install strands-mongodb` installs no runtime dependencies and gives no working backend.
- The `strands_mongodb` import path and the public names it exports.

<!-- No link reference definitions below. Keep a Changelog puts compare links here; this repository
     has no tags to compare yet. They go in with the first release, alongside that release's date. -->
