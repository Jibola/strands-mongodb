# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), with one deviation the
format does not cover: **no version below carries a date, because none of them has been released.**

The repository's landing page publishes a version table saying what `0.0.1` and `0.1.0` contain, and
both `README.md` files carry a marker above their install blocks saying the package is not on PyPI
yet. A changelog is the third document making claims about the same thing, and three documents that
nobody compares is how a release note comes to describe a version that does not exist.

So the comparison is a test. `AT-164` asserts that the versions here are exactly the versions the
landing page publishes, in both directions, and that nothing here is dated while either README still
says the package is unpublished. The day the name is claimed on PyPI, removing those markers without
dating a release fails the suite — which is the point, because that is the day the two documents are
most likely to drift.

Versions are `MAJOR.MINOR.PATCH` per [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and
a release is cut by pushing a `python-v` tag. [RELEASING.md](RELEASING.md) is the procedure.

## [Unreleased]

Work that has landed on `main` and is not yet inside a tagged version. Entries move down into a
version heading when that version is tagged.

## [0.1.0]

The first version with working code in it. Planned, not released.

### Added

- `MongoDBStorage`, implementing all five `Storage` contract methods plus `namespace()` and
  `delete_prefix()`, so an `Agent` given one gets sessions, context offloading and context history.
- `MongoDBMemoryStore`, with `initialize()`, `add()`, `search()`, `delete()`, `delete_all()` and
  `get_tools()`.
- Three search tiers behind `MongoDBMemoryStore.search()`: `vector`, `text`, and `hybrid` fusing the
  two with `$rankFusion`. `auto` picks among them.
- Record expiry, armed differently on each surface: `MongoDBMemoryStore.initialize()` provisions the
  TTL index itself, while `MongoDBStorage` leaves it to an explicit `await storage.ensure_indexes()`
  at startup, because an `Agent` builds its storage where an operator never sees it and DDL should
  not be issued from there.
- `appName` attribution on every client the package builds from a connection string: `strands-mongodb`,
  or the caller's own value with `-strands-mongodb` appended. An injected client is left as it is.
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
     has no tags to compare, and a link to a tag that does not exist is a claim that it does. They go
     in with the first release, alongside that release's date. -->
