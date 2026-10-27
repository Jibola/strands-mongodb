# Working in this repository

The rules that apply every time you touch the code. [CONTRIBUTING.md](CONTRIBUTING.md) is the process
around a change: how to set up, what a pull request has to pass, where to report a bug. This file is what
the change itself has to look like.

Written for whoever arrives next, human or agent. Where a rule can be checked by a tool it is, and the
test that checks it is named, because a convention nothing measures is a preference.

## Layout

One repository, one published package. A TypeScript sibling publishing the same name to npm is deferred
pending the parity spike, and the layout reserves the shape so it can be added without moving
files.

| Path | Holds |
|---|---|
| `python/` | The `strands-mongodb` package. `src/`, `tests/`, `examples/`, `pyproject.toml`, and the reference `README.md` and `MIGRATION.md` |
| `.github/workflows/ci-python.yml` | Every gate a pull request meets |
| `README.md` | The landing page. Short on purpose: `python/README.md` is the reference |
| `docs/images/` | The diagrams the READMEs and `CONTRIBUTING.md` show, as PNG |
| `AGENTS.md`, `CONTRIBUTING.md` | This file and the process |

Everything with a version number lives under `python/`. When a second language lands, this file gains a
per-language section rather than splitting into two copies, because a split is how two files that describe
one repository start disagreeing.

## Where the reasoning lives, and how tests are tagged

The tests are the record of what this package guarantees. A change in behaviour comes with the test that
proves it, written first, and the reason for a non-obvious choice lives in the code that makes it.

**Every test names the item it proves, in the first paragraph of its docstring.** Either its own docstring
or its class's:

```python
async def test_a_bound_above_the_lowest_score_drops_exactly_that_result(self) -> None:
    """AT-059b. Measure first, then bound just above the measurement's floor."""
```

The release gate in `tests/conftest.py` collects these tags to decide which checklist items have evidence.
It reads the first paragraph only, so that a later sentence such as "this tier does not cover X" is never
counted as evidence that X is covered. Use the tag nowhere else: comments, later paragraphs and the
documentation describe the behaviour in words.

A `TODO(...)` in a file names the thing that has to be settled before the placeholder can go, and the test
suite asserts that every placeholder in this file and in `CONTRIBUTING.md` names something. Do not write a
bare `TODO`.

## Commands

Every command runs from `python/`. [Hatch](https://hatch.pypa.io) builds the environment from
`pyproject.toml`, so nothing needs installing first.

| Command | Does |
|---|---|
| `hatch run test` | The suite. Arguments pass through, so `hatch run test tests/test_storage.py -x` works |
| `hatch run lint` | Ruff, over `src/`, `tests/`, `scripts/` and `examples/` |
| `hatch run format` | Ruff's formatter, in place |
| `hatch run format-check` | The same, read-only. What CI runs |
| `hatch run typecheck` | Mypy, over the same four directories |
| `hatch run cov` | The suite plus both coverage floors: 90% aggregate and 70% per module |
| `hatch run prepare` | `format`, `lint`, `typecheck`, `test`, in that order. Run this before pushing |

Pass no path to `lint` or `typecheck`. An explicit path overrides the file list in `pyproject.toml`, which
is the one list both tools read.

`cov` is deliberately not part of `test`. A floor that fails on a half-finished module trains everyone to
reach for `--no-cov`, so `test` stays runnable at any point mid-change and `cov` is the release gate.

Working in a plain virtualenv instead is fine: `pip install -e ".[all,dev]"` from `python/` installs the
same set, and `python -m pytest` replaces `hatch run test`. Python 3.10 is the floor, so use an
interpreter at 3.10 or newer.

## Test tiers

Most of the suite needs nothing. The tiers that need a deployment are gated on an environment variable
being non-empty, and a tier whose variable is unset skips rather than fails, so a first run on a laptop
with no cluster is green and fast.

| Marker | Gate | Needs |
|---|---|---|
| unmarked | none | Nothing. Around 700 tests, roughly 50 seconds |
| `contract` | `MONGODB_URI` | Any MongoDB deployment, including a local one |
| `integration` | `MONGODB_ATLAS_URI` | Atlas, or an Atlas CLI local deployment with `mongot` |
| `autoembed` | `MONGODB_AUTOEMBED_URI` | An Atlas cluster that runs automated embedding: M10 or larger, storage auto-scaling on. `python/scripts/atlas_ci_cluster.py up` creates one |
| `build` | `STRANDS_MONGODB_BUILD_TIER` | Network. Builds a wheel and installs it into an empty venv. No deployment and no secret |
| `compat` | none, CI cron or dispatch only | Resolves the latest published `strands-agents` rather than the pinned one |
| `example` | its backend's tier | Runs a shipped `examples/` script as a subprocess |
| `gate` | none | The release-gate assertion. Ordered last, so it sees every other outcome |

The variables are read from the process environment, and also from a `.env` file beside this one if you
keep one there. `.env` is gitignored and must stay that way: it holds a connection string with a password
in it.

**The gotcha that will get you once.** Because `.env` is loaded automatically, a bare `hatch run test` on a
machine that has one starts the contract tier, which takes minutes rather than seconds. Opting out means
setting the variables to **empty**, not unsetting them:

```bash
MONGODB_URI= MONGODB_ATLAS_URI= MONGODB_AUTOEMBED_URI= hatch run test
```

To run one tier on purpose, select it: `hatch run test -m contract`, or `hatch run test -m "not integration"`
to keep everything else. `--strict-markers` is on, so a marker that is not registered in `pyproject.toml`
is a collection error rather than a silently ignored decorator. Register it there before you use it.

Two things to know before you wait on a cluster. A first `add()` against a collection with no search index
creates two indexes and waits for both to report `READY`, which takes minutes. A document is then not
searchable the instant its write returns, because the search process indexes it asynchronously. Test
helpers poll to a 180-second ceiling rather than sleeping, so a fast cluster is not slowed down.

## House conventions

The Strands Python SDK's own contributor guide is where these come from. Seven of the nine are checked by a
tool or a test, and the table names which. The two that are not are review's job, and saying so is more
useful than pretending otherwise.

| Convention | Checked by |
|---|---|
| `X \| None`, never `Optional[X]` | `tests/test_conventions.py` |
| `TypedDict` or `dataclass` for structure. No pydantic, no runtime validator | `tests/test_conventions.py` |
| An explicit `__all__` in every module, so the public surface is a decision | `tests/test_conventions.py` |
| `%s` placeholders in log calls, never an f-string | Ruff's `G` rules, and `tests/test_conventions.py` asserts they are selected |
| `warnings.warn` always passes `stacklevel`, so the warning points at the caller | `tests/test_conventions.py`, written for the first call anyone adds |
| A `Raises:` section on anything public that raises | `tests/test_conventions.py`, over every public function and method |
| Google-style docstrings. `snake_case` functions, `_leading_underscore` for private | Review |
| No `asyncio.TaskGroup` or `asyncio.timeout` without a version gate. Both are 3.11+ and the floor is 3.10 | `tests/test_conventions.py` |
| Conventional commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`) | Review. `CONTRIBUTING.md` has the format |

**One deviation, and it is deliberate.** The SDK's log messages are shaped
`field=<%s>, other=<%s> | lowercase message`. This package does not copy that shape, and writes
`strands_mongodb: ` followed by a sentence with `%s` placeholders in it:

```python
_SCORE_BOUND_APPLIED = "strands_mongodb: %s %r on %s kept %d of %d results (applied after the query)."

logger.debug(_SCORE_BOUND_APPLIED, name, bound, self._target, len(kept), len(scored))
```

Message bodies are module constants with docstrings of their own, not literals inside the call. That is
where the reason for a message lives, and it is also what lets a test assert on one without copying the
string.

Two reasons, both about being a third-party integration rather than the SDK itself. The prefix names the
package in a log stream where the logger name is usually not rendered, and several of these messages are
explanatory rather than a field list: the one for a deployment with no Atlas Search names the server error
code the probe saw and what to do about it, which does not fit a key-value line. Every log call in the
package follows this shape, and `tests/test_telemetry.py` pins each one, so reshaping them is a change with
tests to update rather than a matter of taste. Placeholders stay `%s`: that half of the rule is enforced by ruff.

Every log call also passes `extra=_telemetry.event_extra(...)`, which adds `strands_mongodb_event` and,
where the record is about one, `strands_mongodb_namespace` and `strands_mongodb_tier`. A new call needs a
new name in `_telemetry.LogEvent`, and `tests/test_telemetry.py` fails on a call that skips the helper.
Spans go through `_telemetry.traced` or `_telemetry.span`, and an attribute is recorded only if it is in
`_telemetry.Attribute`, the allow-list; anything else is dropped with a `DEBUG` record. Never add an
attribute that carries caller data: a key, content, metadata, query text, filter values, a scope or a
memory id. The README's observability section lists every span, attribute and event, and
`tests/test_readme.py` fails until a new one is named there.

## What not to do

- **Do not silence a warning to make a run quiet.** Two known ones are tracked and neither is ours to fix:
  `pytest-asyncio` 0.26 calling a deprecated event-loop API, and pymongo closing a periodic executor in a
  weakref callback after the loop has gone. A
  `filterwarnings` entry that hides those also hides the next one, which will be ours.
- **Do not put a connection string in a document, a test or an example.** `tests/test_conventions.py` walks every Markdown file
  in the repository looking for one with credentials in it, and the examples redact the userinfo before they
  print a URI.
- **Do not infer an SDK API.** Read the installed `strands` package. The floor is `strands-agents` 1.54.0
  because that is the first release with every module this package imports, and a guess about an earlier
  release is a guess about somebody's install. Do not raise or lower a floor by reading a changelog either:
  the `Dependency floors` job installs the declared floors and runs the suite against them, which is the only
  check of what the code actually needs.
- **Do not hand-edit the index and privilege commands in `python/README.md`.** They are compared against the
  renderer in `src/strands_mongodb/_indexes.py` that produces them, so an edit there fails a test rather
  than shipping a command the package does not run.
- **Do not add a runtime dependency to `dependencies`.** It is empty on purpose. Every dependency belongs to
  the extra that needs it, so `pip install 'strands-mongodb[storage]'` never pulls vector-search machinery.
- **Do not skip a test to get to green.** A skip and a pass are the same colour.
