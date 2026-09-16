"""The migration guide, checked against three packages, only one of which is in this repository.

The acceptance set ``AT-098`` through ``AT-102``. The requirements carry the reasoning;
this file carries the checks.

``python/MIGRATION.md`` is the only document in this repository whose subject is somebody else's code.
That makes it the only one where a test cannot prove the claims are current, and pretending otherwise
would be worse than admitting it. So the mechanism is deliberately uneven, and the unevenness is visible
in the class names below:

* The archived ``strands-tools`` plugin lives in this workspace, so its settings and document fields are
  extracted from the module text and compared against the guide. When the workspace is not present, the
  transcribed inventory below is still compared against the guide, so a bare clone of this repository
  still runs a meaningful test.
* The two community packages are not vendored and are not fetched at test time. Their inventories are
  transcribed here, with the URL and the date they were read. These tests catch the guide drifting from
  the transcript. Nothing here can catch the transcript drifting from upstream, which is ``REQ-E-102``'s
  reason for existing and is stated in the guide rather than hidden in this docstring.

What the guide *runs* is ``test_readme.py``'s subject: the guide joined that file's block sweep, so every
fenced snippet in it carries a run directive or a skip reason (``REQ-E-097``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from strands_mongodb import _indexes
from strands_mongodb.types import MongoDBConfig

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    # `tomli` is the parser `tomllib` was adopted from, and 2.x is the API `tomllib` shipped, so the
    # alias makes the two interchangeable and every caller below reads one name. It is a dev dependency
    # declared without a version marker on purpose; `pyproject.toml` says why, and the reason is mypy
    # rather than this branch.
    import tomli as tomllib

# tests/ -> python/ -> the repository root -> the workspace that holds the archived tool.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIGRATION_GUIDE = _PROJECT_ROOT / "MIGRATION.md"
_PACKAGE_README = _PROJECT_ROOT / "README.md"
_ROOT_README = _PROJECT_ROOT.parent / "README.md"
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"

#: The archived plugin, when this checkout sits inside the workspace that also holds ``tools/``. Read
#: rather than imported: the module needs ``boto3`` and the ``mongodb-memory`` extra, and this package
#: depends on neither. Absent in a bare clone, which is what ``_workspace_only`` handles.
_ARCHIVED_TOOL = _PROJECT_ROOT.parent.parent / "tools" / "src" / "strands_tools" / "mongodb_memory.py"

_workspace_only = pytest.mark.skipif(
    not _ARCHIVED_TOOL.is_file(),
    reason=f"the archived strands-tools plugin is not at {_ARCHIVED_TOOL}; the transcript tests still run",
)

#: Opt-in gate for the sdist build, shared in spirit with ``test_packaging.py``'s ``AT-002``/``AT-003``.
#: Building a distribution costs seconds and needs a build backend, so the cheap manifest assertion runs
#: every time and the artifact assertion runs behind this.
BUILD_TIER_VARIABLE = "STRANDS_MONGODB_BUILD_TIER"

_BUILD_TIER_GATE = pytest.mark.skipif(
    not os.environ.get(BUILD_TIER_VARIABLE),
    reason=f"build tier needs {BUILD_TIER_VARIABLE} set; it builds an sdist",
)

_TABLE_ROW = re.compile(r"^\|(?P<cells>.*)\|\s*$")


@dataclass(frozen=True)
class Source:
    """One package the guide has to speak to.

    Attributes:
        slug: Marker suffix used in ``MIGRATION.md``, as ``<!-- settings: <slug> -->``.
        distribution: The name a reader would recognise, and searches the guide for.
        read_from: Workspace path or URL the inventory below was taken from.
        read_on: ISO date it was read, or ``""`` for a source in this workspace, where the file itself
            is the record and a date would go stale without meaning anything.
        settings: Every knob the package has, named as the package names it.
        fields: Every field the package writes into a document.
    """

    slug: str
    distribution: str
    read_from: str
    read_on: str
    settings: tuple[str, ...]
    fields: tuple[str, ...]


#: The archived plugin. ``vector_index_name`` is in the settings list and has no environment variable,
#: which is the row a reader most needs and the one an env-var sweep alone would miss (`:778`).
_ARCHIVED = Source(
    slug="tools-memory",
    distribution="strands-tools",
    read_from="tools/src/strands_tools/mongodb_memory.py",
    read_on="",
    settings=(
        "MONGODB_ATLAS_CLUSTER_URI",
        "MONGODB_DATABASE_NAME",
        "MONGODB_COLLECTION_NAME",
        "MONGODB_EMBEDDING_MODEL",
        "MONGODB_NAMESPACE",
        "AWS_REGION",
        "vector_index_name",
    ),
    fields=("memory_id", "content", "embedding", "namespace", "timestamp", "metadata"),
)

#: ``strands-mongodb-store`` 0.2.0. Thirteen constructor keywords, all keyword-only, no environment
#: variables at all. Five of them (``name``, ``description``, ``max_search_results``, ``writable``,
#: ``extraction``) are the SDK's own ``MemoryStoreConfig`` keys and survive the migration unchanged,
#: which is worth a reader knowing and is therefore in the table rather than filtered out of it.
_COMMUNITY_MEMORY = Source(
    slug="community-memory",
    distribution="strands-mongodb-store",
    read_from=(
        "https://raw.githubusercontent.com/skamalj/strands-agents-session/main"
        "/memory/mongodb/src/strands_mongodb_store/__init__.py"
    ),
    read_on="2026-09-01",
    settings=(
        "name",
        "connection_string",
        "database_name",
        "collection_name",
        "client",
        "description",
        "max_search_results",
        "writable",
        "extraction",
        "model",
        "index_name",
        "num_candidates",
        "create_index",
    ),
    fields=("_id", "content", "metadata", "createdAt"),
)

#: ``strands-mongodb-store``'s ``DEFAULT_MODEL``, from ``__init__.py:28`` of the URL above. It happens to
#: be the same string as this package's ``_indexes.DEFAULT_EMBEDDING_MODEL``, and that coincidence is why
#: it is spelled out here rather than borrowed from ours: two packages agreeing on a default today is not
#: a reason for a test to stop being able to tell them apart.
_COMMUNITY_MEMORY_DEFAULT_MODEL = "voyage-4-lite"

#: ``strands-mongodb-storage`` 0.1.0. Two document fields and nothing else, which is what makes this the
#: one migration that moves no data.
_COMMUNITY_STORAGE = Source(
    slug="community-storage",
    distribution="strands-mongodb-storage",
    read_from=(
        "https://raw.githubusercontent.com/skamalj/strands-agents-session/main"
        "/storage/mongodb/src/strands_mongodb_storage/__init__.py"
    ),
    read_on="2026-09-01",
    settings=("connection_string", "database_name", "collection_name", "client", "prefix"),
    fields=("_id", "data"),
)

_SOURCES = (_ARCHIVED, _COMMUNITY_MEMORY, _COMMUNITY_STORAGE)
_REMOTE_SOURCES = tuple(source for source in _SOURCES if source.read_on)


def _guide() -> str:
    """Return the guide's text.

    Returns:
        The whole file, so a prose claim counts as much as a table cell.
    """
    return _MIGRATION_GUIDE.read_text()


def _table(marker: str) -> str:
    """Return the Markdown table introduced by an HTML comment marker.

    A marker rather than a heading, because a heading is prose a writer will reasonably reword and a
    marker is a handle a writer knows not to touch. The same convention ``test_readme.py`` uses for the
    configuration tables.

    Args:
        marker: Marker text without the comment delimiters, such as ``"settings: tools-memory"``.

    Returns:
        The table's rows, newline-joined, header included.

    Raises:
        AssertionError: If the marker is absent or nothing table-shaped follows it.
    """
    lines = _guide().splitlines()
    needle = f"<!-- {marker} -->"
    try:
        start = lines.index(needle)
    except ValueError:
        raise AssertionError(f"MIGRATION.md has no '{needle}' marker") from None

    rows: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() and not rows:
            continue
        if not _TABLE_ROW.match(line):
            break
        rows.append(line)
    assert rows, f"nothing table-shaped follows '{needle}' in MIGRATION.md"
    return "\n".join(rows)


def _manifest() -> dict[str, Any]:
    """Return ``pyproject.toml`` parsed.

    Returns:
        The manifest as a mapping.
    """
    return tomllib.loads(_PYPROJECT.read_text())


class TestTheGuideReachesThePeopleWhoNeedIt:
    """``AT-098`` / ``REQ-E-098`` — a guide only on GitHub misses everybody who installed from PyPI.

    Which is exactly the population with old data. The manifest assertion is the one that fails while
    somebody is editing ``pyproject.toml``; the sdist assertion is the one that matches what the
    requirement actually says, so both are here.
    """

    def test_the_guide_exists(self) -> None:
        """AT-098 -- named first, because every other test in this file fails obscurely without it."""
        assert _MIGRATION_GUIDE.is_file(), f"MIGRATION.md missing from {_PROJECT_ROOT}"

    def test_the_sdist_include_list_names_the_guide(self) -> None:
        """AT-098 -- the include list is explicit, so a new top-level file is invisible until it is added."""
        include = _manifest()["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
        assert "MIGRATION.md" in include, (
            f"MIGRATION.md is not in the sdist include list {include}; hatch's include is a whitelist, "
            f"so the guide would ship to GitHub and to nobody who installed the package"
        )

    @pytest.mark.build
    @_BUILD_TIER_GATE
    def test_the_built_sdist_contains_the_guide(self, tmp_path: Path) -> None:
        """AT-098 -- the outcome the requirement states, rather than the manifest key that implies it.

        A build backend can honour an include list and still drop a file, and the include list is a proxy
        either way. So this one builds the artifact and looks inside it.

        Through a throwaway venv with ``build`` installed, for the same reason ``test_packaging.py``
        reaches for ``pip wheel``: neither ``build`` nor ``hatchling`` is in the development venv, and
        adding either to make a test pass would make the test dependent on a tool the package does not
        otherwise need. ``pip`` has no sdist verb, so a wheel cannot stand in here either. The wheel does
        not carry ``MIGRATION.md`` at all: hatch's sdist ``include`` governs the sdist, and the wheel
        target is ``packages = ["src/strands_mongodb"]``.

        Built into ``tmp_path`` so the project's own ``dist/`` is left alone.
        """
        root = tmp_path / "buildenv"
        created = subprocess.run([sys.executable, "-m", "venv", str(root)], capture_output=True, text=True, check=False)
        assert created.returncode == 0, created.stdout + created.stderr

        interpreter = root / ("Scripts" if os.name == "nt" else "bin") / "python"
        installed = subprocess.run(
            [str(interpreter), "-m", "pip", "install", "--quiet", "--no-input", "build"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert installed.returncode == 0, installed.stdout + installed.stderr

        destination = tmp_path / "sdist"
        completed = subprocess.run(
            [str(interpreter), "-m", "build", "--sdist", "--outdir", str(destination), str(_PROJECT_ROOT)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"building the sdist failed:\n{completed.stdout}\n{completed.stderr}"

        archives = sorted(destination.glob("*.tar.gz"))
        assert len(archives) == 1, f"expected one sdist in {destination}, found {[a.name for a in archives]}"
        with tarfile.open(archives[0]) as archive:
            members = {Path(name).name for name in archive.getnames()}
        assert "MIGRATION.md" in members, f"the built sdist {archives[0].name} does not contain MIGRATION.md"


class TestBothReadmesPointAtTheGuide:
    """``AT-102`` / ``REQ-E-098`` — a migration guide nobody is linked to is a file in a repository.

    Both files, because they have different readers: the root README is what GitHub renders for somebody
    evaluating the package, and ``python/README.md`` is what ships and what a reader with old data is
    holding.
    """

    @pytest.mark.parametrize("readme", [_PACKAGE_README, _ROOT_README], ids=["package", "root"])
    def test_the_readme_links_to_the_guide(self, readme: Path) -> None:
        """AT-102 -- a Markdown link, not a bare mention, so the reader can actually get there."""
        text = readme.read_text()
        assert re.search(r"\]\([^)]*MIGRATION\.md[^)]*\)", text), (
            f"{readme.name} does not link to MIGRATION.md; a mention without a link is not a pointer"
        )


class TestTheGuideSaysWhereItsClaimsCameFrom:
    """``AT-102`` / ``REQ-E-102`` — two of the three sources can change without this repository noticing.

    The guide cannot be made to notice. It can be made to say when it last looked, which is the
    difference between a reader who can check a claim in a minute and a reader who cannot check it at all.
    """

    @pytest.mark.parametrize("source", _SOURCES, ids=[source.slug for source in _SOURCES])
    def test_the_guide_names_the_source_it_was_read_from(self, source: Source) -> None:
        """AT-102 -- a path for the in-workspace source, a URL for the two that are not."""
        assert source.read_from in _guide(), (
            f"MIGRATION.md makes claims about {source.distribution} without naming {source.read_from}"
        )

    @pytest.mark.parametrize("source", _REMOTE_SOURCES, ids=[source.slug for source in _REMOTE_SOURCES])
    def test_a_source_outside_this_workspace_carries_the_date_it_was_read(self, source: Source) -> None:
        """AT-102 -- the date is what makes a stale claim detectable rather than merely possible."""
        assert source.read_on in _guide(), (
            f"MIGRATION.md cites {source.distribution} without the date it was read ({source.read_on}); "
            f"nothing in this suite can tell a reader whether that source still says what the guide says"
        )

    def test_the_guide_admits_what_no_test_here_can_check(self) -> None:
        """AT-102 -- the limit is stated in the document, not only in this file's docstring.

        Asserted by phrase rather than by structure because there is nothing structural to assert: the
        claim is that the guide is honest about its own staleness, and the only mechanical proxy is that
        it says so where a reader will see it.
        """
        guide = _guide().lower()
        assert "not vendored" in guide or "not fetched" in guide, (
            "MIGRATION.md does not tell the reader that the two community packages are read from "
            "upstream and not checked by this suite (REQ-E-102)"
        )


class TestEverySourcesSurfaceIsMapped:
    """``AT-099`` and ``AT-100`` / ``REQ-E-099`` — completeness measured in the *source's* direction.

    A mapping table that lists the settings with an obvious equivalent and quietly omits the rest is how
    a reader loses ``MONGODB_NAMESPACE``: it has no counterpart here, which is precisely why it needs a
    row saying so. So the assertion is over the source's full surface, and a setting whose answer is
    "there is no equivalent" still has to appear.
    """

    @pytest.mark.parametrize("source", _SOURCES, ids=[source.slug for source in _SOURCES])
    def test_every_setting_has_a_row(self, source: Source) -> None:
        """AT-099 / AT-100 -- every knob the package has, named as that package names it."""
        table = _table(f"settings: {source.slug}")
        missing = [setting for setting in source.settings if setting not in table]
        assert missing == [], (
            f"the {source.distribution} settings table has no row for {missing}. A setting with no "
            f"equivalent here still needs a row: that is the row a migrating reader is looking for"
        )

    @pytest.mark.parametrize("source", _SOURCES, ids=[source.slug for source in _SOURCES])
    def test_every_document_field_has_a_row(self, source: Source) -> None:
        """AT-099 / AT-100 -- and every field it writes, since that is what a transformation reads."""
        table = _table(f"fields: {source.slug}")
        missing = [field for field in source.fields if f"`{field}`" not in table]
        assert missing == [], (
            f"the {source.distribution} document table has no row for {missing}, so a reader writing a "
            f"copy script would not know what to do with those fields"
        )


@_workspace_only
class TestTheArchivedToolsTranscriptIsCurrent:
    """``AT-099`` / ``REQ-E-102`` — the one source that can be checked against code rather than a memo.

    The module is read as text, not imported: it needs ``boto3`` and the ``mongodb-memory`` extra, and
    this package depends on neither, so importing it would make the test's own dependencies wider than
    the thing it tests.
    """

    def test_the_transcribed_environment_variables_are_the_ones_the_module_reads(self) -> None:
        """AT-099 -- a variable added to the plugin fails here rather than being silently unmapped."""
        text = _ARCHIVED_TOOL.read_text()
        read = set(re.findall(r'os\.getenv\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', text))
        transcribed = set(_ARCHIVED.settings)
        assert read <= transcribed, (
            f"{_ARCHIVED_TOOL.name} reads {sorted(read - transcribed)}, which this file's inventory and "
            f"therefore MIGRATION.md do not mention"
        )

    def test_every_default_the_module_declares_is_a_value_in_the_table(self) -> None:
        """AT-099 -- the defaults are what a reader compares against, so the values are checked too.

        Values shorter than five characters are excluded: the plugin's ``DEFAULT_MAX_RESULTS = 10`` and
        ``DEFAULT_EMBEDDING_DIMS = 1024`` would match any table containing any number, and an assertion
        that cannot fail is worse than no assertion. Both are covered by
        :meth:`TestTheGuideStatesTheBehaviourDifferences.test_the_guide_states_the_titan_dimension_count`.
        """
        text = _ARCHIVED_TOOL.read_text()
        declared = re.findall(r'^DEFAULT_[A-Z_]+ = "?([^"\n]+?)"?(?:\s+#.*)?$', text, re.MULTILINE)
        table = _table(f"settings: {_ARCHIVED.slug}")
        missing = [value for value in declared if len(value) >= 5 and value not in table]
        assert missing == [], f"the {_ARCHIVED.distribution} settings table does not carry the defaults {missing}"

    def test_every_transcribed_document_field_is_written_by_the_module(self) -> None:
        """AT-099 -- the inventory checked in the other direction, so it cannot invent a field."""
        text = _ARCHIVED_TOOL.read_text()
        absent = [field for field in _ARCHIVED.fields if f'"{field}":' not in text]
        assert absent == [], f"this file claims {_ARCHIVED_TOOL.name} writes {absent}, and it does not"


class TestTheGuideSaysWhichMigrationsMoveData:
    """``AT-100`` / ``REQ-E-100`` — three sources, and the answer is not the same for all three.

    Storage migrates by pointing the new configuration at the existing collection: the two packages agree
    on ``_id`` and on ``data``, and both normalise keys with the SDK's own function, so each reads the
    other's documents. Both memory paths need a copy, because our read predicate names fields their
    documents do not have. One procedure covering all three would be wrong for two of them.
    """

    @pytest.mark.parametrize("source", _SOURCES, ids=[source.slug for source in _SOURCES])
    def test_each_source_gets_a_verdict(self, source: Source) -> None:
        """AT-100 -- a reader deciding whether to schedule downtime needs this in one place."""
        table = _table("verdict")
        assert source.distribution in table, (
            f"the verdict table does not say whether migrating from {source.distribution} moves data"
        )

    def test_the_verdicts_are_not_all_the_same(self) -> None:
        """AT-100 -- the split is the finding, so a table answering "yes" three times has lost it."""
        table = _table("verdict").lower()
        assert "no" in table and "yes" in table, (
            "the verdict table does not distinguish the storage migration, which moves nothing, from the "
            "two memory migrations, which are copies with a transformation"
        )


class TestTheGuideStatesTheBehaviourDifferences:
    """``AT-100`` / ``REQ-E-099`` — the four gaps that make a memory migration a copy, plus two more.

    Each of these was measured against source rather than reasoned about, and each changes what a reader
    has to do. They are asserted individually because a hazards section that drops one of them still
    looks complete.
    """

    @pytest.mark.parametrize(
        ("needle", "why"),
        [
            ("store_name", "our read predicate names it and neither source writes it"),
            ("content_hash", "deduplication is a unique partial index on it, and neither source has it"),
            ("createdAt", "the community store's timestamp is an ISO string where ours is a BSON date"),
            ("timestamp", "the archived plugin's timestamp is an ISO string where ours is a BSON date"),
            ("_score", "the community store puts the relevance score under a different metadata key"),
        ],
    )
    def test_the_hazard_table_names_it(self, needle: str, why: str) -> None:
        """AT-100 -- one row per difference a reader's code or data has to survive."""
        assert needle in _table("hazards"), f"the hazards table does not mention `{needle}`: {why}"

    def test_the_guide_states_that_a_pipeline_cannot_compute_the_hash(self) -> None:
        """AT-100 -- the cheap migration path has a cost, and it is this one.

        Aggregation has no SHA-256 operator, so a ``$merge`` cannot fill ``content_hash`` in. Migrated
        memories therefore do not deduplicate against newly added identical content until something
        rewrites them from a client. A guide that omitted this would be recommending a path whose
        consequence the reader discovers later.
        """
        guide = _guide()
        assert "content_hash" in guide and re.search(r"sha[- ]?256", guide, re.IGNORECASE), (
            "MIGRATION.md does not explain why content_hash cannot be filled in by an aggregation"
        )

    def test_the_guide_states_the_titan_dimension_count(self) -> None:
        """AT-100 -- the number a reader needs to reuse existing vectors instead of re-embedding."""
        assert "1024" in _guide(), (
            "MIGRATION.md does not give the archived plugin's embedding dimension count, which is what "
            "`embedding_dimensions` has to be set to for existing vectors to remain queryable"
        )

    def test_the_guide_says_the_archived_plugin_is_not_a_memory_store(self) -> None:
        """AT-100 -- the first thing a migrating reader needs told, and the easiest thing to leave out.

        The plugin is one ``@tool`` taking an ``action`` string. ``MongoDBMemoryStore`` is a store the
        SDK's ``MemoryManager`` drives. So the agent's wiring changes shape, and a reader shown only a
        field mapping will map the fields and then find that nothing calls them.
        """
        guide = _guide()
        assert "MemoryManager" in guide, (
            "MIGRATION.md does not explain that the archived plugin is a tool the model calls while this "
            "package is a store the SDK's MemoryManager drives"
        )


#: The archived plugin's search-index definition, transcribed from ``mongodb_memory.py:207-222``. A
#: legacy ``search``-type index with a ``knnVector`` field, under the name all three packages default to.
#: Atlas 8.0.30 accepts it and echoes it back in ``latestDefinition`` unchanged, measured 2026-09-01,
#: which is what makes it usable as the foreign index in the test below.
_LEGACY_INDEX = {
    "name": _indexes.DEFAULT_VECTOR_INDEX_NAME,
    "definition": {
        "mappings": {
            "dynamic": False,
            "fields": {
                "embedding": {"type": "knnVector", "dimensions": 1024, "similarity": "cosine"},
                "namespace": {"type": "string"},
            },
        }
    },
}

#: ``strands-mongodb-store``'s definition, transcribed from ``__init__.py:105-117`` of the source named in
#: ``_COMMUNITY_MEMORY``. A ``vectorSearch`` index like ours, on the same name, over the same ``content``
#: path, differing only in the two ``filter`` entries this package needs for scope and store isolation.
#: The near miss is the point: it is the collision most likely to be mistaken for a match.
_COMMUNITY_STORE_INDEX = {
    "name": _indexes.DEFAULT_VECTOR_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "autoEmbed",
                "modality": "text",
                "path": "content",
                "model": _COMMUNITY_MEMORY_DEFAULT_MODEL,
            }
        ]
    },
}

#: Both foreign definitions, with the substring each one's error is expected to name under
#: ``_MANUAL_VECTOR_CONFIG``. The legacy ``search`` index has no ``fields`` list at all, so the comparator
#: reports the container as absent. The community one has a ``fields`` list, so it reports the first
#: declared field nothing in it covers, which under manual vectors is the ``vector`` field on ``embedding``
#: rather than a ``filter`` entry: the config decides which mismatch is found first, and this pair asserts
#: what each one actually says. The filter-entry message a reader migrating from that package will really
#: see is the ``autoembed`` case below, since that package embeds server-side and so will they.
_FOREIGN_INDEXES = (
    pytest.param(_LEGACY_INDEX, "definition.fields is absent", id="tools-memory"),
    pytest.param(
        _COMMUNITY_STORE_INDEX,
        "nothing reported at definition.fields covers it",
        id="community-memory",
        marks=pytest.mark.autoembed,
    ),
)

#: The same two without their expected substrings, for the test that asserts on the deployment's index set
#: rather than on the message.
_FOREIGN_INDEXES_ONLY = (
    pytest.param(_LEGACY_INDEX, id="tools-memory"),
    pytest.param(_COMMUNITY_STORE_INDEX, id="community-memory", marks=pytest.mark.autoembed),
)

#: Manual-vector mode: the specs *under test* need no registered embedding model, and the collision is
#: about the index *name*, which is the same either way. That is why the class sits on ``integration``.
#:
#: It is not the whole tier story, and reading it as such was a bug (``AT-132``). A tier has to cover the
#: setup as well as the assertion, and one of the two foreign indexes these tests plant *first* is itself
#: an ``autoEmbed`` definition naming a model. Planting it needs that model registered however little the
#: assertion afterwards cares, so that parameter carries ``autoembed`` on top of the class marker.
_MANUAL_VECTOR_CONFIG: MongoDBConfig = {
    "database": "unused",
    "embedding_dimensions": 4,
    "similarity": "cosine",
}


@pytest.mark.integration
class TestTheIndexNameCollisionFailsLoudly:
    """``AT-101`` / ``REQ-E-101``, ``REQ-E-069`` — the sharpest hazard, measured rather than reasoned.

    All three source packages default their vector index to ``vector_index``, and so does this one. Two of
    them also default their collection to a name this package defaults to. So the ordinary migration is a
    reader pointing this package at a collection that already holds a ``vector_index`` of the wrong shape,
    and what happens next is the claim in the guide a reader is most likely to disbelieve: every other
    tool in this space treats a matching name as a matching index.

    Against a deployment, because the whole point is that a real ``mongot`` accepts the foreign definition
    first. A fake that accepts it proves nothing about the one that matters.
    """

    @pytest.mark.parametrize(("foreign", "expected"), _FOREIGN_INDEXES)
    async def test_a_foreign_index_of_the_same_name_raises_naming_the_gap(
        self, foreign: dict[str, Any], expected: str, atlas_database: Any, unique_collection_name: str
    ) -> None:
        """AT-101 -- the error names the missing field and carries both remedies, and drops nothing.

        Two sources rather than one, because they fail differently and the guide claims both. The legacy
        ``search`` index has no ``fields`` list to compare against; ``strands-mongodb-store``'s has one that
        is a strict subset of ours, which is the case a containment check could plausibly get wrong.
        """
        from strands.types.exceptions import StorageError

        collection = atlas_database[unique_collection_name]
        await atlas_database.create_collection(unique_collection_name)
        await collection.create_search_index(foreign)

        specs = _indexes.memory_search_specs(_MANUAL_VECTOR_CONFIG)
        with pytest.raises(StorageError) as caught:
            await _indexes.ensure(collection, search=specs, create=True)

        message = str(caught.value)
        assert expected in message, f"the gap is not named in: {message}"
        assert "dropSearchIndex" in message, "the error does not carry the drop command"
        assert "createSearchIndexes" in message, "the error does not carry the create command"

    @pytest.mark.parametrize("foreign", _FOREIGN_INDEXES_ONLY)
    async def test_nothing_was_dropped_or_created_on_the_way_to_the_error(
        self, foreign: dict[str, Any], atlas_database: Any, unique_collection_name: str
    ) -> None:
        """AT-101 -- the half that makes the guide's advice safe to follow against live data.

        A migration attempt that discarded somebody's built index before reporting the conflict would be
        worse than a silent wrong answer. The foreign index is still there afterwards, and the text index
        this package would otherwise have created is not, because the conflict is raised while the specs
        are being checked and before anything is issued.
        """
        from strands.types.exceptions import StorageError

        collection = atlas_database[unique_collection_name]
        await atlas_database.create_collection(unique_collection_name)
        await collection.create_search_index(foreign)

        with pytest.raises(StorageError):
            await _indexes.ensure(collection, search=_indexes.memory_search_specs(_MANUAL_VECTOR_CONFIG), create=True)

        after = {index["name"] async for index in await collection.list_search_indexes()}
        assert after == {_indexes.DEFAULT_VECTOR_INDEX_NAME}, (
            f"expected the foreign index alone and nothing else, found {sorted(after)}"
        )


class TestEveryForeignIndexIsGatedByWhatPlantingItNeeds:
    """``AT-132`` / ``REQ-E-129`` — a tier covers the setup, not only the assertion.

    Found by running the integration tier against an Atlas CLI local deployment, which ``conftest.py``
    names as a supported target for ``MONGODB_ATLAS_URI``. Both ``[community-memory]`` parameters of
    ``TestTheIndexNameCollisionFailsLoudly`` failed there with ``CanonicalModel: voyage-4-lite not
    registered yet, supported models are: []`` — raised by ``create_search_index`` while planting the
    foreign index, long before the assertion the tier had been chosen for.

    The reasoning that put them on ``integration`` was sound about the specs under test and silent about
    the fixture, so the whole integration tier could not be run clean on its own documented deployment.
    This grades the wiring instead of the reasoning: a parameter whose index embeds server-side is gated
    on ``autoembed``, and one that does not is not. Needs no deployment, so it runs everywhere.
    """

    @staticmethod
    def _embeds_server_side(index: dict[str, Any]) -> bool:
        """Report whether planting ``index`` requires a registered embedding model.

        Args:
            index: A search index definition as it would be handed to ``create_search_index``.

        Returns:
            ``True`` if any declared field asks the deployment to embed, which is what needs the model.
        """
        return any(field.get("type") == "autoEmbed" for field in index["definition"].get("fields", ()))

    @pytest.mark.parametrize("params", [_FOREIGN_INDEXES, _FOREIGN_INDEXES_ONLY], ids=["with-message", "index-set"])
    def test_a_server_side_index_is_gated_on_autoembed_and_a_manual_one_is_not(self, params: tuple[Any, ...]) -> None:
        """AT-132 -- the marker on each parameter matches what planting that parameter costs.

        Both directions, because an over-gated parameter is a test that silently stops running and is
        just as wrong as one that fails.
        """
        for param in params:
            index = param.values[0]
            gated = any(mark.name == "autoembed" for mark in param.marks)
            needs_model = self._embeds_server_side(index)
            assert gated == needs_model, (
                f"{param.id}: planting it {'needs' if needs_model else 'does not need'} a registered "
                f"embedding model, but it is {'gated on' if gated else 'not gated on'} autoembed"
            )


@pytest.mark.autoembed
class TestTheNearMissNamesTheMissingFilters:
    """``AT-101`` / ``REQ-E-101`` — the collision a reader of ``strands-mongodb-store`` will actually hit.

    Its index and this package's default index agree on the one field it declares: ``autoEmbed`` over
    ``content``, same model string. So on this path the comparator gets past the vector field and reports
    what is genuinely missing, which is the two ``filter`` entries scope and store isolation are enforced
    with. The guide quotes that clause, so it is asserted here rather than reasoned from the manual-vector
    message, which names a different field first.

    On the ``autoembed`` tier because the default path needs a registered embedding model. Its own class so
    the marker applies to it alone: the tests above run wherever Atlas Search does.
    """

    async def test_the_message_names_the_absent_filter_entries(
        self, autoembed_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-101 -- a near-identical index is still refused, and the report says which two fields are why."""
        from pymongo import AsyncMongoClient
        from strands.types.exceptions import StorageError

        client: AsyncMongoClient[Any] = AsyncMongoClient(autoembed_uri)
        try:
            collection = client[test_database_name][unique_collection_name]
            await client[test_database_name].create_collection(unique_collection_name)
            await collection.create_search_index(_COMMUNITY_STORE_INDEX)

            specs = _indexes.memory_search_specs({"database": test_database_name})
            with pytest.raises(StorageError) as caught:
                await _indexes.ensure(collection, search=specs, create=True)

            message = str(caught.value)
            assert "'type': 'filter'" in message, f"the absent filter entry is not named in: {message}"
            assert "dropSearchIndex" in message, "the error does not carry the drop command"
        finally:
            await client.drop_database(test_database_name)
            await client.close()
