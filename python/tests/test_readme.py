"""The two READMEs and the migration guide, read as source rather than as prose.

The README acceptance set, ``AT-092`` through ``AT-097``. Every test here reads
``python/README.md``, the repository-root ``README.md`` or ``python/MIGRATION.md`` and compares it against
something that cannot drift silently: a running interpreter, a ``TypedDict``, a command renderer, a
deployment, or ``pyproject.toml``.

The guide was added to this sweep rather than given a sweep of its own, so its procedures obey
the same directive rule as a quickstart. ``test_migration.py`` covers what the guide claims; this file
covers what it runs.

Each fenced block declares what happens to it in an HTML comment on the line above the fence:

======================================  ===================================================================
Directive                               Means
======================================  ===================================================================
``<!-- run: unit -->``                  executed with every tier variable emptied, so no deployment
                                        is reachable
``<!-- run: contract -->``              executed against ``MONGODB_URI``
``<!-- run: autoembed -->``             executed against ``MONGODB_AUTOEMBED_URI``, which needs a
                                        registered Atlas embedding model
``<!-- run: mongosh -->``               a ``db.runCommand`` line, compared against the renderer that
                                        produces it and sent to the deployment where it is cheap
``<!-- skip: <reason> -->``             not executed, and the reason is part of the document
======================================  ===================================================================

A block with an executable language and no directive fails ``AT-097``. That is the point of the
mechanism: executing a snippet is the default, and not executing one is a decision somebody wrote down
where a reviewer can see it.

Snippets read their deployment and database from the environment, which is what makes them runnable
here and is also the standing rule: a literal connection string in a document is how a password
reaches a document.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import pytest
from pymongo import AsyncMongoClient, monitoring
from pymongo.auth import MECHANISMS
from pymongo.errors import OperationFailure
from strands.types.exceptions import StorageError

from strands_mongodb import _indexes, _pipeline
from strands_mongodb.memory_store import DEFAULT_COLLECTION, MongoDBMemoryStore
from strands_mongodb.storage import MAX_DOCUMENT_BYTES, MAX_KEY_BYTES, MongoDBStorage
from strands_mongodb.types import (
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_OVERSAMPLING_FACTOR,
    RESERVED_METADATA_KEYS,
    MongoDBConfig,
    MongoDBMemoryStoreConfig,
    MongoDBSearchQuery,
    ResolvedSearchType,
)

# tests/ -> python/ -> the repository root, which holds the landing-page README.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_README = _PROJECT_ROOT / "README.md"
_ROOT_README = _PROJECT_ROOT.parent / "README.md"
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"

# Languages a reader can paste into something that runs it. `bash` is on the list on purpose: exempting
# it by language would put the exemption in this file, where a reader of the README cannot see it. `text`
# and an unlabelled fence are not, since neither claims to run anywhere.
_EXECUTABLE_LANGUAGES = frozenset({"python", "py", "bash", "sh", "shell", "javascript", "js"})

_DIRECTIVE = re.compile(r"^<!--\s*(?P<kind>run|skip):\s*(?P<value>[^>]+?)\s*-->$")
_FENCE = re.compile(r"^```(?P<language>[A-Za-z0-9_+-]*)\s*$")

# A credential in a document, in any file, in any language. Matches `test_examples.py`'s guard on the
# example output for the same reason: the redaction placeholder is the only thing allowed in the
# userinfo position, and a partial redaction still reads as a leak.
_CREDENTIALED_URI = re.compile(r"mongodb(\+srv)?://(?!<redacted>@)[^\s/`]*@")

# What a documented `db.runCommand(...)` line looks like, so the argument can be compared as JSON
# rather than as text. `_indexes` renders it with `json.dumps`, so the spacing is fixed, but comparing
# parsed arguments means a reader may reformat the block without breaking the test.
_RUN_COMMAND = re.compile(r"^db\.runCommand\((?P<argument>.+)\)$")


@dataclass(frozen=True)
class Block:
    """One fenced code block, with the directive that governs it.

    Attributes:
        path: File the block came from.
        line: 1-based line of the opening fence, so a failure message points at it.
        language: Fence language, lowercased. ``""`` for an unlabelled fence.
        kind: ``"run"``, ``"skip"``, or ``""`` when the block carries no directive.
        value: Tier name for ``run``, reason for ``skip``, ``""`` for neither.
        body: The block's contents, without the fences.
    """

    path: Path
    line: int
    language: str
    kind: str
    value: str
    body: str

    @property
    def where(self) -> str:
        """Return a clickable location for a failure message.

        Returns:
            ``<path>:<line>``, with the path relative to the repository root.
        """
        return f"{self.path.relative_to(_ROOT_README.parent)}:{self.line}"

    @property
    def executable(self) -> bool:
        """Whether this block's language is one a reader can run.

        Returns:
            ``True`` for a language in :data:`_EXECUTABLE_LANGUAGES`.
        """
        return self.language in _EXECUTABLE_LANGUAGES


def _blocks(path: Path) -> tuple[Block, ...]:
    """Parse every fenced code block out of one Markdown file.

    A hand-rolled scan rather than a Markdown library, because the thing being read is a directive in
    an HTML comment immediately above a fence, which is a document convention rather than Markdown
    structure. It also keeps the test suite's dependency set as it is.

    Args:
        path: The Markdown file.

    Returns:
        Every block in document order, each carrying the directive that preceded it, if any.
    """
    found: list[Block] = []
    directive: tuple[str, str] | None = None
    language: str | None = None
    body: list[str] = []
    opened_at = 0

    for number, line in enumerate(path.read_text().splitlines(), start=1):
        fence = _FENCE.match(line)
        if language is None:
            if fence:
                language, opened_at = fence.group("language").lower(), number
                continue
            match = _DIRECTIVE.match(line.strip())
            # A directive only governs the fence directly below it. Anything else between the two
            # resets it, so a comment left behind by an edit cannot silently adopt the next block.
            directive = (match.group("kind"), match.group("value")) if match else None
            continue
        if fence and not fence.group("language"):
            kind, value = directive or ("", "")
            found.append(
                Block(
                    path=path,
                    line=opened_at,
                    language=language,
                    kind=kind,
                    value=value,
                    body="\n".join(body) + "\n",
                )
            )
            directive, language, body = None, None, []
            continue
        body.append(line)

    assert language is None, f"{path.name} has an unclosed fence opened at line {opened_at}"
    return tuple(found)


_PACKAGE_BLOCKS = _blocks(_PACKAGE_README)
_ROOT_BLOCKS = _blocks(_ROOT_README)

# The migration guide joins the sweep rather than getting its own copy of it (`REQ-E-102`).
# Its procedures are the snippets in this repository most likely to be run against real data, so the
# directive rule, the credential rule and the tier runners apply to them by inheritance. What the guide
# *says* is `test_migration.py`'s subject; what it *runs* is this file's.
_MIGRATION_GUIDE = _PROJECT_ROOT / "MIGRATION.md"
_MIGRATION_BLOCKS = _blocks(_MIGRATION_GUIDE)

_DOCUMENTS = (_PACKAGE_README, _ROOT_README, _MIGRATION_GUIDE)
_ALL_BLOCKS = _PACKAGE_BLOCKS + _ROOT_BLOCKS + _MIGRATION_BLOCKS


def _selected(tier: str) -> list[Block]:
    """Return every block whose directive runs it on one tier.

    Args:
        tier: ``"unit"``, ``"contract"``, ``"autoembed"`` or ``"mongosh"``.

    Returns:
        The blocks, in document order across both files.
    """
    return [block for block in _ALL_BLOCKS if block.kind == "run" and block.value == tier]


def _identifiers(blocks: list[Block]) -> list[str]:
    """Return parametrize ids naming the file and line of each block.

    Args:
        blocks: Blocks about to be parametrized over.

    Returns:
        One id per block, so a failure names the location rather than an index.
    """
    return [block.where for block in blocks]


def _run(body: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute one snippet in a child interpreter.

    ``check=False``, so a failing snippet is reported with its own traceback rather than as a
    ``CalledProcessError`` whose message hides it.

    Args:
        body: The snippet, exactly as the README carries it.
        environment: Full environment for the child.

    Returns:
        The completed process.
    """
    return subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        env=environment,
        cwd=_PROJECT_ROOT,
        check=False,
    )


def _unconfigured() -> dict[str, str]:
    """Return an environment with every tier variable emptied.

    A ``unit`` snippet has to run where nothing is reachable, and this process inherits a live
    ``MONGODB_URI`` whenever ``.env`` holds one, so emptying is the only way to test what a reader
    with no cluster sees. Empty rather than unset, matching the suite's own opt-out.

    Returns:
        A copy of this environment with the three tier variables set to ``""``.
    """
    return {**os.environ, "MONGODB_URI": "", "MONGODB_ATLAS_URI": "", "MONGODB_AUTOEMBED_URI": ""}


def _documented_keys(marker: str) -> dict[str, str]:
    """Read one configuration table out of the package README.

    The table is found by an HTML comment naming the type it documents, so a reordering of the README
    cannot make the test read the wrong table.

    Args:
        marker: The type name in ``<!-- keys: <name> -->``.

    Returns:
        Key name to the verbatim ``Required`` cell, for every row of that table.
    """
    lines = _PACKAGE_README.read_text().splitlines()
    try:
        start = lines.index(f"<!-- keys: {marker} -->")
    except ValueError:  # pragma: no cover - the failure is asserted by the caller
        return {}

    rows: dict[str, str] = {}
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"Key", ""} or set(cells[0]) <= {"-", ":"}:
            continue
        rows[cells[0].strip("`")] = cells[2]
    return rows


def _annotation_text(annotation: object) -> str:
    """Return one annotation as the source text the module wrote.

    Three shapes reach here for the same declaration. Under ``from __future__ import annotations`` on
    3.10 through 3.13 the value is a string; on 3.14 PEP 649 hands back a ``ForwardRef`` wrapping that
    string; and an eagerly evaluated annotation is the object itself.

    Args:
        annotation: A value from ``__annotations__``.

    Returns:
        Text containing ``Required[`` when the key was declared required.
    """
    if isinstance(annotation, str):
        return annotation
    return str(getattr(annotation, "__forward_arg__", annotation))


def _annotated_keys(typed_dict: Any) -> tuple[frozenset[str], frozenset[str]]:
    """Return the key set and the required subset of one ``TypedDict``.

    ``__annotations__`` rather than ``get_type_hints``, because ``types.py`` imports
    ``AsyncMongoClient`` under ``TYPE_CHECKING`` and resolving the annotations therefore raises
    ``NameError``. ``TypedDict`` merges ``__annotations__`` across bases, so this still sees the five
    keys ``MongoDBMemoryStoreConfig`` inherits from the SDK, which are part of what a caller passes and
    so part of what the README owes them.

    Required-ness comes from the annotation text and not from ``__required_keys__``, which
    ``test_types.py`` measured as an empty frozenset for all three of these classes: postponed
    evaluation means ``TypedDict`` never sees the ``Required[...]`` wrapper at class creation. Type
    checkers read the source and disagree with the runtime here, and the README documents what the type
    checker enforces.

    Args:
        typed_dict: The class.

    Returns:
        Every key, then the ones declared ``Required``.
    """
    annotations = dict(typed_dict.__annotations__)
    required = {name for name, value in annotations.items() if "Required[" in _annotation_text(value)}
    return frozenset(annotations), frozenset(required)


class TestEveryBlockDeclaresWhatHappensToIt:
    """``AT-097`` -- the sweep that makes every other test in this file mandatory rather than optional.

    Executing the snippets proves the executed ones work. It says nothing about the snippet somebody
    adds next week, which is the one that will be wrong. So the default is execution and an exemption
    has to be written down.
    """

    def test_every_executable_block_carries_a_directive(self) -> None:
        """AT-097 -- a ``python`` or ``javascript`` block with no directive fails here (``REQ-E-097``)."""
        undeclared = [block.where for block in _ALL_BLOCKS if block.executable and not block.kind]
        assert undeclared == [], (
            f"these code blocks declare neither a run tier nor a skip reason: {undeclared}. Add "
            f"'<!-- run: unit -->', '<!-- run: contract -->', '<!-- run: autoembed -->', "
            f"'<!-- run: mongosh -->' or '<!-- skip: <reason> -->' on the line above the fence."
        )

    def test_every_skip_states_a_reason(self) -> None:
        """AT-097 -- ``skip`` with an empty or placeholder reason is the same as no directive."""
        unexplained = [block.where for block in _ALL_BLOCKS if block.kind == "skip" and len(block.value.split()) < 3]
        assert unexplained == [], f"these skips do not say why in a sentence: {unexplained}"

    def test_every_run_directive_names_a_tier_the_suite_runs(self) -> None:
        """AT-097 -- a typo in a tier name would silently exempt a block from every test below."""
        known = {"unit", "contract", "autoembed", "mongosh"}
        unknown = [
            (block.where, block.value) for block in _ALL_BLOCKS if block.kind == "run" and block.value not in known
        ]
        assert unknown == [], f"unrecognised run tiers, which no test below would pick up: {unknown}"

    def test_no_document_in_the_sweep_carries_a_credential(self) -> None:
        """AT-097 -- ``REQ-E-093``: a literal connection string in a document is how a password ships.

        Over the whole file rather than over the code blocks, because prose is where a URI gets pasted
        while somebody explains what a URI looks like. The migration guide is the highest-risk of the
        three: its reader has an old deployment and a new one, so it is the document where somebody is
        most tempted to write both out.
        """
        for path in _DOCUMENTS:
            found = _CREDENTIALED_URI.findall(path.read_text())
            assert found == [], f"{path.name} contains a connection string with credentials in it"

    def test_every_connecting_snippet_reads_its_deployment_from_the_environment(self) -> None:
        """AT-097 -- ``REQ-E-093``, the positive half: the snippets show the habit, not just avoid the leak.

        A reader copies the snippet before reading the paragraph above it, so where the URI comes from is
        the snippet's job to demonstrate.
        """
        for block in _ALL_BLOCKS:
            if block.language != "python":
                continue
            if "connection_string" not in block.body and "AsyncMongoClient(" not in block.body:
                continue
            assert "os.environ" in block.body, (
                f"{block.where} builds a connection without reading the URI from the environment"
            )


class TestTheQuickstartsRun:
    """``AT-092`` -- ``REQ-E-092``: the quickstart is executed from the README text, not transcribed.

    A quickstart is the one snippet every reader runs, so it is the one snippet that must not be stale.
    Copying it into a test file would only prove the copy works.
    """

    def test_there_is_a_quickstart_for_each_backend(self) -> None:
        """AT-092 -- both backends need one, and an empty parametrize below would pass silently."""
        bodies = "\n".join(block.body for block in _ALL_BLOCKS if block.kind == "run")
        assert "MongoDBStorage" in bodies, "no executed snippet constructs a MongoDBStorage"
        assert "MongoDBMemoryStore" in bodies, "no executed snippet constructs a MongoDBMemoryStore"

    @pytest.mark.parametrize("block", _selected("unit"), ids=_identifiers(_selected("unit")))
    def test_the_unit_blocks_run_with_nothing_reachable(self, block: Block) -> None:
        """AT-092 -- import-only and construction-only snippets, run where no deployment answers."""
        completed = _run(block.body, _unconfigured())
        assert completed.returncode == 0, f"{block.where} failed:\n{completed.stderr}"

    @pytest.mark.contract
    @pytest.mark.parametrize("block", _selected("contract"), ids=_identifiers(_selected("contract")))
    def test_the_contract_blocks_run_against_a_deployment(
        self, block: Block, contract_uri: str, test_database_name: str
    ) -> None:
        """AT-092 -- run against ``MONGODB_URI``, in the throwaway database ``conftest`` drops."""
        environment = {**os.environ, "MONGODB_URI": contract_uri, "MONGODB_DATABASE": test_database_name}
        completed = _run(block.body, environment)
        assert completed.returncode == 0, f"{block.where} failed:\n{completed.stderr}"

    @pytest.mark.autoembed
    @pytest.mark.parametrize("block", _selected("autoembed"), ids=_identifiers(_selected("autoembed")))
    def test_the_autoembed_blocks_run_against_a_registered_model(
        self, block: Block, autoembed_uri: str, test_database_name: str
    ) -> None:
        """AT-092 -- the memory quickstart takes the AutoEmbeddings default, so it needs a model.

        Which is also why it is gated: ``MONGODB_AUTOEMBED_URI`` is the narrowest tier in the suite, and
        the snippet a reader is most likely to run is the one this repository can least often run.
        """
        environment = {
            **os.environ,
            "MONGODB_URI": autoembed_uri,
            "MONGODB_AUTOEMBED_URI": autoembed_uri,
            "MONGODB_DATABASE": test_database_name,
        }
        completed = _run(block.body, environment)
        assert completed.returncode == 0, f"{block.where} failed:\n{completed.stderr}"


class TestTheTwoReadmesAgree:
    """``AT-093`` -- the drift guard on the split between the two files.

    Two files stating one fact is the arrangement that rots, and it rots asymmetrically: the file people
    edit stays right and the other one keeps its old number. So the shared facts are compared, and the
    version floors are compared against ``pyproject.toml`` rather than against each other, because two
    files agreeing on a stale number is the failure this is meant to catch.
    """

    def test_the_install_section_is_identical_in_both(self) -> None:
        """AT-093 -- the whole ``## Install`` section, compared verbatim rather than one block out of it.

        This compared ``[0]`` of the blocks containing ``pip install``, which was the entire shared
        surface while the PyPI commands were the only install block. Two things make that too narrow now.
        The section gained a second block -- how to install before the package is published -- and the
        first element of a list is exactly what keeps matching while the part somebody edited drifts. And
        a block comparison never saw the prose between the blocks, where the reason for each command is
        written; the caveat about quoting brackets could disagree between the two files indefinitely.

        The section rather than the file, because these are not the same document: the split makes the root
        file a landing page and ``python/README.md`` the reference, and the reference carries a
        ``pip install -e`` line for contributors that the landing page has no reason to repeat. What the
        two are required to agree on is what they tell a reader to install, and that is this section.
        """
        sections = {}
        for path in (_PACKAGE_README, _ROOT_README):
            text = path.read_text()
            start = text.find("\n## Install\n")
            assert start != -1, f"{path.name} has no '## Install' section, so nothing here is comparing it"
            end = text.find("\n## ", start + 1)
            sections[path] = text[start:end] if end != -1 else text[start:]

        assert sections[_PACKAGE_README] == sections[_ROOT_README], (
            "the two READMEs' Install sections differ. They are the one passage both files carry in full, "
            "so a reader who arrives at either is told the same thing:\n"
            f"--- python/README.md ---\n{sections[_PACKAGE_README]}\n"
            f"--- README.md ---\n{sections[_ROOT_README]}"
        )

    @pytest.mark.parametrize(
        ("row", "declaration"),
        [
            ("Python", 'requires-python = ">=3.10"'),
            ("`strands-agents`", '"strands-agents>=1.54.0"'),
        ],
    )
    def test_both_files_state_the_floor_pyproject_declares(self, row: str, declaration: str) -> None:
        """AT-093 -- a floor stated in three places, checked against the one place that enforces it."""
        assert declaration in _PYPROJECT.read_text(), (
            f"pyproject.toml no longer contains {declaration!r}, so this test is checking the READMEs "
            f"against a floor nothing enforces"
        )
        floor = declaration.split(">=")[1].strip('" ')
        for path in (_PACKAGE_README, _ROOT_README):
            stated = [line for line in path.read_text().splitlines() if line.startswith(f"| {row} |")]
            assert stated, f"{path.name} has no requirements row for {row}"
            assert f">= {floor}" in stated[0], f"{path.name} states {stated[0]!r}, not >= {floor}"

    def test_the_landing_page_points_at_the_reference(self) -> None:
        """AT-093 -- what the root file does instead of repeating: it links, and that is asserted."""
        assert "python/README.md" in _ROOT_README.read_text(), (
            "the root README does not link to python/README.md, so nothing leads a reader to the reference"
        )


class TestTheConfigurationTablesMatchTheTypes:
    """``AT-094`` -- ``REQ-E-094``: every key of the three configuration ``TypedDict``s is documented.

    This is the requirement that paid for itself before it was written. ``filter_fields``,
    ``search_type`` and ``max_score`` were all added during the build, and the scaffold's README
    documented none of them.
    """

    @pytest.mark.parametrize(
        ("marker", "typed_dict"),
        [
            ("MongoDBConfig", MongoDBConfig),
            ("MongoDBMemoryStoreConfig", MongoDBMemoryStoreConfig),
            ("MongoDBSearchQuery", MongoDBSearchQuery),
        ],
    )
    def test_the_table_lists_every_key_and_no_others(self, marker: str, typed_dict: Any) -> None:
        """AT-094 -- both directions, because a key documented and then removed misleads just as well."""
        documented = _documented_keys(marker)
        assert documented, f"no table marked '<!-- keys: {marker} -->' in {_PACKAGE_README.name}"
        keys, _ = _annotated_keys(typed_dict)
        assert set(documented) == set(keys), (
            f"{marker}: undocumented {sorted(keys - set(documented))}, "
            f"documented but gone {sorted(set(documented) - keys)}"
        )

    @pytest.mark.parametrize(
        ("marker", "typed_dict"),
        [
            ("MongoDBConfig", MongoDBConfig),
            ("MongoDBMemoryStoreConfig", MongoDBMemoryStoreConfig),
            ("MongoDBSearchQuery", MongoDBSearchQuery),
        ],
    )
    def test_the_required_column_matches_the_declaration(self, marker: str, typed_dict: Any) -> None:
        """AT-094 -- required-ness read from the annotation text, for the reason in :func:`_annotated_keys`."""
        documented = _documented_keys(marker)
        _, required = _annotated_keys(typed_dict)
        for key, cell in documented.items():
            claimed = cell.strip().lower().startswith("yes")
            assert claimed == (key in required), (
                f"{marker}.{key}: the README says required={claimed!r}, the declaration says {key in required!r}"
            )
        assert required <= set(documented), f"{marker}: required keys missing from the table"


# Every value the limits and defaults sections quote, and the renderings a cell may use for it. Derived
# rather than typed, so a change to the constant changes what the README has to say. `MAX_DOCUMENT_BYTES`
# gets two forms because nobody writes 16777216 in a document and nobody should have to.
_TRACEABLE_VALUES: dict[str, tuple[str, ...]] = {
    "MAX_KEY_BYTES": (str(MAX_KEY_BYTES),),
    "MAX_DOCUMENT_BYTES": (f"{MAX_DOCUMENT_BYTES // 1024 // 1024} MB", str(MAX_DOCUMENT_BYTES)),
    "DEFAULT_MAX_SEARCH_RESULTS": (str(DEFAULT_MAX_SEARCH_RESULTS),),
    "DEFAULT_OVERSAMPLING_FACTOR": (str(DEFAULT_OVERSAMPLING_FACTOR),),
    "DEFAULT_EMBEDDING_MODEL": (_indexes.DEFAULT_EMBEDDING_MODEL,),
    "DEFAULT_VECTOR_INDEX_NAME": (_indexes.DEFAULT_VECTOR_INDEX_NAME,),
    "DEFAULT_TEXT_INDEX_NAME": (_indexes.DEFAULT_TEXT_INDEX_NAME,),
    "MAX_DIMENSIONS": (str(_indexes.MAX_DIMENSIONS),),
    "READY_TIMEOUT_SECONDS": (f"{_indexes.READY_TIMEOUT_SECONDS:g}",),
    "POLL_INTERVAL_SECONDS": (f"{_indexes.POLL_INTERVAL_SECONDS:g}",),
}

# The open questions and requirements `REQ-E-095` names, which have a limit to state and no constant behind
# it. A row cites one of these in its source column instead of a symbol.
_TRACEABLE_IDS = frozenset(
    {
        "REQ-E-017a",
        "REQ-E-017b",
        "REQ-E-017c",
        "REQ-E-091",
        "store search failed",
        "Storage.list()",
        "MongoDBStorage.ensure_indexes()",
    }
)
"""The sources ``REQ-E-095``'s rows cite that are not package constants.

``strands_expires_at_ttl`` was one of them until ``REQ-E-119`` closed it: the row it labelled said no
code path creates the storage TTL index, and ``MongoDBStorage.ensure_indexes()`` is that code path now.
A row citing a method rather than an index name is one a reader can act on without leaving the document.
"""


def _limit_rows() -> list[tuple[str, str, str]]:
    """Read the limits table out of the package README.

    Returns:
        One ``(limit, value, source)`` triple per row, in document order.
    """
    lines = _PACKAGE_README.read_text().splitlines()
    try:
        start = lines.index("<!-- limits -->")
    except ValueError:  # pragma: no cover - the failure is asserted by the caller
        return []

    rows: list[tuple[str, str, str]] = []
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] == "Limit" or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append((cells[0], cells[1], cells[2].strip("`")))
    return rows


class TestTheDocumentedValuesAreTheValues:
    """``AT-095`` -- ``REQ-E-095``: one limits section, every row traceable to the thing that sets it.

    The third column is what makes this testable. A limits section is prose about numbers, and prose
    about numbers is exactly what goes stale; naming the symbol beside each number turns the section into
    something a test can walk.
    """

    def test_the_limits_section_exists_and_every_row_names_its_source(self) -> None:
        """AT-095 -- a row nobody can trace fails, since an untraceable limit cannot be rechecked."""
        rows = _limit_rows()
        assert rows, f"no table marked '<!-- limits -->' in {_PACKAGE_README.name}"
        untraceable = [
            (limit, source)
            for limit, _, source in rows
            if source not in _TRACEABLE_VALUES and source not in _TRACEABLE_IDS
        ]
        assert untraceable == [], (
            f"these limits name a source this test cannot resolve: {untraceable}. Name a constant from "
            f"the package or a requirement id, and add it to _TRACEABLE_VALUES or _TRACEABLE_IDS."
        )

    def test_every_limit_requirement_e_095_names_is_present(self) -> None:
        """AT-095 -- the coverage direction: ``REQ-E-095`` lists eight limits and all eight are rows."""
        cited = {source for _, _, source in _limit_rows()}
        assert _TRACEABLE_IDS <= cited, f"limits dropped from the section: {sorted(_TRACEABLE_IDS - cited)}"

    def test_every_row_quoting_a_constant_quotes_its_current_value(self) -> None:
        """AT-095 -- the drift direction: a ceiling raised in code and not in the document fails here."""
        for limit, value, source in _limit_rows():
            if source not in _TRACEABLE_VALUES:
                continue
            accepted = _TRACEABLE_VALUES[source]
            assert any(form in value for form in accepted), (
                f"the {limit!r} row says {value!r}, and {source} is now one of {accepted}"
            )

    def test_the_defaults_the_readme_quotes_are_the_defaults(self) -> None:
        """AT-095 -- the values a reader plans around, checked wherever in the document they appear."""
        text = _PACKAGE_README.read_text()
        for name in (
            "DEFAULT_EMBEDDING_MODEL",
            "DEFAULT_VECTOR_INDEX_NAME",
            "DEFAULT_TEXT_INDEX_NAME",
            "DEFAULT_MAX_SEARCH_RESULTS",
            "DEFAULT_OVERSAMPLING_FACTOR",
        ):
            assert any(form in text for form in _TRACEABLE_VALUES[name]), (
                f"the README does not state the current {name}, {_TRACEABLE_VALUES[name][0]!r}"
            )
        assert DEFAULT_COLLECTION in text, "the README does not name the default memory collection"
        storage_collection = inspect.signature(MongoDBStorage.__init__).parameters["collection"].default
        assert storage_collection in text, "the README does not name the default storage collection"

    def test_the_reserved_metadata_keys_are_all_listed(self) -> None:
        """AT-095 -- a caller cannot avoid a reserved key that the document does not name."""
        text = _PACKAGE_README.read_text()
        missing = sorted(key for key in RESERVED_METADATA_KEYS if f"`{key}`" not in text)
        assert missing == [], f"reserved metadata keys the README never mentions: {missing}"

    def test_the_authentication_mechanisms_are_the_drivers(self) -> None:
        """AT-095 -- ``REQ-E-093``'s list, taken from the installed driver rather than guessed at.

        ``DEFAULT`` is excluded: it is how the driver is told to negotiate, not a mechanism an operator
        configures on a user.
        """
        text = _PACKAGE_README.read_text()
        expected = {name for name in MECHANISMS if name != "DEFAULT"}
        missing = sorted(name for name in expected if f"`{name}`" not in text)
        assert missing == [], f"the driver supports these and the README does not name them: {missing}"


def _privilege_rows() -> list[tuple[str, dict[str, bool]]]:
    """Read the privileges table out of the package README.

    The header row names one built-in role per column from the third onwards, and each body row claims
    whether that role grants the action in the first column. Both are read from the document so that the
    test checks what the README says rather than what this file remembers it saying.

    Returns:
        One ``(action, {role: granted})`` pair per row, in document order.
    """
    lines = _PACKAGE_README.read_text().splitlines()
    try:
        start = lines.index("<!-- privileges -->")
    except ValueError:  # pragma: no cover - the failure is asserted by the caller
        return []

    roles: list[str] = []
    rows: list[tuple[str, dict[str, bool]]] = []
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            if roles:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= {"-", ":"}:
            continue
        if not roles:
            roles = [cell.strip("`") for cell in cells[2:]]
            continue
        claims = {role: cells[2 + offset].lower().startswith("yes") for offset, role in enumerate(roles)}
        rows.append((cells[0].strip("`"), claims))
    return rows


def _documented_commands() -> list[str]:
    """Return every ``db.runCommand`` line from the README's ``mongosh`` blocks.

    Returns:
        The lines, in document order, with blank lines and comments dropped.
    """
    lines: list[str] = []
    for block in _selected("mongosh"):
        for line in block.body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                lines.append(stripped)
    return lines


def _rendered_commands() -> list[str]:
    """Render the commands ``initialize()`` would need, in the order the README documents them.

    The storage collection first, then the memory collection's ordinary indexes, then its two search
    indexes under the zero-configuration defaults. ``memory_search_specs`` is given an empty config
    beyond the database name, which resolves to the AutoEmbeddings path, matching what
    ``REQ-E-092``'s quickstart does.

    Returns:
        One ``db.runCommand({...})`` string per index the package creates.
    """
    storage_collection = inspect.signature(MongoDBStorage.__init__).parameters["collection"].default
    config: MongoDBConfig = {"database": "agents"}
    return [
        *(spec.creation_command(storage_collection) for spec in _indexes.storage_specs()),
        *(spec.creation_command(DEFAULT_COLLECTION) for spec in _indexes.memory_specs()),
        *(spec.creation_command(DEFAULT_COLLECTION) for spec in _indexes.memory_search_specs(config)),
    ]


# The privilege actions each command the package can send requires, for `AT-116`. An empty set means the
# server authorizes the command without any privilege, which is true of the handshake, the session calls
# and the cursor calls: a client that could not run them could not connect at all, so no role has to
# grant them.
#
# A set rather than one action per command, because the command name and the action are not one to one in
# either direction. `update` is the case that matters here and it is measured rather than reasoned:
# `write` and `add` always upsert, and on 8.3.8 a user holding only `find` and `update` is refused an
# upsert with `Unauthorized` while a plain update succeeds. So this package needs `insert` without ever
# sending an `insert` command, which is why the README table lists that action for `write` and `add`.
#
# The two directions are checked separately: a command absent from this mapping fails the test as
# unmapped, and an action absent from the README fails it as undocumented. Adding a command here without
# adding its action to the README therefore does not silence anything.
_COMMAND_ACTIONS: dict[str, frozenset[str]] = {
    "aggregate": frozenset({"find"}),  # `search` and the memory pipelines; reads, so `find`
    "buildInfo": frozenset(),
    "count": frozenset({"find"}),
    # Sent only where search exists, and only just before `createSearchIndexes`, which fails with code 26
    # against a missing namespace where `createIndexes` would have created it implicitly.
    "create": frozenset({"createCollection"}),
    "createIndexes": frozenset({"createIndex"}),  # the wire name is plural, the action is not
    "createSearchIndexes": frozenset({"createSearchIndexes"}),
    "delete": frozenset({"remove"}),  # the wire name and the action differ here too
    "endSessions": frozenset(),
    "find": frozenset({"find"}),
    "getMore": frozenset(),  # continuing a cursor the `find` or `aggregate` already authorized
    "hello": frozenset(),
    "insert": frozenset({"insert"}),
    "killCursors": frozenset(),
    "listCollections": frozenset({"listCollections"}),
    "listIndexes": frozenset({"listIndexes"}),
    "listSearchIndexes": frozenset({"listSearchIndexes"}),
    "ping": frozenset(),
    # The SCRAM handshake. No role grants these and none could usefully: authentication is what decides
    # which roles the connection has, so it necessarily runs before any of them apply. `saslStart` is
    # often folded into `hello` as speculative authentication and so does not always appear on the wire.
    "saslContinue": frozenset(),
    "saslStart": frozenset(),
    "update": frozenset({"update", "insert"}),  # every write this package sends is an upsert
}


class _CommandNames(monitoring.CommandListener):
    """Record the name of every command the driver sends.

    A listener rather than a proxy or a patched method, because it observes the wire rather than the call
    site: a command the package sends indirectly, through an index helper or a cursor continuation, is
    recorded the same as one it sends directly. That is the whole point of ``AT-116``, since the commands
    a reviewer forgets are the indirect ones.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def started(self, event: Any) -> None:
        """Record a command as it goes out.

        Args:
            event: The driver's ``CommandStartedEvent``.
        """
        self.names.add(event.command_name)

    def succeeded(self, event: Any) -> None:
        """Ignore the reply, since only the request needs a privilege.

        Args:
            event: The driver's ``CommandSucceededEvent``.
        """

    def failed(self, event: Any) -> None:
        """Ignore the failure, which still tells us the command was sent and so needed its action.

        Args:
            event: The driver's ``CommandFailedEvent``.
        """


def _fused_score_examples() -> dict[str, str]:
    """The two fused scores worth writing down, computed rather than remembered (``AT-151``).

    Both documents quote these, and quoting a number is how a document goes stale. Deriving them from
    :data:`_pipeline.RANK_FUSION_PENALTY` here means the test fails when the constant moves, instead of
    passing against prose that is now wrong.

    Returns:
        Each score, rounded the way the documents write it, mapped to what it describes.
    """
    penalty = _pipeline.RANK_FUSION_PENALTY
    both = 2 / (penalty + 1)
    one = 1 / (penalty + 1)
    return {f"{both:.3f}": "ranked first by both tiers", f"{one:.3f}": "ranked first by one tier"}


class TestTheFusedScaleIsStatedWhereABoundIsChosen:
    """AT-151 -- ``REQ-E-146``: the number a caller needs in order to pick a bound for the hybrid tier.

    ``REQ-E-145`` protects a caller on ``"auto"`` by declining fusion when a bound is set. ``REQ-E-146``
    removes that protection the moment they ask for ``"hybrid"`` by name, which is correct -- it is their
    tier -- and leaves them owing a number they have no way to derive. Reciprocal rank fusion is not a
    similarity, there is no 0-to-1 range to reason from, and the ceiling depends on a rank constant that
    lives in the server.

    So both documents state it, and both are checked against :data:`_pipeline.RANK_FUSION_PENALTY` rather
    than against 60. The constant exists to be quoted; if it ever stops being 60, the document that still
    says 60 should fail here rather than mislead someone into a floor that empties their result set.
    """

    def test_the_readme_explains_the_fused_scale_and_quotes_the_live_penalty(self) -> None:
        """AT-151 -- the operator half, in the passage a reader reaches from the tier table."""
        text = _PACKAGE_README.read_text()
        assert "<!-- rrf -->" in text, f"no passage marked '<!-- rrf -->' in {_PACKAGE_README.name}"
        section = text.split("<!-- rrf -->", 1)[1].split("\n## ", 1)[0]

        assert str(_pipeline.RANK_FUSION_PENALTY) in section, (
            f"the RRF passage does not state the rank constant, now {_pipeline.RANK_FUSION_PENALTY}"
        )
        for number, describes in _fused_score_examples().items():
            assert number in section, f"the RRF passage does not state {number}, the score of a document {describes}"
        assert "min_score" in section and "max_score" in section, (
            "the RRF passage does not name the two keys whose values this scale governs"
        )

    def test_the_config_docstring_explains_it_where_the_bound_is_set(self) -> None:
        """AT-151 -- the API half. A caller typing ``min_score=`` is in an editor, not in the README."""
        doc = inspect.getdoc(MongoDBMemoryStoreConfig) or ""
        bounds = doc.split("min_score:", 1)
        assert len(bounds) == 2, "MongoDBMemoryStoreConfig no longer documents min_score"

        assert "RANK_FUSION_PENALTY" in bounds[1], (
            "the min_score documentation does not name the constant holding the rank penalty, so a "
            "reader has to find the scale somewhere else or guess it"
        )
        assert all(number in bounds[1] for number in _fused_score_examples()), (
            "the min_score documentation states no worked value, which is the part that makes the "
            "difference between the two scales obvious rather than merely mentioned"
        )


class TestTheIndexAndPrivilegesReference:
    """``AT-096`` -- ``REQ-E-096``: the block an operator pastes, generated rather than written.

    ``REQ-E-096`` asks for copy-pasteable creation commands so a DBA can provision before first run. A command
    block that does not paste is worse than no block, because it fails in their terminal with our name on
    it. So the block is compared against the renderers ``REQ-E-061`` already ships, and the ordinary
    commands are also sent to a real deployment.
    """

    def test_the_documented_commands_are_the_rendered_commands(self) -> None:
        """AT-096 -- exact equality, in order, against ``creation_command()``."""
        assert _documented_commands() == _rendered_commands(), (
            "the README's command block is not what the package renders. It is generated content: copy "
            "the output of _rendered_commands() rather than editing the block by hand."
        )

    def test_every_index_the_package_creates_has_a_documented_command(self) -> None:
        """AT-096 -- the coverage direction, by name, so a new index cannot ship undocumented."""
        text = _PACKAGE_README.read_text()
        storage_collection = inspect.signature(MongoDBStorage.__init__).parameters["collection"].default
        config: MongoDBConfig = {"database": "agents"}
        names = [
            *(spec.name for spec in _indexes.storage_specs()),
            *(spec.name for spec in _indexes.memory_specs()),
            *(spec.name for spec in _indexes.memory_search_specs(config)),
        ]
        missing = sorted({name for name in names if name not in text})
        assert missing == [], f"indexes this package creates and the README never names: {missing}"
        assert storage_collection in text and DEFAULT_COLLECTION in text

    @pytest.mark.contract
    async def test_the_documented_ordinary_commands_run_on_a_deployment(self, contract_database: Any) -> None:
        """AT-096 -- the strongest form of the claim: the server accepts the lines as written.

        Ordinary indexes only. ``createSearchIndexes`` needs mongot, which the contract tier does not
        promise, and the integration tier already covers that path against Atlas.
        """
        for command in _documented_commands():
            match = _RUN_COMMAND.match(command)
            assert match, f"not a runCommand line: {command}"
            argument = json.loads(match.group("argument"))
            if "createIndexes" not in argument:
                continue
            result = await contract_database.command(argument)
            assert result.get("ok") == 1, f"the deployment rejected a documented command: {command}"

    @pytest.mark.contract
    async def test_the_privileges_table_matches_the_deployment(self, contract_database: Any) -> None:
        """AT-096b -- the measured half of ``REQ-E-096``, rechecked rather than trusted.

        It was assumed that a ``readWrite`` user cannot create indexes. Measured on Atlas 8.0.30 it can,
        including search indexes, which is why the README says so and why this test exists: the claim is
        about a specific server's role definitions, and role definitions are the server's to change.

        Skips rather than fails where ``rolesInfo`` goes unanswered, since a deployment may withhold it
        from the credential the tier was given, and that is not a documentation defect.
        """
        rows = _privilege_rows()
        assert rows, f"no table marked '<!-- privileges -->' in {_PACKAGE_README.name}"

        granted: dict[str, set[str]] = {}
        for role in sorted({role for _, claims in rows for role in claims}):
            try:
                answer = await contract_database.command(
                    {"rolesInfo": {"role": role, "db": contract_database.name}, "showPrivileges": True}
                )
            except OperationFailure as error:
                pytest.skip(f"this deployment does not answer rolesInfo for {role!r}: {error}")
            definitions = answer.get("roles") or []
            if not definitions or "privileges" not in definitions[0]:
                pytest.skip(f"this deployment returns no privilege list for {role!r}")
            granted[role] = {
                action for privilege in definitions[0]["privileges"] for action in privilege.get("actions", ())
            }

        for action, claims in rows:
            for role, claimed in claims.items():
                assert claimed == (action in granted[role]), (
                    f"the README says {role} {'grants' if claimed else 'does not grant'} {action!r}, and "
                    f"this deployment says otherwise"
                )

    @pytest.mark.contract
    async def test_the_package_needs_no_action_the_documented_role_withholds(
        self, contract_uri: str, test_database_name: str, unique_collection_name: str
    ) -> None:
        """AT-116 -- ``REQ-E-113``: every command the package sends is one ``readWrite`` authorizes.

        The table above says what each action is needed for. This says the converse, which is the half
        that rots: nothing the package sends needs an action the table omits. That was verified by
        running the whole suite under a credential holding only ``readWriteAnyDatabase``, and a run is
        evidence that expires. The eleventh command a later change adds would fail at a customer's
        deployment, not here, so the property is asserted against the commands rather than the run.

        Outcomes are deliberately not asserted. Which of these calls succeeds depends on the deployment,
        and a contract-tier deployment promises no search process, so ``search`` on either surface may
        raise. What the test observes is the set of command names the driver sent.
        """
        observed = _CommandNames()
        client: AsyncMongoClient[Any] = AsyncMongoClient(contract_uri, event_listeners=[observed])
        config: MongoDBConfig = {"client": client, "database": test_database_name}
        storage = MongoDBStorage(config, collection=unique_collection_name)
        store = MongoDBMemoryStore(
            name="at116",
            config=config,
            collection=f"{unique_collection_name}_memory",
            scope="at116",
            writable=True,
        )
        try:
            await storage.write("session/at116", b'{"kind": "snapshot"}')
            await storage.read("session/at116")
            await storage.list("session/")
            await storage.search("snapshot")
            await storage.delete("session/at116")
            with contextlib.suppress(StorageError):
                await store.add("a memory this test writes")
            with contextlib.suppress(StorageError):
                await store.search("memory")
        finally:
            await storage.close()
            await store.close()
            await client.close()

        assert observed.names, "the listener saw no commands, so this test proved nothing"
        unmapped = sorted(name for name in observed.names if name not in _COMMAND_ACTIONS)
        assert unmapped == [], (
            f"commands this package sent that {Path(__file__).name} has no privilege mapping for: "
            f"{unmapped}. Add each one with the action it needs, then check the README table still "
            f"covers it"
        )

        documented = {action for action, _ in _privilege_rows()}
        needed = {action for name in observed.names for action in _COMMAND_ACTIONS[name]}
        missing = sorted(action for action in needed if action not in documented)
        assert missing == [], (
            f"the package needs these privilege actions and the README privileges table names none of them: {missing}"
        )


# The nine subjects, in ``REQ-E-118``'s own order, that the README has to cover before the
# package is documented. Each pair is (subject, what locates it), and the locator is a heading rather than a phrase
# so that a section renamed fails here and a sentence reworded does not. Six of the nine are asserted only
# as present, because whether their prose is right is asserted by the tests above, against the code.
_REQUIRED_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("install", "## Install"),
    ("quickstart", "## Quickstart"),
    ("index creation commands", "### Creating them by hand"),
    ("the search tiers", "## Search: what answers a query"),
    ("reserved metadata keys", "### Reserved metadata keys"),
    ("the tenancy models", "## Tenancy"),
    ("the support matrix", "## Requirements"),
    ("the 16 MB document ceiling", "## Limits"),
    ("the erasure path", "## Erasing memories"),
)


class TestTheReadmeCoversEverySubjectTheRequirementNames:
    """``AT-120`` -- ``REQ-E-118`` lists nine subjects, read as a list rather
    than assumed.

    The gap this closes is small and was real. It is one sentence naming nine things, and until
    this test was written nothing asserted eight of them; the README happened to cover all nine, so
    a deleted section would have taken a subject with it and no test would have noticed. Deliberately the
    weakest assertion in this file: presence of a section, plus one value check where the subject is a
    number the code owns. Everything about whether the prose is *correct* is asserted elsewhere, against
    the code, and repeating it here would be a second place for the same fact to drift.
    """

    def test_every_subject_has_a_section(self) -> None:
        """A subject with no section is an undocumented subject, whatever the rest of the file says."""
        headings = {line.rstrip() for line in _PACKAGE_README.read_text().splitlines() if line.startswith("#")}
        absent = {subject: heading for subject, heading in _REQUIRED_SUBJECTS if heading not in headings}

        assert absent == {}, f"required subjects whose section is gone from {_PACKAGE_README.name}: {absent}"

    def test_the_nine_subjects_are_nine(self) -> None:
        """The list above is a transcription of one sentence, and a transcription can lose a clause."""
        assert len(_REQUIRED_SUBJECTS) == 9
        assert len({heading for _, heading in _REQUIRED_SUBJECTS}) == 9

    def test_the_document_ceiling_is_stated_as_a_number(self) -> None:
        """Item 24 asks for the caveat, not for a heading. 16 MB is the one subject that is a value.

        The value comes from the package rather than from this file, so a ceiling changed in code and not
        in the document fails here. Which duplicates ``AT-095``'s limits-table check on purpose: that one
        reads the table by its marker, and this one reads the section, so deleting the table fails one and
        deleting the section fails the other.
        """
        section = _PACKAGE_README.read_text().split("## Limits", 1)[1].split("\n## ", 1)[0]
        megabytes = MAX_DOCUMENT_BYTES // (1024 * 1024)

        assert f"{megabytes} MB" in section, (
            f"the limits section does not state the {megabytes} MB ceiling that MAX_DOCUMENT_BYTES sets"
        )


def _marked_section(marker: str) -> str:
    """Return the lines following an HTML marker comment, up to the next heading.

    The marker is what makes a passage addressable. A test that searched the whole document for a
    sentence would pass when the sentence drifted into an unrelated section, and a test that searched by
    heading would fail the day somebody rewords the heading.

    Args:
        marker: The comment's inner text, without the ``<!--`` and ``-->``.

    Returns:
        The section body, or ``""`` when the marker is absent.
    """
    lines = _PACKAGE_README.read_text().splitlines()
    try:
        start = lines.index(f"<!-- {marker} -->")
    except ValueError:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        body.append(line)
    return "\n".join(body)


def _verify_rows() -> list[tuple[str, str, str]]:
    """Read the ensure-versus-verify table out of the package README.

    Returns:
        One ``(index kind, what ensure does, what verify does)`` triple per row, in document order.
    """
    rows: list[tuple[str, str, str]] = []
    for line in _marked_section("verify").splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or set("".join(cells)) <= {"-", ":"} or cells[0].startswith("Index"):
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


class TestTheManualVectorPathIsDocumentedAtItsRealWidth:
    """``AT-128`` -- ``REQ-E-125``: a read half that works and no write half.

    That gap was never an API defect. A `vector` keyword on `add` would be reachable only by a caller who
    has already stepped around `MemoryManager`, and that caller can write the document. What was wrong
    was the document: it offered `embedding_dimensions` and `similarity` as two more configuration keys
    and never said that the field they index is one nothing in this package fills.

    So both directions are asserted. The README has to *say* it, and the fact it states has to still be
    true of the code -- the day `add` grows a vector parameter, the sentence becomes a lie and this class
    is what notices.
    """

    def test_the_readme_names_the_field_the_index_covers(self) -> None:
        """AT-128 -- the name comes from the package, so renaming the field fails here.

        A reader who has to supply the vector needs the field name above everything else, and it appears
        nowhere in the document today: `embedding_dimensions` is a config key, not a path.
        """
        section = _marked_section("manual-vectors")
        assert section, f"no passage marked '<!-- manual-vectors -->' in {_PACKAGE_README.name}"
        assert f"`{_indexes.VECTOR_FIELD}`" in section, (
            f"the manual-vector passage never names {_indexes.VECTOR_FIELD!r}, the field "
            f"_indexes.VECTOR_FIELD makes the index cover"
        )

    def test_the_readme_says_the_package_never_produces_a_vector(self) -> None:
        """AT-128 -- the statement `REQ-E-125` asks for, read as two clauses rather than one sentence."""
        section = _marked_section("manual-vectors").lower()
        for clause, phrases in (
            ("that this package indexes and queries the field", ("index", "quer")),
            ("that this package never produces a vector", ("never",)),
            ("that writing the field is the caller's", ("you", "caller")),
        ):
            assert any(phrase in section for phrase in phrases), f"the manual-vector passage does not state {clause}"

    def test_add_has_no_vector_parameter_for_the_readme_to_offer(self) -> None:
        """AT-128 -- the negative half of ``REQ-E-125``, asserted against the signature it is about.

        ``REQ-E-125`` forbids presenting ``embedding_dimensions`` as something ``add`` can fill. This is
        the load-bearing half: prose can be re-read, but the reason the prose is correct is that the
        parameter does not exist, and if somebody adds one the document has to change with it.
        """
        parameters = set(inspect.signature(MongoDBMemoryStore.add).parameters)

        assert not parameters & {"vector", "embedding", "embeddings"}, (
            f"MongoDBMemoryStore.add now takes {sorted(parameters & {'vector', 'embedding', 'embeddings'})}, "
            f"so the write half exists and the README's manual-vector passage is out of date"
        )

    def test_the_passage_sits_where_a_reader_configuring_the_manual_path_will_be(self) -> None:
        """AT-128 -- placement, because a true sentence in the wrong section is not a statement.

        The two keys that select the path are in the `MongoDBConfig` table; the consequence of selecting
        it belongs with the search tiers, which is where a reader goes to find out what answers a query.
        """
        text = _PACKAGE_README.read_text()
        search = text.index("## Search: what answers a query")
        marker = text.index("<!-- manual-vectors -->")
        following = text.index("\n## ", search + 1)

        assert search < marker < following, (
            "the manual-vector passage is outside the search section, where a reader who has just set "
            "embedding_dimensions is not looking"
        )


class TestBothHalvesOfVerifyModeAreStated:
    """``AT-129`` -- ``REQ-E-126``: the section said verify mode never raises, and it always could.

    The sentence was wrong about both halves. ``_ensure_ordinary`` raises on a missing ordinary index and
    always has (``REQ-E-061``), and ``_ensure_search`` raises too; what is tolerant is one level up, in
    ``_ensure_search_tiers``, and only under ``search_type: "auto"``, where a failed tier is dropped
    because another one may answer. A reader planning a no-DDL deployment needs to know which of their
    indexes stops startup and which one only degrades it, and the old text told them neither.

    The search half's *behaviour* is asserted in ``test_memory_store.py``, where the warning is raised;
    this class asserts the document, plus the one fact the document promises about the ordinary half that
    can be computed here without a deployment.
    """

    def test_the_table_covers_every_index_kind_verify_mode_treats_differently(self) -> None:
        """AT-129 -- three behaviours, so a two-row table has folded two of them together."""
        rows = _verify_rows()
        assert rows, f"no table marked '<!-- verify -->' in {_PACKAGE_README.name}"
        kinds = [kind.lower() for kind, _, _ in rows]

        assert len(rows) == 3, f"verify mode has three behaviours and the table has {len(rows)}: {kinds}"
        assert any("ordinary" in kind for kind in kinds), f"no row for ordinary indexes: {kinds}"
        assert sum("search" in kind for kind in kinds) == 2, (
            f"the two search rows are what separate 'auto' from a named search_type: {kinds}"
        )

    def test_the_ordinary_row_says_it_raises_and_the_auto_row_says_it_warns(self) -> None:
        """AT-129 -- the statement itself, read per row rather than as a search of the whole section."""
        rows = {kind.lower(): verify.lower() for kind, _, verify in _verify_rows()}
        ordinary = next((verify for kind, verify in rows.items() if "ordinary" in kind), "")
        auto = next((verify for kind, verify in rows.items() if "auto" in kind), "")
        named = next((verify for kind, verify in rows.items() if "search" in kind and "auto" not in kind), "")

        assert "rais" in ordinary, f"the ordinary row does not say a missing index raises: {ordinary!r}"
        assert "warn" in auto, f"the auto search row does not say a missing index warns: {auto!r}"
        assert "rais" in named, f"the named search_type row does not say a missing index raises: {named!r}"
        assert "warn" not in ordinary, (
            f"the ordinary row still describes the search half's warning, which is the confusion "
            f"REQ-E-126 is about: {ordinary!r}"
        )

    def test_the_sentence_that_said_verify_never_raises_is_gone(self) -> None:
        """AT-129 -- the regression direction, since a corrected table under a wrong paragraph is worse.

        A reader who reaches the paragraph first stops there. Phrased against the claim rather than the
        wording, so a reworded version of the same wrong claim fails too.
        """
        text = _PACKAGE_README.read_text().lower()
        wrong = [phrase for phrase in ("never raises", "does not raise", "never fails") if phrase in text]

        assert wrong == [], f"the README still claims verify mode cannot raise: {wrong}"

    def test_the_ordinary_rows_promised_message_is_the_message_the_code_builds(self) -> None:
        """AT-129 -- the fact check: the row promises a work order, and this is the work order.

        Computed rather than quoted. ``_missing_ordinary_message`` is what a caller in verify mode reads,
        and if it ever stops carrying the creation commands the row's promise is empty.
        """
        specs = _indexes.storage_specs()
        message = _indexes._missing_ordinary_message("storage", "appdb.storage", specs)

        assert "ensure_indexes is False" in message
        for spec in specs:
            assert spec.creation_command("storage") in message, (
                f"the verify-mode message omits the command that creates {spec.name!r}, so the README's "
                f"ordinary row promises something the package does not send"
            )


class TestTheTtlCaveatNamesTheMethodThatCreatesTheIndex:
    """``AT-128``'s sibling -- ``REQ-E-119``'s documentation half, the last place that gap was recorded.

    Two rows said no code path creates the storage TTL index. One of them was in the limits table, cited
    to ``strands_expires_at_ttl``, which is how a reader was expected to find out that ``ttl_seconds``
    expired nothing. ``ensure_indexes()`` is that code path now, so both rows name it and the citation
    moves from an index name to the method, which a reader can call rather than read about.
    """

    def test_the_limits_row_cites_the_method(self) -> None:
        """AT-128 -- the row is still there and its source is the method rather than the id."""
        sources = {source for _, _, source in _limit_rows()}

        assert "strands_expires_at_ttl" not in sources, "the limits table still cites the bare index name"
        assert "MongoDBStorage.ensure_indexes()" in sources, (
            "no limits row names ensure_indexes(), so a reader whose ttl_seconds documents are not "
            "expiring has nothing to call"
        )

    def test_the_method_the_readme_names_is_a_method(self) -> None:
        """AT-128 -- a citation to a name that does not resolve is worth less than no citation."""
        assert callable(getattr(MongoDBStorage, "ensure_indexes", None)), (
            "the README names MongoDBStorage.ensure_indexes() and the class does not have it"
        )

    def test_the_index_table_row_no_longer_says_nothing_creates_it(self) -> None:
        """AT-128 -- the other row, in the index table, which said the same thing in different words."""
        text = _PACKAGE_README.read_text()
        section = text.split("### What gets created", 1)[1].split("\n### ", 1)[0]
        ttl = next(line for line in section.splitlines() if "strands_expires_at_ttl" in line)

        assert "nothing creates this one for you" not in ttl, (
            f"the index table still says nothing creates the storage TTL index: {ttl!r}"
        )
        assert "ensure_indexes()" in ttl, f"the index table's TTL row does not name the method: {ttl!r}"


class TestTheReadmeSaysWhichSurfaceCreatesTheTtlIndex:
    """``AT-158`` -- ``REQ-E-149``'s documentation half, made necessary and false in one step.

    There was once one TTL story and the README told it once: you stamp ``expires_at``, nothing creates
    the index, call ``ensure_indexes()``. The memory store has the opposite arrangement --
    ``initialize()`` creates the index whether or not any store sets a ``ttl_seconds`` -- which turned
    the single story into a claim that is true of one surface and wrong about the other. A reader who
    believes the unscoped version calls a method that does not exist on ``MongoDBMemoryStore``; a reader
    who believes its inverse leaves storage documents unexpiring forever. So both rows name their
    surface.
    """

    def test_the_limits_row_scopes_itself_and_names_the_other_surface(self) -> None:
        """AT-158 -- the caveat is about `MongoDBStorage`, and it says so before it says anything else."""
        row = next((limit, value) for limit, value, _ in _limit_rows() if "ttl_seconds" in limit and "index" in limit)
        limit, value = row

        assert "MongoDBStorage" in limit, (
            f"the limits row reads as a claim about both surfaces, and it is false on one: {limit!r}"
        )
        assert "initialize()" in value, (
            f"the row does not say the memory store creates its own TTL index, so a reader goes looking "
            f"for an ensure_indexes() that surface does not have: {value!r}"
        )

    def test_the_memory_index_row_names_the_method_that_creates_it(self) -> None:
        """AT-158 -- the index table's memory row, which said what expires and not who armed it."""
        text = _PACKAGE_README.read_text()
        section = text.split("### What gets created", 1)[1].split("\n### ", 1)[0]
        rows = [line for line in section.splitlines() if "strands_expires_at_ttl" in line]

        assert len(rows) == 2, f"expected a TTL row per collection, found {len(rows)}: {rows!r}"
        memory = next(line for line in rows if line.lstrip("| ").startswith("`memory`"))

        assert "initialize()" in memory, (
            f"the memory TTL row does not say initialize() creates it, which is the whole point: {memory!r}"
        )


class TestTheLandingPageDescribesThePackageAsItIs:
    """``AT-161`` / ``REQ-E-155`` -- the root README's status block, which nothing checked.

    ``test_the_landing_page_points_at_the_reference`` is the only assertion this file made about the
    root document, and it checks that the file links to the reference rather than anything it says. So
    the landing page accumulated claims that were true when written and were not revisited when the
    package moved: that a store with a ``ttl_seconds`` never expires anything, which stopped being true
    when ``MongoDBStorage.ensure_indexes()`` arrived; and that ``search()`` spans a vector tier and a
    text tier, which the ``hybrid`` tier made two-thirds of the answer.

    It is the first page a reader of the public repository sees, and it was telling them a shipped
    feature silently does nothing. ``AT-158`` holds the *package* README to the TTL story; this holds
    the landing page to the same one, and to the tier list the package actually implements.

    Both assertions read the claim out of the source rather than restating it here, so a third tier or a
    second surface that needs provisioning fails this file instead of ageing quietly in the document.
    """

    def test_a_ttl_claim_names_the_call_that_arms_it(self) -> None:
        """AT-161 -- the paragraph that mentions ``ttl_seconds`` says what makes it act."""
        text = _ROOT_README.read_text()
        paragraphs = [
            " ".join(line.lstrip("> ").strip() for line in block.splitlines()) for block in re.split(r"\n>?\s*\n", text)
        ]
        mentions = [p for p in paragraphs if "ttl_seconds" in p]

        assert mentions, "the landing page no longer mentions ttl_seconds; drop this test or the claim"
        for paragraph in mentions:
            assert "ensure_indexes()" in paragraph or "initialize()" in paragraph, (
                f"the landing page discusses ttl_seconds without naming the call that provisions the "
                f"index it needs, which is the difference between a working feature and a silent "
                f"no-op: {paragraph!r}"
            )

    def test_every_tier_that_can_answer_a_query_is_named(self) -> None:
        """AT-161 -- read the tiers out of ``ResolvedSearchType``, so adding one fails here."""
        tiers = get_args(ResolvedSearchType)
        text = _ROOT_README.read_text()

        assert len(tiers) >= 2, f"ResolvedSearchType no longer lists tiers to check: {tiers!r}"
        missing = [tier for tier in tiers if tier not in text]

        assert not missing, (
            f"the landing page describes the search surface without naming {missing}; it is the first "
            f"page a reader sees and it would undersell what the package resolves to"
        )
