"""The two documents a contributor arrives at, read as claims about this repository.

The conventions acceptance set, ``AT-112`` through ``AT-115`` (``REQ-E-111`` and
``REQ-E-112``). ``AGENTS.md`` and ``CONTRIBUTING.md`` sit at the repository root and describe how to work
here: the commands, the test tiers, the gates, the house conventions.

**Why these are tests and not prose.** A conventions document is copied from somewhere, and a copied
document's failure mode is not bad writing. It is naming a command, an environment variable, a marker, a
version floor, a path or a CI job that this repository does not have, which sends the first contributor
to debug their own machine over a sentence that was never true here. So every such claim is resolved
against the file that owns it: ``hatch run <script>`` against ``pyproject.toml``, a tier variable against
``conftest.TIER_VARIABLES``, a marker against the registered list, a Python version against
``requires-python`` and the matrix, a backticked path against the filesystem, a job name against the
workflow.

``AT-115`` goes the other way. It takes the nine conventions ``AGENTS.md`` states and checks the package
obeys them, because a convention nothing measures is a preference. Two of the nine are review-only and
say so. One is stated as a deviation from the SDK's own rule rather than as a copy of it, and the
deviation is the decision recorded in ``REQ-E-112``, not an oversight this file should report.

``AT-181`` came later, when the two sections that had shipped as placeholders got their answers. It asks a
different kind of question from the rest of the file: not whether a name resolves against this repository,
but whether an address reaches the company that can act on the report.

What this file cannot tell you: whether the documents are *useful*. Every claim here is a resolvable
reference, and a document can pass all of it while explaining nothing.
"""

from __future__ import annotations

import ast
import asyncio
import gc
import inspect
import re
import sys
import warnings
import weakref
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from conftest import TIER_VARIABLES
from pymongo.periodic_executor import AsyncPeriodicExecutor

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    # `tomli` is the parser `tomllib` was adopted from, and 2.x is the API `tomllib` shipped, so the
    # alias makes the two interchangeable and every caller below reads one name. It is a dev dependency
    # declared without a version marker on purpose; `pyproject.toml` says why, and the reason is mypy
    # rather than this branch.
    import tomli as tomllib

# tests/ -> python/ -> the repository root, which is where a contributor lands.
_PACKAGE = Path(__file__).resolve().parent.parent
_ROOT = _PACKAGE.parent
_AGENTS = _ROOT / "AGENTS.md"
_CONTRIBUTING = _ROOT / "CONTRIBUTING.md"
_PYPROJECT = _PACKAGE / "pyproject.toml"
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci-python.yml"
_GITIGNORE = _ROOT / ".gitignore"
_SECURITY = _ROOT / "SECURITY.md"
_SOURCE = _PACKAGE / "src" / "strands_mongodb"

#: The two documents every test below reads. Named separately above because most tests want one of them.
_CONVENTION_DOCUMENTS: Final = (_AGENTS, _CONTRIBUTING)

#: The gate variable that is not a tier: ``build`` needs network and no deployment, so it carries no URI
#: and cannot come from ``TIER_VARIABLES``.
_BUILD_VARIABLE: Final = "STRANDS_MONGODB_BUILD_TIER"

#: Directories the Markdown sweep does not enter: they hold copies of files this repository does not
#: own, or working notes that are not part of the package.
_UNSWEPT_DIRECTORIES: Final = frozenset(
    {".git", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__", "node_modules"}
)

#: Every Markdown file that ships or is read in place, checked against the sweep so that a walk which
#: silently found nothing cannot pass as a clean sweep.
_SWEPT_AT_LEAST: Final = (
    "AGENTS.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "python/MIGRATION.md",
    "python/README.md",
    "python/examples/README.md",
)

# A credential in a document, in any file. The same expression ``test_readme.py`` and
# ``test_examples.py`` use, for the same reason: the redaction placeholder is the only thing allowed in
# the userinfo position, and a partial redaction still reads as a leak.
_CREDENTIALED_URI = re.compile(r"mongodb(\+srv)?://(?!<redacted>@)[^\s/`]*@")

# What a path looks like inside backticks. Deliberately narrow: a span with a space, a `*`, an `=` or a
# `:` in it is a command, a glob or a URL rather than a file, and the existence check has nothing to say
# about those.
_PATH_SHAPE = re.compile(r"^[\w.][\w./-]*/?$")
_PATH_SUFFIXES: Final = frozenset({".md", ".py", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".env", ".lock"})

#: Paths the documents name that a clone does not contain, each one a thing a contributor creates. The
#: exemption is checked rather than trusted: ``AT-113`` asserts ``.gitignore`` is the reason.
_ABSENT_FROM_A_CLONE: Final = (".env", "python/.venv")

#: The placeholders this repository has declared open. Empty of real ones today: the code of conduct and
#: the security address were the two, and both are answered in ``CONTRIBUTING.md`` and ``SECURITY.md``.
#: ``...`` is the shape ``AGENTS.md`` teaches when it explains the convention, not a placeholder anyone
#: has to settle.
_OPEN_PLACEHOLDERS: Final = frozenset({"..."})

_HATCH_SCRIPT = re.compile(r"hatch run ([a-z][\w-]*)")
_TIER_VARIABLE = re.compile(r"\b(MONGODB_[A-Z_]+|STRANDS_MONGODB_[A-Z_]+)\b")
_MARK_ATTRIBUTE = re.compile(r"pytest\.mark\.(\w+)")
_DASH_M = re.compile(r"-m\s+[\"']?([a-z_]+(?:\s+(?:and|or|not)\s+[a-z_]+)*)")
_PYTHON_VERSION = re.compile(r"\b3\.\d+\b")
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*$", re.MULTILINE)
_TODO = re.compile(r"TODO\(([^)]*)\)")

#: A mail address in a document. ``AT-181`` reads these to check which company a report would reach.
_MAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

#: The one ``user@host`` in these documents that is not a mailbox: the SSH form of a git remote, which
#: ``CONTRIBUTING.md`` names in its clone command. Excluded by shape -- the trailing colon a remote has and
#: an address does not -- rather than by naming ``git@github.com``, so a real address at that host would
#: still be reported by the check below.
_SSH_REMOTE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}:")

#: The address the owner supplied on 2026-09-17 for a private security report (``REQ-E-176``).
#: Declared here rather than read out of whichever document is being checked, so that the two documents are
#: compared against one string and changing it stays a deliberate edit rather than a typo that propagates.
_SECURITY_ADDRESS: Final = "mongodb-partners@mongodb.com"
_TABLE_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|", re.MULTILINE)


def _text(path: Path) -> str:
    """Read a document, or fail naming what is missing.

    Absence is the defect this file was written to catch, so it is an assertion rather than a skip: a
    skipped test reports the same green as a passing one.

    Args:
        path: File to read.

    Returns:
        The file's text.
    """
    assert path.is_file(), f"REQ-E-111: {path.relative_to(_ROOT)} is not in the repository"
    return path.read_text()


def _headings(text: str) -> list[str]:
    """Every ATX heading in a document, lowercased.

    Args:
        text: The document.

    Returns:
        Heading texts, in document order, without their leading hashes.
    """
    return [match.group(1).strip().lower() for match in _HEADING.finditer(text)]


def _covers(text: str, alternatives: tuple[str, ...]) -> bool:
    """Whether some heading names one of these topics.

    Substrings rather than exact headings, because the wording of a heading is the author's and the
    subject is the contract. ``("tier",)`` is satisfied by "Test tiers" and by "The tiers and what they
    need"; it is not satisfied by a document that never mentions tiers.

    Args:
        text: The document.
        alternatives: Lowercased substrings, any one of which counts.

    Returns:
        ``True`` if at least one heading contains at least one alternative.
    """
    return any(alternative in heading for heading in _headings(text) for alternative in alternatives)


def _config() -> dict[str, Any]:
    """Parse ``pyproject.toml``.

    Returns:
        The parsed document.
    """
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _scripts() -> set[str]:
    """Every hatch script a contributor can run.

    Returns:
        The keys of ``[tool.hatch.envs.default.scripts]``.
    """
    return set(_config()["tool"]["hatch"]["envs"]["default"]["scripts"])


def _markers() -> set[str]:
    """Every marker ``--strict-markers`` will accept.

    Returns:
        The marker names, without their descriptions.
    """
    declared = _config()["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0].strip() for entry in declared}


def _supported_versions() -> set[str]:
    """Every Python version this package claims, from the two places that claim it.

    The classifiers are the published claim and the CI matrix is the tested one. A version in a document
    that is in neither is a support promise nothing backs.

    Returns:
        Version strings such as ``"3.10"``.
    """
    configured = _config()
    classified = {
        entry.rsplit(" ", 1)[1]
        for entry in configured["project"]["classifiers"]
        if entry.startswith("Programming Language :: Python :: 3.")
    }
    matrix = {
        str(version)
        for version in yaml.safe_load(_WORKFLOW.read_text())["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    }
    return classified | matrix | {configured["project"]["requires-python"].lstrip(">=")}


def _canonical_job(name: str) -> str:
    """Cut a job name at its first variable part, so both sides of a comparison can be compared.

    ``Test - Python ${{ matrix.python-version }}`` in the workflow and ``Test - Python <version>`` in a
    document are the same job named for two audiences. Everything from the first ``${{`` or ``<`` is the
    part that varies.

    Args:
        name: A job name from either side.

    Returns:
        The literal prefix, stripped.
    """
    return re.split(r"\$\{\{|<", name)[0].strip()


def _path_candidates(text: str) -> list[str]:
    """Every backticked span that claims to be a file or a directory in this repository.

    Args:
        text: The document.

    Returns:
        The candidate spans, in document order, without duplicates.
    """
    found: list[str] = []
    for span in _BACKTICKED.findall(text):
        if "*" in span or not _PATH_SHAPE.match(span):
            continue
        if "/" not in span and Path(span).suffix not in _PATH_SUFFIXES:
            continue
        if span not in found:
            found.append(span)
    return found


def _source_modules() -> list[Path]:
    """Every module in the package, which is what the conventions apply to.

    Returns:
        The module paths, sorted.
    """
    modules = sorted(_SOURCE.glob("*.py"))
    assert len(modules) >= 10, f"the conventions sweep found {len(modules)} modules, which is not this package"
    return modules


def _public_definitions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Module-level functions and the methods of public classes, and nothing nested inside them.

    The definition ``REQ-E-112`` records, and it has to be stated rather than implied: the decorator
    factories in ``memory_store.py`` and ``storage.py`` each hold two inner functions with public names
    and no docstring, and a rule that called those public would report four violations of a convention
    nobody meant to apply to closures.

    Args:
        tree: A parsed module.

    Returns:
        The definitions the conventions govern.
    """
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.append(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            found.extend(child for child in node.body if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef))
    return [node for node in found if not node.name.startswith("_") or node.name == "__init__"]


def _markers_named(text: str) -> set[str]:
    """Every marker a document tells a reader to use.

    Two forms: ``pytest.mark.<name>`` in a snippet, and ``-m <expression>`` on a command line. The second
    is read per line and only on lines that run pytest, because ``-m`` is also how a module is run, and
    ``python -m pytest -m contract`` therefore yields ``pytest`` as well. That one is subtracted by name
    rather than by pattern, so a future marker actually called ``pytest`` would be caught by the reverse
    check below instead of silently exempted here.

    Args:
        text: The document.

    Returns:
        The marker names, boolean operators removed.
    """
    found = set(_MARK_ATTRIBUTE.findall(text))
    for line in text.splitlines():
        if "pytest" not in line and "hatch run test" not in line:
            continue
        for expression in _DASH_M.findall(line):
            found.update(token for token in expression.split() if token not in {"and", "or", "not"})
    return found - {"pytest"}


#: The subjects ``AGENTS.md`` has to cover, as heading substrings. Any one alternative satisfies its row.
_AGENTS_TOPICS: Final = {
    "what is in the repository and where": ("layout",),
    "the commands": ("command",),
    "the test tiers and what each one needs": ("tier",),
    "the house conventions": ("convention",),
    "where the reasoning lives": ("spec", "reasoning"),
    "the things not to do": ("not to do", "never do", "avoid"),
}

#: The same for ``CONTRIBUTING.md``, which is about the process rather than about the code.
_CONTRIBUTING_TOPICS: Final = {
    "reporting a bug or asking for a feature": ("bug", "issue", "report"),
    "getting a working environment": ("environment", "setting up", "setup"),
    "running the tests": ("test",),
    "what a pull request has to pass": ("gate", "pull request"),
    "the code of conduct": ("code of conduct",),
    "reporting a vulnerability": ("security",),
    "licensing": ("licens",),
}


class TestBothConventionDocumentsExist:
    """``AT-112`` — ``REQ-E-111``, which asked for two files and for a long time got neither.

    They are "the SDK's house conventions copied into this repo", carried as a recommended
    item ``[RECOMMENDED]``. Recommended is why nothing failed while they were missing: no test read them,
    so the gap was invisible to everything except a reader looking for them.
    """

    def test_both_documents_are_at_the_repository_root(self) -> None:
        """AT-112 — ``REQ-E-111``: one copy of each, at the root, not two split by language.

        The SDK splits its conventions across a root file and a per-language file. This repository has one
        package, so a split would duplicate exactly what the README split refused to duplicate for the two READMEs.
        """
        missing = [path.name for path in _CONVENTION_DOCUMENTS if not path.is_file()]
        assert missing == [], f"REQ-E-111 asks for these at the repository root and they are absent: {missing}"

    def test_the_agents_document_covers_what_a_contributor_needs_before_writing_code(self) -> None:
        """AT-112 — the six subjects a contributor needs before their first commit, by heading."""
        text = _text(_AGENTS)
        uncovered = [subject for subject, alternatives in _AGENTS_TOPICS.items() if not _covers(text, alternatives)]
        assert uncovered == [], f"AGENTS.md has no heading covering: {uncovered}"

    def test_the_contributing_document_covers_the_process(self) -> None:
        """AT-112 — the seven subjects a first-time contributor needs, by heading."""
        text = _text(_CONTRIBUTING)
        uncovered = [
            subject for subject, alternatives in _CONTRIBUTING_TOPICS.items() if not _covers(text, alternatives)
        ]
        assert uncovered == [], f"CONTRIBUTING.md has no heading covering: {uncovered}"

    def test_the_contributing_document_defers_to_agents_rather_than_restating_it(self) -> None:
        """AT-112 — two documents that both list the conventions are two documents that will disagree.

        The split is deliberate: ``CONTRIBUTING.md`` is the process a contributor follows once, and
        ``AGENTS.md`` is the rules they read every time. Restating the second inside the first is how one
        of them goes stale, and the stale one is always the copy.
        """
        text = _text(_CONTRIBUTING)
        assert "AGENTS.md" in text, "CONTRIBUTING.md never points at AGENTS.md, so the conventions are unreachable"
        restated = [heading for heading in _headings(text) if "convention" in heading]
        assert restated == [], f"CONTRIBUTING.md restates the conventions instead of linking them: {restated}"


@pytest.mark.parametrize("document", _CONVENTION_DOCUMENTS, ids=lambda path: path.name)
class TestEveryClaimResolvesAgainstThisRepository:
    """``AT-113`` — ``REQ-E-111``, the six classes of claim a copied conventions document gets wrong.

    Not one of these is about writing. Each is a name that either resolves against a file in this
    repository or sends a contributor to debug their own machine over a sentence that was never true here.
    Both documents are checked against all six, because which one holds a given claim is an editorial
    decision and moving a table between them should not move it out of the tests.
    """

    def test_every_hatch_command_names_a_real_script(self, document: Path) -> None:
        """AT-113 — a documented command that is not a script fails with hatch's error, not the tool's.

        ``pip`` is exempt and is the only exemption: ``hatch run pip install --upgrade strands-agents`` is
        what the canary job runs, and hatch resolves it in the environment rather than as a script.
        """
        named = set(_HATCH_SCRIPT.findall(_text(document))) - {"pip"}
        unknown = sorted(named - _scripts())
        assert unknown == [], (
            f"{document.name} documents 'hatch run' for scripts pyproject.toml does not define: {unknown}"
        )

    def test_every_environment_variable_named_is_a_gate(self, document: Path) -> None:
        """AT-113 — a misspelled tier variable is a test suite that skips everything and reports green."""
        known = set(TIER_VARIABLES.values()) | {_BUILD_VARIABLE}
        unknown = sorted(set(_TIER_VARIABLE.findall(_text(document))) - known)
        assert unknown == [], f"{document.name} names variables no gate reads: {unknown}. The gates are {sorted(known)}"

    def test_every_marker_used_is_registered(self, document: Path) -> None:
        """AT-113 — ``--strict-markers`` is on, so an unregistered marker is a collection error."""
        unknown = sorted(_markers_named(_text(document)) - _markers())
        assert unknown == [], f"{document.name} uses markers pyproject.toml does not register: {unknown}"

    def test_every_python_version_named_is_one_the_package_supports(self, document: Path) -> None:
        """AT-113 — a version in a document and in neither the classifiers nor the matrix is a promise."""
        unsupported = sorted(set(_PYTHON_VERSION.findall(_text(document))) - _supported_versions())
        assert unsupported == [], (
            f"{document.name} names Python versions that are neither classified nor in the CI matrix: "
            f"{unsupported}. Supported: {sorted(_supported_versions())}"
        )

    def test_the_version_floor_is_stated(self, document: Path) -> None:
        """AT-113 — the floor is the one version fact a contributor needs before installing anything."""
        floor = _config()["project"]["requires-python"].lstrip(">=")
        assert floor in _text(document), f"{document.name} never states the {floor} floor from requires-python"

    def test_every_path_named_exists(self, document: Path) -> None:
        """AT-113 — the claim that goes stale first, because a file moves and a document does not follow.

        Resolved against the repository root and against ``python/``, so both ``python/pyproject.toml`` and
        a bare ``pyproject.toml`` are valid ways to write the same file.
        """
        exempt = set(_ABSENT_FROM_A_CLONE)
        missing = [
            candidate
            for candidate in _path_candidates(_text(document))
            if candidate not in exempt and not (_ROOT / candidate).exists() and not (_PACKAGE / candidate).exists()
        ]
        assert missing == [], f"{document.name} names paths that are in neither the root nor python/: {missing}"

    def test_every_placeholder_names_a_tracked_item(self, document: Path) -> None:
        """AT-113 — ``REQ-E-111``: a ``TODO`` with nothing behind it is how an open decision ships as prose.

        The code of conduct this repository adopts and the address a security report goes to were the two
        this check was written for, and both are filled in now. It stays for the next one, and ``AT-181`` now
        holds the two sections to naming an address rather than to naming the item that will name one.

        The open set is declared in :data:`_OPEN_PLACEHOLDERS` rather than inferred, so that adding a
        placeholder is a deliberate act: a ``TODO`` nobody has claimed fails here, and so does a bare
        ``TODO()``, whose empty name is in no set.
        """
        untracked = [item for item in _TODO.findall(_text(document)) if item not in _OPEN_PLACEHOLDERS]
        assert untracked == [], f"{document.name} holds placeholders nothing has claimed: {untracked}"


class TestTheDocumentsAndTheRepositoryAgreeBothWays:
    """``AT-113``, the reverse direction: what the repository has and the documents do not mention.

    The tests above catch a document that names something absent. These catch a document that is merely
    incomplete, which is the failure a copied file starts with rather than acquires: the script, marker or
    job that exists here and not in whatever the file was copied from.
    """

    def test_every_script_a_contributor_can_run_is_documented(self) -> None:
        """AT-113 — an undocumented script is one nobody outside this repository knows to run."""
        documented = set(_HATCH_SCRIPT.findall(_text(_AGENTS)))
        undocumented = sorted(_scripts() - documented)
        assert undocumented == [], f"AGENTS.md documents no 'hatch run' for: {undocumented}"

    def test_every_gate_variable_is_documented(self) -> None:
        """AT-113 — an undocumented gate is a tier that silently never runs on a contributor's machine."""
        documented = set(_TIER_VARIABLE.findall(_text(_AGENTS)))
        undocumented = sorted((set(TIER_VARIABLES.values()) | {_BUILD_VARIABLE}) - documented)
        assert undocumented == [], f"AGENTS.md never names these gates: {undocumented}"

    def test_every_registered_marker_is_documented(self) -> None:
        """AT-113 — seven markers are registered; a contributor who knows four writes tests in the wrong tier."""
        text = _text(_AGENTS)
        backticked = set(_BACKTICKED.findall(text))
        undocumented = sorted(marker for marker in _markers() if marker not in backticked)
        assert undocumented == [], f"AGENTS.md never mentions these registered markers: {undocumented}"

    def test_the_documented_gates_are_the_workflow_jobs(self) -> None:
        """AT-113 — set equality, both directions, against every job in ``ci-python.yml``.

        Read from the first column of ``CONTRIBUTING.md``'s table rather than from its prose, because the
        claim is a list and a list is the one thing a reader will assume is complete. Job names are
        compared with their variable part cut off, so the matrix job matches under one name.
        """
        jobs = {_canonical_job(job["name"]) for job in yaml.safe_load(_WORKFLOW.read_text())["jobs"].values()}
        documented = {_canonical_job(cell) for cell in _TABLE_ROW.findall(_text(_CONTRIBUTING))}
        assert jobs <= documented, (
            f"CONTRIBUTING.md's table omits CI jobs a pull request has to pass: {sorted(jobs - documented)}"
        )
        stale = {cell for cell in documented if cell and cell[0].isupper()} - jobs
        assert stale == set(), f"CONTRIBUTING.md's table names jobs the workflow does not define: {sorted(stale)}"

    def test_each_exempted_path_is_exempt_because_git_ignores_it(self) -> None:
        """AT-113 — the two paths the existence check skips, and the reason checked rather than trusted.

        ``.env`` and ``python/.venv`` are both named in the documents and absent from a clone, because both
        are things a contributor creates. An exemption list that drifted into holding a real path would
        quietly stop checking it, so each entry has to be one ``.gitignore`` accounts for.
        """
        ignored = _text(_GITIGNORE)
        unaccounted = [path for path in _ABSENT_FROM_A_CLONE if Path(path).name not in ignored]
        assert unaccounted == [], f".gitignore does not explain why the path check skips: {unaccounted}"


def _swept_documents() -> list[Path]:
    """Every Markdown file in the repository that a reader outside it can reach.

    Found by walking rather than by listing, which is the point of ``AT-114``: a list has to be extended
    when a document is added, and the document nobody adds to the list is the one that leaks.

    Returns:
        The paths, sorted, relative parts filtered against :data:`_UNSWEPT_DIRECTORIES`.
    """
    return sorted(
        path for path in _ROOT.rglob("*.md") if not _UNSWEPT_DIRECTORIES & set(path.relative_to(_ROOT).parts[:-1])
    )


class TestNoShippedDocumentCarriesACredential:
    """``AT-114`` — ``REQ-E-111`` and ``REQ-E-093``, widened from a list of three files to a walk.

    ``AT-097`` sweeps the two READMEs and the migration guide, named one at a time. That list never
    included ``python/examples/README.md``, which the sdist ships, and it could not have included the two
    documents this task adds. Walking the tree covers all of them and everything added next.
    """

    def test_the_sweep_reaches_every_document_that_ships(self) -> None:
        """AT-114 — the vacuity guard: a walk that found nothing would pass the leak test below.

        A superset rather than an equality, so adding a document does not fail this test. Removing one from
        the walk's reach does.
        """
        reached = {str(path.relative_to(_ROOT)) for path in _swept_documents()}
        unreached = [name for name in _SWEPT_AT_LEAST if name not in reached]
        assert unreached == [], f"the walk did not reach documents that ship: {unreached}. It found {sorted(reached)}"

    def test_no_swept_document_carries_a_credential(self) -> None:
        """AT-114 — ``REQ-E-093``: a literal connection string in a document is how a password ships."""
        leaking = [
            str(path.relative_to(_ROOT)) for path in _swept_documents() if _CREDENTIALED_URI.search(path.read_text())
        ]
        assert leaking == [], f"these documents contain a connection string with credentials in it: {leaking}"

    def test_the_detector_would_catch_one(self, tmp_path: Path) -> None:
        """AT-114 — the sweep above passes on an empty tree and on a broken expression alike.

        So the expression is tested against a planted credential and against the redacted form the probe
        writes, which is the shape that has to keep passing.
        """
        planted = tmp_path / "leak.md"
        planted.write_text("Connect with `mongodb+srv://admin:SUPERSECRET@cluster0.example.mongodb.net/`.\n")
        assert _CREDENTIALED_URI.search(planted.read_text()), "the credential expression matches nothing"

        redacted = tmp_path / "clean.md"
        redacted.write_text("Connect with `mongodb+srv://<redacted>@cluster0.example.mongodb.net/`.\n")
        assert not _CREDENTIALED_URI.search(redacted.read_text()), "the redacted form must not read as a leak"


class TestTheHouseConventionsAreEnforcedNotDescribed:
    """``AT-115`` — ``REQ-E-112``: the nine conventions ``AGENTS.md`` states, checked against the package.

    Seven of the nine are mechanical, and each is checked here. Two are not: Google-style docstrings and
    naming are review's job, and conventional commit messages are git's history rather than the tree.

    Six of the seven held when this task measured them, which is the reason they are worth asserting now
    rather than fixing later: a convention that holds by habit stops holding on the first contribution from
    somebody who never read the document. The seventh, ``__all__``, held in nine modules of ten.
    """

    def test_no_module_spells_an_optional_type_as_optional(self) -> None:
        """AT-115 — ``X | None``, never ``Optional[X]``, which is the SDK's rule and this package's floor."""
        offending = [path.name for path in _source_modules() if "Optional[" in path.read_text()]
        assert offending == [], f"these modules use Optional[X] instead of X | None: {offending}"

    def test_every_module_declares_what_it_exports(self) -> None:
        """AT-115 — an ``__all__`` on every module, so the public surface is a decision rather than a leak.

        ``_extras.py`` was the one module without one, alone among the four private modules. This test added
        it rather than writing the exemption into this test, because the exemption would have had to explain
        why one module's exports are nobody's business.
        """
        silent = []
        for path in _source_modules():
            tree = ast.parse(path.read_text())
            declared = any(
                isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "__all__" for target in node.targets)
                for node in tree.body
            )
            if not declared:
                silent.append(path.name)
        assert silent == [], f"these modules declare no __all__: {silent}"

    def test_the_logging_rule_is_a_selected_lint_rule(self) -> None:
        """AT-115 — ``%s`` and not f-strings in log calls, enforced the way the SDK enforces it.

        All ten log calls in this package already pass ``G``, and ``LOG`` too. The rule was not selected,
        which made a passing tree a coincidence rather than a gate, so it is selected here.
        """
        selected = _config()["tool"]["ruff"]["lint"]["select"]
        assert "G" in selected, (
            f"ruff's flake8-logging-format rules are not selected, so nothing enforces %s in log calls: {selected}"
        )

    def test_every_warning_passes_a_stacklevel(self) -> None:
        """AT-115 — vacuous today, and stated as vacuous: ``src`` makes no ``warnings.warn`` call at all.

        Written anyway because the convention exists for the call somebody adds, and a warning without
        ``stacklevel`` points at this package's own line instead of at the caller's.
        """
        careless = []
        for path in _source_modules():
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call) or not ast.unparse(node.func).endswith("warnings.warn"):
                    continue
                if "stacklevel" not in [keyword.arg for keyword in node.keywords]:
                    careless.append(f"{path.name}:{node.lineno}")
        assert careless == [], f"these warnings.warn calls pass no stacklevel: {careless}"

    def test_every_public_definition_that_raises_documents_it(self) -> None:
        """AT-115 — a ``Raises:`` section wherever a caller can be raised at, which nothing checked before.

        46 module-level functions and public-class methods when this was measured, and no gaps. The
        definition of "public" excludes closures, and :func:`_public_definitions` says why.
        """
        undocumented, counted = [], 0
        for path in _source_modules():
            for node in _public_definitions(ast.parse(path.read_text())):
                counted += 1
                raises = any(isinstance(child, ast.Raise) for child in ast.walk(node))
                if raises and "Raises:" not in (ast.get_docstring(node) or ""):
                    undocumented.append(f"{path.name}:{node.lineno} {node.name}")
        assert counted >= 40, f"the sweep found only {counted} public definitions, which is not this package"
        assert undocumented == [], f"these raise without documenting it: {undocumented}"

    def test_no_version_gated_asyncio_api_is_used_ungated(self) -> None:
        """AT-115 — ``TaskGroup`` and ``asyncio.timeout`` are 3.11+, and the floor is 3.10.

        Held by absence rather than by discipline: neither appears in ``src``. The test is the thing that
        keeps the floor from being broken by one convenient import.
        """
        ungated = []
        for path in _source_modules():
            source = path.read_text()
            for api in ("TaskGroup", "asyncio.timeout"):
                if api in source and "sys.version_info" not in source:
                    ungated.append(f"{path.name}: {api}")
        assert ungated == [], f"these use a 3.11+ asyncio API with no version gate: {ungated}"

    def test_nothing_in_the_package_imports_pydantic(self) -> None:
        """AT-115 — ``TypedDict`` and ``dataclass`` for structure; a runtime validator is a dependency."""
        importing = [path.name for path in _source_modules() if "pydantic" in path.read_text()]
        assert importing == [], f"these modules reach for pydantic: {importing}"

    def test_the_one_convention_this_package_declines_is_written_down(self) -> None:
        """AT-115 — ``REQ-E-112``: the log message shape, and the deviation stated as a decision.

        The SDK's format is ``field=<%s> | lowercase message``. This package prefixes ``strands_mongodb: ``
        and writes a sentence, because a message such as the one naming the server error code a capability
        probe saw is explanatory rather than a field list, and because the logger name is not rendered in
        most agent processes. A
        deviation nobody wrote down is indistinguishable from not knowing the rule, and contributor number
        two writes log call eleven either way.
        """
        text = _text(_AGENTS)
        assert "strands_mongodb: " in text, "AGENTS.md does not show the log prefix this package actually uses"
        assert any(word in text.lower() for word in ("deviat", "differs from", "does not copy")), (
            "AGENTS.md states the log format without saying it deviates from the SDK's, which reads as ignorance"
        )


class TestADeprecationIsAFailureRatherThanAScrollbackLine:
    """``AT-162`` / ``REQ-E-156`` -- the warning filter, and the floor that lets the suite survive it.

    The pin below sat at ``<1.0.0`` from this package's first commit, and the cost of leaving it there
    looked like tidiness: 2,269 ``DeprecationWarning``s per run, every one of them ``pytest_asyncio``
    0.26 calling ``asyncio.get_event_loop_policy()``, which Python removes in 3.16. The real cost was that
    a deprecation raised by *this* package would have arrived as line 2,270 of a wall nobody reads, and
    :class:`TestTheHouseConventionsAreEnforcedNotDescribed.test_every_warning_passes_a_stacklevel` is
    written for exactly the ``warnings.warn`` call somebody adds later.

    So the two halves are one change: the filter makes the category fatal, and the floor is what makes the
    filter survivable. Measured on the same tree under the same filter -- 915 errors on ``pytest-asyncio``
    0.26.0, 912 passed on 1.4.0.

    Both tests below observe the running interpreter rather than reading ``pyproject.toml`` back. A test
    that parses the manifest and asserts the manifest says what the manifest says passes while the setting
    does nothing -- the same failure as a document checked only against itself, which is how the root
    ``README.md`` came to describe a fixed defect for months with a passing test over it.
    """

    def test_a_deprecation_warning_raised_here_stops_the_test(self) -> None:
        """AT-162 -- the filter is live, proven by raising one rather than by reading the config.

        This is the whole of the guarantee. Delete ``filterwarnings`` from ``pyproject.toml`` and this
        test fails; misspell the category and it fails; add an ``ignore`` that shadows it and it fails.
        Nothing else in the suite can say that, because every other test passes whether warnings are fatal
        or merely counted.
        """
        with pytest.raises(DeprecationWarning):
            warnings.warn("AT-162 raises this to prove the filter catches it", DeprecationWarning, stacklevel=2)

    def test_the_async_plugin_is_past_the_version_that_trips_that_filter(self) -> None:
        """AT-162 -- read from the installed distribution, so the failure names the cause.

        Strictly redundant: on 0.26 the filter turns 915 tests into errors and the run is unmistakably
        broken. It is here for what the *first* line of that report says. 915 errors inside pytest's own
        plugin is a morning of bisecting; one failure naming the plugin and the floor is a ``pip install``.
        A test can be redundant about the outcome and load-bearing about the diagnosis.
        """
        installed = version("pytest-asyncio")
        major = int(installed.split(".")[0])
        assert major >= 1, (
            f"pytest-asyncio {installed} is installed, and below 1.0 it calls asyncio.get_event_loop_policy() "
            f"on every async test -- which pyproject.toml's filterwarnings turns into an error. Install the "
            f"declared floor (>=1.0.0,<2.0.0); the rest of this run's failures are that, not your change"
        )


class TestTheWarningTheFilterLeavesCountedIsNotOursToFix:
    """``AT-191`` / ``REQ-E-156`` -- the reason the filter names one category instead of erroring on all.

    ``filterwarnings = ["error::DeprecationWarning"]`` is deliberately narrower than ``["error"]``, and the
    comment in ``pyproject.toml`` gives two reasons. One of them was read off the tail of a 1,922-second
    contract run and never reproduced: nine ``PytestUnraisableExceptionWarning``s, diagnosed as
    pymongo's periodic executor cancelling its task from a weakref callback after the loop it belongs to
    has closed. The test named in that report was whichever one happened to be running, so the diagnosis
    was an inference from a stack trace, and a justification nobody can reproduce is a justification that
    gets argued away the next time somebody wants the strict setting.

    Reproduced here in five lines, deterministically and with no deployment: exactly the shape
    ``monitor.py`` builds, exactly one ``RuntimeError: Event loop is closed``, arriving where no ``except``
    can see it.

    **Not fixed upstream, and not by anything this project can pin.** ``AsyncPeriodicExecutor.close`` is
    byte-identical in 4.14.0 and in 4.18.1, the newest release at the time of writing: it calls
    ``self._task.cancel()`` with nothing asking whether the loop is still alive. So the filter stays narrow,
    and if that ever changes these tests fail -- which is the good direction, and the signal that
    ``["error"]`` has become affordable.

    **Not claimed: that the nine warnings in that run were each this.** One executor produces one, and the
    count matched the number of clients that outlived their loops; that is consistent rather than proven.
    """

    def test_the_executors_close_raises_when_its_loop_is_gone(self) -> None:
        """AT-191 -- the mechanism, reproduced rather than inferred from a stack trace.

        The executor is pymongo's own class with a trivial target, so what is under test is the driver's
        cancellation path and not a re-implementation of it. ``sys.unraisablehook`` is swapped for the
        duration because the exception is raised inside a garbage-collection callback, where the only way
        to observe it is the hook pytest's own plugin installs for the same reason.

        The closed loop is the cause rather than the coincidence: the same five lines with the loop left
        open collect the stand-in and raise nothing.
        """

        async def target() -> bool:
            return True

        async def start(executor: AsyncPeriodicExecutor) -> None:
            executor.open()
            await asyncio.sleep(0)  # Let the task start, for determinism; measured, it fires either way.

        executor = AsyncPeriodicExecutor(interval=60, min_interval=0.1, target=target, name="AT-191")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(start(executor))
        # The pending task is collected after the loop it ran on, and asyncio reports that through the
        # loop's exception handler rather than through a warning. Silenced because it is not what is being
        # measured and it would otherwise arrive as an unexplained error line in an unrelated test's output.
        loop.set_exception_handler(lambda _loop, _context: None)
        loop.close()

        class Monitored:
            """Stands in for the object whose collection fires ``executor.close``."""

        monitored = Monitored()
        finalizer = weakref.ref(monitored, executor.close)  # monitor.py's shape, verbatim.
        unraisable: list[str] = []
        original_hook = sys.unraisablehook
        sys.unraisablehook = lambda args: unraisable.append(f"{args.exc_type.__name__}: {args.exc_value}")
        try:
            del monitored
            gc.collect()
        finally:
            sys.unraisablehook = original_hook

        assert finalizer() is None, "the stand-in survived collection, so the callback never ran"
        assert unraisable == ["RuntimeError: Event loop is closed"], (
            f"pymongo {version('pymongo')} raised {unraisable} from a weakref callback on a closed loop; "
            f"if that list is empty the driver now guards the cancel and pyproject.toml's filterwarnings "
            f"can be widened to ['error']"
        )

    def test_the_driver_still_calls_that_close_from_a_weakref_callback(self) -> None:
        """AT-191 -- the premise of the test above, which is the part that could go stale quietly.

        The wiring is not reachable through behaviour: the callback is registered inside ``Monitor.open``
        on an object that closes over ``self``, and nothing exposes it. So the module is read. A brittle
        assertion on purpose -- if pymongo stops passing ``executor.close`` to ``weakref.ref``, the
        reproduction above is testing a shape the driver no longer builds, and this is the only test that
        would say so.
        """
        import pymongo.asynchronous.monitor as monitor

        assert "weakref.ref(self, executor.close)" in inspect.getsource(monitor), (
            f"pymongo {version('pymongo')} no longer registers the executor's close as a weakref callback, "
            f"so AT-191's reproduction no longer mirrors the driver -- re-derive it from monitor.py"
        )


def _section(document: Path, heading: str) -> str:
    """The body of one ``##`` section of a Markdown document, matched on the heading text.

    Sections rather than whole documents, because the questions ``AT-181`` asks are local: the address in
    the security section is the one a finder uses, and an address anywhere else in the file is a different
    claim. Two sections with the same heading would be concatenated, which no document here has.

    Args:
        document: The file to read.
        heading: The heading text, without the ``##`` and matched case-insensitively.

    Returns:
        The lines between that heading and the next one at the same level, or ``""`` if it is absent --
        which every caller below treats as a failure rather than as an empty section.
    """
    body: list[str] = []
    collecting = False
    for line in _text(document).splitlines():
        if line.startswith("## "):
            collecting = line[3:].strip().casefold() == heading.casefold()
            continue
        if collecting:
            body.append(line)
    return "\n".join(body)


def _addresses(text: str) -> set[str]:
    """Every mail address in a passage, with git's SSH remotes taken out first.

    Args:
        text: The passage to read.

    Returns:
        The addresses, deduplicated.
    """
    return set(_MAIL.findall(_SSH_REMOTE.sub("", text)))


class TestTheCodeOfConductAndTheSecurityAddressResolve:
    """``AT-181`` — ``REQ-E-176``: the two sections that shipped as placeholders for a long time.

    Neither was a hard question; both were somebody else's. The template these documents were modelled on
    answers both with Amazon's, and ``AT-113`` held the placeholders to naming a tracked item so that going
    quiet could not resolve them. The owner supplied both on 2026-09-17, and the claim moves with them: from
    "a placeholder names its item" to "an address resolves, and belongs to the company that can act on it".

    The company check is the one worth having. A wrong address is a report nobody reads, and the specific way
    this repository could have got there was inheriting a contact from the template -- which is a valid
    address at a company that cannot fix a defect in a MongoDB package.
    """

    def test_the_security_policy_is_at_the_filename_github_reads(self) -> None:
        """AT-181 — ``SECURITY.md`` at the root, which is what *Report a vulnerability* points a finder at.

        ``CONTRIBUTING.md`` has the same address in it and is not where a finder looks: GitHub surfaces this
        filename and only this filename.
        """
        assert _SECURITY.is_file(), f"REQ-E-176 asks for a security policy at {_SECURITY.name} and it is absent"

    def test_the_code_of_conduct_section_links_a_document_rather_than_describing_one(self) -> None:
        """AT-181 — a link, because "be respectful" is a sentiment and a code of conduct is a document."""
        section = _section(_CONTRIBUTING, "Code of conduct")
        assert section.strip(), "CONTRIBUTING.md has no 'Code of conduct' section for this test to read"
        assert "https://" in section, f"the code of conduct section links no document: {section.strip()!r}"

    def test_the_security_section_names_the_address_the_owner_supplied(self) -> None:
        """AT-181 — the address itself, pinned, so that editing it is a deliberate act in this file too."""
        section = _section(_CONTRIBUTING, "Security")
        assert section.strip(), "CONTRIBUTING.md has no 'Security' section for this test to read"
        assert _SECURITY_ADDRESS in section, f"the security section does not name {_SECURITY_ADDRESS}"

    def test_the_two_documents_name_one_reporting_address_and_the_same_one(self) -> None:
        """AT-181 — set equality against one string, because two copies of an address are two addresses.

        A second address in either place is the failure this asserts against, not a redundancy: a finder
        picks one, and the one they pick is the one nobody is reading.
        """
        found = _addresses(_text(_SECURITY)) | _addresses(_section(_CONTRIBUTING, "Security"))
        assert found == {_SECURITY_ADDRESS}, f"the security route is not one address: {sorted(found)}"

    def test_no_address_any_of_the_documents_names_belongs_to_another_company(self) -> None:
        """AT-181 — the failure exactly: a valid address at a company that cannot act on the report.

        Both addresses in this repository are at ``mongodb.com`` -- the reporting one the owner supplied,
        and ``community@mongodb.com``, which is the address MongoDB's own code of conduct names. The count
        guards the walk: an expression that matched nothing would pass the domain check silently.
        """
        found = {address for path in (_AGENTS, _CONTRIBUTING, _SECURITY) for address in _addresses(_text(path))}
        assert len(found) >= 2, f"the mail expression found fewer addresses than this repository has: {sorted(found)}"
        elsewhere = sorted(address for address in found if not address.endswith("@mongodb.com"))
        assert elsewhere == [], f"these documents route mail to a company that does not own this package: {elsewhere}"

    def test_the_one_address_shaped_string_that_is_not_an_address_is_a_git_remote(self) -> None:
        """AT-181 — the exclusion :data:`_SSH_REMOTE` makes, checked rather than trusted.

        An exclusion that quietly grew to cover a real mailbox would turn the company check above into a
        test that reports green on the failure it exists to catch. So what it removes is asserted to be
        non-empty and asserted to be written in the shape a remote has.
        """
        text = _text(_CONTRIBUTING)
        removed = set(_MAIL.findall(text)) - _addresses(text)
        assert removed, "CONTRIBUTING.md names no SSH remote any more, so the exclusion covers nothing"
        written_as_a_remote = sorted(address for address in removed if f"{address}:" in text)
        assert written_as_a_remote == sorted(removed), (
            f"excluded as a git remote but written as a mail address: {sorted(removed)}"
        )

    def test_both_documents_tell_a_finder_not_to_open_a_public_issue(self) -> None:
        """AT-181 — the outcome the section exists to prevent, stated in both places a finder arrives at.

        A public issue describing an unfixed vulnerability is a disclosure, and it is the default a finder
        falls back to when a repository says nothing.
        """
        silent = [
            path.name
            for path, text in ((_SECURITY, _text(_SECURITY)), (_CONTRIBUTING, _section(_CONTRIBUTING, "Security")))
            if "public issue" not in text
        ]
        assert silent == [], f"these documents never tell a finder to keep a vulnerability off the tracker: {silent}"

    def test_no_placeholder_stands_where_an_address_or_a_link_belongs(self) -> None:
        """AT-181 — ``AT-113`` asserts a placeholder names a tracked item; this asserts there is none left.

        The two are different claims and both are worth keeping. A tracked placeholder was the right answer
        while the decision was open, and it is the wrong answer now that it is closed: the address exists, so
        a ``TODO`` in either section would mean somebody replaced an answer with a promise.
        """
        sections = {
            "CONTRIBUTING.md 'Code of conduct'": _section(_CONTRIBUTING, "Code of conduct"),
            "CONTRIBUTING.md 'Security'": _section(_CONTRIBUTING, "Security"),
            _SECURITY.name: _text(_SECURITY),
        }
        held = {name: _TODO.findall(text) for name, text in sections.items() if _TODO.search(text)}
        assert held == {}, f"a placeholder stands where REQ-E-176 asks for an address or a link: {held}"
