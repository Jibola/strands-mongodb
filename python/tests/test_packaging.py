"""Characterization tests for the declared and resolved package metadata.

Covers ``CH-002`` and ``CH-006``, and stands at the boundary from the
declared manifest to resolved distribution. Both sides are asserted, because they disagree in ways
that matter.

``importlib.metadata`` reads what pip actually installed. That is the stronger assertion for
dependency and extra questions, and it is the only one that catches a manifest whose extras resolve
differently than they read. The manifest is read directly only for properties the build **erases**:
hatchling flattens the self-referential ``all`` extra, and ``hatch-vcs`` resolves a version that
depends on git state the manifest cannot show.

``importlib.metadata`` has one weakness, and ``AT-002``/``AT-003`` at the bottom of this file exist
because of it. It describes whichever install last ran, which here is an editable one made before the
workspace became a git repository: it still reports version ``0.0.0``, hours after a fresh build
stopped producing that number. Those two tests build a wheel and install it into an empty venv, so
what they read is this tree.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib.metadata import metadata
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    # Imported for annotations only. It is re-exported from `importlib.metadata` on every supported
    # interpreter, but it is a protocol defined in a private submodule, so keeping the import out of
    # runtime avoids depending on that re-export staying put.
    from importlib.metadata import PackageMetadata

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    # `tomli` is the parser `tomllib` was adopted from, and 2.x is the API `tomllib` shipped, so the
    # alias makes the two interchangeable and every caller below reads one name. It is a dev dependency
    # declared without a version marker on purpose; `pyproject.toml` says why, and the reason is mypy
    # rather than this branch.
    import tomli as tomllib

DISTRIBUTION = "strands-mongodb"

# tests/ -> python/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

#: Opt-in gate for the build tier (`AT-002`, `AT-003`). Every other test in this file reads metadata
#: that is already on disk and costs milliseconds. These two build a wheel and resolve dependencies
#: over the network, which was 4 to 10 seconds per operation when they were written, so they are gated
#: rather than left to slow every run down.
#:
#: Gated here and not in `conftest.py`'s `TIER_VARIABLES`, because that mapping is also the list of
#: names `fill_tier_variables_from_env` reads out of a `.env` file, and every one of them is a
#: connection string. This tier needs network and no secret, which is why the split exists.
BUILD_TIER_VARIABLE = "STRANDS_MONGODB_BUILD_TIER"

#: Both halves are needed. The marker makes the tier selectable with ``-m build`` and is declared in
#: ``pyproject.toml`` under ``--strict-markers``; the skip is what keeps an ordinary run fast, and it
#: names the variable, so the report says what to set rather than just "skipped".
_BUILD_TIER = pytest.mark.build
_BUILD_TIER_GATE = pytest.mark.skipif(
    not os.environ.get(BUILD_TIER_VARIABLE),
    reason=f"build tier needs {BUILD_TIER_VARIABLE} set; it builds a wheel and resolves dependencies over the network",
)


def _requirements_by_extra(dist_metadata: PackageMetadata) -> dict[str, set[str]]:
    """Group ``Requires-Dist`` entries by the extra that gates them.

    Args:
        dist_metadata: Metadata of the installed distribution.

    Returns:
        A mapping of extra name to the set of requirement specifiers it pulls in. Ungated
        requirements are collected under the empty string.
    """
    by_extra: dict[str, set[str]] = {}
    for requirement in dist_metadata.get_all("Requires-Dist") or []:
        spec, _, marker = requirement.partition(";")
        extra = marker.strip().removeprefix("extra ==").strip().strip("'\"")
        by_extra.setdefault(extra, set()).add(spec.strip())
    return by_extra


def _distribution_names(specifiers: set[str]) -> set[str]:
    """Strip version constraints, leaving bare distribution names.

    Args:
        specifiers: Requirement specifiers such as ``pymongo>=4.14.0``.

    Returns:
        The distribution names alone.
    """
    return {spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip() for spec in specifiers}


def _normalized(name: str) -> str:
    """Return a distribution name in PEP 503 normalized form.

    Needed because pip's ``--report`` echoes each distribution's own ``Name``, and distributions
    disagree about the separator: ``typing_extensions`` publishes an underscore, ``pymongo-search-utils``
    a hyphen. Comparing a resolution against names taken from ``pyproject.toml`` without normalizing
    fails on the underscore, which is how ``AT-002`` first ran RED.

    Args:
        name: A distribution name in whatever form its own metadata uses.

    Returns:
        Lowercase, with runs of ``-``, ``_`` and ``.`` collapsed to a single ``-``.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


@pytest.fixture(scope="module")
def dist_metadata() -> PackageMetadata:
    """Metadata of the installed distribution."""
    return metadata(DISTRIBUTION)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """Parsed ``pyproject.toml``.

    Returns:
        The manifest as a mapping.
    """
    return tomllib.loads(PYPROJECT.read_text())


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A wheel built from this tree, through pip's PEP 517 path.

    ``pip wheel`` rather than ``hatch build``, and the reason is not convenience: neither ``hatch`` nor
    ``hatchling`` is installed in the development venv. The backend is named in ``pyproject.toml``, pip
    fetches it into an isolated build environment, and that is the same path a consumer's
    ``pip install strands-mongodb`` takes from an sdist. Building through whichever tool a maintainer
    happens to have would be a test of that tool.

    Returns:
        Path to the single built wheel.
    """
    destination = tmp_path_factory.mktemp("wheelhouse")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-input",
            "--wheel-dir",
            str(destination),
            str(PROJECT_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    wheels = sorted(destination.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {[w.name for w in wheels]}"
    return wheels[0]


@pytest.fixture(scope="module")
def resolve_extra(built_wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], set[str]]:
    """Return a callable giving the distribution names pip would install for one extra of the wheel.

    ``--dry-run`` with ``--report``, so the resolution is real and nothing is downloaded into an
    environment. ``--ignore-installed`` is what makes the answer meaningful: without it pip treats the
    development venv's ``strands-agents`` and ``pymongo`` as already satisfying the requirement, the
    report names the wheel and almost nothing else, and a test asserting on that set would be reading
    this machine's history rather than the artifact.

    Results are cached per extra, because each resolution took about 4 seconds when this was written.

    Returns:
        A callable taking an extra name and returning the full transitive set of distribution names.
    """
    reports = tmp_path_factory.mktemp("reports")
    cache: dict[str, set[str]] = {}

    def resolve(extra: str) -> set[str]:
        if extra not in cache:
            report = reports / f"{extra}.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--ignore-installed",
                    "--no-input",
                    "--quiet",
                    "--report",
                    str(report),
                    f"{built_wheel}[{extra}]",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            resolution = json.loads(report.read_text())
            cache[extra] = {_normalized(entry["metadata"]["name"]) for entry in resolution["install"]}
        return cache[extra]

    return resolve


@pytest.fixture(scope="module")
def storage_only_venv(built_wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty venv with the built wheel installed under ``[storage]`` and nothing else.

    ``[storage]`` rather than ``[all]`` on purpose. It is the only environment where the missing-extra
    path is real: ``pymongo-search-utils`` is genuinely absent, so touching ``MongoDBMemoryStore``
    raises for the reason a user's would rather than because a ``sys.meta_path`` finder blocked it,
    which is how ``AT-008b`` has to simulate it. This venv was first built by hand as a demo, and
    nothing since has replaced it; this fixture is it, executed.

    Returns:
        Path to the venv's interpreter.
    """
    root = tmp_path_factory.mktemp("clean") / "venv"
    created = subprocess.run([sys.executable, "-m", "venv", str(root)], capture_output=True, text=True, check=False)
    assert created.returncode == 0, created.stdout + created.stderr

    interpreter = root / ("Scripts" if os.name == "nt" else "bin") / "python"
    installed = subprocess.run(
        [str(interpreter), "-m", "pip", "install", "--quiet", "--no-input", f"{built_wheel}[storage]"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    return interpreter


def _in_venv(interpreter: Path, source: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a venv's interpreter and return the completed process.

    Args:
        interpreter: Path to the interpreter to run.
        source: Python source to execute.

    Returns:
        The completed process, never raising on a non-zero status: the tests assert on it.
    """
    return subprocess.run([str(interpreter), "-c", dedent(source)], capture_output=True, text=True, check=False)


class TestExtras:
    """CH-002 — three runtime extras, plus dev, and ``all`` is their union."""

    def test_declares_the_expected_extras(self, dist_metadata: PackageMetadata) -> None:
        """Exactly four extras: three runtime surfaces and ``dev``.

        ``dev`` counts. An earlier draft said "exactly three extras" and would have
        failed against the package as built.
        """
        assert sorted(dist_metadata.get_all("Provides-Extra") or []) == ["all", "dev", "memory", "storage"]

    def test_base_install_has_no_runtime_dependency(self, dist_metadata: PackageMetadata) -> None:
        """REQ-E-001 — every ``Requires-Dist`` entry is gated behind an extra.

        A dependency that slipped out of an extra would be installed by a bare
        ``pip install strands-mongodb``, which is exactly what the extras exist to prevent.
        """
        assert _requirements_by_extra(dist_metadata).get("", set()) == set()

    def test_all_is_the_union_of_storage_and_memory(self, dist_metadata: PackageMetadata) -> None:
        """``[all]`` cannot resolve to a split pair, because it names nothing of its own.

        This is the observable form of the self-referential ``all``. Hatchling flattens
        ``strands-mongodb[memory,storage]`` into the resolved dependency list at build time, so the
        self-reference itself is invisible here (see ``test_all_extra_is_self_referential``). The
        union property is what protects the user and it survives the flattening.
        """
        by_extra = _requirements_by_extra(dist_metadata)

        assert by_extra["all"] == by_extra["storage"] | by_extra["memory"]

    def test_memory_extra_is_a_superset_of_storage(self, dist_metadata: PackageMetadata) -> None:
        """REQ-E-002 — ``[memory]`` adds ``pymongo-search-utils`` and nothing else."""
        by_extra = _requirements_by_extra(dist_metadata)

        assert _distribution_names(by_extra["memory"] - by_extra["storage"]) == {"pymongo-search-utils"}

    def test_storage_extra_pulls_no_search_machinery(self, dist_metadata: PackageMetadata) -> None:
        """REQ-E-002 — the point of splitting the extras, asserted directly.

        The union test above would still pass if ``[storage]`` quietly gained
        ``pymongo-search-utils``, since the union would simply be equal. This is the assertion that
        the split buys the user something.

        ``typing-extensions`` joined this set later (see
        ``test_every_directly_imported_third_party_module_is_declared``). ``opentelemetry-api`` joined it
        with the package's own spans (``AT-212``).
        """
        by_extra = _requirements_by_extra(dist_metadata)

        assert _distribution_names(by_extra["storage"]) == {
            "strands-agents",
            "pymongo",
            "typing-extensions",
            "opentelemetry-api",
        }

    def test_every_directly_imported_third_party_module_is_declared(self, dist_metadata: PackageMetadata) -> None:
        """REQ-E-002 — a dependency we ``import`` is one we name, not one we inherit.

        ``types.py`` imports ``Required`` and ``TypedDict`` from ``typing_extensions``, because
        ``typing.Required`` is 3.11+ and the declared floor is 3.10. It resolves today whether or not
        we ask for it: ``strands-agents`` pins ``typing-extensions>=4.13.2,<5.0.0`` unconditionally
        and is in both runtime extras.

        Relying on that is the thing this test rejects. A transitive pin is a fact about someone
        else's current release, not a promise to us, and the failure it produces is a bare
        ``ModuleNotFoundError`` at import in a resolution nobody tested. Both extras name it, so
        The invariant is unaffected: neither surface gained a dependency the other lacks.
        """
        by_extra = _requirements_by_extra(dist_metadata)

        for extra in ("storage", "memory"):
            assert "typing-extensions" in _distribution_names(by_extra[extra]), sorted(by_extra[extra])

    def test_both_store_extras_declare_the_tracing_api_they_import(self, dist_metadata: PackageMetadata) -> None:
        """AT-212 / REQ-E-207 — ``_telemetry`` imports ``opentelemetry``, so both extras name it.

        The same argument as ``typing-extensions`` above. ``strands-agents`` brings it today, and that is a
        fact about their release rather than a promise to us. The floor is theirs, so declaring it changes no
        resolution. The bare install is unaffected: ``dependencies`` is asserted empty elsewhere.
        """
        by_extra = _requirements_by_extra(dist_metadata)

        for extra in ("storage", "memory"):
            assert "opentelemetry-api>=1.30.0" in by_extra[extra], sorted(by_extra[extra])

    def test_all_extra_is_self_referential_in_the_manifest(self, manifest: dict[str, Any]) -> None:
        """The manifest expresses ``all`` by reference, so the union cannot drift by editing one list.

        Asserted against the source, since the built metadata has this flattened away. Both
        assertions are needed: this one keeps the mechanism, the union test keeps the outcome.
        """
        assert manifest["project"]["optional-dependencies"]["all"] == ["strands-mongodb[memory,storage]"]

    def test_dev_extra_declares_a_coverage_tool(self, dist_metadata: PackageMetadata) -> None:
        """REQ-E-005 — the coverage floor is a release gate, so the tool that measures it is declared.

        The scaffold declared a 90% floor and no ``pytest-cov`` in ``dev``. A gate no
        installed tool can measure is not a gate.
        """
        dev = _requirements_by_extra(dist_metadata)["dev"]

        assert "pytest-cov" in _distribution_names(dev), sorted(dev)

    def test_dev_extra_declares_a_yaml_parser(self, manifest: dict[str, Any]) -> None:
        """REQ-E-104 — the same argument as the coverage tool, one release gate over.

        ``tests/test_ci.py`` parses the CI workflow, so PyYAML is a test dependency. It is already
        importable in a development venv as a runtime dependency of ``strands-agents``, and that is the
        reason to declare it rather than a reason not to: an inherited dependency is a fact about their
        current release, not a promise to us.

        Read from the manifest rather than from the built metadata, deliberately. The claim here is about
        what the project declares; whether the declared parser is actually installed is asserted by
        ``import yaml`` at the top of ``test_ci.py``, which fails collection rather than a test.
        """
        dev = manifest["project"]["optional-dependencies"]["dev"]

        assert any(requirement.startswith("pyyaml") for requirement in dev), dev


class TestCoreMetadata:
    """CH-006 — interpreter floor, SDK floor, license, version source."""

    def test_requires_python_floor_is_3_10(self, dist_metadata: PackageMetadata) -> None:
        """The floor the classifiers and the mypy target both assume."""
        assert dist_metadata["Requires-Python"] == ">=3.10"

    def test_license_is_an_apache_2_expression(self, dist_metadata: PackageMetadata) -> None:
        """PEP 639 license expression, matching the extension template's own declaration."""
        assert dist_metadata["License-Expression"] == "Apache-2.0"

    @pytest.mark.parametrize("extra", ["storage", "memory"])
    def test_strands_agents_floor_is_1_54_on_both_runtime_extras(
        self, dist_metadata: PackageMetadata, extra: str
    ) -> None:
        """1.54.0 is the first release shipping every ``strands`` module ``storage.py`` imports.

        ``strands.storage`` and ``StorageError`` arrive at 1.48.0, which is what this floor said until
        the floors were installed rather than read. ``strands.storage.search.keyword`` -- whose
        ``tokenize`` the lexical path calls -- arrives at 1.54.0, and below it the storage backend
        raises ``ModuleNotFoundError`` on import.

        Applied to both extras deliberately, though ``strands.memory`` landed in 1.44.0, so ``[all]``
        cannot resolve to a split pair.
        """
        requirements = [r for r in (dist_metadata.get_all("Requires-Dist") or []) if f"extra == '{extra}'" in r]
        floors = [r for r in requirements if r.startswith("strands-agents")]
        assert floors, requirements
        assert all(">=1.54.0" in r for r in floors), floors

    @pytest.mark.parametrize("extra", ["storage", "memory"])
    def test_pymongo_floor_is_4_14_on_both_runtime_extras(self, dist_metadata: PackageMetadata, extra: str) -> None:
        """4.14.0 is the first release carrying ``AsyncMongoClient.append_metadata``.

        The floor is what makes :func:`_client._attribute_injected_client` a plain call. Below 4.14 the
        attribute still resolves -- ``AsyncMongoClient.__getattr__`` hands back a ``Database`` of that
        name -- so an unguarded call raises ``TypeError`` rather than failing to import. A guard would
        make that silent, and a silent no-op is exactly what the floor exists to rule out.

        It was ``>=4.12.0``, inherited from ``pymongo-search-utils`` 0.3.0's own floor rather than chosen.
        Nothing here needs 4.12, and 4.14.0 supports Python 3.9 upward, which clears our 3.10 floor.
        """
        requirements = [r for r in (dist_metadata.get_all("Requires-Dist") or []) if f"extra == '{extra}'" in r]
        floors = [r for r in requirements if r.startswith("pymongo") and not r.startswith("pymongo-search-utils")]
        assert floors, requirements
        assert all(">=4.14.0" in r for r in floors), floors

    def test_version_is_derived_from_a_git_tag_not_a_file(self, manifest: dict[str, Any]) -> None:
        """``CH-006``'s version half, asserted against the manifest.

        The resolved version cannot demonstrate this, for two reasons that both survive. The workspace
        carries no ``python-v*`` tag, so ``hatch-vcs`` reports a guess for an untagged tree rather than
        anything a tag produced; and installed metadata describes whichever build last ran ``pip``,
        which for this venv predates the repository existing at all and reads ``0.0.0``. Asserting
        either would assert the absence of a tag, which is the opposite of the invariant. ``AT-003``
        covers what a fresh build does carry, and ``AT-004`` is the unwritten test that needs a tag.
        """
        assert manifest["project"]["dynamic"] == ["version"]
        assert "version" not in manifest["project"]
        assert manifest["tool"]["hatch"]["version"]["source"] == "vcs"

    def test_release_tags_are_namespaced_to_the_python_artifact(self, manifest: dict[str, Any]) -> None:
        """``python-v*``, so a future TypeScript half versions independently off the same repo."""
        raw_options = manifest["tool"]["hatch"]["version"]["raw-options"]
        assert raw_options["tag_regex"] == "^python-v(?P<version>.+)$"
        assert raw_options["root"] == ".."

    def test_untagged_builds_are_unmistakable(self, manifest: dict[str, Any]) -> None:
        """The fallback version is ``0.0.0``, which no release will ever be."""
        assert manifest["tool"]["hatch"]["version"]["raw-options"]["fallback_version"] == "0.0.0"


class TestBuildPathsResolveInsideTheProjectRoot:
    """REQ-E-003 — the defect that made the scaffold unbuildable.

    ``pyproject.toml`` declared ``readme = "README.md"`` and ``license-files = ["LICENSE", "NOTICE"]``
    while all three files lived at the repository root, one directory above the project root.
    ``pip install -e .`` failed at metadata generation with
    ``OSError: Readme file does not exist: README.md``, so 0.0.1 was never built and never published.
    """

    @pytest.mark.parametrize("filename", ["README.md", "LICENSE", "NOTICE"])
    def test_referenced_file_exists_in_the_project_root(self, filename: str) -> None:
        """Every path the manifest references resolves without escaping ``python/``."""
        assert (PROJECT_ROOT / filename).is_file(), f"{filename} missing from {PROJECT_ROOT}"

    def test_manifest_references_those_files_by_relative_name(self, manifest: dict[str, Any]) -> None:
        """No ``../`` in the readme or license paths: a build backend rejects paths outside the root."""
        assert manifest["project"]["readme"] == "README.md"
        assert manifest["project"]["license-files"] == ["LICENSE", "NOTICE"]


class TestTheDistributionAdvertisesItsTypes:
    """AT-007c / REQ-E-007a — the PEP 561 marker, and the consequence of shipping without it.

    Found by running a demo from a directory outside the project. ``mypy`` on a snippet with
    a ``MongoDBConfig`` literal missing its ``Required`` ``database`` key exited **0**. Not because
    the type was wrong: because the package carried no ``py.typed``, so mypy declined to read any of
    it and made every symbol ``Any``.

    That makes the marker load bearing for the whole typing story rather than a packaging detail.
    ``Required`` keys, the narrow ``SearchType`` literal and the ``TYPE_CHECKING`` declarations behind
    ``REQ-E-007`` are all instructions to a consumer's type checker, and a consumer's type checker
    reads none of them without it. Every reference ships one: ``strands`` itself, the extension
    template this package was scaffolded from, and ``pymongo``.
    """

    def test_the_marker_file_exists_in_the_package(self, dist_metadata: PackageMetadata) -> None:
        """A file whose presence is the entire signal (PEP 561), asserted next to the claim it backs.

        The manifest has carried the ``Typing :: Typed`` classifier since the scaffold, so the
        omission was two lines of one file disagreeing rather than a decision. The classifier is a
        claim on PyPI and the marker is the mechanism; asserting both together is what keeps them from
        drifting apart again in either direction.
        """
        assert "Typing :: Typed" in (dist_metadata.get_all("Classifier") or [])
        assert (PROJECT_ROOT / "src" / "strands_mongodb" / "py.typed").is_file()

    def test_the_marker_sits_inside_the_directory_the_wheel_packages(self, manifest: dict[str, Any]) -> None:
        """Placement is what ships it, and misplacing it fails silently.

        ``hatchling`` includes every file under the declared package directory, so no explicit
        ``artifacts`` or ``force-include`` entry is needed. A marker one level up in ``src/`` would
        satisfy the test above and never reach the wheel.

        The stronger form of this assertion inspects a built wheel. That needs a build backend this
        environment does not install, so the release workflow's own build step is where it belongs.
        """
        packages = manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]

        assert "src/strands_mongodb" in packages

    def test_a_consumer_type_checker_reads_our_required_keys(
        self, run_mypy_as_installed: Callable[[str], subprocess.CompletedProcess[str]]
    ) -> None:
        """The failure the marker actually prevents, asserted the way a consumer would hit it.

        Resolved through the install with no ``MYPYPATH``, this is the only test in the suite that can
        fail on a missing marker. The rest set ``MYPYPATH`` and analyze ``src`` directly, which is a
        path PEP 561 does not gate, so all of them pass either way.

        The positive half matters as much as the negative one. A missing marker produces exit 1 too,
        from ``import-untyped``, so the returncode alone proves nothing.
        """
        result = run_mypy_as_installed(
            """
            from strands_mongodb.types import MongoDBConfig

            config: MongoDBConfig = {"connection_string": "mongodb://localhost"}
            """
        )

        assert "import-untyped" not in result.stdout, result.stdout
        assert 'Missing key "database" for TypedDict "MongoDBConfig"' in result.stdout, result.stdout


@_BUILD_TIER
@_BUILD_TIER_GATE
class TestEachExtraResolvesFromTheBuiltWheel:
    """AT-002 / REQ-E-002 — the manifest boundary from the artifact side: every extra resolves.

    ``TestExtras`` above reads the same facts out of ``importlib.metadata`` and is the faster test, so
    this one has to justify the network. It does two things that one cannot.

    It resolves. An extra is a name plus a list of requirement strings, and a metadata assertion proves
    the strings are what the manifest said. Whether a resolver can satisfy them together is a different
    question, and the answer moves without this repository changing: a yanked release, a new upper bound
    in a transitive dependency, a floor that no longer has a wheel for a supported interpreter. All of
    those turn a correct manifest into an uninstallable package.

    And it resolves the **transitive closure**, which is where the extras split is actually decided.
    ``[storage]`` naming no search dependency of its own does not mean a ``[storage]`` install lacks
    one; any of its 49 transitive dependencies could pull ``pymongo-search-utils`` in and the metadata
    assertion would still pass.
    """

    @pytest.mark.parametrize("extra", ["storage", "memory", "all"])
    def test_the_extra_resolves_and_carries_the_declared_floors(
        self, resolve_extra: Callable[[str], set[str]], extra: str
    ) -> None:
        """A resolution that fails at all fails in the fixture, which is the primary assertion here.

        What is left to check is that the resolution is of this package with its dependencies, rather
        than a resolution that succeeded by installing something else: pip reports ``strands-mongodb``
        among the names, and the three distributions every runtime extra declares are present.
        """
        resolved = resolve_extra(extra)

        assert "strands-mongodb" in resolved, sorted(resolved)
        for required in ("strands-agents", "pymongo", "typing-extensions"):
            assert required in resolved, f"{required} missing from [{extra}]: {sorted(resolved)}"

    def test_only_the_memory_side_pulls_the_search_builders(self, resolve_extra: Callable[[str], set[str]]) -> None:
        """REQ-E-002 — the split buys the user something, asserted after the resolver has had its say.

        One distribution of difference across the whole closure, and it is the one the manifest names.
        A ``[storage]`` install that arrived at ``pymongo-search-utils`` by some other route would fail
        here and pass every metadata assertion in this file.
        """
        storage = resolve_extra("storage")
        memory = resolve_extra("memory")

        assert memory - storage == {"pymongo-search-utils"}, sorted(memory - storage)
        assert storage - memory == set(), sorted(storage - memory)

    def test_all_resolves_to_exactly_what_memory_resolves_to(self, resolve_extra: Callable[[str], set[str]]) -> None:
        """``[all]`` is the union, and ``[memory]`` is a superset of ``[storage]``, so the union is it.

        The self-reference hatchling flattens is invisible to a resolver, so this is the only form in
        which ``all = ["strands-mongodb[memory,storage]"]`` can be checked end to end: whatever the
        flattening produced, it installs the same environment.
        """
        assert resolve_extra("all") == resolve_extra("memory")


@_BUILD_TIER
@_BUILD_TIER_GATE
class TestTheBuiltWheelInstallsIntoAnEmptyVenv:
    """AT-003 / REQ-E-003 / REQ-E-007a — the artifact and lazy-import boundaries, nothing else installed.

    Every other assertion in this file describes the development venv's editable install. That install
    was made before this workspace became a git repository and it still reports version ``0.0.0``, hours
    after a fresh build stopped producing that number, which is the shape of the problem: an assertion
    about installed metadata can be an assertion about an install old enough to predate the change it
    was meant to catch. The wheel here is built from the tree as it stands, in this session.

    Four things are only observable this way: what the wheel *contains*, which is where ``py.typed``
    either ships or silently does not; what a first import pulls in when the interpreter has never seen
    this package; what the missing-extra path does when the dependency is genuinely absent; and what
    version a build off this tree resolves to.
    """

    def test_the_first_import_pulls_no_driver(self, storage_only_venv: Path) -> None:
        """REQ-E-006 — the lazy export, in an interpreter with no import cache of ours.

        ``AT-006d`` asserts this already, in a subprocess of the development venv where every
        dependency is installed and something else could have imported ``pymongo`` first. Here nothing
        else is installed, so the observation is unambiguous.
        """
        result = _in_venv(
            storage_only_venv,
            """
            import sys
            import strands_mongodb

            print("pymongo" in sys.modules)
            print(strands_mongodb.MongoDBStorage.__module__)
            print("pymongo" in sys.modules)
            """,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.split() == ["False", "strands_mongodb.storage", "True"], result.stdout

    def test_the_missing_extra_message_names_the_extra_that_is_actually_missing(self, storage_only_venv: Path) -> None:
        """The error a user gets after installing the narrower extra, produced by the install.

        ``AT-008b`` covers this with a ``sys.meta_path`` finder that blocks the distribution, because in
        the development venv it is present. A blocked import and an absent one raise the same class and
        need not carry the same message: the finder answers for the module name it was told about, and
        the install answers for whatever ``pip`` actually put on the path.
        """
        result = _in_venv(
            storage_only_venv,
            """
            import strands_mongodb

            try:
                strands_mongodb.MongoDBMemoryStore
            except ImportError as error:
                print(error)
            else:
                print("NO ERROR")
            """,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "strands_mongodb.memory_store requires the 'memory' extra" in result.stdout, result.stdout
        assert "pip install 'strands-mongodb[memory]'" in result.stdout, result.stdout

    def test_the_type_marker_ships_inside_the_installed_package(self, storage_only_venv: Path) -> None:
        """REQ-E-007a — the assertion ``test_the_marker_sits_inside_the_directory_the_wheel_packages`` defers.

        That test reads the manifest's ``packages`` list and says so in its own docstring: the stronger
        form inspects a built wheel. This is that form, one step further on, reading the file out of the
        directory pip unpacked it into. A marker that the build dropped, or placed one level up in
        ``src/``, is absent here and present in the manifest.
        """
        result = _in_venv(
            storage_only_venv,
            """
            import pathlib
            import strands_mongodb

            marker = pathlib.Path(strands_mongodb.__file__).parent / "py.typed"
            print(marker.is_file())
            """,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "True", result.stdout

    def test_the_installed_version_is_the_one_this_tree_builds(
        self, storage_only_venv: Path, built_wheel: Path
    ) -> None:
        """``CH-006``'s version half, as far as an untagged repository allows it to be checked.

        The wheel's filename carries the version ``hatch-vcs`` resolved at build time, and the installed
        metadata carries what pip recorded from it. Equal, and equal to neither ``0.0.0`` nor the stale
        number the development venv still reports, is the whole claim. The tag half needs a ``python-v*``
        tag and is ``AT-004``, which is unwritten.

        The last assertion presumes git metadata is reachable, which is a property of this tier rather
        than of the package: it runs for a maintainer or in CI, both of which have a checkout. Run from
        an unpacked sdist it would fail on the fallback version, correctly reporting that the build
        cannot tell a release from a download.
        """
        expected = built_wheel.name.split("-")[1]
        result = _in_venv(
            storage_only_venv,
            """
            from importlib.metadata import version

            print(version("strands-mongodb"))
            """,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == expected, f"{result.stdout.strip()!r} != {expected!r}"
        assert expected != "0.0.0", "a build off this tree resolved the no-git fallback version"


class TestTheDevExtraAndTheHatchEnvironmentAgree:
    """AT-160 / REQ-E-154 — the same dependency list is written twice, so assert the copies are equal.

    ``[project.optional-dependencies].dev`` is what a contributor installs with
    ``pip install -e '.[all,dev]'``. ``[tool.hatch.envs.default].dependencies`` is what CI installs,
    because every job runs through ``hatch``. Nothing derives one from the other, and nothing compared
    them until this test: a package added to the first alone is present locally and absent in CI, which
    makes a green matrix a statement about the maintainer's laptop.

    Found by having to add the same dependency to both lists and noticing that nothing would have said so.

    Compared as sets of requirement strings rather than as lists, so ordering is free, and in both
    directions for the same reason ``AT-105`` is: a copy that is merely a superset is still a copy that
    drifted, and which side is ahead is the first thing a reader needs to know.
    """

    def test_the_two_dependency_lists_are_equal(self, manifest: dict[str, Any]) -> None:
        """Read from both tables, so neither can be edited alone."""
        extra = set(manifest["project"]["optional-dependencies"]["dev"])
        environment = set(manifest["tool"]["hatch"]["envs"]["default"]["dependencies"])

        assert extra, "the dev extra declares nothing, so there is nothing to compare"
        assert extra == environment, (
            f"the dev extra and the hatch default environment have drifted; "
            f"in the extra only: {sorted(extra - environment)}, "
            f"in the environment only: {sorted(environment - extra)}"
        )

    def test_the_hatch_environment_installs_every_runtime_extra(self, manifest: dict[str, Any]) -> None:
        """``features = ["all"]`` is the other half: CI gets the runtime dependencies too.

        Without it the dev list would be equal and complete and CI would still fail to import the
        package, which is a different failure the equality above cannot see.
        """
        assert manifest["tool"]["hatch"]["envs"]["default"]["features"] == ["all"]
