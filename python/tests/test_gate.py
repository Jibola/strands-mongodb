"""The contract checklist, written out as a table the run can be measured against.

The checklist enumerates 29 behaviours "a user would notice if we got it wrong" and then names a subset
as the release gate: items 1 to 9 and 16 to 19, with the remainder expected to exist but a single
failure among them "a judgement call, not an automatic block". This file is what turns the checklist
from a list into something CI can fail on.

Two halves, and they fail for different reasons:

* The static half runs with no backend at all. It asserts the table still matches the checklist (29
  items, numbered once each, the gate subset exactly as drawn) and that every acceptance-test id it
  names resolves to a test that was actually collected. **A checklist item with no test is a failure,
  not a silence**, which is the one thing a suite of 640 passing tests cannot tell you about itself.
* The dynamic half needs the rest of the run. It reads what each gate item's tests did and refuses to
  report a green gate when they skipped for want of a database, because a skipped release gate that
  prints as a pass is worse than no gate.

The link between an item and its tests is the acceptance-test id in the test's docstring, which is the
convention the whole suite already follows, resolved against the tests pytest collected rather than
against the files on disk. ``conftest.py``'s ``ChecklistGate`` does the collecting, and explains when an
id on a class docstring stands in for the tests inside it.

One rule for anything written here: **no acceptance-test id may appear in a docstring in this file.**
Ids in the table below are data. An id in a docstring would make this file its own evidence, and the
gate would start answering yes because it asked itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import pytest
from conftest import ChecklistGate, RunOutcome


class Item(NamedTuple):
    """One numbered item from the contract checklist.

    Attributes:
        number: The item's number, which is how the checklist refers to it.
        gate: Whether the checklist puts it in the release gate.
        behaviour: The behaviour, shortened from the checklist's own wording.
        ids: The acceptance-test ids covering it. More than one where the item has halves that live in
            different files, or where the covering id is declared on a class rather than a test.
    """

    number: int
    gate: bool
    behaviour: str
    ids: tuple[str, ...]


# The checklist, in its own order and numbering. Items 10 and 11 name the same two ids on purpose: both
# are halves of the normalisation decision, and the two classes covering it declare their ids at class
# level, so nothing finer is available to point at without minting ids that do not exist.
CHECKLIST: tuple[Item, ...] = (
    Item(1, True, "read of an absent key returns None rather than raising", ("AT-021",)),
    Item(2, True, "read raises StorageError for a failure other than a missing key", ("AT-022",)),
    Item(3, True, "write overwrites an existing value rather than appending or erroring", ("AT-020",)),
    Item(4, True, "bytes round-trip unchanged, including empty, every byte value and a large payload", ("AT-024",)),
    Item(5, True, "delete of an absent key is a no-op", ("AT-023",)),
    Item(6, True, "list('') returns every key", ("AT-025",)),
    Item(7, True, "list(prefix) returns full keys rather than suffixes", ("AT-026",)),
    Item(8, True, "list results sort lexicographically, asserted against Python's sorted()", ("AT-027",)),
    Item(9, True, "list on a prefix matching nothing returns an empty list", ("AT-028",)),
    Item(
        10, False, "keys normalise the SDK's way, so which keys collide is a documented decision", ("AT-018", "AT-019")
    ),
    Item(11, False, "a key or prefix containing a '..' segment is rejected", ("AT-018", "AT-019")),
    Item(12, False, "search scores are similarity-oriented, higher being better", ("AT-032d", "AT-033")),
    Item(13, False, "scores stay in the documented range and data is populated as asked", ("AT-032b", "AT-033")),
    Item(14, False, "a transport failure raises StorageError with the cause chained", ("AT-022", "AT-036", "AT-080")),
    Item(
        15,
        False,
        "the session-snapshot key shapes round-trip and UUID7 keys sort chronologically",
        ("AT-027", "AT-081"),
    ),
    Item(16, True, "search returns MemoryEntry with content, store_name and the reserved metadata", ("AT-040",)),
    Item(17, True, "max_search_results precedence, and below 1 raises ValueError", ("AT-041",)),
    Item(18, True, "adding identical content twice leaves one document and reports one duplicate", ("AT-042",)),
    Item(19, True, "a concurrent duplicate insert surfaces as success, not DuplicateKeyError", ("AT-043",)),
    Item(20, False, "add_messages stays absent, so the SDK routes extraction through add", ("AT-044a", "AT-044b")),
    Item(21, False, "initialize is idempotent and the first search performs it", ("AT-045a", "AT-045b")),
    Item(22, False, "concurrent initialize across instances makes one client and one index set", ("AT-012", "AT-065")),
    Item(23, False, "a filter naming an undeclared field raises a ValueError naming the field", ("AT-049",)),
    Item(24, False, "a filter containing an MQL operator key is rejected", ("AT-050a", "AT-050b", "AT-050c")),
    Item(
        25,
        False,
        "two scopes never see each other's documents, for search and for add",
        ("AT-046", "AT-047a", "AT-047b"),
    ),
    Item(26, False, "an injected client is never closed by the store", ("AT-010", "AT-011")),
    Item(27, False, "driver metadata is attached on both construction paths", ("AT-013a", "AT-013b")),
    Item(28, False, "an oversized payload raises StorageError naming the key, size and limit", ("AT-029", "AT-038a")),
    Item(
        29,
        False,
        "vector on a deployment without search raises, auto degrades and reports",
        ("AT-053a", "AT-053c", "AT-054"),
    ),
)

# The checklist's own sentence: "items 1-9 and 16-19 are the release gate". Written out separately from the
# table so a flag flipped by accident in one of them contradicts the other.
RELEASE_GATE = frozenset(range(1, 10)) | frozenset(range(16, 20))


def _narrowing(config: pytest.Config) -> str:
    """Describe every argument that could have kept a test out of this run.

    The gate distinguishes "this behaviour has no test" from "this run did not collect its test", and
    the two look identical from inside a test. Anything reported here means the second reading is
    available, so absence becomes a skip rather than a failure.

    Args:
        config: The active pytest configuration.

    Returns:
        A comma-separated description of the narrowing arguments, or an empty string for a run that
        collected the whole suite.
    """
    option = config.option
    narrowing = []
    if option.keyword:
        narrowing.append(f"-k {option.keyword!r}")
    if option.markexpr:
        narrowing.append(f"-m {option.markexpr!r}")
    if getattr(option, "deselect", None):
        narrowing.append(f"--deselect ({len(option.deselect)})")
    if getattr(option, "lf", False) or getattr(option, "ff", False):
        narrowing.append("--last-failed / --failed-first")
    targets = [str(target) for target in option.file_or_dir]
    if targets and targets != [str(path) for path in config.getini("testpaths")]:
        narrowing.append(f"paths {targets}")
    return ", ".join(narrowing)


def _carriers(gate: ChecklistGate, item: Item) -> list[str]:
    """Return the node ids of every collected test covering one checklist item.

    Args:
        gate: The registered gate, holding this run's index of ids to node ids.
        item: The checklist item.

    Returns:
        Node ids, sorted and deduplicated, since two of an item's ids can be declared on one class.
    """
    return sorted({node for identifier in item.ids for node in gate.by_id.get(identifier, ())})


def _verdict(gate: ChecklistGate, items: Iterable[Item], narrowing: str) -> tuple[list[str], dict[int, list[str]]]:
    """Classify a set of checklist items against what this run's tests did.

    Written as a function taking the gate rather than as part of the test, so that the four judgements
    below are themselves testable without arranging a whole pytest session to produce each one. The
    class at the end of this file does that.

    Args:
        gate: The registered gate, holding the id index and the recorded outcomes.
        items: The checklist items to judge.
        narrowing: The result of :func:`_narrowing`, which decides whether an item with no collected test
            is a missing test or a deselected one.

    Returns:
        A pair. The first is the problems, each a sentence naming the items involved, empty when there
        are none. The second maps an item number to the reasons its tests gave for not producing a pass,
        which is a skip rather than a failure.
    """
    absent: list[int] = []
    failed: dict[int, list[str]] = {}
    unrun: dict[int, list[str]] = {}
    unverified: dict[int, list[str]] = {}

    for item in items:
        nodes = _carriers(gate, item)
        if not nodes:
            # Deselected rather than missing, if this run collected a subset. Either way the item is
            # unverified here; only which of the two lists it lands in differs.
            if narrowing:
                unverified[item.number] = [f"deselected by {narrowing}"]
            else:
                absent.append(item.number)
            continue
        recorded = {node: gate.outcomes.get(node) for node in nodes}
        if never_ran := [node for node, outcome in recorded.items() if outcome is None]:
            unrun[item.number] = never_ran
        outcomes = [outcome for outcome in recorded.values() if outcome is not None]
        if broke := [node for node, outcome in recorded.items() if outcome is not None and outcome.outcome == "failed"]:
            failed[item.number] = broke
        if not any(outcome.outcome == "passed" for outcome in outcomes):
            unverified[item.number] = sorted({outcome.detail for outcome in outcomes if outcome.detail}) or [
                "no reason recorded"
            ]

    problems = []
    if absent:
        problems.append(f"no test at all for items {absent}")
    if failed:
        problems.append(f"failing tests: {failed}")
    if unrun:
        problems.append(f"tests that never ran, so this gate was not reordered last: {unrun}")
    return problems, unverified


class TestTheTableStillMatchesTheChecklist:
    """The table above is a transcription, and a transcription can drift from its source.

    Nothing here needs a database or the rest of the run. These are the assertions that would catch a
    row deleted while rewording something, a number typed twice, or an item quietly promoted out of the
    release gate, all of which would leave the gate green and narrower than the checklist.
    """

    def test_the_checklists_twenty_nine_items_are_each_here_once(self) -> None:
        assert [item.number for item in CHECKLIST] == list(range(1, 30))

    def test_the_release_gate_subset_is_the_one_the_checklist_draws(self) -> None:
        """Items 1 to 9 and 16 to 19, and the reason for the split is in this file's docstring."""
        assert {item.number for item in CHECKLIST if item.gate} == RELEASE_GATE

    def test_every_item_names_a_test(self) -> None:
        """An item with an empty id tuple would pass every other assertion here and cover nothing."""
        assert [item.number for item in CHECKLIST if not item.ids] == []

    def test_every_item_describes_the_behaviour_rather_than_only_numbering_it(self) -> None:
        """The behaviour text is what a failure message has to be readable from."""
        assert [item.number for item in CHECKLIST if len(item.behaviour) < 20] == []


class TestEveryChecklistItemHasATestInThisRun:
    """Do the ids in the table resolve to tests pytest actually collected?

    This is the half that catches the failure the checklist is written against: a checklist item nobody wrote a
    test for. It runs with no backend configured, because collection does not need one, so it fails on a
    laptop and in CI alike rather than only in the nightly job that has the databases.

    An id resolves through the docstrings of the collected tests. That means a test renamed or moved
    still counts, and a test deleted stops counting immediately.
    """

    def test_every_id_the_table_names_resolves_to_a_collected_test(
        self, checklist_gate: ChecklistGate, request: pytest.FixtureRequest
    ) -> None:
        unresolved = {
            item.number: [identifier for identifier in item.ids if identifier not in checklist_gate.by_id]
            for item in CHECKLIST
        }
        unresolved = {number: ids for number, ids in unresolved.items() if ids}
        if unresolved and (narrowing := _narrowing(request.config)):
            pytest.skip(f"this run collected a subset ({narrowing}); items {sorted(unresolved)} unresolved here")

        assert unresolved == {}, (
            f"checklist items name ids no collected test declares: {unresolved}. Either the test was "
            f"deleted or its docstring no longer carries the id."
        )

    def test_no_checklist_item_is_left_without_a_test(
        self, checklist_gate: ChecklistGate, request: pytest.FixtureRequest
    ) -> None:
        """The same question asked per item rather than per id, which is the form the checklist asks it in."""
        uncovered = [item.number for item in CHECKLIST if not _carriers(checklist_gate, item)]
        if uncovered and (narrowing := _narrowing(request.config)):
            pytest.skip(f"this run collected a subset ({narrowing}); uncovered here: {uncovered}")

        assert uncovered == [], f"checklist items with no collected test: {uncovered}"


@pytest.mark.gate
class TestTheReleaseGate:
    """Did the release-gate items run, and did they pass?

    The gate assertion the release requires, and the reason it is a test rather than a script: it has to
    read the same run CI reads, including the skips. It is reordered to last by the plugin in
    ``conftest.py``, because a gate that ran before the tests it measures would report all of them as
    never having run.

    Three outcomes, and the third is the one worth designing for:

    * A gate item with a failing test, or with a test that never ran, **fails** this test. So does an
      item with no test in a run that collected the whole suite.
    * All gate items with a passing test: this passes, and it is the only thing in the suite that says
      so about the checklist as a whole.
    * A gate item whose every test skipped, which is what a run with no ``MONGODB_URI`` produces:
      **skips**, naming the items and quoting the skip reasons. Reporting a pass there would be the
      exact failure the tier gate in ``conftest.py`` exists to prevent, one level up.

    Individual failures are still reported by the tests that own them. What this adds is the checklist's
    own verdict, which no individual test can give.
    """

    def test_every_release_gate_item_ran_and_passed(
        self, checklist_gate: ChecklistGate, request: pytest.FixtureRequest
    ) -> None:
        if hasattr(request.config, "workerinput"):
            pytest.skip("outcomes are recorded per process, so under xdist no worker sees the whole run")

        gated = [item for item in CHECKLIST if item.gate]
        problems, unverified = _verdict(checklist_gate, gated, _narrowing(request.config))

        assert not problems, "release gate (checklist items 1-9, 16-19): " + "; ".join(problems)

        if unverified:
            # Grouped by reason rather than listed per item, because the common case is one missing
            # environment variable skipping a dozen items with the same sentence.
            grouped: dict[tuple[str, ...], list[int]] = {}
            for number, reasons in sorted(unverified.items()):
                grouped.setdefault(tuple(reasons), []).append(number)
            detail = "; ".join(f"items {numbers}: {' / '.join(reasons)}" for reasons, numbers in grouped.items())
            pytest.skip(f"release gate unverified in this run: {detail}")


class TestTheVerdictTheGateReaches:
    """The gate's own judgement, each branch arranged directly rather than by staging a whole session.

    This is the same argument ``test_tier_gating.py`` makes about the tier gate, applied one level up. A
    gate that cannot fail reports green forever, and a green run is the one nobody investigates. The
    difference from the tier gate is that no nested session is needed here: the judgement is a function
    of two dictionaries, so the cases can be built by hand.

    The id below is deliberately not one this suite uses. It is data, never a docstring, so nothing here
    can register itself as evidence for a real checklist item.
    """

    ITEM = Item(1, True, "the behaviour whose verdict is under test", ("AT-999",))

    @staticmethod
    def _gate(nodes: tuple[str, ...], outcomes: dict[str, RunOutcome]) -> ChecklistGate:
        gate = ChecklistGate()
        gate.by_id = {"AT-999": nodes}
        gate.outcomes = outcomes
        return gate

    def test_a_passing_test_leaves_nothing_to_report(self) -> None:
        gate = self._gate(("t.py::one",), {"t.py::one": RunOutcome("passed", "")})

        assert _verdict(gate, [self.ITEM], "") == ([], {})

    def test_a_failing_test_is_a_problem_naming_the_item_and_the_node(self) -> None:
        gate = self._gate(("t.py::one",), {"t.py::one": RunOutcome("failed", "call failed")})

        problems, unverified = _verdict(gate, [self.ITEM], "")

        assert problems == ["failing tests: {1: ['t.py::one']}"]
        assert unverified == {1: ["call failed"]}

    def test_one_pass_among_several_carriers_is_enough_but_a_failure_is_still_reported(self) -> None:
        """Both halves matter. A class-level id can put a dozen tests behind one item, and an item is
        covered when any of them passes, but a failure among them is not absolved by that pass."""
        gate = self._gate(
            ("t.py::one", "t.py::two"),
            {"t.py::one": RunOutcome("passed", ""), "t.py::two": RunOutcome("failed", "call failed")},
        )

        problems, unverified = _verdict(gate, [self.ITEM], "")

        assert problems == ["failing tests: {1: ['t.py::two']}"]
        assert unverified == {}

    def test_every_carrier_skipping_is_unverified_rather_than_a_problem(self) -> None:
        """The no-database case, and the reason the gate skips instead of passing."""
        gate = self._gate(("t.py::one",), {"t.py::one": RunOutcome("skipped", "Skipped: needs MONGODB_URI")})

        assert _verdict(gate, [self.ITEM], "") == ([], {1: ["Skipped: needs MONGODB_URI"]})

    def test_a_carrier_with_no_recorded_outcome_means_the_gate_ran_too_early(self) -> None:
        """The ordering failure the plugin's ``trylast`` exists to prevent, and it must be loud: a gate
        that measured tests which had not run yet would report the checklist as uncovered, not as green."""
        gate = self._gate(("t.py::one",), {})

        problems, unverified = _verdict(gate, [self.ITEM], "")

        assert problems == ["tests that never ran, so this gate was not reordered last: {1: ['t.py::one']}"]
        assert unverified == {1: ["no reason recorded"]}

    def test_an_item_with_no_test_fails_a_whole_run_and_skips_a_narrowed_one(self) -> None:
        """The distinction the checklist turns on. Absence is the failure it is written against, but
        under ``-k`` or ``-m`` the same absence is the caller's own filter and must not be reported as a
        missing test."""
        gate = self._gate((), {})

        assert _verdict(gate, [self.ITEM], "") == (["no test at all for items [1]"], {})
        assert _verdict(gate, [self.ITEM], "-m 'not integration'") == (
            [],
            {1: ["deselected by -m 'not integration'"]},
        )
