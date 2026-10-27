"""Fail if any single module falls below a coverage floor.

``--cov-fail-under`` checks the package as a whole, which can pass while one module is untested. This
reads the ``coverage.json`` written by ``pytest --cov-report=json`` and exits non-zero if any module is
below the floor. Run it in addition to the aggregate check.

Usage::

    python scripts/check_module_coverage.py [--floor 70] [--report coverage.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Lower than the 90% aggregate floor: one module may trail the average, but not by this much.
DEFAULT_FLOOR = 70.0

# Modules exempt from the floor, mapped to the reason. Kept in the repository, not a CI flag, so
# adding one goes through review.
EXEMPT: dict[str, str] = {}


def main() -> int:
    """Check every measured module against the floor.

    Returns:
        ``0`` when every non-exempt module meets the floor. ``1`` when one does not, or when the report
        is missing or measured no files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR, help="minimum percent per module")
    parser.add_argument("--report", type=Path, default=Path("coverage.json"), help="coverage json path")
    args = parser.parse_args()

    if not args.report.exists():
        print(
            f"error: {args.report} not found. Run pytest with --cov-report=json first.",
            file=sys.stderr,
        )
        return 1

    report = json.loads(args.report.read_text())
    files = report.get("files", {})
    if not files:
        print(f"error: {args.report} measured no files at all.", file=sys.stderr)
        return 1

    failures: list[tuple[str, float, int]] = []
    for path, data in sorted(files.items()):
        summary = data["summary"]
        # A module with no statements cannot be under-covered.
        if summary["num_statements"] == 0:
            continue
        if path in EXEMPT:
            print(f"exempt  {path}: {EXEMPT[path]}")
            continue
        percent = summary["percent_covered"]
        if percent < args.floor:
            failures.append((path, percent, summary["missing_lines"]))

    if not failures:
        print(f"ok: every module meets the {args.floor:.0f}% floor ({len(files)} measured)")
        return 0

    print(f"\nfail: {len(failures)} module(s) below the {args.floor:.0f}% per-module floor\n", file=sys.stderr)
    for path, percent, missing in failures:
        print(f"  {percent:6.2f}%  {path}  ({missing} uncovered lines)", file=sys.stderr)
    print(
        "\nThe aggregate floor can pass while these fail. That is the point: aggregate coverage\n"
        "hides an untested adapter behind well-tested pure functions.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
