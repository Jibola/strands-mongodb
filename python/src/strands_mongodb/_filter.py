"""Caller-filter validation and scope composition, shared by the search paths.

Three guards and one composer, all pure functions over plain mappings. Nothing here imports pymongo or
does I/O, so the security guards can be tested without a deployment.

The guards are separate because they answer different questions with different remedies: what a key
is (a string with no leading ``$``), what a value does (no regex), and whether the index declared the
field. One combined pass would give one message for three unrelated problems.

Private module: the validated filter shape is a promise, these signatures are not.
"""

from __future__ import annotations

import copy
import re

# Imported at runtime, not under ``TYPE_CHECKING``: the guards' ``isinstance`` checks need ``Mapping``.
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

__all__ = ["compose", "validate"]

# Fields the vector index always declares filterable, before `filter_fields`. `_indexes` builds the
# declared set from this, and `validate` takes that set as an argument rather than reading this.
BAKED_IN_FILTER_FIELDS = ("scope", "store_name")

# `bson.Regex`, matched by name so this module needs no pymongo import. The MRO is walked so a subclass
# cannot slip past. In pymongo 4.x the class is `bson.regex.Regex`.
_REGEX_TYPE_NAMES = frozenset({"bson.regex.Regex"})


def _reject_operator_keys(value: Any) -> None:
    """Raise if any key at any depth is ``$``-prefixed or not a string.

    Walks every mapping, list and tuple, which are the only containers bson encodes. A dict inside a list
    is an equality operand to ``$match`` and cannot act as operators, but that is the server's guarantee,
    not this function's, so lists are walked too.

    Only keys are checked: a key is what MQL dispatches on, and a value is data (the string ``"$where"``
    as a value is just a string). The one value that is not data is a regex, which
    :func:`_reject_regex_values` refuses. Values are checked as the Python objects passed in, so a custom
    codec on the client that encodes some other type as a subdocument is outside this guard.

    Recursion is on ``collections.abc.Mapping``, not ``dict``. bson encodes any mapping as a
    subdocument, so ``{"tier": UserDict({"$ne": "gold"})}`` would otherwise pass this guard and reach
    the server as ``{"tier": {"$ne": "gold"}}``, matching every document whose tier is not gold.

    Args:
        value: A filter, or any value reached while walking one.

    Raises:
        ValueError: Naming the offending key.
    """
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or key.startswith("$"):
                raise ValueError(
                    f"strands_mongodb: invalid filter key {key!r}. MongoDB operator keys and "
                    f"non-string keys are not allowed anywhere in a filter, at any depth. A filter is "
                    f"a mapping of field path to exact value, such as {{'metadata.tier': 'gold'}}."
                )
            _reject_operator_keys(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_operator_keys(item)


def _reject_regex_values(value: Any) -> None:
    """Raise if any value at any depth is a regex.

    A filter maps field paths to exact values. Given a regex, ``$match`` and ``$eq`` pattern-match
    instead of comparing: ``{"tier": re.compile("^gol")}`` matches a document whose tier is ``"gold"``.
    The operator guard cannot catch this because there is no ``$`` key. Pattern filters are refused
    rather than supported, because their cost and reach depend on the pattern rather than on what a
    reader sees.

    Both ``re.Pattern`` and ``bson.Regex`` are refused, since both reach the server as a BSON regex.
    Lists and tuples are walked as in the operator guard.

    Args:
        value: A filter, or any value reached while walking one.

    Raises:
        ValueError: Naming the field path that carries the regex, when there is one to name.
    """
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _is_regex(nested):
                raise ValueError(
                    f"strands_mongodb: invalid filter value for {key!r}: a regular expression. A filter "
                    f"is a mapping of field path to exact value, and a regex is matched as a pattern "
                    f"rather than compared, so it would select documents this filter does not name. Pass "
                    f"the literal value to match, such as {{{key!r}: 'gold'}}."
                )
            _reject_regex_values(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if _is_regex(item):
                raise ValueError(
                    "strands_mongodb: invalid filter value: a regular expression inside a list. A filter "
                    "is a mapping of field path to exact value, at any depth, and a regex is matched as a "
                    "pattern rather than compared."
                )
            _reject_regex_values(item)


def _is_regex(value: Any) -> bool:
    """Return whether a value would reach the server as a BSON regex.

    Args:
        value: Any value found in a filter.

    Returns:
        ``True`` for a compiled ``re.Pattern`` and for ``bson.Regex`` or any subclass of it.
    """
    if isinstance(value, re.Pattern):
        return True
    return any(f"{cls.__module__}.{cls.__qualname__}" in _REGEX_TYPE_NAMES for cls in type(value).__mro__)


def _reject_undeclared_fields(caller_filter: Mapping[str, Any], declared: Collection[str]) -> None:
    """Raise if the filter names a path the index did not declare filterable.

    mongot rejects such a filter itself (code 8, ``Path 'x' needs to be indexed as filter``, seen on
    8.3.8; not verified on Atlas proper). This guard still runs first so the error names the config key
    to edit, the index rebuild it needs, and what is filterable now, without a round trip.

    Checking top-level keys only is exact: the operator guard has already rejected ``$and`` and ``$or``,
    so no field path can sit below the top level, and a dict value is exact subdocument equality on the
    key above it.

    Args:
        caller_filter: A filter that has already passed the operator guard.
        declared: Every path the index declares filterable: ``scope``, ``store_name`` and the config's
            ``filter_fields``. Passed in so the index definition stays the single authority.

    Raises:
        ValueError: Naming every undeclared path, not only the first, and naming what is available.
    """
    undeclared = sorted(set(caller_filter) - set(declared))
    if not undeclared:
        return

    available = ", ".join(sorted(declared)) or "none"
    raise ValueError(
        f"strands_mongodb: filter field(s) {', '.join(repr(field) for field in undeclared)} are not "
        f"declared filterable on this index, and the server rejects a filter over an undeclared path. "
        f"Filterable here: {available}. Add the field to 'filter_fields' in MongoDBConfig and recreate "
        f"the index to filter on it."
    )


def validate(caller_filter: Mapping[str, Any], *, declared_fields: Iterable[str]) -> None:
    """Check a caller-supplied filter against all three guards.

    A valid filter is a mapping from declared field paths to exact values. In detail:

    * Every key, at any depth (inside nested mappings, lists and tuples), is a string that does not
      start with ``$``. So no operators: no ``$and``, ``$or``, ``$ne``, ``$in``, ``$where``, ``$expr``.
    * No value, at any depth, is a regex (``re.Pattern`` or ``bson.Regex``).
    * Every top-level key is in ``declared_fields``. Dotted paths such as ``"metadata.tier"`` are allowed
      when declared exactly so.

    The order is part of the contract: operator keys, then regex values, then undeclared fields.
    ``{"$where": ...}`` is both forbidden and undeclared, and a caller told to declare ``$where`` in
    ``filter_fields`` might try to. An operator key is the more serious problem, so it is reported
    before a regex.

    Args:
        caller_filter: The filter as the caller supplied it. Not modified.
        declared_fields: Paths the index declares filterable.

    Raises:
        ValueError: From any guard, naming what was wrong.
    """
    _reject_operator_keys(caller_filter)
    _reject_regex_values(caller_filter)
    _reject_undeclared_fields(caller_filter, tuple(declared_fields))


def compose(
    *caller_filters: Mapping[str, Any] | None,
    scope: str,
    store_name: str,
) -> dict[str, Any]:
    """Compose caller filters with the scope predicate as a conjunction.

    ``$and`` rather than a merged dict: ``{**scope_predicate, **caller_filter}`` would let a caller's
    ``scope`` key overwrite the store's. Under ``$and`` a caller term can only narrow, and a
    contradictory one (``{"scope": "tenant-b"}`` against ``tenant-a``) matches nothing.

    Variadic because a memory store has two caller filters: the one configured on the store and the one
    on a single ``MongoDBSearchQuery``. Each stays its own term, since merging them would need a ``$and``
    that :func:`validate` rejects.

    Does not validate. Callers run :func:`validate` first.

    Args:
        *caller_filters: Caller filters, already validated, in the order they should appear.
            ``None`` and ``{}`` both add no term.
        scope: This store's scope. Never taken from caller input; an absent scope resolves to a sentinel
            upstream, never to unfiltered.
        store_name: This store's name, isolating it from other stores in the same collection.

    Returns:
        A ``$and`` predicate with the scope term first, then the store term, then every non-empty caller
        filter. Deep-copied, so the caller's dicts and the returned predicate share no structure.
    """
    terms: list[dict[str, Any]] = [
        {"scope": {"$eq": scope}},
        {"store_name": {"$eq": store_name}},
    ]
    for caller_filter in caller_filters:
        if caller_filter:
            # Deep, so mutating a nested value after composing cannot alter a query already in flight.
            terms.append(copy.deepcopy(dict(caller_filter)))
    return {"$and": terms}
