"""Caller-filter validation and scope composition, shared by the search paths.

Three guards and one composer, all pure functions over plain mappings. No pymongo import, no I/O, nothing
that needs a deployment, which is deliberate: this is the code path a cross-scope read would run through, and
a security guard that can only be tested against a live cluster gets tested less often.

The guards are independent on purpose. ``REQ-E-050`` is about what a key *is*, ``REQ-E-174`` is
about what a value *does*, and ``REQ-E-049`` is about whether the index declared it. Collapsing them into
one pass would produce one message for three unrelated problems with three unrelated remedies.

Private, like ``_client``. The validated filter shape is a promise; these function signatures are not.
"""

from __future__ import annotations

import copy
import re

# Imported at runtime rather than under ``TYPE_CHECKING``, and that is the fix for one of ``AT-179``'s two
# bypasses: the operator guard recurses on what it can prove is a mapping, and ``dict`` is not that test.
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

__all__ = ["compose", "validate"]

# Field paths the vector index definition always declares filterable, before `filter_fields`. Named
# here for documentation only: callers pass the set in, so the index definition stays the one authority.
BAKED_IN_FILTER_FIELDS = ("scope", "store_name")

# `bson.Regex`, matched by name instead of by import (``REQ-E-174``). This module states at the top that it
# imports no part of pymongo, and that is worth more than the two characters an import would save: a guard
# every scope check runs through is a guard that has to be cheap to test, and `bson` is a compiled
# extension in the default install. The MRO is walked rather than `type(value)` compared, so a subclass
# cannot slip past. Verified against pymongo 4.x, where the class is `bson.regex.Regex`.
_REGEX_TYPE_NAMES = frozenset({"bson.regex.Regex"})


def _reject_operator_keys(value: Any) -> None:
    """Raise if any key at any depth is ``$``-prefixed or not a string (``REQ-E-050``).

    MongoDB's own ``_validate_filter``, byte-identical in
    ``langgraph-checkpoint-mongodb/.../utils.py:46`` and ``langgraph-store-mongodb/.../base.py:49``,
    with one difference: it recurses into ``dict`` values only, and this recurses into lists and tuples
    too. Measured against both copies, ``{"a": [{"$where": "sleep(1000)"}]}`` passes them, and
    ``REQ-E-050`` says *at any depth*.

    That extension is defense in depth rather than a patched vulnerability: probed on
    MongoDB 8.3.8, a dict inside a list is an equality operand and matches nothing, and every construct
    that would read it as operators is a ``$`` key already rejected here. It is closed because that
    safety is a property of ``$match`` rather than of this function.

    Keys are checked, and values are checked only for *being* a mapping. A value is data whatever it
    holds; a key is what MQL dispatches on. The string ``"$where"`` in a value position is a
    five-character string. The one value that is not data is a regex, which MQL dispatches on as surely as
    it dispatches on a key, and :func:`_reject_regex_values` is where that is refused.

    Recursion is on ``collections.abc.Mapping`` and not on ``dict``, which is the whole of ``AT-179``'s
    first half. bson encodes any mapping as a subdocument, so a value the guard declined to walk still
    arrives at the server as one: measured on MongoDB 8.0.32, a filter of
    ``{"tier": UserDict({"$ne": "gold"})}`` passed this guard, composed into
    ``{"$and": [..., {"tier": {"$ne": "gold"}}]}``, and matched the document whose tier is *not* gold --
    the exact opposite of the exact-match policy, from a filter that never contained a ``dict``. A nested
    one, ``{"tier": {"inner": UserDict({"$ne": None})}}``, passed for the same reason. Reported by a
    production-readiness review; the scope conjunction held throughout, so this was a filter-policy bypass
    rather than a cross-scope read.

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
    """Raise if any value at any depth is a regex (``REQ-E-174``).

    The third guard, and independent of the other two for the same reason they are independent of each
    other: it is a different question with a different remedy. ``REQ-E-050`` asks what a key *is*,
    ``REQ-E-049`` asks whether the index declared it, and this asks what a value *does*.

    A filter is documented as a mapping of field path to exact value, and every value in one is inert --
    with one exception, which is why this function exists. Given a regex, ``$match`` and ``$eq`` both
    pattern-match instead of comparing: measured on MongoDB 8.0.32, ``{"tier": re.compile("^gol")}`` and
    ``{"tier": bson.Regex("^gol")}`` each matched the document whose tier is ``"gold"``, where an exact
    match on the six characters ``^gol`` matches nothing. Neither was rejected by the operator guard,
    because neither contains a key at all. Reported by a production-readiness review.

    Refused rather than supported, and that is a policy choice with a date on it. A pattern filter is a
    reasonable thing to want; it is also the difference between a predicate a reader can evaluate by
    looking at it and one whose cost and blast radius depend on the pattern. If it is wanted later it
    should arrive as a named, declared capability rather than as a value type that happens to work.

    Both flavours are refused, because both reach the server as a BSON regex: Python's own
    ``re.Pattern``, and pymongo's ``bson.Regex``, which exists precisely to carry a pattern this package's
    filters must not carry. Lists and tuples are walked for the same reason the operator guard walks them.

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
    """Raise if the filter names a path the index did not declare filterable (``REQ-E-049``).

    It was assumed Atlas answers a pre-filter over an undeclared path with unfiltered results rather than an
    error, which made this guard the only thing between a caller and silently cross-scope reads.
    Measured on mongot 8.3.8, it is code 8, ``Path 'x' needs to be indexed as filter``. The
    guard stays for a smaller but real reason: the server names the path and stops there, while a caller
    who typed a field name needs the config key to edit, the index rebuild it implies, and the list of
    what is filterable now. It also fails before the round trip rather than after it. ``REQ-E-071``
    records the correction; unverified on Atlas proper, so the loud behaviour is not assumed everywhere.

    Top-level keys only, and that is exact rather than approximate. ``_reject_operator_keys`` has
    already rejected ``$and`` and ``$or``, so no caller filter can place a field path below the top
    level, and a dict value is MQL exact-subdocument equality whose path is the key above it.

    Args:
        caller_filter: A filter that has already passed the operator guard.
        declared: Every path the index declares filterable: ``scope``, ``store_name`` and the config's
            ``filter_fields``. Passed in rather than derived, so the index definition stays the single
            authority.

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

    ``REQ-E-050``, ``REQ-E-174`` and ``REQ-E-049``, run in that order, and the order is part of the
    contract. The two guards that ask what the filter *is* run before the one that asks whether the index
    declared its fields: ``{"$where": ...}`` is both forbidden and undeclared, and a caller told their
    ``$where`` key is missing from ``filter_fields`` would reasonably try to declare it. The same argument
    puts the regex guard second rather than last, and it is second rather than first because an operator
    key is the more serious of the two and should be the message a caller sees when a filter has both.

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
    """Compose caller filters with the scope predicate as a conjunction (``REQ-E-051``).

    ``$and`` rather than a merged dict, and that is the requirement rather than a style preference.
    ``{**scope_predicate, **caller_filter}`` lets a caller key of the same name win, and the key whose
    name is worth overwriting is ``scope``. Under a conjunction the caller's term can only narrow.

    Verified on MongoDB 8.3.8: ``{"$and": [{"scope": {"$eq": "tenant-a"}}, {"scope": "tenant-b"}]}``
    matches 0 documents. A contradictory caller term yields nothing rather than yielding their term,
    which is what makes conjunction sufficient here and lets the tests assert on the emitted dict.

    Variadic because a memory store has **two** caller filters and they arrive from different places:
    the one configured on the store, which applies to every query, and the one on a single
    ``MongoDBSearchQuery``. Merging the pair before composing is not available, since the merge would
    have to be a ``$and`` of its own and :func:`validate` rejects a ``$``-prefixed key at any depth. Two
    terms in one conjunction is the same predicate either way, and each stays a plain field-to-value
    mapping that the guards can still read.

    Does not validate. Callers run :func:`validate` first, because the two have different arguments and
    a caller with no declared-field set to hand should still be able to compose.

    Args:
        *caller_filters: Caller filters, already validated, in the order they should appear.
            ``None`` and ``{}`` both add no term, so a call site with nothing to add passes either.
        scope: This store's scope. Never taken from caller input (``REQ-E-046``); an absent scope
            resolves to a sentinel upstream, never to unfiltered (``REQ-E-048``).
        store_name: This store's name, isolating it from other stores in the same collection.

    Returns:
        A ``$and`` predicate with the scope term first, then the store term, then every caller filter
        that has anything in it. Deep-copied, so the caller's dicts and the returned predicate share no
        structure in either direction.
    """
    terms: list[dict[str, Any]] = [
        {"scope": {"$eq": scope}},
        {"store_name": {"$eq": store_name}},
    ]
    for caller_filter in caller_filters:
        if caller_filter:
            # Deep, not shallow. A caller who mutates a nested value after composing must not be able
            # to alter a query that is already in flight.
            terms.append(copy.deepcopy(dict(caller_filter)))
    return {"$and": terms}
