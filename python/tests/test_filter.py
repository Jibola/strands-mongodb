"""Caller-filter validation and scope composition.

Isolation between scopes is a predicate in a query rather than a database boundary. A caller filter that
widened the read scope, or one that silently failed to narrow it, would return another scope's memories.
These tests pin the guards that prevent both.

The functions under test are pure functions over dicts, so all of this runs in the mock tier. Two questions
depend on what the server does with the dict: whether an operator nested in a list is read as an operator,
and whether a conjunction can be widened by a contradictory term. Both were answered against MongoDB 8.3.8,
and the tests below assert the shape this package emits rather than re-probing the server on every run.
"""

from __future__ import annotations

import re
from collections import ChainMap, UserDict
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import pytest

from strands_mongodb import _filter


class _MinimalMapping(Mapping[str, Any]):
    """A mapping that is not a ``dict`` and does not inherit from one.

    ``UserDict`` wraps a ``dict``, ``MappingProxyType`` proxies one and ``ChainMap`` composes them, so all
    three could be dismissed as library types with a ``dict`` inside. This one is what a caller writes: a
    class registered as a mapping and nothing more. bson encodes it, and the mapping guard has to walk it.
    """

    def __init__(self, contents: dict[str, Any]) -> None:
        self._contents = contents

    def __getitem__(self, key: str) -> Any:
        return self._contents[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._contents)

    def __len__(self) -> int:
        return len(self._contents)


# The paths the search index definition always declares, before any caller-supplied `filter_fields`.
BAKED_IN = ("scope", "store_name")


class TestOperatorKeysAreRejectedAtAnyDepth:
    """AT-050a, AT-050b, AT-050c — operator keys are rejected at any depth, including inside lists.

    The guard rejects keys rather than values, as MongoDB's reference implementation does: an ordinary
    value is data no matter what it contains, while a key is what MQL dispatches on. ``$where`` as a
    *value* is the string ``"$where"``; as a *key* it is server-side JavaScript.
    """

    @pytest.mark.parametrize(
        "operator",
        ["$where", "$expr", "$function", "$accumulator", "$or", "$and", "$nor", "$jsonSchema"],
    )
    def test_a_top_level_operator_is_rejected_by_name(self, operator: str) -> None:
        """AT-050a — the message names the key, so the caller knows which of their keys was refused."""
        with pytest.raises(ValueError, match=r"\$"):
            _filter.validate({operator: "anything"}, declared_fields=BAKED_IN)

    def test_the_message_names_the_offending_key_itself(self) -> None:
        """The message names the refused key, so the caller does not have to bisect their filter to find it."""
        with pytest.raises(ValueError) as caught:
            _filter.validate({"$where": "sleep(1000)"}, declared_fields=BAKED_IN)

        assert "$where" in str(caught.value)

    def test_an_operator_nested_one_level_down_is_rejected(self) -> None:
        """AT-050b — the check recurses into nested values.

        A top-level-only check is bypassed by nesting one level. ``{"a": {"$ne": None}}`` is the canonical
        case: it matches every document where ``a`` holds any value at all, which turns a narrowing filter
        into a match-all.
        """
        with pytest.raises(ValueError, match=r"\$ne"):
            _filter.validate({"a": {"$ne": None}}, declared_fields=("a",))

    def test_an_operator_nested_three_levels_down_is_rejected(self) -> None:
        """Depth is unbounded, so a deeply nested operator is rejected too."""
        with pytest.raises(ValueError, match=r"\$gt"):
            _filter.validate({"a": {"b": {"c": {"$gt": 1}}}}, declared_fields=("a",))

    def test_an_operator_inside_a_list_is_rejected(self) -> None:
        """An operator inside a list is rejected, which MongoDB's reference guard does not do.

        The reference guard recurses into ``dict`` values only, so ``{"a": [{"$where": "sleep(1000)"}]}``
        passes it. This package treats a list as one more level of depth.

        This is defense in depth. In a ``$match``, a dict inside a list is an equality operand rather than
        an operator position: on MongoDB 8.3.8 that filter matches 0 of 2 documents, and every construct
        that would read it as operators is a ``$`` key the guard already rejects. It is rejected anyway
        because that safety is a property of ``$match``, not of this function, and the same dict handed to
        a different stage would not inherit it.
        """
        with pytest.raises(ValueError, match=r"\$where"):
            _filter.validate({"a": [{"$where": "sleep(1000)"}]}, declared_fields=("a",))

    def test_an_operator_inside_a_nested_list_is_rejected(self) -> None:
        """Lists inside lists are walked too: ``{"a": [[{"$expr": {}}]]}`` also passes the reference guard."""
        with pytest.raises(ValueError, match=r"\$expr"):
            _filter.validate({"a": [[{"$expr": {}}]]}, declared_fields=("a",))

    def test_an_operator_inside_a_tuple_is_rejected(self) -> None:
        """A tuple reaches BSON as an array, so an operator inside one is rejected as inside a list.

        It has its own test because ``isinstance(value, list)`` is the obvious implementation and it misses
        tuples.
        """
        with pytest.raises(ValueError, match=r"\$where"):
            _filter.validate({"a": ({"$where": "x"},)}, declared_fields=("a",))

    @pytest.mark.parametrize("key", [1, 2.5, None, True, (1, 2)])
    def test_a_non_string_key_is_rejected(self, key: Any) -> None:
        """AT-050c — non-string keys are rejected.

        BSON cannot encode them either (``bson.encode({1: "x"})`` raises ``InvalidDocument``), so this
        check decides where the failure lands. This one names the caller's filter; BSON's would arrive from
        inside the driver mid-query, after the scope predicate has been composed around the filter.
        """
        with pytest.raises(ValueError):
            _filter.validate({key: "x"}, declared_fields=BAKED_IN)

    def test_a_non_string_key_nested_in_a_dict_is_rejected(self) -> None:
        """The recursion carries the key-type check with it, not only the ``$`` check."""
        with pytest.raises(ValueError):
            _filter.validate({"a": {1: "x"}}, declared_fields=("a",))

    def test_a_list_of_plain_values_is_allowed(self) -> None:
        """A list with no operator in it passes, so the list walk does not reject clean lists.

        ``{"tier": ["gold", "silver"]}`` is ordinary MQL, matching a document whose ``tier`` array equals
        that list, and rejecting it would break a documented filter shape. Every list in the tests above
        raises, so this is also the only test of the path where the walk finishes normally.
        """
        _filter.validate({"tier": ["gold", "silver"]}, declared_fields=("tier",))
        _filter.validate({"tier": []}, declared_fields=("tier",))
        _filter.validate({"tier": [{"nested": "ok"}]}, declared_fields=("tier",))

    def test_a_dollar_sign_that_is_not_a_prefix_is_allowed(self) -> None:
        """A ``$`` that is not the first character is allowed: a field named ``price$usd`` is not an operator.

        This is the negative half of the guard. A check written as ``"$" in key`` passes every test above
        and rejects valid field names.
        """
        _filter.validate({"price$usd": 10}, declared_fields=("price$usd",))

    def test_a_benign_operator_is_rejected_too_and_that_is_the_requirement(self) -> None:
        """Comparison operators such as ``$gt`` are rejected too, because the guard has no allow-list.

        A caller filter is a flat map of path to exact value. Allowing comparison operators would need a
        threat model of its own, so this test pins the current policy and makes any widening a deliberate
        change.
        """
        with pytest.raises(ValueError, match=r"\$gt"):
            _filter.validate({"age": {"$gt": 5}}, declared_fields=("age",))


class TestTheGuardsCoverEveryMappingAndRefuseEveryRegex:
    """AT-179 — operator keys are found in every mapping type, and regular-expression values are refused.

    Both guards keep a caller filter to exact-value matching, which is the filter policy this package
    documents:

    * bson encodes **any** mapping as a subdocument, not only a ``dict``. A guard that walked only ``dict``
      values would never see the ``$ne`` inside a ``UserDict``, and the server would still receive it as an
      operator;
    * a regex value is dispatched on by ``$match`` just as an operator key is: it pattern-matches where the
      policy promises comparison.

    On MongoDB 8.0.32, against two documents whose ``tier`` is ``"gold"`` and ``"silver"``, a guard without
    these checks lets each of the following through ``validate`` and ``compose``:

    * ``{"tier": UserDict({"$ne": "gold"})}`` composes to
      ``{"$and": [{"scope": ...}, {"store_name": ...}, {"tier": {"$ne": "gold"}}]}`` and matches the silver
      document, the inverse of the filter's stated meaning;
    * ``{"tier": {"inner": UserDict({"$ne": None})}}`` passes for the same reason;
    * ``{"tier": re.compile("^gol")}`` matches the gold document, where an exact match on the four
      characters ``^gol`` matches nothing;
    * ``{"tier": bson.Regex("^gol")}`` does the same.

    The scope conjunction holds in every case: ``compose`` pins ``scope`` and ``store_name`` itself, and a
    caller term can only narrow the read within that scope. These guards make the caller's own term mean
    what it says.
    """

    DECLARED = ("tier", "scope", "store_name")

    @staticmethod
    def _mappings(payload: dict[str, Any]) -> dict[str, Any]:
        """Return the payload wrapped in each mapping type that is not a ``dict``.

        Args:
            payload: The contents to wrap.

        Returns:
            One entry per mapping flavour, keyed by a name a parametrized test id can carry.
        """
        return {
            "UserDict": UserDict(payload),
            "MappingProxyType": MappingProxyType(payload),
            "ChainMap": ChainMap(payload),
            "custom Mapping": _MinimalMapping(payload),
        }

    def test_bson_encodes_a_non_dict_mapping_exactly_as_a_dict(self) -> None:
        """bson encodes a non-dict mapping byte for byte as it encodes a ``dict``.

        This is the premise of the mapping guard. Because the two encodings are identical, an operator key
        inside a ``UserDict`` reaches the server as an operator, so the guard has to walk every mapping.
        """
        import bson

        assert bson.encode({"tier": UserDict({"$ne": "gold"})}) == bson.encode({"tier": {"$ne": "gold"}})

    @pytest.mark.parametrize("flavour", ["UserDict", "MappingProxyType", "ChainMap", "custom Mapping"])
    def test_an_operator_inside_a_mapping_value_is_rejected(self, flavour: str) -> None:
        """An operator key inside any mapping value is rejected, not only inside a ``dict``.

        The four here are three standard-library mapping types and a minimal custom ``Mapping``.
        """
        value = self._mappings({"$ne": "gold"})[flavour]

        with pytest.raises(ValueError, match=r"\$ne"):
            _filter.validate({"tier": value}, declared_fields=self.DECLARED)

    def test_an_operator_inside_a_mapping_nested_one_level_down_is_rejected(self) -> None:
        """The mapping check applies at every depth, not only to a top-level value."""
        with pytest.raises(ValueError, match=r"\$ne"):
            _filter.validate({"tier": {"inner": UserDict({"$ne": None})}}, declared_fields=self.DECLARED)

    def test_an_operator_inside_a_mapping_inside_a_list_is_rejected(self) -> None:
        """A mapping inside a list is walked too, combining the list walk with the mapping check."""
        with pytest.raises(ValueError, match=r"\$where"):
            _filter.validate({"tier": [UserDict({"$where": "sleep(1000)"})]}, declared_fields=self.DECLARED)

    def test_a_non_dict_mapping_is_accepted_as_the_filter_itself(self) -> None:
        """A non-dict mapping is accepted as the filter itself, so walking every mapping value does not
        break the top level."""
        _filter.validate(UserDict({"tier": "gold"}), declared_fields=self.DECLARED)

    def test_a_clean_mapping_value_is_still_allowed(self) -> None:
        """A mapping value with no operator key in it passes. It is exact subdocument equality, a documented
        filter shape, and a guard that rejected every mapping would pass every test above while breaking
        callers."""
        _filter.validate({"tier": UserDict({"inner": "gold"})}, declared_fields=self.DECLARED)
        _filter.validate({"tier": MappingProxyType({"inner": "gold"})}, declared_fields=self.DECLARED)

    def test_a_native_regex_value_is_rejected_and_the_message_names_the_field(self) -> None:
        """A compiled ``re`` pattern is rejected and the message names the field. It is the regex type a
        Python caller is most likely to pass."""
        with pytest.raises(ValueError) as caught:
            _filter.validate({"tier": re.compile("^gol")}, declared_fields=self.DECLARED)

        assert "'tier'" in str(caught.value)
        assert "regular expression" in str(caught.value)

    def test_a_bson_regex_value_is_rejected(self) -> None:
        """A ``bson.Regex`` value is rejected.

        ``_filter`` imports no part of pymongo, so it recognises this type by class name rather than with
        ``isinstance``. This test fails if that name ever changes.
        """
        import bson

        with pytest.raises(ValueError, match="regular expression"):
            _filter.validate({"tier": bson.Regex("^gol")}, declared_fields=self.DECLARED)

    def test_a_subclass_of_the_bson_regex_is_rejected_too(self) -> None:
        """A subclass of ``bson.Regex`` is rejected too, which is why the guard walks the MRO rather than
        comparing ``type(value)``."""
        import bson

        class Sneaky(bson.Regex):  # type: ignore[type-arg,misc]
            """A ``bson.Regex`` under a different class name."""

        with pytest.raises(ValueError, match="regular expression"):
            _filter.validate({"tier": Sneaky("^gol")}, declared_fields=self.DECLARED)

    def test_a_regex_nested_in_a_mapping_or_a_list_is_rejected(self) -> None:
        """A regex is rejected at any depth, on the same terms as an operator key, including inside a
        non-dict mapping."""
        with pytest.raises(ValueError, match="regular expression"):
            _filter.validate({"tier": {"inner": re.compile("x")}}, declared_fields=self.DECLARED)
        with pytest.raises(ValueError, match="regular expression"):
            _filter.validate({"tier": [re.compile("x")]}, declared_fields=self.DECLARED)
        with pytest.raises(ValueError, match="regular expression"):
            _filter.validate({"tier": UserDict({"inner": re.compile("x")})}, declared_fields=self.DECLARED)

    def test_a_string_that_looks_like_a_pattern_is_still_a_string(self) -> None:
        """A string that looks like a pattern is a plain string and passes.

        ``"^gol"`` as a value is compared character by character. Rejecting it would refuse legitimate
        filters over fields whose contents contain regex punctuation, such as paths, version ranges or
        formulas.
        """
        _filter.validate({"tier": "^gol.*$"}, declared_fields=self.DECLARED)
        _filter.validate({"tier": ["^gol", "silver"]}, declared_fields=self.DECLARED)

    def test_an_operator_key_is_the_message_when_a_filter_has_both_faults(self) -> None:
        """When a filter has both an operator key and a regex value, the message reports the operator key.

        An operator key is the more serious fault, and its remedy is not "use a different value", so it is
        reported first. The order is asserted because the guards are independent functions, and reordering
        them is a one-line edit with no other symptom.
        """
        with pytest.raises(ValueError) as caught:
            _filter.validate({"tier": {"$ne": re.compile("x")}}, declared_fields=self.DECLARED)

        assert "$ne" in str(caught.value)
        assert "regular expression" not in str(caught.value)


class TestUndeclaredFieldsAreRejectedBeforeQuerying:
    """AT-049 — a filter on a path the search index does not declare is rejected before the query runs.

    The server rejects such a query too: mongot 8.3.8 answers code 8, ``Path 'x' needs to be indexed as
    filter``. This guard fails before the round trip, and it names the remedy the server error does not:
    the ``filter_fields`` option and an index rebuild.
    """

    def test_an_undeclared_field_raises_naming_the_field(self) -> None:
        """AT-049 — an undeclared path is rejected and the message names it."""
        with pytest.raises(ValueError, match=r"metadata\.foo"):
            _filter.validate({"metadata.foo": "x"}, declared_fields=("scope", "store_name"))

    def test_the_message_names_what_was_declared_as_well(self) -> None:
        """Naming only the rejected field leaves the caller guessing what they may filter on.

        The remedy for this error is to add the path to ``filter_fields`` and recreate the index, and the
        message is where a caller learns that the option exists.
        """
        with pytest.raises(ValueError) as caught:
            _filter.validate({"metadata.foo": "x"}, declared_fields=("scope", "store_name"))

        message = str(caught.value)
        assert "filter_fields" in message
        assert "scope" in message and "store_name" in message

    def test_a_declared_field_passes(self) -> None:
        """The positive half, including a caller-supplied path alongside the baked-in ones."""
        _filter.validate({"metadata.tier": "gold"}, declared_fields=("scope", "store_name", "metadata.tier"))

    def test_the_baked_in_paths_are_filterable_without_being_configured(self) -> None:
        """``scope`` and ``store_name`` are filterable without configuration, because the index definition
        always declares them.

        Validating against ``filter_fields`` alone would reject a filter that Atlas applies correctly.
        """
        _filter.validate({"scope": "tenant-a"}, declared_fields=("scope", "store_name"))
        _filter.validate({"store_name": "facts"}, declared_fields=("scope", "store_name"))

    def test_every_undeclared_field_is_named_not_just_the_first(self) -> None:
        """A caller who got three paths wrong learns that in one round trip, not three.

        The paths are distinctive and several characters long on purpose: asserting that ``"a"`` appears in
        an error message proves nothing, since almost any English sentence contains it.
        """
        with pytest.raises(ValueError) as caught:
            _filter.validate(
                {"metadata.tier": 1, "region": 2, "created_at": 3},
                declared_fields=("scope",),
            )

        message = str(caught.value)
        assert "metadata.tier" in message
        assert "region" in message
        assert "created_at" in message

    def test_only_top_level_keys_are_treated_as_field_paths(self) -> None:
        """``{"metadata": {"tier": "gold"}}`` filters on ``metadata``, not on ``metadata.tier``.

        That is MQL's own reading: a dict value is exact-subdocument equality. Because the operator guard
        rejects ``$and`` and ``$or``, no caller filter can put a field path anywhere but the top level, so
        the check needs no path walking, and this test pins that reasoning.
        """
        _filter.validate({"metadata": {"tier": "gold"}}, declared_fields=("metadata",))

        with pytest.raises(ValueError, match="metadata"):
            _filter.validate({"metadata": {"tier": "gold"}}, declared_fields=("metadata.tier",))

    def test_an_empty_filter_is_accepted(self) -> None:
        """``{}`` names no field, so there is nothing to reject.

        The realistic source is a caller passing through an unset option. Raising here would make the
        absence of a filter an error.
        """
        _filter.validate({}, declared_fields=())

    def test_the_operator_guard_runs_before_the_declared_field_guard(self) -> None:
        """The operator rejection is reported before the undeclared-field one, since ``{"$where": ...}`` is
        both.

        A caller told their ``$where`` key is *not in filter_fields* would reasonably try declaring it. The
        security rejection is the one they need to see.
        """
        with pytest.raises(ValueError) as caught:
            _filter.validate({"$where": "sleep(1000)"}, declared_fields=())

        message = str(caught.value)
        assert "$where" in message
        assert "filter_fields" not in message


class TestScopeCannotBeWidenedByACallerFilter:
    """AT-051 — a caller filter cannot widen the scope, asserted on the emitted predicate.

    The shape assertions rest on measured server behaviour: on MongoDB 8.3.8,
    ``{"$and": [{"scope": {"$eq": "tenant-a"}}, {"scope": "tenant-b"}]}`` matches 0 documents. A
    conjunction with a contradictory term returns nothing, not the caller's term. That makes composition by
    ``$and`` sufficient, and it lets these tests assert on the dict without a server round trip.
    """

    def test_the_scope_predicate_is_always_present(self) -> None:
        """The base case: no caller filter at all still narrows."""
        composed = _filter.compose(None, scope="tenant-a", store_name="facts")

        assert {"scope": {"$eq": "tenant-a"}} in composed["$and"]
        assert {"store_name": {"$eq": "facts"}} in composed["$and"]

    def test_a_caller_filter_is_conjoined_rather_than_merged(self) -> None:
        """The caller filter is conjoined with the scope predicate, never merged into it.

        Under ``$and`` the caller's term is an additional constraint. Under a merge such as
        ``{**scope_pred, **caller}`` a duplicate key wins and *replaces* the scope term, and the read is no
        longer limited to the store's scope.
        """
        composed = _filter.compose({"metadata.tier": "gold"}, scope="tenant-a", store_name="facts")

        assert {"scope": {"$eq": "tenant-a"}} in composed["$and"]
        assert {"metadata.tier": "gold"} in composed["$and"]

    def test_a_caller_filter_naming_scope_cannot_replace_the_scope_predicate(self) -> None:
        """A caller filter naming another ``scope`` adds a term and leaves the store's scope term in place.

        Both terms survive. The server then matches nothing, which is the correct outcome: a caller asking
        for another scope gets no documents rather than that scope's documents.
        """
        composed = _filter.compose({"scope": "tenant-b"}, scope="tenant-a", store_name="facts")

        assert {"scope": {"$eq": "tenant-a"}} in composed["$and"]
        assert {"scope": "tenant-b"} in composed["$and"]

    def test_an_empty_caller_filter_adds_no_term(self) -> None:
        """``{}`` in a ``$and`` is a match-everything term, so it is dropped rather than appended.

        Harmless to MongoDB, but not to a reader: an empty clause inside the predicate that enforces
        isolation invites the misreading that something in there matches everything.
        """
        composed = _filter.compose({}, scope="tenant-a", store_name="facts")

        assert composed["$and"] == [{"scope": {"$eq": "tenant-a"}}, {"store_name": {"$eq": "facts"}}]

    def test_the_scope_term_comes_first(self) -> None:
        """The scope term is always first in the conjunction, where a reader looks for it.

        No MQL semantics depend on the order. A human scanning a logged pipeline for the isolation
        predicate does, and putting it anywhere but position 0 makes that scan a search.
        """
        composed = _filter.compose({"metadata.tier": "gold"}, scope="tenant-a", store_name="facts")

        assert composed["$and"][0] == {"scope": {"$eq": "tenant-a"}}

    def test_the_caller_filter_is_not_mutated(self) -> None:
        """The caller's dict is theirs. Composing must not write the scope into it.

        A caller reusing one filter dict across two stores with different scopes would otherwise carry the
        first store's scope into the second query.
        """
        caller = {"metadata.tier": "gold"}

        _filter.compose(caller, scope="tenant-a", store_name="facts")

        assert caller == {"metadata.tier": "gold"}

    def test_the_composed_predicate_shares_no_structure_with_the_caller_filter(self) -> None:
        """Copied, not aliased. A caller mutating their dict afterwards must not alter a live query.

        This needs its own assertion: a shallow ``list(...)`` append passes the mutation test above while
        still holding the caller's object by reference.
        """
        caller: dict[str, Any] = {"metadata.tier": "gold"}
        composed = _filter.compose(caller, scope="tenant-a", store_name="facts")

        caller["metadata.tier"] = "bronze"

        assert {"metadata.tier": "gold"} in composed["$and"]

    def test_the_sentinel_scope_composes_like_any_other(self) -> None:
        """The sentinel an absent scope resolves to composes like any other scope value.

        The memory store, not ``compose``, chooses the sentinel. What is asserted here is that ``compose``
        has no special case for it: the sentinel becomes an ordinary ``$eq`` term, and a ``None`` scope can
        never silently drop the predicate.
        """
        composed = _filter.compose(None, scope="__no_scope__", store_name="facts")

        assert composed["$and"][0] == {"scope": {"$eq": "__no_scope__"}}
