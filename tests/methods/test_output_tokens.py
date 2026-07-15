"""Unified ``output_tokens`` resolver (#291 phase 1, #292).

Covers the explicit per-value token-form declaration and the single resolver
over it: form-group dedup, tokenizer-aware score-token ids, the derived string
checker, and the back-compat guarantee that ``get_output_token_ids`` is
byte-identical to the legacy path when no ``output_tokens`` is declared.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from causalab.causal.causal_model import (
    CausalModel,
    _validate_match_modes,
    _validate_output_tokens,
    build_output_tokens,
    derive_checker,
)
from causalab.causal.trace import Mechanism, input_var
from causalab.methods.output_tokens import (
    form_group_labels,
    form_groups,
    resolve_score_token_ids,
)
from causalab.runner.helpers import get_output_token_ids


def _tiny_model(**kwargs) -> CausalModel:
    """Smallest valid CausalModel (just the required raw_input/raw_output)."""
    mechanisms = {
        "raw_input": input_var(["a"]),
        "raw_output": Mechanism(parents=["raw_input"], compute=lambda t: "out"),
    }
    values = {"raw_input": ["a"], "raw_output": ["out"]}
    return CausalModel(mechanisms, values, **kwargs)


pytestmark = pytest.mark.unit


class _StubTokenizer:
    """Maps a few strings to fixed ids; unknown strings encode to 2 tokens so
    they are filtered out as multi-token (mirrors test_metric._StubTokenizer)."""

    _TABLE = {
        " blue": [5],
        "blue": [6],
        " red": [7],
        "red": [8],
        " green": [11, 12],  # multi-token both spacings → no single-token form
        "green": [13, 14],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._TABLE.get(text, [98, 99]))


class TestBuildOutputTokens:
    def test_default_prefix_emits_both_spacings(self):
        assert build_output_tokens(["Monday", "Tuesday"]) == {
            "Monday": [" Monday", "Monday"],
            "Tuesday": [" Tuesday", "Tuesday"],
        }

    def test_custom_prefix(self):
        assert build_output_tokens(["A"], prefix=":") == {"A": [":A", "A"]}

    def test_empty_prefix_dedupes_to_one_form(self):
        assert build_output_tokens(["x"], prefix="") == {"x": ["x"]}

    def test_non_string_values_stringified_for_bare_form(self):
        assert build_output_tokens([1]) == {1: [" 1", "1"]}


class TestFormGroups:
    def test_distinct_lists_preserved_in_order(self):
        var_map = {"blue": [" blue", "blue"], "red": [" red", "red"]}
        assert form_groups(var_map) == [[" blue", "blue"], [" red", "red"]]

    def test_identical_form_lists_collapse(self):
        # The 2D case: many (entity, group) tuples share one entity's forms.
        var_map = {
            ("Mon", "g1"): [" Mon", "Mon"],
            ("Mon", "g2"): [" Mon", "Mon"],
            ("Tue", "g1"): [" Tue", "Tue"],
            ("Tue", "g2"): [" Tue", "Tue"],
        }
        assert form_groups(var_map) == [[" Mon", "Mon"], [" Tue", "Tue"]]


class TestFormGroupLabels:
    def test_one_stripped_label_per_distinct_group(self):
        # Labels name the score columns; deduped union of forms, not the keys.
        var_map = {
            ("Mon", "g1"): [" Mon", "Mon"],
            ("Mon", "g2"): [" Mon", "Mon"],
            ("Tue", "g1"): [" Tue", "Tue"],
        }
        assert form_group_labels(var_map) == ["Mon", "Tue"]

    def test_labels_align_with_form_groups(self):
        var_map = build_output_tokens(["jam", "pie", "bread"])
        assert form_group_labels(var_map) == ["jam", "pie", "bread"]


class TestResolveScoreTokenIds:
    def test_keeps_single_token_forms_across_spacings(self):
        tok = _StubTokenizer()
        var_map = {"blue": [" blue", "blue"], "red": [" red", "red"]}
        assert resolve_score_token_ids(tok, var_map) == [[5, 6], [7, 8]]

    def test_multi_token_falls_back_to_first_form_sequence(self):
        tok = _StubTokenizer()
        var_map = {"green": [" green", "green"]}
        # No single-token form → fall back to the first form's full sequence.
        assert resolve_score_token_ids(tok, var_map) == [[11, 12]]

    def test_dedup_collapses_token_groups(self):
        tok = _StubTokenizer()
        var_map = {
            ("blue", "g1"): [" blue", "blue"],
            ("blue", "g2"): [" blue", "blue"],
            ("red", "g1"): [" red", "red"],
        }
        # 3 values, 2 distinct form groups → 2 token-id groups.
        assert resolve_score_token_ids(tok, var_map) == [[5, 6], [7, 8]]


class TestDeriveChecker:
    def test_exact_matches_declared_form(self):
        check = derive_checker({"Monday": [" Monday", "Monday"]}, "exact")
        assert check({"string": "Monday"}, "Monday")
        assert check({"string": " Monday "}, "Monday")  # stripped both sides

    def test_exact_rejects_substring_and_continuation(self):
        check = derive_checker({"Ann": [" Ann", "Ann"]}, "exact")
        assert not check({"string": "Anna"}, "Ann")  # the #167 over-match
        assert not check({"string": "Ann loves"}, "Ann")

    def test_prefix_accepts_continuation(self):
        check = derive_checker({"bread": [" bread", "bread"]}, "prefix")
        assert check({"string": "bread\n\nAnn loves"}, "bread")
        assert not check({"string": "toast"}, "bread")

    def test_fallback_to_literal_when_value_undeclared(self):
        # causal_output names no declared value → matched literally (strip-tolerant).
        check = derive_checker({"Monday": [" Monday", "Monday"]}, "exact")
        assert check({"string": "1"}, "1")
        assert not check({"string": "0"}, "1")

    def test_invalid_match_mode_raises(self):
        with pytest.raises(ValueError, match="exact.*prefix|match_mode"):
            derive_checker({"a": ["a"]}, "fuzzy")


class TestGetOutputTokenIds:
    """``get_output_token_ids`` derives the score-token ids from the variable's
    ``output_tokens`` declaration, and returns ``(None, None)`` when there is no
    declaration to resolve (the legacy ``result_token_pattern`` fallback was
    removed in #291 phase 3)."""

    def _task(self, *, output_tokens):
        return SimpleNamespace(
            intervention_variable="weekday",
            causal_model=SimpleNamespace(output_tokens=output_tokens),
        )

    def _pipeline(self):
        return SimpleNamespace(tokenizer=_StubTokenizer())

    def test_declared_resolves_to_score_token_ids(self):
        task = self._task(
            output_tokens={"weekday": build_output_tokens(["blue", "red"])}
        )
        ids, n = get_output_token_ids(task, self._pipeline())
        assert ids == [[5, 6], [7, 8]]
        assert n == 2

    def test_none_declaration_returns_none(self):
        task = self._task(output_tokens=None)
        assert get_output_token_ids(task, self._pipeline()) == (None, None)

    def test_empty_declaration_returns_none(self):
        task = self._task(output_tokens={"weekday": {}})
        assert get_output_token_ids(task, self._pipeline()) == (None, None)

    def test_no_target_variable_returns_none(self):
        task = SimpleNamespace(
            intervention_variable=None,
            causal_model=SimpleNamespace(output_tokens=None),
        )
        assert get_output_token_ids(task, self._pipeline()) == (None, None)


class TestValidateOutputTokens:
    def test_valid_map_passes(self):
        _validate_output_tokens({"weekday": build_output_tokens(["Monday"])})

    def test_non_dict_top_level_raises(self):
        with pytest.raises(TypeError, match="dict keyed by variable"):
            _validate_output_tokens(["Monday"])

    def test_non_dict_var_map_raises(self):
        with pytest.raises(TypeError, match=r"\[value: \[forms\]\] dict|value"):
            _validate_output_tokens({"weekday": ["Monday"]})

    def test_forms_not_list_of_str_raises(self):
        with pytest.raises(TypeError, match="list\\[str\\]"):
            _validate_output_tokens({"weekday": {"Monday": "Monday"}})
        with pytest.raises(TypeError, match="list\\[str\\]"):
            _validate_output_tokens({"weekday": {"Monday": [1, 2]}})

    def test_empty_forms_raises(self):
        with pytest.raises(ValueError, match="no forms"):
            _validate_output_tokens({"weekday": {"Monday": []}})


class TestValidateMatchModes:
    def test_valid_modes_pass(self):
        _validate_match_modes({"a": "exact", "b": "prefix"})

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="exact.*prefix|match_modes"):
            _validate_match_modes({"a": "fuzzy"})


class TestCausalModelConstructionValidation:
    def test_good_declaration_constructs_and_sets_attrs(self):
        ot = {"raw_output": build_output_tokens(["out"])}
        model = _tiny_model(output_tokens=ot, match_modes={"raw_output": "prefix"})
        assert model.output_tokens == ot
        assert model.match_modes == {"raw_output": "prefix"}

    def test_none_is_the_back_compat_default(self):
        model = _tiny_model()
        assert model.output_tokens is None
        assert model.match_modes is None

    def test_malformed_output_tokens_raises_at_construction(self):
        with pytest.raises(TypeError):
            _tiny_model(output_tokens={"raw_output": "notadict"})

    def test_bad_match_mode_raises_at_construction(self):
        with pytest.raises(ValueError, match="exact.*prefix|match_modes"):
            _tiny_model(match_modes={"raw_output": "fuzzy"})


class TestCanonicalMatchModes:
    """One canonical match-mode set, shared by the validator and the checker (#296)."""

    def test_derive_checker_accepts_every_canonical_mode(self):
        from causalab.causal.causal_model import MATCH_MODES

        for mode in MATCH_MODES:
            # Each declared mode builds a checker without raising.
            assert callable(derive_checker({"a": ["a"]}, mode))

    def test_validator_and_checker_share_one_set(self):
        """A mode outside ``MATCH_MODES`` is rejected by *both* consumers — so the
        allowed set cannot drift between ``_validate_match_modes`` and
        ``derive_checker``."""
        from causalab.causal.causal_model import MATCH_MODES

        bogus = "contains"
        assert bogus not in MATCH_MODES
        with pytest.raises(ValueError):
            _validate_match_modes({"a": bogus})
        with pytest.raises(ValueError):
            derive_checker({"a": ["a"]}, bogus)


class TestResolveChecker:
    """``load_task`` checker resolution: a shipped ``checker.py`` is the bespoke
    override and wins when present; otherwise the checker is derived from
    ``output_tokens`` (#291 phase 3)."""

    def test_derives_from_output_tokens_when_declared(self, monkeypatch):
        import causalab.tasks.loader as loader_mod

        monkeypatch.setattr(loader_mod, "load_task_checker", lambda name: None)
        model = _tiny_model(
            output_tokens={"raw_output": build_output_tokens(["bread"])},
            match_modes={"raw_output": "prefix"},
        )
        check = loader_mod._resolve_checker(model, "anytask", "raw_output")
        assert check.__module__ == "causalab.causal.causal_model"
        # prefix mode came from match_modes → accepts a continuation.
        assert check({"string": "bread\n\nAnn loves"}, "bread")
        assert not check({"string": "toast"}, "bread")

    def test_defaults_to_exact_when_no_match_mode_entry(self, monkeypatch):
        import causalab.tasks.loader as loader_mod

        monkeypatch.setattr(loader_mod, "load_task_checker", lambda name: None)
        model = _tiny_model(output_tokens={"raw_output": build_output_tokens(["Ann"])})
        check = loader_mod._resolve_checker(model, "anytask", "raw_output")
        assert check({"string": "Ann"}, "Ann")
        assert not check({"string": "Anna"}, "Ann")  # exact, not prefix

    def test_uses_shipped_checker_py_when_present(self, monkeypatch):
        """A shipped ``checker.py`` is used whether or not ``output_tokens``
        covers the variable — the bespoke matcher is always available."""
        import causalab.tasks.loader as loader_mod

        sentinel = object()
        monkeypatch.setattr(loader_mod, "load_task_checker", lambda name: sentinel)
        # output_tokens absent entirely.
        model = _tiny_model()
        assert loader_mod._resolve_checker(model, "sometask", "raw_output") is sentinel
        # output_tokens declared, but not for the intervention variable.
        model2 = _tiny_model(output_tokens={"other": build_output_tokens(["x"])})
        assert loader_mod._resolve_checker(model2, "sometask", "raw_output") is sentinel

    def test_bespoke_checker_py_wins_over_derivation(self, monkeypatch):
        """When a task ships a ``checker.py`` *and* declares ``output_tokens`` for
        its variable, the bespoke checker takes precedence (#291 phase 3)."""
        import causalab.tasks.loader as loader_mod

        sentinel = object()
        monkeypatch.setattr(loader_mod, "load_task_checker", lambda name: sentinel)
        model = _tiny_model(output_tokens={"raw_output": build_output_tokens(["x"])})
        assert loader_mod._resolve_checker(model, "sometask", "raw_output") is sentinel

    def test_raises_when_neither_checker_nor_output_tokens(self, monkeypatch):
        """A task with no ``checker.py`` and no ``output_tokens`` for its variable
        cannot grade its output and fails loud."""
        import causalab.tasks.loader as loader_mod

        monkeypatch.setattr(loader_mod, "load_task_checker", lambda name: None)
        model = _tiny_model()
        with pytest.raises(ValueError, match="cannot grade its output"):
            loader_mod._resolve_checker(model, "sometask", "raw_output")
