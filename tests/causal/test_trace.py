"""Tests for ``causalab.causal.trace``.

``trace.py`` defines the runtime substrate of every causal model: ``Mechanism``
(dataclass binding parents → compute → lazy flag), ``input_var`` (factory for
free-input variables that sample from a list), and ``CausalTrace`` (the
stateful object that holds variable values during a single forward pass and
supports interventions). ``CausalModel``, counterfactual dataset generation,
interchange interventions, and the runner pipeline all instantiate
``CausalTrace`` objects per example. Wrong values here silently corrupt every
downstream counterfactual sample and intervention experiment.
"""

from __future__ import annotations

import pytest

from causalab.causal.trace import CausalTrace, Mechanism, input_var


def _ab_chain() -> dict[str, Mechanism]:
    """Two-node mechanisms dict: A is input, B = A."""
    return {
        "A": input_var([0, 1]),
        "B": Mechanism(parents=["A"], compute=lambda t: t["A"]),
    }


def _abc_chain() -> dict[str, Mechanism]:
    """Three-node mechanisms dict: A -> B -> C, each step adds 1."""
    return {
        "A": input_var([0, 1, 2]),
        "B": Mechanism(parents=["A"], compute=lambda t: t["A"] + 1),
        "C": Mechanism(parents=["B"], compute=lambda t: t["B"] + 1),
    }


class TestMechanismUnit:
    """``Mechanism`` dataclass: parents/compute/lazy fields and ``__call__`` delegation."""

    pytestmark = pytest.mark.unit

    def test_default_lazy_is_false(self):
        mech = Mechanism(parents=["A"], compute=lambda t: 1)
        assert mech.lazy is False

    def test_call_delegates_to_compute(self):
        mech = Mechanism(parents=[], compute=lambda t: 42)
        # __call__(trace) returns compute(trace); trace can be None here since
        # the compute closure ignores it.
        assert mech(None) == 42

    def test_parents_field_is_stored(self):
        mech = Mechanism(parents=["A", "B"], compute=lambda t: 0)
        assert mech.parents == ["A", "B"]

    def test_lazy_can_be_set(self):
        mech = Mechanism(parents=["A"], compute=lambda t: 1, lazy=True)
        assert mech.lazy is True


class TestInputVarUnit:
    """``input_var`` factory: parentless mechanism sampling from a fixed value list."""

    pytestmark = pytest.mark.unit

    def test_returns_mechanism_with_empty_parents(self):
        mech = input_var([0, 1, 2])
        assert isinstance(mech, Mechanism)
        assert mech.parents == []

    def test_sampled_value_in_value_list(self):
        values = [10, 20, 30]
        mech = input_var(values)
        # Call repeatedly; every draw must be in the list.
        for _ in range(20):
            assert mech(None) in values


class TestCausalTraceUnit:
    """``CausalTrace``: forward-evaluation, intervention, copy, and to_dict semantics."""

    pytestmark = pytest.mark.unit

    def test_init_auto_computes_descendants(self):
        trace = CausalTrace(_ab_chain(), inputs={"A": 1})
        assert trace["A"] == 1
        assert trace["B"] == 1

    def test_get_raises_keyerror_for_uncomputed(self):
        # Construct with no inputs -> B is never computed.
        trace = CausalTrace(_ab_chain())
        with pytest.raises(KeyError):
            trace.get("B")

    def test_contains_protocol(self):
        trace = CausalTrace(_ab_chain(), inputs={"A": 0})
        assert "A" in trace
        assert "B" in trace
        assert "Z" not in trace

    def test_delitem_removes_cached_value(self):
        trace = CausalTrace(_ab_chain(), inputs={"A": 1})
        assert "B" in trace
        del trace["B"]
        assert "B" not in trace

    def test_intervene_breaks_causal_link(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        # Without intervention: A=0, B=1, C=2.
        assert trace["B"] == 1
        assert trace["C"] == 2

        # Intervene on B -> 99. C should recompute to 100.
        trace.intervene("B", 99)
        assert trace["B"] == 99
        assert trace["C"] == 100

        # Mutate A — B must remain pinned at 99 because intervene replaced
        # its mechanism with a constant.
        trace.intervene("A", 1)
        assert trace["B"] == 99
        assert trace["C"] == 100

    def test_setitem_calls_intervene(self):
        # __setitem__ has intervention semantics (per the docstring).
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace["B"] = 50
        assert trace["B"] == 50
        # Ancestor mutation must not overwrite the pinned B.
        trace.intervene("A", 2)
        assert trace["B"] == 50

    def test_copy_is_independent(self):
        original = CausalTrace(_abc_chain(), inputs={"A": 0})
        clone = original.copy()
        clone.intervene("B", 77)
        # Original is untouched.
        assert original["B"] == 1
        assert original["C"] == 2
        # Clone reflects the intervention.
        assert clone["B"] == 77
        assert clone["C"] == 78

    def test_to_dict_returns_plain_dict(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        d = trace.to_dict()
        assert isinstance(d, dict)
        assert d == {"A": 0, "B": 1, "C": 2}

    def test_to_dict_returns_copy_not_alias(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        d = trace.to_dict()
        d["A"] = 999
        # Mutating the returned dict must not bleed into the trace.
        assert trace["A"] == 0

    def test_lazy_mechanism_resolved_on_access(self):
        mechs = {
            "A": input_var([0, 1]),
            "B": Mechanism(parents=["A"], compute=lambda t: t["A"] + 1, lazy=True),
        }
        trace = CausalTrace(mechs, inputs={"A": 1})
        # B is lazy; not stored eagerly during init.
        assert "B" not in trace
        # First access computes it.
        assert trace["B"] == 2
        assert "B" in trace


class TestCausalTraceProperty:
    """Invariants of ``CausalTrace`` that hold across mechanism shapes."""

    pytestmark = pytest.mark.property

    def test_intervention_persists_under_ancestor_change(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace.intervene("B", 7)
        # Vary A across several values; B must stay pinned.
        for a_val in [0, 1, 2]:
            trace.intervene("A", a_val)
            assert trace["B"] == 7

    def test_copy_idempotent_round_trip(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 2})
        clone = trace.copy().copy()
        for var in ("A", "B", "C"):
            assert clone[var] == trace[var]

    def test_to_dict_matches_per_variable_get(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 1})
        d = trace.to_dict()
        for var, value in d.items():
            assert trace.get(var) == value

    def test_setitem_observationally_equals_intervene(self):
        trace_a = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace_b = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace_a["B"] = 42
        trace_b.intervene("B", 42)
        assert trace_a.to_dict() == trace_b.to_dict()

    @pytest.mark.parametrize("a_value", [0, 1, 2])
    def test_descendant_recompute_terminates_on_chain(self, a_value):
        # A 3-node chain: descendant recompute is bounded by graph depth.
        trace = CausalTrace(_abc_chain(), inputs={"A": a_value})
        assert trace["A"] == a_value
        assert trace["B"] == a_value + 1
        assert trace["C"] == a_value + 2

    def test_fresh_trace_equivalent_to_post_intervention(self):
        # Forward-pass with A=1 should produce the same dict as
        # forward-pass with A=0 followed by intervene("A", 1).
        fresh = CausalTrace(_abc_chain(), inputs={"A": 1})
        intervened = CausalTrace(_abc_chain(), inputs={"A": 0})
        intervened.intervene("A", 1)
        assert fresh.to_dict() == intervened.to_dict()

    def test_lazy_value_invalidated_on_ancestor_intervene(self):
        mechs = {
            "A": input_var([0, 1, 2]),
            "B": Mechanism(parents=["A"], compute=lambda t: t["A"] * 10, lazy=True),
        }
        trace = CausalTrace(mechs, inputs={"A": 1})
        # Force B to materialize.
        assert trace["B"] == 10
        # Now intervene on A; lazy descendant must be invalidated.
        trace.intervene("A", 2)
        assert trace["B"] == 20
