"""Tests for ``causalab.causal.counterfactual_dataset``.

Defines the two ``TypedDict`` schemas (``CounterfactualExample``,
``LabeledCounterfactualExample``) that every counterfactual sample conforms
to. The base dict pairs a ``CausalTrace`` input with a list of counterfactual
``CausalTrace`` inputs and is produced by
``causalab.causal.causal_utils.generate_counterfactual_samples`` and consumed
by ``io.counterfactuals`` (save/load), ``methods.filter.filter_dataset``,
``methods.metric.*``, ``methods.pca``, and
``CausalModel.label_counterfactual_data``. Wrong/missing keys break
serialization round-trips, DAS/DBM training loss (needs ``label``), and every
runner that ingests a counterfactual dataset.

NOTE: ``TypedDict`` is runtime-permissive (extra keys accepted; missing keys
not raised on construction). The tests below assert *structural* invariants
that consumers depend on, not schema enforcement.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.causal.trace import CausalTrace

from tests._helpers.tiny import tiny_chain_model


@pytest.fixture
def example_pair() -> tuple[CausalTrace, CausalTrace]:
    """Base/counterfactual trace pair from the 3-node A→B→C model."""
    model = tiny_chain_model()
    base = model.new_trace({"A": 0})
    cf = model.new_trace({"A": 1})
    return base, cf


class TestCounterfactualExampleUnit:
    """Schema for the base ``CounterfactualExample`` consumed by all downstream code."""

    pytestmark = pytest.mark.unit

    def test_dict_literal_constructs(self, example_pair):
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
        }
        assert isinstance(example, dict)

    def test_required_keys_reachable(self, example_pair):
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
        }
        assert example["input"] is base
        assert example["counterfactual_inputs"] == [cf]

    def test_type_annotations_declared(self):
        # Static type contract — consumers rely on `input: CausalTrace` and
        # `counterfactual_inputs: list[CausalTrace]`.
        # `CausalTrace` is a TYPE_CHECKING-only forward ref in the module; pass
        # it via localns so get_type_hints can resolve.
        hints = get_type_hints(
            CounterfactualExample, localns={"CausalTrace": CausalTrace}
        )
        assert "input" in hints
        assert "counterfactual_inputs" in hints

    def test_extra_keys_accepted_at_runtime(self, example_pair):
        # TypedDict is runtime-permissive; consumers must not break if extra
        # keys appear (e.g. "label", "setting").
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "extra_field": "anything",  # type: ignore[typeddict-unknown-key]
        }
        assert example["extra_field"] == "anything"  # type: ignore[typeddict-item]


class TestLabeledCounterfactualExampleUnit:
    """``LabeledCounterfactualExample`` extends the base schema with ``label``."""

    pytestmark = pytest.mark.unit

    def test_dict_literal_with_label_constructs(self, example_pair):
        base, cf = example_pair
        example: LabeledCounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "label": "any-value",
        }
        assert example["label"] == "any-value"

    def test_label_key_is_in_type_hints(self):
        hints = get_type_hints(
            LabeledCounterfactualExample, localns={"CausalTrace": CausalTrace}
        )
        assert "label" in hints

    def test_inherits_base_keys(self):
        # Static-type inheritance: the labeled schema must include the base
        # keys, so downstream code typed against `CounterfactualExample`
        # accepts labeled examples too.
        hints = get_type_hints(
            LabeledCounterfactualExample, localns={"CausalTrace": CausalTrace}
        )
        assert "input" in hints
        assert "counterfactual_inputs" in hints


class TestCounterfactualExampleProperty:
    """Structural invariants that downstream consumers (``filter_dataset``,
    ``save_counterfactual_examples``, ``label_counterfactual_data``) depend on."""

    pytestmark = pytest.mark.property

    def test_counterfactual_inputs_is_indexable_list(self, example_pair):
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf, cf, cf],
        }
        # Must be a list (consumers index it via [0], [i], etc.).
        assert isinstance(example["counterfactual_inputs"], list)
        # Indexable.
        assert example["counterfactual_inputs"][0] is cf
        assert example["counterfactual_inputs"][2] is cf

    def test_input_and_cf_share_node_set(self, example_pair):
        # Every consumer assumes the cf traces have the same variables as the
        # input trace (otherwise interchange dies). Build both from the same
        # model and confirm.
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
        }
        input_vars = set(example["input"].to_dict())
        for cf_trace in example["counterfactual_inputs"]:
            assert set(cf_trace.to_dict()) == input_vars

    def test_empty_counterfactual_list_allowed(self, example_pair):
        # No consumer crashes on an empty list (it just no-ops).
        base, _ = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [],
        }
        assert example["counterfactual_inputs"] == []


class TestLabeledCounterfactualExampleProperty:
    """A ``LabeledCounterfactualExample`` must be drop-in compatible with every
    consumer typed against the base ``CounterfactualExample``."""

    pytestmark = pytest.mark.property

    def test_labeled_satisfies_base_schema(self, example_pair):
        base, cf = example_pair
        labeled: LabeledCounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "label": 42,
        }
        # Re-narrow to the base type — consumers that only read input /
        # counterfactual_inputs must not break.
        as_base: CounterfactualExample = labeled  # type: ignore[assignment]
        assert as_base["input"] is base
        assert as_base["counterfactual_inputs"] == [cf]

    @pytest.mark.parametrize(
        "label_value", [0, 1, "string", None, [1, 2, 3], {"k": "v"}]
    )
    def test_label_accepts_arbitrary_types(self, example_pair, label_value):
        # `label: Any` — no runtime constraint.
        base, cf = example_pair
        example: LabeledCounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "label": label_value,
        }
        assert example["label"] == label_value
