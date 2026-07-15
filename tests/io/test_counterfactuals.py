"""Tests for ``causalab.io.counterfactuals`` (save/load round-trip).

Coverage for ``generate_counterfactual_samples`` and
``display_counterfactual_examples`` (both in ``causalab.causal.causal_utils``)
lives in ``tests/causal/test_causal_utils.py``, where it gets direct
attribution for that module.
"""

import tempfile
import os
import pytest

from causalab.io.counterfactuals import (
    save_counterfactual_examples,
    load_counterfactual_examples,
)
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.causal.causal_model import CausalModel


pytestmark = pytest.mark.unit


class TestSaveLoadCounterfactualExamples:
    """Tests for save and load counterfactual examples functions."""

    @pytest.fixture
    def simple_causal_model(self):
        """Create a simple CausalModel for testing."""
        mechanisms = {
            "raw_input": Mechanism(parents=[], compute=lambda t: "test"),
            "var_a": Mechanism(parents=["raw_input"], compute=lambda t: 0),
            "var_b": Mechanism(parents=["raw_input"], compute=lambda t: ""),
            "raw_output": Mechanism(
                parents=["var_a", "var_b"],
                compute=lambda t: f"{t['var_a']}_{t['var_b']}",
            ),
        }
        values = {
            "raw_input": ["input1", "input2", "cf1_1", "cf1_2", "cf2_1"],
            "var_a": [0, 1, 2, 3, 10, 20],
            "var_b": ["", "hello", "world", "test", "foo", "bar"],
            "raw_output": ["0_", "1_hello"],
        }
        return CausalModel(mechanisms=mechanisms, values=values)

    def test_roundtrip(self, simple_causal_model: CausalModel) -> None:
        """Test saving and loading counterfactual examples."""
        # Create traces using the causal model
        input1 = simple_causal_model.new_trace(
            {"raw_input": "input1", "var_a": 1, "var_b": "hello"}
        )
        cf1_1 = simple_causal_model.new_trace(
            {"raw_input": "cf1_1", "var_a": 2, "var_b": "world"}
        )
        cf1_2 = simple_causal_model.new_trace(
            {"raw_input": "cf1_2", "var_a": 3, "var_b": "test"}
        )
        input2 = simple_causal_model.new_trace(
            {"raw_input": "input2", "var_a": 10, "var_b": "foo"}
        )
        cf2_1 = simple_causal_model.new_trace(
            {"raw_input": "cf2_1", "var_a": 20, "var_b": "bar"}
        )

        examples: list[CounterfactualExample] = [
            {"input": input1, "counterfactual_inputs": [cf1_1, cf1_2]},
            {"input": input2, "counterfactual_inputs": [cf2_1]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_examples.json")

            # Save
            save_counterfactual_examples(examples, path)
            assert os.path.exists(path)

            # Load (returns CausalTrace objects)
            loaded = load_counterfactual_examples(path, simple_causal_model)

            # Verify
            assert len(loaded) == 2

            # Check first example (accessed via CausalTrace)
            assert loaded[0]["input"]["raw_input"] == "input1"
            assert loaded[0]["input"]["var_a"] == 1
            assert loaded[0]["input"]["var_b"] == "hello"
            assert len(loaded[0]["counterfactual_inputs"]) == 2

            # Check second example
            assert loaded[1]["input"]["raw_input"] == "input2"
            assert len(loaded[1]["counterfactual_inputs"]) == 1

            # Verify loaded items are CausalTrace objects
            assert isinstance(loaded[0]["input"], CausalTrace)
            assert isinstance(loaded[0]["counterfactual_inputs"][0], CausalTrace)
