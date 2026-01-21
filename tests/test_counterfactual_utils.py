"""Tests for counterfactual utility functions in causal_utils."""

import tempfile
import os
import pytest
from typing import Any
from unittest import mock

from causalab.causal.causal_utils import (
    generate_counterfactual_samples,
    display_counterfactual_examples,
    save_counterfactual_examples,
    load_counterfactual_examples,
)
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.causal.causal_model import CausalModel


class TestGenerateCounterfactualSamples:
    """Tests for generate_counterfactual_samples function."""

    @pytest.fixture
    def simple_mechanisms(self):
        """Create simple mechanisms for testing."""
        return {
            "raw_input": Mechanism(parents=[], compute=lambda t: "test input"),
            "var": Mechanism(parents=[], compute=lambda t: 1),
        }

    def test_basic_generation(self, simple_mechanisms: Any) -> None:
        """Test generating samples from a simple sampler."""

        def sampler() -> CounterfactualExample:
            trace = CausalTrace(
                simple_mechanisms, {"raw_input": "original input", "var": 1}
            )
            cf1 = CausalTrace(simple_mechanisms, {"raw_input": "cf 1", "var": 2})
            cf2 = CausalTrace(simple_mechanisms, {"raw_input": "cf 2", "var": 3})
            return {"input": trace, "counterfactual_inputs": [cf1, cf2]}

        samples = generate_counterfactual_samples(5, sampler)

        assert len(samples) == 5
        for sample in samples:
            assert "input" in sample
            assert "counterfactual_inputs" in sample
            assert sample["input"]["raw_input"] == "original input"
            assert len(sample["counterfactual_inputs"]) == 2

    def test_generation_with_filter(self, simple_mechanisms: Any) -> None:
        """Test generating samples with a filter function."""
        counter = [0]

        def sampler() -> CounterfactualExample:
            counter[0] += 1
            trace = CausalTrace(
                simple_mechanisms,
                {"raw_input": f"input {counter[0]}", "var": counter[0]},
            )
            cf = CausalTrace(
                simple_mechanisms,
                {"raw_input": f"cf {counter[0]}", "var": counter[0]},
            )
            return {"input": trace, "counterfactual_inputs": [cf]}

        # Filter that only accepts even-numbered inputs
        def filter_fn(sample: CounterfactualExample) -> bool:
            return sample["input"]["var"] % 2 == 0

        samples = generate_counterfactual_samples(3, sampler, filter=filter_fn)

        assert len(samples) == 3
        for sample in samples:
            assert sample["input"]["var"] % 2 == 0


class TestDisplayCounterfactualExamples:
    """Tests for display_counterfactual_examples function."""

    def test_display_verbose(self):
        """Test displaying examples with verbose output."""
        # Use plain dicts - display_counterfactual_examples works with any dict-like
        examples = [
            {
                "input": {"raw_input": "input1", "var": 1},
                "counterfactual_inputs": [
                    {"raw_input": "cf1_1", "var": 2},
                    {"raw_input": "cf1_2", "var": 3},
                ],
            },
            {
                "input": {"raw_input": "input2", "var": 4},
                "counterfactual_inputs": [{"raw_input": "cf2_1", "var": 5}],
            },
        ]

        with mock.patch("builtins.print") as mock_print:
            result = display_counterfactual_examples(examples, num_examples=2)

            # Verify prints were made
            assert mock_print.call_count > 0
            # Verify result structure
            assert len(result) == 2
            assert 0 in result
            assert 1 in result

    def test_display_quiet(self):
        """Test displaying examples without verbose output."""
        examples = [
            {
                "input": {"raw_input": "input1", "var": 1},
                "counterfactual_inputs": [{"raw_input": "cf1", "var": 2}],
            },
        ]

        with mock.patch("builtins.print") as mock_print:
            result = display_counterfactual_examples(examples, verbose=False)

            # Verify no prints were made
            mock_print.assert_not_called()
            # But result should still contain data
            assert len(result) == 1

    def test_display_limited_examples(self):
        """Test that num_examples limits output."""
        examples = [
            {
                "input": {"raw_input": f"input{i}", "var": i},
                "counterfactual_inputs": [],
            }
            for i in range(10)
        ]

        result = display_counterfactual_examples(
            examples, num_examples=3, verbose=False
        )
        assert len(result) == 3


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
