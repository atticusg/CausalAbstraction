"""
Integration tests for Notebook 01: MCQA Task Definition

Tests the causal model definition, interchange interventions,
and counterfactual dataset generation without using neural networks.
"""

import pytest
import random
from causalab.tasks.MCQA.counterfactuals import (
    sample_answerable_question,
    different_symbol,
)
from causalab.causal.causal_utils import generate_counterfactual_samples


pytestmark = pytest.mark.slow


class TestMCQACausalModel:
    """Test the MCQA causal model structure and operations."""

    def test_load_causal_model(self, causal_model):
        """Test that the positional causal model loads correctly."""
        assert causal_model is not None
        assert causal_model.id is not None

    def test_sample_answerable_question(self):
        """Test sampling questions from the task."""
        example = sample_answerable_question()

        # Check that required keys are present
        assert "template" in example
        assert "object" in example
        assert "color" in example
        assert "symbol0" in example
        assert "symbol1" in example
        assert "choice0" in example
        assert "choice1" in example
        assert "raw_input" in example

    def test_causal_model_forward(self, causal_model):
        """Test running the causal model forward."""
        # sample_answerable_question() already returns a fully computed CausalTrace
        full_setting = sample_answerable_question()

        # Check that output variables are computed
        assert "answer_position" in full_setting
        assert "answer" in full_setting
        assert "raw_output" in full_setting

        # Verify answer_position is valid (0 or 1)
        assert full_setting["answer_position"] in [0, 1]

        # Verify answer is one of the symbols
        assert full_setting["answer"] in [
            full_setting["symbol0"],
            full_setting["symbol1"],
        ]

    def test_causal_model_intervention(self, causal_model):
        """Test fixing a variable in the causal model."""
        example_trace = sample_answerable_question()
        original_position = example_trace["answer_position"]

        # Intervene on answer_position using copy().intervene()
        new_position = int(not original_position)
        intervened_trace = example_trace.copy().intervene(
            "answer_position", new_position
        )

        # Check that the intervention took effect
        assert intervened_trace["answer_position"] == new_position
        assert intervened_trace["answer_position"] != original_position

    def test_interchange_intervention(self, causal_model):
        """Test interchange interventions on the causal model."""
        # Generate two examples with different answer positions
        # sample_answerable_question() returns a fully computed CausalTrace
        setting1 = sample_answerable_question()
        setting2 = sample_answerable_question()

        # Keep sampling until we get different positions
        max_attempts = 50
        attempts = 0
        while (
            setting1["answer_position"] == setting2["answer_position"]
            and attempts < max_attempts
        ):
            setting2 = sample_answerable_question()
            attempts += 1

        if attempts < max_attempts:
            # Perform interchange intervention
            intervened_setting = causal_model.run_interchange(
                setting1, {"answer_position": setting2}
            )

            # Verify the intervention took the value from example2
            assert intervened_setting["answer_position"] == setting2["answer_position"]


class TestCounterfactualDatasets:
    """Test counterfactual dataset generation and discriminative power."""

    def test_different_symbol_dataset(self, small_different_symbol_dataset):
        """Test different_symbol counterfactual dataset generation."""
        assert len(small_different_symbol_dataset) == 8

        # Check first example structure
        example = small_different_symbol_dataset[0]
        assert "input" in example
        assert "counterfactual_inputs" in example
        assert len(example["counterfactual_inputs"]) == 1

        # Verify that symbols are different between original and counterfactual
        orig_symbols = {example["input"]["symbol0"], example["input"]["symbol1"]}
        cf_symbols = {
            example["counterfactual_inputs"][0]["symbol0"],
            example["counterfactual_inputs"][0]["symbol1"],
        }
        assert orig_symbols != cf_symbols

    def test_same_symbol_different_position_dataset(
        self, small_same_symbol_diff_position_dataset
    ):
        """Test same_symbol_different_position counterfactual dataset generation."""
        assert len(small_same_symbol_diff_position_dataset) == 8

        # Check structure
        example = small_same_symbol_diff_position_dataset[0]
        assert "input" in example
        assert "counterfactual_inputs" in example

    def test_random_counterfactual_dataset(self, small_random_dataset):
        """Test random counterfactual dataset generation."""
        assert len(small_random_dataset) == 8

        # Check structure
        example = small_random_dataset[0]
        assert "input" in example
        assert "counterfactual_inputs" in example

    def test_can_distinguish_with_dataset(self, causal_model):
        """Test the can_distinguish_with_dataset function."""
        from causalab.causal.causal_utils import can_distinguish_with_dataset

        # Create a small dataset
        dataset = generate_counterfactual_samples(8, different_symbol)

        # Test distinguishing answer from no intervention
        result = can_distinguish_with_dataset(dataset, causal_model, ["answer"], None)

        # Verify result structure
        assert "proportion" in result
        assert "count" in result
        assert 0 <= result["proportion"] <= 1
        assert 0 <= result["count"] <= len(dataset)

    def test_confounding_example(self, causal_model):
        """Test creating confounded and deconfounded counterfactuals."""
        # sample_answerable_question() returns a fully computed CausalTrace
        full_setting = sample_answerable_question()

        # Extract input variables only for creating modified traces
        input_vars = {var: full_setting[var] for var in causal_model.inputs}

        # Create confounded counterfactual (change object and color)
        confounded_cf = input_vars.copy()
        confounded_cf["object"] = "toy"
        confounded_cf["color"] = full_setting[
            f"choice{int(not full_setting['answer_position'])}"
        ]
        confounded_cf_setting = causal_model.new_trace(confounded_cf)

        # Verify the counterfactual has different answer position
        assert (
            confounded_cf_setting["answer_position"] != full_setting["answer_position"]
        )

        # Create deconfounded counterfactual (different symbols)
        deconfounded_cf = input_vars.copy()
        deconfounded_cf["object"] = "toy"
        deconfounded_cf["color"] = full_setting[
            f"choice{int(not full_setting['answer_position'])}"
        ]
        # Change symbols
        available_symbols = list(
            {"A", "B", "C", "D"}.difference(
                {full_setting["symbol0"], full_setting["symbol1"]}
            )
        )
        if available_symbols:
            deconfounded_cf["symbol0"] = random.choice(available_symbols)
            deconfounded_cf_setting = causal_model.new_trace(deconfounded_cf)

            # Verify structure is valid
            assert "answer" in deconfounded_cf_setting
            assert "answer_position" in deconfounded_cf_setting
