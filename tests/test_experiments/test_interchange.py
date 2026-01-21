# tests/test_experiments/test_interchange.py
"""
Tests for experiments/metric.py - causal_score_intervention_outputs function.
"""

import pytest
from typing import Any
from unittest.mock import MagicMock

from causalab.experiments.metric import causal_score_intervention_outputs


def create_mock_cf_dataset(size: int = 3) -> Any:
    """Create a list of mock CounterfactualExample dicts for testing."""
    return [
        {"input": {"text": f"input_{i}"}, "counterfactual_inputs": []}
        for i in range(size)
    ]


class TestCausalScoreInterventionOutputs:
    """Test causal_score_intervention_outputs function."""

    def test_returns_expected_result_structure(self):
        """Test that causal_score_intervention_outputs returns dict with expected keys."""
        mock_cf_dataset = create_mock_cf_dataset(3)

        mock_causal_model = MagicMock()
        mock_causal_model.label_counterfactual_data = MagicMock(
            return_value=[
                {"input": {"text": "test"}, "label": "A"},
                {"input": {"text": "test2"}, "label": "B"},
                {"input": {"text": "test3"}, "label": "A"},
            ]
        )

        raw_results = {("test",): {"string": ["A", "B", "A"]}}

        result = causal_score_intervention_outputs(
            raw_results=raw_results,
            dataset=mock_cf_dataset,
            causal_model=mock_causal_model,
            target_variable_groups=[("answer",)],
            metric=lambda x, y: x.get("string") == y,
        )

        # Check result structure
        assert "avg_score" in result
        assert "scores_by_variable" in result
        assert "results_by_key" in result

    def test_computes_correct_score(self):
        """Test that causal_score_intervention_outputs computes accuracy correctly."""
        mock_cf_dataset = create_mock_cf_dataset(4)

        mock_causal_model = MagicMock()
        # 4 examples with labels A, B, A, B
        mock_causal_model.label_counterfactual_data = MagicMock(
            return_value=[
                {"label": "A"},
                {"label": "B"},
                {"label": "A"},
                {"label": "B"},
            ]
        )

        # Model outputs: A, A, A, A (correct on 2/4 = 50%)
        raw_results = {("test",): {"string": ["A", "A", "A", "A"]}}

        result = causal_score_intervention_outputs(
            raw_results=raw_results,
            dataset=mock_cf_dataset,
            causal_model=mock_causal_model,
            target_variable_groups=[("answer",)],
            metric=lambda x, y: x.get("string") == y,
        )

        # 2 correct out of 4 = 0.5
        assert result["avg_score"] == 0.5
        assert result["scores_by_variable"][("answer",)] == 0.5

    def test_handles_multiple_target_variables(self):
        """Test that causal_score_intervention_outputs handles multiple target variables."""
        mock_cf_dataset = create_mock_cf_dataset(2)

        mock_causal_model = MagicMock()
        # Return different labels for each target variable
        mock_causal_model.label_counterfactual_data = MagicMock(
            side_effect=[
                [{"label": "A"}, {"label": "A"}],  # For "answer"
                [{"label": "X"}, {"label": "Y"}],  # For "position"
            ]
        )

        raw_results = {("test",): {"string": ["A", "A"]}}

        result = causal_score_intervention_outputs(
            raw_results=raw_results,
            dataset=mock_cf_dataset,
            causal_model=mock_causal_model,
            target_variable_groups=[("answer",), ("position",)],
            metric=lambda x, y: x.get("string") == y,
        )

        # Check scores for each variable group
        assert ("answer",) in result["scores_by_variable"]
        assert ("position",) in result["scores_by_variable"]
        # answer: 2/2 correct = 1.0
        assert result["scores_by_variable"][("answer",)] == 1.0
        # position: 0/2 correct = 0.0
        assert result["scores_by_variable"][("position",)] == 0.0
        # Overall: average of 1.0 and 0.0 = 0.5
        assert result["avg_score"] == 0.5


class TestCausalScoreNestedOutputs:
    """Test handling of nested output structures."""

    def test_handles_nested_string_outputs(self):
        """Test that nested list outputs are flattened correctly."""
        mock_cf_dataset = create_mock_cf_dataset(4)

        mock_causal_model = MagicMock()
        mock_causal_model.label_counterfactual_data = MagicMock(
            return_value=[
                {"label": "A"},
                {"label": "B"},
                {"label": "C"},
                {"label": "D"},
            ]
        )

        # Return nested structure (batched outputs)
        raw_results = {("test",): {"string": [["A", "B"], ["C", "D"]]}}

        result = causal_score_intervention_outputs(
            raw_results=raw_results,
            dataset=mock_cf_dataset,
            causal_model=mock_causal_model,
            target_variable_groups=[("answer",)],
            metric=lambda x, y: x.get("string") == y,
        )

        # All 4 should match after flattening
        assert result["avg_score"] == 1.0


class TestCausalScoreMultipleKeys:
    """Test handling of multiple target keys."""

    def test_handles_multiple_keys(self):
        """Test that multiple keys are scored independently."""
        mock_cf_dataset = create_mock_cf_dataset(2)

        mock_causal_model = MagicMock()
        mock_causal_model.label_counterfactual_data = MagicMock(
            return_value=[
                {"label": "A"},
                {"label": "B"},
            ]
        )

        # Two keys with different results
        raw_results = {
            ("key1",): {"string": ["A", "B"]},  # 100% correct
            ("key2",): {"string": ["X", "Y"]},  # 0% correct
        }

        result = causal_score_intervention_outputs(
            raw_results=raw_results,
            dataset=mock_cf_dataset,
            causal_model=mock_causal_model,
            target_variable_groups=[("answer",)],
            metric=lambda x, y: x.get("string") == y,
        )

        # Check per-key scores
        assert result["results_by_key"][("key1",)]["avg_score"] == 1.0
        assert result["results_by_key"][("key2",)]["avg_score"] == 0.0
        # Overall average: (1.0 + 0.0) / 2 = 0.5
        assert result["avg_score"] == 0.5


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
