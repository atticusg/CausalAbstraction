# tests/test_experiments/test_interchange.py
"""
Tests for experiments/metric.py - causal_score_intervention_outputs function.
"""

import pytest
import torch
from typing import Any
from unittest.mock import MagicMock

from causalab.methods.metric import causal_score_intervention_outputs
from causalab.neural.pipeline import GenerationResult


pytestmark = pytest.mark.unit


def create_mock_cf_dataset(size: int = 3) -> Any:
    """Create a list of mock CounterfactualExample dicts for testing."""
    return [
        {"input": {"text": f"input_{i}"}, "counterfactual_inputs": []}
        for i in range(size)
    ]


def _result(strings: list[str]) -> GenerationResult:
    """A flat GenerationResult with the given decoded outputs (EU5b, #487)."""
    return GenerationResult(
        sequences=torch.zeros((len(strings), 1), dtype=torch.long),
        strings=strings,
    )


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

        results = {("test",): _result(["A", "B", "A"])}

        result = causal_score_intervention_outputs(
            results=results,
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
        results = {("test",): _result(["A", "A", "A", "A"])}

        result = causal_score_intervention_outputs(
            results=results,
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

        results = {("test",): _result(["A", "A"])}

        result = causal_score_intervention_outputs(
            results=results,
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

    def test_embeds_io_view_as_raw_results(self):
        """Each results_by_key entry embeds the legacy one-synthetic-batch
        ``raw_results`` dict (``to_raw_results()``) — the io boundary's
        stored-artifact schema, unchanged by EU5b (#487)."""
        mock_cf_dataset = create_mock_cf_dataset(2)

        mock_causal_model = MagicMock()
        mock_causal_model.label_counterfactual_data = MagicMock(
            return_value=[{"label": "A"}, {"label": "B"}]
        )

        generation = _result(["A", "B"])
        result = causal_score_intervention_outputs(
            results={("test",): generation},
            dataset=mock_cf_dataset,
            causal_model=mock_causal_model,
            target_variable_groups=[("answer",)],
            metric=lambda x, y: x.get("string") == y,
        )

        raw = result["results_by_key"][("test",)]["raw_results"]
        assert raw["string"] == [["A", "B"]]  # ONE synthetic batch
        assert len(raw["sequences"]) == 1
        assert torch.equal(raw["sequences"][0], generation.sequences)


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
        results = {
            ("key1",): _result(["A", "B"]),  # 100% correct
            ("key2",): _result(["X", "Y"]),  # 0% correct
        }

        result = causal_score_intervention_outputs(
            results=results,
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
