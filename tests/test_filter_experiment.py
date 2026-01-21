"""
Test suite for experiments/filter.py

Tests the filter_dataset() function which filters list[CounterfactualExample]
based on agreement between neural pipeline and causal model outputs.
"""

import pytest
from typing import Any
from unittest import mock
import torch

from causalab.experiments.filter import filter_dataset


# ---------------------- Fixtures ---------------------- #


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline with compute_outputs method."""
    pipeline = mock.MagicMock()
    return pipeline


@pytest.fixture
def mock_causal_model():
    """Create a mock causal model with new_trace method."""
    causal_model = mock.MagicMock()
    return causal_model


@pytest.fixture
def simple_dataset():
    """Create a simple list of mock CounterfactualExample dicts for testing."""
    return [
        {
            "input": {"text": "input_0", "raw_output": "expected_0"},
            "counterfactual_inputs": [
                {"text": "cf_0_0", "raw_output": "cf_expected_0_0"},
                {"text": "cf_0_1", "raw_output": "cf_expected_0_1"},
            ],
        },
        {
            "input": {"text": "input_1", "raw_output": "expected_1"},
            "counterfactual_inputs": [
                {"text": "cf_1_0", "raw_output": "cf_expected_1_0"},
                {"text": "cf_1_1", "raw_output": "cf_expected_1_1"},
            ],
        },
        {
            "input": {"text": "input_2", "raw_output": "expected_2"},
            "counterfactual_inputs": [
                {"text": "cf_2_0", "raw_output": "cf_expected_2_0"},
                {"text": "cf_2_1", "raw_output": "cf_expected_2_1"},
            ],
        },
    ]


@pytest.fixture
def metric_all_pass():
    """Metric that always returns True."""
    return lambda pred, expected: True


@pytest.fixture
def metric_all_fail():
    """Metric that always returns False."""
    return lambda pred, expected: False


def make_batch_output(strings):
    """Helper to create a batch output dict."""
    return {
        "sequences": torch.tensor([[i] for i in range(len(strings))]),
        "string": strings,
    }


# ---------------------- Tests ---------------------- #


class TestFilterDataset:
    """Tests for the filter_dataset function."""

    def test_filter_all_pass(
        self, mock_pipeline, mock_causal_model, simple_dataset, metric_all_pass
    ):
        """All examples pass metric check - all should be kept."""
        # Setup pipeline to return outputs for 3 base + 6 counterfactuals
        # compute_outputs returns flattened per-example outputs
        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
                {"string": "out_1", "sequences": torch.tensor([[1]])},
                {"string": "out_2", "sequences": torch.tensor([[2]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_out_0", "sequences": torch.tensor([[0]])},
                {"string": "cf_out_1", "sequences": torch.tensor([[1]])},
                {"string": "cf_out_2", "sequences": torch.tensor([[2]])},
                {"string": "cf_out_3", "sequences": torch.tensor([[3]])},
                {"string": "cf_out_4", "sequences": torch.tensor([[4]])},
                {"string": "cf_out_5", "sequences": torch.tensor([[5]])},
            ],
        }

        result = filter_dataset(
            simple_dataset,
            mock_pipeline,
            mock_causal_model,
            metric_all_pass,
        )

        assert len(result) == 3

    def test_filter_all_fail(
        self, mock_pipeline, mock_causal_model, simple_dataset, metric_all_fail
    ):
        """No examples pass metric check - empty list returned."""
        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
                {"string": "out_1", "sequences": torch.tensor([[1]])},
                {"string": "out_2", "sequences": torch.tensor([[2]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_0", "sequences": torch.tensor([[0]])},
                {"string": "cf_1", "sequences": torch.tensor([[1]])},
                {"string": "cf_2", "sequences": torch.tensor([[2]])},
                {"string": "cf_3", "sequences": torch.tensor([[3]])},
                {"string": "cf_4", "sequences": torch.tensor([[4]])},
                {"string": "cf_5", "sequences": torch.tensor([[5]])},
            ],
        }

        result = filter_dataset(
            simple_dataset,
            mock_pipeline,
            mock_causal_model,
            metric_all_fail,
        )

        # Now returns empty list instead of raising IndexError
        assert len(result) == 0

    def test_filter_mixed_results(
        self, mock_pipeline, mock_causal_model, simple_dataset
    ):
        """Some examples pass, some fail - only passing ones kept."""
        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
                {"string": "out_1", "sequences": torch.tensor([[1]])},
                {"string": "out_2", "sequences": torch.tensor([[2]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_0", "sequences": torch.tensor([[0]])},
                {"string": "cf_1", "sequences": torch.tensor([[1]])},
                {"string": "cf_2", "sequences": torch.tensor([[2]])},
                {"string": "cf_3", "sequences": torch.tensor([[3]])},
                {"string": "cf_4", "sequences": torch.tensor([[4]])},
                {"string": "cf_5", "sequences": torch.tensor([[5]])},
            ],
        }

        # Return different expected values for each call
        mock_causal_model.new_trace.return_value = {
            "raw_output": "expected",
            "raw_input": "raw",
        }

        # Metric passes only for first example (index 0)
        call_count = [0]

        def selective_metric(pred, expected):
            idx = call_count[0]
            call_count[0] += 1
            # Pass for indices 0, 1, 2 (first example's base + 2 counterfactuals)
            return idx < 3

        result = filter_dataset(
            simple_dataset,
            mock_pipeline,
            mock_causal_model,
            selective_metric,
        )

        assert len(result) == 1

    def test_filter_base_fails_skips_counterfactuals(
        self, mock_pipeline, mock_causal_model, simple_dataset
    ):
        """If base input fails, counterfactuals should not be checked."""
        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
                {"string": "out_1", "sequences": torch.tensor([[1]])},
                {"string": "out_2", "sequences": torch.tensor([[2]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_0", "sequences": torch.tensor([[0]])},
                {"string": "cf_1", "sequences": torch.tensor([[1]])},
                {"string": "cf_2", "sequences": torch.tensor([[2]])},
                {"string": "cf_3", "sequences": torch.tensor([[3]])},
                {"string": "cf_4", "sequences": torch.tensor([[4]])},
                {"string": "cf_5", "sequences": torch.tensor([[5]])},
            ],
        }

        metric_calls = []

        def tracking_metric(pred, expected):
            metric_calls.append(pred)
            # Fail all base inputs
            return "cf_" in pred.get("string", "")

        result = filter_dataset(
            simple_dataset,
            mock_pipeline,
            mock_causal_model,
            tracking_metric,
        )

        # All bases fail -> empty list returned
        assert len(result) == 0
        # Should only have 3 calls (one per base input), no CF calls since bases fail
        assert len(metric_calls) == 3

    def test_filter_counterfactual_fails(
        self, mock_pipeline, mock_causal_model, simple_dataset
    ):
        """Base passes but counterfactual fails - example excluded."""
        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
                {"string": "out_1", "sequences": torch.tensor([[1]])},
                {"string": "out_2", "sequences": torch.tensor([[2]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_0", "sequences": torch.tensor([[0]])},
                {"string": "cf_1", "sequences": torch.tensor([[1]])},
                {"string": "cf_2", "sequences": torch.tensor([[2]])},
                {"string": "cf_3", "sequences": torch.tensor([[3]])},
                {"string": "cf_4", "sequences": torch.tensor([[4]])},
                {"string": "cf_5", "sequences": torch.tensor([[5]])},
            ],
        }

        call_count = [0]

        def base_pass_cf_fail(pred, expected):
            idx = call_count[0]
            call_count[0] += 1
            # Pass base (idx 0), fail first CF (idx 1)
            return idx == 0

        result = filter_dataset(
            simple_dataset,
            mock_pipeline,
            mock_causal_model,
            base_pass_cf_fail,
        )

        # All examples excluded -> empty list
        assert len(result) == 0

    def test_filter_empty_dataset(
        self, mock_pipeline, mock_causal_model, metric_all_pass
    ):
        """Empty input dataset - returns empty list."""
        empty_dataset = []

        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [],
            "counterfactual_outputs": [],
        }

        result = filter_dataset(
            empty_dataset,
            mock_pipeline,
            mock_causal_model,
            metric_all_pass,
        )

        assert len(result) == 0

    def test_filter_validate_counterfactuals_false(
        self, mock_pipeline, mock_causal_model, simple_dataset
    ):
        """validate_counterfactuals=False skips CF validation."""
        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
                {"string": "out_1", "sequences": torch.tensor([[1]])},
                {"string": "out_2", "sequences": torch.tensor([[2]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_0", "sequences": torch.tensor([[0]])},
                {"string": "cf_1", "sequences": torch.tensor([[1]])},
                {"string": "cf_2", "sequences": torch.tensor([[2]])},
                {"string": "cf_3", "sequences": torch.tensor([[3]])},
                {"string": "cf_4", "sequences": torch.tensor([[4]])},
                {"string": "cf_5", "sequences": torch.tensor([[5]])},
            ],
        }

        mock_causal_model.new_trace.return_value = {
            "raw_output": "expected",
            "raw_input": "raw",
        }

        metric_calls = []

        def tracking_metric(pred, expected):
            metric_calls.append(pred)
            return True

        result = filter_dataset(
            simple_dataset,
            mock_pipeline,
            mock_causal_model,
            tracking_metric,
            validate_counterfactuals=False,
        )

        # Should only have 3 calls (base inputs only)
        assert len(metric_calls) == 3
        assert len(result) == 3

    def test_filter_preserves_example_data(
        self, mock_pipeline, mock_causal_model, metric_all_pass
    ):
        """Filtered dataset preserves the original example data."""
        # Mock data for testing
        dataset: Any = [
            {
                "input": {"text": "input_0", "raw_output": "expected_0"},
                "counterfactual_inputs": [
                    {"text": "cf_0", "raw_output": "cf_expected_0"}
                ],
            },
        ]

        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_out_0", "sequences": torch.tensor([[0]])},
            ],
        }

        result = filter_dataset(
            dataset,
            mock_pipeline,
            mock_causal_model,
            metric_all_pass,
        )

        assert len(result) == 1
        assert result[0]["input"]["text"] == "input_0"
