"""
Test suite for experiments/filter.py

Tests the filter_dataset() function which filters CounterfactualDatasets
based on agreement between neural pipeline and causal model outputs.
"""

import pytest
from unittest import mock
import torch

from causalab.experiments.filter import filter_dataset
from causalab.causal.counterfactual_dataset import CounterfactualDataset


# ---------------------- Fixtures ---------------------- #


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline with compute_outputs method."""
    pipeline = mock.MagicMock()
    return pipeline


@pytest.fixture
def mock_causal_model():
    """Create a mock causal model with run_forward method."""
    causal_model = mock.MagicMock()
    return causal_model


@pytest.fixture
def simple_dataset():
    """Create a simple CounterfactualDataset with 3 examples."""
    return CounterfactualDataset.from_dict(
        {
            "input": [
                {"text": "input_0"},
                {"text": "input_1"},
                {"text": "input_2"},
            ],
            "counterfactual_inputs": [
                [{"text": "cf_0_0"}, {"text": "cf_0_1"}],
                [{"text": "cf_1_0"}, {"text": "cf_1_1"}],
                [{"text": "cf_2_0"}, {"text": "cf_2_1"}],
            ],
        },
        id="test_dataset",
    )


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

        # Setup causal model to return expected outputs
        mock_causal_model.run_forward.return_value = {
            "raw_output": "expected",
            "raw_input": "raw",
        }

        result = filter_dataset(
            simple_dataset,
            mock_pipeline,
            mock_causal_model,
            metric_all_pass,
        )

        assert len(result) == 3
        assert result.id == "test_dataset"

    def test_filter_all_fail(
        self, mock_pipeline, mock_causal_model, simple_dataset, metric_all_fail
    ):
        """No examples pass metric check - raises IndexError due to empty dataset creation."""
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

        mock_causal_model.run_forward.return_value = {
            "raw_output": "expected",
            "raw_input": "raw",
        }

        # CounterfactualDataset doesn't support empty datasets (tries to access [0])
        with pytest.raises(IndexError):
            filter_dataset(
                simple_dataset,
                mock_pipeline,
                mock_causal_model,
                metric_all_fail,
            )

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
        mock_causal_model.run_forward.return_value = {
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

        mock_causal_model.run_forward.return_value = {
            "raw_output": "expected",
            "raw_input": "raw",
        }

        metric_calls = []

        def tracking_metric(pred, expected):
            metric_calls.append(pred)
            # Fail all base inputs
            return "cf_" in pred.get("string", "")

        # All bases fail -> empty dataset -> IndexError from CounterfactualDataset
        with pytest.raises(IndexError):
            filter_dataset(
                simple_dataset,
                mock_pipeline,
                mock_causal_model,
                tracking_metric,
            )

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

        mock_causal_model.run_forward.return_value = {
            "raw_output": "expected",
            "raw_input": "raw",
        }

        call_count = [0]

        def base_pass_cf_fail(pred, expected):
            idx = call_count[0]
            call_count[0] += 1
            # Pass base (idx 0), fail first CF (idx 1)
            return idx == 0

        # All examples excluded -> empty dataset -> IndexError
        with pytest.raises(IndexError):
            filter_dataset(
                simple_dataset,
                mock_pipeline,
                mock_causal_model,
                base_pass_cf_fail,
            )

    def test_filter_empty_dataset(
        self, mock_pipeline, mock_causal_model, metric_all_pass
    ):
        """Empty input dataset - CounterfactualDataset doesn't support empty datasets."""
        # CounterfactualDataset.from_dict raises IndexError for empty datasets
        # because it tries to access dataset[0] in __init__
        with pytest.raises(IndexError):
            CounterfactualDataset.from_dict(
                {"input": [], "counterfactual_inputs": []},
                id="empty",
            )

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

        mock_causal_model.run_forward.return_value = {
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

    def test_filter_preserves_dataset_id(
        self, mock_pipeline, mock_causal_model, metric_all_pass
    ):
        """Filtered dataset preserves the original dataset id."""
        dataset = CounterfactualDataset.from_dict(
            {
                "input": [{"text": "input_0"}],
                "counterfactual_inputs": [[{"text": "cf_0"}]],
            },
            id="my_custom_id",
        )

        mock_pipeline.compute_outputs.return_value = {
            "base_outputs": [
                {"string": "out_0", "sequences": torch.tensor([[0]])},
            ],
            "counterfactual_outputs": [
                {"string": "cf_out_0", "sequences": torch.tensor([[0]])},
            ],
        }

        mock_causal_model.run_forward.return_value = {
            "raw_output": "expected",
            "raw_input": "raw",
        }

        result = filter_dataset(
            dataset,
            mock_pipeline,
            mock_causal_model,
            metric_all_pass,
        )

        assert result.id == "my_custom_id"
