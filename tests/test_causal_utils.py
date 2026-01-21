"""
Tests for causal_utils.py functions, particularly compute_interchange_scores.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock
from causalab.causal.causal_utils import compute_interchange_scores


class MockDataset:
    """Mock dataset for testing (simulates list-like dataset interface)."""

    def __init__(self, data, dataset_id="test_dataset"):
        self.data = data
        self.id = dataset_id

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def mock_causal_model():
    """Create a mock causal model."""
    model = Mock()

    def mock_label(dataset, target_vars):
        """Mock label_counterfactual_data - adds 'label' field to each example."""
        labeled_data = []
        for i, sample in enumerate(dataset):
            labeled_sample = sample.copy()
            # Add the 'label' field that compute_interchange_scores expects
            labeled_sample["label"] = f"expected_{i}"
            labeled_data.append(labeled_sample)

        return MockDataset(labeled_data, dataset_id=dataset.id)

    model.label_counterfactual_data = mock_label
    return model


@pytest.fixture
def mock_raw_results():
    """Create mock raw results from perform_interventions."""
    return {
        "experiment_id": "test_exp",
        "method_name": "PatchResidualStream",
        "model_name": "TestModel",
        "dataset": {
            "test_dataset": {
                "model_unit": {
                    "[[unit_1]]": {
                        "raw_outputs": [
                            {
                                "string": "output_0",
                                "sequences": torch.tensor([[1, 2, 3]]),
                            },
                            {
                                "string": "expected_1",
                                "sequences": torch.tensor([[4, 5, 6]]),
                            },
                            {
                                "string": "expected_2",
                                "sequences": torch.tensor([[7, 8, 9]]),
                            },
                        ],
                        "causal_model_inputs": [
                            {
                                "base_input": {"id": 0, "raw_input": "input_0"},
                                "counterfactual_inputs": [
                                    {"id": 0, "raw_input": "cf_0"}
                                ],
                            },
                            {
                                "base_input": {"id": 1, "raw_input": "input_1"},
                                "counterfactual_inputs": [
                                    {"id": 1, "raw_input": "cf_1"}
                                ],
                            },
                            {
                                "base_input": {"id": 2, "raw_input": "input_2"},
                                "counterfactual_inputs": [
                                    {"id": 2, "raw_input": "cf_2"}
                                ],
                            },
                        ],
                        "metadata": {"layer": 5, "position": "last"},
                        "feature_indices": None,
                    }
                }
            }
        },
    }


@pytest.fixture
def mock_dataset():
    """Create a mock dataset (list-like interface)."""
    data = [
        {"id": 0, "raw_input": "input_0", "counterfactual_inputs": [{"id": 0}]},
        {"id": 1, "raw_input": "input_1", "counterfactual_inputs": [{"id": 1}]},
        {"id": 2, "raw_input": "input_2", "counterfactual_inputs": [{"id": 2}]},
    ]
    return MockDataset(data, dataset_id="test_dataset")


@pytest.fixture
def exact_match_checker():
    """Create a simple exact match checker."""

    def checker(output, expected):
        return 1.0 if output["string"] == expected else 0.0

    return checker


class TestComputeInterchangeScores:
    """Tests for compute_interchange_scores function."""

    def test_basic_score_computation(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        """Test basic score computation adds expected fields."""
        target_variables_list = [["output"]]

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            exact_match_checker,
        )

        # Check structure is preserved
        assert "dataset" in results
        assert "test_dataset" in results["dataset"]
        assert "model_unit" in results["dataset"]["test_dataset"]

        # Check scores were added
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "output" in unit_data

        # Check score structure
        score_data = unit_data["output"]
        assert "scores" in score_data
        assert "average_score" in score_data
        assert isinstance(score_data["scores"], list)
        assert isinstance(score_data["average_score"], (float, np.floating))

    def test_score_accuracy(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        """Test that scores are computed correctly."""
        target_variables_list = [["output"]]

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            exact_match_checker,
        )

        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        scores = unit_data["output"]["scores"]

        # Based on our mock data:
        # output_0 != expected_0 -> 0.0
        # output_1 == expected_1 -> 1.0  (string matches "expected_1")
        # output_2 == expected_2 -> 1.0  (string matches "expected_2")
        assert len(scores) == 3
        assert scores[0] == 0.0
        assert scores[1] == 1.0
        assert scores[2] == 1.0

        # Average should be 2/3
        avg = unit_data["output"]["average_score"]
        assert abs(avg - 2 / 3) < 0.001

    def test_multiple_target_variables(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        """Test computing scores for multiple target variable groups."""
        target_variables_list = [["output"], ["answer"]]

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            exact_match_checker,
        )

        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]

        # Both target variable groups should have scores
        assert "output" in unit_data
        assert "answer" in unit_data
        assert "average_score" in unit_data["output"]
        assert "average_score" in unit_data["answer"]

    def test_single_dataset_conversion(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        """Test that dict format works (single dataset conversion tested in integration)."""
        target_variables_list = [["output"]]

        # Pass dataset as dict (the standard usage)
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},  # Dict format
            target_variables_list,
            exact_match_checker,
        )

        # Should work with dict format
        assert "dataset" in results
        assert "test_dataset" in results["dataset"]
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "output" in unit_data

    def test_raw_results_preserved(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        """Test that original raw_outputs and causal_model_inputs are preserved."""
        target_variables_list = [["output"]]

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            exact_match_checker,
        )

        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]

        # Raw data should still be there
        assert "raw_outputs" in unit_data
        assert "causal_model_inputs" in unit_data
        assert "metadata" in unit_data

        # Check they're unchanged
        assert len(unit_data["raw_outputs"]) == 3
        assert len(unit_data["causal_model_inputs"]) == 3
        assert unit_data["metadata"]["layer"] == 5

    def test_does_not_modify_input(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        """Test that compute_interchange_scores doesn't modify the input."""
        target_variables_list = [["output"]]

        # Store original keys
        original_unit_keys = set(
            mock_raw_results["dataset"]["test_dataset"]["model_unit"][
                "[[unit_1]]"
            ].keys()
        )

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            exact_match_checker,
        )

        # Original should be unchanged
        current_unit_keys = set(
            mock_raw_results["dataset"]["test_dataset"]["model_unit"][
                "[[unit_1]]"
            ].keys()
        )
        assert original_unit_keys == current_unit_keys

        # New results should have the added fields
        result_unit_keys = set(
            results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"].keys()
        )
        assert "output" in result_unit_keys
        assert "output" not in original_unit_keys

    def test_tensor_score_conversion(
        self, mock_raw_results, mock_causal_model, mock_dataset
    ):
        """Test that tensor scores are converted to floats."""
        target_variables_list = [["output"]]

        # Checker that returns tensors
        def tensor_checker(output, expected):
            return torch.tensor(1.0 if output["string"] == expected else 0.0)

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            tensor_checker,
        )

        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        scores = unit_data["output"]["scores"]

        # All scores should be Python floats, not tensors
        for score in scores:
            assert isinstance(score, float)
            assert not isinstance(score, torch.Tensor)

    def test_average_score_calculation(
        self, mock_raw_results, mock_causal_model, mock_dataset
    ):
        """Test that average_score is calculated correctly."""
        target_variables_list = [["output"]]

        # Checker with known values
        test_scores = [0.25, 0.5, 0.75]
        score_idx = [0]

        def score_checker(output, expected):
            score = test_scores[score_idx[0]]
            score_idx[0] += 1
            return score

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            score_checker,
        )

        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]

        # Check individual scores
        assert unit_data["output"]["scores"] == test_scores

        # Check average
        expected_avg = np.mean(test_scores)
        assert abs(unit_data["output"]["average_score"] - expected_avg) < 0.001
