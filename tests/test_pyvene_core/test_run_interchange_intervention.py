# tests/test_pyvene_core/test_run_interchange_intervention.py

import pytest
import torch
from typing import Any
from unittest.mock import MagicMock, patch, ANY

from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.neural.model_units import InterchangeTarget


class TestRunInterchangeInterventions:
    """Tests for the _run_interchange_interventions function."""

    @pytest.fixture
    def mock_intervenable_model(self):
        """Create a mock intervenable model."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            MagicMock(sequences=torch.tensor([[1, 2, 3], [4, 5, 6]])),
        ]
        return mock_model

    def test_basic_intervention_run(self, mock_tiny_lm: Any, model_units_list: Any):
        """Test basic functionality for running interventions."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Test dataset (mock data)
        test_dataset: Any = [
            {"input": "input1", "counterfactual_inputs": ["cf1"]},
            {"input": "input2", "counterfactual_inputs": ["cf2"]},
            {"input": "input3", "counterfactual_inputs": ["cf3"]},
        ]

        # Mock prepare_intervenable_model
        mock_model = MagicMock()
        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_model,
            ) as mock_prepare,
            patch(
                "causalab.neural.pyvene_core.interchange.batched_interchange_intervention"
            ) as mock_batched,
            patch(
                "causalab.neural.pyvene_core.interchange.delete_intervenable_model"
            ) as mock_delete,
        ):
            # Set up mock return values for batched_interchange_intervention
            # With batch_size=2, we'll have 2 batches (2 examples, then 1 example)
            mock_batched.side_effect = [
                {
                    "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]])
                },  # First batch (2 examples)
                {"sequences": torch.tensor([[7, 8, 9]])},  # Second batch (1 example)
            ]

            # Call the function with batch_size=2
            results = run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=test_dataset,
                interchange_target=interchange_target,
                batch_size=2,
                output_scores=False,
            )

            # Verify that prepare_intervenable_model was called correctly
            mock_prepare.assert_called_once_with(
                mock_tiny_lm, interchange_target, intervention_type="interchange"
            )

            # Verify that batched_interchange_intervention was called for each batch
            assert mock_batched.call_count == 2

            # Verify that delete_intervenable_model was called to clean up
            mock_delete.assert_called_once_with(mock_model)

            # Verify results - check keys exists after restructuring
            assert "sequences" in results
            assert len(results["sequences"]) == 2  # One per batch

    def test_with_output_scores(self, mock_tiny_lm: Any, model_units_list: Any):
        """Test when output_scores is an int (top-k format), verifying scores are properly returned."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Test dataset (mock data)
        test_dataset: Any = [
            {"input": "input1", "counterfactual_inputs": ["cf1"]},
            {"input": "input2", "counterfactual_inputs": ["cf2"]},
            {"input": "input3", "counterfactual_inputs": ["cf3"]},
        ]

        # Mock prepare_intervenable_model
        mock_model = MagicMock()
        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch(
                "causalab.neural.pyvene_core.interchange.batched_interchange_intervention"
            ) as mock_batched,
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
            # Also mock _convert_to_top_k to avoid actual conversion
            patch(
                "causalab.neural.pyvene_core.interchange._convert_to_top_k"
            ) as mock_convert,
        ):
            # For scores, we expect dict outputs with sequences and scores
            mock_output_batches = [
                # First batch (2 examples)
                {
                    "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "scores": [
                        torch.tensor(
                            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
                        ),
                    ],
                },
                # Second batch (1 example)
                {
                    "sequences": torch.tensor([[7, 8, 9]]),
                    "scores": [
                        torch.tensor([[2.1, 2.2, 2.3, 2.4, 2.5]]),
                    ],
                },
            ]
            mock_batched.side_effect = mock_output_batches
            # Mock _convert_to_top_k to return the same outputs (just pass through)
            mock_convert.return_value = mock_output_batches

            # Call the function with output_scores=10 (top-k format), batch_size=2
            results = run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=test_dataset,
                interchange_target=interchange_target,
                batch_size=2,
                output_scores=10,  # Use int for top-k format
            )

            # Verify that batched_interchange_intervention was called with output_scores=10
            mock_batched.assert_called_with(
                mock_tiny_lm,
                mock_model,
                ANY,
                interchange_target,
                output_scores=10,
                source_pipeline=None,
                source_intervenable_model=None,
            )

            # Verify that _convert_to_top_k was called
            mock_convert.assert_called_once()

            # Verify results - dict with sequences and scores
            assert "sequences" in results
            assert "scores" in results

    def test_with_tqdm_progress(self, mock_tiny_lm: Any, model_units_list: Any):
        """Test that tqdm progress is controlled by logging level."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Test dataset (mock data)
        test_dataset: Any = [
            {"input": "input1", "counterfactual_inputs": ["cf1"]},
        ]

        # Mock prepare_intervenable_model
        mock_model = MagicMock()
        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch(
                "causalab.neural.pyvene_core.interchange.batched_interchange_intervention",
                return_value={"sequences": torch.tensor([[1, 2, 3]])},
            ),
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
            patch("causalab.neural.pyvene_core.interchange.tqdm") as mock_tqdm,
        ):
            # Make tqdm return an iterable over the range
            mock_tqdm.return_value = range(0, 1, 1)

            # Call the function (tqdm is now controlled by logging level)
            run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=test_dataset,
                interchange_target=interchange_target,
                batch_size=6,
                output_scores=False,
            )

            # Verify that tqdm was used to wrap the range
            mock_tqdm.assert_called_once()

    def test_with_small_batch_size(self, mock_tiny_lm: Any, model_units_list: Any):
        """Test behavior with a small batch size, requiring more processing batches."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Test dataset (mock data) - 5 examples
        test_dataset: Any = [
            {"input": "input1", "counterfactual_inputs": ["cf1"]},
            {"input": "input2", "counterfactual_inputs": ["cf2"]},
            {"input": "input3", "counterfactual_inputs": ["cf3"]},
            {"input": "input4", "counterfactual_inputs": ["cf4"]},
            {"input": "input5", "counterfactual_inputs": ["cf5"]},
        ]

        # Mock prepare_intervenable_model
        mock_model = MagicMock()
        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch(
                "causalab.neural.pyvene_core.interchange.batched_interchange_intervention"
            ) as mock_batched,
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
        ):
            # Set up return values for each batch (5 examples with batch_size=1 = 5 batches)
            mock_batched.side_effect = [
                {"sequences": torch.tensor([[1, 2, 3]])},  # Batch 1
                {"sequences": torch.tensor([[4, 5, 6]])},  # Batch 2
                {"sequences": torch.tensor([[7, 8, 9]])},  # Batch 3
                {"sequences": torch.tensor([[10, 11, 12]])},  # Batch 4
                {"sequences": torch.tensor([[13, 14, 15]])},  # Batch 5
            ]

            # Call the function with a small batch size
            results = run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=test_dataset,
                interchange_target=interchange_target,
                batch_size=1,  # Very small batch size - one example per batch
                output_scores=False,
            )

            # Verify that batched_interchange_intervention was called multiple times
            assert mock_batched.call_count == 5  # 5 batches

            # Verify results - dict with sequences list
            assert "sequences" in results
            assert len(results["sequences"]) == 5

    def test_error_handling(self, mock_tiny_lm: Any, model_units_list: Any):
        """Test handling of errors during intervention."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Test dataset (mock data)
        test_dataset: Any = [
            {"input": "input1", "counterfactual_inputs": ["cf1"]},
        ]

        # Mock prepare_intervenable_model
        mock_model = MagicMock()

        # Use simple mocking approach rather than trying to override the function
        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch(
                "causalab.neural.pyvene_core.interchange.batched_interchange_intervention",
                side_effect=RuntimeError("Test error"),
            ),
        ):
            # Call the function - should propagate the error
            with pytest.raises(RuntimeError) as exc_info:
                run_interchange_interventions(
                    pipeline=mock_tiny_lm,
                    counterfactual_dataset=test_dataset,
                    interchange_target=interchange_target,
                    batch_size=6,
                    output_scores=False,
                )

            # Verify that the error message is as expected
            assert "Test error" in str(exc_info.value)

    def test_empty_dataset(self, mock_tiny_lm: Any, model_units_list: Any):
        """Test behavior with an empty dataset."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Empty dataset
        empty_dataset = []

        # Mock prepare_intervenable_model
        mock_model = MagicMock()
        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_model,
            ) as mock_prepare,
            patch(
                "causalab.neural.pyvene_core.interchange.batched_interchange_intervention"
            ) as mock_batched,
            patch(
                "causalab.neural.pyvene_core.interchange.delete_intervenable_model"
            ) as _mock_delete,
        ):
            # Call the function with an empty dataset - should raise IndexError due to empty all_outputs
            # The function tries to access all_outputs[0].keys() when restructuring
            with pytest.raises(IndexError):
                run_interchange_interventions(
                    pipeline=mock_tiny_lm,
                    counterfactual_dataset=empty_dataset,
                    interchange_target=interchange_target,
                    batch_size=6,
                    output_scores=False,
                )

            # Verify that prepare_intervenable_model was still called
            mock_prepare.assert_called_once()

            # Verify that batched_interchange_intervention was not called
            mock_batched.assert_not_called()
