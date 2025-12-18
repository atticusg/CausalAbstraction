# tests/test_pyvene_core/test_run_interchange_intervention.py

import pytest
import torch
from unittest.mock import MagicMock, patch, ANY  # Added ANY import

from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.neural.model_units import InterchangeTarget


class TestRunInterchangeInterventions:
    """Tests for the _run_interchange_interventions function."""

    @pytest.fixture
    def mock_counterfactual_dataset(self):
        """Create a mock counterfactual dataset."""
        # Create mock dataset with required features
        mock_dataset = MagicMock()
        mock_dataset.dataset = MagicMock()
        mock_dataset.dataset.__getitem__.side_effect = lambda i: {
            "input": f"input_{i}",
            "counterfactual_inputs": [f"cf_{i}_1", f"cf_{i}_2"],
        }
        mock_dataset.dataset.__len__.return_value = 10
        return mock_dataset

    @pytest.fixture
    def mock_intervenable_model(self):
        """Create a mock intervenable model."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            MagicMock(sequences=torch.tensor([[1, 2, 3], [4, 5, 6]])),
        ]
        return mock_model

    def test_basic_intervention_run(
        self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset
    ):
        """Test basic functionality for running interventions."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Create mock batches to be returned by dataloader
        mock_batches = [
            {
                "input": ["input1", "input2"],
                "counterfactual_inputs": [["cf1"], ["cf2"]],
            },
            {"input": ["input3"], "counterfactual_inputs": [["cf3"]]},
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
            patch(
                "causalab.neural.pyvene_core.interchange.DataLoader",
                return_value=mock_batches,
            ),
        ):
            # Set up mock return values for batched_interchange_intervention
            # Return different dicts (with sequences) for each batch to ensure results are properly collected
            mock_batched.side_effect = [
                {"sequences": torch.tensor([[1, 2, 3], [4, 5, 6]])},  # First batch
                {"sequences": torch.tensor([[7, 8, 9]])},  # Second batch
            ]

            # Call the function
            results = run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=mock_counterfactual_dataset,
                interchange_target=interchange_target,
                batch_size=6,
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

    def test_with_output_scores(
        self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset
    ):
        """Test when output_scores is an int (top-k format), verifying scores are properly returned."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Create mock batches to be returned by dataloader
        mock_batches = [
            {
                "input": ["input1", "input2"],
                "counterfactual_inputs": [["cf1"], ["cf2"]],
            },
            {"input": ["input3"], "counterfactual_inputs": [["cf3"]]},
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
            patch(
                "causalab.neural.pyvene_core.interchange.DataLoader",
                return_value=mock_batches,
            ),
            # Also mock _convert_to_top_k to avoid actual conversion
            patch(
                "causalab.neural.pyvene_core.interchange._convert_to_top_k"
            ) as mock_convert,
        ):
            # For scores, we expect dict outputs with sequences and scores
            mock_output_batches = [
                # First batch
                {
                    "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "scores": [
                        torch.tensor(
                            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
                        ),
                    ],
                },
                # Second batch
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

            # Call the function with output_scores=10 (top-k format)
            results = run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=mock_counterfactual_dataset,
                interchange_target=interchange_target,
                batch_size=6,
                output_scores=10,  # Use int for top-k format
            )

            # Verify that batched_interchange_intervention was called with output_scores=10
            mock_batched.assert_called_with(
                mock_tiny_lm, mock_model, ANY, interchange_target, output_scores=10
            )

            # Verify that _convert_to_top_k was called
            mock_convert.assert_called_once()

            # Verify results - dict with sequences and scores
            assert "sequences" in results
            assert "scores" in results

    def test_with_tqdm_progress(
        self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset
    ):
        """Test that tqdm progress is controlled by logging level."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Create mock batches to be returned by dataloader
        mock_batches = [
            {"input": ["input1"], "counterfactual_inputs": [["cf1"]]},
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
            patch(
                "causalab.neural.pyvene_core.interchange.DataLoader",
                return_value=mock_batches,
            ),
            patch("causalab.neural.pyvene_core.interchange.tqdm") as mock_tqdm,
        ):
            # Make tqdm return an iterable (the mock_batches)
            mock_tqdm.return_value = mock_batches

            # Call the function (tqdm is now controlled by logging level)
            run_interchange_interventions(
                pipeline=mock_tiny_lm,
                counterfactual_dataset=mock_counterfactual_dataset,
                interchange_target=interchange_target,
                batch_size=6,
                output_scores=False,
            )

            # Verify that tqdm was used to wrap the dataloader
            mock_tqdm.assert_called_once()

    def test_with_small_batch_size(
        self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset
    ):
        """Test behavior with a small batch size, requiring more processing batches."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Create mock batches - 5 batches to simulate small batch size processing
        mock_batches = [
            {"input": ["input1"], "counterfactual_inputs": [["cf1"]]},
            {"input": ["input2"], "counterfactual_inputs": [["cf2"]]},
            {"input": ["input3"], "counterfactual_inputs": [["cf3"]]},
            {"input": ["input4"], "counterfactual_inputs": [["cf4"]]},
            {"input": ["input5"], "counterfactual_inputs": [["cf5"]]},
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
            patch(
                "causalab.neural.pyvene_core.interchange.DataLoader",
                return_value=mock_batches,
            ),
        ):
            # Set up return values for each batch
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
                counterfactual_dataset=mock_counterfactual_dataset,
                interchange_target=interchange_target,
                batch_size=2,  # Small batch size
                output_scores=False,
            )

            # Verify that batched_interchange_intervention was called multiple times
            assert mock_batched.call_count == 5  # 5 batches

            # Verify results - dict with sequences list
            assert "sequences" in results
            assert len(results["sequences"]) == 5

    def test_error_handling(
        self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset
    ):
        """Test handling of errors during intervention."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Create mock batches to be returned by dataloader
        mock_batches = [
            {"input": ["input1"], "counterfactual_inputs": [["cf1"]]},
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
            patch(
                "causalab.neural.pyvene_core.interchange.DataLoader",
                return_value=mock_batches,
            ),
        ):
            # Call the function - should propagate the error
            with pytest.raises(RuntimeError) as exc_info:
                run_interchange_interventions(
                    pipeline=mock_tiny_lm,
                    counterfactual_dataset=mock_counterfactual_dataset,
                    interchange_target=interchange_target,
                    batch_size=6,
                    output_scores=False,
                )

            # Verify that the error message is as expected
            assert "Test error" in str(exc_info.value)

    def test_empty_dataset(self, mock_tiny_lm, model_units_list):
        """Test behavior with an empty dataset."""
        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Create an empty dataset
        empty_mock = MagicMock()
        empty_mock.dataset = MagicMock()
        empty_mock.dataset.__len__.return_value = 0

        # Empty dataloader - no batches
        mock_batches = []

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
            patch(
                "causalab.neural.pyvene_core.interchange.DataLoader",
                return_value=mock_batches,
            ),
        ):
            # Call the function with an empty dataset - should raise IndexError due to empty all_outputs
            # The function tries to access all_outputs[0].keys() when restructuring
            with pytest.raises(IndexError):
                run_interchange_interventions(
                    pipeline=mock_tiny_lm,
                    counterfactual_dataset=empty_mock,
                    interchange_target=interchange_target,
                    batch_size=6,
                    output_scores=False,
                )

            # Verify that prepare_intervenable_model was still called
            mock_prepare.assert_called_once()

            # Verify that batched_interchange_intervention was not called
            mock_batched.assert_not_called()
