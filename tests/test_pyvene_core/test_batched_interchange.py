# tests/test_pyvene_core/test_batched_intervention.py

import pytest
import torch
from unittest.mock import MagicMock, patch

from causalab.neural.pyvene_core.interchange import batched_interchange_intervention
from causalab.neural.model_units import InterchangeTarget


class TestBatchedInterchangeIntervention:
    """Tests for the batched_interchange_intervention function."""

    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch with base and counterfactual inputs."""
        return {
            "input": ["input1", "input2"],
            "counterfactual_inputs": [["cf1_1", "cf1_2"], ["cf2_1", "cf2_2"]],
        }

    @pytest.fixture
    def mock_prepared_inputs(self):
        """Create mock prepared inputs as would be returned by prepare_intervenable_inputs."""
        # Create base inputs
        base_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        base_attention_mask = torch.ones_like(base_input_ids)
        batched_base = {
            "input_ids": base_input_ids,
            "attention_mask": base_attention_mask,
        }

        # Create counterfactual inputs
        cf1_input_ids = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.long)
        cf1_attention_mask = torch.ones_like(cf1_input_ids)
        cf2_input_ids = torch.tensor([[13, 14, 15], [16, 17, 18]], dtype=torch.long)
        cf2_attention_mask = torch.ones_like(cf2_input_ids)

        batched_counterfactuals = [
            {"input_ids": cf1_input_ids, "attention_mask": cf1_attention_mask},
            {"input_ids": cf2_input_ids, "attention_mask": cf2_attention_mask},
        ]

        # Create inv_locations and feature_indices
        inv_locations = {
            "sources->base": (
                [[[0, 1], [0, 1]], [[0, 1], [0, 1]]],  # counterfactual_indices
                [[[0, 1], [0, 1]], [[0, 1], [0, 1]]],  # base_indices
            )
        }

        feature_indices = [
            [[0, 1, 2], [0, 1, 2]],  # for first unit
            [[0, 1, 2], [0, 1, 2]],  # for second unit
        ]

        return batched_base, batched_counterfactuals, inv_locations, feature_indices

    def test_basic_intervention(
        self, mock_tiny_lm, model_units_list, mock_batch, mock_prepared_inputs
    ):
        """Test basic intervention functionality."""
        batched_base, batched_counterfactuals, inv_locations, feature_indices = (
            mock_prepared_inputs
        )

        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Mock intervenable model
        mock_intervenable_model = MagicMock()

        # Mock prepare_intervenable_inputs
        with patch(
            "causalab.neural.pyvene_core.interchange.prepare_intervenable_inputs",
            return_value=(
                batched_base,
                batched_counterfactuals,
                inv_locations,
                feature_indices,
            ),
        ):
            # Mock pipeline.intervenable_generate - returns a dict with 'sequences' key
            expected_sequences = torch.tensor([[100, 101, 102], [103, 104, 105]])
            expected_output = {"sequences": expected_sequences}
            mock_tiny_lm.intervenable_generate = MagicMock(return_value=expected_output)

            # Call the function
            output = batched_interchange_intervention(
                mock_tiny_lm, mock_intervenable_model, mock_batch, interchange_target
            )

            # Verify the output
            assert torch.equal(output["sequences"], expected_sequences)

            # Verify that intervenable_generate was called with correct arguments
            mock_tiny_lm.intervenable_generate.assert_called_once_with(
                mock_intervenable_model,
                batched_base,
                batched_counterfactuals,
                inv_locations,
                feature_indices,
                source_representations=None,
                output_scores=True,
            )

            # Verify that tensors were moved to CPU
            for batched in [batched_base] + batched_counterfactuals:
                for k, v in batched.items():
                    if isinstance(v, torch.Tensor):
                        assert v.device.type == "cpu"

    def test_with_output_scores(
        self, mock_tiny_lm, model_units_list, mock_batch, mock_prepared_inputs
    ):
        """Test intervention with output_scores=True."""
        batched_base, batched_counterfactuals, inv_locations, feature_indices = (
            mock_prepared_inputs
        )

        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # Mock intervenable model
        mock_intervenable_model = MagicMock()

        # Mock prepare_intervenable_inputs
        with patch(
            "causalab.neural.pyvene_core.interchange.prepare_intervenable_inputs",
            return_value=(
                batched_base,
                batched_counterfactuals,
                inv_locations,
                feature_indices,
            ),
        ):
            # Mock pipeline.intervenable_generate to return logits (scores)
            # In this case, we'll return a 3D tensor (batch_size, seq_len, vocab_size)
            expected_sequences = torch.randn(
                2, 3, 10
            )  # (batch_size=2, seq_len=3, vocab_size=10)
            expected_output = {"sequences": expected_sequences}
            mock_tiny_lm.intervenable_generate = MagicMock(return_value=expected_output)

            # Call the function with output_scores=True
            output = batched_interchange_intervention(
                mock_tiny_lm,
                mock_intervenable_model,
                mock_batch,
                interchange_target,
                output_scores=True,
            )

            # Verify the output
            assert torch.equal(output["sequences"], expected_sequences)

            # Verify that intervenable_generate was called with output_scores=True
            mock_tiny_lm.intervenable_generate.assert_called_once_with(
                mock_intervenable_model,
                batched_base,
                batched_counterfactuals,
                inv_locations,
                feature_indices,
                source_representations=None,
                output_scores=True,
            )

    def test_tensor_device_handling(
        self, mock_tiny_lm, model_units_list, mock_batch, mock_prepared_inputs
    ):
        """Test that tensors are properly moved to CPU."""
        batched_base, batched_counterfactuals, inv_locations, feature_indices = (
            mock_prepared_inputs
        )

        # Extract units from fixture and create InterchangeTarget
        model_units_sublist = model_units_list[0]
        all_units = []
        for units in model_units_sublist:
            for unit in units:
                all_units.append(unit)
        interchange_target = InterchangeTarget([all_units])

        # If CUDA is available, move tensors to GPU first
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move tensors to device
        for k, v in batched_base.items():
            if isinstance(v, torch.Tensor):
                batched_base[k] = v.to(device)

        for batched in batched_counterfactuals:
            for k, v in batched.items():
                if isinstance(v, torch.Tensor):
                    batched[k] = v.to(device)

        # Mock intervenable model
        mock_intervenable_model = MagicMock()

        # Mock prepare_intervenable_inputs
        with patch(
            "causalab.neural.pyvene_core.interchange.prepare_intervenable_inputs",
            return_value=(
                batched_base,
                batched_counterfactuals,
                inv_locations,
                feature_indices,
            ),
        ):
            # Mock pipeline.intervenable_generate
            expected_output = torch.tensor([[100, 101, 102], [103, 104, 105]])
            mock_tiny_lm.intervenable_generate = MagicMock(return_value=expected_output)

            # Call the function
            batched_interchange_intervention(
                mock_tiny_lm, mock_intervenable_model, mock_batch, interchange_target
            )

            # Verify that all tensors have been moved to CPU
            for k, v in batched_base.items():
                if isinstance(v, torch.Tensor):
                    assert v.device.type == "cpu"

            for batched in batched_counterfactuals:
                for k, v in batched.items():
                    if isinstance(v, torch.Tensor):
                        assert v.device.type == "cpu"
