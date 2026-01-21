import pytest
import torch
import logging
from unittest.mock import MagicMock, patch

from causalab.neural.pyvene_core.collect import collect_features


class TestCollectFeatures:
    """Tests for the collect_features function."""

    @pytest.fixture
    def model_units(self):
        """Create a flat list of mock model units for testing."""
        unit1 = MagicMock()
        unit1.id = "ResidualStream(Layer:0,Token:last_token)"
        unit1.index_component.return_value = [[0, 1], [0, 1]]  # batch of indices

        unit2 = MagicMock()
        unit2.id = "ResidualStream(Layer:2,Token:last_token)"
        unit2.index_component.return_value = [[0, 1], [0, 1]]  # batch of indices

        return [unit1, unit2]

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        dataset = [
            {"input": "input_1"},
            {"input": "input_2"},
            {"input": "input_3"},
        ]
        return dataset

    @pytest.fixture
    def mock_loaded_inputs(self):
        """Create mock loaded inputs from the pipeline."""
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

    def test_basic_feature_collection(
        self,
        mock_tiny_lm,
        model_units,
        mock_dataset,
        mock_loaded_inputs,
    ):
        """Test basic feature collection functionality."""
        # Mock the intervenable model
        mock_intervenable_model = MagicMock()

        def model_side_effect(*args, **kwargs):
            # Return 2 activation tensors (one per model unit)
            activations = [torch.randn(2, 32), torch.randn(2, 32)]
            return (MagicMock(), activations), None

        mock_intervenable_model.side_effect = model_side_effect

        with (
            patch(
                "causalab.neural.pyvene_core.collect.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ) as mock_prepare,
            patch("causalab.neural.pyvene_core.collect.delete_intervenable_model"),
            patch.object(mock_tiny_lm, "load", return_value=mock_loaded_inputs),
        ):
            # Call the function
            result = collect_features(
                mock_dataset,
                mock_tiny_lm,
                model_units,
                batch_size=2,
            )

            # Verify that prepare_intervenable_model was called with "collect" intervention type
            mock_prepare.assert_called_once_with(
                mock_tiny_lm, model_units, intervention_type="collect"
            )

            # Verify the result structure: dict mapping unit IDs to tensors
            assert isinstance(result, dict)
            assert len(result) == len(model_units)

            # Each result should be keyed by unit ID and contain a tensor
            for unit in model_units:
                assert unit.id in result
                assert isinstance(result[unit.id], torch.Tensor)

    def test_verbose_output(
        self,
        mock_tiny_lm,
        model_units,
        mock_dataset,
        mock_loaded_inputs,
        caplog,
    ):
        """Test verbose output functionality."""
        mock_intervenable_model = MagicMock()
        mock_intervenable_model.side_effect = lambda *args, **kwargs: (
            (MagicMock(), [torch.randn(2, 32), torch.randn(2, 32)]),
            None,
        )

        with (
            patch(
                "causalab.neural.pyvene_core.collect.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ),
            patch("causalab.neural.pyvene_core.collect.delete_intervenable_model"),
            patch.object(mock_tiny_lm, "load", return_value=mock_loaded_inputs),
            caplog.at_level(
                logging.DEBUG, logger="causalab.neural.pyvene_core.collect"
            ),
        ):
            collect_features(
                mock_dataset,
                mock_tiny_lm,
                model_units,
                batch_size=2,
            )

            # Check that diagnostic information was logged
            assert "Collected features for" in caplog.text
            assert "Feature tensor shape:" in caplog.text

    def test_memory_management(
        self,
        mock_tiny_lm,
        model_units,
        mock_dataset,
        mock_loaded_inputs,
    ):
        """Test that tensors are moved to CPU for memory efficiency."""
        # Create mock model that returns tensors on a specific device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_intervenable_model = MagicMock()

        def model_side_effect(*args, **kwargs):
            activations = [
                torch.randn(2, 32, device=device),
                torch.randn(2, 32, device=device),
            ]
            return (MagicMock(), activations), None

        mock_intervenable_model.side_effect = model_side_effect

        with (
            patch(
                "causalab.neural.pyvene_core.collect.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ),
            patch("causalab.neural.pyvene_core.collect.delete_intervenable_model"),
            patch("torch.cuda.empty_cache"),
            patch.object(mock_tiny_lm, "load", return_value=mock_loaded_inputs),
        ):
            # Call the function
            result = collect_features(
                mock_dataset,
                mock_tiny_lm,
                model_units,
                batch_size=2,
            )

            # Verify all tensors are on CPU
            for _unit_id, tensor in result.items():
                assert tensor.device.type == "cpu"

    def test_result_shape(
        self,
        mock_tiny_lm,
        model_units,
        mock_dataset,
        mock_loaded_inputs,
    ):
        """Test that result tensors have correct shape (n_samples, n_features)."""
        hidden_size = 32
        mock_intervenable_model = MagicMock()

        def model_side_effect(*args, **kwargs):
            # Return activations with shape (batch_size, hidden_size)
            activations = [
                torch.randn(2, hidden_size),
                torch.randn(2, hidden_size),
            ]
            return (MagicMock(), activations), None

        mock_intervenable_model.side_effect = model_side_effect

        with (
            patch(
                "causalab.neural.pyvene_core.collect.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ),
            patch("causalab.neural.pyvene_core.collect.delete_intervenable_model"),
            patch.object(mock_tiny_lm, "load", return_value=mock_loaded_inputs),
        ):
            result = collect_features(
                mock_dataset,
                mock_tiny_lm,
                model_units,
                batch_size=2,
            )

            # Each tensor should have shape (n_samples, hidden_size)
            for _unit_id, tensor in result.items():
                assert tensor.ndim == 2
                assert tensor.shape[1] == hidden_size


class TestCollectFeaturesPyvene18Plus:
    """Test suite for collect_features with pyvene 0.1.8+ format."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        return [
            {"input": "input_1"},
            {"input": "input_2"},
            {"input": "input_3"},
        ]

    @pytest.fixture
    def mock_loaded_inputs(self):
        """Create mock loaded inputs from the pipeline."""
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

    @pytest.fixture
    def attention_head_units(self):
        """Create attention head model units for testing."""
        unit1 = MagicMock()
        unit1.id = "AttentionHead(Layer:0,Head:0)"
        unit1.index_component.return_value = [[0], [0]]

        unit2 = MagicMock()
        unit2.id = "AttentionHead(Layer:0,Head:1)"
        unit2.index_component.return_value = [[0], [0]]

        return [unit1, unit2]

    @pytest.fixture
    def residual_stream_units(self):
        """Create residual stream model units for testing."""
        unit1 = MagicMock()
        unit1.id = "ResidualStream(Layer:0)"
        unit1.index_component.return_value = [[0], [0]]

        unit2 = MagicMock()
        unit2.id = "ResidualStream(Layer:1)"
        unit2.index_component.return_value = [[0], [0]]

        return [unit1, unit2]

    def test_attention_head_activation_processing(
        self,
        mock_tiny_lm,
        attention_head_units,
        mock_dataset,
        mock_loaded_inputs,
    ):
        """Test processing of 4D attention head activations in pyvene 0.1.8+ format."""
        # Mock pyvene 0.1.8+ format: one tensor per unit with 4D shape
        # Shape: (batch_size=2, seq_len=1, num_heads=4, head_dim=8)
        mock_activations = [
            torch.randn(2, 1, 4, 8),  # First attention head unit
            torch.randn(2, 1, 4, 8),  # Second attention head unit
        ]

        mock_model = MagicMock()
        mock_model.side_effect = lambda *args, **kwargs: (
            (MagicMock(), mock_activations),
            None,
        )

        with (
            patch(
                "causalab.neural.pyvene_core.collect.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch("causalab.neural.pyvene_core.collect.delete_intervenable_model"),
            patch.object(mock_tiny_lm, "load", return_value=mock_loaded_inputs),
        ):
            result = collect_features(
                mock_dataset, mock_tiny_lm, attention_head_units, batch_size=2
            )

            # Verify correct processing - result is dict keyed by unit ID
            assert len(result) == 2  # Two units

            # Each head should have extracted activations with shape (total_samples, head_dim)
            for unit in attention_head_units:
                assert unit.id in result
                tensor = result[unit.id]
                assert tensor.shape[1] == 8  # head_dim = 8
                assert tensor.shape[0] > 0  # Should have some samples

    def test_residual_stream_activation_processing(
        self,
        mock_tiny_lm,
        residual_stream_units,
        mock_dataset,
        mock_loaded_inputs,
    ):
        """Test processing of 3D residual stream activations in pyvene 0.1.8+ format."""
        # Mock pyvene 0.1.8+ format: one tensor per unit with 3D shape
        # Shape: (batch_size=2, seq_len=1, hidden_dim=32)
        mock_activations = [
            torch.randn(2, 1, 32),  # First residual stream unit
            torch.randn(2, 1, 32),  # Second residual stream unit
        ]

        mock_model = MagicMock()
        mock_model.side_effect = lambda *args, **kwargs: (
            (MagicMock(), mock_activations),
            None,
        )

        with (
            patch(
                "causalab.neural.pyvene_core.collect.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch("causalab.neural.pyvene_core.collect.delete_intervenable_model"),
            patch.object(mock_tiny_lm, "load", return_value=mock_loaded_inputs),
        ):
            result = collect_features(
                mock_dataset, mock_tiny_lm, residual_stream_units, batch_size=2
            )

            # Verify correct processing - result is dict keyed by unit ID
            assert len(result) == 2  # Two units

            # Each unit should have activations with shape (total_samples, hidden_dim)
            for unit in residual_stream_units:
                assert unit.id in result
                tensor = result[unit.id]
                assert tensor.shape[1] == 32  # hidden_dim = 32
                assert tensor.shape[0] > 0  # Should have some samples

    def test_mixed_activation_shapes(
        self, mock_tiny_lm, mock_dataset, mock_loaded_inputs
    ):
        """Test handling of different activation shapes in the same call."""
        # Mixed units: some 2D, some 3D
        unit1 = MagicMock()
        unit1.id = "Unit1"
        unit1.index_component.return_value = [[0], [0]]

        unit2 = MagicMock()
        unit2.id = "Unit2"
        unit2.index_component.return_value = [[0], [0]]

        mixed_units = [unit1, unit2]

        # Mock different shapes: 2D and 3D
        mock_activations = [
            torch.randn(2, 64),  # Already 2D: (batch_size, feature_dim)
            torch.randn(2, 1, 32),  # 3D: (batch_size, seq_len, hidden_dim)
        ]

        mock_model = MagicMock()
        mock_model.side_effect = lambda *args, **kwargs: (
            (MagicMock(), mock_activations),
            None,
        )

        with (
            patch(
                "causalab.neural.pyvene_core.collect.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch("causalab.neural.pyvene_core.collect.delete_intervenable_model"),
            patch.object(mock_tiny_lm, "load", return_value=mock_loaded_inputs),
        ):
            # MagicMock objects used as test doubles for AtomicModelUnit
            result = collect_features(
                mock_dataset,
                mock_tiny_lm,
                mixed_units,  # pyright: ignore[reportArgumentType]
                batch_size=2,
            )

            # Verify both shapes are handled correctly
            assert len(result) == 2

            # First unit (2D) should remain unchanged
            assert result["Unit1"].shape[1] == 64
            assert result["Unit1"].shape[0] > 0

            # Second unit (3D) should be squeezed
            assert result["Unit2"].shape[1] == 32
            assert result["Unit2"].shape[0] > 0


# Run tests when file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
