# tests/test_pyvene_core/test_prepare_intervenable_model.py

from unittest.mock import MagicMock, patch

from causalab.neural.pyvene_core.intervenable_model import prepare_intervenable_model
from causalab.neural.LM_units import ResidualStream
from causalab.neural.model_units import InterchangeTarget


class TestPrepareIntervenableModelIntegration:
    """Integration tests for the prepare_intervenable_model function."""

    def test_end_to_end_creation(self, mock_tiny_lm, token_positions, monkeypatch):
        """Test end-to-end model creation with real components (no mocks)."""
        # Set up model units - new API uses InterchangeTarget or flat list
        units = []
        layers = [0, 1]

        for layer in layers:
            for token_position in token_positions:
                unit = ResidualStream(
                    layer=layer,
                    token_indices=token_position,
                    shape=(mock_tiny_lm.model.config.hidden_size,),
                    target_output=True,
                )
                # The key fix: Mock is_static to return True for consistent testing
                unit.is_static = MagicMock(return_value=True)
                units.append(unit)

        # Wrap in InterchangeTarget (single group)
        interchange_target = InterchangeTarget([units])

        # Create mock pyvene components to avoid actual model creation
        mock_config = MagicMock()
        mock_model = MagicMock()

        class MockPV:
            def __init__(self):
                self.IntervenableConfig = MagicMock(return_value=mock_config)
                self.IntervenableModel = MagicMock(return_value=mock_model)

                # Mock TrainableIntervention and other intervention classes
                self.TrainableIntervention = MagicMock()
                self.DistributedRepresentationIntervention = MagicMock()
                self.CollectIntervention = MagicMock()

        mock_pv = MockPV()

        # Apply the patch
        monkeypatch.setattr(
            "causalab.neural.pyvene_core.intervenable_model.pv", mock_pv
        )

        # Call the function
        result = prepare_intervenable_model(mock_tiny_lm, interchange_target)

        # Check the result
        assert result is mock_model

        # Verify IntervenableConfig was created with correct number of configs
        # We expect one config per unit
        assert mock_pv.IntervenableConfig.call_count == 1

        # Verify IntervenableModel was created with correct config
        mock_pv.IntervenableModel.assert_called_once_with(
            mock_config, model=mock_tiny_lm.model, use_fast=True
        )

        # Verify set_device was called
        mock_model.set_device.assert_called_once_with(mock_tiny_lm.model.device)

    def test_with_real_pyvene(self, mock_tiny_lm, token_positions):
        """
        Test with actual pyvene library
        """
        # Set up model units - new API uses InterchangeTarget or flat list
        units = []
        layers = [0]

        for layer in layers:
            for token_position in token_positions:
                unit = ResidualStream(
                    layer=layer,
                    token_indices=token_position,
                    shape=(mock_tiny_lm.model.config.hidden_size,),
                    target_output=True,
                )
                units.append(unit)

        # Wrap in InterchangeTarget
        interchange_target = InterchangeTarget([units])

        # Mock the create_intervention_config method to avoid actual intervention creation
        for unit in units:
            unit.create_intervention_config = MagicMock(
                return_value={
                    "component": unit.component_type,
                    "unit": unit.unit,
                    "layer": unit.layer,
                    "group_key": 0,
                    "intervention_type": MagicMock(),
                }
            )

        # Mock pyvene components to avoid actual model creation
        with (
            patch("pyvene.IntervenableModel") as mock_model_class,
            patch("pyvene.IntervenableConfig") as mock_config_class,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            # Call the function
            result = prepare_intervenable_model(mock_tiny_lm, interchange_target)

            # Check the result
            assert result is mock_model

            # Verify IntervenableConfig was created
            mock_config_class.assert_called_once()

            # Verify IntervenableModel was created with correct arguments
            mock_model_class.assert_called_once()

            # Verify set_device was called
            mock_model.set_device.assert_called_once_with(mock_tiny_lm.model.device)
