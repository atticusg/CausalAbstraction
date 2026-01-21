# tests/test_pyvene_core/test_train_intervention.py

import pytest
import torch
from unittest.mock import MagicMock, patch

from causalab.neural.pyvene_core.interchange import train_interventions
from causalab.neural.model_units import InterchangeTarget


class MockTqdm:
    """Mock tqdm that wraps an iterable and provides set_postfix method."""

    def __init__(self, iterable, **kwargs):
        self.iterable = iterable
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        return len(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


class TestTrainInterventions:
    """Tests for train_interventions function."""

    @pytest.fixture
    def interchange_target(self):
        """Create an InterchangeTarget for testing."""
        model_unit1 = MagicMock()
        model_unit1.id = "ResidualStream(Layer:0,Token:last_token)"
        model_unit1.is_static.return_value = True
        model_unit1.create_intervention_config.return_value = MagicMock()
        model_unit1.set_feature_indices = MagicMock()
        model_unit1.get_feature_indices.return_value = [0, 1, 2]

        model_unit2 = MagicMock()
        model_unit2.id = "ResidualStream(Layer:2,Token:last_token)"
        model_unit2.is_static.return_value = True
        model_unit2.create_intervention_config.return_value = MagicMock()
        model_unit2.set_feature_indices = MagicMock()
        model_unit2.get_feature_indices.return_value = [0, 1, 2]

        target = InterchangeTarget([[model_unit1], [model_unit2]])
        return target

    @pytest.fixture
    def mock_counterfactual_dataset(self):
        """Create a mock counterfactual dataset."""
        return [
            {
                "input": "input_0",
                "counterfactual_inputs": ["cf_0_1", "cf_0_2"],
                "label": "label_0",
            },
            {
                "input": "input_1",
                "counterfactual_inputs": ["cf_1_1"],
                "label": "label_1",
            },
        ]

    @pytest.fixture
    def mock_intervenable_model(self):
        """Create a mock intervenable model."""
        model = MagicMock()
        model.disable_model_gradients = MagicMock()
        model.eval = MagicMock()
        model.count_parameters = MagicMock(return_value=100)
        model.set_zero_grad = MagicMock()

        # Create mock intervention with parameters
        mock_intervention = MagicMock()
        mock_param = torch.nn.Parameter(torch.zeros(10))
        mock_intervention.parameters.return_value = [mock_param]
        mock_intervention.get_sparsity_loss.return_value = torch.tensor(0.1)
        mock_intervention.set_temperature = MagicMock()
        mock_intervention.mask = torch.nn.Parameter(torch.zeros(10))

        model.interventions = {"test_intervention": mock_intervention}
        return model

    @pytest.fixture
    def mock_loss_metric_fn(self):
        """Create a mock loss and metric function."""
        mock_fn = MagicMock()
        mock_fn.return_value = (
            torch.tensor(0.5, requires_grad=True),
            {"accuracy": 0.75},
            {"preds": ["pred1"], "labels": ["label1"]},
        )
        return mock_fn

    @pytest.fixture
    def mock_config(self):
        """Create a config dictionary for testing."""
        return {
            "train_batch_size": 2,
            "training_epoch": 2,
            "init_lr": 1e-3,
            "memory_cleanup_freq": 1,
            "patience": None,
            "scheduler_type": "constant",
            "masking": {
                "regularization_coefficient": 0.1,
                "temperature_schedule": (1.0, 0.01),
                "temperature_annealing_fraction": 0.5,
            },
            "featurizer_kwargs": {"tie_masks": False},
        }

    def test_interchange_intervention_training(
        self,
        mock_tiny_lm,
        interchange_target,
        mock_counterfactual_dataset,
        mock_intervenable_model,
        mock_loss_metric_fn,
        mock_config,
    ):
        """Test training with interchange intervention type."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.001]
        mock_scheduler._step_count = 0

        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ) as mock_prepare,
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
            patch("causalab.neural.pyvene_core.interchange.tqdm", MockTqdm),
            patch("torch.optim.AdamW", return_value=MagicMock()),
            patch(
                "causalab.neural.pyvene_core.interchange.transformers.get_scheduler",
                return_value=mock_scheduler,
            ),
        ):
            result = train_interventions(
                pipeline=mock_tiny_lm,
                interchange_target=interchange_target,
                counterfactual_dataset=mock_counterfactual_dataset,
                intervention_type="interchange",
                config=mock_config,
                loss_and_metric_fn=mock_loss_metric_fn,
            )

            # Verify prepare_intervenable_model was called correctly
            mock_prepare.assert_called_once_with(
                mock_tiny_lm, interchange_target, intervention_type="interchange"
            )

            # Verify result is a summary string
            assert isinstance(result, str)
            assert "Trained intervention" in result

            # Verify model was put in eval mode
            mock_intervenable_model.eval.assert_called()
            mock_intervenable_model.disable_model_gradients.assert_called()

    def test_mask_intervention_training(
        self,
        mock_tiny_lm,
        interchange_target,
        mock_counterfactual_dataset,
        mock_intervenable_model,
        mock_loss_metric_fn,
        mock_config,
    ):
        """Test training with mask intervention type."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.001]
        mock_scheduler._step_count = 0

        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ) as mock_prepare,
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
            patch("causalab.neural.pyvene_core.interchange.tqdm", MockTqdm),
            patch("torch.optim.AdamW", return_value=MagicMock()),
            patch(
                "causalab.neural.pyvene_core.interchange.transformers.get_scheduler",
                return_value=mock_scheduler,
            ),
        ):
            result = train_interventions(
                pipeline=mock_tiny_lm,
                interchange_target=interchange_target,
                counterfactual_dataset=mock_counterfactual_dataset,
                intervention_type="mask",
                config=mock_config,
                loss_and_metric_fn=mock_loss_metric_fn,
            )

            # Verify prepare_intervenable_model was called with mask type
            mock_prepare.assert_called_once_with(
                mock_tiny_lm, interchange_target, intervention_type="mask"
            )

            # Verify result is a summary string
            assert isinstance(result, str)

    def test_early_stopping(
        self,
        mock_tiny_lm,
        interchange_target,
        mock_counterfactual_dataset,
        mock_intervenable_model,
        mock_config,
    ):
        """Test early stopping functionality."""
        # Set patience to trigger early stopping
        mock_config["patience"] = 1
        mock_config["training_epoch"] = 10  # More epochs than we'll run

        # Track loss values - start low then increase to trigger early stopping
        loss_call_count = [0]

        def increasing_loss(*args, **kwargs):
            loss_call_count[0] += 1
            # Return increasing loss to trigger early stopping
            loss_value = 0.5 + (loss_call_count[0] * 0.1)
            return (
                torch.tensor(loss_value, requires_grad=True),
                {"accuracy": 0.5},
                {},
            )

        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.001]
        mock_scheduler._step_count = 0

        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ),
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
            patch("causalab.neural.pyvene_core.interchange.tqdm", MockTqdm),
            patch("torch.optim.AdamW", return_value=MagicMock()),
            patch(
                "causalab.neural.pyvene_core.interchange.transformers.get_scheduler",
                return_value=mock_scheduler,
            ),
        ):
            train_interventions(
                pipeline=mock_tiny_lm,
                interchange_target=interchange_target,
                counterfactual_dataset=mock_counterfactual_dataset,
                intervention_type="interchange",
                config=mock_config,
                loss_and_metric_fn=increasing_loss,
            )

            # Verify early stopping occurred by checking loss was called fewer times
            # than full training (10 epochs * 1 batch = 10 calls without early stopping)
            # With patience=1 and increasing loss, should stop after 2 epochs
            assert loss_call_count[0] < 10, "Early stopping should have triggered"
            assert loss_call_count[0] == 2, "Should stop after 2 epochs with patience=1"

    def test_custom_loss_function(
        self,
        mock_tiny_lm,
        interchange_target,
        mock_counterfactual_dataset,
        mock_intervenable_model,
        mock_config,
    ):
        """Test using a custom loss function."""
        # Track calls to custom loss function
        custom_loss_called = [0]

        def custom_loss_fn(
            pipeline: MagicMock,
            model: MagicMock,
            batch: dict[str, list[str]],
            target: InterchangeTarget,
            source_pipeline: MagicMock | None = None,
            source_intervenable_model: MagicMock | None = None,
        ) -> tuple[torch.Tensor, dict[str, float], dict[str, str]]:
            custom_loss_called[0] += 1
            return (
                torch.tensor(0.3, requires_grad=True),
                {"custom_metric": 0.9},
                {"custom_info": "test"},
            )

        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.001]
        mock_scheduler._step_count = 0

        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ),
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
            patch("causalab.neural.pyvene_core.interchange.tqdm", MockTqdm),
            patch("torch.optim.AdamW", return_value=MagicMock()),
            patch(
                "causalab.neural.pyvene_core.interchange.transformers.get_scheduler",
                return_value=mock_scheduler,
            ),
        ):
            train_interventions(
                pipeline=mock_tiny_lm,
                interchange_target=interchange_target,
                counterfactual_dataset=mock_counterfactual_dataset,
                intervention_type="interchange",
                config=mock_config,
                loss_and_metric_fn=custom_loss_fn,
            )

            # Verify custom loss was called (epochs * batches = 2 * 1)
            assert custom_loss_called[0] == 2

    def test_memory_cleanup(
        self,
        mock_tiny_lm,
        interchange_target,
        mock_counterfactual_dataset,
        mock_intervenable_model,
        mock_loss_metric_fn,
        mock_config,
    ):
        """Test memory cleanup during training."""
        # Set memory cleanup frequency
        mock_config["memory_cleanup_freq"] = 1

        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.001]
        mock_scheduler._step_count = 0

        with (
            patch(
                "causalab.neural.pyvene_core.interchange.prepare_intervenable_model",
                return_value=mock_intervenable_model,
            ),
            patch("causalab.neural.pyvene_core.interchange.delete_intervenable_model"),
            patch("causalab.neural.pyvene_core.interchange.tqdm", MockTqdm),
            patch("torch.optim.AdamW", return_value=MagicMock()),
            patch(
                "causalab.neural.pyvene_core.interchange.transformers.get_scheduler",
                return_value=mock_scheduler,
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
        ):
            train_interventions(
                pipeline=mock_tiny_lm,
                interchange_target=interchange_target,
                counterfactual_dataset=mock_counterfactual_dataset,
                intervention_type="interchange",
                config=mock_config,
                loss_and_metric_fn=mock_loss_metric_fn,
            )

            # Verify empty_cache was called (at least once per epoch for step 0)
            assert mock_empty_cache.call_count >= 2


# Run tests when file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
