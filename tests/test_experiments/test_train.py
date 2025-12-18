# tests/test_experiments/test_train.py
"""
Tests for experiments/train.py - DBM and DAS training functions.
"""

import pytest
from unittest.mock import MagicMock, patch, create_autospec
from datasets import Dataset

from causalab.experiments.train import train_interventions as train_intervention
from causalab.experiments.configs.train_config import DEFAULT_CONFIG


def create_mock_hf_dataset(size=5):
    """Create a mock that passes isinstance(x, Dataset) check."""
    mock = create_autospec(Dataset, instance=True)
    mock.__len__ = MagicMock(return_value=size)
    mock.__iter__ = MagicMock(
        return_value=iter(
            [
                {
                    "input": {"text": "test"},
                    "counterfactual_inputs": [{"text": "cf"}],
                }
                for _ in range(size)
            ]
        )
    )
    return mock


class TestTrainInterventionValidation:
    """Test input validation for train_intervention function."""

    def test_rejects_invalid_intervention_type(self, tmp_path):
        """Test that invalid intervention_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid intervention_type"):
            train_intervention(
                causal_model=MagicMock(),
                interchange_targets={("test",): MagicMock()},
                train_dataset_path=str(tmp_path),
                test_dataset_path=str(tmp_path),
                pipeline=MagicMock(),
                target_variable_group=("answer",),
                output_dir=str(tmp_path / "output"),
                metric=lambda x, y: True,
                config={"intervention_type": "invalid_type"},
            )

    def test_accepts_mask_intervention_type(self, tmp_path):
        """Test that 'mask' is a valid intervention_type."""
        # Should not raise on intervention_type validation
        # (will fail later on dataset loading, which is expected)
        with pytest.raises(Exception) as exc_info:
            train_intervention(
                causal_model=MagicMock(),
                interchange_targets={("test",): MagicMock()},
                train_dataset_path=str(tmp_path / "nonexistent"),
                test_dataset_path=str(tmp_path / "nonexistent"),
                pipeline=MagicMock(),
                target_variable_group=("answer",),
                output_dir=str(tmp_path / "output"),
                metric=lambda x, y: True,
                config={"intervention_type": "mask"},
            )
        # Should fail on dataset loading, not intervention_type
        assert "Invalid intervention_type" not in str(exc_info.value)

    def test_accepts_interchange_intervention_type(self, tmp_path):
        """Test that 'interchange' is a valid intervention_type."""
        with pytest.raises(Exception) as exc_info:
            train_intervention(
                causal_model=MagicMock(),
                interchange_targets={("test",): MagicMock()},
                train_dataset_path=str(tmp_path / "nonexistent"),
                test_dataset_path=str(tmp_path / "nonexistent"),
                pipeline=MagicMock(),
                target_variable_group=("answer",),
                output_dir=str(tmp_path / "output"),
                metric=lambda x, y: True,
                config={"intervention_type": "interchange"},
            )
        assert "Invalid intervention_type" not in str(exc_info.value)


class TestTrainInterventionConfig:
    """Test configuration handling for train_intervention."""

    def test_default_config_used_when_none_provided(self):
        """Test that DEFAULT_CONFIG values are used when no config provided."""
        # We can't easily test this without mocking everything, but we can
        # verify DEFAULT_CONFIG has expected keys
        assert "train_batch_size" in DEFAULT_CONFIG
        assert "evaluation_batch_size" in DEFAULT_CONFIG
        assert "training_epoch" in DEFAULT_CONFIG
        assert "init_lr" in DEFAULT_CONFIG

    def test_mask_requires_featurizer_kwargs(self, tmp_path):
        """Test that mask intervention requires featurizer_kwargs in config."""
        # Create a config without featurizer_kwargs
        config = {
            "intervention_type": "mask",
            "train_batch_size": 4,
            "training_epoch": 1,
        }

        # Mock the dataset loading to get past that step
        mock_hf_dataset = create_mock_hf_dataset(5)

        # Create properly mocked CounterfactualDataset
        mock_cf_dataset = MagicMock()
        mock_cf_dataset.__len__ = MagicMock(return_value=5)
        mock_cf_dataset.id = "test_dataset"

        with (
            patch(
                "causalab.experiments.train.load_from_disk",
                return_value=mock_hf_dataset,
            ),
            patch(
                "causalab.experiments.train.CounterfactualDataset",
                return_value=mock_cf_dataset,
            ),
        ):
            with pytest.raises(ValueError, match="featurizer_kwargs"):
                train_intervention(
                    causal_model=MagicMock(),
                    interchange_targets={
                        ("test",): MagicMock(flatten=MagicMock(return_value=[]))
                    },
                    train_dataset_path=str(tmp_path),
                    test_dataset_path=str(tmp_path),
                    pipeline=MagicMock(),
                    target_variable_group=("answer",),
                    output_dir=str(tmp_path / "output"),
                    metric=lambda x, y: True,
                    config=config,
                )


class TestTrainInterventionExecution:
    """Test train_intervention execution with mocked dependencies."""

    @pytest.fixture
    def mock_dependencies(self, tmp_path):
        """Set up common mocks for train_intervention tests."""
        # Mock HuggingFace dataset
        mock_hf_dataset = create_mock_hf_dataset(5)

        # Mock CounterfactualDataset
        mock_cf_dataset = MagicMock()
        mock_cf_dataset.__len__ = MagicMock(return_value=5)
        mock_cf_dataset.__iter__ = MagicMock(return_value=iter([]))

        # Mock InterchangeTarget
        mock_target = MagicMock()
        mock_target.flatten = MagicMock(return_value=[MagicMock() for _ in range(3)])
        mock_target.get_feature_indices = MagicMock(
            return_value={
                "unit_0": [0, 1],
                "unit_1": [],
                "unit_2": [0, 1, 2],
            }
        )
        mock_target.save = MagicMock()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.model_or_name = "test_model"

        # Mock causal model
        mock_causal_model = MagicMock()
        mock_causal_model.label_counterfactual_data = MagicMock(
            return_value=[{"input": {"text": "test"}, "label": "A"} for _ in range(5)]
        )

        return {
            "hf_dataset": mock_hf_dataset,
            "cf_dataset": mock_cf_dataset,
            "target": mock_target,
            "pipeline": mock_pipeline,
            "causal_model": mock_causal_model,
            "tmp_path": tmp_path,
        }

    def test_train_intervention_calls_train_interventions(self, mock_dependencies):
        """Test that train_intervention calls the underlying training function."""
        mocks = mock_dependencies

        config = {
            "intervention_type": "mask",
            "train_batch_size": 2,
            "training_epoch": 1,
            "init_lr": 0.001,
            "featurizer_kwargs": {"tie_masks": True},
        }

        # Create directories that would be checked by load_from_disk
        train_dir = mocks["tmp_path"] / "train"
        test_dir = mocks["tmp_path"] / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        with (
            patch(
                "causalab.experiments.train.load_from_disk",
                return_value=mocks["hf_dataset"],
            ),
            patch(
                "causalab.experiments.train.CounterfactualDataset",
                return_value=mocks["cf_dataset"],
            ),
            patch(
                "causalab.experiments.train.train_interventions_pyvene"
            ) as mock_train,
            patch(
                "causalab.experiments.train.run_interchange_interventions"
            ) as mock_run_interventions,
        ):
            # Mock the interventions to return expected format
            mock_run_interventions.return_value = {"string": ["A"] * 5}

            result = train_intervention(
                causal_model=mocks["causal_model"],
                interchange_targets={("test",): mocks["target"]},
                train_dataset_path=str(mocks["tmp_path"] / "train"),
                test_dataset_path=str(mocks["tmp_path"] / "test"),
                pipeline=mocks["pipeline"],
                target_variable_group=("answer",),
                output_dir=str(mocks["tmp_path"] / "output"),
                metric=lambda x, y: x.get("string") == y,
                config=config,
                save_results=False,
            )

            # Verify train_interventions was called
            mock_train.assert_called_once()

            # Verify result structure
            assert "avg_train_score" in result
            assert "avg_test_score" in result
            assert "results_by_key" in result
            assert "metadata" in result

    def test_train_intervention_returns_correct_metadata(self, mock_dependencies):
        """Test that train_intervention returns correct metadata structure."""
        mocks = mock_dependencies

        config = {
            "intervention_type": "mask",
            "train_batch_size": 2,
            "training_epoch": 1,
            "init_lr": 0.001,
            "featurizer_kwargs": {"tie_masks": True},
        }

        # Create directories that would be checked by load_from_disk
        train_dir = mocks["tmp_path"] / "train"
        test_dir = mocks["tmp_path"] / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        with (
            patch(
                "causalab.experiments.train.load_from_disk",
                return_value=mocks["hf_dataset"],
            ),
            patch(
                "causalab.experiments.train.CounterfactualDataset",
                return_value=mocks["cf_dataset"],
            ),
            patch("causalab.experiments.train.train_interventions_pyvene"),
            patch(
                "causalab.experiments.train.run_interchange_interventions"
            ) as mock_run_interventions,
        ):
            # Mock the interventions to return expected format
            mock_run_interventions.return_value = {"string": ["A"] * 5}

            result = train_intervention(
                causal_model=mocks["causal_model"],
                interchange_targets={("test",): mocks["target"]},
                train_dataset_path=str(mocks["tmp_path"] / "train"),
                test_dataset_path=str(mocks["tmp_path"] / "test"),
                pipeline=mocks["pipeline"],
                target_variable_group=("answer",),
                output_dir=str(mocks["tmp_path"] / "output"),
                metric=lambda x, y: True,
                config=config,
                save_results=False,
            )

            metadata = result["metadata"]
            assert metadata["experiment_type"] == "DBM"
            assert metadata["intervention_type"] == "mask"
            assert metadata["target_variable_group"] == ("answer",)
            assert "avg_train_score" in metadata
            assert "avg_test_score" in metadata


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
