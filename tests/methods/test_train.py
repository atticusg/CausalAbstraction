# tests/test_experiments/test_train.py
"""
Tests for methods/trained_subspace/train.py - train_interventions public API.
"""

import pytest
from unittest.mock import MagicMock, patch

import torch

from causalab.neural.pipeline import GenerationResult

from causalab.methods.trained_subspace.train import train_interventions
from causalab.configs.train_config import (
    DEFAULT_CONFIG,
    merge_with_defaults,
)


pytestmark = pytest.mark.unit


def create_mock_dataset(size: int = 5) -> list[dict[str, object]]:
    return [
        {
            "input": {"text": "test", "raw_input": "test"},
            "counterfactual_inputs": [{"text": "cf", "raw_input": "cf"}],
        }
        for _ in range(size)
    ]


class TestTrainInterventionValidation:
    def test_rejects_invalid_intervention_type(self, tmp_path):
        config = merge_with_defaults(None)
        config["intervention_type"] = "invalid_type"  # type: ignore[typeddict-item]
        config["log_dir"] = str(tmp_path / "logs")
        with pytest.raises(ValueError, match="Invalid intervention_type"):
            # config is TypedDict ExperimentConfig; signature expects plain dict
            train_interventions(
                causal_model=MagicMock(),
                grid={("test",): [[MagicMock()]]},
                train_dataset=[],
                test_dataset=[],
                pipeline=MagicMock(),
                target_variable_group=("answer",),
                metric=lambda x, y: True,
                config=config,  # pyright: ignore[reportArgumentType]
            )


class TestTrainInterventionConfig:
    def test_default_config_has_required_keys(self):
        assert "train_batch_size" in DEFAULT_CONFIG
        assert "evaluation_batch_size" in DEFAULT_CONFIG
        assert "training_epoch" in DEFAULT_CONFIG
        assert "init_lr" in DEFAULT_CONFIG

    def test_default_config_includes_featurizer_kwargs(self):
        assert "featurizer_kwargs" in DEFAULT_CONFIG
        assert "tie_masks" in DEFAULT_CONFIG["featurizer_kwargs"]


class TestTrainInterventionExecution:
    @pytest.fixture
    def mock_dependencies(self, tmp_path):
        mock_dataset = create_mock_dataset(5)

        # Nested spec groups: one group of three mock specs whose featurizer
        # id is non-"null" so the functional initializer passes them through.
        mock_specs = []
        for i in range(3):
            spec = MagicMock()
            spec.key = f"unit_{i}"
            spec.fsite.featurizer.id = "pretrained"
            mock_specs.append(spec)
        mock_groups = [mock_specs]
        mock_feature_indices = {
            "unit_0": [0, 1],
            "unit_1": [],
            "unit_2": [0, 1, 2],
        }

        mock_pipeline = MagicMock()
        mock_pipeline.model_or_name = "test_model"

        mock_causal_model = MagicMock()
        mock_causal_model.label_counterfactual_data = MagicMock(
            return_value=[{"input": {"text": "test"}, "label": "A"} for _ in range(5)]
        )

        return {
            "dataset": mock_dataset,
            "groups": mock_groups,
            "feature_indices": mock_feature_indices,
            "pipeline": mock_pipeline,
            "causal_model": mock_causal_model,
            "tmp_path": tmp_path,
        }

    def _make_config(self, tmp_path):
        config = merge_with_defaults(
            {
                "intervention_type": "mask",
                "train_batch_size": 2,
                "training_epoch": 1,
                "init_lr": 0.001,
                "featurizer_kwargs": {"tie_masks": True},
            }
        )
        config["log_dir"] = str(tmp_path / "logs")
        return config

    def test_train_intervention_calls_train_loop(self, mock_dependencies):
        mocks = mock_dependencies
        config = self._make_config(mocks["tmp_path"])

        with (
            patch(
                "causalab.methods.trained_subspace.train._run_training_loop"
            ) as mock_train,
            patch(
                "causalab.methods.trained_subspace.train.run_interchange_interventions"
            ) as mock_run_interventions,
        ):
            mock_train.return_value = (
                mocks["groups"],
                mocks["feature_indices"],
                "summary",
            )
            mock_run_interventions.return_value = GenerationResult(
                sequences=torch.zeros((5, 1), dtype=torch.long),
                strings=["A"] * 5,
            )

            # config is TypedDict ExperimentConfig; signature expects plain dict
            result = train_interventions(
                causal_model=mocks["causal_model"],
                grid={("test",): mocks["groups"]},
                train_dataset=mocks["dataset"],
                test_dataset=mocks["dataset"],
                pipeline=mocks["pipeline"],
                target_variable_group=("answer",),
                metric=lambda x, y: x.get("string") == y,
                config=config,  # pyright: ignore[reportArgumentType]
            )

            mock_train.assert_called_once()
            assert "avg_train_score" in result
            assert "avg_test_score" in result
            assert "results_by_key" in result
            assert "metadata" in result

    def test_train_intervention_returns_correct_metadata(self, mock_dependencies):
        mocks = mock_dependencies
        config = self._make_config(mocks["tmp_path"])

        with (
            patch(
                "causalab.methods.trained_subspace.train._run_training_loop"
            ) as mock_train,
            patch(
                "causalab.methods.trained_subspace.train.run_interchange_interventions"
            ) as mock_run_interventions,
        ):
            mock_train.return_value = (
                mocks["groups"],
                mocks["feature_indices"],
                "summary",
            )
            mock_run_interventions.return_value = GenerationResult(
                sequences=torch.zeros((5, 1), dtype=torch.long),
                strings=["A"] * 5,
            )

            # config is TypedDict ExperimentConfig; signature expects plain dict
            result = train_interventions(
                causal_model=mocks["causal_model"],
                grid={("test",): mocks["groups"]},
                train_dataset=mocks["dataset"],
                test_dataset=mocks["dataset"],
                pipeline=mocks["pipeline"],
                target_variable_group=("answer",),
                metric=lambda x, y: True,
                config=config,  # pyright: ignore[reportArgumentType]
            )

            metadata = result["metadata"]
            assert metadata["intervention_type"] == "mask"
            assert metadata["target_variable_group"] == ["answer"]
            assert "avg_train_score" in metadata
            assert "avg_test_score" in metadata
            # Research-question fields must NOT be stamped by methods/ layer
            assert "experiment_type" not in metadata
            assert "model" not in metadata
            assert "train_dataset_path" not in metadata
            assert "test_dataset_path" not in metadata


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
