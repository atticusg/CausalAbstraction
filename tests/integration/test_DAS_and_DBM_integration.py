"""
Integration test for DAS (Distributed Alignment Search) and DBM (Desiderata-Based Masking).

This test is modeled after the 04_train_DAS_and_DBM.ipynb notebook and verifies:
1. Creating MCQA task datasets
2. Filtering datasets based on model performance
3. Training DAS interventions on residual stream to localize causal variables
4. Training DBM interventions on attention heads to identify responsible heads
5. Evaluating on held-out test data for generalization
"""

import pytest
import tempfile
import os

from causalab.tasks.MCQA.token_positions import create_token_positions
from causalab.causal.causal_utils import save_counterfactual_examples
from causalab.tasks.MCQA.counterfactuals import (
    different_symbol,
    same_symbol_different_position,
)
from causalab.causal.causal_utils import generate_counterfactual_samples
from causalab.experiments.filter import filter_dataset
from causalab.experiments.jobs.DAS_grid import train_DAS
from causalab.experiments.jobs.DBM_binary_grid import train_DBM_binary_heatmaps
from causalab.experiments.interchange_targets import (
    build_residual_stream_targets,
    build_attention_head_targets,
)
from causalab.neural.token_position_builder import get_all_tokens

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


class TestDASIntegration:
    """Integration tests for Distributed Alignment Search (DAS)."""

    def test_das_training(self, pipeline, causal_model, checker):
        """Test DAS training flow with real model."""
        # Generate and filter datasets
        train_dataset = generate_counterfactual_samples(
            16, same_symbol_different_position
        )
        test_dataset = generate_counterfactual_samples(
            8, same_symbol_different_position
        )

        filtered_train = filter_dataset(
            dataset=train_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        filtered_test = filter_dataset(
            dataset=test_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_train) < 2 or len(filtered_test) < 2:
            pytest.skip("Not enough examples passed filtering")

        # Create token positions (use minimal set for speed)
        token_positions_dict = create_token_positions(pipeline)
        limited_positions = [token_positions_dict["last_token"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save datasets to disk
            train_path = os.path.join(tmpdir, "train_dataset.json")
            test_path = os.path.join(tmpdir, "test_dataset.json")
            save_counterfactual_examples(filtered_train, train_path)
            save_counterfactual_examples(filtered_test, test_path)

            # Build residual stream targets for DAS
            layers = [0, 1]  # Use only 2 layers for speed
            targets = build_residual_stream_targets(
                pipeline=pipeline,
                layers=layers,
                token_positions=limited_positions,
                mode="one_target_per_unit",
            )

            # Train DAS
            result = train_DAS(
                causal_model=causal_model,
                interchange_targets=targets,
                train_dataset_path=train_path,
                test_dataset_path=test_path,
                pipeline=pipeline,
                target_variable_group=("answer_position",),
                output_dir=os.path.join(tmpdir, "das_results"),
                metric=checker,
                save_results=False,
                verbose=False,
            )

        # Verify results structure
        assert result is not None
        assert "train_scores" in result
        assert "test_scores" in result
        assert "metadata" in result

        # Verify scores exist (train_DAS returns scores by key, not by variable)
        assert len(result["train_scores"]) > 0
        assert len(result["test_scores"]) > 0

    def test_das_generalization(self, pipeline, causal_model, checker):
        """Test that DAS results can be evaluated on held-out test data."""
        # Generate and filter datasets
        train_dataset = generate_counterfactual_samples(
            16, same_symbol_different_position
        )
        test_dataset = generate_counterfactual_samples(
            8, same_symbol_different_position
        )

        filtered_train = filter_dataset(
            dataset=train_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        filtered_test = filter_dataset(
            dataset=test_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_train) < 2 or len(filtered_test) < 2:
            pytest.skip("Not enough examples passed filtering")

        token_positions_dict = create_token_positions(pipeline)
        limited_positions = [token_positions_dict["last_token"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train_dataset.json")
            test_path = os.path.join(tmpdir, "test_dataset.json")
            save_counterfactual_examples(filtered_train, train_path)
            save_counterfactual_examples(filtered_test, test_path)

            # Build residual stream targets for DAS
            layers = [0]  # Single layer for speed
            targets = build_residual_stream_targets(
                pipeline=pipeline,
                layers=layers,
                token_positions=limited_positions,
                mode="one_target_per_unit",
            )

            result = train_DAS(
                causal_model=causal_model,
                interchange_targets=targets,
                train_dataset_path=train_path,
                test_dataset_path=test_path,
                pipeline=pipeline,
                target_variable_group=("answer_position",),
                output_dir=os.path.join(tmpdir, "das_results"),
                metric=checker,
                save_results=False,
                verbose=False,
            )

        # Test scores should exist (generalization gap is expected)
        train_scores = result["train_scores"]
        test_scores = result["test_scores"]

        # Both should have scores
        assert len(train_scores) > 0
        assert len(test_scores) > 0

        # Metadata should have summary statistics
        assert "train_max_score" in result["metadata"]
        assert "test_max_score" in result["metadata"]


class TestDBMIntegration:
    """Integration tests for Desiderata-Based Masking (DBM)."""

    def test_dbm_training(self, pipeline, causal_model, checker):
        """Test DBM training flow with real model."""
        # Generate and filter datasets
        train_dataset = generate_counterfactual_samples(16, different_symbol)
        test_dataset = generate_counterfactual_samples(8, different_symbol)

        filtered_train = filter_dataset(
            dataset=train_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        filtered_test = filter_dataset(
            dataset=test_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_train) < 2 or len(filtered_test) < 2:
            pytest.skip("Not enough examples passed filtering")

        # Create token position for all tokens
        sample_input = filtered_train[0]["input"]
        all_tokens = get_all_tokens(sample_input, pipeline, padding=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save datasets to disk
            train_path = os.path.join(tmpdir, "train_dataset.json")
            test_path = os.path.join(tmpdir, "test_dataset.json")
            save_counterfactual_examples(filtered_train, train_path)
            save_counterfactual_examples(filtered_test, test_path)

            # Build attention head targets for DBM
            layers = [0, 1]  # Use only 2 layers for speed
            num_heads = pipeline.model.config.num_attention_heads
            heads = list(range(num_heads))
            targets = build_attention_head_targets(
                pipeline=pipeline,
                layers=layers,
                heads=heads,
                token_position=all_tokens,
                mode="one_target_all_units",
            )

            # Train DBM
            result = train_DBM_binary_heatmaps(
                causal_model=causal_model,
                interchange_target=targets[("all",)],
                train_dataset_path=train_path,
                test_dataset_path=test_path,
                pipeline=pipeline,
                target_variable_group=("answer",),
                output_dir=os.path.join(tmpdir, "dbm_results"),
                metric=checker,
                tie_masks=True,
                save_results=False,
                verbose=False,
            )

        # Verify results structure
        assert result is not None
        assert "train_score" in result
        assert "test_score" in result
        assert "selected_units" in result
        assert "metadata" in result

        # Verify selected_units is a list of tuples
        assert isinstance(result["selected_units"], list)

    def test_dbm_mask_extraction(self, pipeline, causal_model, checker):
        """Test extracting binary masks from DBM results."""
        train_dataset = generate_counterfactual_samples(16, different_symbol)
        test_dataset = generate_counterfactual_samples(8, different_symbol)

        filtered_train = filter_dataset(
            dataset=train_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        filtered_test = filter_dataset(
            dataset=test_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_train) < 2 or len(filtered_test) < 2:
            pytest.skip("Not enough examples passed filtering")

        sample_input = filtered_train[0]["input"]
        all_tokens = get_all_tokens(sample_input, pipeline, padding=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train_dataset.json")
            test_path = os.path.join(tmpdir, "test_dataset.json")
            save_counterfactual_examples(filtered_train, train_path)
            save_counterfactual_examples(filtered_test, test_path)

            # Build attention head targets for DBM
            layers = [0]  # Single layer for speed
            num_heads = pipeline.model.config.num_attention_heads
            heads = list(range(num_heads))
            targets = build_attention_head_targets(
                pipeline=pipeline,
                layers=layers,
                heads=heads,
                token_position=all_tokens,
                mode="one_target_all_units",
            )

            result = train_DBM_binary_heatmaps(
                causal_model=causal_model,
                interchange_target=targets[("all",)],
                train_dataset_path=train_path,
                test_dataset_path=test_path,
                pipeline=pipeline,
                target_variable_group=("answer",),
                output_dir=os.path.join(tmpdir, "dbm_results"),
                metric=checker,
                tie_masks=True,
                save_results=False,
                verbose=False,
            )

        # Verify feature_indices structure
        assert "feature_indices" in result
        feature_indices = result["feature_indices"]
        assert isinstance(feature_indices, dict)

        # Verify metadata has head information
        assert "num_selected_units" in result["metadata"]
        assert "num_units" in result["metadata"]


class TestFilterExperimentIntegration:
    """Integration tests for filter_dataset used before DAS/DBM."""

    def test_filter_dataset_with_mcqa_task(self, pipeline, causal_model, checker):
        """Test filtering datasets based on model performance."""
        # Generate datasets
        dataset = generate_counterfactual_samples(16, different_symbol)

        # Filter dataset
        filtered = filter_dataset(
            dataset=dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        # Verify filtering worked
        assert isinstance(filtered, list)
        assert len(filtered) <= len(dataset)

    def test_filter_multiple_datasets(self, pipeline, causal_model, checker):
        """Test filtering multiple datasets."""
        datasets = {
            "different_symbol": generate_counterfactual_samples(8, different_symbol),
            "same_symbol_different_position": generate_counterfactual_samples(
                8, same_symbol_different_position
            ),
        }

        filtered_datasets = {}
        for name, dataset in datasets.items():
            filtered_datasets[name] = filter_dataset(
                dataset=dataset,
                pipeline=pipeline,
                causal_model=causal_model,
                metric=checker,
                batch_size=8,
            )

        # Verify all datasets were filtered
        assert len(filtered_datasets) == 2
        for name in datasets.keys():
            assert name in filtered_datasets
            assert len(filtered_datasets[name]) <= len(datasets[name])


class TestEndToEndWorkflow:
    """End-to-end integration tests simulating the notebook workflow."""

    def test_complete_das_workflow(self, pipeline, causal_model, checker):
        """Test complete DAS workflow: filter -> train -> evaluate."""
        # Step 1: Generate and filter datasets
        train_dataset = generate_counterfactual_samples(
            16, same_symbol_different_position
        )
        test_dataset = generate_counterfactual_samples(
            8, same_symbol_different_position
        )

        filtered_train = filter_dataset(
            dataset=train_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        filtered_test = filter_dataset(
            dataset=test_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_train) < 2 or len(filtered_test) < 2:
            pytest.skip("Not enough examples passed filtering")

        # Step 2: Create token positions
        token_positions_dict = create_token_positions(pipeline)
        limited_positions = [token_positions_dict["last_token"]]

        # Step 3: Train and evaluate DAS
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train_dataset.json")
            test_path = os.path.join(tmpdir, "test_dataset.json")
            save_counterfactual_examples(filtered_train, train_path)
            save_counterfactual_examples(filtered_test, test_path)

            # Build residual stream targets for DAS
            layers = [0]  # Minimal for speed
            targets = build_residual_stream_targets(
                pipeline=pipeline,
                layers=layers,
                token_positions=limited_positions,
                mode="one_target_per_unit",
            )

            result = train_DAS(
                causal_model=causal_model,
                interchange_targets=targets,
                train_dataset_path=train_path,
                test_dataset_path=test_path,
                pipeline=pipeline,
                target_variable_group=("answer_position",),
                output_dir=os.path.join(tmpdir, "das_results"),
                metric=checker,
                save_results=False,
                verbose=False,
            )

        # Verify complete workflow results
        assert result is not None
        assert "train_scores" in result
        assert "test_scores" in result
        assert result["metadata"]["train_max_score"] is not None
        assert result["metadata"]["test_max_score"] is not None

    def test_complete_dbm_workflow(self, pipeline, causal_model, checker):
        """Test complete DBM workflow: filter -> train -> evaluate."""
        # Step 1: Generate and filter datasets
        train_dataset = generate_counterfactual_samples(16, different_symbol)
        test_dataset = generate_counterfactual_samples(8, different_symbol)

        filtered_train = filter_dataset(
            dataset=train_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        filtered_test = filter_dataset(
            dataset=test_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_train) < 2 or len(filtered_test) < 2:
            pytest.skip("Not enough examples passed filtering")

        # Step 2: Create token position for all tokens
        sample_input = filtered_train[0]["input"]
        all_tokens = get_all_tokens(sample_input, pipeline, padding=True)

        # Step 3: Train and evaluate DBM
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train_dataset.json")
            test_path = os.path.join(tmpdir, "test_dataset.json")
            save_counterfactual_examples(filtered_train, train_path)
            save_counterfactual_examples(filtered_test, test_path)

            # Build attention head targets for DBM
            layers = [0]  # Minimal for speed
            num_heads = pipeline.model.config.num_attention_heads
            heads = list(range(num_heads))
            targets = build_attention_head_targets(
                pipeline=pipeline,
                layers=layers,
                heads=heads,
                token_position=all_tokens,
                mode="one_target_all_units",
            )

            result = train_DBM_binary_heatmaps(
                causal_model=causal_model,
                interchange_target=targets[("all",)],
                train_dataset_path=train_path,
                test_dataset_path=test_path,
                pipeline=pipeline,
                target_variable_group=("answer",),
                output_dir=os.path.join(tmpdir, "dbm_results"),
                metric=checker,
                tie_masks=True,
                save_results=False,
                verbose=False,
            )

        # Verify complete workflow results
        assert result is not None
        assert "train_score" in result
        assert "test_score" in result
        assert "selected_units" in result
        assert result["metadata"]["num_selected_units"] is not None


# Run tests when file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
