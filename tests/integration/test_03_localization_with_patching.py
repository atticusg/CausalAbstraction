"""
Integration tests for Notebook 03: Localization with Patching

Tests filtering datasets, running residual stream patching experiments,
and localizing answer and answer_position variables.
"""

import pytest
import tempfile
import os
from causalab.tasks.MCQA import token_positions
from causalab.causal.causal_utils import save_counterfactual_examples
from causalab.tasks.MCQA.counterfactuals import sample_answerable_question
from causalab.causal.causal_utils import generate_counterfactual_samples
from causalab.experiments.filter import filter_dataset
from causalab.experiments.metric import causal_score_intervention_outputs
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.neural.pyvene_core.interchange import run_interchange_interventions


pytestmark = [pytest.mark.slow, pytest.mark.gpu]


class TestDatasetFiltering:
    """Test filtering counterfactual datasets based on model performance."""

    def test_filter_dataset(self, pipeline, causal_model, checker):
        """Test filtering a single counterfactual dataset."""
        from causalab.tasks.MCQA.counterfactuals import different_symbol

        # Generate a small dataset
        dataset = generate_counterfactual_samples(8, different_symbol)

        # Filter the dataset
        filtered = filter_dataset(
            dataset=dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        # Verify that we get a filtered dataset back
        assert isinstance(filtered, list)
        assert len(filtered) <= len(dataset)

    def test_filter_multiple_datasets(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset,
        small_same_symbol_diff_position_dataset,
        small_random_dataset,
    ):
        """Test filtering multiple counterfactual datasets."""
        datasets = {
            "different_symbol": small_different_symbol_dataset,
            "same_symbol_different_position": small_same_symbol_diff_position_dataset,
            "random_counterfactual": small_random_dataset,
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

        # Verify that we get filtered datasets back
        assert isinstance(filtered_datasets, dict)
        assert len(filtered_datasets) == 3

        # Verify each filtered dataset is not larger than original
        for key in datasets.keys():
            assert key in filtered_datasets
            assert len(filtered_datasets[key]) <= len(datasets[key])

    def test_filtered_dataset_structure(
        self, pipeline, causal_model, checker, small_different_symbol_dataset
    ):
        """Test that filtered datasets maintain proper structure."""
        filtered = filter_dataset(
            dataset=small_different_symbol_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        # Check structure is preserved
        if len(filtered) > 0:
            example = filtered[0]
            assert "input" in example
            assert "counterfactual_inputs" in example


class TestTokenPositions:
    """Test token position definitions for the MCQA task."""

    def test_create_token_positions(self, pipeline):
        """Test creating token positions for MCQA task."""
        token_positions_dict = token_positions.create_token_positions(pipeline)

        assert isinstance(token_positions_dict, dict)
        assert len(token_positions_dict) > 0

        # Check expected positions exist
        expected_positions = ["symbol0", "symbol1", "correct_symbol", "last_token"]
        for pos_name in expected_positions:
            assert pos_name in token_positions_dict

    def test_token_position_selection(self, pipeline, causal_model):
        """Test that token positions correctly select tokens."""

        token_positions_dict = token_positions.create_token_positions(pipeline)
        example = sample_answerable_question()

        # Test that each position can highlight a token
        for _, token_pos in token_positions_dict.items():
            highlighted = token_pos.highlight_selected_token(example)
            assert isinstance(highlighted, str)
            assert "**" in highlighted  # Check that highlighting occurred


class TestResidualStreamPatching:
    """Test activation patching experiments on residual stream."""

    def test_run_patching_experiment(
        self, pipeline, causal_model, checker, small_different_symbol_dataset
    ):
        """Test running patching experiment and verify results structure."""
        # Filter dataset first
        filtered_dataset = filter_dataset(
            dataset=small_different_symbol_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_dataset) == 0:
            pytest.skip("No examples passed filtering")

        # Create token positions
        token_positions_dict = token_positions.create_token_positions(pipeline)
        # Use only first 2 positions for speed
        limited_positions = dict(list(token_positions_dict.items())[:2])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save dataset to disk
            dataset_path = os.path.join(tmpdir, "test_dataset.json")
            save_counterfactual_examples(filtered_dataset, dataset_path)

            # Build residual stream targets
            layers = list(range(min(3, pipeline.get_num_layers())))
            targets = build_residual_stream_targets(
                pipeline=pipeline,
                layers=layers,
                token_positions=list(limited_positions.values()),
                mode="one_target_per_unit",
            )

            # Run interventions directly
            raw_results = {}
            for key, target in targets.items():
                raw_results[key] = run_interchange_interventions(
                    pipeline=pipeline,
                    counterfactual_dataset=filtered_dataset,
                    interchange_target=target,
                    batch_size=8,
                    output_scores=False,
                )

            # Score the results
            results = causal_score_intervention_outputs(
                raw_results=raw_results,
                dataset=filtered_dataset,
                causal_model=causal_model,
                target_variable_groups=[("answer",)],
                metric=checker,
            )

        # Verify results structure
        assert results is not None
        assert "results_by_key" in results
        assert "scores_by_variable" in results

        # Verify scores structure
        assert ("answer",) in results["scores_by_variable"]
        assert results["scores_by_variable"][("answer",)] >= 0

    def test_patching_with_multiple_target_variables(
        self, pipeline, causal_model, checker, small_different_symbol_dataset
    ):
        """Test performing patching with multiple target variables."""
        # Filter dataset first
        filtered_dataset = filter_dataset(
            dataset=small_different_symbol_dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=checker,
            batch_size=8,
        )

        if len(filtered_dataset) == 0:
            pytest.skip("No examples passed filtering")

        # Create token positions (use just 1 for speed)
        token_positions_dict = token_positions.create_token_positions(pipeline)
        limited_positions = {"last_token": token_positions_dict["last_token"]}

        # Build residual stream targets
        layers = list(range(min(2, pipeline.get_num_layers())))
        targets = build_residual_stream_targets(
            pipeline=pipeline,
            layers=layers,
            token_positions=list(limited_positions.values()),
            mode="one_target_per_unit",
        )

        # Run interventions directly
        raw_results = {}
        for key, target in targets.items():
            raw_results[key] = run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=filtered_dataset,
                interchange_target=target,
                batch_size=8,
                output_scores=False,
            )

        # Score with multiple target variables
        results = causal_score_intervention_outputs(
            raw_results=raw_results,
            dataset=filtered_dataset,
            causal_model=causal_model,
            target_variable_groups=[("answer",), ("answer_position",)],
            metric=checker,
        )

        # Verify both target variable groups have results
        assert ("answer",) in results["scores_by_variable"]
        assert ("answer_position",) in results["scores_by_variable"]


class TestIntegrationWorkflow:
    """Test the complete workflow from notebook 03."""

    def test_full_workflow(
        self,
        pipeline,
        causal_model,
        checker,
        small_different_symbol_dataset,
        small_same_symbol_diff_position_dataset,
    ):
        """Test the complete workflow: filter -> create experiment -> run interventions."""
        # Step 1: Filter datasets
        datasets = {
            "different_symbol": small_different_symbol_dataset,
            "same_symbol_different_position": small_same_symbol_diff_position_dataset,
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

        # Check if we have any data to work with
        total_examples = sum(len(d) for d in filtered_datasets.values())
        if total_examples == 0:
            pytest.skip("No examples passed filtering")

        # Step 2: Create token positions (minimal for speed)
        token_positions_dict = token_positions.create_token_positions(pipeline)
        limited_positions = {"last_token": token_positions_dict["last_token"]}

        # Step 3: Run patching on first non-empty dataset
        for name, dataset in filtered_datasets.items():
            if len(dataset) > 0:
                # Build residual stream targets
                layers = list(range(min(2, pipeline.get_num_layers())))
                targets = build_residual_stream_targets(
                    pipeline=pipeline,
                    layers=layers,
                    token_positions=list(limited_positions.values()),
                    mode="one_target_per_unit",
                )

                # Step 4: Run interventions directly
                raw_results = {}
                for key, target in targets.items():
                    raw_results[key] = run_interchange_interventions(
                        pipeline=pipeline,
                        counterfactual_dataset=dataset,
                        interchange_target=target,
                        batch_size=8,
                        output_scores=False,
                    )

                # Score the results
                results = causal_score_intervention_outputs(
                    raw_results=raw_results,
                    dataset=dataset,
                    causal_model=causal_model,
                    target_variable_groups=[("answer",)],
                    metric=checker,
                )

                # Verify end-to-end results
                assert results is not None
                assert "scores_by_variable" in results
                assert len(results["scores_by_variable"]) > 0
                break  # Only test one dataset for speed
