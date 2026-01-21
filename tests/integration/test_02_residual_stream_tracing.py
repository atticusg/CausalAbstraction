"""
Integration tests for Notebook 02: Residual Stream Tracing

Tests loading the Llama model and running residual stream tracing
to trace information flow through the residual stream.
"""

import pytest
import random
import torch
import tempfile
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.tasks.MCQA.counterfactuals import sample_answerable_question
from causalab.experiments.jobs.residual_stream_tracing import (
    run_residual_stream_tracing,
)
from causalab.neural.token_position_builder import get_list_of_each_token


pytestmark = [pytest.mark.slow, pytest.mark.gpu]


def _make_trace(text: str) -> CausalTrace:
    """Helper to create a simple CausalTrace from a string."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _create_counterfactual_pair(
    causal_model: CausalModel,
) -> tuple[CausalTrace, CausalTrace]:
    """Create a valid original/counterfactual pair for testing."""
    # Sample original - already a fully computed CausalTrace
    full_setting = sample_answerable_question()

    # Create a counterfactual by changing the answer symbol
    input_vars = {var: full_setting[var] for var in causal_model.inputs}
    answer_symbol_key = f"symbol{full_setting['answer_position']}"
    new_symbols = list({"A", "B", "C"}.difference({full_setting[answer_symbol_key]}))
    input_vars[answer_symbol_key] = random.choice(new_symbols)
    counterfactual_setting = causal_model.new_trace(input_vars)

    return full_setting, counterfactual_setting


class TestModelLoading:
    """Test loading the Llama model via LMPipeline."""

    def test_pipeline_loaded(self, pipeline):
        """Test that pipeline is loaded correctly."""
        assert pipeline is not None
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None

    def test_pipeline_device(self, pipeline, device):
        """Test that model is on expected device."""
        assert str(pipeline.model.device).startswith(device.split(":")[0])

    def test_pipeline_generation(self, pipeline):
        """Test basic generation with the pipeline."""
        test_input = "The sky is blue. What color is the sky?\nA. red\nB. blue\nAnswer:"
        output = pipeline.generate([_make_trace(test_input)])

        # Verify output is a dict with sequences
        assert isinstance(output, dict)
        assert "sequences" in output
        assert isinstance(output["sequences"], torch.Tensor)
        assert output["sequences"].numel() > 0

    def test_pipeline_dump(self, pipeline):
        """Test decoding output with pipeline.dump."""
        test_input = "Test"
        output = pipeline.generate([_make_trace(test_input)])
        decoded = pipeline.dump(output["sequences"])

        # Verify decoded output is a string
        assert isinstance(decoded, str)


class TestResidualStreamTracing:
    """Test residual stream tracing experiments using the functional API."""

    def test_run_tracing_on_valid_pair(self, pipeline, causal_model, checker):
        """Test running tracing experiment on a valid input pair."""
        # Sample original - already a fully computed CausalTrace
        full_setting = sample_answerable_question()

        # Create a counterfactual by changing the answer symbol
        # Extract input variables for modification
        input_vars = {var: full_setting[var] for var in causal_model.inputs}
        answer_symbol_key = f"symbol{full_setting['answer_position']}"
        new_symbols = list(
            {"A", "B", "C"}.difference({full_setting[answer_symbol_key]})
        )
        input_vars[answer_symbol_key] = random.choice(new_symbols)
        counterfactual_setting = causal_model.new_trace(input_vars)

        # Get token positions for the prompt
        token_positions = get_list_of_each_token(full_setting["raw_input"], pipeline)

        # Run tracing with temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_residual_stream_tracing(
                pipeline=pipeline,
                prompt=full_setting["raw_input"],
                counterfactual_prompt=counterfactual_setting["raw_input"],
                token_positions=token_positions,
                output_dir=tmpdir,
                layers=list(
                    range(min(3, pipeline.get_num_layers()))
                ),  # Use fewer layers for speed
                generate_visualization=False,
                save_results=True,
            )

        # Verify results structure
        assert results is not None
        assert "intervention_results" in results
        assert "metadata" in results
        assert len(results["intervention_results"]) > 0

    def test_tracing_results_structure(self, pipeline, causal_model, checker):
        """Test that tracing results have expected structure."""
        # Sample inputs using helper
        full_setting, counterfactual_setting = _create_counterfactual_pair(causal_model)

        # Get token positions
        token_positions = get_list_of_each_token(full_setting["raw_input"], pipeline)

        # Use only first 2 token positions and 2 layers for speed
        token_positions = token_positions[:2]
        layers = list(range(min(2, pipeline.get_num_layers())))

        # Run tracing
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_residual_stream_tracing(
                pipeline=pipeline,
                prompt=full_setting["raw_input"],
                counterfactual_prompt=counterfactual_setting["raw_input"],
                token_positions=token_positions,
                output_dir=tmpdir,
                layers=layers,
                generate_visualization=False,
                save_results=False,
            )

        # Check metadata structure
        metadata = results["metadata"]
        assert "num_layers" in metadata
        assert "layers_used" in metadata
        assert "num_token_positions" in metadata
        assert "total_interventions" in metadata

        # Verify intervention results have correct keys
        intervention_results = results["intervention_results"]
        assert len(intervention_results) == len(layers) * len(token_positions)

        # Check that keys are (layer, token_position_id) tuples
        for key in intervention_results.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            layer, pos_id = key
            assert isinstance(layer, int)
            assert isinstance(pos_id, str)

    def test_tracing_with_same_length_inputs(self, pipeline, causal_model, checker):
        """Test that tracing works with same-length inputs."""
        # Sample inputs using helper
        full_setting, counterfactual_setting = _create_counterfactual_pair(causal_model)

        # Get token positions
        token_positions = get_list_of_each_token(full_setting["raw_input"], pipeline)[
            :2
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_residual_stream_tracing(
                pipeline=pipeline,
                prompt=full_setting["raw_input"],
                counterfactual_prompt=counterfactual_setting["raw_input"],
                token_positions=token_positions,
                output_dir=tmpdir,
                layers=[0, 1],
                generate_visualization=False,
                save_results=False,
            )

        assert results is not None
        assert len(results["intervention_results"]) > 0
