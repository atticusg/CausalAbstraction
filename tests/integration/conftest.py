"""Shared fixtures for integration tests."""

import pytest
import torch
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from causalab.tasks.MCQA.causal_models import positional_causal_model
from causalab.tasks.MCQA.counterfactuals import (
    different_symbol,
    same_symbol_different_position,
    random_counterfactual,
)
from causalab.tasks.MCQA.token_positions import create_token_positions, TEMPLATES
from causalab.neural.token_position_builder import TokenPosition


@pytest.fixture(scope="module")
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def pipeline(device):
    """Load Llama model pipeline (shared across module for efficiency)."""
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    pipeline = LMPipeline(
        model_name,
        max_new_tokens=1,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_length=32,
    )
    pipeline.tokenizer.padding_side = "left"
    return pipeline


@pytest.fixture(scope="module")
def causal_model():
    """Load MCQA positional causal model."""
    return positional_causal_model


@pytest.fixture
def checker():
    """Checker function for comparing neural and causal outputs."""

    def _checker(neural_output, causal_output):
        # Handle case where neural_output is a dict with 'string' key
        if isinstance(neural_output, dict) and "string" in neural_output:
            neural_output = neural_output["string"]
        return causal_output in neural_output or neural_output in causal_output

    return _checker


@pytest.fixture
def small_different_symbol_dataset():
    """Generate small different_symbol counterfactual dataset."""
    return CounterfactualDataset.from_sampler(8, different_symbol)


@pytest.fixture
def small_same_symbol_diff_position_dataset():
    """Generate small same_symbol_different_position counterfactual dataset."""
    return CounterfactualDataset.from_sampler(8, same_symbol_different_position)


@pytest.fixture
def small_random_dataset():
    """Generate small random counterfactual dataset."""
    return CounterfactualDataset.from_sampler(8, random_counterfactual)


@pytest.fixture
def answer_token_position(pipeline: LMPipeline) -> TokenPosition:
    """Create answer token position for the given pipeline."""
    return create_token_positions(pipeline, template=TEMPLATES[0])["correct_symbol"]
