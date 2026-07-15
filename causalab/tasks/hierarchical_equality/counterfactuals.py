"""
Counterfactual generator functions for the task.
"""

import random

from causalab.causal.counterfactual_dataset import CounterfactualExample

from .causal_models import CAUSAL_MODEL
from .config import PATTERNS
from .templates import TEMPLATES, _sample_pattern_values  # pyright: ignore[reportPrivateUsage]


def sample_balanced_input():
    """Sample a balanced input across the four patterns."""
    pattern = random.choice(PATTERNS)
    v1, v2, v3, v4 = _sample_pattern_values(pattern)
    template = random.choice(TEMPLATES)
    return CAUSAL_MODEL.new_trace(
        {
            "template": template,
            "var_1": v1,
            "var_2": v2,
            "var_3": v3,
            "var_4": v4,
        }
    )


def random_counterfactual():
    """Generate a random counterfactual by sampling two independent balanced inputs."""
    input_sample = sample_balanced_input()
    counterfactual = sample_balanced_input()

    return CounterfactualExample(
        input=input_sample, counterfactual_inputs=[counterfactual]
    )


COUNTERFACTUAL_GENERATORS = {
    "random_counterfactual": random_counterfactual,
}


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate n counterfactual examples using balanced sampling."""
    state = random.getstate()
    random.seed(seed)
    examples: list[CounterfactualExample] = [
        {
            "input": sample_balanced_input(),
            "counterfactual_inputs": [sample_balanced_input()],
        }
        for _ in range(n)
    ]
    random.setstate(state)
    return examples
