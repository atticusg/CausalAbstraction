"""
Counterfactual generator functions for the task.
"""

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace
from .causal_models import causal_model


def is_valid(setting: CausalTrace) -> bool:
    """Check that the input is semantically valid."""
    raise NotImplementedError("TODO: implement")


def sample_valid_input():
    """Sample a valid input"""
    input_sample = causal_model.sample_input(filter_func=is_valid)
    return input_sample


def random_counterfactual():
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_valid_input()
    counterfactual = sample_valid_input()

    return CounterfactualExample(
        input=input_sample, counterfactual_inputs=[counterfactual]
    )


# TODO: implement ALL counterfactual types from the specification.
# Each type listed in the spec MUST have its own generator function.
# Do not skip any counterfactual type.


COUNTERFACTUAL_GENERATORS = {
    "random_counterfactual": random_counterfactual,
    # TODO: add ALL other counterfactual types from the specification
}
