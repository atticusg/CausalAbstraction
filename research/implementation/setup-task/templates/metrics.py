"""
Metrics for evaluating model outputs on the task.

IMPORTANT: The metric function signature is ALWAYS metric(neural_output, causal_output) -> bool.
It receives only the model's output string and the expected causal output string.
It does NOT have access to logits, the pipeline, or the tokenizer.
Do not implement logit-based or probability-based metrics here — those are computed
separately during experiments.
"""

from typing import Any


def metric(neural_output: dict[str, Any], causal_output: str) -> bool:
    """Check if the causal output matches the neural output string.

    Args:
        neural_output: Dict with "string" key containing model output
        causal_output: Expected output string from causal model

    Returns:
        True if the outputs match
    """
    expected = causal_output.strip()
    actual = neural_output["string"].strip()

    return actual == expected or actual.startswith(expected)
