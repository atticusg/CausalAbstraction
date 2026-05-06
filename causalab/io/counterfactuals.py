"""Disk I/O for counterfactual example datasets."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Sequence, cast

from causalab.causal.counterfactual_dataset import CounterfactualExample

if TYPE_CHECKING:
    from causalab.causal.causal_model import CausalModel


def save_counterfactual_examples(
    examples: list[CounterfactualExample],
    path: str,
) -> None:
    """Save counterfactual examples to disk as JSON."""

    def serialize_example(ex: CounterfactualExample) -> dict[str, Any]:
        return {
            "input": ex["input"].to_dict(),
            "counterfactual_inputs": [t.to_dict() for t in ex["counterfactual_inputs"]],
        }

    serialized = [serialize_example(ex) for ex in examples]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


def load_counterfactual_examples(
    path: str, causal_model: "CausalModel"
) -> list[CounterfactualExample]:
    """Load counterfactual examples from disk and rehydrate CausalTrace objects.

    Supports both list format ``[{"input": ..., "counterfactual_inputs": [...]}, ...]``
    and dict format ``{"input": [...], "counterfactual_inputs": [[...], ...]}``.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "input" in data and "counterfactual_inputs" in data:
        data = [
            {"input": inp, "counterfactual_inputs": cf}
            for inp, cf in zip(data["input"], data["counterfactual_inputs"])
        ]

    return deserialize_counterfactual_examples(
        cast(list[CounterfactualExample], data), causal_model
    )


def deserialize_counterfactual_examples(
    dataset: Sequence[CounterfactualExample], causal_model: "CausalModel"
) -> list[CounterfactualExample]:
    """Convert dicts loaded from disk back to CausalTrace objects."""
    result = []
    for example in dataset:
        input_data = example["input"]
        cf_inputs_data = example["counterfactual_inputs"]

        if isinstance(input_data, dict):
            input_trace = causal_model.new_trace(input_data)
        else:
            input_trace = input_data

        cf_traces = []
        for cf_data in cf_inputs_data:
            if isinstance(cf_data, dict):
                cf_traces.append(causal_model.new_trace(cf_data))
            else:
                cf_traces.append(cf_data)

        result.append({"input": input_trace, "counterfactual_inputs": cf_traces})

    return result
