"""
data_utils.py
=============
Shared utilities for pyvene-based interventions.

Provides common functions for output processing and data loading used by
both interchange and steering intervention modules.
"""

from __future__ import annotations

from typing import Any

import torch

from causalab.neural.pipeline import Pipeline


def convert_to_top_k(
    outputs: list[dict[str, Any]], pipeline: Pipeline, k: int
) -> list[dict[str, Any]]:
    """
    Convert full vocabulary logits to top-k format to reduce memory usage.

    This processes outputs to extract only the top-k logits, indices, and tokens,
    dramatically reducing memory footprint (e.g., from ~256K to 10 values per token).

    Args:
        outputs: List of output dictionaries with 'scores' (list of tensors on GPU)
        pipeline: Pipeline with tokenizer for decoding tokens
        k: Number of top logits to keep (must be > 0)

    Returns:
        Modified outputs where 'scores' contains top-k structured data
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    converted_outputs = []
    for batch_dict in outputs:
        # Carry over all keys except "scores" (which gets converted below)
        converted_batch = {
            key: val for key, val in batch_dict.items() if key != "scores"
        }

        # Convert scores to top-k format if present
        scores = batch_dict.get("scores")
        if scores:
            top_k_scores = []
            for position_logits in scores:
                # position_logits shape: (batch_size, vocab_size)
                vocab_size = position_logits.shape[1]
                k_actual = min(k, vocab_size)

                # Get top-k values and indices for entire batch
                top_k_values, top_k_indices = torch.topk(
                    position_logits, k=k_actual, dim=1
                )

                # Decode tokens for entire batch - flatten and batch decode for efficiency
                batch_size = position_logits.shape[0]
                flat_indices = top_k_indices.flatten().tolist()
                flat_tokens = pipeline.tokenizer.batch_decode(
                    [[idx] for idx in flat_indices], skip_special_tokens=False
                )

                # Reshape back to (batch_size, k)
                top_k_tokens = [
                    flat_tokens[i * k_actual : (i + 1) * k_actual]
                    for i in range(batch_size)
                ]

                top_k_scores.append(
                    {
                        "top_k_logits": top_k_values,  # Tensor[batch, k]
                        "top_k_indices": top_k_indices,  # Tensor[batch, k]
                        "top_k_tokens": top_k_tokens,  # List[batch][k]
                    }
                )

            converted_batch["scores"] = top_k_scores

        converted_outputs.append(converted_batch)

    return converted_outputs


def move_outputs_to_cpu(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Move all tensors in outputs to CPU and detach from computation graph.
    Handles nested structures including top-K formatted scores.

    Args:
        outputs: List of output dictionaries

    Returns:
        Same structure with all tensors moved to CPU and detached
    """

    def move_to_cpu(value: Any) -> Any:
        """Recursively move value to CPU, handling various types."""
        if value is None:
            return None
        elif isinstance(value, torch.Tensor):
            return value.detach().cpu()
        elif isinstance(value, dict):
            return {k: move_to_cpu(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(move_to_cpu(item) for item in value)
        else:
            # Primitives (str, int, float, etc.) - return as-is
            return value

    return [move_to_cpu(batch_dict) for batch_dict in outputs]
