"""
Attention Pattern Analysis

This module provides functions to extract and analyze attention patterns from
language models. It supports extracting attention weights for specific attention
heads across prompts and token positions.

Key Features:
1. Extract attention patterns for specific layer/head combinations
2. Support for batch processing of multiple prompts
3. Optional token position filtering (returns all positions if not specified)
4. Returns attention matrices per prompt and token position

Usage Example:
==============
```python
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.methods.attention_pattern_analysis import get_attention_patterns

# Setup pipeline
pipeline = LMPipeline("Qwen/Qwen3-14B", max_new_tokens=4)

# Define prompts
prompts = [
    {"raw_input": "The capital of France is"},
    {"raw_input": "The sum of 2 and 3 is"},
]

# Extract attention patterns for layer 10, head 5
patterns = get_attention_patterns(
    pipeline=pipeline,
    layer=10,
    head=5,
    prompts=prompts,
    token_positions=None,  # Returns all token positions
)

# Or specify specific token positions
from causalab.neural.token_positions import get_all_tokens

token_pos = get_all_tokens(prompts[0], pipeline)
patterns = get_attention_patterns(
    pipeline=pipeline,
    layer=10,
    head=5,
    prompts=prompts,
    token_positions=token_pos,
)
```
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from causalab.causal.trace import CausalTrace
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition

logger = logging.getLogger(__name__)


def get_attention_patterns(
    pipeline: LMPipeline,
    layer: int,
    head: int,
    prompts: List[CausalTrace],
    token_positions: Optional[Union[TokenPosition, List[int]]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract attention patterns for a specific attention head across prompts.

    This function extracts the attention weights from a specified layer and head
    for each prompt. If token_positions is provided, only the attention patterns
    for those positions are returned; otherwise, all token positions are included.

    Args:
        pipeline: LMPipeline with the model to extract attention from
        layer: Layer index (0-indexed)
        head: Head index (0-indexed)
        prompts: List of CausalTrace dicts, each with 'raw_input' key
        token_positions: Optional token positions to filter results.
            Can be:
            - None: Return attention for all token positions (default)
            - TokenPosition: Dynamic indexer that returns positions per prompt
            - List[int]: Fixed list of token indices to use for all prompts
        verbose: Whether to print progress information (default: False)

    Returns:
        List of dicts, one per prompt, each containing:
            - 'prompt': The original prompt dict
            - 'layer': Layer index
            - 'head': Head index
            - 'attention_pattern': Full attention matrix (seq_len x seq_len) as numpy array
            - 'token_positions': List of token positions included
            - 'filtered_attention': Attention rows for specified token positions
                                   (seq_len,) for each position if token_positions specified,
                                   otherwise same as attention_pattern

    Raises:
        ValueError: If layer or head index is out of bounds
    """
    # Validate layer and head indices
    num_layers = pipeline.model.config.num_hidden_layers
    num_heads = pipeline.model.config.num_attention_heads

    if layer < 0 or layer >= num_layers:
        raise ValueError(
            f"Layer index {layer} out of range. Valid range: 0-{num_layers - 1}"
        )
    if head < 0 or head >= num_heads:
        raise ValueError(
            f"Head index {head} out of range. Valid range: 0-{num_heads - 1}"
        )

    if verbose:
        logger.info(f"Extracting attention patterns for Layer {layer}, Head {head}")
        logger.info(f"Processing {len(prompts)} prompts")

    results = []

    # Process prompts
    for prompt in prompts:
        # Tokenize input (load expects a list)
        encoded = pipeline.load([prompt])

        # Forward pass with attention output
        with torch.no_grad():
            outputs = pipeline.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_attentions=True,
            )

        # Get attention from specified layer
        # Shape: (batch=1, num_heads, seq_len, seq_len)
        layer_attention = outputs.attentions[layer]

        # Extract specific head: (seq_len, seq_len)
        head_attention = layer_attention[0, head].cpu().float().numpy()

        # Get non-pad token mask
        attention_mask = encoded["attention_mask"][0].cpu().numpy().astype(bool)
        seq_len = int(attention_mask.sum())

        # Filter out pad tokens from attention pattern for visualization
        # Keep only rows and columns where mask is True (non-pad tokens)
        head_attention_filtered = head_attention[attention_mask][:, attention_mask]

        # Determine token positions
        if token_positions is None:
            # Return all non-pad token positions
            positions = list(range(seq_len))
            filtered_attention = head_attention_filtered
        elif isinstance(token_positions, TokenPosition):
            # Use dynamic indexer
            positions = token_positions.index(prompt)
            # Filter attention to only the specified positions (rows)
            filtered_attention = head_attention_filtered[positions, :]
        else:
            # Fixed list of positions
            positions = list(token_positions)
            # Filter attention to only the specified positions (rows)
            filtered_attention = head_attention_filtered[positions, :]

        result = {
            "prompt": prompt,
            "layer": layer,
            "head": head,
            "attention_pattern": head_attention_filtered,
            "attention_pattern_unfiltered": head_attention,  # Full pattern for TokenPosition indexing
            "token_positions": positions,
            "filtered_attention": filtered_attention,
            "seq_len": seq_len,
        }

        results.append(result)

    if verbose:
        logger.info(f"Extracted attention patterns for {len(results)} prompts")

    return results


def compute_average_attention(
    attention_results: List[Dict[str, Any]],
    ignore_first_token: bool = True,
) -> Dict[str, Any]:
    """
    Compute average attention pattern across multiple prompts of the same length.

    All prompts must have the same sequence length. This simplifies averaging
    and ensures consistent attention pattern dimensions.

    Args:
        attention_results: List of results from get_attention_patterns.
                          All results must have the same seq_len.
        ignore_first_token: If True, exclude the first token (attention sink)
                           from the averaged pattern. Default is True.

    Returns:
        Dict containing:
            - 'average_pattern': Averaged attention matrix
            - 'seq_len': Sequence length of samples (after removing first token if applicable)
            - 'num_samples': Number of samples averaged
            - 'ignored_first_token': Whether first token was ignored

    Raises:
        ValueError: If attention_results is empty or prompts have different lengths
    """
    if not attention_results:
        raise ValueError("attention_results cannot be empty")

    # Check all prompts have the same length
    seq_len = attention_results[0]["seq_len"]
    for i, result in enumerate(attention_results):
        if result["seq_len"] != seq_len:
            raise ValueError(
                f"All prompts must have the same sequence length. "
                f"Prompt 0 has length {seq_len}, but prompt {i} has length {result['seq_len']}"
            )

    # Stack attention patterns
    patterns = np.stack([r["attention_pattern"] for r in attention_results], axis=0)

    # Optionally ignore first token (attention sink)
    if ignore_first_token:
        patterns = patterns[:, 1:, 1:]
        seq_len = seq_len - 1

    avg_pattern = np.mean(patterns, axis=0)

    return {
        "average_pattern": avg_pattern,
        "seq_len": seq_len,
        "num_samples": len(attention_results),
        "ignored_first_token": ignore_first_token,
    }


def compute_average_attention_by_token_type(
    attention_results: List[Dict[str, Any]],
    source_positions: List[TokenPosition],
    target_positions: List[TokenPosition],
) -> Dict[str, Any]:
    """
    Compute average attention between specific token types across multiple prompts.

    This function computes the average attention from source token types (queries)
    to target token types (keys/values). Unlike compute_average_attention which
    requires all prompts to have the same length, this function handles variable-length
    prompts by averaging attention values rather than full matrices.

    Args:
        attention_results: List of results from get_attention_patterns.
        source_positions: List of TokenPosition objects for source positions (queries).
            These are the "FROM" positions.
        target_positions: List of TokenPosition objects for target positions (keys/values).
            These are the "TO" positions.

    Returns:
        Dict containing:
            - 'attention_matrix': 2D numpy array of shape (num_source_types, num_target_types)
                                 with average attention from each source type to each target type
            - 'source_ids': List of source position IDs (row labels)
            - 'target_ids': List of target position IDs (column labels)
            - 'num_samples': Number of samples averaged
            - 'std_matrix': Standard deviation across samples
            - 'per_sample_attention': List of per-sample attention matrices

    Example:
        >>> from causalab.neural.token_positions import TokenPosition
        >>> # Create token positions
        >>> last_tok = TokenPosition(lambda x: [-1], pipeline, id="last_token")
        >>> subj_pos = TokenPosition(lambda x: [2, 3], pipeline, id="subject")
        >>> obj_pos = TokenPosition(lambda x: [5], pipeline, id="object")
        >>>
        >>> result = compute_average_attention_by_token_type(
        ...     attention_results,
        ...     source_positions=[last_tok],
        ...     target_positions=[subj_pos, obj_pos],
        ... )
        >>> # result['attention_matrix'][0, 1] = avg attention from last_token to object
    """
    if not attention_results:
        raise ValueError("attention_results cannot be empty")

    source_ids = [pos.id for pos in source_positions]
    target_ids = [pos.id for pos in target_positions]

    # Collect attention values per sample
    per_sample_matrices = []

    for result in attention_results:
        prompt = result["prompt"]
        # Use unfiltered pattern if available (for correct TokenPosition indexing)
        # Fall back to filtered pattern for backwards compatibility
        pattern = result.get(
            "attention_pattern_unfiltered", result["attention_pattern"]
        )

        # Build attention matrix for this sample
        sample_matrix = np.zeros((len(source_positions), len(target_positions)))

        for i, src_pos in enumerate(source_positions):
            # Get source token indices for this prompt
            # index() returns list[int] for non-batch input
            src_indices: list[int] = src_pos.index(prompt)  # type: ignore[assignment]

            for j, tgt_pos in enumerate(target_positions):
                # Get target token indices for this prompt
                # index() returns list[int] for non-batch input
                tgt_indices: list[int] = tgt_pos.index(prompt)  # type: ignore[assignment]

                # Extract attention values from source to target
                # pattern[src, tgt] = attention from src (query) to tgt (key)
                seq_len = pattern.shape[0]
                attention_values = []
                for src_idx in src_indices:
                    for tgt_idx in tgt_indices:
                        # Skip if index is out of bounds
                        if src_idx >= seq_len or tgt_idx >= seq_len:
                            continue
                        # Only include if target is within causal mask (tgt <= src)
                        if tgt_idx <= src_idx:
                            attention_values.append(pattern[src_idx, tgt_idx])

                # Average attention from this source type to this target type
                if attention_values:
                    sample_matrix[i, j] = np.mean(attention_values)
                else:
                    sample_matrix[i, j] = np.nan  # No valid attention (causal mask)

        per_sample_matrices.append(sample_matrix)

    # Stack and compute statistics across samples
    stacked = np.stack(per_sample_matrices, axis=0)  # (num_samples, num_src, num_tgt)
    avg_matrix = np.nanmean(stacked, axis=0)  # (num_src, num_tgt)
    std_matrix = np.nanstd(stacked, axis=0)  # (num_src, num_tgt)

    return {
        "attention_matrix": avg_matrix,
        "source_ids": source_ids,
        "target_ids": target_ids,
        "num_samples": len(attention_results),
        "std_matrix": std_matrix,
        "per_sample_attention": per_sample_matrices,
    }


def analyze_attention_statistics(
    attention_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute statistics about attention patterns.

    Args:
        attention_results: List of results from get_attention_patterns

    Returns:
        Dict containing:
            - 'avg_entropy': Average entropy of attention distributions
            - 'avg_max_attention': Average maximum attention weight
            - 'avg_diagonal': Average attention to same position (self-attention)
            - 'avg_previous': Average attention to previous token
    """
    if not attention_results:
        raise ValueError("attention_results cannot be empty")

    entropies = []
    max_attentions = []
    diagonal_attentions = []
    previous_attentions = []

    for result in attention_results:
        pattern = result["attention_pattern"]
        seq_len = result["seq_len"]

        for i in range(seq_len):
            row = pattern[i, : i + 1]  # Only attend to positions up to i (causal)
            row = row + 1e-10  # Avoid log(0)

            # Entropy
            entropy = -np.sum(row * np.log2(row))
            entropies.append(entropy)

            # Max attention
            max_attentions.append(np.max(row))

            # Diagonal (self-attention)
            diagonal_attentions.append(pattern[i, i])

            # Previous token attention
            if i > 0:
                previous_attentions.append(pattern[i, i - 1])

    return {
        "avg_entropy": float(np.mean(entropies)),
        "avg_max_attention": float(np.mean(max_attentions)),
        "avg_diagonal": float(np.mean(diagonal_attentions)),
        "avg_previous": float(np.mean(previous_attentions))
        if previous_attentions
        else 0.0,
    }
