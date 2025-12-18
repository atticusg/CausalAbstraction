"""
Text-based analysis and output functions for intervention results.

These functions produce text-based output for intervention results, useful for
console output, log files, and non-graphical environments.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _find_contiguous_ranges(layers: List[int]) -> List[Tuple[int, int]]:
    """
    Find contiguous ranges in a sorted list of layer numbers.

    Args:
        layers: Sorted list of layer numbers

    Returns:
        List of (start, end) tuples representing contiguous ranges

    Example:
        [1, 2, 3, 5, 6, 8] -> [(1, 3), (5, 6), (8, 8)]
    """
    if not layers:
        return []

    ranges = []
    start = layers[0]
    prev = layers[0]

    for layer in layers[1:]:
        if layer == prev + 1:
            # Continue the range
            prev = layer
        else:
            # End the current range and start a new one
            ranges.append((start, prev))
            start = layer
            prev = layer

    # Add the final range
    ranges.append((start, prev))

    return ranges


def _save_text_output(save_path: str, content: str) -> None:
    """Save text content to a file, creating directories as needed."""
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(content)
    print(f"\nText analysis saved to: {save_path}")


def print_residual_stream_patching_analysis(
    scores: Dict[Tuple[int, Any], float],
    layers: List[int],
    token_position_ids: List[Any],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Print a detailed text-based analysis of intervention accuracy scores using region-based breakdown.

    This function identifies distinct regions in the score matrix (groups of cells with similar scores)
    and reports statistics for each region, capturing both high AND low accuracy patterns.

    Args:
        scores: Dict mapping (layer, position_id) tuples to scores (0.0 to 1.0).
        layers: List of layer indices.
        token_position_ids: List of position IDs.
        title: Optional custom title for the analysis.
        save_path: Optional path to save the text output to a file.

    Returns:
        The full text output as a string.
    """
    output_lines = []
    output_lines.append("=" * 80)
    if title:
        output_lines.append(title.upper())
    else:
        output_lines.append("ACTIVATION PATCHING TEXT ANALYSIS")
    output_lines.append("=" * 80)

    # Build score matrix
    score_matrix = np.full((len(layers), len(token_position_ids)), np.nan)
    for (layer, pos_id), score in scores.items():
        if layer in layers and pos_id in token_position_ids:
            layer_idx = layers.index(layer)
            pos_idx = token_position_ids.index(pos_id)
            score_matrix[layer_idx, pos_idx] = score

    # Check if we have valid data
    if np.all(np.isnan(score_matrix)):
        output_lines.append("\n  No data available")
        full_output = "\n".join(output_lines)
        print(full_output)
        if save_path:
            _save_text_output(save_path, full_output)
        return full_output

    # Identify regions using clustering on score values
    output_lines.append("\nRegion Breakdown (using natural clustering):")

    # Get unique score values (excluding NaN) and cluster them
    flat_scores = score_matrix[~np.isnan(score_matrix)].flatten()
    if len(flat_scores) == 0:
        output_lines.append("  No valid scores found")
        full_output = "\n".join(output_lines)
        print(full_output)
        if save_path:
            _save_text_output(save_path, full_output)
        return full_output

    unique_scores = np.unique(flat_scores)

    # Use simple thresholding to identify clusters:
    # Group scores that are within 10% of each other
    clusters: List[List[float]] = []
    for score in sorted(unique_scores):
        # Check if this score belongs to an existing cluster
        added = False
        for cluster in clusters:
            if abs(score - np.mean(cluster)) <= 0.10:
                cluster.append(float(score))
                added = True
                break
        if not added:
            clusters.append([float(score)])

    # Merge small clusters that are close together
    merged_clusters: List[List[float]] = []
    for cluster in clusters:
        if (
            merged_clusters
            and abs(np.mean(cluster) - np.mean(merged_clusters[-1])) <= 0.15
        ):
            merged_clusters[-1].extend(cluster)
        else:
            merged_clusters.append(cluster)

    # For each cluster, create a region
    for cluster_idx, cluster in enumerate(merged_clusters):
        cluster_min = min(cluster)
        cluster_max = max(cluster)

        # Find all cells in this score range
        mask = (score_matrix >= cluster_min - 0.01) & (
            score_matrix <= cluster_max + 0.01
        )
        mask = mask & ~np.isnan(score_matrix)  # Exclude NaN values

        if np.sum(mask) == 0:
            continue

        region_scores = score_matrix[mask]
        region_mean = float(region_scores.mean())
        region_std = float(region_scores.std())
        region_size = int(np.sum(mask))
        total_valid = int(np.sum(~np.isnan(score_matrix)))

        # Collect actual (layer, position) pairs in this region
        region_pairs: List[Tuple[int, Any, float]] = []
        for idx in np.ndindex(score_matrix.shape):
            if mask[idx]:
                layer = layers[idx[0]]
                pos = token_position_ids[idx[1]]
                score = float(score_matrix[idx])
                region_pairs.append((layer, pos, score))

        # Sort by score descending
        region_pairs.sort(key=lambda x: x[2], reverse=True)

        # Report region
        output_lines.append(
            f"\n  Region {cluster_idx + 1}: {cluster_min:.0%}-{cluster_max:.0%}"
        )
        output_lines.append(f"    Mean: {region_mean:.1%} +/- {region_std:.1%}")
        output_lines.append(
            f"    Size: {region_size} cells ({region_size / total_valid * 100:.1f}%)"
        )

        # Show actual (layer, position) pairs
        if region_size <= 20:
            # Small region - show all pairs
            output_lines.append("    Cells:")
            for layer, pos, score in region_pairs:
                output_lines.append(f"      L{layer:2d} @ {str(pos):15s} ({score:.1%})")
        else:
            # Large region - show contiguous layer ranges with statistics per position
            output_lines.append("    Contiguous layer ranges by position:")

            # Group pairs by position, keeping scores
            position_to_layers_scores: Dict[Any, List[Tuple[int, float]]] = {}
            for layer, pos, score in region_pairs:
                if pos not in position_to_layers_scores:
                    position_to_layers_scores[pos] = []
                position_to_layers_scores[pos].append((layer, score))

            # For each position, find contiguous ranges with stats
            for pos in sorted(position_to_layers_scores.keys(), key=str):
                layers_scores = sorted(
                    position_to_layers_scores[pos], key=lambda x: x[0]
                )
                layers_only = [layer for layer, _score in layers_scores]
                ranges = _find_contiguous_ranges(layers_only)

                # For each range, compute statistics
                range_strs = []
                for start, end in ranges:
                    # Get layers and scores for this range
                    range_data = [
                        (layer, score)
                        for layer, score in layers_scores
                        if start <= layer <= end
                    ]
                    range_scores_list = [score for _, score in range_data]
                    mean_score = np.mean(range_scores_list)
                    std_score = np.std(range_scores_list)

                    # Find min and max with their layers
                    min_idx = int(np.argmin(range_scores_list))
                    max_idx = int(np.argmax(range_scores_list))
                    min_layer, min_score = range_data[min_idx]
                    max_layer, max_score = range_data[max_idx]

                    if start == end:
                        range_str = f"L{start} ({mean_score:.1%})"
                    else:
                        range_str = (
                            f"L{start}-L{end} (mu={mean_score:.1%}+/-{std_score:.1%}, "
                            f"min={min_score:.1%}@L{min_layer}, max={max_score:.1%}@L{max_layer})"
                        )
                    range_strs.append(range_str)

                output_lines.append(f"      {str(pos):15s}: {', '.join(range_strs)}")

    output_lines.append("\n" + "=" * 80)
    output_lines.append("ANALYSIS COMPLETE")
    output_lines.append("=" * 80)

    # Join all lines
    full_output = "\n".join(output_lines)

    # Print to console
    print(full_output)

    # Optionally save to file
    if save_path:
        _save_text_output(save_path, full_output)

    return full_output
