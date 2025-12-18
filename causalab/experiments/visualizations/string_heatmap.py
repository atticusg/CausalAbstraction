"""
String-based heatmap visualization functions.

These functions display text/tokens in heatmap cells rather than numeric scores.
"""

import os
from collections import Counter
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import TokenPosition


def plot_residual_stream_intervention_heatmap(
    intervention_results: Dict[tuple[Any, ...], Dict[str, Any]],
    prompt: str,
    layers: List[int],
    token_positions: List[TokenPosition],
    pipeline: LMPipeline,
    correct_answer: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_scores: bool = True,
    color_by_frequency: bool = False,
) -> None:
    """
    Create heatmap showing output tokens after interventions at each (layer, position).

    Args:
        intervention_results: Dict with (layer, token_pos_id) keys and intervention outputs.
        prompt: Original prompt text to extract token labels.
        layers: List of layer indices that were intervened on.
        token_positions: List of TokenPosition objects used in interventions.
        pipeline: LMPipeline object for token decoding.
        correct_answer: Optional correct answer to highlight matches.
        title: Optional custom title.
        save_path: Optional path to save the figure.
        show_scores: If True, color-code by correctness; if False, white background.
        color_by_frequency: If True, color cells by output token frequency (top 5 get
            unique colors). This mode is useful for tracing experiments where you want
            to see which outputs are most common. Overrides show_scores when True.
    """
    # Extract actual tokens from prompt for x-axis labels
    loaded = pipeline.load(prompt)
    token_ids = loaded["input_ids"][0]
    token_labels = []
    for i, token_pos in enumerate(token_positions):
        # Get the token index from the position (returns list[int] when batch=False)
        token_idx_result = token_pos.index(prompt)
        # Extract first index - index() returns list[int] for single input
        first_item = token_idx_result[0] if token_idx_result else 0
        # Handle nested list case (batch mode returns list[list[int]])
        token_idx: int = first_item[0] if isinstance(first_item, list) else first_item

        # Decode the token at this position
        if token_idx < len(token_ids):
            token_str = pipeline.tokenizer.decode([token_ids[token_idx]])
            # Clean up for display
            token_str = token_str.strip().replace("\n", "\\n")
            if len(token_str) > 15:
                token_str = token_str[:12] + "..."
            token_labels.append(f"{token_idx}: {token_str}")
        else:
            token_labels.append(f"{token_idx}: ???")

    # Create matrices for visualization
    num_layers = len(layers)
    num_positions = len(token_positions)
    token_matrix = np.empty((num_layers, num_positions), dtype=object)
    score_matrix = np.zeros((num_layers, num_positions))

    # Fill matrices with intervention results
    for layer_idx, layer in enumerate(layers):
        for pos_idx, token_pos in enumerate(token_positions):
            key = (layer, token_pos.id)
            if key in intervention_results:
                result = intervention_results[key]

                # Extract output token from string results (first item of batch)
                output_token = ""
                if "string" in result and result["string"]:
                    output_token = (
                        result["string"][0].strip() if result["string"] else ""
                    )

                # Store the output token
                token_matrix[layer_idx, pos_idx] = output_token if output_token else "∅"

                # Set score based on correctness
                if correct_answer and output_token == correct_answer:
                    score_matrix[layer_idx, pos_idx] = 1.0
                else:
                    score_matrix[layer_idx, pos_idx] = 0.0
            else:
                token_matrix[layer_idx, pos_idx] = "?"
                score_matrix[layer_idx, pos_idx] = 0.5  # Gray for missing data

    # Flip matrices vertically so layer -1 is at the bottom
    token_matrix = np.flipud(token_matrix)
    score_matrix = np.flipud(score_matrix)

    # Build frequency-based color mapping if requested
    output_color_map: Dict[str, str] = {}
    if color_by_frequency:
        # Count output frequencies across all cells
        output_counter: Counter[str] = Counter()
        for row in token_matrix:
            for token in row:
                if token and token not in ("?", "∅"):
                    output_counter[str(token)] += 1

        # Define light colors for the top 5 most frequent outputs
        light_colors = [
            "#FFFFE0",  # Light yellow
            "#FFE4E1",  # Light red/pink
            "#E0FFE0",  # Light green
            "#E0E0FF",  # Light blue
            "#F5DEB3",  # Light brown/wheat
        ]

        # Get top 5 most frequent outputs and create color mapping
        top_outputs = [output for output, _ in output_counter.most_common(5)]
        output_color_map = {
            output: light_colors[i] for i, output in enumerate(top_outputs)
        }

    # Create the figure - add extra width for legend when using frequency coloring
    fig_width = max(14, num_positions * 1.2)
    if color_by_frequency and output_color_map:
        fig_width += 3  # Extra space for legend
    _fig, ax = plt.subplots(figsize=(fig_width, max(8, num_layers * 0.5)))

    # Create heatmap background
    im = None  # Initialize to satisfy type checker
    if color_by_frequency:
        # Frequency coloring mode: create a color matrix based on token frequencies
        color_matrix = np.zeros((num_layers, num_positions, 3))
        for i in range(num_layers):
            for j in range(num_positions):
                token = str(token_matrix[i, j])
                if token in output_color_map:
                    # Convert hex color to RGB
                    hex_color = output_color_map[token]
                    color_matrix[i, j] = [
                        int(hex_color[1:3], 16) / 255,
                        int(hex_color[3:5], 16) / 255,
                        int(hex_color[5:7], 16) / 255,
                    ]
                else:
                    # Default white for non-top outputs
                    color_matrix[i, j] = [1.0, 1.0, 1.0]
        ax.imshow(color_matrix, aspect="auto")
    elif show_scores:
        # Color-coded mode: show correctness
        if correct_answer:
            # Use a colormap with good contrast: blue for wrong, red for correct
            im = ax.imshow(score_matrix, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
        else:
            # If no correct answer, use a light neutral background
            im = ax.imshow(
                np.ones_like(score_matrix) * 0.9,
                cmap="gray_r",
                aspect="auto",
                vmin=0,
                vmax=1,
            )
    else:
        # White background mode: just show tokens
        # Use 0.0 with gray_r to get white, or use Greys colormap
        im = ax.imshow(
            np.zeros_like(score_matrix), cmap="Greys", aspect="auto", vmin=0, vmax=1
        )

    # Add text annotations showing the predicted tokens
    for i in range(num_layers):
        for j in range(num_positions):
            token = token_matrix[i, j]
            if token:
                if color_by_frequency:
                    # Frequency coloring mode: always black text on colored background
                    ax.text(
                        j,
                        i,
                        str(token),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=11,
                        fontweight="bold"
                        if str(token) in output_color_map
                        else "normal",
                    )
                elif show_scores:
                    # Color-coded text based on correctness
                    if correct_answer and token == correct_answer:
                        # White text on red background (correct answers)
                        ax.text(
                            j,
                            i,
                            str(token),
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=11,
                            fontweight="bold",
                        )
                    else:
                        # Dark text on blue/gray background (incorrect/missing)
                        text_color = "black" if score_matrix[i, j] < 0.5 else "white"
                        ax.text(
                            j,
                            i,
                            str(token),
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=10,
                        )
                else:
                    # White background mode: always black text
                    ax.text(
                        j,
                        i,
                        str(token),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=11,
                    )

    # Set ticks and labels
    ax.set_xticks(np.arange(num_positions))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)

    # Reverse y-axis labels since we flipped the matrix
    y_labels = [f"L{layer}" if layer >= 0 else "Embed" for layer in layers]
    ax.set_yticklabels(reversed(y_labels))

    # Add colorbar or legend
    if color_by_frequency and output_color_map:
        # Add legend for frequency-based coloring
        legend_patches = []
        for output, color in output_color_map.items():
            # Truncate long labels
            label = output if len(output) <= 15 else output[:12] + "..."
            # Show quotes for whitespace-only strings
            if not output.strip():
                label = f'"{output}"'
            legend_patches.append(mpatches.Patch(color=color, label=label))

        # Add "Other" entry for non-top outputs
        legend_patches.append(mpatches.Patch(color="white", label="Other", ec="gray"))

        ax.legend(
            handles=legend_patches,
            title="Top Outputs",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            title_fontsize=10,
        )
    elif show_scores and im is not None:
        if correct_answer:
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1])
            cbar.set_label(f'Output = "{correct_answer}"', rotation=270, labelpad=20)
            cbar.ax.set_yticklabels(["Incorrect", "Missing", "Correct"])
        else:
            plt.colorbar(im, ax=ax)

    # Set title and labels
    if title is None:
        title = "Residual Stream Intervention: Output Tokens by Layer and Position"
    if show_scores and correct_answer:
        title += f'\n(Correct Answer: "{correct_answer}")'

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)

    # Add grid for clarity
    ax.set_xticks(np.arange(num_positions + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_layers + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.show()
    plt.close()
