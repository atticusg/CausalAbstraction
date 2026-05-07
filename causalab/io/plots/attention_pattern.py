"""
Visualization functions for attention pattern heatmaps.

These visualizations display attention weights from transformer attention heads,
showing where each query position attends to in the key positions.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from causalab.io.plots.figure_format import path_with_figure_format
import numpy as np
from numpy.typing import NDArray


def plot_attention_heatmap(
    attention_pattern: NDArray[np.floating[Any]],
    title: str = "Attention Pattern",
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    ignore_first_token: bool = True,
    pad_token: Optional[str] = None,
    figsize: tuple[float, float] = (12, 10),
    cmap: str = "Blues",
    show_colorbar: bool = True,
    xlabel: str = "Key Position (attending to)",
    ylabel: str = "Query Position (attending from)",
    figure_format: str = "pdf",
) -> None:
    """
    Plot attention pattern as a heatmap.

    Args:
        attention_pattern: 2D attention matrix (seq_len x seq_len)
        title: Plot title
        tokens: Optional list of token strings for axis labels
        save_path: Optional path to save figure
        ignore_first_token: If True, exclude the first token (attention sink)
                           from visualization. Default is True.
        pad_token: Optional pad token string to filter out from tokens list.
                  Common values: "<pad>", "<|pad|>", "[PAD]". If provided,
                  tokens matching this value will be removed from display.
        figsize: Figure size as (width, height)
        cmap: Colormap to use (default: "Blues")
        show_colorbar: Whether to show the colorbar (default: True)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figure_format: ``png`` or ``pdf`` for static output.
    """
    pattern = attention_pattern.copy()

    # Filter out pad tokens from the tokens list
    if tokens is not None and pad_token is not None:
        tokens = [t for t in tokens if t != pad_token]

    # Optionally ignore the first token (attention sink)
    if ignore_first_token:
        pattern = pattern[1:, 1:]
        if tokens is not None:
            tokens = tokens[1:]

    _fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pattern, cmap=cmap, aspect="auto")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight")

    # Add token labels if provided
    if tokens is not None:
        display_tokens = tokens[: pattern.shape[0]]
        ax.set_xticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(len(display_tokens)))
        ax.set_yticklabels(display_tokens, fontsize=8)

    plt.tight_layout()

    if save_path:
        out = path_with_figure_format(save_path, figure_format)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_attention_comparison(
    attention_patterns: List[NDArray[np.floating[Any]]],
    labels: List[str],
    title: str = "Attention Pattern Comparison",
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    ignore_first_token: bool = True,
    pad_token: Optional[str] = None,
    max_tokens: Optional[int] = 40,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "Blues",
    ncols: int = 2,
    figure_format: str = "pdf",
) -> None:
    """
    Plot multiple attention patterns side by side for comparison.

    Args:
        attention_patterns: List of 2D attention matrices to compare
        labels: List of labels for each pattern (e.g., "Layer 10, Head 5")
        title: Overall figure title
        tokens: Optional list of token strings for axis labels
        save_path: Optional path to save figure
        ignore_first_token: If True, exclude the first token (attention sink)
                           from visualization. Default is True.
        pad_token: Optional pad token string to filter out from tokens list.
                  Common values: "<pad>", "<|pad|>", "[PAD]". If provided,
                  tokens matching this value will be removed from display.
        max_tokens: Maximum number of tokens to display (None = show all)
        figsize: Figure size. If None, auto-calculated based on number of patterns.
        cmap: Colormap to use (default: "Blues")
        ncols: Number of columns in the grid (default: 2)
        figure_format: ``png`` or ``pdf`` for static output.
    """
    n_patterns = len(attention_patterns)
    if n_patterns != len(labels):
        raise ValueError("Number of patterns must match number of labels")

    # Filter out pad tokens from the tokens list
    if tokens is not None and pad_token is not None:
        tokens = [t for t in tokens if t != pad_token]

    nrows = (n_patterns + ncols - 1) // ncols

    if figsize is None:
        figsize = (7 * ncols, 6 * nrows)

    _fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for idx, (pattern, label) in enumerate(zip(attention_patterns, labels)):
        ax = axes[idx]
        plot_pattern = pattern.copy()

        # Optionally ignore first token
        if ignore_first_token:
            plot_pattern = plot_pattern[1:, 1:]
            plot_tokens = tokens[1:] if tokens is not None else None
        else:
            plot_tokens = tokens

        # Limit tokens if specified
        if max_tokens is not None:
            plot_pattern = plot_pattern[:max_tokens, :max_tokens]
            if plot_tokens is not None:
                plot_tokens = plot_tokens[:max_tokens]

        im = ax.imshow(plot_pattern, cmap=cmap, aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.colorbar(im, ax=ax)

        # Add token labels if provided and not too many
        if plot_tokens is not None and len(plot_tokens) <= 30:
            ax.set_xticks(range(len(plot_tokens)))
            ax.set_xticklabels(plot_tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(len(plot_tokens)))
            ax.set_yticklabels(plot_tokens, fontsize=6)

    # Hide unused subplots
    for idx in range(n_patterns, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        out = path_with_figure_format(save_path, figure_format)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_attention_statistics(
    statistics: Dict[str, float],
    title: str = "Attention Pattern Statistics",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
    figure_format: str = "pdf",
) -> None:
    """
    Plot attention pattern statistics as a bar chart.

    Args:
        statistics: Dict containing statistics from analyze_attention_statistics.
                   Expected keys: avg_entropy, avg_max_attention, avg_diagonal, avg_previous
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size as (width, height)
        figure_format: ``png`` or ``pdf`` for static output.
    """
    # Define display names and descriptions
    stat_info = {
        "avg_entropy": ("Entropy (bits)", "Higher = more distributed"),
        "avg_max_attention": ("Max Attention", "Higher = more focused"),
        "avg_diagonal": ("Self-Attention", "Attention to current position"),
        "avg_previous": ("Previous Token", "Attention to preceding token"),
    }

    # Filter to only include known statistics
    filtered_stats = {k: v for k, v in statistics.items() if k in stat_info}

    _fig, ax = plt.subplots(figsize=figsize)

    names = [stat_info[k][0] for k in filtered_stats.keys()]
    values = list(filtered_stats.values())
    descriptions = [stat_info[k][1] for k in filtered_stats.keys()]

    bars = ax.bar(names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    ax.set_ylabel("Value")
    ax.set_title(title)

    # Add value labels on bars
    for bar, desc in zip(bars, descriptions):
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}\n({desc})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        out = path_with_figure_format(save_path, figure_format)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_layer_head_attention_grid(
    attention_results: List[Dict[str, Any]],
    title: str = "Attention Patterns Across Layers and Heads",
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    ignore_first_token: bool = True,
    pad_token: Optional[str] = None,
    max_tokens: Optional[int] = 40,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "Blues",
    figure_format: str = "pdf",
) -> None:
    """
    Plot attention patterns organized by layer and head.

    This is a convenience wrapper around plot_attention_comparison that
    automatically generates labels from attention_results.

    Args:
        attention_results: List of results from get_attention_patterns,
                          each containing 'layer', 'head', and 'attention_pattern'
        title: Overall figure title
        tokens: Optional list of token strings for axis labels
        save_path: Optional path to save figure
        ignore_first_token: If True, exclude the first token (attention sink)
        pad_token: Optional pad token string to filter out from tokens list.
                  Common values: "<pad>", "<|pad|>", "[PAD]". If provided,
                  tokens matching this value will be removed from display.
        max_tokens: Maximum number of tokens to display
        figsize: Figure size
        cmap: Colormap to use
        figure_format: ``png`` or ``pdf`` for static output.
    """
    patterns = [r["attention_pattern"] for r in attention_results]
    labels = [f"Layer {r['layer']}, Head {r['head']}" for r in attention_results]

    plot_attention_comparison(
        attention_patterns=patterns,
        labels=labels,
        title=title,
        tokens=tokens,
        save_path=save_path,
        ignore_first_token=ignore_first_token,
        pad_token=pad_token,
        max_tokens=max_tokens,
        figsize=figsize,
        cmap=cmap,
        figure_format=figure_format,
    )


def plot_token_type_attention_heatmap(
    attention_matrix: NDArray[np.floating[Any]],
    source_ids: List[str],
    target_ids: List[str],
    title: str = "Average Attention by Token Type",
    save_path: Optional[str] = None,
    std_matrix: Optional[NDArray[np.floating[Any]]] = None,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "Blues",
    xlabel: str = "Target Token Type (TO)",
    ylabel: str = "Source Token Type (FROM)",
    figure_format: str = "pdf",
) -> None:
    """Plot a source-type x target-type attention matrix as a heatmap.

    Args:
        attention_matrix: 2D array (num_source_types, num_target_types).
        source_ids: Row labels (FROM positions).
        target_ids: Column labels (TO positions).
        title: Plot title.
        save_path: Optional path to save figure.
        std_matrix: Optional std matrix; when provided, cell annotations
            show ``mean ± std``.
        figsize: Figure size. Auto-calculated if None.
        cmap: Matplotlib colormap name.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figure_format: ``png`` or ``pdf`` for static output.
    """
    n_rows, n_cols = attention_matrix.shape
    if figsize is None:
        figsize = (max(8, 1.5 * n_cols), max(4, 1.2 * n_rows))

    if std_matrix is not None:
        annot_array = np.empty_like(attention_matrix, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                annot_array[i, j] = (
                    f"{attention_matrix[i, j]:.3f}\n±{std_matrix[i, j]:.3f}"
                )
        annot: Any = annot_array
        fmt = ""
    else:
        annot = True
        fmt = ".3f"

    _fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        attention_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=target_ids,
        yticklabels=source_ids,
        ax=ax,
        cbar_kws={"label": "Attention Weight"},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        out = path_with_figure_format(save_path, figure_format)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()
