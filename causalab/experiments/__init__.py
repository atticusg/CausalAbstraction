"""Experiment implementations for causal analysis."""

# Import only what actually exists after refactoring
from .filter import filter_dataset

# Import visualization functions
from .visualizations import (
    # Score heatmaps
    plot_attention_head_heatmap,
    plot_residual_stream_heatmap,
    plot_score_heatmap,
    # Component type detection and grid extraction utilities
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
    # String heatmaps
    plot_residual_stream_intervention_heatmap,
    # Binary mask heatmaps (unified)
    plot_binary_mask,
    get_selected_units,
    plot_attention_head_mask,
    plot_residual_stream_mask,
    plot_mlp_mask,
    get_selected_heads,
    get_selected_residual_positions,
    get_selected_mlps,
    # Unit ID parsing helpers
    extract_layer_head_from_unit_id,
    # Text analysis functions
    print_residual_stream_patching_analysis,
)

__all__ = [
    # filter module
    "filter_dataset",
    # Score heatmaps
    "plot_attention_head_heatmap",
    "plot_residual_stream_heatmap",
    "plot_score_heatmap",
    # Component type detection and grid extraction utilities
    "detect_component_type_from_targets",
    "extract_grid_dimensions_from_targets",
    # String heatmaps
    "plot_residual_stream_intervention_heatmap",
    # Binary mask heatmaps (unified)
    "plot_binary_mask",
    "get_selected_units",
    "plot_attention_head_mask",
    "plot_residual_stream_mask",
    "plot_mlp_mask",
    "get_selected_heads",
    "get_selected_residual_positions",
    "get_selected_mlps",
    # Unit ID parsing helpers
    "extract_layer_head_from_unit_id",
    # Text analysis functions
    "print_residual_stream_patching_analysis",
]
