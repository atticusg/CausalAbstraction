from .score_heatmap import (
    plot_attention_head_heatmap,
    plot_residual_stream_heatmap,
    plot_score_heatmap,
    plot_variable_localization_heatmap,
)
from causalab.neural.activations.targets import (
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
)
from .string_heatmap import (
    plot_residual_stream_intervention_heatmap,
    plot_single_pair_trace_heatmap,
    build_token_labels,
)
from .binary_mask import (
    # Unified dispatchers
    plot_binary_mask,
    get_selected_units,
    # Component-specific binary mask functions
    plot_attention_head_mask,
    plot_residual_stream_mask,
    plot_mlp_mask,
    # Selection extractors
    get_selected_heads,
    get_selected_residual_positions,
    get_selected_mlps,
)
from .unit_id import (
    # Unit ID parsing helpers (shared by binary_mask and feature_masks)
    extract_layer_from_unit_id,
    extract_token_position_from_unit_id,
    extract_layer_head_from_unit_id,
    detect_component_type,
    is_per_layer_mode,
    extract_grid_dimensions,
)
from .feature_masks import (
    # Feature count plotting (DBM with tie_masks=False)
    plot_feature_counts,
    plot_attention_head_feature_counts,
    plot_residual_stream_feature_counts,
    plot_mlp_feature_counts,
    count_selected_features,
    NFeatures,
)
from .text_analysis import (
    print_residual_stream_patching_analysis,
)
from .pca_scatter import (
    plot_pca_scatter,
    plot_features_2d,
)
from .mds import mds_embed
from .distance_plots import plot_distance_scatter, plot_dual_mds
from .figure_format import (
    ALLOWED_FIGURE_FORMATS,
    FigureFormat,
    normalize_figure_format,
    path_with_figure_format,
    resolve_figure_format_from_analysis,
)

__all__ = [
    # Score heatmaps
    "plot_attention_head_heatmap",
    "plot_residual_stream_heatmap",
    "plot_score_heatmap",
    "plot_variable_localization_heatmap",
    # Component type detection and grid extraction utilities
    "detect_component_type_from_targets",
    "extract_grid_dimensions_from_targets",
    # String heatmaps
    "plot_residual_stream_intervention_heatmap",
    "plot_single_pair_trace_heatmap",
    "build_token_labels",
    # Binary mask heatmaps (unified)
    "plot_binary_mask",
    "get_selected_units",
    "plot_attention_head_mask",
    "plot_residual_stream_mask",
    "plot_mlp_mask",
    "get_selected_heads",
    "get_selected_residual_positions",
    "get_selected_mlps",
    # Unit ID parsing helpers (shared)
    "extract_layer_from_unit_id",
    "extract_token_position_from_unit_id",
    "extract_layer_head_from_unit_id",
    "detect_component_type",
    "is_per_layer_mode",
    "extract_grid_dimensions",
    # Feature count plotting (DBM with tie_masks=False)
    "plot_feature_counts",
    "plot_attention_head_feature_counts",
    "plot_residual_stream_feature_counts",
    "plot_mlp_feature_counts",
    "count_selected_features",
    "NFeatures",
    # Text analysis functions
    "print_residual_stream_patching_analysis",
    # PCA scatter plots
    "plot_pca_scatter",
    "plot_features_2d",
    # MDS and distance plots
    "mds_embed",
    "plot_distance_scatter",
    "plot_dual_mds",
    # Figure output format (PNG / PDF)
    "ALLOWED_FIGURE_FORMATS",
    "FigureFormat",
    "normalize_figure_format",
    "path_with_figure_format",
    "resolve_figure_format_from_analysis",
]
