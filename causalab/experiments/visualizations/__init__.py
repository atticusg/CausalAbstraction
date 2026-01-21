from .score_heatmap import (
    plot_attention_head_heatmap,
    plot_residual_stream_heatmap,
    plot_score_heatmap,
)
from causalab.experiments.interchange_targets import (
    detect_component_type_from_targets,
    extract_grid_dimensions_from_targets,
)
from .string_heatmap import (
    plot_residual_stream_intervention_heatmap,
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
)

__all__ = [
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
]
