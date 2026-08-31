from .score_heatmap import (
    plot_attention_head_heatmap,
    plot_residual_stream_heatmap,
    plot_variable_localization_heatmap,
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
from .grid_cells import (
    GridCell,
    cell_grid_dimensions,
)
from .feature_masks import (
    # Feature count plotting (DBM with tie_masks=False)
    plot_feature_counts,
    plot_attention_head_feature_counts,
    plot_residual_stream_feature_counts,
    plot_mlp_feature_counts,
)
from .text_analysis import (
    print_residual_stream_patching_analysis,
)
from .pca_scatter import (
    plot_pca_scatter,
    plot_features_2d,
)
from .receptive_field import (
    build_receptive_field_figure,
    plot_receptive_field,
)
from .mds import mds_embed
from .distance_plots import plot_distance_scatter, plot_dual_mds
from .figure_format import (
    ALLOWED_FIGURE_FORMATS,
    FigureFormat,
    normalize_figure_format,
    path_with_figure_format,
)
from .causal_graph import (
    DEFAULT_COLORS,
    build_forward_pass_app,
    build_interchange_app,
    build_setting_figure,
    build_structure_app,
    build_structure_figure,
    display_forward_pass,
    display_interchange,
    display_structure,
    print_setting,
    print_structure,
)

__all__ = [
    # Score heatmaps
    "plot_attention_head_heatmap",
    "plot_residual_stream_heatmap",
    "plot_variable_localization_heatmap",
    # Binary mask heatmaps (unified)
    "plot_binary_mask",
    "get_selected_units",
    "plot_attention_head_mask",
    "plot_residual_stream_mask",
    "plot_mlp_mask",
    "get_selected_heads",
    "get_selected_residual_positions",
    "get_selected_mlps",
    # Structured grid-cell records
    "GridCell",
    "cell_grid_dimensions",
    # Feature count plotting (DBM with tie_masks=False)
    "plot_feature_counts",
    "plot_attention_head_feature_counts",
    "plot_residual_stream_feature_counts",
    "plot_mlp_feature_counts",
    # Text analysis functions
    "print_residual_stream_patching_analysis",
    # PCA scatter plots
    "plot_pca_scatter",
    "plot_features_2d",
    # Receptive-field decision map
    "build_receptive_field_figure",
    "plot_receptive_field",
    # MDS and distance plots
    "mds_embed",
    "plot_distance_scatter",
    "plot_dual_mds",
    # Figure output format (PNG / PDF)
    "ALLOWED_FIGURE_FORMATS",
    "FigureFormat",
    "normalize_figure_format",
    "path_with_figure_format",
    # Causal graph visualization (Dash + matplotlib)
    "DEFAULT_COLORS",
    "build_forward_pass_app",
    "build_interchange_app",
    "build_setting_figure",
    "build_structure_app",
    "build_structure_figure",
    "display_forward_pass",
    "display_interchange",
    "display_structure",
    "print_setting",
    "print_structure",
]
