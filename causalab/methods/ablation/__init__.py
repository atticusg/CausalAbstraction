"""Component ablation methods — zero/mean replacement + behavioral-drop scoring."""

from causalab.methods.ablation.reference_vectors import (
    make_mean_vectors,
    make_zero_vectors,
)
from causalab.methods.ablation.run import (
    run_ablation,
    run_ablation_combo,
    run_ablation_combo_multi,
    run_ablation_scan,
    run_ablation_scan_multi,
)

__all__ = [
    "make_mean_vectors",
    "make_zero_vectors",
    "run_ablation",
    "run_ablation_combo",
    "run_ablation_combo_multi",
    "run_ablation_scan",
    "run_ablation_scan_multi",
]
