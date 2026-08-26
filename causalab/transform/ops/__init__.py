"""The registered transform ops.

Importing this package runs each op module's ``@register`` decorator, which is
what populates the registry — so every new op must be imported here or it does
not exist as far as a document is concerned. None of these imports pulls in
torch: an op's numerics live inside its function body.
"""

from __future__ import annotations

from causalab.transform.ops import fit_pca, head_stats, paired_ttest

__all__ = ["fit_pca", "head_stats", "paired_ttest"]
