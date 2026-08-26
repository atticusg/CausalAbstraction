"""``fit_pca@1`` — a principal basis over a saved read.

Deterministic without a seed: a *full* SVD has no randomness to pin (unlike
sklearn's randomized solver), and the one genuine ambiguity — the sign of each
component, which SVD leaves free — is fixed here rather than left to the
backend's mood. That is the registry's admission criterion met by
construction, which is why this op declares no ``seed`` parameter: a seed that
changes nothing would be a lie in the record.
"""

from __future__ import annotations

from typing import Any, Mapping

from causalab.transform.registry import register
from causalab.transform.schema import Int, Table, Tensor, TransformError

__all__ = ["fit_pca"]


@register(
    name="fit_pca",
    version=1,
    inputs={
        "acts": Tensor(
            description="a saved read; leading dimensions are flattened, so "
            "[examples, positions, d] and [rows, d] both mean rows of d"
        )
    },
    outputs={
        "weight": Tensor(
            description="the principal basis as (d, k) — the orientation and "
            "the slot name of the 'pca' featurizer's weight "
            "(FEATURIZER_SLOTS, canonical.py's [width, k] shape rule), so the "
            "bundle loads straight into a downstream protocol step"
        ),
        "spectrum": Table(
            columns={
                "pc": "int64",
                "explained_variance": "float64",
                "explained_variance_ratio": "float64",
            },
            description="one row per retained component, in descending order",
        ),
    },
    params={"k": Int(min=1, description="number of components to retain")},
    identity_from_params={"k": "k"},
    description="Principal component basis of a saved read, by full SVD.",
)
def fit_pca(*, inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd
    import torch

    acts = inputs["acts"]
    if acts.ndim < 2:
        raise TransformError(
            f"fit_pca@1: 'acts' needs at least 2 dimensions, got shape "
            f"{tuple(acts.shape)}"
        )
    # float64 throughout: the fit is the numerically delicate part, and a
    # float32 SVD of a near-degenerate covariance is not reproducible across
    # devices at the bit level, which the registry requires.
    rows = acts.reshape(-1, acts.shape[-1]).to(torch.float64)
    n, d = int(rows.shape[0]), int(rows.shape[1])
    k = int(params["k"])
    if k > min(n, d):
        raise TransformError(
            f"fit_pca@1: k={k} exceeds the rank available from {n} rows of {d} "
            "dimensions"
        )
    if n < 2:
        raise TransformError("fit_pca@1: a variance needs at least 2 rows")
    centered = rows - rows.mean(dim=0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:k].clone()
    # sign convention: SVD fixes each component only up to a sign, so pin it —
    # the entry of largest magnitude is made positive. Without this the same
    # input can produce two bases that are both "correct" and not equal.
    leading = components.abs().argmax(dim=1)
    signs = torch.sign(components.gather(1, leading.unsqueeze(1))).squeeze(1)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    components = components * signs.unsqueeze(1)
    variance = (singular**2) / (n - 1)
    total = variance.sum()
    ratio = variance / total if total > 0 else torch.zeros_like(variance)
    spectrum = pd.DataFrame(
        {
            "pc": list(range(k)),
            "explained_variance": [float(v) for v in variance[:k]],
            "explained_variance_ratio": [float(v) for v in ratio[:k]],
        }
    )
    # (d, k), not (k, d): a featurizer's weight maps d -> k, and matching that
    # convention here is what lets a protocol step load this bundle unchanged
    return {"weight": components.T.contiguous().to(torch.float32), "spectrum": spectrum}
