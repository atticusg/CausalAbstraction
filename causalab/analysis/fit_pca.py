"""``causalab.analysis.fit_pca`` — a principal basis over a saved read.

```json
"fit": {
  "type": "script", "script": {"module": "causalab.analysis.fit_pca"},
  "inputs": {"acts": {"step": "harvest", "file": "acts.safetensors"}, "k": 8},
  "outputs": {"weight": "basis.safetensors",
              "spectrum": {"file": "spectrum.json",
                           "columns": {"pc": "int64",
                                       "explained_variance": "float64",
                                       "explained_variance_ratio": "float64"}}}
}
```

Deterministic without a seed: a *full* SVD has no randomness to pin (unlike
sklearn's randomized solver), and the one genuine ambiguity — the sign of each
component, which SVD leaves free — is fixed here rather than left to the
backend's mood.

Formerly ``fit_pca@1`` in the transform-op registry. The numerics are
unchanged; what went away is the registry, whose admission-by-pull-request rule
also made a one-off reduction inexpressible. A script's identity is its content
hash, which the workflow digest carries (§7), so pinning the numerics no longer
needs a version in the name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, write_table, write_tensor

__all__ = ["main"]


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    from causalab.io.step_io import read_tensor

    acts = inputs["acts"]
    if isinstance(acts, (str, Path)):
        acts = read_tensor(Path(acts), what="fit_pca: 'acts'")
    if acts.ndim < 2:
        raise StepError(
            f"fit_pca: 'acts' needs at least 2 dimensions, got shape "
            f"{tuple(acts.shape)}"
        )
    k = int(inputs["k"])
    if k < 1:
        raise StepError(f"fit_pca: k must be >= 1, got {k}")
    # float64 throughout: the fit is the numerically delicate part, and a
    # float32 SVD of a near-degenerate covariance is not reproducible across
    # devices at the bit level.
    rows = acts.reshape(-1, acts.shape[-1]).to(torch.float64)
    n, d = int(rows.shape[0]), int(rows.shape[1])
    if k > min(n, d):
        raise StepError(
            f"fit_pca: k={k} exceeds the rank available from {n} rows of {d} dimensions"
        )
    if n < 2:
        raise StepError("fit_pca: a variance needs at least 2 rows")
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

    # (d, k), not (k, d): a featurizer's weight maps d -> k, and matching that
    # convention is what lets a protocol step load this bundle unchanged
    write_tensor(
        outputs["weight"],
        components.T.contiguous().to(torch.float32),
        slot="weight",
        # the rank is a parameter, not something inheritable from the input, and
        # a consuming `pca` featurizer's identity check requires it
        identity={"k": k},
    )
    write_table(
        Path(outputs["spectrum"]),
        [
            {
                "pc": i,
                "explained_variance": float(variance[i]),
                "explained_variance_ratio": float(ratio[i]),
            }
            for i in range(k)
        ],
    )
