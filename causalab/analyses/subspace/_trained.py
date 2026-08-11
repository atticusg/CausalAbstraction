"""Shared helpers over ``train_interventions`` result cells (WU5, #507).

Training is functional post where-unification: the trained featurizer /
feature ids arrive on the specs a result cell carries under
``"trained_specs"`` (a flat spec sequence or nested spec groups), never by
mutating the caller's specs. These helpers normalize that shape for the
single-cell subspace flows.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from causalab.neural.specs import SiteSpec


def trained_specs(result_cell: Mapping[str, Any]) -> list[SiteSpec]:
    """The cell's trained specs, flattened (flat sequence or nested groups)."""
    trained: Sequence[Any] = result_cell["trained_specs"]
    flat: list[SiteSpec] = []
    for item in trained:
        if isinstance(item, SiteSpec):
            flat.append(item)
        else:
            flat.extend(item)
    return flat


def trained_spec(result_cell: Mapping[str, Any]) -> SiteSpec:
    """The cell's single trained spec (single-site flows)."""
    flat = trained_specs(result_cell)
    if len(flat) != 1:
        raise ValueError(
            f"expected a single trained spec in the result cell, got {len(flat)}"
        )
    return flat[0]
