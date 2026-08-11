"""Structured grid-cell records for mask/feature-count plots (WU5, #507).

The spec-era replacement for :mod:`causalab.io.plots.unit_id`'s id-string
parsing: plot axes are joined **structurally** from the specs of a built grid
(:data:`~causalab.neural.activations.site_grids.SiteGrid`), never parsed out
of result keys. Post-migration the per-key dicts that feed these plots
(``feature_indices`` from trained-subspace results, WU1 bundle records) are
keyed by ``spec.key``, which is opaque by contract — :func:`cells_from_site_grid`
recovers each key's ``(component, layer, head / position)`` from the grid's
own :class:`~causalab.neural.specs.SiteSpec` values.

Layer semantics match the legacy id-parsing exactly: a residual ``layer=-1``
grid cell is built as ``Site("block_output"/"block_input", 0)`` and its legacy
unit id said ``Layer-0``, so these plots labelled it ``L0`` — deriving the
layer from the spec's engine site reproduces that.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

from causalab.neural.activations.site_grids import SiteGrid, grid_component
from causalab.neural.head_view import HeadSite
from causalab.neural.specs import SiteSpec

__all__ = [
    "GridCell",
    "cell_grid_dimensions",
    "cells_from_site_grid",
    "cells_from_specs",
]


@dataclasses.dataclass(frozen=True)
class GridCell:
    """One plotted grid cell: a spec's structural coordinates plus its
    selection state.

    ``indices`` follows the trained-mask convention: ``None`` means every
    feature is selected (binary mask = on), a list is the selected feature
    subset (binary mask = off when empty is not constructible; mask plots
    treat any non-``None`` list as off, feature-count plots count it).
    """

    key: str
    layer: int
    head: Optional[int] = None
    position: Optional[str] = None
    indices: Optional[tuple[int, ...]] = None
    n_features: Optional[int] = None


def _cell_from_spec(
    spec: SiteSpec,
    indices: Optional[Sequence[int]],
    n_features: Optional[int],
) -> GridCell:
    site = spec.fsite.site
    head = site.head if isinstance(site, HeadSite) else None
    position = getattr(spec.positions, "id", None)
    return GridCell(
        key=spec.key,
        layer=int(site.layer),
        head=head,
        position=position if isinstance(position, str) else None,
        indices=None if indices is None else tuple(int(i) for i in indices),
        n_features=n_features,
    )


def _n_features_for(
    n_features: int | Mapping[str, int] | None, spec: SiteSpec
) -> Optional[int]:
    """Per-spec feature count: a shared int, a per-key mapping (missing keys
    fall back to the spec's featurizer ``n_features``), or ``None`` →
    featurizer-derived when available."""
    if isinstance(n_features, int):
        return n_features
    if n_features is not None and spec.key in n_features:
        return int(n_features[spec.key])
    derived = spec.fsite.featurizer.n_features
    return None if derived is None else int(derived)


def cells_from_specs(
    specs: Sequence[SiteSpec],
    feature_indices: Mapping[str, Optional[Sequence[int]]],
    n_features: int | Mapping[str, int] | None = None,
) -> list[GridCell]:
    """Join per-key ``feature_indices`` with the specs that produced them.

    Only specs whose ``key`` appears in ``feature_indices`` contribute a cell
    (mirroring the legacy plots, which skipped unit ids absent from the dict).
    Keys are matched exactly and never parsed.
    """
    cells: list[GridCell] = []
    for spec in specs:
        if spec.key not in feature_indices:
            continue
        cells.append(
            _cell_from_spec(
                spec, feature_indices[spec.key], _n_features_for(n_features, spec)
            )
        )
    return cells


def cells_from_site_grid(
    sites_dict: SiteGrid,
    feature_indices: Mapping[str, Optional[Sequence[int]]],
    n_features: int | Mapping[str, int] | None = None,
) -> tuple[str, list[GridCell]]:
    """Join a built grid's structure with per-key ``feature_indices``.

    Works for every grouping mode (per-unit, per-layer, fused ``("all",)``)
    because the coordinates come from each :class:`SiteSpec`'s engine site
    and position resolver, not from the dict keys.

    Args:
        sites_dict: A built grid (any grouping mode).
        feature_indices: ``{spec.key: indices-or-None}`` — e.g. a trained
            result's ``feature_indices`` dict, keyed by ``spec.key``.
        n_features: Total features per cell — a shared int, a per-key
            mapping, or ``None`` to read each spec's featurizer.

    Returns:
        ``(component_type, cells)`` where ``component_type`` is
        :func:`~causalab.neural.activations.site_grids.grid_component`'s
        structural detection over the grid.
    """
    component_type = grid_component(sites_dict)
    specs = [
        spec for groups in sites_dict.values() for group in groups for spec in group
    ]
    return component_type, cells_from_specs(specs, feature_indices, n_features)


def cell_grid_dimensions(
    component_type: str, cells: Sequence[GridCell]
) -> dict[str, list[Any]]:
    """Grid axes from structured cells — the ``extract_grid_dimensions``
    successor (id parsing retired).

    Returns ``{"layers", "heads"}`` for ``attention_head`` grids (both
    sorted) and ``{"layers", "token_position_ids"}`` otherwise (layers
    sorted, position ids in first-seen order — the axis-order contract
    ``score_heatmap`` shares).
    """
    if component_type == "attention_head":
        layers = sorted({c.layer for c in cells})
        heads = sorted({c.head for c in cells if c.head is not None})
        return {"layers": layers, "heads": heads}
    layers = sorted({c.layer for c in cells})
    position_ids: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.position is not None and cell.position not in seen:
            position_ids.append(cell.position)
            seen.add(cell.position)
    return {"layers": layers, "token_position_ids": position_ids}
