"""Spec-native grid builders — the ``targets.py`` successor (WU2, #504).

The spec-layer rebuild of the retired ``causalab.neural.activations.targets``
(design-of-record: ``docs/REBASE_CAUSALAB_ON_NNTERP.md`` Part 6, "Grid
builders"; the legacy module was deleted by the WU6 sweep, #508). Four
builders turn ``(pipeline, layers, positions/heads, mode)`` into
``dict[key_tuple, list[list[SiteSpec]]]`` — the plain nested-sequence
groups form that replaced the legacy grouping container — plus the
structural grid detection the score-heatmap axis code consumes.

Contracts carried over 1:1 from the legacy builders:

* **Key tuples.** ``(layer, pos.id)`` for residual/MLP/attention-output grids,
  ``(layer, head)`` for attention-head grids — scans, grids, and
  ``causalab/io/plots/score_heatmap.py`` keep their axis semantics.
* **Grouping modes** (``mode=``, names kept verbatim so WU4/WU5 call sites and
  configs migrate mechanically; semantics identical to
  ``_group_units_into_targets``):

  - ``"one_target_all_units"`` → ``{("all",): [[spec, ...]]}`` — one fused
    single-group run over every site;
  - ``"one_target_per_unit"`` → ``{key: [[spec]], ...}`` — one single-spec
    group per grid cell (the scan shape);
  - ``"one_target_per_layer"`` → ``{(layer,): [[spec, ...]], ...}`` — all of
    a layer's sites in one group (the knockout shape); layers that
    contributed no sites are dropped.

  Every value is a *single-group* nesting (``len(groups) == 1``), exactly as
  the legacy modes built; group ``g`` reads its
  sources from ``example["counterfactual_inputs"][g]`` (the Plan-input naming
  contract, Part 6).
* **The residual ``layer=-1`` special case.** ``-1`` means embeddings: the
  spec's engine site is ``Site("block_input", 0)`` (the engine requires
  ``layer >= 0``) while the dict key — and the caller-facing axis — keeps
  ``-1``.
* **Config width probing.** ``hidden_size`` for residual / MLP-in-out /
  attention-output; ``mlp_activation`` walks ``n_inner`` (present and
  non-``None``) → ``intermediate_size`` → ``hidden_size * 4``; head width is
  ``head_dim`` else ``hidden_size // (n_head | num_attention_heads |
  num_heads)``. Each spec's ``width`` is bound from these (the DAS/DBM
  rotation-sizing currency).

New over legacy: every generated :class:`~causalab.neural.specs.SiteSpec`
carries an explicit, **opaque** ``key`` (unique across the grid — they feed
collect-result dicts in WU3; nothing may parse them), and a duplicate grid
cell — e.g. two token positions sharing an ``id`` — is refused loudly instead
of silently overwriting (per-unit mode) or double-running (fused mode).

Like :mod:`causalab.neural.specs`, this module is deliberately **not**
re-exported from :mod:`causalab.neural.activations` — import it directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec

if TYPE_CHECKING:
    from causalab.neural.pipeline import LMPipeline
    from causalab.neural.token_positions import TokenPosition

__all__ = [
    "GROUPING_MODES",
    "SiteGrid",
    "build_attention_head_sites",
    "build_attention_output_sites",
    "build_mlp_sites",
    "build_residual_stream_sites",
    "extract_grid_dimensions_from_targets",
    "grid_component",
]

#: A built grid: grid-cell key tuple → single-group nested spec lists
#: (``(layer, pos.id)`` or ``(layer, head)`` keys — or ``("all",)`` /
#: ``(layer,)`` under the collapsing modes).
SiteGrid = dict[tuple[Any, ...], list[list[SiteSpec]]]

#: The three grouping modes, unchanged from the legacy builders (semantics
#: pinned by ``_group_units_into_targets``; see the module docstring).
GROUPING_MODES = (
    "one_target_all_units",
    "one_target_per_unit",
    "one_target_per_layer",
)


# =============================================================================
# Grid Component Detection
# =============================================================================


def grid_component(sites_dict: SiteGrid) -> str:
    """The grid's component type, read **structurally** off its first spec.

    The ``detect_component_type_from_targets`` successor: instead of substring-
    matching legacy unit-id strings, it reads ``site.component`` /
    ``HeadSite`` kind-ness from the first spec's engine site — after the sweep
    nothing in the tree parses a where-identifier out of a string.

    Args:
        sites_dict: A built grid (any grouping mode).

    Returns:
        One of ``"attention_head"``, ``"attention_output"``,
        ``"residual_stream"``, ``"mlp"``.

    Raises:
        ValueError: If the dict is empty, its first grid value holds no specs,
            or the first spec's site has no grid component type (e.g. an
            ``embeddings`` :class:`Site`, which no builder emits).
    """
    if not sites_dict:
        raise ValueError("sites dict is empty")
    groups = next(iter(sites_dict.values()))
    specs = [spec for group in groups for spec in group]
    if not specs:
        raise ValueError("grid value has no specs")
    site = specs[0].fsite.site
    # Any per-head site is an attention-head grid, whatever its kind — the
    # structural counterpart of the legacy "AttentionHead" id-substring match
    # (which also caught AttentionHeadValue/AttentionHeadQuery ids).
    if isinstance(site, HeadSite):
        return "attention_head"
    if isinstance(site, Site):
        if site.component in ("block_input", "block_output"):
            return "residual_stream"
        if site.component in ("mlp_input", "mlp_activation", "mlp_output"):
            return "mlp"
        if site.component == "attention_output":
            return "attention_output"
    raise ValueError(f"Unknown grid component type for site: {site!r}")


# =============================================================================
# Grid Dimension Extraction
# =============================================================================


def extract_grid_dimensions_from_targets(
    component_type: str,
    sites_dict: SiteGrid,
) -> dict[str, list[Any]]:
    """Extract grid dimensions from the keys of a built grid.

    Ported as-is from the legacy module (it reads only the dict keys, which
    are unchanged). For ``one_target_per_unit`` grids:

    - ``attention_head``: keys are ``(layer, head)``;
    - ``residual_stream`` / ``mlp`` / ``attention_output``: keys are
      ``(layer, token_position_id)``.

    Args:
        component_type: One of ``"attention_head"``, ``"residual_stream"``,
            ``"mlp"``, ``"attention_output"`` (as returned by
            :func:`grid_component`).
        sites_dict: A built grid.

    Returns:
        - ``attention_head``: ``{"layers": [...], "heads": [...]}`` (sorted);
        - otherwise: ``{"layers": [...], "token_position_ids": [...]}`` —
          layers sorted, position ids in **first-seen insertion order**
          (``causalab/io/plots/score_heatmap.py`` relies on this axis order).

    Raises:
        ValueError: If ``component_type`` is not one of the four known types.
    """
    keys = list(sites_dict.keys())

    if component_type == "attention_head":
        layers = sorted(set(k[0] for k in keys))
        heads = sorted(set(k[1] for k in keys))
        return {"layers": layers, "heads": heads}
    elif component_type in ("residual_stream", "mlp", "attention_output"):
        # keys are (layer, position_id)
        layers = sorted(set(k[0] for k in keys))
        # Token position IDs may be strings, preserve insertion order
        # (downstream score_heatmap.py relies on this axis order).
        position_ids = []
        seen = set()
        for k in keys:
            if k[1] not in seen:
                position_ids.append(k[1])
                seen.add(k[1])
        return {"layers": layers, "token_position_ids": position_ids}
    else:
        raise ValueError(
            f"Unknown component_type: {component_type!r}. Expected one of: "
            f"'attention_head', 'residual_stream', 'mlp', 'attention_output'."
        )


# =============================================================================
# Grid Builder Functions
# =============================================================================


def _group_specs_into_grid(
    specs_with_keys: list[tuple[tuple[Any, ...], SiteSpec]],
    layers: Sequence[int],
    mode: str,
) -> SiteGrid:
    """Group specs into a grid dict of ``[[SiteSpec]]`` values (the retired
    legacy builders' grouping semantics, verbatim).

    Args:
        specs_with_keys: ``(key_tuple, spec)`` pairs, in build order; the
            key tuple's first element is the (caller-facing) layer.
        layers: Layer indices, for ``one_target_per_layer`` grouping.
        mode: One of :data:`GROUPING_MODES`.

    Raises:
        ValueError: On an unknown ``mode``, or on duplicate grid keys (which
            would silently overwrite per-unit cells and collide the specs'
            collect-result keys).
    """
    keys = [key for key, _ in specs_with_keys]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(
            f"duplicate grid keys {dupes}: every (layer, position-id) / "
            "(layer, head) combination must be unique — duplicate token-"
            "position ids or repeated layers/heads collide the specs' keys."
        )

    if mode == "one_target_all_units":
        specs = [spec for _, spec in specs_with_keys]
        return {("all",): [specs]}

    elif mode == "one_target_per_unit":
        return {key: [[spec]] for key, spec in specs_with_keys}

    elif mode == "one_target_per_layer":
        layer_groups: dict[int, list[SiteSpec]] = {layer: [] for layer in layers}
        for (layer, *_), spec in specs_with_keys:
            layer_groups[layer].append(spec)
        return {(layer,): [group] for layer in layers if (group := layer_groups[layer])}

    else:
        raise ValueError(
            f"Invalid mode: {mode}. "
            f"Expected: 'one_target_all_units', 'one_target_per_unit', "
            f"'one_target_per_layer'"
        )


def build_residual_stream_sites(
    pipeline: LMPipeline,
    layers: Sequence[int],
    token_positions: Sequence[TokenPosition],
    mode: str = "one_target_per_unit",
    target_output: bool = True,
) -> SiteGrid:
    """Build a residual-stream grid of :class:`SiteSpec` groups.

    Args:
        pipeline: ``LMPipeline`` whose ``model.config.hidden_size`` binds each
            spec's ``width``.
        layers: Layer indices to intervene on. ``-1`` means embeddings: the
            spec's engine site becomes ``Site("block_input", 0)`` while the
            dict key keeps ``-1`` (the caller-facing axis).
        token_positions: Declarative position resolvers (any
            ``PositionResolver`` with an ``id``, e.g. ``TokenPosition``); each
            contributes a grid column keyed by its ``id``.
        mode: One of :data:`GROUPING_MODES` (see the module docstring).
        target_output: Target ``block_output`` (True) or ``block_input``
            (False). Overridden to ``block_input`` by the ``layer=-1`` rewrite.

    Returns:
        ``{(layer, pos.id): [[spec]], ...}`` under ``one_target_per_unit``;
        ``{("all",): [[...]]}`` / ``{(layer,): [[...]]}`` under the
        collapsing modes.
    """
    hidden_size = pipeline.model.config.hidden_size

    specs_with_keys = []
    for layer in layers:
        for pos in token_positions:
            # Handle layer -1 special case (embeddings/block_input at layer 0)
            actual_layer = layer
            actual_target_output = target_output
            if layer == -1:
                actual_layer = 0
                actual_target_output = False

            component = "block_output" if actual_target_output else "block_input"
            spec = SiteSpec(
                fsite=FeaturizedSite(Site(component, actual_layer)),
                positions=pos,
                # Opaque; component included so block_input(layer=-1) keys
                # never collide with a block_output grid over layer 0.
                key=f"residual_stream.L{layer}.{component}.{pos.id}",
                width=hidden_size,
            )
            specs_with_keys.append(((layer, pos.id), spec))

    return _group_specs_into_grid(specs_with_keys, layers, mode)


def build_mlp_sites(
    pipeline: LMPipeline,
    layers: Sequence[int],
    token_positions: Sequence[TokenPosition],
    mode: str = "one_target_per_unit",
    location: str = "mlp_output",
) -> SiteGrid:
    """Build an MLP grid of :class:`SiteSpec` groups.

    Args:
        pipeline: ``LMPipeline`` whose config binds each spec's ``width``:
            ``hidden_size`` for ``mlp_input``/``mlp_output``; for
            ``mlp_activation`` the legacy fallback chain ``n_inner`` (present
            and non-``None``) → ``intermediate_size`` → ``hidden_size * 4``.
        layers: Layer indices to intervene on.
        token_positions: Declarative position resolvers; each contributes a
            grid column keyed by its ``id``.
        mode: One of :data:`GROUPING_MODES` (see the module docstring).
        location: Which MLP component to target — ``"mlp_input"``,
            ``"mlp_output"`` (default), or ``"mlp_activation"`` (the engine's
            :class:`Site` vocabulary; anything else is refused at
            construction).

    Returns:
        ``{(layer, pos.id): [[spec]], ...}`` under ``one_target_per_unit``;
        collapsed keys under the other modes.
    """
    # Get dimension from model config based on location
    p_config = pipeline.model.config
    if location == "mlp_activation":
        if hasattr(p_config, "n_inner") and p_config.n_inner is not None:
            feature_size = p_config.n_inner
        elif hasattr(p_config, "intermediate_size"):
            feature_size = p_config.intermediate_size
        else:
            feature_size = p_config.hidden_size * 4
    else:
        feature_size = p_config.hidden_size

    specs_with_keys = []
    for layer in layers:
        for pos in token_positions:
            spec = SiteSpec(
                fsite=FeaturizedSite(Site(location, layer)),
                positions=pos,
                key=f"mlp.L{layer}.{location}.{pos.id}",
                width=feature_size,
            )
            specs_with_keys.append(((layer, pos.id), spec))

    return _group_specs_into_grid(specs_with_keys, layers, mode)


def build_attention_output_sites(
    pipeline: LMPipeline,
    layers: Sequence[int],
    token_positions: Sequence[TokenPosition],
    mode: str = "one_target_per_unit",
) -> SiteGrid:
    """Build a whole-attention-sublayer grid of :class:`SiteSpec` groups.

    Targets the engine's ``attention_output`` component — the full attention
    sublayer output (all heads jointly) written back into the residual
    stream. The layer-level analogue of :func:`build_attention_head_sites`
    (which targets a single head's value stream), mirroring
    :func:`build_mlp_sites` in shape and grouping. The merged sublayer output
    is already ``hidden_size``-wide, so this is GQA-correct without per-head
    handling.

    Args:
        pipeline: ``LMPipeline`` whose ``model.config.hidden_size`` binds each
            spec's ``width``.
        layers: Layer indices to intervene on.
        token_positions: Declarative position resolvers; each contributes a
            grid column keyed by its ``id``.
        mode: One of :data:`GROUPING_MODES` (see the module docstring).

    Returns:
        ``{(layer, pos.id): [[spec]], ...}`` under ``one_target_per_unit``;
        collapsed keys under the other modes.
    """
    hidden_size = pipeline.model.config.hidden_size

    specs_with_keys = []
    for layer in layers:
        for pos in token_positions:
            spec = SiteSpec(
                fsite=FeaturizedSite(Site("attention_output", layer)),
                positions=pos,
                key=f"attention_output.L{layer}.{pos.id}",
                width=hidden_size,
            )
            specs_with_keys.append(((layer, pos.id), spec))

    return _group_specs_into_grid(specs_with_keys, layers, mode)


def build_attention_head_sites(
    pipeline: LMPipeline,
    layers: Sequence[int],
    heads: Sequence[int],
    token_position: TokenPosition,
    mode: str = "one_target_per_unit",
) -> SiteGrid:
    """Build an attention-head grid of :class:`SiteSpec` groups.

    Each spec's engine site is ``HeadSite(kind="attention_value", layer,
    head)`` — the per-head o-projection input (the sender component the
    legacy ``AttentionHead`` unit targeted as pyvene's
    ``head_attention_value_output``). The ``"value"``/``"query"`` kinds are
    receiver realizations built directly by path patching and deliberately
    have no grid builder.

    Args:
        pipeline: ``LMPipeline`` whose config binds each spec's ``width``:
            ``head_dim`` when present, else ``hidden_size //`` the head count
            from the ``n_head`` → ``num_attention_heads`` → ``num_heads``
            attribute chain.
        layers: Layer indices.
        heads: Head indices.
        token_position: One declarative position resolver shared by every
            ``(layer, head)`` cell.
        mode: One of :data:`GROUPING_MODES`; ``one_target_per_layer`` groups
            all of a layer's heads into one group (the knockout shape).

    Returns:
        ``{(layer, head): [[spec]], ...}`` under ``one_target_per_unit``;
        collapsed keys under the other modes.

    Raises:
        ValueError: If the config exposes none of ``head_dim`` / ``n_head`` /
            ``num_attention_heads`` / ``num_heads``.
    """
    # Calculate head dimension from model config
    p_config = pipeline.model.config
    if hasattr(p_config, "head_dim"):
        head_size = p_config.head_dim
    else:
        if hasattr(p_config, "n_head"):
            num_heads = p_config.n_head
        elif hasattr(p_config, "num_attention_heads"):
            num_heads = p_config.num_attention_heads
        elif hasattr(p_config, "num_heads"):
            num_heads = p_config.num_heads
        else:
            raise ValueError(
                "Could not determine number of heads from model config. "
                "Expected one of: head_dim, n_head, num_attention_heads, num_heads"
            )
        head_size = pipeline.model.config.hidden_size // num_heads

    specs_with_keys = []
    for layer in layers:
        for head in heads:
            spec = SiteSpec(
                fsite=FeaturizedSite(
                    HeadSite(kind="attention_value", layer=layer, head=head)
                ),
                positions=token_position,
                key=f"attention_head.L{layer}.H{head}.{token_position.id}",
                width=head_size,
            )
            specs_with_keys.append(((layer, head), spec))

    return _group_specs_into_grid(specs_with_keys, layers, mode)
