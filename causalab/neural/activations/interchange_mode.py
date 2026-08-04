"""
interchange.py
==============
Core utilities for running interchange intervention experiments.

This module provides functions for running interventions on neural networks using
the pyvene library. It focuses on interchange interventions where activations are
swapped between base and counterfactual inputs.

For training intervention models, see train.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from tqdm import tqdm
from pyvene import IntervenableModel  # type: ignore[import-untyped]

from causalab.causal.counterfactual_dataset import (
    CounterfactualExample,
    LabeledCounterfactualExample,
)
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import AtomicModelUnit, InterchangeTarget
from causalab.neural.activations.engine import (
    build_plans,
    collect_unit_activations,
    collect_unit_activations_under,
    generate_with_interventions,
)
from causalab.neural.activations.intervenable_model import (
    delete_intervenable_model,
)
from causalab.neural.activations.collect import collect_source_representations
from causalab.neural.activations.data_utils import (
    convert_to_top_k,
    move_outputs_to_cpu,
)

# Configure logging
logger = logging.getLogger(__name__)


def _first_index_shape_mismatch(
    base: Any, cf: Any, path: tuple[int, ...] = ()
) -> tuple[tuple[int, ...], Any, Any] | None:
    """Return the first point where two nested index structures diverge in shape.

    Interchange pairs base and counterfactual (source) positions element-wise, so
    pyvene's gather/scatter requires their nested list shapes to match exactly.
    Walks ``base`` and ``cf`` in parallel and returns ``(path, base_len, cf_len)``
    at the first list whose lengths differ (or where one side is a list and the
    other a scalar index); returns ``None`` when the shapes match. Generic over
    nesting depth, so it covers both the single-axis ``ResidualStream`` layout
    (per-example list of position lists) and the multi-axis ``AttentionHead``
    layout (``[head_axis, position_axis]``).
    """
    base_is_list = isinstance(base, list)
    cf_is_list = isinstance(cf, list)
    if base_is_list != cf_is_list:
        return path, base, cf
    if not base_is_list:
        return None
    if len(base) != len(cf):
        return path, len(base), len(cf)
    for i, (b, c) in enumerate(zip(base, cf)):
        mismatch = _first_index_shape_mismatch(b, c, path + (i,))
        if mismatch is not None:
            return mismatch
    return None


def _first_ragged_span(
    group: Any,
) -> tuple[int, tuple[tuple[int, ...], Any, Any]] | None:
    """Return the first batch example in a *group* whose index shape diverges from example 0.

    ``group`` is a batch of per-example locations that pyvene stacks with a
    single ``torch.tensor(...)`` call (``[example_0, example_1, ...]``), so every
    example must contribute the same nested shape. A *ragged* span — 1 token for
    some examples and 2 for others — surfaces as the cryptic ``expected sequence
    of length N at dim 1`` error that poisons the context. A *uniform*
    multi-token span stacks fine and is allowed.

    Returns ``(example_index, mismatch)`` for the first example whose shape
    differs from example 0 (``mismatch`` is the `_first_index_shape_mismatch`
    tuple), or ``None`` when every example shares the same shape. Comparing each
    example against example 0 — rather than flagging any inner list of length
    > 1 — distinguishes a legitimate uniform span from a genuinely ragged one.

    The caller must pass a single tensorized group. A single-axis ``pos`` unit's
    whole structure is one group; a multi-axis ``h.pos`` unit's head and position
    axes are stacked by ``torch.tensor`` *separately* (``gather_neurons`` indexes
    ``unit_locations_as_list[0]`` and ``[1]``), so each axis is its own group —
    do NOT pass the combined ``[head_axis, position_axis]`` list here, or the two
    axes (which legitimately differ in width) would read as ragged.
    """
    if not isinstance(group, list) or len(group) < 2:
        return None
    first = group[0]
    for example_index in range(1, len(group)):
        mismatch = _first_index_shape_mismatch(first, group[example_index])
        if mismatch is not None:
            return example_index, mismatch
    return None


def prepare_intervenable_inputs(
    pipeline: Pipeline,
    examples: list[CounterfactualExample] | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[list[list[int] | None]],
]:
    """
    Prepare the inputs for the intervenable model.

    This function loads the base and counterfactual inputs, and prepares the indices
    for the model units.

    Args:
        pipeline: The pipeline containing the model
        examples: List of counterfactual examples, each with "input" and
            "counterfactual_inputs" keys.
        interchange_target: InterchangeTarget containing the model units to be intervened on.
            Groups in the target correspond to counterfactual inputs.

    Returns:
        Tuple of (batched_base, batched_counterfactuals, inv_locations, feature_indices)
    """
    # Extract and batch inputs from examples
    raw_base = [ex["input"] for ex in examples]
    # Shape: (batch_size, num_counterfactuals) -> (num_counterfactuals, batch_size)
    # Convert zip tuples to lists for pipeline.load() compatibility
    raw_counterfactuals = [
        list(cf_tuple)
        for cf_tuple in zip(*[ex["counterfactual_inputs"] for ex in examples])
    ]

    batched_base = pipeline.load(raw_base)
    batched_counterfactuals = [
        pipeline.load(batched_counterfactual)
        for batched_counterfactual in raw_counterfactuals
    ]

    # Positions returned by `index_component` are in each example's unpadded
    # tokenization frame. Passing the per-batch `attention_mask` shifts them
    # into the padded-batch frame (`argmax(attention_mask, dim=1)` per row).
    # shape: (num_model_units, batch_size, num_component_indices)
    base_indices = [
        model_unit.index_component(
            raw_base,
            batch=True,
            is_original=True,
            attention_mask=batched_base["attention_mask"],
        )
        for group in interchange_target
        for model_unit in group
    ]

    # shape: (num_model_units, batch_size, num_component_indices)
    counterfactual_indices = [
        model_unit.index_component(
            batched_counterfactual,
            batch=True,
            is_original=False,
            attention_mask=batched_cf["attention_mask"],
        )
        for group, batched_counterfactual, batched_cf in zip(
            interchange_target, raw_counterfactuals, batched_counterfactuals
        )
        for model_unit in group
    ]

    # shape: (num_model_units, batch_size, num_feature_indices)
    feature_indices = [
        [model_unit.get_feature_indices() for _ in range(len(raw_base))]
        for group in interchange_target
        for model_unit in group
    ]

    # pyvene gathers source (counterfactual) activations and scatters them onto
    # the base at the paired indices, so the two index structures must share the
    # same nested shape. A mismatch reaches the CUDA gather/scatter as an
    # out-of-bounds assertion that poisons the context; catch it on CPU here with
    # an actionable error instead. The per-position bounds check in
    # `_apply_padding_shift` (#176) only validates individual indices, so a
    # length mismatch where every index is in-bounds slips past it.
    mismatch = _first_index_shape_mismatch(base_indices, counterfactual_indices)
    if mismatch is not None:
        _path, base_len, cf_len = mismatch
        raise ValueError(
            f"Base and counterfactual token positions have mismatched shapes at "
            f"index path {list(_path)}: base selects {base_len}, counterfactual "
            f"selects {cf_len}. Interchange pairs positions, so both must select "
            f"the same number of tokens per example — a mismatch reaches pyvene's "
            f"gather/scatter as a CUDA assertion that poisons the context. Usual "
            f"cause: a counterfactual value string that tokenizes to a different "
            f"number of tokens than the base value. Use single-token values or a "
            f"fixed single position."
        )

    # Multi-token positions are supported as long as the span is *uniform* across
    # the batch (every example selects the same number of tokens). pyvene stacks a
    # unit's per-example positions into one `(batch, num_positions)` tensor and
    # gathers/scatters every position simultaneously, and the feature
    # interventions set `keep_last_dim=True` so the featurizer sees
    # `(batch, num_positions, d)` and applies per position (its d-sized weights
    # broadcast across the span). The remaining failure mode is a *ragged* span —
    # a width that varies across the batch. `torch.tensor` in `gather_neurons`
    # can't build a rectangular tensor from it and raises the cryptic `expected
    # sequence of length N at dim 1` error that poisons the context. The base/CF
    # mismatch check above only compares the two sides element-wise, so a span
    # that is ragged *within* one side (identically on both) slips past it; catch
    # it here per unit with an actionable message. Each unit reports how pyvene
    # tensorizes its index axes via `tensorized_index_groups` — a single-axis
    # `pos` unit is one group, a multi-axis `h.pos` unit splits into head and
    # position axes (stacked separately) — and we check each group on its own so
    # the head and position axes' legitimately-different widths are never misread
    # as ragged.
    model_units = [model_unit for group in interchange_target for model_unit in group]
    for label, all_indices in (
        ("base token position", base_indices),
        ("counterfactual token position", counterfactual_indices),
    ):
        for unit_idx, unit_indices in enumerate(all_indices):
            # The unit class knows how pyvene tensorizes its index axes: a plain
            # `pos` unit is one group; a multi-axis `h.pos` unit splits into
            # `[head_axis, position_axis]`. Asking the unit keeps that layout
            # knowledge on the class instead of re-deriving it from the unit
            # string here.
            tensorized_groups = model_units[unit_idx].tensorized_index_groups(
                unit_indices
            )
            for group in tensorized_groups:
                ragged = _first_ragged_span(group)
                if ragged is not None:
                    example_index, (_path, len0, len_j) = ragged
                    raise ValueError(
                        f"A {label} selects a variable number of "
                        f"tokens across the batch (unit {unit_idx}: example 0 "
                        f"selects {len0}, example {example_index} selects {len_j} "
                        f"at index path {list(_path)}). Interchange stacks a "
                        f"unit's per-example positions with `torch.tensor`, so "
                        f"every example must select the same number of tokens — a "
                        f"ragged span reaches pyvene's gather as an `expected "
                        f"sequence of length N at dim 1` error that poisons the "
                        f"context. Usual cause: a value string that tokenizes to a "
                        f"different number of tokens for different examples. Use a "
                        f"fixed-width position (e.g. the last token of the value)."
                    )

    inv_locations = {"sources->base": (counterfactual_indices, base_indices)}
    return batched_base, batched_counterfactuals, inv_locations, feature_indices


# --------------------------------------------------------------------------- #
#  Trace-engine batch preparation                                             #
# --------------------------------------------------------------------------- #
@dataclass
class InterchangeBatch:
    """One batch's tokenized inputs and resolved positions, ready for the engine.

    Positions are per unit, in the padded-batch frame, and already validated: the
    base/counterfactual shapes match and no span is ragged across the batch (see
    :func:`prepare_intervenable_inputs` for what those two failures look like and
    why they must be caught here rather than deep in a gather).
    """

    base_encoding: dict[str, Any]
    source_encodings: list[dict[str, Any]]
    base_positions: list[list[list[int]]]
    source_positions: list[list[list[int]]]
    feature_indices: list[list[int] | None]
    units: list[AtomicModelUnit]
    group_sizes: list[int]


def _validate_positions(
    units: Sequence[AtomicModelUnit],
    base_indices: Sequence[Any],
    counterfactual_indices: Sequence[Any],
    *,
    role: str = "base token position",
) -> None:
    """Reject shape mismatches and ragged spans before they reach a gather.

    Both failures used to surface as a CUDA assertion that poisoned the context;
    they are cheap to detect here and impossible to diagnose there. The
    restriction they encode — a uniform, matched span width — is what lets the
    engine build one rectangular index tensor per unit.
    """
    mismatch = _first_index_shape_mismatch(
        list(base_indices), list(counterfactual_indices)
    )
    if mismatch is not None:
        _path, base_len, cf_len = mismatch
        raise ValueError(
            f"Base and counterfactual token positions have mismatched shapes at "
            f"index path {list(_path)}: base selects {base_len}, counterfactual "
            f"selects {cf_len}. Interchange pairs positions, so both must select "
            f"the same number of tokens per example. Usual cause: a counterfactual "
            f"value string that tokenizes to a different number of tokens than the "
            f"base value. Use single-token values or a fixed single position."
        )
    for label, all_indices in (
        (role, base_indices),
        ("counterfactual token position", counterfactual_indices),
    ):
        for unit_idx, unit_indices in enumerate(all_indices):
            for group in units[unit_idx].tensorized_index_groups(unit_indices):
                ragged = _first_ragged_span(group)
                if ragged is not None:
                    example_index, (_path, len0, len_j) = ragged
                    raise ValueError(
                        f"A {label} selects a variable number of "
                        f"tokens across the batch (unit {unit_idx}: example 0 "
                        f"selects {len0}, example {example_index} selects {len_j} "
                        f"at index path {list(_path)}). A unit's per-example "
                        f"positions are stacked into one index tensor, so every "
                        f"example must select the same number of tokens. Usual "
                        f"cause: a value string that tokenizes to a different "
                        f"number of tokens for different examples. Use a "
                        f"fixed-width position (e.g. the last token of the value)."
                    )


def prepare_interchange_batch(
    pipeline: Pipeline,
    examples: list[CounterfactualExample] | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    source_pipeline: Pipeline | None = None,
) -> InterchangeBatch:
    """Tokenize a batch and resolve every unit's base and source positions.

    ``source_pipeline`` (cross-model patching) changes only where the *source*
    side is tokenized and indexed: a different model means a different tokenizer,
    so the counterfactual positions must be computed against it, while the base
    stays on ``pipeline``.
    """
    source_pipeline = source_pipeline or pipeline
    raw_base = [ex["input"] for ex in examples]
    raw_sources = [
        list(cf_tuple)
        for cf_tuple in zip(*[ex["counterfactual_inputs"] for ex in examples])
    ]

    base_encoding = pipeline.load(raw_base)
    source_encodings = [source_pipeline.load(group) for group in raw_sources]

    units = interchange_target.flatten()
    group_sizes = [len(group) for group in interchange_target]

    base_positions = [
        unit.resolve_positions(
            raw_base,
            attention_mask=base_encoding["attention_mask"],
            is_original=True,
        )
        for unit in units
    ]
    source_positions = [
        unit.resolve_positions(
            raw_sources[group_index],
            attention_mask=source_encodings[group_index]["attention_mask"],
            is_original=False,
        )
        for group_index, group in enumerate(interchange_target)
        for unit in group
    ]

    # Validate on the full index structures (which carry the head axis for
    # per-head units), not the extracted positions.
    _validate_positions(
        units,
        [
            unit.index_component(
                raw_base,
                batch=True,
                is_original=True,
                attention_mask=base_encoding["attention_mask"],
            )
            for unit in units
        ],
        [
            unit.index_component(
                raw_sources[group_index],
                batch=True,
                is_original=False,
                attention_mask=source_encodings[group_index]["attention_mask"],
            )
            for group_index, group in enumerate(interchange_target)
            for unit in group
        ],
    )

    return InterchangeBatch(
        base_encoding=base_encoding,
        source_encodings=source_encodings,
        base_positions=base_positions,
        source_positions=source_positions,
        feature_indices=[unit.get_feature_indices() for unit in units],
        units=units,
        group_sizes=group_sizes,
    )


def collect_group_sources(
    source_pipeline: Pipeline, batch: InterchangeBatch
) -> list[Tensor]:
    """Read every unit's source activation, one forward per counterfactual group.

    Groups are separate forwards because each has its own counterfactual input —
    the same reason the pyvene backbone ran one collection pass per source.

    Read ``raw``: the interchange intervention featurizes the source itself, so
    reading through the unit's featurizer here would apply it twice.
    """
    sources: list[Tensor] = []
    offset = 0
    for group_index, size in enumerate(batch.group_sizes):
        units = batch.units[offset : offset + size]
        positions = batch.source_positions[offset : offset + size]
        offset += size
        plans = build_plans(units, positions, "collect", raw=True)
        collected = collect_unit_activations(
            source_pipeline, batch.source_encodings[group_index], plans
        )
        assert isinstance(collected, list)
        sources.extend(collected)
    return sources


def batched_interchange_intervention(
    pipeline: Pipeline,
    intervenable_model: IntervenableModel,
    examples: list[CounterfactualExample],
    interchange_target: InterchangeTarget,
    output_scores: bool | int = True,
    source_pipeline: Pipeline | None = None,
    source_intervenable_model: IntervenableModel | None = None,
) -> dict[str, Any]:
    """
    Perform interchange interventions on batched inputs using an intervenable model.

    This function executes the core intervention logic by:
    1. Preparing the base and counterfactual inputs for intervention
    2. Running the model with interventions at specified locations
    3. Moving tensors back to CPU to free GPU memory

    Args:
        pipeline: Target pipeline where interventions are applied
        intervenable_model: PyVENE model with preset intervention locations
        examples: List of counterfactual examples
        interchange_target: InterchangeTarget containing model components to intervene on
        output_scores: Whether to include scores in output dictionary (default: True)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching.
        source_intervenable_model: Optional pre-created intervenable model for the source
            pipeline. If provided with source_pipeline, this model will be reused for
            collecting activations instead of creating a new one per batch.

    Returns:
        dict: Dictionary with 'sequences' and optionally 'scores' keys
    """
    try:
        # Prepare inputs for intervention (using target pipeline for base)
        batched_base, batched_counterfactuals, inv_locations, feature_indices = (
            prepare_intervenable_inputs(pipeline, examples, interchange_target)
        )

        # Collect source representations if using cross-model patching
        source_representations = None
        if source_pipeline is not None:
            source_representations = collect_source_representations(
                source_pipeline,
                examples,
                interchange_target,
            )
            # When using cross-model patching, we don't pass counterfactuals to pyvene
            # because we're using pre-collected source_representations instead
            batched_counterfactuals = None

        # Execute the intervention via the pipeline
        gen_kwargs = {"output_scores": output_scores}
        output = pipeline.intervenable_generate(
            intervenable_model,
            batched_base,
            batched_counterfactuals,
            inv_locations,
            feature_indices,
            source_representations=source_representations,
            **gen_kwargs,
        )

        # Move tensors to CPU to free GPU memory
        if batched_counterfactuals is not None:
            for batched in [batched_base] + batched_counterfactuals:
                for k, v in batched.items():
                    batched[k] = v.cpu()
        else:
            for k, v in batched_base.items():
                batched_base[k] = v.cpu()

        return output
    finally:
        delete_intervenable_model(intervenable_model)


def interchange_batch(
    pipeline: Pipeline,
    examples: list[CounterfactualExample] | list[LabeledCounterfactualExample],
    interchange_target: InterchangeTarget,
    output_scores: bool | int = True,
    source_pipeline: Pipeline | None = None,
) -> dict[str, Any]:
    """Swap each unit's source activation into the base run, for one batch.

    Two forwards per counterfactual group plus one generation: the sources are
    read first (from ``source_pipeline`` when patching across models), then
    written into the base's prefill. See :mod:`causalab.neural.activations.engine`
    for why the sources are a separate pass rather than a batched invoke.
    """
    batch = prepare_interchange_batch(
        pipeline, examples, interchange_target, source_pipeline
    )
    sources = collect_group_sources(source_pipeline or pipeline, batch)
    plans = build_plans(
        batch.units,
        batch.base_positions,
        "interchange",
        sources=sources,
        feature_indices=batch.feature_indices,
    )
    result = generate_with_interventions(
        pipeline, batch.base_encoding, plans, output_scores=bool(output_scores)
    )
    return pipeline.format_generation(result, batch.base_encoding, output_scores)


def run_interchange_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: list[CounterfactualExample],
    interchange_target: InterchangeTarget,
    batch_size: int = 32,
    output_scores: bool | int = True,
    source_pipeline: Pipeline | None = None,
) -> dict[str, list[Any]]:
    """
    Run interchange interventions on a full counterfactual dataset in batches.

    This function:
    1. Prepares an intervenable model configured for interchange interventions
    2. Processes the dataset in batches, applying interventions to each batch
    3. Converts scores to top-k format if requested (for memory efficiency)
    4. Moves all outputs to CPU to free GPU memory
    5. Collects and returns results from all batches

    Args:
        pipeline: Target pipeline where interventions are applied
        counterfactual_dataset: List of counterfactual examples
        interchange_target: InterchangeTarget containing model components to intervene on,
                           where groups share counterfactual inputs
        batch_size: Number of examples to process in each batch
        output_scores: Controls score output format:
            - False: No scores
            - True: Full vocabulary scores (on CPU)
            - int (e.g., 10): Top-k scores (on CPU, memory efficient)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching where activations
            from source_pipeline are patched into pipeline (the target).

    Returns:
        List[dict]: List of dictionaries, each with 'sequences' (on CPU) and optionally
                   'scores' keys (on CPU, in top-k format if int was provided)
    """
    all_outputs = []

    # Process each batch with progress tracking
    for start in tqdm(
        range(0, len(counterfactual_dataset), batch_size),
        desc="Processing batches",
        disable=not logger.isEnabledFor(logging.DEBUG),
        leave=False,
    ):
        examples = counterfactual_dataset[start : start + batch_size]
        with torch.no_grad():  # Disable gradient tracking for inference
            all_outputs.append(
                interchange_batch(
                    pipeline,
                    examples,
                    interchange_target,
                    output_scores=output_scores,
                    source_pipeline=source_pipeline,
                )
            )

    # Convert to top-k format if requested (while still on GPU for efficiency)
    if not isinstance(output_scores, bool) and output_scores > 0:
        all_outputs = convert_to_top_k(all_outputs, pipeline, k=output_scores)

    # Move all outputs to CPU
    all_outputs = move_outputs_to_cpu(all_outputs)

    # remove batch structure from outputs
    if not all_outputs:
        return {"sequences": [[]], "string": [[]]}
    all_outputs = {
        k: [output[k] for output in all_outputs] for k in all_outputs[0].keys()
    }

    return all_outputs


def run_two_pass_path_patching(
    pipeline: Pipeline,
    counterfactual_dataset: list[CounterfactualExample],
    interchange_target: InterchangeTarget,
    receiver: AtomicModelUnit | list[AtomicModelUnit],
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> dict[str, list[Any]]:
    """Two-pass path patching for one or a *set* of internal receivers.

    Unlike the degenerate ``receiver = output`` case (a single intervened forward,
    handled by :func:`run_interchange_interventions`), an internal receiver needs two
    passes, because the sender's perturbation also reaches the output through routes
    that *bypass* the receiver (the sender's own residual path, and sender→other
    downstream components). Those must be stripped out:

    * **PASS 1** runs one forward in which the sender reads its source and the
      restorers read the base (the ``interchange_target`` groups), and *collects* the
      receivers' activations ``v*`` — the values the receivers read when only the direct
      sender→receiver path is perturbed. Hook firing order (forward order) makes this
      exact: by the time execution reaches each receiver, every upstream restorer has
      already overwritten its base value.
    * **PASS 2** re-runs on the **clean base** (sender un-perturbed) with only the
      receivers' activations replaced by ``v*`` (injected via ``source_representations``,
      reusing the cross-model patching path). Every sender→output route that bypasses
      the receivers is therefore clean, leaving exactly the sender→receiver edge effect.

    ``receiver`` may be a single :class:`AtomicModelUnit` or a list of them. A set is
    patched *simultaneously* (collected and injected in one shot) — not summed over
    single receivers — because the metric is a nonlinear function of the patched
    activations and the receivers interact downstream (the IOI Fig. 4 / Fig. 5 case:
    one sender into all name-mover queries or all S-inhibition values at once). The
    same ``receiver`` list drives both passes, and the engine returns activations in
    the order it was given units, so PASS 2 injects what PASS 1 collected with no
    reordering step to get wrong.

    PASS 1 mixes roles in one forward — the sender/restorer units write while the
    receivers read. Forward order is what makes it exact, and the engine visits
    sites in that order, so each receiver is read only after every upstream
    restorer has written.

    ``counterfactual_dataset`` must already be presented so each example's
    ``counterfactual_inputs`` lead with the source (group 0) followed by the base for
    each restorer group — i.e. the ``[source, base]`` arrangement the path-patching
    runner builds. Returns the same ``{"string", "sequences", "scores"}`` dict shape as
    :func:`run_interchange_interventions`, so existing scoring applies unchanged.
    """
    receivers = list(receiver) if isinstance(receiver, (list, tuple)) else [receiver]

    all_outputs: list[dict[str, Any]] = []
    for start in tqdm(
        range(0, len(counterfactual_dataset), batch_size),
        desc="path-patch two-pass",
        disable=not logger.isEnabledFor(logging.DEBUG),
        leave=False,
    ):
        examples = counterfactual_dataset[start : start + batch_size]
        with torch.no_grad():
            batch = prepare_interchange_batch(pipeline, examples, interchange_target)
            raw_base = [ex["input"] for ex in examples]
            receiver_positions = [
                r.resolve_positions(
                    raw_base,
                    attention_mask=batch.base_encoding["attention_mask"],
                    is_original=True,
                )
                for r in receivers
            ]
            # The receivers bypass prepare_interchange_batch's guards (they are
            # not interchange groups), so validate them on the same path — a
            # ragged receiver span would otherwise fail deep in an index build.
            receiver_indices = [
                r.index_component(
                    raw_base,
                    batch=True,
                    is_original=True,
                    attention_mask=batch.base_encoding["attention_mask"],
                )
                for r in receivers
            ]
            _validate_positions(
                receivers,
                receiver_indices,
                receiver_indices,
                role="path-patching receiver",
            )

            # PASS 1 — write the sender + restorers, read v* at the receivers.
            sources = collect_group_sources(pipeline, batch)
            write_plans = build_plans(
                batch.units,
                batch.base_positions,
                "interchange",
                sources=sources,
                feature_indices=batch.feature_indices,
            )
            read_plans = build_plans(receivers, receiver_positions, "collect", raw=True)
            v_star = collect_unit_activations_under(
                pipeline, batch.base_encoding, write_plans, read_plans
            )

            # PASS 2 — inject v* on an otherwise-clean base run.
            inject_plans = build_plans(
                receivers,
                receiver_positions,
                "interchange",
                sources=v_star,
                feature_indices=[r.get_feature_indices() for r in receivers],
            )
            result = generate_with_interventions(
                pipeline,
                batch.base_encoding,
                inject_plans,
                output_scores=bool(output_scores),
            )
            all_outputs.append(
                pipeline.format_generation(result, batch.base_encoding, output_scores)
            )

    if not isinstance(output_scores, bool) and output_scores > 0:
        all_outputs = convert_to_top_k(all_outputs, pipeline, k=output_scores)
    all_outputs = move_outputs_to_cpu(all_outputs)
    if not all_outputs:
        return {"sequences": [[]], "string": [[]]}
    return {k: [output[k] for output in all_outputs] for k in all_outputs[0].keys()}
