"""Batched dataset execution — PL3 (#405); spec-typed surface — WU3 (#505).

The layer between the Plan compiler (PL1/PL2) and the public wrappers: it
takes a *counterfactual dataset* (paired base / counterfactual examples), the
declarative spec surface (:class:`~causalab.neural.specs.SiteSpec` /
:class:`~causalab.neural.specs.EditSpec`, WU1 #503), and runs the whole
dataset in batches on the nnsight backbone:

* **One tokenization per batch side** — the batch's run encoding
  (``pipeline.load(traces, return_offsets_mapping=True)``) is both what the
  model runs and what positions resolve against
  (:func:`causalab.neural.positions.resolve_positions_batched`), so per-row
  indices are born in the padded frame. Ragged spans batch as flat
  gather/scatters (:class:`causalab.neural.site.RaggedIndex`) — no
  length-bucketing, no per-example fallback.
* **Split-forward layout, derived by the engine** (EU4, #485): generation
  builds ONE generation :class:`~causalab.neural.plan.Plan` per batch — an
  :class:`~causalab.neural.plan.EditOp` on the base input per
  :class:`~causalab.neural.specs.EditSpec` (:func:`_edit_spec_to_edit`),
  each source-needing group's counterfactual batch as its own plan input
  (:func:`cf_input_key`) read through cross-input ``ReadSource``s, and a
  :class:`~causalab.neural.plan.GenerateSpec` — and
  :func:`causalab.neural.plan.run_plan` lowers it to exactly the
  pyvene-parity layout this module used to hand-code: one fused,
  early-stopped collect stage per counterfactual group, then ONE terminal
  ``model.generate`` trace with every edit applied during the prefill —
  measured (and pinned by the tests) to persist through cached decode steps
  and to match a raw-hook oracle exactly. Collect-only execution lowers
  onto the same :func:`~causalab.neural.plan.run_plan` (fused,
  early-stopped).
* **One flat output shape** (EU5a, #486): generation returns the unified
  :class:`~causalab.neural.pipeline.GenerationResult` — flattened across the
  internal batches, so the batch split never leaks into the result; feature
  collection returns ``{spec.key: (n_samples, n_features)}`` rows in
  example-major order.

The public wrappers (``collect_features``, ``run_interchange_interventions``,
``run_steering_interventions``, ``run_ablation``, ``run_causal_trace``)
reroute onto these two entry points; cross-model patching (a source captured
on a *different* model, SH2 #411) rides
:attr:`~causalab.neural.plan.Plan.models` — the ``source_pipeline``
threading.

Design + as-landed record: ``docs/REBASE_CAUSALAB_ON_NNTERP.md`` Part 5
("engine unification", #480) and Part 6 ("where-unification", #491) — this
module's reroute is EU4 (#485); its spec-typed surface is WU3 (#505); the
legacy unit vocabulary and its boundary coercions were deleted by the WU6
sweep (#508).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from causalab.neural import modes
from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.modes import SeededNoise
from causalab.neural.pipeline import (
    GenerationResult,
    LMPipeline,
    ensure_position_ids,
    right_pad_sequences,
)
from causalab.neural.plan import CollectOp, EditOp, GenerateSpec, Plan, run_plan
from causalab.neural.positions import resolve_positions_batched
from causalab.neural.site import forward_key
from causalab.neural.specs import EditSpec, SiteSpec

__all__ = [
    "cf_input_key",
    "collect_dataset_features",
    "forward_inputs",
    "resolve_spec_positions",
    "run_intervened_generation",
]


def _edit_spec_to_edit(
    edit: EditSpec,
    fsite: FeaturizedSite,
    positions: list[list[int]],
    source_input: str | None,
    source_positions: list[list[int]] | None,
    batch_lo: int,
    batch_hi: int,
    noise_stream: SeededNoise | None = None,
) -> Edit:
    """THE single spec→engine conversion point (the renamed
    ``_unit_edit_to_edit``, WU3 #505): one :class:`~causalab.neural.specs.
    EditSpec` on one batch → one writing :class:`~causalab.neural.edit.Edit`,
    through the ED2 mode constructor its ``mode`` names —
    :mod:`causalab.neural.modes` stays the single Edit vocabulary (no
    parallel mode table here; the constructors also carry the
    construction-time ``_check_width`` guard, so a feature-width mismatch
    fails legibly before any trace). No other function translates an
    ``EditSpec`` into an ``Edit``.

    ``positions`` are the edit's resolved base-side rows (``Edit.positions``,
    the base encoding's frame). The source-needing modes (interchange /
    interpolate) read the SAME featurized site under the plan input
    ``source_input`` at ``source_positions`` (resolved in the counterfactual
    encoding's own frame) via a cross-input
    :class:`~causalab.neural.edit.ReadSource` — the read the scheduler
    force-stages into the group's collect pass. Vector-fed modes slice/shape
    their vector for this batch here, at plan-build time. ``"add"`` is
    :func:`modes.steer` with ``factor=scale``; ``"replace"`` passes ``scale``
    through to the constructor (which owns the rank-preserving
    ``expand_as``); ``"noise"`` with ``vector`` set uses it as a per-feature
    noise scale (times ``scale`` — the causal-tracing ROME ``noise_scale·σ``
    contract).

    ``noise_stream`` is the run-shared :class:`SeededNoise` instance for this
    edit's ``seed`` — built once per distinct seed per run, *before* the
    batch loop, so draws advance across edits and batches. Passing the raw
    ``edit.seed`` int to :func:`modes.noise` here would re-seed a fresh
    stream per plan build, repeating the same noise across batches (the
    hazard documented on :class:`SeededNoise`); this function refuses to do
    that."""
    if edit.mode == "interchange":
        return modes.interchange(
            fsite,
            fsite,
            source_positions=source_positions,
            source_input=source_input,
            positions=positions,
        )
    if edit.mode == "interpolate":
        return modes.interpolate(
            fsite,
            fsite,
            edit.interpolate_fn,  # type: ignore[arg-type]
            source_positions=source_positions,
            source_input=source_input,
            positions=positions,
            **edit.interpolate_params,
        )
    if edit.mode == "noise":
        if noise_stream is None:
            raise ValueError(
                f"edit {edit.site.key!r}: noise edits must be lowered with a "
                "run-shared SeededNoise stream (one per distinct seed, built "
                "before the batch loop) — re-seeding from the raw int per "
                "batch would repeat the same noise across batch boundaries "
                "(see causalab.neural.modes.SeededNoise)."
            )
        noise_scale: Any = edit.scale
        if edit.vector is not None:
            noise_scale = edit.scale * _values_for_rows(
                edit.vector, positions, batch_lo, batch_hi
            )
        return modes.noise(
            fsite,
            noise_scale,
            seed=noise_stream,
            positions=positions,
        )

    value = _values_for_rows(edit.vector, positions, batch_lo, batch_hi)  # type: ignore[arg-type]
    if edit.mode == "replace":
        return modes.replace(fsite, value, scale=edit.scale, positions=positions)
    return modes.steer(  # "add": f + scale·v
        fsite, value, factor=edit.scale, positions=positions
    )


# --------------------------------------------------------------------------- #
#  shared helpers                                                              #
# --------------------------------------------------------------------------- #
_MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "position_ids")


def _model_inputs(encoding: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The tensor keys a forward/generate consumes — resolution-only extras
    (offset_mapping, chat metadata) must not reach the model."""
    return {k: encoding[k] for k in _MODEL_INPUT_KEYS if k in encoding}


def forward_inputs(encoding: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Model inputs for a *plain forward* (collect passes): left-padded rows
    on absolute-position models need a prompt-shaped ``position_ids`` or every
    activation in them is numbered from the pad tokens (no-op for RoPE models
    and for multi-step generate, which numbers its own steps). Also strips the
    resolution-only extras (offset_mapping, chat metadata) a ``pipeline.load``
    encoding carries — those must not reach the model."""
    return ensure_position_ids(_model_inputs(encoding))


def cf_input_key(g: int) -> str:
    """The plan-input key for counterfactual group ``g`` — the naming half of
    the grouping contract (Part 6 "Grouping"): group ``g`` of ``groups``
    reads its sources from ``example["counterfactual_inputs"][g]``, which
    :func:`run_intervened_generation` binds to plan input ``cf_input_key(g)``
    (cross-input ``ReadSource``s force-staged into one collect stage per
    group — the EU3/EU4 machinery)."""
    return f"cf_{g}"


def resolve_spec_positions(
    spec: SiteSpec,
    traces: Sequence[Any],
    encoding: Any,
    *,
    is_original: bool | None = None,
) -> list[list[int]]:
    """A spec's token positions over ``traces``, in ``encoding``'s padded
    frame — the spec-level face of
    :func:`causalab.neural.positions.resolve_positions_batched` (WU3 #505).

    Resolves :attr:`~causalab.neural.specs.SiteSpec.positions` (a declarative
    resolver or literal rows) against the batch's own run encoding
    (``pipeline.load(traces, return_offsets_mapping=True)``), so rows are
    born in the frame the model runs on. ``is_original`` routes paired
    base/counterfactual resolvers to the right side. Rows may be ragged.

    ``spec.positions is None`` — an *unbound* spec, e.g. one loaded via
    :func:`~causalab.neural.specs.load_site_specs` without its
    ``token_positions`` mapping — is **refused loudly**. The dataset entry
    points deliberately do not read ``None`` as "the whole sequence": on a
    padded batch that would silently include pad positions (and misalign
    base/counterfactual pairs of different lengths). Bind positions first —
    construct the spec with a resolver or literal rows, derive a view with
    ``spec.with_positions(...)``, or reload the bundle with
    ``load_site_specs(dir, token_positions=...)``.
    """
    if spec.positions is None:
        raise ValueError(
            f"spec {spec.key!r} has positions=None (unbound): the dataset "
            "entry points require every spec to say where on the sequence "
            "axis it reads/writes — None is NOT read as 'the whole "
            "sequence' (on a padded batch that would silently include pad "
            "positions). Bind positions via spec.with_positions(...), "
            "literal rows, or load_site_specs(dir, token_positions=...)."
        )
    return resolve_positions_batched(
        spec.positions, traces, encoding, is_original=is_original
    )


def _place_spec_featurizer(pipeline: LMPipeline, spec: SiteSpec) -> None:
    """Move the spec's featurizer modules to its site's layer device.

    The pyvene model build did this implicitly (per-key ``intervention.to``
    from the ``hf_device_map``); on the engine it is one explicit pass,
    mirroring ``trainable.place_edit_parameters``. In-place and idempotent —
    shipped featurizers that already coerce per call are unaffected."""
    from causalab.neural.pipeline import device_for_layer

    device = device_for_layer(pipeline, spec.fsite.site.layer)  # type: ignore[attr-defined]
    spec.fsite.featurizer.featurizer.to(device)
    spec.fsite.featurizer.inverse_featurizer.to(device)


def _widths(rows: Sequence[Sequence[int]]) -> list[int]:
    return [len(row) for row in rows]


def _check_pairwise_widths(
    key: str,
    base_rows: Sequence[Sequence[int]],
    source_rows: Sequence[Sequence[int]],
) -> None:
    """Interchange writes source features onto base positions example-by-
    example, so each example must contribute the same number of positions on
    both sides — otherwise the flat forms would silently misalign."""
    base_w, source_w = _widths(base_rows), _widths(source_rows)
    if base_w != source_w:
        mismatch = next(i for i, (b, s) in enumerate(zip(base_w, source_w)) if b != s)
        raise ValueError(
            f"site {key!r}: base and counterfactual position widths differ "
            f"for example {mismatch} ({base_w[mismatch]} vs {source_w[mismatch]}). "
            "An interchange pairs positions example-by-example; ragged widths "
            "may differ across examples but not between a pair's sides."
        )


def _values_for_rows(
    vector: torch.Tensor,
    rows: Sequence[Sequence[int]],
    batch_lo: int,
    batch_hi: int,
) -> torch.Tensor:
    """Shape a feature-space vector for a positional write at ``rows``.

    ``(n_features,)`` broadcasts as-is over every selected position (both the
    ``(batch, k, d)`` equal-width and flat ragged ``(total, d)`` forms).
    ``(n_examples, n_features)`` is sliced to this batch and expanded to one
    row per selected position: ``(batch, 1, d)`` for equal-width rows, or
    ``repeat_interleave`` by per-row widths for ragged rows.
    """
    if vector.dim() == 1:
        return vector
    if vector.dim() != 2:
        raise ValueError(
            f"per-example vectors must be (n_examples, n_features), got shape "
            f"{tuple(vector.shape)}"
        )
    batch_vec = vector[batch_lo:batch_hi]
    widths = _widths(rows)
    if len(set(widths)) > 1:
        return batch_vec.repeat_interleave(
            torch.tensor(widths, device=batch_vec.device), dim=0
        )
    return batch_vec.unsqueeze(1)


def _batches(n: int, batch_size: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + batch_size, n)) for lo in range(0, n, batch_size)]


# --------------------------------------------------------------------------- #
#  collect at scale                                                            #
# --------------------------------------------------------------------------- #
def collect_dataset_features(
    pipeline: LMPipeline,
    dataset: Sequence[Mapping[str, Any]],
    sites: Sequence[SiteSpec],
    batch_size: int = 32,
    collect_output_logits: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Collect every site's features over the dataset's *base* inputs.

    ``sites`` is a flat sequence of :class:`~causalab.neural.specs.SiteSpec`.
    Every spec must carry bound positions — ``positions=None`` is refused
    (see :func:`resolve_spec_positions`). Duplicate ``spec.key``\\s are
    refused: the result is keyed by ``spec.key`` and duplicates would
    silently merge.

    One ``pipeline.load`` and one fused, early-stopped forward per batch
    (:func:`run_plan` with one ``CollectOp`` per spec; saving logits keeps
    the full forward). Returns ``{spec.key: (n_samples, n_features)}`` with
    rows in example-major order — ``n_samples = Σ positions per example``
    (``n_examples × k`` when every example selects ``k`` positions) — the
    ``collect_features`` contract. With ``collect_output_logits`` also
    returns each example's full ``(seq, vocab)`` logits.
    """
    specs = list(sites)
    keys = [spec.key for spec in specs]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"duplicate site keys in sites: {dupes}")
    # Pin eval mode (dropout/batch-norm leakage guard): HF loads eval, but a
    # caller-trained or pre-loaded model may arrive in train mode — the old
    # collect_features asserted this and the engine must too (#449 finding 5).
    pipeline.model.eval()
    for spec in specs:
        _place_spec_featurizer(pipeline, spec)

    per_key: dict[str, list[torch.Tensor]] = {spec.key: [] for spec in specs}
    logits_rows: list[torch.Tensor] = []
    for lo, hi in _batches(len(dataset), batch_size):
        traces = [example["input"] for example in dataset[lo:hi]]
        encoding = pipeline.load(traces, return_offsets_mapping=True)
        ops = []
        for spec in specs:
            rows = resolve_spec_positions(spec, traces, encoding, is_original=True)
            ops.append(CollectOp("base", spec.fsite, key=spec.key, positions=rows))
        plan = Plan(
            inputs={"base": forward_inputs(encoding)},
            ops=tuple(ops),
            save_logits=("base",) if collect_output_logits else (),
        )
        with torch.no_grad():
            result = run_plan(pipeline.model, plan)
        for spec in specs:
            value = result.collects[spec.key]
            per_key[spec.key].append(value.reshape(-1, value.shape[-1]))
        if collect_output_logits:
            logits_rows.extend(row for row in result.logits["base"])

    features = {key: torch.cat(chunks, dim=0) for key, chunks in per_key.items()}
    if collect_output_logits:
        return features, logits_rows
    return features


# --------------------------------------------------------------------------- #
#  intervened generation at scale                                              #
# --------------------------------------------------------------------------- #
def run_intervened_generation(
    pipeline: LMPipeline,
    dataset: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[EditSpec]],
    *,
    batch_size: int = 32,
    output_scores: bool = True,
    source_pipeline: LMPipeline | None = None,
    **gen_kwargs: Any,
) -> GenerationResult:
    """Run every example through one intervened generation, in batches.

    **Grouping contract** (Part 6 "Grouping"): the outer index of ``groups``
    picks which counterfactual input feeds the group's source reads — group
    ``g`` reads ``example["counterfactual_inputs"][g]``, bound to plan input
    :func:`cf_input_key(g) <cf_input_key>` — so ``len(groups)`` must match
    ``len(example["counterfactual_inputs"])`` on every example (a group with
    no source-needing edits never touches its slot). Every edit across all
    groups applies simultaneously in the same base run.

    ``groups`` is a nested sequence of
    :class:`~causalab.neural.specs.EditSpec`. Every edit's spec must carry
    bound positions — ``positions=None`` is refused (see
    :func:`resolve_spec_positions`).

    **Empty-selection contract**: an empty feature selection (e.g. a DBM
    mask that switched every feature off) is a no-op edit. It is not
    constructible as a spec (``FeaturizedSite`` refuses empty
    ``feature_ids``) — express a no-op by omitting the edit.

    **Noise lowering rule** (#505): each noise edit draws from ONE
    :class:`SeededNoise` stream per distinct ``seed`` per call, built
    *before* the batch loop and passed to :func:`modes.noise
    <causalab.neural.modes.noise>` as the instance — never the raw int per
    batch (which would re-seed per plan build and repeat the same noise
    across batch boundaries). Draws therefore advance across edits *and*
    batches.

    Per batch this builds ONE generation
    :class:`~causalab.neural.plan.Plan` — an
    :class:`~causalab.neural.plan.EditOp` on the base input per
    :class:`~causalab.neural.specs.EditSpec` (:func:`_edit_spec_to_edit`),
    each source-needing group's counterfactual batch as its own plan input
    (:func:`cf_input_key`) read through cross-input ``ReadSource``s, and a
    :class:`~causalab.neural.plan.GenerateSpec` carrying the generation
    knobs — and :func:`~causalab.neural.plan.run_plan` derives the
    split-forward layout this function hand-coded before EU4 (#485): one
    fused, early-stopped collect stage per source-needing group, then ONE
    terminal ``model.generate`` trace on the base batch with all edits
    applied during the prefill — the same layout pyvene used, so cost is
    unchanged. EditOps are declared in the retired path's exact apply order
    (the stable sort below), so same-site write order and shared
    :class:`SeededNoise` draw order stay bit-identical.

    ``source_pipeline`` makes the source reads **cross-model** (SH2, #411):
    counterfactual inputs are tokenized by the source pipeline and bound to
    *its* model via :attr:`~causalab.neural.plan.Plan.models`, so each
    group's collect stage runs there and the features captured there are
    written into ``pipeline``'s base run (the featurize happens once, at the
    source read — the engine's constant machinery carries the values across
    with device/dtype coercion at the consuming site). Positions still come
    from each spec's own resolver, per tokenization side.

    Returns ONE flat :class:`~causalab.neural.pipeline.GenerationResult`
    (EU5a, #486) — ``sequences`` ``(n_examples, max_new_tokens)`` CPU,
    ``strings`` always a list, per-step ``scores`` ``(n_examples, vocab)``
    when ``output_scores`` (``None`` otherwise) — concatenated across the
    internal batches, so the batch split never leaks into the result.
    Sequences keep the fixed **pipeline** ``max_new_tokens`` width even
    under a ``gen_kwargs`` override — the deliberate legacy width contract
    (:func:`~causalab.neural.pipeline.right_pad_sequences` — early EOS is
    right-padded with ``pad_token_id``, an over-budget override is
    truncated back while ``scores`` keeps every generated step). Flattening
    ``scores`` requires every batch to generate the same number of steps;
    a ragged early-EOS split across batches is refused loudly (the legacy
    batch-nested shape silently mis-aligned or crashed downstream on it).
    A run whose groups carry no edits at all is the un-intervened baseline
    — deliberately not a Plan (an op-less plan is refused) — and takes
    :func:`_plain_generate` instead.
    """
    model = pipeline.model
    src_pipeline = source_pipeline if source_pipeline is not None else pipeline
    src_model = src_pipeline.model
    # Pin eval mode on both sides (see collect_dataset_features).
    model.eval()
    src_model.eval()
    for group in groups:
        for edit in group:
            _place_spec_featurizer(pipeline, edit.site)

    # The noise lowering rule (#505): ONE SeededNoise per distinct seed per
    # call, built BEFORE the batch loop so draws advance across edits and
    # batches (modes.noise would otherwise re-seed a fresh stream per plan
    # build — the repeat-noise hazard documented on SeededNoise).
    noise_stream_of: dict[int, SeededNoise] = {}
    stream_per_seed: dict[int, SeededNoise] = {}
    for group in groups:
        for edit in group:
            if edit.mode == "noise":
                seed = int(edit.seed)  # noise __post_init__ defaulted it
                stream = stream_per_seed.setdefault(seed, SeededNoise(seed))
                noise_stream_of[id(edit)] = stream

    gen_kwargs = dict(gen_kwargs)
    max_new_tokens = int(gen_kwargs.pop("max_new_tokens", pipeline.max_new_tokens))
    spec: GenerateSpec | None = None
    if any(groups):
        # The legacy generate defaults: max_new_tokens / output_scores are
        # GenerateSpec fields, pad_token_id rides kwargs (caller-overridable,
        # like every remaining HF knob); the emitter owns do_sample=False,
        # return_dict_in_generate=True and use_cache=True.
        spec = GenerateSpec(
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            kwargs={"pad_token_id": pipeline.tokenizer.pad_token_id, **gen_kwargs},
        )

    seq_batches: list[torch.Tensor] = []
    strings: list[str] = []
    score_batches: list[list[torch.Tensor]] = []

    for lo, hi in _batches(len(dataset), batch_size):
        batch = dataset[lo:hi]
        base_traces = [example["input"] for example in batch]
        base_encoding = pipeline.load(base_traces, return_offsets_mapping=True)
        base_inputs = _model_inputs(base_encoding)
        if max_new_tokens == 1:
            # The prefill-only case is the one shape where a prompt-shaped
            # position_ids is exactly right (see pipeline.ensure_position_ids);
            # multi-step generate numbers its own per-step positions and the
            # engine refuses the key up front (plan._check_generate_inputs).
            base_inputs = ensure_position_ids(base_inputs)

        if spec is None:
            gen, step_scores = _plain_generate(
                pipeline,
                base_inputs,
                output_scores=output_scores,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
        else:
            # Build the batch's generation Plan: resolve every edit's
            # positions per side on that side's own run encoding, then one
            # EditOp per EditSpec on the base input. Each source-needing
            # group's counterfactual batch becomes its own plan input, read
            # through cross-input ReadSources the scheduler force-stages
            # into one early-stopped collect trace per group.
            plan_inputs: dict[str, Any] = {"base": base_inputs}
            bound_models: dict[str, Any] = {}
            entries: list[tuple[tuple[int, int, int], Edit]] = []
            for g, group in enumerate(groups):
                source_edits = [edit for edit in group if edit.needs_source]
                cf_key = cf_input_key(g)
                source_rows_of: dict[int, list[list[int]]] = {}
                if source_edits:
                    cf_traces = [
                        example["counterfactual_inputs"][g] for example in batch
                    ]
                    cf_encoding = src_pipeline.load(
                        cf_traces, return_offsets_mapping=True
                    )
                    plan_inputs[cf_key] = forward_inputs(cf_encoding)
                    if source_pipeline is not None:
                        bound_models[cf_key] = src_model
                    for edit in source_edits:
                        source_rows_of[id(edit)] = resolve_spec_positions(
                            edit.site, cf_traces, cf_encoding, is_original=False
                        )

                for order, edit in enumerate(group):
                    fsite = edit.site.fsite
                    base_rows = resolve_spec_positions(
                        edit.site, base_traces, base_encoding, is_original=True
                    )
                    source_rows = None
                    if edit.needs_source:
                        source_rows = source_rows_of[id(edit)]
                        _check_pairwise_widths(edit.site.key, base_rows, source_rows)
                    entries.append(
                        (
                            (*forward_key(fsite.site, model), order),
                            _edit_spec_to_edit(
                                edit,
                                fsite,
                                base_rows,
                                cf_key if edit.needs_source else None,
                                source_rows,
                                lo,
                                hi,
                                noise_stream=noise_stream_of.get(id(edit)),
                            ),
                        )
                    )

            # Declare EditOps in the retired path's exact apply order — the
            # stable sort on (layer, forward rank, within-group order) its
            # sorted writes ran in. The generate trace's tap sort
            # (layer, rank, op-declaration-index) is the identity on this
            # sequence, so same-site write order and shared-SeededNoise draw
            # order stay bit-identical to the retired _generate_with_edits.
            plan = Plan(
                inputs=plan_inputs,
                ops=tuple(
                    EditOp("base", edit)
                    for _, edit in sorted(entries, key=lambda item: item[0])
                ),
                models=bound_models,
                generate=spec,
            )
            with torch.no_grad():
                result = run_plan(model, plan)
            gen = result.sequences["base"]
            step_scores = result.scores["base"] if output_scores else []

        # The fixed-width contract: pad/truncate to the PIPELINE's budget
        # (not a gen_kwargs max_new_tokens override's) — the deliberate
        # legacy width choice (EU4 #485; re-affirmed for EU5a #486).
        sequences = right_pad_sequences(
            gen, pipeline.max_new_tokens, pipeline.tokenizer.pad_token_id
        )
        seq_batches.append(sequences)
        decoded = pipeline.dump(sequences, is_logits=False)
        strings.extend([decoded] if isinstance(decoded, str) else decoded)
        if output_scores:
            score_batches.append(step_scores)

    flat_sequences = (
        torch.cat(seq_batches, dim=0)
        if seq_batches
        else torch.empty((0, pipeline.max_new_tokens), dtype=torch.long)
    )
    flat_scores: list[torch.Tensor] | None = None
    if output_scores:
        step_counts = {len(batch_steps) for batch_steps in score_batches}
        if len(step_counts) > 1:
            raise ValueError(
                "cannot flatten per-step scores: the internal batches "
                f"generated unequal step counts {sorted(step_counts)} (early "
                "EOS stops a batch when all ITS rows finish). Use a single "
                "batch (batch_size >= len(dataset)) or force a fixed length "
                "(e.g. min_new_tokens=max_new_tokens)."
            )
        n_steps = next(iter(step_counts)) if step_counts else 0
        flat_scores = [
            torch.cat([batch_steps[t] for batch_steps in score_batches], dim=0)
            for t in range(n_steps)
        ]
    return GenerationResult(
        sequences=flat_sequences, strings=strings, scores=flat_scores
    )


def _plain_generate(
    pipeline: LMPipeline,
    inputs: dict[str, torch.Tensor],
    *,
    output_scores: bool = True,
    **gen_kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The un-intervened per-batch baseline: ONE traced ``model.generate``
    over ``inputs``, no edits, returning ``(generated tokens, per-step
    scores)`` shaped like the engine emitter's output.

    Deliberately NOT a Plan — a generation plan derives its generated input
    from its ops, so an op-less plan is refused ("an un-intervened
    generation baseline is not a Plan at all",
    :func:`causalab.neural.plan._emit_generate_trace`); this traced fallback
    owns that shape instead. Traced — not ``pipeline.generate``'s raw-HF
    call — so persistent edits compose (:mod:`causalab.neural.persistent`).
    Defaults mirror the Plan reroute's
    :class:`~causalab.neural.plan.GenerateSpec` exactly; the caller applies
    the single-step ``position_ids`` fix before calling.
    """
    defaults: dict[str, Any] = dict(
        max_new_tokens=pipeline.max_new_tokens,
        pad_token_id=pipeline.tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=output_scores,
        do_sample=False,
        use_cache=True,
    )
    defaults.update(gen_kwargs)
    model = pipeline.model
    with torch.no_grad():
        with model.generate(dict(inputs), **defaults):
            out = model.generator.output.save()
    prompt_len = int(inputs["input_ids"].shape[-1])
    sequences = out.sequences[:, prompt_len:].detach().cpu()
    return sequences, [step.detach().cpu() for step in (out.scores or [])]
