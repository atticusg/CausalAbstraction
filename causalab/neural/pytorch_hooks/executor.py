"""Point-protocol execution over raw pytorch hooks (spec §4).

One :class:`PointExecutor` runs one concrete document: forward groups are
``original`` on every input it is read on plus each intervened model on
its declared input; groups run lazily in operand-dependency order (the
validated-acyclic model graph); within a group every in-force write applies
at its address — absolute first, additive deltas summed, renormalize last
against the pre-write norm — and reads see the fully written state because
pytorch fires hooks in module-execution order and chains their return
values at one module.

The hook mechanics mirror the oracle library
(``tests/neural/activations/hook_oracle.py``): reads/writes on a module's
``out`` side ride ``register_forward_hook`` (tuple outputs normalized),
``in``-side taps ride ``register_forward_pre_hook``; writes install before
captures at the same module so a same-address read sees the write.

v1 boundaries, refused legibly rather than approximated: ragged per-row
position widths on an *write* (equal width required to batch the scatter),
and ragged widths on a *saved* read (nothing to stack into a tensor file).
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator, Mapping

import torch

from causalab.neural.pytorch_hooks.encoding import (
    Continuation,
    EncodedBatch,
    encode,
    resolve_position,
    resolve_steps,
    select_field,
)
from causalab.neural.pytorch_hooks.featurizers import (
    FeaturizerStack,
    Stage,
    build_stack,
)
from causalab.neural.pytorch_hooks.loading import ModelBundle
from causalab.neural.pytorch_hooks.mechanisms import (
    apply_absolute,
    apply_delta,
    apply_renormalize,
    is_additive,
    operand_names,
)
from causalab.neural.pytorch_hooks.attention_interface import (
    InterfaceTap,
    attention_interface_taps,
)
from causalab.neural.pytorch_hooks.attention_probs import post_softmax_value_multiply
from causalab.neural.pytorch_hooks.sites import (
    NORMALIZED_TAPS,
    READ_ONLY_COMPONENTS,
    SWAP_ONLY_COMPONENTS,
    ResolvedSite,
    resolve_site,
)
from causalab.neural.pytorch_hooks.layout import (
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)
from causalab.protocol.bundles import entry_selection, selector_slot
from causalab.protocol.errors import ProtocolError
from causalab.protocol.plan import generated_budget
from causalab.protocol.registry import component_width
from causalab.protocol.shapes import FeatureShape
from causalab.protocol.schema import (
    ALL_POSITIONS,
    Document,
    PositionSpec,
    ReadSpec,
    SiteSpec,
    WriteSpec,
)

__all__ = ["PointExecutor", "RaggedValue", "document_seed"]


import dataclasses


def document_seed(doc: Document) -> int:
    """The one seed a document implies: ``train.seed``, or **0** when it
    declares no fit.

    Read in one place so the three consumers cannot drift apart: the
    ``subspace`` featurizer's initial rotation (:func:`build_stack`),
    ``torch.manual_seed`` at train-loop entry, and the batch-order RNG.

    The 0 for a document with no ``train`` block is deliberate rather than
    accidental: an apply/inference document has no seed to name, and pinning
    it means the same document builds the same (unfitted) featurizer whether
    or not a fit is running — which a global-RNG init could not promise."""
    train = doc.train
    if train is None:
        return 0
    return int(train.seed) if isinstance(train.seed, int) else 0


@dataclasses.dataclass(frozen=True)
class RaggedValue:
    """A read over per-row windows of unequal width: the flat
    ``(total_positions, d)`` gather plus per-row widths, re-nestable via
    ``torch.split(flat, widths)`` — the RaggedIndex contract of the old
    resolver, kept as the protocol's ragged-read surface."""

    flat: torch.Tensor
    widths: tuple[int, ...]

    def detach_cpu(self) -> "RaggedValue":
        return RaggedValue(flat=self.flat.detach().cpu(), widths=self.widths)


def _hidden_of(out: Any) -> torch.Tensor:
    """The historical tuple rule. Kept for callers with no tap to consult."""
    return out[0] if isinstance(out, tuple) else out


#: What identifies a tap for capture-sink sharing — see :func:`_tap_key`.
TapKey = tuple[int, str, FeatureShape, int | None, str | None]


def _tap_key(site: "ResolvedSite") -> TapKey:
    """Identity of a tap for capture-sink sharing.

    Two sites may share a module and side yet mean different tensors — a
    different tuple element, or the same tensor read through a different shape
    (round 2's ``attention_gate`` and ``attention_query_pre_rope`` are two
    splits of one projection) — so the shape and tuple index are part of the
    identity. Keying on the module alone would let one tap read another's
    tensor.
    """
    return (
        id(site.module),
        site.kind,
        site.shape,
        site.tuple_index,
        site.interface_slot,
    )


class PointExecutor:
    """Execute one concrete document against one loaded model."""

    def __init__(
        self,
        doc: Document,
        bundle: ModelBundle,
        *,
        role_rows: Mapping[str, list[dict[str, Any]]],
        role_fields: Mapping[str, str],
        load_tensors: Callable[[str], Any],
        stage_cache: dict[str, Stage] | None = None,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
    ) -> None:
        self.doc = doc
        self.bundle = bundle
        self.role_rows = dict(role_rows)
        self.role_fields = dict(role_fields)
        self.load_tensors = load_tensors
        self.stage_cache: dict[str, Stage] = (
            stage_cache if stage_cache is not None else {}
        )
        self.grad_enabled = grad_enabled
        # the seed every freshly built featurizer initialises from; the stage
        # cache is keyed by name alone, so it belongs to this one point
        self.seed = document_seed(doc)
        #: this point's sweep coordinates — they select the matching entry of
        #: a swept bundle a loaded featurizer/param points at (§2.5)
        self.coords = dict(coords or {})
        self._read_values: dict[str, torch.Tensor | RaggedValue] = {}
        self._groups_run: set[tuple[str, str]] = set()
        self._batches: dict[str, EncodedBatch] = {}
        self._continuations: dict[tuple[str, str], Continuation] = {}
        #: per generate read, the decode steps each row addresses — the
        #: same list the gather used, kept because metrics need to know
        #: *which* steps a value covers (and that a row covered none)
        self._read_steps: dict[str, list[list[int]]] = {}

    # ------------------------------------------------------------------ #
    # public surface
    # ------------------------------------------------------------------ #

    def read_value(self, name: str) -> "torch.Tensor | RaggedValue":
        """The (featurized, dims-selected) value of one read; runs its
        group (and, transitively, operand groups) on first use."""
        if name not in self._read_values:
            read = self.doc.reads[name]
            self._run_group(str(read.model), str(read.input))
        return self._read_values[name]

    def dense_value(self, name: str) -> torch.Tensor:
        """A read value that must be a dense tensor (metric inputs): a
        ragged read has no per-example position alignment to reduce."""
        value = self.read_value(name)
        if isinstance(value, RaggedValue):
            raise ProtocolError(
                "P2",
                f"read {name!r} is ragged (unequal per-row position widths) — "
                "metrics reduce one aligned position per example",
            )
        return value

    def is_generated(self, name: str) -> bool:
        """Whether this read addresses the continuation frame (§2.3)."""
        return generated_budget(self.doc, self.doc.reads[name].pos) is not None

    def windowed_value(self, name: str) -> list[torch.Tensor]:
        """One read's value split per example: ``(positions_i, …)`` each.

        The metric surface for a continuation read. Unlike
        :meth:`dense_value` it welcomes ragged widths, because in the
        continuation frame they are the answer rather than a
        misalignment — a row that stopped early, or never said the value a
        ``variable`` anchor looks for, contributes an **empty** tensor.
        """
        value = self.read_value(name)
        if isinstance(value, RaggedValue):
            widths = list(value.widths)
            return list(torch.split(value.flat, widths)) if widths else []
        if value.dim() == 2:  # one position per row, already squeezed
            return [value[i].unsqueeze(0) for i in range(value.shape[0])]
        return [value[i] for i in range(value.shape[0])]

    def addressed_steps(self, name: str) -> list[list[int]]:
        """Per example, the decode steps one generate read covers."""
        if name not in self._read_steps:
            self.read_value(name)  # materialize the group that fills it
        return self._read_steps[name]

    def generated_ids(self, name: str) -> list[list[int]]:
        """Per example, the token ids at a generate read's addressed steps.

        The ``ids`` metric domain (§2.10): these come from the decode
        itself, so a metric that only needs them obliges no vocabulary
        projection anywhere.
        """
        read = self.doc.reads[name]
        steps = self.addressed_steps(name)
        continuation = self._continuations[(str(read.model), str(read.input))]
        return [
            [int(continuation.token_ids[row, step]) for step in row_steps]
            for row, row_steps in enumerate(steps)
        ]

    def run_all(self) -> None:
        """Run every group the document implies (all reads materialize)."""
        for name in self.doc.reads:
            self.read_value(name)

    def stage(self, name: str) -> Stage:
        """The (shared) featurizer stage instance for one declared name."""
        if name not in self.stage_cache:
            build_stack(
                name,
                dict(self.doc.featurizers),
                width=self._featurizer_width(name),
                load_tensors=self.load_tensors,
                stage_cache=self.stage_cache,
                device=self.bundle.device,
                seed=self.seed,
                coords=self.coords,
            )
        return self.stage_cache[name]

    def _featurizer_width(self, name: str) -> int:
        """The input width of one declared featurizer: the site width of a
        chain that uses it, folded through the stages before it (§2.5
        composition — a gate after a k=3 rotation is 3-wide)."""
        from causalab.neural.pytorch_hooks.featurizers import stage_output_width

        for entry in (*self.doc.reads.values(), *self.doc.writes.values()):
            ref = entry.featurizer
            chain = (
                (ref,)
                if isinstance(ref, str)
                else tuple(ref)
                if isinstance(ref, tuple)
                else ()
            )
            if name not in chain:
                continue
            site = resolve_site(self.bundle, self.doc.sites[str(entry.site)])
            running = (
                site.feature_slice.stop - site.feature_slice.start
                if site.feature_slice is not None
                else component_width(self.bundle.info, site.component, head=None)
            )
            for member in chain:
                if member == name:
                    return running
                out = stage_output_width(self.doc.featurizers[member], running)
                if out is None:
                    raise ProtocolError(
                        "P2",
                        f"cannot size {name!r}: {member!r} before it in the "
                        "chain has no spec-derivable output width",
                    )
                running = out
        raise ProtocolError("P2", f"featurizer {name!r} is used by no read or write")

    def rows_for_metrics(self) -> list[dict[str, Any]]:
        """Metric columns resolve against the base rows — the pairing
        anchor (§2.2: rows are paired; one base row + its counterfactuals form one
        example)."""
        return self.role_rows["base"]

    def reset_reads(self) -> None:
        """Drop cached read values and group state (training steps re-run
        forwards with updated featurizer parameters)."""
        self._read_values.clear()
        self._groups_run.clear()
        self._continuations.clear()
        self._read_steps.clear()

    # ------------------------------------------------------------------ #
    # group execution
    # ------------------------------------------------------------------ #

    def _batch(self, role: str) -> EncodedBatch:
        if role not in self._batches:
            rows = self.role_rows[role]
            field = self.role_fields[role]
            texts = [str(select_field(row, field)) for row in rows]
            self._batches[role] = encode(
                self.bundle.tokenizer, texts, device=self.bundle.device
            )
        return self._batches[role]

    def _run_group(self, model: str, input_role: str) -> None:
        if (model, input_role) in self._groups_run:
            return
        # operands first — the acyclic model graph is the schedule skeleton
        write_names: tuple[str, ...] = ()
        if model != "original":
            im = self.doc.intervened_models[model]
            write_names = tuple(im.writes) if isinstance(im.writes, tuple) else ()
            for ename in write_names:
                for operand in operand_names(self.doc.writes[ename].do.payload):
                    if operand in self.doc.reads:
                        self.read_value(operand)

        batch = self._batch(input_role)
        all_taps = [
            (rname, read)
            for rname, read in self.doc.reads.items()
            if str(read.model) == model and str(read.input) == input_role
        ]
        depth = 0
        taps: list[tuple[str, ReadSpec]] = []
        gen_taps: list[tuple[str, ReadSpec]] = []
        for rname, read in all_taps:
            budget = generated_budget(self.doc, read.pos)
            if budget is None:
                taps.append((rname, read))
            else:
                gen_taps.append((rname, read))
                depth = max(depth, budget)

        write_hooks = self._build_write_hooks(write_names, input_role, batch)
        capture: dict[TapKey, torch.Tensor] = {}
        capture_sites = {
            (rname): resolve_site(self.bundle, self.doc.sites[str(read.site)])
            for rname, read in taps
        }
        # A continuation read at lm_head is served from kept ln_final
        # activations (d_model, not vocab) and projected at its addressed
        # steps — the same value, without ever building the whole vocabulary
        # for every step. Any other site is captured as itself.
        gen_sites = {
            rname: resolve_site(self.bundle, self.doc.sites[str(read.site)])
            for rname, read in gen_taps
        }
        gen_capture_sites = {
            rname: (
                resolve_site(self.bundle, SiteSpec(component="ln_final"))
                if site.component == "lm_head"
                else site
            )
            for rname, site in gen_sites.items()
        }

        with contextlib.ExitStack() as hooks:
            # Four of the mixer's tensors — and the attention pattern's *write*
            # — are not module boundaries: transformers computes them inside one
            # `attention_interface(...)` call, so a forward hook on the mixer
            # fires after they have already been consumed. They are collected
            # first and installed together, because the interception is one
            # registry entry and nesting two of them would let the inner
            # wrapper's edits replace the outer's.
            interface: dict[int, list[InterfaceTap]] = {}
            batch_size = batch.input_ids.shape[0]

            for site, fn in write_hooks:
                if site.interface_slot is not None:
                    interface.setdefault(id(site.module), []).append(
                        InterfaceTap(
                            slot=site.interface_slot,
                            edit=_interface_edit(site, fn, batch_size),
                        )
                    )
                    continue
                hooks.enter_context(
                    _installed(
                        site.module,
                        site.kind,
                        fn,
                        shape=site.shape,
                        tuple_index=site.tuple_index,
                        batch_size=batch_size,
                    )
                )
            for site in capture_sites.values():
                key = _tap_key(site)
                if key in capture:
                    continue
                capture[key] = torch.empty(0)  # placeholder; filled by the tap
                if site.kind == "interface":
                    assert site.interface_slot is not None
                    interface.setdefault(id(site.module), []).append(
                        InterfaceTap(
                            slot=site.interface_slot,
                            read=_interface_capture(capture, key, site, batch_size),
                        )
                    )
                    continue
                hooks.enter_context(
                    _capturing(
                        site.module,
                        site.kind,
                        capture,
                        key,
                        shape=site.shape,
                        tuple_index=site.tuple_index,
                        batch_size=batch_size,
                    )
                )
            hooks.enter_context(
                attention_interface_taps(
                    {mid: tuple(entries) for mid, entries in interface.items()},
                    post_softmax=post_softmax_value_multiply,
                )
            )
            with torch.enable_grad() if self.grad_enabled else torch.no_grad():
                prefill = self.bundle.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids(),
                    use_cache=depth > 0,
                )

        for rname, read in taps:
            site = capture_sites[rname]
            raw = capture[_tap_key(site)]
            self._read_values[rname] = self._finalize_read(
                rname, read, site, raw, batch, input_role
            )

        if depth:
            self._decode(
                model,
                input_role,
                batch=batch,
                prefill=prefill,
                depth=depth,
                gen_taps=gen_taps,
                gen_sites=gen_sites,
                gen_capture_sites=gen_capture_sites,
            )
        self._groups_run.add((model, input_role))

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #

    def _decode(
        self,
        model: str,
        input_role: str,
        *,
        batch: EncodedBatch,
        prefill: Any,
        depth: int,
        gen_taps: list[tuple[str, ReadSpec]],
        gen_sites: Mapping[str, ResolvedSite],
        gen_capture_sites: Mapping[str, ResolvedSite],
    ) -> None:
        """Greedy-decode this group's continuation and finalize its reads.

        The prefill above produced the first token; each step here consumes
        the token before it, so ``depth`` tokens need ``depth`` steps and
        every generated position has activations — including the last, whose
        ``lm_head`` value is the distribution *after* it (§2.3).

        **Write hooks are gone by now**: they lived in the prefill's
        ``ExitStack``, which closed before this runs. That is the whole of
        "writes are prefill-only" — an intervention reaches the continuation
        through the first token's logits and through what it left in the KV
        cache, and nothing re-fires per step.
        """
        eos = self.bundle.tokenizer.eos_token_id
        mask = batch.attention_mask
        next_pos = batch.position_ids()[:, -1:]
        nxt = prefill.logits[:, -1:, :].argmax(dim=-1)
        rows = int(mask.shape[0])

        tokens: list[torch.Tensor] = [nxt]
        steps: dict[TapKey, list[torch.Tensor]] = {}
        cache = prefill.past_key_values
        with contextlib.ExitStack() as hooks:
            interface: dict[int, list[InterfaceTap]] = {}
            batch_size = batch.input_ids.shape[0]
            for name, site in gen_capture_sites.items():
                _refuse_unstackable(name, site)
                key = _tap_key(site)
                if key in steps:
                    continue
                steps[key] = []
                if site.kind == "interface":
                    assert site.interface_slot is not None
                    interface.setdefault(id(site.module), []).append(
                        InterfaceTap(
                            slot=site.interface_slot,
                            read=_interface_accumulate(steps[key], site, batch_size),
                        )
                    )
                    continue
                hooks.enter_context(
                    _accumulating(
                        site.module,
                        site.kind,
                        steps[key],
                        shape=site.shape,
                        tuple_index=site.tuple_index,
                        batch_size=batch_size,
                    )
                )
            hooks.enter_context(
                attention_interface_taps(
                    {mid: tuple(entries) for mid, entries in interface.items()},
                    post_softmax=post_softmax_value_multiply,
                )
            )
            for _ in range(depth):
                mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
                next_pos = next_pos + 1
                with torch.enable_grad() if self.grad_enabled else torch.no_grad():
                    out = self.bundle.model(
                        input_ids=nxt,
                        attention_mask=mask,
                        position_ids=next_pos,
                        past_key_values=cache,
                        use_cache=True,
                    )
                cache = out.past_key_values
                nxt = out.logits[:, -1:, :].argmax(dim=-1)
                tokens.append(nxt)

        # tokens[i] entered step i; the last draw is never consumed, so the
        # generated sequence is exactly the first `depth` of them
        generated = torch.cat(tokens[:depth], dim=1)
        widths: list[int] = []
        for row in range(rows):
            width = depth
            if eos is not None:
                hit = (generated[row] == eos).nonzero()
                if hit.numel():
                    width = int(hit[0].item())
            widths.append(width)
        continuation = _continuation_frame(
            self.bundle.tokenizer, generated, tuple(widths)
        )
        self._continuations[(model, input_role)] = continuation

        head = None
        for rname, read in gen_taps:
            site = gen_sites[rname]
            capture_site = gen_capture_sites[rname]
            stacked = torch.cat(steps[_tap_key(capture_site)], 1)
            dataset_rows = self.role_rows[input_role]
            field = self.role_fields[input_role]
            per_row = [
                resolve_steps(
                    self._spec(read.pos),
                    continuation,
                    row,
                    dataset_row=dataset_rows[row],
                    field=field,
                )
                for row in range(rows)
            ]
            self._read_steps[rname] = per_row
            project = None
            if capture_site is not site:  # ln_final kept, lm_head owed
                head = (
                    head
                    or resolve_site(self.bundle, SiteSpec(component="lm_head")).module
                )
                project = head
            self._read_values[rname] = self._finalize_read(
                rname,
                read,
                site,
                stacked,
                batch,
                input_role,
                per_row=per_row,
                project=project,
            )

    def _spec(self, pos: Any) -> PositionSpec:
        spec = self.doc.positions[pos] if isinstance(pos, str) else pos
        if not isinstance(spec, PositionSpec):
            raise ProtocolError("P2", f"unresolved position {pos!r}")
        return spec

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #

    def _positions(
        self, pos: Any, batch: EncodedBatch, input_role: str
    ) -> list[list[int]]:
        spec = pos
        if isinstance(spec, str):
            spec = self.doc.positions[spec]
        if not isinstance(spec, PositionSpec):
            raise ProtocolError("P2", f"unresolved position {pos!r}")
        rows = self.role_rows[input_role]
        field = self.role_fields[input_role]
        return [
            resolve_position(spec, batch, i, dataset_row=rows[i], field=field)
            for i in range(len(rows))
        ]

    @staticmethod
    def _gather(
        tensor: torch.Tensor, per_row: list[list[int]], what: str
    ) -> "torch.Tensor | RaggedValue":
        widths = {len(row) for row in per_row}
        if len(widths) == 1:
            idx = torch.tensor(per_row, dtype=torch.long, device=tensor.device)
            rows = torch.arange(tensor.shape[0], device=tensor.device).unsqueeze(1)
            return tensor[rows, idx]
        # ragged: one flat advanced index, (total_positions, ...) + widths
        row_ids = torch.tensor(
            [i for i, row in enumerate(per_row) for _ in row],
            dtype=torch.long,
            device=tensor.device,
        )
        col_ids = torch.tensor(
            [p for row in per_row for p in row], dtype=torch.long, device=tensor.device
        )
        return RaggedValue(
            flat=tensor[row_ids, col_ids], widths=tuple(len(row) for row in per_row)
        )

    def _read_stack(
        self, read: ReadSpec | WriteSpec, site: ResolvedSite
    ) -> FeaturizerStack:
        # Lazily: `build_stack` ignores the width entirely when no featurizer is
        # referenced (it returns an Identity stack), and some components have no
        # width to give — `input_ids` carries integer ids on a position axis and
        # `expert_idx` a routing table, so both refuse rather than invent one
        # (§5.4). Asking for the width up front made an unfeaturized read of
        # those impossible, which is not what the refusal is for: it exists to
        # reject a *featurizer*, not a read.
        if read.featurizer is None:
            width = 0
        elif site.feature_slice is not None:
            width = site.feature_slice.stop - site.feature_slice.start
        else:
            # `head=site.head` matters only for a *derived* component, which
            # carries no feature_slice: a head there narrows the value's width
            # without narrowing the captured tensor's.
            width = component_width(self.bundle.info, site.component, head=site.head)
        return build_stack(
            read.featurizer,
            dict(self.doc.featurizers),
            width=width,
            load_tensors=self.load_tensors,
            stage_cache=self.stage_cache,
            device=self.bundle.device,
            seed=self.seed,
            coords=self.coords,
        )

    def _finalize_read(
        self,
        rname: str,
        read: ReadSpec,
        site: ResolvedSite,
        raw: torch.Tensor,
        batch: EncodedBatch,
        input_role: str,
        *,
        per_row: list[list[int]] | None = None,
        project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> "torch.Tensor | RaggedValue":
        """One read's value: gather at its positions, then featurize.

        ``per_row`` overrides position resolution — the continuation frame
        resolves to decode steps, which the caller has already worked out
        against the decode. ``project`` runs on the gathered slice before
        anything else, which is how an ``lm_head`` continuation read is
        served from kept ``ln_final`` activations: the vocabulary projection
        happens at the addressed positions and nowhere else.
        """
        if not site.shape.has_contract_form:
            return _whole_native_tensor(rname, read, raw, site)
        if per_row is None:
            per_row = self._positions(read.pos, batch, input_role)
        gathered = self._gather(raw, per_row, f"read {rname!r}")
        if project is not None:
            if isinstance(gathered, RaggedValue):
                gathered = RaggedValue(
                    flat=project(gathered.flat), widths=gathered.widths
                )
            else:
                gathered = project(gathered)
        ragged = isinstance(gathered, RaggedValue)
        value = gathered.flat if isinstance(gathered, RaggedValue) else gathered
        if site.derivation is not None:
            # After the gather, deliberately: the value is `heads` times wider
            # than the tensor it comes from, so deriving it before the gather
            # would cost `seq · H · hidden` where this costs
            # `n_positions · H · hidden`.
            value = _derive(site, value, rname)
        if site.feature_slice is not None:
            value = value[..., site.feature_slice]
        stack = self._read_stack(read, site)
        if not stack.is_identity:
            value, _errs = stack.featurize(value)
        if isinstance(read.dims, tuple):
            dims = torch.tensor(list(read.dims), dtype=torch.long, device=value.device)
            value = value.index_select(-1, dims)
        if not self.grad_enabled:
            value = value.detach().cpu()
        if ragged:
            assert isinstance(gathered, RaggedValue)
            return RaggedValue(flat=value, widths=gathered.widths)
        return value

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #

    def _operand_lookup(self, value: Any) -> torch.Tensor | float:
        if not isinstance(value, str):
            return float(value)
        if value in self.doc.reads:
            stored = self._read_values[value]
            if isinstance(stored, RaggedValue):
                raise NotImplementedError(
                    f"operand read {value!r} is ragged — pairing ragged windows "
                    "into a write is not batchable in the v1 reference backend"
                )
            return stored.to(self.bundle.device)
        if "." in value:
            fname, slot = value.split(".", 1)
            if fname in self.doc.featurizers:
                params = self.stage(fname).slot_params()
                if slot in params:
                    return params[slot]
        if value in self.doc.params:
            spec = self.doc.params[value]
            if isinstance(spec.file_path, str):
                want, implicit = entry_selection(spec.entry, self.coords, value)
                slot = selector_slot(spec.entry, "value")
                what = f"params entry {value!r} ({spec.file_path})"
                point = self.load_tensors(spec.file_path).point(
                    slot, want, what=what, implicit=implicit
                )
                return point.tensor(slot)
            raise NotImplementedError(
                f"trainable free params ({value!r}) arrive with the train loop"
            )
        raise ProtocolError("P2", f"operand {value!r} did not resolve at run time")

    def _build_write_hooks(
        self, write_names: tuple[str, ...], input_role: str, batch: EncodedBatch
    ) -> list[tuple[ResolvedSite, Callable[[torch.Tensor], None]]]:
        """One in-place hook per written-to tap, applying every write at that
        address in class order.

        Addresses are keyed by :func:`_tap_key`, so two components that share a
        module but mean different tensors (a different tuple element, or a
        different shape) get their own hook rather than one overwriting the
        other's view.
        """
        by_address: dict[Any, list[tuple[str, WriteSpec, ResolvedSite]]] = {}
        addresses: dict[Any, ResolvedSite] = {}
        for ename in write_names:
            write = self.doc.writes[ename]
            site = resolve_site(self.bundle, self.doc.sites[str(write.site)])
            if site.component in READ_ONLY_COMPONENTS:
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} targets {site.component!r}, which no write "
                    "may change: "
                    f"{READ_ONLY_COMPONENTS[site.component]}. Refusing at the "
                    "plan, before anything runs.",
                )
            if (
                site.component in SWAP_ONLY_COMPONENTS
                and str(write.do.mechanism) != "swap"
            ):
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} applies {str(write.do.mechanism)!r} to "
                    f"{site.component!r}, which only a whole-value 'swap' may "
                    f"change: {SWAP_ONLY_COMPONENTS[site.component]}. Refusing "
                    "rather than doing arithmetic on values that are labels.",
                )
            key = _tap_key(site)
            by_address.setdefault(key, []).append((ename, write, site))
            addresses[key] = site

        hooks: list[tuple[ResolvedSite, Callable[[torch.Tensor], None]]] = []
        for key, entries in by_address.items():
            site = addresses[key]
            hooks.append((site, self._address_writer(entries, input_role, batch)))
        return hooks

    def _address_writer(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
    ) -> Callable[[torch.Tensor], None]:
        def class_rank(entry: tuple[str, WriteSpec, ResolvedSite]) -> int:
            do = entry[1].do
            if str(do.mechanism) == "renormalize":
                return 2  # after the deltas — the only order where it acts (§2.8 note)
            return 1 if is_additive(do) else 0  # absolute first, then additive

        ordered = sorted(entries, key=class_rank)

        def apply(tensor: torch.Tensor) -> None:
            for ename, write, site in ordered:
                if not site.shape.has_contract_form:
                    # Symmetric with the read (see _whole_native_tensor): this
                    # tensor's feature axis is a position axis, so the position
                    # gather below would index heads with positions and `dims`
                    # would slice key positions as features. Both are refused
                    # there; what is left is the whole tensor, edited whole.
                    _whole_native_tensor(ename, write, tensor, site)
                    mechanism = str(write.do.mechanism)
                    if site.component in NORMALIZED_TAPS and mechanism != "swap":
                        raise ProtocolError(
                            "P4",
                            f"write {ename!r} applies {mechanism!r} to "
                            f"{site.component!r}, which only a whole-tensor "
                            f"'swap' may change: "
                            f"{NORMALIZED_TAPS[site.component]}.",
                        )
                    if mechanism == "swap":
                        replacement = self._operand_lookup(write.do.payload)
                        if not isinstance(replacement, torch.Tensor):
                            raise ProtocolError(
                                "P2",
                                f"write {ename!r} swaps {site.component!r} with "
                                "a scalar; a whole-tensor interchange needs a "
                                "tensor operand read from elsewhere",
                            )
                        if replacement.shape != tensor.shape:
                            raise ProtocolError(
                                "P2",
                                f"write {ename!r} replaces the whole "
                                f"{site.component!r} tensor, but its operand has "
                                f"shape {tuple(replacement.shape)} and the tap is "
                                f"{tuple(tensor.shape)} — an interchange needs "
                                "both inputs to have the same number of positions",
                            )
                        tensor.copy_(replacement.to(tensor.dtype))
                    elif mechanism == "gaussian":
                        # 📐 The noise is drawn as (batch, position, feature)
                        # and its `axis` names the feature axis' tensor-parallel
                        # semantics. This tap has no feature axis — its last
                        # axis is key positions — so there is nothing for either
                        # to mean, and the draw does not even fit (measured:
                        # "shape '[1, 8, 5, 5]' is invalid for input of size
                        # 40"). Refused by name rather than reshaped into
                        # something that would run.
                        raise ProtocolError(
                            "P4",
                            f"write {ename!r} applies 'gaussian' to "
                            f"{site.component!r}, whose shape is "
                            f"{site.shape.describe()}: the noise is drawn per "
                            "(batch, position, feature) and its 'axis' names "
                            "how the feature axis is sharded, and this tap has "
                            "no feature axis at all. Swap in a noise tensor of "
                            "the tap's own shape instead.",
                        )
                    else:
                        # 📐 Arithmetic on the whole tensor, with no gather: for
                        # `attention_scores` this is the point of the component.
                        # `_written_value` broadcasts a scalar operand over any
                        # rank, and `dims` and featurizers are already refused
                        # above, so there is no feature axis for it to mis-slice.
                        tensor.copy_(
                            self._written_value(ename, write, site, tensor).to(
                                tensor.dtype
                            )
                        )
                    continue
                per_row = self._positions(write.pos, batch, input_role)
                widths = {len(row) for row in per_row}
                if len(widths) != 1:
                    raise NotImplementedError(
                        f"write {ename!r}: ragged position widths {sorted(widths)} "
                        "are not batchable in the v1 reference backend — an "
                        "all-positions or variable write needs every row to be "
                        "the same length"
                    )
                idx = torch.tensor(per_row, dtype=torch.long, device=tensor.device)
                rows = torch.arange(tensor.shape[0], device=tensor.device).unsqueeze(1)
                fslice = site.feature_slice or slice(None)
                v_pre = tensor[rows, idx][..., fslice]
                v_new = self._written_value(ename, write, site, v_pre)
                slice_view = tensor[rows, idx]
                slice_view[..., fslice] = v_new.to(tensor.dtype)
                tensor[rows, idx] = slice_view

        return apply

    def _written_value(
        self, ename: str, write: WriteSpec, site: ResolvedSite, v_pre: torch.Tensor
    ) -> torch.Tensor:
        """featurize → class-ordered do → inverse, honoring dims and the
        error-term contract."""
        stack = self._read_stack(write, site)
        f0, errs = stack.featurize(v_pre)
        dims = None
        if isinstance(write.dims, tuple):
            dims = torch.tensor(list(write.dims), dtype=torch.long, device=f0.device)

        def select(f: torch.Tensor) -> torch.Tensor:
            return f if dims is None else f.index_select(-1, dims)

        f = f0.clone()
        do = write.do
        batch_size, n_pos = v_pre.shape[0], v_pre.shape[1]
        if str(do.mechanism) == "renormalize":
            pass  # applied last, below
        elif is_additive(do):
            delta = apply_delta(
                do, select(f0), self._operand_lookup, batch=batch_size, n_pos=n_pos
            )
            if dims is None:
                f = f + delta
            else:
                f.index_copy_(-1, dims, select(f) + delta)
        else:
            written = apply_absolute(do, select(f0), self._operand_lookup)
            written = written.broadcast_to(select(f0).shape).to(f.dtype)
            if dims is None:
                f = written.clone()
            else:
                f.index_copy_(-1, dims, written)
        if str(do.mechanism) == "renormalize":
            if dims is None:
                f = apply_renormalize(f, f0)
            else:
                f.index_copy_(-1, dims, apply_renormalize(select(f), select(f0)))
        return stack.inverse(f, errs)


# --------------------------------------------------------------------------- #
# hook plumbing (mirrors the oracle's _install / capture helpers)
# --------------------------------------------------------------------------- #


def _whole_native_tensor(
    rname: str, read: "ReadSpec | WriteSpec", raw: torch.Tensor, site: "ResolvedSite"
) -> torch.Tensor:
    """A read of a tap with **no contract form**: the whole native tensor.

    The one such tap is the attention pattern, ``(batch, heads, query, key)``,
    and the reason this is a bypass rather than a gather is its second position
    axis: ``_gather`` (dim 0 batch, dim 1 position) would index the head axis
    with position indices, and ``dims`` would slice the key axis as though it
    were features. Both produce plausible numbers from the wrong tensor.

    All three refusals below are *generated* from
    :class:`~causalab.protocol.shapes.FeatureShape` — position addressing needs
    one position axis, a featurizer needs a feature space, ``dims`` needs a
    feature axis to index — so a later tap with the same problem is refused by
    declaring its axes rather than by adding a branch here. What remains — the
    whole tensor, at ``pos: "all"`` — is exactly what an interchange on
    attention needs, and what nnterp's own check exercises
    (``self[layer] = rnd``).
    """
    shape = site.shape
    what = f"{site.component!r} ({shape.describe()})"
    pos = read.pos
    whole = getattr(pos, "all", None) is True or pos == ALL_POSITIONS
    if not whole:
        axes = ", ".join(a.label for a in shape.position_axes)
        raise ProtocolError(
            "P4",
            f"read {rname!r} addresses positions on {what}, which has "
            f"{len(shape.position_axes)} position axes ({axes}) — a position "
            "index would be ambiguous between them. Read the whole tensor with "
            'pos: "all".',
        )
    if read.featurizer is not None:
        raise ProtocolError(
            "P4",
            f"read {rname!r} featurizes {what}: {shape.refusal('it')} A "
            "featurizer would be fitted across an axis that is not a basis.",
        )
    if isinstance(read.dims, tuple):
        raise ProtocolError(
            "P4",
            f"read {rname!r} slices 'dims' on {what}: that would select "
            f"{shape.axes[-1].label} entries as though they were features.",
        )
    return raw


def _derive(site: "ResolvedSite", value: torch.Tensor, rname: str) -> torch.Tensor:
    """Compute a derived component from the tensor its tap captured."""
    if site.derivation == "attention_result":
        return _attention_result(site, value)
    raise ProtocolError("P2", f"read {rname!r}: unknown derivation {site.derivation!r}")


def _attention_result(site: "ResolvedSite", premix: torch.Tensor) -> torch.Tensor:
    """Head ``h``'s contribution to the residual stream.

    ``result[..., h, :] = premix[..., h·d:(h+1)·d] @ W_o[:, h·d:(h+1)·d].T`` —
    the part of the block's attention output that head ``h`` is responsible for.
    The model never forms it: it projects the whole premix at once, so what it
    computes is the *sum* over heads (plus the o-projection's bias, if it has
    one). ``sum_h result == attention_output - bias`` is the identity that
    defines this component, and the test suite pins it.

    Computed by **masking and re-projecting** rather than by slicing the weight
    matrix. That is deliberate: ``nn.Linear`` stores ``(out, in)`` and
    transformers' ``Conv1D`` (GPT-2's ``c_proj``) stores ``(in, out)``, so a
    weight-slicing implementation has to know which family it is looking at and
    is silently wrong if it guesses. Running the projection the model's own
    module defines cannot be wrong about its own layout, and the bias — which is
    *not* attributable to any head — is subtracted back off explicitly.
    """
    module = site.module
    bias = getattr(module, "bias", None)
    heads = site.shape.head_space
    assert heads is not None  # the premix tap always has a head axis
    per_head = premix.shape[-1] // heads

    def contribution(head: int) -> torch.Tensor:
        masked = torch.zeros_like(premix)
        window = slice(head * per_head, (head + 1) * per_head)
        masked[..., window] = premix[..., window]
        out = module(masked)
        return out if bias is None else out - bias

    if site.head is not None:
        return contribution(site.head)
    # The whole tensor: `heads` times wider than `attention_output`. On a real
    # A3B that is 64x at hidden 4096, which is why naming a `head` is
    # encouraged — but it is a documented cost, not a refusal.
    return torch.cat([contribution(head) for head in range(heads)], dim=-1)


def _interface_edit(
    site: "ResolvedSite", write: Callable[[torch.Tensor], None], batch_size: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapt an in-place contract-shaped writer to the interface's protocol.

    The manager hands out a clone and takes back a replacement, which is why
    this can convert, mutate and convert back without any of it reaching the
    model's own storage.
    """

    def edit(native: torch.Tensor) -> torch.Tensor:
        contract = to_contract(native, site.shape, batch_size=batch_size)
        write(contract)
        return from_contract(contract, site.shape, batch_size=batch_size, native=native)

    return edit


def _interface_capture(
    sink: dict[Any, torch.Tensor],
    key: Any,
    site: "ResolvedSite",
    batch_size: int,
) -> Callable[[torch.Tensor], None]:
    """The read half — the same contract shape ``_capturing`` produces."""

    def read(native: torch.Tensor) -> None:
        sink[key] = to_contract(native, site.shape, batch_size=batch_size)

    return read


def _interface_accumulate(
    sink: list[torch.Tensor], site: "ResolvedSite", batch_size: int
) -> Callable[[torch.Tensor], None]:
    """The read half for a decode: append per step, as ``_accumulating`` does."""

    def read(native: torch.Tensor) -> None:
        sink.append(to_contract(native, site.shape, batch_size=batch_size))

    return read


def _refuse_unstackable(name: str, site: "ResolvedSite") -> None:
    """Refuse a continuation read whose steps do not stack into a frame.

    📐 A decode step attends over the whole KV cache, so a tensor indexed by the
    positions being attended *to* is ``prompt + step`` long at step ``step``
    while the query axis stays 1. The accumulating sink concatenates steps on
    dim 1, which for an ordinary tap is the position axis, so such a tap either
    raises a bare torch size error (measured: "Expected size 9 but got size 10")
    or, for a single-step budget, silently returns one step shaped like a frame.
    Neither is a continuation read.

    Both conditions are read off the declared axes rather than a component list:

    * **two position axes** — the attention pattern and the scores. There is no
      non-arbitrary answer to which of them the steps stack along;
    * **one position axis, over the keys** — ``attention_key``. Its own length
      is what grows, so consecutive steps are different lengths.

    ``attention_query`` and ``attention_z`` are query-axis-shaped and accumulate
    correctly, which is why this is a property of the axes and not of "anything
    inside the attention function".
    """
    shape = site.shape
    if shape.has_contract_form and not any(
        axis.kind == "position" and axis.name == "key" for axis in shape.axes
    ):
        return
    why = (
        "it has two position axes, so there is no single axis the decode steps "
        "stack along"
        if not shape.has_contract_form
        else "its position axis runs over the positions being attended to, "
        "which under a KV cache is the whole prefix and grows by one per step"
    )
    raise ProtocolError(
        "P4",
        f"read {name!r} reads {site.component!r} in the generated frame, whose "
        f"shape is {shape.describe()}: {why}, so the steps do not stack into "
        "one tensor. Read it in the prompt frame.",
    )


@contextlib.contextmanager
def _installed(
    module: Any,
    kind: str,
    write: Callable[[torch.Tensor], None],
    *,
    shape: FeatureShape,
    tuple_index: int | None = None,
    batch_size: int = 1,
) -> Iterator[None]:
    """Install an in-place write hook, converting to the executor's contract.

    ``write`` always sees a ``(batch, position, feature)`` tensor and mutates it
    in place; the model always gets its native shape back. ``native`` is handed
    to :func:`from_contract` because a fused tap's other splits live in it and
    have to survive the write untouched.
    """
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> Any:
            native = tap_tensor(out, tuple_index).clone()
            contract = to_contract(native, shape, batch_size=batch_size)
            write(contract)
            return rebuild_payload(
                out,
                tuple_index,
                from_contract(contract, shape, batch_size=batch_size, native=native),
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            native = args[0].clone()
            contract = to_contract(native, shape, batch_size=batch_size)
            write(contract)
            return (
                from_contract(contract, shape, batch_size=batch_size, native=native),
                *args[1:],
            )

        handle = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def _capturing(
    module: Any,
    kind: str,
    sink: dict[Any, torch.Tensor],
    key: Any,
    *,
    shape: FeatureShape,
    tuple_index: int | None = None,
    batch_size: int = 1,
) -> Iterator[None]:
    """Capture a tap's tensor, in the executor's contract shape."""
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> None:
            sink[key] = to_contract(
                tap_tensor(out, tuple_index), shape, batch_size=batch_size
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> None:
            sink[key] = to_contract(args[0], shape, batch_size=batch_size)

        handle = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def _accumulating(
    module: Any,
    kind: str,
    sink: list[torch.Tensor],
    *,
    shape: FeatureShape,
    tuple_index: int | None = None,
    batch_size: int = 1,
) -> Iterator[None]:
    """Like :func:`_capturing`, but append instead of overwrite.

    A decode calls the same modules once per step, so the single-tensor sink
    would keep only the last step. Continuation reads need every step, and
    stacking them on the sequence axis gives a ``(batch, steps, …)`` tensor
    that gathers exactly like a padded frame does.
    """
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> None:
            sink.append(
                to_contract(tap_tensor(out, tuple_index), shape, batch_size=batch_size)
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> None:
            sink.append(to_contract(args[0], shape, batch_size=batch_size))

        handle = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def _continuation_frame(
    tokenizer: Any, generated: torch.Tensor, widths: tuple[int, ...]
) -> Continuation:
    """Build the frame the decode produced, characters included.

    Token spans come from incremental detokenization — decode the row's
    first ``k`` tokens, then ``k + 1``, and the growth is token ``k``'s
    span. Re-encoding the finished text would not do: a tokenizer is free
    to merge across a boundary the decode never saw, and the spans have to
    describe the tokens the model actually emitted.
    """
    texts: list[str] = []
    offsets: list[tuple[tuple[int, int], ...]] = []
    for row, width in enumerate(widths):
        ids = [int(t) for t in generated[row, :width]]
        spans: list[tuple[int, int]] = []
        text = ""
        for k in range(width):
            grown = tokenizer.decode(ids[: k + 1], skip_special_tokens=True)
            spans.append((len(text), len(grown)))
            text = grown
        texts.append(text)
        offsets.append(tuple(spans))
    return Continuation(
        token_ids=generated.detach().cpu(),
        widths=widths,
        texts=tuple(texts),
        offsets=tuple(offsets),
    )
