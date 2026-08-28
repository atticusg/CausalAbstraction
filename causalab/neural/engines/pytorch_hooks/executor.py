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

from causalab.neural.shared.encoding import (
    Continuation,
    EncodedBatch,
    encode,
    resolve_position,
    resolve_steps,
    select_field,
)
from causalab.neural.engines.pytorch_hooks.featurizers import (
    FeaturizerStack,
    Stage,
    build_stack,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.mechanisms import (
    apply_absolute,
    apply_delta,
    apply_renormalize,
    is_additive,
    operand_names,
)
from causalab.neural.engines.pytorch_hooks.attention_probs import eager_attention_writes
from causalab.neural.engines.pytorch_hooks.sites import (
    READ_ONLY_COMPONENTS,
    SWAP_ONLY_COMPONENTS,
    ResolvedSite,
    resolve_site,
)
from causalab.neural.shared.layout import (
    Layout,
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)
from causalab.protocol.bundles import entry_selection, selector_slot
from causalab.protocol.errors import ProtocolError
from causalab.protocol.plan import generated_budget
from causalab.protocol.registry import component_width
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


def _tap_key(site: "ResolvedSite") -> tuple[int, str, str, int | None]:
    """Identity of a tap for capture-sink sharing.

    Two sites may share a module and side yet mean different tensors — a
    different tuple element, or the same tensor in a different layout — so the
    layout and tuple index are part of the identity. Keying on the module alone
    would let one tap read another's tensor.
    """
    return (id(site.module), site.kind, site.layout, site.tuple_index)


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
        from causalab.neural.engines.pytorch_hooks.featurizers import stage_output_width

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
        capture: dict[tuple[int, str], torch.Tensor] = {}
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
            # An attention-pattern write cannot ride a forward hook: the hook
            # fires after the pattern has been consumed, so the edit would be a
            # silent no-op. It goes through the eager attention function instead.
            pattern_edits = {
                id(module): fn
                for module, kind, w_layout, w_tuple_index, fn in write_hooks
                if w_layout == "native"
            }
            hooks.enter_context(eager_attention_writes(pattern_edits))
            for module, kind, w_layout, w_tuple_index, fn in write_hooks:
                if w_layout == "native":
                    continue
                hooks.enter_context(
                    _installed(
                        module,
                        kind,
                        fn,
                        layout=w_layout,
                        tuple_index=w_tuple_index,
                        batch_size=batch.input_ids.shape[0],
                    )
                )
            for site in capture_sites.values():
                key = _tap_key(site)
                if key not in capture:
                    capture[key] = torch.empty(0)  # placeholder; filled by hook
                    hooks.enter_context(
                        _capturing(
                            site.module,
                            site.kind,
                            capture,
                            key,
                            layout=site.layout,
                            tuple_index=site.tuple_index,
                            batch_size=batch.input_ids.shape[0],
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
        steps: dict[tuple[int, str], list[torch.Tensor]] = {}
        cache = prefill.past_key_values
        with contextlib.ExitStack() as hooks:
            for name, site in gen_capture_sites.items():
                if site.component == "attention_probs":
                    # 📐 Each decode step attends over the whole cache, so the
                    # pattern is (batch, heads, 1, prompt + step): the key axis
                    # GROWS by one per step while the query axis stays 1. The
                    # accumulating sink stacks steps on dim 1, which for every
                    # other component is the position axis and here is heads —
                    # so the concat either raises a bare torch size error
                    # (measured: "Expected size 9 but got size 10") or, for a
                    # single-step budget, silently returns one step's pattern
                    # shaped like a frame. Neither is a continuation read.
                    #
                    # Saying what a correct one would mean — a ragged key axis
                    # per step, addressed per query row — is exactly the typed
                    # feature-shape descriptor, follow-up F1. Refuse by name
                    # until then, as round 1 does for every other thing the
                    # descriptor is needed for.
                    raise ProtocolError(
                        "P4",
                        f"read {name!r} reads 'attention_probs' in the "
                        "generated frame, which round 1 does not support: with "
                        "a KV cache each step's pattern has a different key "
                        "width, so the steps do not stack into one tensor. "
                        "Read it in the prompt frame, or wait on the typed "
                        "feature-shape descriptor (follow-up F1).",
                    )
                key = _tap_key(site)
                if key not in steps:
                    steps[key] = []
                    hooks.enter_context(
                        _accumulating(
                            site.module,
                            site.kind,
                            steps[key],
                            layout=site.layout,
                            tuple_index=site.tuple_index,
                            batch_size=batch.input_ids.shape[0],
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
            width = component_width(self.bundle.info, site.component, head=None)
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
        if site.component == "attention_probs":
            return _whole_attention_pattern(rname, read, raw)
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
                    "into a write is not batchable in the v1 reference engine"
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
    ) -> list[tuple[Any, str, Layout, int | None, Callable[[torch.Tensor], None]]]:
        """One in-place hook per written-to tap, applying every write at that
        address in class order.

        Addresses are keyed by :func:`_tap_key`, so two components that share a
        module but mean different tensors (a different tuple element, or a
        different layout) get their own hook rather than one overwriting the
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

        hooks: list[
            tuple[Any, str, Layout, int | None, Callable[[torch.Tensor], None]]
        ] = []
        for key, entries in by_address.items():
            site = addresses[key]
            hooks.append(
                (
                    site.module,
                    site.kind,
                    site.layout,
                    site.tuple_index,
                    self._address_writer(entries, input_role, batch),
                )
            )
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
                if site.component == "attention_probs":
                    # Symmetric with the read (see _whole_attention_pattern):
                    # the pattern's feature axis is a position axis, so the
                    # position gather below would index heads with positions.
                    # Round 1 replaces the whole pattern — which is what only
                    # `swap` means. Any other mechanism (a delta, a scale, a
                    # clamp) would leave rows that no longer sum to 1, and the
                    # code below would treat its payload as a whole pattern
                    # anyway; refuse by name rather than produce one that is
                    # plausible and wrong.
                    if str(write.do.mechanism) != "swap":
                        raise ProtocolError(
                            "P4",
                            f"write {ename!r} applies {str(write.do.mechanism)!r} "
                            "to 'attention_probs' — round 1 supports only a "
                            "whole-pattern 'swap' (an interchange). Anything "
                            "that re-weights rows needs the typed feature-shape "
                            "descriptor and a renormalization story "
                            "(follow-up F1).",
                        )
                    _whole_attention_pattern(ename, write, tensor)
                    replacement = self._operand_lookup(write.do.payload)
                    if replacement.shape != tensor.shape:
                        raise ProtocolError(
                            "P2",
                            f"write {ename!r} replaces the whole attention "
                            f"pattern, but its operand has shape "
                            f"{tuple(replacement.shape)} and the pattern is "
                            f"{tuple(tensor.shape)} — an attention-pattern "
                            "interchange needs both inputs to have the same "
                            "number of positions",
                        )
                    tensor.copy_(replacement.to(tensor.dtype))
                    continue
                per_row = self._positions(write.pos, batch, input_role)
                widths = {len(row) for row in per_row}
                if len(widths) != 1:
                    raise NotImplementedError(
                        f"write {ename!r}: ragged position widths {sorted(widths)} "
                        "are not batchable in the v1 reference engine — an "
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


def _whole_attention_pattern(
    rname: str, read: "ReadSpec | WriteSpec", raw: torch.Tensor
) -> torch.Tensor:
    """An ``attention_probs`` read: the whole (batch, heads, query, key) pattern.

    Round-1 scope, and the reason it is a bypass rather than a gather: this
    tensor has **two** position axes and its feature axis *is* a position axis,
    so ``_gather`` (dim 0 batch, dim 1 position) would index the head axis with
    position indices, and ``dims`` would slice the key axis as though it were
    features. Both would produce plausible numbers from the wrong tensor.

    Rather than approximate, the two forms that need a typed feature-shape
    descriptor are refused and named as follow-up F1. What remains — the whole
    pattern, at ``pos: "all"`` — is exactly what an interchange on attention
    needs, and what nnterp's own check exercises (``self[layer] = rnd``).
    """
    pos = read.pos
    whole = getattr(pos, "all", None) is True or pos == ALL_POSITIONS
    if not whole:
        raise ProtocolError(
            "P4",
            f"read {rname!r} addresses positions on 'attention_probs', which "
            "has two position axes (batch, heads, query, key) — its feature "
            "axis IS a position axis. Round 1 exposes the whole pattern: use "
            'pos: "all". Addressing one query row, or slicing the key axis, '
            "needs the typed feature-shape descriptor (follow-up F1).",
        )
    if read.featurizer is not None:
        raise ProtocolError(
            "P4",
            f"read {rname!r} featurizes 'attention_probs', whose feature axis "
            "is a position axis — a featurizer would be fitted across key "
            "positions, which is not a feature space. Needs the typed "
            "feature-shape descriptor (follow-up F1).",
        )
    if isinstance(read.dims, tuple):
        raise ProtocolError(
            "P4",
            f"read {rname!r} slices 'dims' on 'attention_probs': that would "
            "select key positions as though they were features. Needs the "
            "typed feature-shape descriptor (follow-up F1).",
        )
    return raw


@contextlib.contextmanager
def _installed(
    module: Any,
    kind: str,
    write: Callable[[torch.Tensor], None],
    *,
    layout: Layout = "bsd",
    tuple_index: int | None = None,
    batch_size: int = 1,
) -> Iterator[None]:
    """Install an in-place write hook, converting to the executor's contract.

    ``write`` always sees a ``(batch, position, feature)`` tensor and mutates it
    in place; the model always gets its native shape back. For the default
    ``"bsd"`` layout both conversions are identity, so the path is unchanged.
    """
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> Any:
            native = tap_tensor(out, tuple_index).clone()
            contract = to_contract(native, layout, batch_size=batch_size)
            write(contract)
            return rebuild_payload(
                out, tuple_index, from_contract(contract, layout, batch_size=batch_size)
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            native = args[0].clone()
            contract = to_contract(native, layout, batch_size=batch_size)
            write(contract)
            return (
                from_contract(contract, layout, batch_size=batch_size),
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
    layout: Layout = "bsd",
    tuple_index: int | None = None,
    batch_size: int = 1,
) -> Iterator[None]:
    """Capture a tap's tensor, in the executor's contract shape."""
    if kind == "out":

        def out_hook(_m: Any, _i: Any, out: Any) -> None:
            sink[key] = to_contract(
                tap_tensor(out, tuple_index), layout, batch_size=batch_size
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> None:
            sink[key] = to_contract(args[0], layout, batch_size=batch_size)

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
    layout: Layout = "bsd",
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
                to_contract(tap_tensor(out, tuple_index), layout, batch_size=batch_size)
            )

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> None:
            sink.append(to_contract(args[0], layout, batch_size=batch_size))

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
