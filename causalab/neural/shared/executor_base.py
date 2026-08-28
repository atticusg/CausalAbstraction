"""The engine-neutral core of a point executor.

Everything here operates on the *contract* tensor shape ``(batch, position,
feature)`` and the document: position resolution, gathers, featurizer stacks,
operand lookup, and the class-ordered write math. What an engine adds is one
method — :meth:`ExecutorBase._run_group` — that produces contract tensors for
this group's taps and lands its writes (hooks in the reference engine, traces
in the nnsight engine). The public surface consumed by
:mod:`causalab.neural.shared.execution` lives here so the two engines cannot
drift apart on what a read means.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping

import torch

from causalab.neural.shared.encoding import (
    Continuation,
    EncodedBatch,
    encode,
    resolve_position,
    select_field,
)
from causalab.neural.shared.featurizers import FeaturizerStack, Stage, build_stack
from causalab.neural.shared.mechanisms import (
    apply_absolute,
    apply_delta,
    apply_renormalize,
    is_additive,
)
from causalab.neural.shared.sites import (
    NORMALIZED_TAPS,
    READ_ONLY_COMPONENTS,
    SWAP_ONLY_COMPONENTS,
    ResolvedSite,
    resolve_site,
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
    WriteSpec,
)

__all__ = ["ExecutorBase", "RaggedValue", "TapKey", "document_seed", "tap_key"]


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


#: What identifies a tap for capture-sink sharing — see :func:`tap_key`.
TapKey = tuple[int, str, FeatureShape, int | None, str | None]


def tap_key(site: ResolvedSite) -> TapKey:
    """Identity of a tap for capture-sink sharing.

    Two sites may share a module and side yet mean different tensors — a
    different tuple element, or the same tensor read through a different shape
    — so the shape and tuple index are part of the identity. Keying on the
    module alone would let one tap read another's tensor.
    """
    return (
        id(site.module),
        site.kind,
        site.shape,
        site.tuple_index,
        site.interface_slot,
    )


def whole_native_tensor(
    rname: str, read: "ReadSpec | WriteSpec", raw: torch.Tensor, site: ResolvedSite
) -> torch.Tensor:
    """A read of a tap with **no contract form**: the whole native tensor.

    The one such tap is the attention pattern, ``(batch, heads, query, key)``,
    and the reason this is a bypass rather than a gather is its second position
    axis: the gather (dim 0 batch, dim 1 position) would index the head axis
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


def _derive(site: ResolvedSite, value: torch.Tensor, rname: str) -> torch.Tensor:
    """Compute a derived component from the tensor its tap captured."""
    if site.derivation == "attention_result":
        return _attention_result(site, value)
    raise ProtocolError("P2", f"read {rname!r}: unknown derivation {site.derivation!r}")


def _attention_result(site: ResolvedSite, premix: torch.Tensor) -> torch.Tensor:
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

    ⚠️ Calls ``site.module`` directly, so it needs a real ``nn.Module``. That is
    why the nnsight engine, whose ``site.module`` is an envoy, does not declare
    this component.
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


class ExecutorBase:
    """Execute one concrete document against one loaded model.

    Subclasses implement :meth:`_run_group` — everything else is shared."""

    def __init__(
        self,
        doc: Document,
        bundle: Any,
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
        from causalab.neural.shared.featurizers import stage_output_width

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
    # what an engine implements
    # ------------------------------------------------------------------ #

    def _run_group(self, model: str, input_role: str) -> None:
        """Run one (model, input role) forward group: land its writes and
        fill ``self._read_values`` for its reads."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # shared plumbing
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

    def _spec(self, pos: Any) -> PositionSpec:
        spec = self.doc.positions[pos] if isinstance(pos, str) else pos
        if not isinstance(spec, PositionSpec):
            raise ProtocolError("P2", f"unresolved position {pos!r}")
        return spec

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
            return whole_native_tensor(rname, read, raw, site)
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
        if site.shape.state_axes:
            return self._state_read(rname, read, site, value, gathered)
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

    def _state_read(
        self,
        rname: str,
        read: ReadSpec,
        site: ResolvedSite,
        value: torch.Tensor,
        gathered: "torch.Tensor | RaggedValue",
    ) -> "torch.Tensor | RaggedValue":
        """The tail of a read whose trailing axes form a state matrix.

        The tensor keeps its native layout — ``(batch, steps, heads, d_k,
        d_v)`` after the position gather — because there is no feature vector
        to flatten to. ``head:`` selects on the head axis directly;
        ``featurizer`` and ``dims`` are refused off the declared axes (the same
        generated refusals the attention pattern gets, with the position gather
        kept, which is what distinguishes the two shapes).
        """
        what = f"{site.component!r} ({site.shape.describe()})"
        if read.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} featurizes {what}: {site.shape.refusal('it')}",
            )
        if isinstance(read.dims, tuple):
            raise ProtocolError(
                "P4",
                f"read {rname!r} slices 'dims' on {what}: that would select "
                "d_v columns of a matrix as though they were features.",
            )
        if site.head is not None:
            # dim 0 of a ragged flat is the gathered rows; dense keeps
            # (batch, steps) in front — the head axis is right after either way
            value = (
                value[:, site.head]
                if isinstance(gathered, RaggedValue)
                else value[:, :, site.head]
            )
        if not self.grad_enabled:
            value = value.detach().cpu()
        if isinstance(gathered, RaggedValue):
            return RaggedValue(flat=value, widths=gathered.widths)
        return value

    def _state_step_writer(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
    ) -> Callable[[int, torch.Tensor], torch.Tensor]:
        """Per-step application of ``delta_state`` writes, for the stepwise
        substitution (round-4 plan §2.3).

        A state edit must feed forward — step ``t``'s replacement is what step
        ``t+1`` decays and writes into — so the shared whole-tensor write math
        cannot land it. Instead each addressed step applies the same
        class-ordered mechanisms to that step's matrix, flattened to one
        row: ``v_pre`` is ``S_t`` as ``(1, 1, heads·d_k·d_v)``, and a tensor
        operand (a ``delta_state`` read) is sliced to the same (row, step)
        before the mechanism sees it, so ``swap`` interchanges step-for-step.

        ``dims`` is refused (a matrix has no feature columns); ``featurizer``
        refuses through the width lookup, as every state read does.
        """

        def class_rank(entry: tuple[str, WriteSpec, ResolvedSite]) -> int:
            do = entry[1].do
            if str(do.mechanism) == "renormalize":
                return 2
            return 1 if is_additive(do) else 0

        prepared: list[tuple[str, WriteSpec, ResolvedSite, list[list[int]]]] = []
        for ename, write, site in sorted(entries, key=class_rank):
            if isinstance(write.dims, tuple):
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} slices 'dims' on {site.component!r} "
                    f"({site.shape.describe()}): that would select d_v columns "
                    "of a matrix as though they were features.",
                )
            prepared.append(
                (ename, write, site, self._positions(write.pos, batch, input_role))
            )

        def edit_state(step: int, state: torch.Tensor) -> torch.Tensor:
            edited = state
            for ename, write, site, per_row in prepared:
                for row, positions in enumerate(per_row):
                    if step not in positions:
                        continue
                    j = positions.index(step)
                    if edited is state:
                        edited = state.clone()
                    v_pre = edited[row : row + 1].reshape(1, 1, -1)
                    v_new = self._written_value(
                        ename,
                        write,
                        site,
                        v_pre,
                        lookup=self._state_operand(
                            ename, row, j, len(positions), v_pre
                        ),
                    )
                    edited[row] = v_new.reshape(edited.shape[1:]).to(edited.dtype)
            return edited

        return edit_state

    def _state_operand(
        self, ename: str, row: int, step_index: int, n_steps: int, v_pre: torch.Tensor
    ) -> Callable[[Any], "torch.Tensor | float"]:
        """Operand lookup for one (row, addressed-step) state application: a
        tensor operand must be a state read — ``(batch, steps, heads, d_k,
        d_v)`` — covering **exactly the write's addressed steps** (the standard
        write path's elementwise rule, stated rather than broadcast), and is
        sliced to this row and step so the mechanism math sees two aligned
        single-step rows."""

        def lookup(value: Any) -> "torch.Tensor | float":
            operand = self._operand_lookup(value)
            if not isinstance(operand, torch.Tensor):
                return operand
            if operand.dim() != 5:
                raise ProtocolError(
                    "P2",
                    f"write {ename!r} hands {value!r} to a 'delta_state' "
                    f"write, but its shape is {tuple(operand.shape)} — a state "
                    "operand is a 'delta_state' read, (batch, steps, heads, "
                    "d_k, d_v), applied step for step",
                )
            if operand.shape[1] != n_steps:
                raise ProtocolError(
                    "P2",
                    f"write {ename!r}: operand {value!r} covers "
                    f"{operand.shape[1]} steps, but the write addresses "
                    f"{n_steps} — the j-th operand step lands on the j-th "
                    "addressed step, so both sides must cover the same steps "
                    "(read the operand at the write's own positions)",
                )
            sliced = operand[row : row + 1, step_index : step_index + 1]
            return sliced.reshape(1, 1, -1).to(v_pre.device)

        return lookup

    # ------------------------------------------------------------------ #
    # writes (the math; landing them is the engine's job)
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

    def _resolve_write_addresses(
        self, write_names: tuple[str, ...]
    ) -> dict[Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]]:
        """Resolve and policy-check this group's writes, grouped by address.

        Addresses are keyed by :func:`tap_key`, so two components that share a
        module but mean different tensors (a different tuple element, or a
        different shape) get their own application rather than one
        overwriting the other's view."""
        by_address: dict[
            Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
        ] = {}
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
            key = tap_key(site)
            if key not in by_address:
                by_address[key] = (site, [])
            by_address[key][1].append((ename, write, site))
        return by_address

    def _apply_writes_to_contract(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
        tensor: torch.Tensor,
    ) -> None:
        """Apply every write at one address, in class order, mutating the
        contract-shaped ``tensor`` in place — absolute first, additive deltas
        summed, renormalize last against the pre-write norm (§2.8)."""

        def class_rank(entry: tuple[str, WriteSpec, ResolvedSite]) -> int:
            do = entry[1].do
            if str(do.mechanism) == "renormalize":
                return 2  # after the deltas — the only order where it acts (§2.8 note)
            return 1 if is_additive(do) else 0  # absolute first, then additive

        for ename, write, site in sorted(entries, key=class_rank):
            if not site.shape.has_contract_form:
                # Symmetric with the read (see whole_native_tensor): this
                # tensor's feature axis is a position axis, so the position
                # gather below would index heads with positions and `dims`
                # would slice key positions as features. Both are refused
                # there; what is left is the whole tensor, edited whole.
                whole_native_tensor(ename, write, tensor, site)
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
                    # 📐 The noise is drawn as (batch, position, feature) and
                    # its `axis` names the feature axis' tensor-parallel
                    # semantics. This tap has no feature axis — its last axis
                    # is key positions — so there is nothing for either to
                    # mean, and the draw does not even fit (measured: "shape
                    # '[1, 8, 5, 5]' is invalid for input of size 40").
                    # Refused by name rather than reshaped into something that
                    # would run.
                    raise ProtocolError(
                        "P4",
                        f"write {ename!r} applies 'gaussian' to "
                        f"{site.component!r}, whose shape is "
                        f"{site.shape.describe()}: the noise is drawn per "
                        "(batch, position, feature) and its 'axis' names how "
                        "the feature axis is sharded, and this tap has no "
                        "feature axis at all. Swap in a noise tensor of the "
                        "tap's own shape instead.",
                    )
                else:
                    # 📐 Arithmetic on the whole tensor, with no gather: for
                    # `attention_scores` this is the point of the component.
                    # `_written_value` broadcasts a scalar operand over any
                    # rank, and `dims` and featurizers are already refused
                    # above, so there is no feature axis for it to mis-slice.
                    tensor.copy_(
                        self._written_value(ename, write, site, tensor).to(tensor.dtype)
                    )
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

    def _written_value(
        self,
        ename: str,
        write: WriteSpec,
        site: ResolvedSite,
        v_pre: torch.Tensor,
        *,
        lookup: "Callable[[Any], torch.Tensor | float] | None" = None,
    ) -> torch.Tensor:
        """featurize → class-ordered do → inverse, honoring dims and the
        error-term contract.

        ``lookup`` overrides operand resolution — the state-write path slices
        tensor operands to one (row, step) so the same mechanism math applies
        per step; everything else uses :meth:`_operand_lookup` unchanged.
        """
        if lookup is None:
            lookup = self._operand_lookup
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
            delta = apply_delta(do, select(f0), lookup, batch=batch_size, n_pos=n_pos)
            if dims is None:
                f = f + delta
            else:
                f.index_copy_(-1, dims, select(f) + delta)
        else:
            written = apply_absolute(do, select(f0), lookup)
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
