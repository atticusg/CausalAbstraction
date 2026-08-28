"""Point-protocol execution over nnsight traces.

One :class:`TracePointExecutor` runs one concrete document: the same lazy
forward groups, position resolution, featurizer stacks and class-ordered
write math as the reference engine (all inherited from
:class:`~causalab.neural.shared.executor_base.ExecutorBase`) — only the
landing differs. Each group is **one trace**: writes assign the envoy's
``input``/``output`` at their address, reads ``nnsight.save`` the contract
tensor, and a read at a written address sees the write because envoy
assignment replaces the value later accesses observe (measured in the N4
probes, mirroring the reference engine's write-before-capture hook order).

Interior components — tensors ``transformers`` computes inside one call, with
no module boundary to hook — are the third landing (N5): the address table in
:mod:`causalab.neural.engines.nnsight_tracing.addresses` names the op, the
executor navigates ``envoy.source`` to it inside the trace, and the same
``to_contract``/``from_contract`` round-trip runs on what it finds. Only the
address differs between engines, never the policy or the payload math.

Operations are issued in forward-execution order (site depth, writes before
reads at the same depth) — module-boundary envoys are order-tolerant, but the
``.source`` interiors are not (``OutOfOrderError``), and the renumbered
attention band of :data:`~causalab.protocol.plan.COMPONENT_RANK` *is* the
in-forward op order, which the test suite pins rather than assumes.

Some interior addresses only exist under a specific implementation — the
fused attention kernels never materialize the scores — so a group whose taps
require one switches it on around its trace and restores the model default
after (D5). The applied set is stamped as execution metadata, never canonical
form: the document and its digest are implementation-blind.

v1 boundary: no generated frame (arrives with N8's ``tracer.iter`` step
anchoring; routing keeps ``generate`` documents on the reference engine). It
still refuses legibly here in case a document arrives unrouted.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from causalab.neural.engines.nnsight_tracing.addresses import (
    ADDRESSES,
    MOE_EXPERTS,
    AddressResolutionError,
    SourceAddress,
    match_op,
)
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.executor_base import ExecutorBase, tap_key
from causalab.neural.shared.layout import (
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)
from causalab.neural.shared.mechanisms import operand_names
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.errors import ProtocolError
from causalab.protocol.plan import generated_budget
from causalab.protocol.schema import WriteSpec

__all__ = ["TracePointExecutor"]


@dataclass(frozen=True)
class ResolvedTap:
    """One tap as this engine lands it: the shared resolution (policy, layout,
    layer — :class:`ResolvedSite` is the cross-engine vocabulary and stays
    untouched) plus, for an interior component, its ``.source`` address."""

    site: ResolvedSite
    #: ``None`` = a module boundary (the N4 path: ``envoy.input``/``.output``).
    source: SourceAddress | None = None


class TracePointExecutor(ExecutorBase):
    """Execute one concrete document against one nnsight-loaded model."""

    def _run_group(self, model: str, input_role: str) -> None:
        if (model, input_role) in self._groups_run:
            return
        if self.grad_enabled:
            raise ProtocolError(
                "P4",
                "the nnsight engine does not run gradient-enabled groups — "
                "training documents route to an engine with the 'grad' "
                "capability",
            )
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
        taps = []
        for rname, read in self.doc.reads.items():
            if str(read.model) != model or str(read.input) != input_role:
                continue
            if generated_budget(self.doc, read.pos) is not None:
                raise ProtocolError(
                    "P4",
                    f"read {rname!r} addresses the generated frame, which the "
                    "nnsight engine does not serve yet (its 'generate' "
                    "capability is absent — the step-anchored trace reads are "
                    "phase N8 of the engine plan). Routing sends such "
                    "documents to the reference engine.",
                )
            taps.append((rname, read))

        read_taps = {
            rname: self._wrap(resolve_site(self.bundle, self.doc.sites[str(read.site)]))
            for rname, read in taps
        }
        write_taps = {
            key: (self._wrap(site), entries)
            for key, (site, entries) in self._resolve_write_addresses(
                write_names
            ).items()
        }

        group_taps = [*read_taps.values(), *(tap for tap, _ in write_taps.values())]
        required = frozenset(
            requirement
            for tap in group_taps
            if tap.source is not None
            for requirement in tap.source.requires
        )
        self.applied_requirements |= required

        # one trace per group, operations in forward-execution order; a write
        # lands before a read at the same depth so the read sees it, and the
        # `.source` interiors demand the order outright
        operations: list[tuple[int, tuple[int, int], str, ResolvedTap, list]] = []
        for tap, entries in write_taps.values():
            operations.append((0, tap.site.depth, "write", tap, entries))
        seen_keys: set = set()
        for tap in read_taps.values():
            key = tap_key(tap.site, tap.source)
            if key not in seen_keys:
                seen_keys.add(key)
                operations.append((1, tap.site.depth, "read", tap, []))
        operations.sort(key=lambda op: (op[1], op[0]))

        import nnsight

        batch_size = int(batch.input_ids.shape[0])
        saves: dict = {}
        #: navigated `.source` values, shared within one trace so two taps on
        #: one op (q and k are one rope call) request it once
        self._trace_sources: dict = {}
        with torch.no_grad():
            with self._switched_implementations(required):
                with self.bundle.model.trace(
                    {
                        "input_ids": batch.input_ids,
                        "attention_mask": batch.attention_mask,
                    },
                    position_ids=batch.position_ids(),
                ):
                    for _, _, op_kind, tap, entries in operations:
                        if op_kind == "write":
                            self._land_writes(tap, entries, input_role, batch)
                        else:
                            saves[tap_key(tap.site, tap.source)] = nnsight.save(
                                self._tap_contract(tap, batch_size)
                            )
        self._trace_sources = {}

        for rname, read in taps:
            tap = read_taps[rname]
            raw = saves[tap_key(tap.site, tap.source)]
            self._read_values[rname] = self._finalize_read(
                rname, read, tap.site, raw, batch, input_role
            )
        self._groups_run.add((model, input_role))

    # ------------------------------------------------------------------ #
    # interior addressing (N5)
    # ------------------------------------------------------------------ #

    def _wrap(self, site: ResolvedSite) -> ResolvedTap:
        """Pair the shared resolution with this engine's landing.

        An ``interface_slot`` marks a component with no module boundary (the
        four function-interior slots, and the pattern — whose *write* has no
        boundary; this engine serves its read from the same op); those are
        looked up in the per-stream address table. ``kind="interior"`` marks
        the per-expert MoE interior, keyed by component. Everything else is
        the envoy path unchanged.
        """
        if site.kind == "interior":
            address = MOE_EXPERTS.get(site.component)
            where = "the MOE_EXPERTS table"
        elif site.interface_slot is not None:
            stream = self.bundle.stream_at(site.layer)
            address = ADDRESSES.get(stream, {}).get(site.component)
            where = f"the {stream!r} table"
        else:
            return ResolvedTap(site=site)
        if address is None:
            raise ProtocolError(
                "P4",
                f"component {site.component!r} has no interior address in "
                f"{where} of the nnsight engine "
                "(neural/engines/nnsight_tracing/addresses.py) — extend the "
                "table, or route to the reference engine.",
            )
        return ResolvedTap(site=site, source=address)

    def _source_value(self, tap: ResolvedTap) -> tuple[Any, Any]:
        """The traced tensor an interior address names, navigated in-trace —
        plus, for an ``align`` address, the kernel's sorted→token permutation.

        Memoized per trace by the address' op identity (before
        ``tuple_index``), so two components on one op — q and k are the two
        elements of one rope call — request it once; ``.source`` ops refuse a
        second request after the model has run past them. The permutation is
        requested *first*, matching op order (the sort fires before anything
        it sorted for).
        """
        address = tap.source
        assert address is not None
        site = tap.site
        key = (
            id(site.module),
            address.op_pattern,
            address.peel,
            address.field,
            address.arg,
        )
        if key not in self._trace_sources:
            root = self._drill(site.module.source, address.op_pattern, site)
            perm = None
            if address.align is not None:
                # its own cache slot: several addresses share one sort op, and
                # the permutation must be requested exactly once per trace
                perm_key = (id(site.module), address.op_pattern, "align", address.align)
                if perm_key not in self._trace_sources:
                    self._trace_sources[perm_key] = self._drill(
                        root.source, address.align, site
                    ).output[1]
                perm = self._trace_sources[perm_key]
            op = root
            for entry in address.peel:
                op = self._drill(op.source, entry, site)
            if address.field is not None:
                op = self._drill(op.source, address.field, site)
            if address.arg is not None:
                index, keyword = address.arg
                self._trace_sources[key] = (op.inputs[index][keyword], perm)
            else:
                self._trace_sources[key] = (op.output, perm)
        value, perm = self._trace_sources[key]
        if address.tuple_index is not None:
            value = value[address.tuple_index]
        return value, perm

    def _present_native(self, tap: ResolvedTap, value: Any, perm: Any) -> Any:
        """The value as the declared shape describes it — semantic order.

        An ``align`` address' rows are in the kernel's expert-sorted order:
        un-sort them (a gather, so the result is a copy — writes go back
        through :meth:`_land_writes`' restore path, never through this).
        ``expert_rows`` re-packs ``(batch·position·top_k, …)`` rows into the
        declared 2-D native ``(batch·position, top_k·…)``.
        """
        address = tap.source
        assert address is not None
        if address.align is not None:
            value = value[torch.argsort(perm)]
        if address.expert_rows:
            # (batch·position·top_k, d) → (batch·position, top_k·d); an
            # integral tap has no feature axis, so its row is the top-k itself
            row_width = tap.site.shape.width or self._topk_width(tap.site)
            value = value.reshape(-1, row_width)
        return value

    @staticmethod
    def _topk_width(site: ResolvedSite) -> int:
        width = next(
            (axis.width for axis in site.shape.axes if axis.kind == "topk"), None
        )
        assert width is not None  # expert_rows implies a top-k axis
        return width

    def _drill(self, source: Any, pattern: str, site: ResolvedSite) -> Any:
        """One matched op on one ``.source``, with the refusal made legible."""

        def line_of(name: str) -> str:
            op = getattr(source, name)
            text, line = getattr(op, "text", None), getattr(op, "line", None)
            if not isinstance(text, str) or not isinstance(line, int):
                return ""
            return text.splitlines()[line - 1]

        try:
            return getattr(source, match_op(pattern, source.names, line_of))
        except AddressResolutionError as error:
            import transformers

            raise ProtocolError(
                "P4",
                f"addressing {site.component!r} at layer {site.layer} of "
                f"{self.bundle.key!r} (transformers "
                f"{transformers.__version__}): {error}",
            ) from error

    @contextlib.contextmanager
    def _switched_implementations(self, required: frozenset[str]) -> Iterator[None]:
        """Run one trace under the implementations its addresses require.

        📐 The runtime switch is verified on the real A3B and the fixture
        (`set_attn_implementation` both directions). The model default is
        restored afterwards, so a document that never touches the pattern
        keeps whatever the checkpoint prefers (sdpa) — D5, decided: on demand.
        """
        model = self.bundle.model
        previous_attn: str | None = None
        previous_experts: str | None = None
        if required:
            # 📐 nnsight dispatches the real model on first trace, and dispatch
            # rebuilds the config — a switch applied before it is silently
            # reset (measured: eager set pre-dispatch, sdpa ran). Dispatch
            # first, so the switch lands on the model that will run.
            if not getattr(model, "dispatched", True):
                model.dispatch()
        if "attn_eager" in required and model.config._attn_implementation != "eager":
            previous_attn = model.config._attn_implementation
            model.set_attn_implementation("eager")
        if "experts_grouped" in required:
            # grouped_mm is the checkpoint default (an unset value means it),
            # so this only ever switches back from an explicit "eager"
            current = getattr(model.config, "_experts_implementation", None)
            if current is not None and current != "grouped_mm":
                previous_experts = current
                model.set_experts_implementation("grouped_mm")
        try:
            yield
        finally:
            if previous_attn is not None:
                model.set_attn_implementation(previous_attn)
            if previous_experts is not None:
                model.set_experts_implementation(previous_experts)

    # ------------------------------------------------------------------ #
    # in-trace plumbing
    # ------------------------------------------------------------------ #

    def _tap_contract(self, tap: ResolvedTap, batch_size: int) -> torch.Tensor:
        """One tap's tensor in the contract shape — read off its envoy, or
        off the op its interior address names."""
        site = tap.site
        if tap.source is not None:
            raw, perm = self._source_value(tap)
            native = self._present_native(tap, raw, perm)
            return to_contract(native, site.shape, batch_size=batch_size)
        envoy = site.module
        if site.kind == "out":
            native = tap_tensor(envoy.output, site.tuple_index)
        else:
            native = envoy.input
        return to_contract(native, site.shape, batch_size=batch_size)

    def _land_writes(
        self,
        tap: ResolvedTap,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
    ) -> None:
        """Apply every write at one address (shared class-ordered math) and
        assign the result back — to the envoy, or into the interior op's
        tensor.

        The interior landing is an in-place fill (``value[:] = new``): the op
        has already produced its tensor when the write runs, and everything
        downstream — the next op in the same forward, a later read of the
        same address — consumes that same object. 📐 Verified consumed on the
        fixture and the real A3B (zeroing the softmax input moved the logits
        by 1.78; a pattern write is #53's finding).
        """
        site = tap.site
        batch_size = int(batch.input_ids.shape[0])
        if tap.source is not None:
            address = tap.source
            raw, perm = self._source_value(tap)
            native = self._present_native(tap, raw, perm).clone()
            contract = to_contract(native, site.shape, batch_size=batch_size)
            self._apply_writes_to_contract(entries, input_role, batch, contract)
            new_native = from_contract(
                contract, site.shape, batch_size=batch_size, native=native
            )
            if address.expert_rows:
                # back to one row per (token, slot), then to the kernel's own
                # order — the exact inverse of _present_native. (The integral
                # expert tap is read-only policy, so a writable one always has
                # a feature axis to size the row by.)
                width = site.shape.width
                assert width is not None
                rows = new_native.reshape(-1, width // self._topk_width(site))
                if address.align is not None:
                    rows = rows[perm]
                raw[:] = rows
            else:
                raw[:] = new_native
            return
        envoy = site.module
        if site.kind == "out":
            payload = envoy.output
            native = tap_tensor(payload, site.tuple_index).clone()
        else:
            payload = None
            native = envoy.input.clone()
        contract = to_contract(native, site.shape, batch_size=batch_size)
        self._apply_writes_to_contract(entries, input_role, batch, contract)
        new_native = from_contract(
            contract, site.shape, batch_size=batch_size, native=native
        )
        if site.kind == "out":
            envoy.output = rebuild_payload(payload, site.tuple_index, new_native)
        else:
            envoy.input = new_native
