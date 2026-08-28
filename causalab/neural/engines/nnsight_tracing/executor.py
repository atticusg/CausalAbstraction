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

Operations are issued in forward-execution order (site depth, writes before
reads at the same depth) — module-boundary envoys are order-tolerant, but the
``.source`` interiors that arrive in N5–N7 are not (``MissedProviderError``),
so the discipline starts here.

v1 boundaries: no generated frame (arrives with N8's ``tracer.iter`` step
anchoring; routing keeps ``generate`` documents on the reference engine) and
no ``attention_probs`` (N5's eager interface taps; likewise routed away).
Both still refuse legibly here in case a document arrives unrouted.
"""

from __future__ import annotations

import torch

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

        read_sites = {
            rname: resolve_site(self.bundle, self.doc.sites[str(read.site)])
            for rname, read in taps
        }
        write_addresses = self._resolve_write_addresses(write_names)
        for site in (
            *read_sites.values(),
            *(site for site, _ in write_addresses.values()),
        ):
            if site.component == "attention_probs":
                raise ProtocolError(
                    "P4",
                    "'attention_probs' is not in the nnsight engine's "
                    "component set yet — its attention-interior taps (the "
                    "eager interface source) are phase N5 of the engine plan; "
                    "the reference engine serves it today.",
                )

        # one trace per group, operations in forward-execution order; a write
        # lands before a read at the same depth so the read sees it
        operations: list[tuple[int, tuple[int, int], str, ResolvedSite, list]] = []
        for site, entries in write_addresses.values():
            operations.append((0, site.depth, "write", site, entries))
        seen_keys: set = set()
        for site in read_sites.values():
            key = tap_key(site)
            if key not in seen_keys:
                seen_keys.add(key)
                operations.append((1, site.depth, "read", site, []))
        operations.sort(key=lambda op: (op[1], op[0]))

        import nnsight

        batch_size = int(batch.input_ids.shape[0])
        saves: dict = {}
        with torch.no_grad():
            with self.bundle.model.trace(
                {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                },
                position_ids=batch.position_ids(),
            ):
                for _, _, op_kind, site, entries in operations:
                    if op_kind == "write":
                        self._land_writes(site, entries, input_role, batch)
                    else:
                        saves[tap_key(site)] = nnsight.save(
                            self._tap_contract(site, batch_size)
                        )

        for rname, read in taps:
            site = read_sites[rname]
            raw = saves[tap_key(site)]
            self._read_values[rname] = self._finalize_read(
                rname, read, site, raw, batch, input_role
            )
        self._groups_run.add((model, input_role))

    # ------------------------------------------------------------------ #
    # in-trace plumbing
    # ------------------------------------------------------------------ #

    def _tap_contract(self, site: ResolvedSite, batch_size: int) -> torch.Tensor:
        """One tap's tensor in the contract shape, read off its envoy."""
        envoy = site.module
        if site.kind == "out":
            native = tap_tensor(envoy.output, site.tuple_index)
        else:
            native = envoy.input
        return to_contract(native, site.layout, batch_size=batch_size)

    def _land_writes(
        self,
        site: ResolvedSite,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
    ) -> None:
        """Apply every write at one address (shared class-ordered math) and
        assign the result back to the envoy."""
        envoy = site.module
        batch_size = int(batch.input_ids.shape[0])
        if site.kind == "out":
            payload = envoy.output
            native = tap_tensor(payload, site.tuple_index).clone()
        else:
            payload = None
            native = envoy.input.clone()
        contract = to_contract(native, site.layout, batch_size=batch_size)
        self._apply_writes_to_contract(entries, input_role, batch, contract)
        new_native = from_contract(contract, site.layout, batch_size=batch_size)
        if site.kind == "out":
            envoy.output = rebuild_payload(payload, site.tuple_index, new_native)
        else:
            envoy.input = new_native
