"""Point-protocol execution over raw pytorch hooks (spec §4).

One :class:`PointExecutor` runs one concrete document: forward groups are
``original`` on every input it is read on plus each intervened model on
its declared input; groups run lazily in operand-dependency order (the
validated-acyclic model graph); within a group every in-force write applies
at its address — absolute first, additive deltas summed, renormalize last
against the pre-write norm — and reads see the fully written state because
pytorch fires hooks in module-execution order and chains their return
values at one module.

The document-and-contract half (positions, gathers, featurizers, the write
math) is :class:`~causalab.neural.shared.executor_base.ExecutorBase`; this
module is the hook half: the group forward, the greedy decode, and the
install/capture plumbing, which mirrors the oracle library
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

from causalab.neural.engines.pytorch_hooks.attention_interface import (
    InterfaceTap,
    attention_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.delta_interface import (
    DeltaTap,
    delta_kernel_taps,
)
from causalab.neural.engines.pytorch_hooks.experts_interface import (
    ExpertsTap,
    experts_interface_taps,
)
from causalab.neural.shared.encoding import (
    EncodedBatch,
    continuation_frame,
    resolve_steps,
)
from causalab.neural.shared.executor_base import (
    ExecutorBase,
    ForwardCache,
    Interning,
    RaggedValue,
    TapKey,
    document_seed,
    refuse_unstackable,
    tap_key,
)
from causalab.protocol.shapes import FeatureShape
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
from causalab.protocol.schema import ReadSpec, SiteSpec

__all__ = [
    "ForwardCache",
    "Interning",
    "PointExecutor",
    "RaggedValue",
    "document_seed",
]


class PointExecutor(ExecutorBase):
    """Execute one concrete document against one loaded model, over hooks."""

    # ------------------------------------------------------------------ #
    # group execution
    # ------------------------------------------------------------------ #

    def _run_group(self, model: str, input_role: str) -> None:
        if (model, input_role) in self._groups_run:
            return
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

        capture_sites = {
            (rname): resolve_site(self.bundle, self.doc.sites[str(read.site)])
            for rname, read in taps
        }
        for rname, site in capture_sites.items():
            _refuse_interior(f"read {rname!r}", site)
        # A continuation read at lm_head is served from kept ln_final
        # activations (d_model, not vocab) and projected at its addressed
        # steps — the same value, without ever building the whole vocabulary
        # for every step. Any other site is captured as itself.
        gen_sites = {
            rname: resolve_site(self.bundle, self.doc.sites[str(read.site)])
            for rname, read in gen_taps
        }
        for rname, site in gen_sites.items():
            _refuse_interior(f"read {rname!r}", site)
        gen_capture_sites = {
            rname: (
                resolve_site(self.bundle, SiteSpec(component="ln_final"))
                if site.component == "lm_head"
                else site
            )
            for rname, site in gen_sites.items()
        }

        # Cross-point interning (§3): a group another point already ran under
        # this digest is served from the shared captures rather than run a
        # second, identical time. Two exemptions, both principled rather than
        # cautious: a **decoding** group's value is the continuation it
        # produced, not activations a later point can replay from a cache (§4
        # exempts it from elision for the same reason); and a **grad-enabled**
        # group's reads have to stay attached to the graph the training step
        # differentiates, which a detached capture from another pass is not.
        digest = (
            self._group_digest(model, input_role)
            if not depth and not self.grad_enabled
            else None
        )
        interned = self._interned(
            digest, (tap_key(site) for site in capture_sites.values())
        )
        prefill: Any = None
        if interned is not None:
            capture, idx_capture = interned
        else:
            capture, idx_capture, prefill = self._forward_group(
                model,
                input_role,
                digest=digest,
                capture_sites=capture_sites,
                depth=depth,
            )

        batch = self._batch(input_role)
        for rname, read in taps:
            site = capture_sites[rname]
            raw = capture[tap_key(site)]
            self._read_values[rname] = self._finalize_read(
                rname,
                read,
                site,
                raw,
                batch,
                input_role,
                expert_idx=idx_capture.get(tap_key(site)),
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

    def _forward_group(
        self,
        model: str,
        input_role: str,
        *,
        digest: str | None,
        capture_sites: Mapping[str, ResolvedSite],
        depth: int,
    ) -> tuple[dict[TapKey, torch.Tensor], dict[TapKey, torch.Tensor], Any]:
        """Run one group's forward; return its raw captures, their routing
        tables, and the prefill output a decode continues from.

        What it captures is this point's taps **unioned with every other
        address the campaign asks of the same digest** (§3): that union is
        what lets the single pass a shared digest earns serve every point, and
        it is why an eliding engine has to stop at the deepest tap of the
        union rather than of the point that happened to run first.
        """
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
        write_hooks = self._build_write_hooks(write_names, input_role, batch)
        capture: dict[TapKey, torch.Tensor] = {}
        # the routing table alongside each experts-interface capture — what the
        # `expert:` sub-axis joins on (executor_base._expert_selected)
        idx_capture: dict[TapKey, torch.Tensor] = {}
        for site, _ in write_hooks:
            _refuse_interior(f"write at {site.component!r}", site)
        # this point's taps first, then the campaign's — the sites another
        # point will ask of this same forward, so it never has to run it again
        shared_sites: list[ResolvedSite] = []
        if digest is not None and self.interning is not None:
            for spec in self.interning.cache.wanted.get(digest, ()):
                site = resolve_site(self.bundle, spec)
                _refuse_interior(f"shared read at {site.component!r}", site)
                shared_sites.append(site)
        tapped: list[ResolvedSite] = [*capture_sites.values(), *shared_sites]
        with contextlib.ExitStack() as hooks:
            # Four of the mixer's tensors — and the attention pattern's *write*
            # — are not module boundaries: transformers computes them inside one
            # `attention_interface(...)` call, so a forward hook on the mixer
            # fires after they have already been consumed. They are collected
            # first and installed together, because the interception is one
            # registry entry and nesting two of them would let the inner
            # wrapper's edits replace the outer's.
            interface: dict[int, list[InterfaceTap]] = {}
            experts: dict[int, list[ExpertsTap]] = {}
            delta: dict[Any, list[DeltaTap]] = {}
            batch_size = batch.input_ids.shape[0]

            for site, fn in write_hooks:
                if site.kind == "experts":
                    assert site.interface_slot is not None
                    experts.setdefault(id(site.module), []).append(
                        ExpertsTap(
                            slot=site.interface_slot,
                            edit=_experts_edit(site, fn, batch_size),
                        )
                    )
                    continue
                if site.kind == "delta":
                    assert site.interface_slot is not None
                    if site.interface_slot == "state":
                        # a state write must feed forward, so it rides the
                        # stepwise substitution's own surface — `fn` here IS
                        # the per-step writer (_build_write_hooks)
                        delta.setdefault(site.module, []).append(
                            DeltaTap(slot="state", edit_state=fn)
                        )
                        continue
                    delta.setdefault(site.module, []).append(
                        DeltaTap(
                            slot=site.interface_slot,
                            edit=_interface_edit(site, fn, batch_size),
                        )
                    )
                    continue
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
            for site in tapped:
                key = tap_key(site)
                if key in capture:
                    continue
                capture[key] = torch.empty(0)  # placeholder; filled by the tap
                if site.kind == "experts":
                    assert site.interface_slot is not None
                    experts.setdefault(id(site.module), []).append(
                        ExpertsTap(
                            slot=site.interface_slot,
                            read=_experts_capture(
                                capture, idx_capture, key, site, batch_size
                            ),
                        )
                    )
                    continue
                if site.kind == "delta":
                    assert site.interface_slot is not None
                    delta.setdefault(site.module, []).append(
                        DeltaTap(
                            slot=site.interface_slot,
                            read=_interface_capture(capture, key, site, batch_size),
                        )
                    )
                    continue
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
                    {mid: tuple(entries) for mid, entries in interface.items()}
                )
            )
            hooks.enter_context(
                experts_interface_taps(
                    {mid: tuple(entries) for mid, entries in experts.items()}
                )
            )
            hooks.enter_context(
                delta_kernel_taps(
                    {mixer: tuple(entries) for mixer, entries in delta.items()}
                )
            )
            with torch.enable_grad() if self.grad_enabled else torch.no_grad():
                prefill = self.bundle.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids(),
                    use_cache=depth > 0,
                )

        # only what a tap actually filled: a placeholder whose module never ran
        # would hand a later point an empty capture instead of letting it run
        # the forward
        self._publish(
            digest,
            f"{model}/{input_role}",
            {key: value for key, value in capture.items() if value.numel()},
            idx_capture,
        )
        return capture, idx_capture, prefill

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
        idx_steps: dict[TapKey, list[torch.Tensor]] = {}
        cache = prefill.past_key_values
        with contextlib.ExitStack() as hooks:
            interface: dict[int, list[InterfaceTap]] = {}
            experts: dict[int, list[ExpertsTap]] = {}
            delta: dict[Any, list[DeltaTap]] = {}
            batch_size = batch.input_ids.shape[0]
            for name, site in gen_capture_sites.items():
                refuse_unstackable(name, site)
                key = tap_key(site)
                if key in steps:
                    continue
                steps[key] = []
                if site.kind == "experts":
                    assert site.interface_slot is not None
                    idx_steps[key] = []
                    experts.setdefault(id(site.module), []).append(
                        ExpertsTap(
                            slot=site.interface_slot,
                            read=_experts_accumulate(
                                steps[key], idx_steps[key], site, batch_size
                            ),
                        )
                    )
                    continue
                if site.kind == "delta":
                    assert site.interface_slot is not None
                    delta.setdefault(site.module, []).append(
                        DeltaTap(
                            slot=site.interface_slot,
                            read=_interface_accumulate(steps[key], site, batch_size),
                        )
                    )
                    continue
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
                    {mid: tuple(entries) for mid, entries in interface.items()}
                )
            )
            hooks.enter_context(
                experts_interface_taps(
                    {mid: tuple(entries) for mid, entries in experts.items()}
                )
            )
            hooks.enter_context(
                delta_kernel_taps(
                    {mixer: tuple(entries) for mixer, entries in delta.items()}
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
        continuation = continuation_frame(
            self.bundle.tokenizer, generated, tuple(widths)
        )
        self._continuations[(model, input_role)] = continuation

        head = None
        for rname, read in gen_taps:
            site = gen_sites[rname]
            capture_site = gen_capture_sites[rname]
            stacked = torch.cat(steps[tap_key(capture_site)], 1)
            stacked_idx = (
                torch.cat(idx_steps[tap_key(capture_site)], 1)
                if idx_steps.get(tap_key(capture_site))
                else None
            )
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
                expert_idx=stacked_idx,
            )

    # ------------------------------------------------------------------ #
    # writes: land the shared math through hooks
    # ------------------------------------------------------------------ #

    def _build_write_hooks(
        self, write_names: tuple[str, ...], input_role: str, batch: EncodedBatch
    ) -> list[tuple[ResolvedSite, Callable[..., Any]]]:
        """One in-place writer per written-to tap, applying every write at that
        address in class order (the shared write math, executor_base).

        Returns the *site* rather than its parts: a tap may be a module
        boundary or an attention-interface slot, and only the site knows which.
        """
        hooks: list[tuple[ResolvedSite, Callable[..., Any]]] = []
        for site, entries in self._resolve_write_addresses(write_names).values():
            if site.kind == "delta" and site.interface_slot == "state":
                # the one address whose writer is per-step (step, S) -> S:
                # a state edit feeds forward, so the whole-tensor contract
                # cannot express it (executor_base._state_step_writer)
                hooks.append(
                    (site, self._state_step_writer(entries, input_role, batch))
                )
                continue
            hooks.append((site, self._address_writer(entries, input_role, batch)))
        return hooks

    def _address_writer(
        self,
        entries: list[Any],
        input_role: str,
        batch: EncodedBatch,
    ) -> Callable[[torch.Tensor], None]:
        def apply(tensor: torch.Tensor) -> None:
            self._apply_writes_to_contract(entries, input_role, batch, tensor)

        return apply


# --------------------------------------------------------------------------- #
# hook plumbing (mirrors the oracle's _install / capture helpers)
# --------------------------------------------------------------------------- #


def _interface_edit(
    site: ResolvedSite, write: Callable[[torch.Tensor], None], batch_size: int
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
    site: ResolvedSite,
    batch_size: int,
) -> Callable[[torch.Tensor], None]:
    """The read half — the same contract shape ``_capturing`` produces."""

    def read(native: torch.Tensor) -> None:
        sink[key] = to_contract(native, site.shape, batch_size=batch_size)

    return read


def _interface_accumulate(
    sink: list[torch.Tensor], site: ResolvedSite, batch_size: int
) -> Callable[[torch.Tensor], None]:
    """The read half for a decode: append per step, as ``_accumulating`` does."""

    def read(native: torch.Tensor) -> None:
        sink.append(to_contract(native, site.shape, batch_size=batch_size))

    return read


def _contract_idx(idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """The routing table in contract form: ``(tokens, top_k)`` token-major →
    ``(batch, position, top_k)`` — the same split every flat_batch shape uses."""
    return idx.reshape(batch_size, -1, idx.shape[-1])


def _experts_edit(
    site: ResolvedSite, write: Callable[[torch.Tensor], None], batch_size: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Adapt an in-place contract-shaped writer to the experts interface.

    Same contract as :func:`_interface_edit`: the manager hands out a clone in
    the taps' token-major form, this converts it to ``(batch, position,
    feature)``, lets the shared write math mutate it, and converts back.

    A site naming an ``expert`` writes only that expert's rows: the write math
    runs over the whole contract tensor as usual, and the merge keeps its
    result exactly where the routing table names that expert — an expert no
    token chose therefore receives a write that lands nowhere, the data-fact
    twin of the width-0 read.
    """

    def edit(native: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        contract = to_contract(native, site.shape, batch_size=batch_size)
        if site.expert is None:
            write(contract)
            return from_contract(
                contract, site.shape, batch_size=batch_size, native=native
            )
        original = contract.clone()
        write(contract)
        idx_c = _contract_idx(idx, batch_size)  # (b, s, top_k)
        top_k = idx_c.shape[-1]
        per_slot = contract.shape[-1] // top_k
        mask = (
            (idx_c == site.expert)
            .unsqueeze(-1)
            .expand(*idx_c.shape, per_slot)
            .reshape(contract.shape)
        )
        merged = torch.where(mask, contract, original)
        return from_contract(merged, site.shape, batch_size=batch_size, native=native)

    return edit


def _experts_capture(
    sink: dict[Any, torch.Tensor],
    idx_sink: dict[Any, torch.Tensor],
    key: Any,
    site: ResolvedSite,
    batch_size: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """The read half — the same contract shape ``_capturing`` produces, plus
    the routing table the ``expert:`` sub-axis joins on."""

    def read(native: torch.Tensor, idx: torch.Tensor) -> None:
        sink[key] = to_contract(native, site.shape, batch_size=batch_size)
        idx_sink[key] = _contract_idx(idx, batch_size)

    return read


def _experts_accumulate(
    sink: list[torch.Tensor],
    idx_sink: list[torch.Tensor],
    site: ResolvedSite,
    batch_size: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """The read half for a decode: append per step, as ``_accumulating`` does.

    Safe for every experts-interface shape: the interior is token-indexed, so a
    decode step is exactly one position per row and the steps stack on the
    position axis (unlike ``attention_key``, nothing here grows with the
    prefix). The routing table accumulates in lockstep — which experts each
    *generated* token was sent to.
    """

    def read(native: torch.Tensor, idx: torch.Tensor) -> None:
        sink.append(to_contract(native, site.shape, batch_size=batch_size))
        idx_sink.append(_contract_idx(idx, batch_size))

    return read


def _refuse_interior(what: str, site: ResolvedSite) -> None:
    """Refuse a tap this engine has no mechanism for, naming the one that does.

    ``kind="interior"`` marks a tensor computed *inside* a fused forward (the
    per-expert MoE interior, N6; the Gated DeltaNet interior, N7): there is no
    module boundary for a hook and no per-family interface registry to wrap,
    so the tap belongs to the nnsight engine's ``.source`` addressing. Routing
    already keeps such documents away (the component is absent from this
    engine's declaration); this refusal is for one arriving unrouted.
    """
    if site.kind != "interior":
        return
    raise ProtocolError(
        "P4",
        f"{what} addresses {site.component!r}, which lives inside a fused "
        "forward where no pytorch hook can reach — the nnsight engine "
        "serves it (its `.source` address table, "
        "neural/engines/nnsight_tracing/addresses.py). Routing sends such "
        "documents there.",
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
    in place; the model always gets its native shape back. For the default
    ``native`` is handed to :func:`from_contract` because a fused tap's other
    splits live in it and have to survive the write untouched.
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
