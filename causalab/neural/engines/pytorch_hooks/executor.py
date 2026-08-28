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
from causalab.neural.engines.pytorch_hooks.experts_interface import (
    ExpertsTap,
    experts_interface_taps,
)
from causalab.neural.shared.encoding import Continuation, EncodedBatch, resolve_steps
from causalab.neural.shared.executor_base import (
    ExecutorBase,
    RaggedValue,
    TapKey,
    document_seed,
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

__all__ = ["PointExecutor", "RaggedValue", "document_seed"]


class PointExecutor(ExecutorBase):
    """Execute one concrete document against one loaded model, over hooks."""

    # ------------------------------------------------------------------ #
    # group execution
    # ------------------------------------------------------------------ #

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
            # Four of the mixer's tensors — and the attention pattern's *write*
            # — are not module boundaries: transformers computes them inside one
            # `attention_interface(...)` call, so a forward hook on the mixer
            # fires after they have already been consumed. They are collected
            # first and installed together, because the interception is one
            # registry entry and nesting two of them would let the inner
            # wrapper's edits replace the outer's.
            interface: dict[int, list[InterfaceTap]] = {}
            experts: dict[int, list[ExpertsTap]] = {}
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
                key = tap_key(site)
                if key in capture:
                    continue
                capture[key] = torch.empty(0)  # placeholder; filled by the tap
                if site.kind == "experts":
                    assert site.interface_slot is not None
                    experts.setdefault(id(site.module), []).append(
                        ExpertsTap(
                            slot=site.interface_slot,
                            read=_experts_capture(capture, key, site, batch_size),
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
            with torch.enable_grad() if self.grad_enabled else torch.no_grad():
                prefill = self.bundle.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids(),
                    use_cache=depth > 0,
                )

        for rname, read in taps:
            site = capture_sites[rname]
            raw = capture[tap_key(site)]
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
            experts: dict[int, list[ExpertsTap]] = {}
            batch_size = batch.input_ids.shape[0]
            for name, site in gen_capture_sites.items():
                _refuse_unstackable(name, site)
                key = tap_key(site)
                if key in steps:
                    continue
                steps[key] = []
                if site.kind == "experts":
                    assert site.interface_slot is not None
                    experts.setdefault(id(site.module), []).append(
                        ExpertsTap(
                            slot=site.interface_slot,
                            read=_experts_accumulate(steps[key], site, batch_size),
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
            stacked = torch.cat(steps[tap_key(capture_site)], 1)
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

    # ------------------------------------------------------------------ #
    # writes: land the shared math through hooks
    # ------------------------------------------------------------------ #

    def _build_write_hooks(
        self, write_names: tuple[str, ...], input_role: str, batch: EncodedBatch
    ) -> list[tuple[ResolvedSite, Callable[[torch.Tensor], None]]]:
        """One in-place writer per written-to tap, applying every write at that
        address in class order (the shared write math, executor_base).

        Returns the *site* rather than its parts: a tap may be a module
        boundary or an attention-interface slot, and only the site knows which.
        """
        hooks: list[tuple[ResolvedSite, Callable[[torch.Tensor], None]]] = []
        for site, entries in self._resolve_write_addresses(write_names).values():
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


def _experts_edit(
    site: ResolvedSite, write: Callable[[torch.Tensor], None], batch_size: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Adapt an in-place contract-shaped writer to the experts interface.

    Same contract as :func:`_interface_edit`: the manager hands out a clone in
    the taps' token-major form, this converts it to ``(batch, position,
    feature)``, lets the shared write math mutate it, and converts back. The
    routing table rides along unused here — round 3.2's ``expert`` sub-axis is
    what consumes it.
    """

    def edit(native: torch.Tensor, _idx: torch.Tensor) -> torch.Tensor:
        contract = to_contract(native, site.shape, batch_size=batch_size)
        write(contract)
        return from_contract(contract, site.shape, batch_size=batch_size, native=native)

    return edit


def _experts_capture(
    sink: dict[Any, torch.Tensor],
    key: Any,
    site: ResolvedSite,
    batch_size: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """The read half — the same contract shape ``_capturing`` produces."""

    def read(native: torch.Tensor, _idx: torch.Tensor) -> None:
        sink[key] = to_contract(native, site.shape, batch_size=batch_size)

    return read


def _experts_accumulate(
    sink: list[torch.Tensor], site: ResolvedSite, batch_size: int
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """The read half for a decode: append per step, as ``_accumulating`` does.

    Safe for every experts-interface shape: the interior is token-indexed, so a
    decode step is exactly one position per row and the steps stack on the
    position axis (unlike ``attention_key``, nothing here grows with the
    prefix).
    """

    def read(native: torch.Tensor, _idx: torch.Tensor) -> None:
        sink.append(to_contract(native, site.shape, batch_size=batch_size))

    return read


def _refuse_unstackable(name: str, site: ResolvedSite) -> None:
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
