"""Case machinery for the SH1 per-mode parity & captured-golden harness (#410).

One declarative registry (:func:`enumerate_cases`) drives three consumers —
the live-oracle sweep (``test_mode_parity.py``), the captured-golden replay
(``test_captured_goldens.py``), and the pin updater (``update_goldens.py``) —
so capture and assertion take exactly the same code path (the
``tests/_helpers/task_pins.py`` principle).

Every :class:`ModeCase` is realized twice:

* :func:`realize_new_stack` — through the nnterp Site/Edit/Plan stack
  (``Edit.apply`` / ``Edit.collect`` in a trace, or ``run_plan`` for the
  cross-input paths);
* :func:`realize_oracle` — through raw ``register_forward_hook`` ground truth
  (:mod:`tests.neural.activations.hook_oracle`), never importing pyvene or
  touching nnsight.

Model families are built fresh from seeded configs with **eager attention
forced** (``config._attn_implementation = "eager"``): the sibling per-module
suites inherit the transformers from-config default (sdpa) — sound for
live A/B equivalence, where both sides share one model instance — but pins
that must replay across runs and survive the SH3 SDPA flip (#424) need the
attention implementation nailed down explicitly.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, Callable, Literal

import torch
from nnterp import StandardizedTransformer

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.featurizer import Featurizer
from causalab.neural.head_view import HeadSite
from causalab.neural.modes import (
    MaskGate,
    collect,
    interchange,
    interpolate,
    mask,
    noise,
    replace,
    steer,
)
from causalab.neural.plan import EditOp, Plan, run_plan
from causalab.neural.site import Site
from causalab.neural.staged import lower_staged

from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import (
    capture_component,
    component_edited_logits,
    component_module,
    head_slice,
    next_token_logits,
    run_with_edits,
)

_TEXT = "the quick brown fox jumps"
_BASE_TEXT = "the quick brown fox jumps"
_SOURCE_TEXT = "a slow green turtle sleeps deeply"

MODES = ("collect", "replace", "steer", "interchange", "interpolate", "noise", "mask")
FAMILIES = ("llama", "gpt2", "gqa")

_NOISE_SEED = 7
_NOISE_SCALE = 3.0
_STEER_FACTOR = 2.5
_ALPHA = 0.3
_IDS = (0, 2)  # feature_ids for the subspace-scatter variant (k=4 featurizer)


# --------------------------------------------------------------------------- #
#  Families — fresh seeded models, eager attention forced and asserted         #
# --------------------------------------------------------------------------- #
def force_eager(cfg: Any) -> None:
    """Pin the attention implementation before construction. From-config HF
    builds default to sdpa (transformers 4.57); pins must be eager, the
    migration-wide parity policy (#410 / design doc Part 4, SH1 row)."""
    cfg._attn_implementation = "eager"


def _gqa(cfg: Any) -> None:
    assert cfg.num_attention_heads % 2 == 0
    cfg.num_key_value_heads = cfg.num_attention_heads // 2


def _decoupled(cfg: Any) -> None:
    # head_dim != hidden // n_heads — the Qwen3 shape pyvene mis-slices (#386).
    cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads + 2


_FAMILY_RECIPES: dict[str, tuple[Callable[..., tuple[Any, Any]], list[Callable]]] = {
    "llama": (fresh_tiny_random_llama, []),
    "gpt2": (fresh_tiny_random_gpt2, []),
    "gqa": (fresh_tiny_random_llama, [_gqa]),
    "decoupled": (fresh_tiny_random_llama, [_gqa, _decoupled]),
}


@dataclasses.dataclass
class ParityCase:
    """A model family, once as the nnterp-wrapped stack and once as the raw-HF
    oracle shim — weights shared, so comparisons are exact."""

    family: str
    st: StandardizedTransformer  # the new stack taps this
    oracle: Any  # SimpleNamespace(hf_model=raw) — for the hook_oracle helpers
    tok: Any

    @property
    def device(self) -> torch.device:
        return next(self.oracle.hf_model.parameters()).device

    def inputs(self, text: str = _TEXT) -> dict[str, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
        }

    def pair(self, t1: str, t2: str) -> tuple[dict, dict]:
        """Two single-row batches in ONE padded frame (tokenized together,
        left-padded) — what a multi-input plan requires."""
        enc = self.tok([t1, t2], padding=True, return_tensors="pt")

        def row(i: int) -> dict:
            return {
                "input_ids": enc["input_ids"][i : i + 1].to(self.device),
                "attention_mask": enc["attention_mask"][i : i + 1].to(self.device),
            }

        return row(0), row(1)

    def capture(self, component: str, layer: int, inputs: Any) -> torch.Tensor:
        """The raw-hook ground truth for ``(component, layer)`` — full ``(b, seq, d)``."""
        module, kind = component_module(self.oracle, layer, component)
        return capture_component(self.oracle, module, kind, inputs)

    def clean_logits(self, inputs: Any) -> torch.Tensor:
        with self.st.trace(inputs):
            clean = self.st.logits[:, -1, :].cpu().save()
        return clean

    def edited_logits(self, edit: Edit, inputs: Any) -> torch.Tensor:
        with self.st.trace(inputs):
            edit.apply(self.st)
            edited = self.st.logits[:, -1, :].cpu().save()
        return edited

    def oracle_edit_logits(
        self,
        site: Site,
        feat: Featurizer,
        g: Callable[[torch.Tensor], torch.Tensor],
        positions: list[int] | None,
        inputs: Any,
    ) -> torch.Tensor:
        """Ground truth: a hand-rolled forward hook applying featurize → ``g`` →
        inverse (with the base error) on the raw activation, offline. ``g``
        receives the **full** featurized slice — ``feature_ids`` scatters are
        emulated inside ``g`` (mirroring ``FeaturizedSite._rewrite``)."""
        module, kind = component_module(self.oracle, site.layer, site.component)

        def edit(h: torch.Tensor) -> None:
            sel = slice(None) if positions is None else positions
            f, err = feat.featurize(h[:, sel])
            h[:, sel] = feat.inverse_featurize(g(f), err).to(h.dtype)

        return component_edited_logits(self.oracle, inputs, module, kind, edit)


def build_family(name: str) -> ParityCase:
    factory, mutators = _FAMILY_RECIPES[name]

    def mutate(cfg: Any) -> None:
        for m in mutators:
            m(cfg)
        force_eager(cfg)

    raw, tok = factory(mutate_config=mutate)
    assert raw.config._attn_implementation == "eager", (
        f"{name}: expected eager attention, got {raw.config._attn_implementation!r}"
    )
    tok.padding_side = "left"  # the pipeline convention; makes [-1] the last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    st = StandardizedTransformer(raw, tokenizer=tok, check_renaming=True)
    # Dispatch materializes real weights so the first trace runs directly instead
    # of nnterp's shape-scan fallback (which, for GPT-2, hits a FakeTensor
    # data-dependent guard) — matching the F3 LMPipeline load path.
    st.dispatch()
    return ParityCase(family=name, st=st, oracle=SimpleNamespace(hf_model=raw), tok=tok)


# --------------------------------------------------------------------------- #
#  The case registry                                                           #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class ModeCase:
    """One cell of the parity matrix. ``src_layer``/``layer`` address the
    source and destination sites of the source-reading modes; single-site
    modes only use ``layer``."""

    family: str
    mode: str
    component: str = "block_output"
    layer: int = 1
    src_layer: int = 0
    featurizer: Literal["identity", "sub3", "ids"] = "sub3"
    positions: Literal["last", "all", "span"] = "last"
    path: Literal["edit", "plan", "staged", "head"] = "edit"
    head_kind: str | None = None  # "value" | "attention_value" (path="head")
    golden: bool = False

    @property
    def case_id(self) -> str:
        bits = [self.family, self.mode]
        if self.path == "head":
            bits.append(f"head_{self.head_kind}")
        else:
            bits.append(self.component)
        bits += [self.featurizer, self.positions]
        if self.path != "edit":
            bits.append(self.path)
        return ".".join(bits)


def enumerate_cases() -> list[ModeCase]:
    """The whole matrix, `case_id`-unique. The ``golden`` flag marks the
    canonical captured-pin subset (one cell per mode × family on
    ``block_output``, plus the gqa head cells)."""
    cases: list[ModeCase] = []

    # Canonical cells (also the captured-golden subset): mode × family on
    # block_output, subspace featurizer, last position.
    for family in FAMILIES:
        for mode in MODES:
            cases.append(ModeCase(family=family, mode=mode, golden=True))

    # Component axis: the non-hidden-width tap (mlp_activation) and the
    # pre-hook kind (block_input) on llama; mlp_activation again on gpt2
    # (a semantically different tensor — c_proj input, not act_fn output).
    for component in ("mlp_activation", "block_input"):
        for mode in MODES:
            cases.append(ModeCase(family="llama", mode=mode, component=component))
    for mode in MODES:
        cases.append(ModeCase(family="gpt2", mode=mode, component="mlp_activation"))

    # One-offs: the layer-less embeddings site (steer constant + interchange
    # from a precomputed tensor — the source_representations pattern) and an
    # attention_output interchange.
    cases += [
        ModeCase(family="llama", mode="steer", component="embeddings", layer=0),
        ModeCase(family="llama", mode="interchange", component="embeddings", layer=0),
        ModeCase(family="llama", mode="interchange", component="attention_output"),
    ]

    # Featurizer axis on llama/block_output: raw activation space (identity)
    # for every mode, and the feature_ids scatter for the three modes whose
    # scatter semantics differ most (swap / constant / gate).
    for mode in MODES:
        cases.append(ModeCase(family="llama", mode=mode, featurizer="identity"))
    for mode in ("interchange", "replace", "mask"):
        cases.append(ModeCase(family="llama", mode=mode, featurizer="ids"))

    # Positions axis on llama/block_output: whole-tensor (None) and a
    # multi-position span [0, last] for every mode.
    for positions in ("all", "span"):
        for mode in MODES:
            cases.append(ModeCase(family="llama", mode=mode, positions=positions))

    # Head sites (llama-family only: FeaturizedSite ranks via `forward_rank`,
    # exact only for separate-projection models): per-head collect and
    # interchange on GQA and on the decoupled-head_dim shape (#386's gap).
    for family in ("gqa", "decoupled"):
        for kind in ("value", "attention_value"):
            for mode in ("collect", "interchange"):
                cases.append(
                    ModeCase(
                        family=family,
                        mode=mode,
                        path="head",
                        head_kind=kind,
                        featurizer="identity",
                        golden=(family == "gqa"),
                    )
                )

    # Cross-input canonical interchange through the one scheduler (EU2 #483):
    # both legacy case ids survive — "plan" runs under lowering="single"
    # strictness, "staged" under the default auto; a structural assertion in
    # _realize_plan pins that both execute the same ONE-fused-trace program.
    for family in FAMILIES:
        for path in ("plan", "staged"):
            cases.append(
                ModeCase(family=family, mode="interchange", layer=0, path=path)
            )

    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case_id in the registry"
    return cases


# --------------------------------------------------------------------------- #
#  Realization — the same ModeCase through the new stack and through hooks     #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Realization:
    """What one side produced: last-position logits (``kind='logits'``, with
    the clean logits for non-vacuity checks) or a collected feature tensor."""

    kind: Literal["logits", "collect"]
    value: torch.Tensor
    clean: torch.Tensor | None = None


@dataclasses.dataclass
class _Ingredients:
    """Everything both realizations share for one ModeCase — computed once so
    the two sides can't drift."""

    inputs: dict[str, torch.Tensor]
    positions: list[int] | None
    n_pos: int
    feat: Featurizer
    feature_ids: tuple[int, ...] | None
    fwidth: int  # the width `g` sees (feature_ids gather when set)
    raw_width: int


def _ingredients(mc: ModeCase, pc: ParityCase) -> _Ingredients:
    inputs = pc.inputs()
    seq = int(inputs["input_ids"].shape[1])
    last = seq - 1
    positions = {"last": [last], "all": None, "span": [0, last]}[mc.positions]
    n_pos = seq if positions is None else len(positions)
    raw_width = int(pc.capture(mc.component, mc.layer, inputs).shape[-1])
    if mc.featurizer == "identity":
        feat, feature_ids, fwidth = Featurizer(), None, raw_width
    elif mc.featurizer == "sub3":
        torch.manual_seed(0)
        feat = SubspaceFeaturizer(shape=(raw_width, 3), trainable=False)
        feature_ids, fwidth = None, 3
    else:  # "ids" — k=4 rotation, scatter into columns _IDS
        torch.manual_seed(0)
        feat = SubspaceFeaturizer(shape=(raw_width, 4), trainable=False)
        feature_ids, fwidth = _IDS, len(_IDS)
    return _Ingredients(
        inputs=inputs,
        positions=positions,
        n_pos=n_pos,
        feat=feat,
        feature_ids=feature_ids,
        fwidth=fwidth,
        raw_width=raw_width,
    )


def _constant(mc: ModeCase, ing: _Ingredients) -> torch.Tensor:
    """The deterministic constant the constant-fed modes use (replace value,
    steering vector, tensor interchange source)."""
    span = {"replace": (-30.0, 30.0), "steer": (3.0, 9.0), "interchange": (-5.0, 5.0)}
    lo, hi = span[mc.mode]
    return torch.linspace(lo, hi, ing.fwidth)


def _gate(ing: _Ingredients) -> MaskGate:
    """A deterministic hard eval-mode gate: on where its logit is positive
    (an ends-anchored linspace turns on the upper half of the features)."""
    gate = MaskGate(ing.fwidth).eval()
    with torch.no_grad():
        gate.mask.copy_(torch.linspace(-2.0, 2.0, ing.fwidth))
    return gate


def _gate_on(ing: _Ingredients) -> list[int]:
    return [
        i
        for i, v in enumerate(torch.linspace(-2.0, 2.0, ing.fwidth).tolist())
        if torch.sigmoid(torch.tensor(v)) > 0.5
    ]


def _dst_fsite(mc: ModeCase, ing: _Ingredients) -> FeaturizedSite:
    return FeaturizedSite(
        Site(mc.component, mc.layer), ing.feat, feature_ids=ing.feature_ids
    )


def _src_fsite(mc: ModeCase, ing: _Ingredients) -> FeaturizedSite:
    return FeaturizedSite(
        Site(mc.component, mc.src_layer), ing.feat, feature_ids=ing.feature_ids
    )


def _oracle_src_features(mc: ModeCase, pc: ParityCase, ing: _Ingredients):
    """Offline source features for the source-reading modes: featurize the
    hook-captured source-site slice (gathered to ``feature_ids`` when set)."""
    sel = slice(None) if ing.positions is None else ing.positions
    f_src, _ = ing.feat.featurize(
        pc.capture(mc.component, mc.src_layer, ing.inputs)[:, sel]
    )
    if ing.feature_ids is not None:
        f_src = f_src[..., list(ing.feature_ids)]
    return f_src


def _scattered(ing: _Ingredients, g_sel: Callable) -> Callable:
    """Lift a gathered-space ``g_sel`` to the full featurized width, emulating
    ``FeaturizedSite._rewrite``'s feature_ids scatter for the oracle side."""
    if ing.feature_ids is None:
        return g_sel
    ids = list(ing.feature_ids)

    def g(f: torch.Tensor) -> torch.Tensor:
        out = f.clone()
        out[..., ids] = g_sel(f[..., ids])
        return out

    return g


def _build_edit(mc: ModeCase, pc: ParityCase, ing: _Ingredients) -> Edit:
    """The new-stack Edit for one single-input ModeCase."""
    fsite = _dst_fsite(mc, ing)
    pos = ing.positions
    if mc.mode == "collect":
        return collect(fsite, positions=pos)
    if mc.mode == "replace":
        return replace(fsite, _constant(mc, ing), positions=pos)
    if mc.mode == "steer":
        return steer(fsite, _constant(mc, ing), factor=_STEER_FACTOR, positions=pos)
    if mc.mode == "noise":
        return noise(fsite, _NOISE_SCALE, seed=_NOISE_SEED, positions=pos)
    if mc.mode == "interchange":
        if mc.component == "embeddings":  # precomputed-tensor source
            return interchange(fsite, _constant(mc, ing), positions=pos)
        return interchange(
            fsite, _src_fsite(mc, ing), source_positions=pos, positions=pos
        )
    if mc.mode == "interpolate":

        def linear(f_base: torch.Tensor, f_src: torch.Tensor, alpha: float):
            return (1 - alpha) * f_base + alpha * f_src

        return interpolate(
            fsite,
            _src_fsite(mc, ing),
            linear,
            source_positions=pos,
            positions=pos,
            alpha=_ALPHA,
        )
    if mc.mode == "mask":
        return mask(
            fsite,
            _src_fsite(mc, ing),
            _gate(ing),
            source_positions=pos,
            positions=pos,
        )
    raise ValueError(f"unknown mode {mc.mode!r}")


def _oracle_g(mc: ModeCase, pc: ParityCase, ing: _Ingredients) -> Callable:
    """The offline feature-space transform matching :func:`_build_edit`.
    Constants are coerced to the features' device/dtype inside ``g`` — the
    new-stack side gets this from the ``ReadSource`` machinery; the hand-rolled
    oracle must do it itself (the tiny CPU families never notice, the GPU
    golden does)."""

    def like(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return v.to(device=f.device, dtype=f.dtype)

    if mc.mode == "replace":
        v = _constant(mc, ing)
        return _scattered(ing, lambda f: like(v, f).expand_as(f))
    if mc.mode == "steer":
        v = _constant(mc, ing)
        return _scattered(ing, lambda f: f + _STEER_FACTOR * like(v, f))
    if mc.mode == "noise":
        b = ing.inputs["input_ids"].shape[0]
        gen = torch.Generator().manual_seed(_NOISE_SEED)
        draw = torch.randn((b, ing.n_pos, ing.fwidth), generator=gen)
        return _scattered(ing, lambda f: f + _NOISE_SCALE * like(draw, f))
    if mc.mode == "interchange":
        if mc.component == "embeddings":
            v = _constant(mc, ing)
            return _scattered(ing, lambda f: like(v, f).expand_as(f))
        f_src = _oracle_src_features(mc, pc, ing)
        return _scattered(ing, lambda f: f_src)
    if mc.mode == "interpolate":
        f_src = _oracle_src_features(mc, pc, ing)
        return _scattered(ing, lambda f: (1 - _ALPHA) * f + _ALPHA * f_src)
    if mc.mode == "mask":
        f_src = _oracle_src_features(mc, pc, ing)
        on = _gate_on(ing)

        def hard_swap(f: torch.Tensor) -> torch.Tensor:
            out = f.clone()
            out[..., on] = f_src[..., on]
            return out

        return _scattered(ing, hard_swap)
    raise ValueError(f"unknown mode {mc.mode!r}")


# ----------------------------- head-site cases ------------------------------ #
_HEAD_DST, _HEAD_SRC = 1, 0  # head indices (kv-heads for "value")


def _head_module(pc: ParityCase, mc: ModeCase) -> tuple[Any, str]:
    """The raw module + hook kind carrying a head kind's flat activation
    (llama-family only — the head cases are restricted to it)."""
    attn = pc.oracle.hf_model.model.layers[mc.layer].self_attn
    if mc.head_kind == "value":
        return attn.v_proj, "out"
    return attn.o_proj, "in"  # attention_value: o_proj input


def _head_capture(pc: ParityCase, mc: ModeCase, head: int, inputs: Any) -> torch.Tensor:
    """Hook-captured per-head slice ``(b, seq, head_dim)`` — column
    ``[head*d:(head+1)*d]`` of the flat projection (honours a decoupled
    ``config.head_dim``, the #386 contract)."""
    module, kind = _head_module(pc, mc)
    flat = capture_component(pc.oracle, module, kind, inputs)
    return flat[:, :, head_slice(pc.oracle, head)]


def _realize_head(mc: ModeCase, pc: ParityCase, side: str) -> Realization:
    inputs = pc.inputs()
    last = int(inputs["input_ids"].shape[1] - 1)
    dst = FeaturizedSite(HeadSite(mc.head_kind, mc.layer, _HEAD_DST))
    if mc.mode == "collect":
        if side == "new":
            got = collect(dst, positions=[last]).collect(pc.st, inputs)
        else:
            got = _head_capture(pc, mc, _HEAD_DST, inputs)[:, [last]]
        return Realization(kind="collect", value=got)
    # interchange: transplant head _HEAD_SRC's activation into head _HEAD_DST
    clean = (
        pc.clean_logits(inputs)
        if side == "new"
        else next_token_logits(pc.oracle, inputs)
    )
    if side == "new":
        src = FeaturizedSite(HeadSite(mc.head_kind, mc.layer, _HEAD_SRC))
        edit = interchange(dst, src, source_positions=[last], positions=[last])
        return Realization("logits", pc.edited_logits(edit, inputs), clean)
    src_vals = _head_capture(pc, mc, _HEAD_SRC, inputs)[:, [last]]
    module, kind = _head_module(pc, mc)
    sl = head_slice(pc.oracle, _HEAD_DST)

    def edit_fn(x: torch.Tensor) -> None:
        x[:, [last], sl] = src_vals

    got = run_with_edits(pc.oracle, inputs, [(module, kind, edit_fn)])
    return Realization("logits", got, clean)


# ----------------------------- plan/staged cases ---------------------------- #
def _realize_plan(mc: ModeCase, pc: ParityCase, side: str) -> Realization:
    """The canonical cross-input interchange (source + base invokes + barrier)
    through ``run_plan`` — strictness ("plan") or auto ("staged"), one
    scheduler either way (EU2, #483)."""
    base, source = pc.pair(_BASE_TEXT, _SOURCE_TEXT)
    last = int(base["input_ids"].shape[1] - 1)
    resid = FeaturizedSite(Site(mc.component, mc.layer))
    if side == "new":
        clean = pc.clean_logits(base)
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        resid,
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(resid, positions=[last], input="source"),
                        ),
                        positions=[last],
                    ),
                ),
            ),
            save_logits=("base",),
        )
        # EU2 (#483): the single/staged lowering fork is gone — one scheduler
        # serves both case ids. The structural assertion replaces the old
        # forcing: the canonical interchange must schedule as ONE fused trace
        # (pass-minimality), so the "plan" case runs it under lowering="single"
        # strictness (which passes through the degenerate schedule) and the
        # "staged" case under the default auto — identical programs by pin.
        program = lower_staged(pc.st, plan)
        assert program.stages == ((("source", "base"),),), program.stages
        lowering = "auto" if mc.path == "staged" else "single"
        result = run_plan(pc.st, plan, lowering=lowering)
        return Realization("logits", result.logits["base"][:, -1, :], clean)
    clean = next_token_logits(pc.oracle, base)
    src_vals = pc.capture(mc.component, mc.layer, source)[:, [last]]
    patched = next_token_logits(
        pc.oracle, base, layer=mc.layer, positions=[last], patch_values=src_vals
    )
    return Realization("logits", patched, clean)


# ----------------------------- entry points --------------------------------- #
def _realize(mc: ModeCase, pc: ParityCase, side: str) -> Realization:
    assert pc.family == mc.family, f"{mc.case_id} realized on {pc.family}"
    if mc.path == "head":
        return _realize_head(mc, pc, side)
    if mc.path in ("plan", "staged"):
        return _realize_plan(mc, pc, side)
    ing = _ingredients(mc, pc)
    if mc.mode == "collect":
        if side == "new":
            got = _build_edit(mc, pc, ing).collect(pc.st, ing.inputs)
        else:
            sel = slice(None) if ing.positions is None else ing.positions
            f, _ = ing.feat.featurize(
                pc.capture(mc.component, mc.layer, ing.inputs)[:, sel]
            )
            got = f if ing.feature_ids is None else f[..., list(ing.feature_ids)]
        return Realization(kind="collect", value=got)
    if side == "new":
        clean = pc.clean_logits(ing.inputs)
        edited = pc.edited_logits(_build_edit(mc, pc, ing), ing.inputs)
        return Realization("logits", edited, clean)
    clean = next_token_logits(pc.oracle, ing.inputs)
    edited = pc.oracle_edit_logits(
        Site(mc.component, mc.layer),
        ing.feat,
        _oracle_g(mc, pc, ing),
        ing.positions,
        ing.inputs,
    )
    return Realization("logits", edited, clean)


def realize_new_stack(mc: ModeCase, pc: ParityCase) -> Realization:
    return _realize(mc, pc, "new")


def realize_oracle(mc: ModeCase, pc: ParityCase) -> Realization:
    return _realize(mc, pc, "oracle")
