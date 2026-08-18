"""Featurizer kinds (spec §2.5): ``featurize(x) → (f, err)``,
``inverse(f, err) → x̂``, per kind, plus composition.

The error-term contract is the load-bearing rule: ``err`` and unselected
dims always come from the **pre-edit** value at the address, so a zero
write ablates only the feature contribution and a ``dims`` write is a
subspace swap. One deliberate interpretation (surfaced in the PR): the
spec's kind table writes ``(Qᵀx, 0)`` for ``subspace``, but the contract
paragraph requires the orthogonal complement to survive a swap — so here
``err = x − QQᵀx`` (identically 0 when ``k = d``), matching the oracle's
lossy-split behavior and DAS semantics.

``gate`` is soft during training (``σ(θ/T) ⊙ x``, temperature annealed by
the train loop) and **hard** in eval (``θ > 0``) — the parity oracle's
mask mode pins the hard-eval split.

Everything here is per-position math on ``(..., d)`` tensors; widths come
from the resolved site, never from the document.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import torch

from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import FeaturizerSpec

__all__ = ["FeaturizerStack", "Stage", "build_stack"]


class Stage(torch.nn.Module):
    """One featurizer stage. Subclasses implement ``featurize`` /
    ``inverse``; parameters registered here are what ``train.params``
    optimizes."""

    kind: str = "identity"

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x, None

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f

    def slot_params(self) -> dict[str, torch.Tensor]:
        """The auto-declared slots (§2.5), for saving and identity checks."""
        return {}


class Identity(Stage):
    kind = "identity"


class Subspace(Stage):
    """An orthonormal ``(d, k)`` map ``Q``; features are the coordinates in
    its column space, ``err`` the complement (module docstring)."""

    kind = "subspace"

    def __init__(
        self, width: int, k: int, parametrization: str, *, seed: int = 0
    ) -> None:
        super().__init__()
        self.k = k
        generator = torch.Generator().manual_seed(seed)
        init = torch.linalg.qr(torch.randn(width, k, generator=generator))[0]
        self.weight = torch.nn.Parameter(init)
        orthogonal_map = {
            "cayley": "cayley",
            "matrix_exp": "matrix_exp",
            # a direct Stiefel point via householder products — the map torch
            # provides for rectangular orthogonal parametrizations
            "stiefel": "householder",
        }[parametrization]
        torch.nn.utils.parametrizations.orthogonal(
            self, "weight", orthogonal_map=orthogonal_map
        )

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        q: torch.Tensor = self.weight
        f = x.to(q.dtype) @ q
        return f, x - f @ q.T

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        q: torch.Tensor = self.weight
        x = f @ q.T
        return x if err is None else x + err

    def slot_params(self) -> dict[str, torch.Tensor]:
        return {"weight": self.weight}


class LoadedLinear(Stage):
    """A fixed ``(d, k)`` map loaded from an artifact (``pca``, or an
    applied ``subspace`` fit): same math as :class:`Subspace`, no
    parametrization, never trainable."""

    weight: torch.Tensor

    def __init__(self, kind: str, weight: torch.Tensor) -> None:
        super().__init__()
        self.kind = kind
        self.register_buffer("weight", weight)

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.weight
        f = x.to(weight.dtype) @ weight
        return f, x - f @ weight.T

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        x = f @ self.weight.T
        return x if err is None else x + err


class Standardize(Stage):
    kind = "standardize"

    mu: torch.Tensor
    sigma: torch.Tensor

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (x - self.mu) / self.sigma, None

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f * self.sigma + self.mu


class Sae(Stage):
    """A loaded sparse autoencoder: ``(enc(x), x − dec(enc(x)))``."""

    kind = "sae"

    enc: torch.Tensor
    dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor

    def __init__(
        self,
        enc: torch.Tensor,
        dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("enc", enc)
        self.register_buffer("dec", dec)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("b_dec", b_dec)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.enc + self.b_enc)

    def _decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.dec + self.b_dec

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        f = self._encode(x)
        return f, x - self._decode(f)

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        x = self._decode(f)
        return x if err is None else x + err


class Gate(Stage):
    """The DBM gate: soft ``σ(θ/T) ⊙ x`` in training, hard ``θ > 0`` in
    eval. ``temperature`` is the anneal target ``<name>.theta.temperature``."""

    kind = "gate"

    def __init__(self, width: int) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(width))
        self.temperature: float = 1.0
        self.hard_eval: bool = True

    def _mask(self) -> torch.Tensor:
        if self.training or not self.hard_eval:
            return torch.sigmoid(self.theta / self.temperature)
        return (self.theta > 0).to(self.theta.dtype)

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask = self._mask().to(x.dtype)
        return mask * x, (1.0 - mask) * x

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f if err is None else f + err

    def slot_params(self) -> dict[str, torch.Tensor]:
        return {"theta": self.theta}


@dataclasses.dataclass
class FeaturizerStack:
    """A left-to-right composition of stages with a per-stage ``err`` list
    (§2.5). ``names`` aligns with ``stages`` for train-param addressing."""

    names: tuple[str, ...]
    stages: tuple[Stage, ...]

    def featurize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        errs: list[torch.Tensor | None] = []
        for stage in self.stages:
            x, err = stage.featurize(x)
            errs.append(err)
        return x, errs

    def inverse(
        self, f: torch.Tensor, errs: Sequence[torch.Tensor | None]
    ) -> torch.Tensor:
        for stage, err in zip(reversed(self.stages), reversed(list(errs))):
            f = stage.inverse(f, err)
        return f

    @property
    def is_identity(self) -> bool:
        return all(isinstance(s, Identity) for s in self.stages)


def stage_output_width(spec: FeaturizerSpec, input_width: int) -> int | None:
    """The feature width a stage emits, given its input width — the chain
    rule of §2.5's composition: subspace/pca project to ``k``, the
    width-preserving kinds pass ``input_width`` through, and a loaded SAE's
    dictionary size is unknowable from the spec alone (``None``)."""
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    if kind in ("subspace", "pca"):
        return spec.k if isinstance(spec.k, int) else None
    if kind == "sae":
        return None  # the dictionary size lives in the bundle, not the spec
    return input_width


def _stage_width(stage: Stage) -> int | None:
    """The input width a built stage was sized for (for cache-reuse checks)."""
    params = stage.slot_params()
    if isinstance(stage, (Subspace, LoadedLinear)):
        weight = stage.weight if isinstance(stage, LoadedLinear) else params["weight"]
        return int(weight.shape[0])
    if isinstance(stage, Gate):
        return int(stage.theta.shape[0])
    if isinstance(stage, Standardize):
        return int(stage.mu.shape[0])
    return None


def build_stack(
    ref: Any,
    specs: dict[str, FeaturizerSpec],
    *,
    width: int,
    load_tensors: Any,
    stage_cache: dict[str, Stage],
) -> FeaturizerStack:
    """Build (or reuse from ``stage_cache``) the stack a read/edit
    references. ``width`` is the SITE width; each later stage in a
    composition is sized to the *previous stage's output* (the §2.5 chain —
    a gate after a k=3 rotation is a 3-wide gate). ``load_tensors`` supplies
    loaded bundles; caching by name keeps one stage instance per declared
    featurizer, so training one featurizer updates every use site — a name
    reused at a different chain width is a contradiction and refuses."""
    if ref is None:
        return FeaturizerStack(names=(), stages=(Identity(),))
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    stages: list[Stage] = []
    running: int | None = width
    for name in chain:
        spec = specs[name]
        if name in stage_cache:
            stage = stage_cache[name]
            built_for = _stage_width(stage)
            if built_for is not None and running is not None and built_for != running:
                raise ProtocolError(
                    "P2",
                    f"featurizer {name!r} is used at width {running} here but was "
                    f"built for width {built_for} — one featurizer, one width",
                )
        else:
            if running is None:
                raise ProtocolError(
                    "P2",
                    f"cannot size featurizer {name!r}: the preceding stage's "
                    "output width is not derivable from its spec",
                )
            stage = _build_stage(name, spec, width=running, load_tensors=load_tensors)
            # inference documents get eval semantics (a gate's hard split);
            # the train loop flips modes around its steps explicitly
            stage.eval()
            stage_cache[name] = stage
        stages.append(stage)
        if isinstance(stage, Sae):
            running = int(stage.enc.shape[1])
        elif running is not None:
            running = stage_output_width(spec, running)
    return FeaturizerStack(names=chain, stages=tuple(stages))


def _build_stage(
    name: str, spec: FeaturizerSpec, *, width: int, load_tensors: Any
) -> Stage:
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    if isinstance(spec.file_path, str):
        tensors = load_tensors(spec.file_path)
        if kind in ("subspace", "pca"):
            return LoadedLinear(kind, tensors["weight"])
        if kind == "standardize":
            return Standardize(tensors["mu"], tensors["sigma"])
        if kind == "sae":
            return Sae(
                tensors["enc"], tensors["dec"], tensors["b_enc"], tensors["b_dec"]
            )
        raise ProtocolError(
            "P2", f"featurizer kind {kind!r} cannot be loaded from a file"
        )
    if kind == "identity":
        return Identity()
    if kind == "subspace":
        k = spec.k if isinstance(spec.k, int) else None
        parametrization = (
            spec.parametrization if isinstance(spec.parametrization, str) else "cayley"
        )
        if k is None:
            raise ProtocolError("P2", f"subspace featurizer {name!r} needs k")
        return Subspace(width, k, parametrization)
    if kind == "gate":
        return Gate(width)
    raise ProtocolError(
        "P2",
        f"featurizer {name!r} of kind {kind!r} needs a file_path — this backend "
        "does not fit it from data at run start",
    )
