"""The seven intervention modes as thin constructors over :class:`Edit` — ED2.

Pyvene shipped one dynamically-built intervention *subclass per mode* (the
``build_feature_*_intervention`` closure factories in
:mod:`causalab.neural.featurizer`). On the new stack the mode is just a choice
of feature-space ``g`` and read-sources over ED1's declarative
:class:`~causalab.neural.edit.Edit`, so each mode collapses to a constructor —
the design doc's mode table, one function per row:

======================  =====================  ===========================================
mode                    reads                  feature-space ``g``
======================  =====================  ===========================================
:func:`collect`         site                   identity → save
:func:`replace`         (shape only)           constant source vector
:func:`steer`           site                   ``f + factor·v``
:func:`interchange`     site + source-site     ``f_src`` (full or ``feature_ids`` swap)
:func:`interpolate`     site + source-site     ``fn(f_base=f, f_src=s, **params)``
:func:`noise`           site                   ``f + scale·randn(generator=seeded)``
:func:`mask`            site + source-site     ``(1−gate)·f + gate·f_src``, gate from θ
======================  =====================  ===========================================

The constructors add **no** runtime machinery: auxiliary values ride
:class:`~causalab.neural.edit.ReadSource` (ED1 owns device/dtype coercion and
the forward-order check), the featurize/inverse wrap and ``feature_ids``
subspace scatter are ST3's (:class:`~causalab.neural.featurized_site.
FeaturizedSite`), and the write itself is ST1's. The two pieces of state a mode
genuinely needs are explicit objects the caller can hold: :class:`SeededNoise`
(the advancing per-device RNG stream ``FeatureNoiseIntervention`` kept per
instance) and :class:`MaskGate` (the DBM gate parameter + temperature that
``FeatureMaskIntervention`` carried).

Semantics are pyvene-parity, pinned against the raw-hook oracle in
``tests/neural/test_modes.py``. One deliberate delta: :class:`MaskGate` needs a
temperature only where one is *used* (training-mode forward and
:meth:`MaskGate.sparsity_loss`); pyvene's ``FeatureMaskIntervention`` also
demanded it for the hard-threshold eval path that never reads it.

Scope
-----
Like :meth:`Edit.apply`, every constructor composes reads and one write within
a single already-open trace over one input — a *source-site* here is an
earlier-firing site on the **same** input, or a precomputed tensor (pyvene's
``source_representations`` pattern). Cross-*input* composition (base vs.
counterfactual via multiple ``invoke`` blocks or staged sequential traces) is
the plan compiler's job (PL1): :func:`interchange` / :func:`interpolate` can
*declare* it (``source_input`` names the plan input the source site is read
under), but only :func:`causalab.neural.plan.run_plan` can execute the
resulting :class:`Edit` (``Edit.apply`` refuses cross-input reads). Training
the :func:`mask` gate (optimizer loop, the differentiable loss slice,
temperature schedule, sharded-device placement) is ED3's, gated on the F6
grad contract.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.site import Positions, Site

__all__ = [
    "MaskGate",
    "SeededNoise",
    "collect",
    "interchange",
    "interpolate",
    "mask",
    "noise",
    "replace",
    "steer",
]


# --------------------------------------------------------------------------- #
#  Shared coercions                                                            #
# --------------------------------------------------------------------------- #
def _featurized(site: Site | FeaturizedSite) -> FeaturizedSite:
    """Accept a bare :class:`Site` (identity feature space) or a full
    :class:`FeaturizedSite` — every constructor takes either."""
    return site if isinstance(site, FeaturizedSite) else FeaturizedSite(site)


def _feature_width(fsite: FeaturizedSite) -> int | None:
    """How wide the feature vectors ``g`` sees are — the ``feature_ids`` gather
    when set, else the featurizer's ``n_features`` (``None`` when unknown, e.g.
    the identity default)."""
    if fsite.feature_ids is not None:
        return len(fsite.feature_ids)
    return fsite.featurizer.n_features


def _check_width(fsite: FeaturizedSite, value: Any, what: str) -> None:
    """Reject a feature-space tensor whose last dim can't broadcast against the
    width ``g`` will see — at construction, where the mismatch is legible,
    instead of as a scatter/matmul error mid-trace."""
    width = _feature_width(fsite)
    if width is None or not isinstance(value, torch.Tensor) or value.dim() == 0:
        return
    if value.shape[-1] not in (width, 1):
        raise ValueError(
            f"{what} has feature width {value.shape[-1]} but this site's "
            f"feature space is {width}-wide "
            f"({'feature_ids gather' if fsite.feature_ids is not None else 'n_features'})"
        )


def _source_read(
    fsite: FeaturizedSite,
    source: Site | FeaturizedSite | torch.Tensor,
    source_positions: Positions | None,
    what: str,
    source_input: str | None = None,
) -> ReadSource:
    """Normalize a mode's ``source`` into one :class:`ReadSource`: an
    earlier-firing site read in the same trace (``Site``/:class:`FeaturizedSite`),
    or a precomputed feature-space tensor (pyvene's ``source_representations``
    pattern — cross-input/cross-model values captured elsewhere).
    ``source_input`` names the *plan input* a site-backed source is read under
    (:attr:`ReadSource.input`) — the cross-input base-vs-counterfactual shape
    only the plan compiler can realize (``Edit.apply`` refuses it); a tensor
    source is input-independent, so ``ReadSource`` itself rejects the
    combination."""
    if isinstance(source, (Site, FeaturizedSite)):
        return ReadSource(
            _featurized(source), positions=source_positions, input=source_input
        )
    if source_positions is not None:
        raise ValueError(
            f"source_positions only applies to a site-backed source; the {what} "
            "source is a precomputed tensor — slice it before passing"
        )
    _check_width(fsite, source, f"{what} source")
    return ReadSource(source, input=source_input)


# --------------------------------------------------------------------------- #
#  Stateful helpers two modes need                                             #
# --------------------------------------------------------------------------- #
class SeededNoise:
    """The seeded, *advancing* Gaussian stream behind :func:`noise` —
    ``FeatureNoiseIntervention``'s reproducibility contract as a value the
    caller can hold.

    One generator per device, seeded once and then advanced across calls:
    successive batches draw independent noise (a fixed re-seed every call would
    hand identical-shape batches the *same* noise, making corruption repeat
    across batch boundaries and depend on batch size). A fresh instance with
    the same seed reproduces the same sequence — reproducible across runs,
    identical across grid cells that each construct their own — and
    :meth:`reset` restarts the stream in place (pyvene's ``reset_noise_rng``,
    used when one instance spans independent example groups, e.g. the length
    buckets of a causal-trace cell).
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        gen = self._generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device).manual_seed(self.seed)
            self._generators[key] = gen
        return gen

    def reset(self) -> None:
        """Drop the cached generators so the next call re-seeds from ``seed``."""
        self._generators.clear()

    def __call__(self, features: torch.Tensor, scale: Any) -> torch.Tensor:
        """``features + scale · randn(features.shape)`` from this stream —
        the noise is generated here (matching the features' shape, device and
        dtype), so callers pass a magnitude, not a pre-shaped tensor."""
        draw = torch.randn(
            features.shape,
            generator=self._generator(features.device),
            device=features.device,
            dtype=features.dtype,
        )
        return features + scale * draw


class MaskGate(torch.nn.Module):
    """The DBM (differentiable binary masking) gate behind :func:`mask` —
    ``FeatureMaskIntervention``'s learnable state as a plain module.

    Per-feature logits ``mask`` (or one tied scalar) gate each feature between
    base and source: soft ``sigmoid(mask / temperature)`` while training, hard
    ``sigmoid(mask) > 0.5`` in eval. The temperature is a settable attribute —
    annealing it (and the outer optimization loop that trains ``mask``) is
    ED3's, not this module's. Gate math runs in the mask's dtype (float32 by
    default) and the parameter is aligned to the features' device per call via
    a non-mutating ``.to`` — explicit placement of the parameter itself (the
    sharded-model case) is likewise ED3's.
    """

    def __init__(self, n_features: int | None = None, *, tie: bool = False) -> None:
        super().__init__()
        if tie:
            self.mask = torch.nn.Parameter(torch.zeros(1))
        else:
            if n_features is None:
                raise ValueError("per-feature gating needs n_features (or tie=True)")
            self.mask = torch.nn.Parameter(torch.zeros(n_features))
        self.temperature: torch.Tensor | None = None

    def set_temperature(self, temperature: float | torch.Tensor) -> None:
        self.temperature = torch.as_tensor(
            temperature, dtype=self.mask.dtype, device=self.mask.device
        )

    def gate(self, device: torch.device | None = None) -> torch.Tensor:
        """The current per-feature gate — soft (needs a temperature) while
        training, hard 0/1 in eval."""
        m = self.mask if device is None else self.mask.to(device)
        if self.training:
            if self.temperature is None:
                raise ValueError(
                    "MaskGate has no temperature; call set_temperature() before "
                    "a training-mode forward"
                )
            return torch.sigmoid(m / self.temperature.to(m.device))
        return (torch.sigmoid(m) > 0.5).to(m.dtype)

    def forward(self, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
        gate = self.gate(f_base.device)
        blended = (1.0 - gate) * f_base.to(gate.dtype) + gate * f_src.to(gate.dtype)
        return blended.to(f_base.dtype)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of the *soft* gate — the DBM sparsity regularizer (always
        the training-mode gate; a hard gate's L1 is not differentiable)."""
        if self.temperature is None:
            raise ValueError(
                "sparsity_loss needs a temperature; call set_temperature()"
            )
        return torch.norm(torch.sigmoid(self.mask / self.temperature), p=1)


# --------------------------------------------------------------------------- #
#  The seven constructors                                                      #
# --------------------------------------------------------------------------- #
def collect(site: Site | FeaturizedSite, *, positions: Positions | None = None) -> Edit:
    """Read-only: featurized read of ``site``, no write (pyvene's
    ``CollectIntervention``). Use :meth:`Edit.collect` — one forward,
    early-stopped, CPU result."""
    return Edit(_featurized(site), positions=positions)


def replace(
    site: Site | FeaturizedSite,
    value: torch.Tensor,
    *,
    scale: float = 1.0,
    positions: Positions | None = None,
) -> Edit:
    """Overwrite the (selected) features with a constant feature-space
    ``value`` (pyvene's ``FeatureReplaceIntervention``): base still contributes
    the reconstruction error and, under ``feature_ids``, the untouched columns
    — a zero ``value`` ablates the feature contribution only. ``scale``
    multiplies the vector (``scale·value`` is written — the
    :class:`~causalab.neural.specs.EditSpec` replace contract, mirroring
    :func:`steer`'s ``factor``). For a site-sourced
    replacement use :func:`interchange`."""
    fsite = _featurized(site)
    _check_width(fsite, value, "replace value")

    def g(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Expand to the base features' shape: a bare broadcast vector would
        # otherwise collapse the rank and break the error-term rebuild of
        # lossy (split) featurizers.
        return (scale * v).expand_as(f)

    return Edit(fsite, g=g, read_sources=(ReadSource(value),), positions=positions)


def steer(
    site: Site | FeaturizedSite,
    vector: torch.Tensor,
    *,
    factor: float = 1.0,
    positions: Positions | None = None,
) -> Edit:
    """Add a direction: ``f + factor·vector`` in feature space (pyvene's
    ``FeatureSteeringIntervention``; ``factor`` folds in nnterp's ``steer``
    scaling so callers keep unit vectors). The vector is coerced to the
    features' device/dtype by the :class:`ReadSource` machinery."""
    fsite = _featurized(site)
    _check_width(fsite, vector, "steering vector")

    def g(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return f + factor * v

    return Edit(fsite, g=g, read_sources=(ReadSource(vector),), positions=positions)


def interchange(
    site: Site | FeaturizedSite,
    source: Site | FeaturizedSite | torch.Tensor,
    *,
    source_positions: Positions | None = None,
    source_input: str | None = None,
    positions: Positions | None = None,
) -> Edit:
    """Swap the (selected) base features for the source's (pyvene's
    ``FeatureInterchangeIntervention`` / ``_do_intervention_by_swap``): under
    ``feature_ids`` only those columns are replaced — the subspace swap — and
    base always keeps its reconstruction error. ``source`` is an
    earlier-firing site read in the same trace over the same input (the
    single-input scope), or a precomputed feature tensor captured elsewhere.
    ``source_input`` names the plan input the source site is read under
    (``ReadSource(..., input=...)``) — the base-vs-counterfactual interchange
    across inputs, runnable only through a Plan
    (:mod:`causalab.neural.plan`; ``Edit.apply`` refuses it)."""
    fsite = _featurized(site)
    src = _source_read(fsite, source, source_positions, "interchange", source_input)

    def g(f: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
        return f_src

    return Edit(fsite, g=g, read_sources=(src,), positions=positions)


def interpolate(
    site: Site | FeaturizedSite,
    source: Site | FeaturizedSite | torch.Tensor,
    fn: Callable[..., torch.Tensor],
    *,
    source_positions: Positions | None = None,
    source_input: str | None = None,
    positions: Positions | None = None,
    **params: Any,
) -> Edit:
    """Patch ``fn(f_base=..., f_src=..., **params)`` back into the model
    (pyvene's ``FeatureInterpolateIntervention``, keeping its keyword
    contract). ``source_input`` makes a site-backed source cross-input,
    exactly as in :func:`interchange` (plan-only):

    .. code-block:: python

        def linear(f_base, f_src, alpha):
            return (1 - alpha) * f_base + alpha * f_src

        edit = interpolate(fsite, src_fsite, linear, alpha=0.5)
    """
    fsite = _featurized(site)
    src = _source_read(fsite, source, source_positions, "interpolate", source_input)

    def g(f: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
        return fn(f_base=f, f_src=f_src, **params)

    return Edit(fsite, g=g, read_sources=(src,), positions=positions)


def noise(
    site: Site | FeaturizedSite,
    scale: float | torch.Tensor,
    *,
    seed: int | SeededNoise = 0,
    positions: Positions | None = None,
) -> Edit:
    """Additive seeded Gaussian corruption, ``f + scale·randn`` (pyvene's
    ``FeatureNoiseIntervention`` — ROME-style causal tracing's corruption
    entry point). ``scale`` is a magnitude — a scalar or a tensor
    broadcastable over the features (e.g. a per-dimension ``3·sigma``); the
    randomness is drawn inside the edit. Pass a :class:`SeededNoise` instance
    as ``seed`` to hold the stream yourself (for :meth:`SeededNoise.reset`
    across example groups); an ``int`` constructs a private one."""
    fsite = _featurized(site)
    _check_width(fsite, scale, "noise scale")
    state = seed if isinstance(seed, SeededNoise) else SeededNoise(seed)
    return Edit(fsite, g=state, read_sources=(ReadSource(scale),), positions=positions)


def mask(
    site: Site | FeaturizedSite,
    source: Site | FeaturizedSite | torch.Tensor,
    gate: MaskGate,
    *,
    source_positions: Positions | None = None,
    positions: Positions | None = None,
) -> Edit:
    """Gate each feature between base and source through a learnable
    :class:`MaskGate`: ``(1−gate)·f_base + gate·f_src`` (pyvene's
    ``FeatureMaskIntervention`` — DBM). The gate module *is* the edit's ``g``,
    so its parameter is introspectable on the :class:`Edit` value. Training it
    — optimizer loop, temperature schedule, loss slice, sharded placement — is
    ED3's; this constructor only wires the forward."""
    fsite = _featurized(site)
    width = _feature_width(fsite)
    if width is not None and gate.mask.numel() not in (width, 1):
        raise ValueError(
            f"MaskGate has {gate.mask.numel()} gate(s) but this site's feature "
            f"space is {width}-wide (pass tie=True for one shared gate)"
        )
    src = _source_read(fsite, source, source_positions, "mask")
    return Edit(fsite, g=gate, read_sources=(src,), positions=positions)
