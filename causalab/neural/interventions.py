"""
interventions.py
================
What to *do* at an intervention site, expressed in a featurizer's feature space.

A :class:`Featurizer` says how to read and write a feature space; this module
says what to write. Each mode is a small ``torch.nn.Module`` taking the base
activation (and, where the mode needs one, a source) and returning the
replacement activation:

    new_activation = inverse_featurize(f(featurize(base), source), base_error)

Every mode preserves the base's *reconstruction error* — the component of the
activation outside the feature space — so intervening in a rank-k subspace
leaves the orthogonal complement untouched.

These replace the seven ``build_feature_*_intervention`` factories that
synthesized pyvene ``TrainableIntervention`` subclasses at runtime. Those existed
because pyvene instantiated intervention *classes* itself and had no way to pass
a featurizer in, so the featurizer had to be captured in a closure over a
dynamically created class — untypeable, and the reason the old module carried a
"type crime" disclaimer. Here a mode is an ordinary object the caller holds.

``nn.Module`` (rather than a plain function) because :class:`MaskIntervention`
owns learnable parameters: the DBM training loop collects them with
``.parameters()`` exactly as it would for any other torch model.
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Callable

import torch
from torch import Tensor

__all__ = [
    "FeatureIntervention",
    "InterchangeIntervention",
    "CollectIntervention",
    "SteeringIntervention",
    "ReplaceIntervention",
    "NoiseIntervention",
    "InterpolationIntervention",
    "MaskIntervention",
    "build_intervention",
    "INTERVENTION_MODES",
]


def _select(f_base: Tensor, f_src: Tensor, feature_indices: list[int] | None) -> Tensor:
    """``f_src`` where ``feature_indices`` selects, ``f_base`` elsewhere.

    ``None`` means no restriction and swaps the whole feature vector. An *empty*
    list is the opposite — it selects nothing, so the base is returned unchanged.
    The two are not interchangeable: a tied mask uses ``[]`` to say "this unit is
    fully dropped", and reading it as "no restriction" would swap everything
    instead of nothing.

    This is the plain-torch equivalent of pyvene's ``subspaces`` argument and its
    ``_do_intervention_by_swap`` dispatch.
    """
    if feature_indices is None:
        return f_src
    out = f_base.clone()
    out[..., feature_indices] = f_src[..., feature_indices].to(out.dtype)
    return out


class FeatureIntervention(torch.nn.Module):
    """Base: hold a featurizer pair, apply a mode between them.

    Subclasses implement :meth:`combine`, which sees the *featurized* base (and
    source) and returns featurized output. The featurize / inverse-featurize
    sandwich, error preservation, and dtype restoration are handled here so no
    mode can forget them.
    """

    #: Whether this mode consumes a ``source`` value.
    needs_source: bool = True

    #: Whether the source still means something at a token the prompt never
    #: contained. A steering direction or a fixed replacement vector does — it is
    #: a point in feature space, not a reading of anything. A source gathered
    #: from another run *at particular positions* does not: there is no such
    #: reading for a token that did not exist when it was collected. Only the
    #: former can be carried into a generation step (see
    #: :func:`~causalab.neural.activations.engine.generate_with_interventions`).
    source_is_positionless: bool = False

    def __init__(self, featurizer: Any) -> None:
        super().__init__()
        self._featurizer = featurizer
        # Register the featurizer's two modules as children. `Featurizer` is a
        # plain container, not an `nn.Module`, so binding it alone registers
        # nothing and `self.parameters()` would miss a *learned* feature space —
        # DAS's rotation above all, whose weights the optimizer must see. Both
        # usually wrap the same underlying layer; `parameters()` de-duplicates.
        self._featurize_module = featurizer.featurizer
        self._inverse_module = featurizer.inverse_featurizer

    def _align_to(self, activation: Tensor) -> None:
        """Put the featurizer on the activation's device.

        A featurizer is built independently of the model — a rotation learned by
        DAS, a PCA fitted offline — so nothing has yet placed it where the
        activations live, and on a sharded model the answer differs per site.
        Doing it here means a featurizer does not have to device-align itself to
        be usable, which is not something a custom one would think to do.
        Probing one tensor is enough; ``.to`` moves the rest.
        """
        for tensor in chain(self.parameters(), self.buffers()):
            if tensor.device != activation.device:
                self.to(activation.device)
            return

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(
        self,
        base: Tensor,
        source: Tensor | None = None,
        feature_indices: list[int] | None = None,
    ) -> Tensor:
        self._align_to(base)
        f_base, base_error = self._featurizer.featurize(base)
        f_src = self._featurize_source(source, f_base)
        f_out = self.combine(f_base, f_src, feature_indices)
        return self._featurizer.inverse_featurize(f_out, base_error).to(base.dtype)

    def _featurize_source(self, source: Tensor | None, f_base: Tensor) -> Tensor | None:
        """Featurize a source activation; modes whose source is *already* a
        feature-space vector (steering, replacement, noise scale) override this."""
        if source is None:
            return None
        return self._featurizer.featurize(source)[0]


class InterchangeIntervention(FeatureIntervention):
    """Write the source's features into the base — the DAS / activation-patching op."""

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:
        assert f_src is not None, "interchange requires a source activation"
        return _select(f_base, f_src, feature_indices)


class CollectIntervention(FeatureIntervention):
    """Read the base's features, changing nothing.

    ``forward`` is overridden rather than ``combine`` because collection returns
    *features*, not a replacement activation — there is nothing to write back.
    """

    needs_source = False

    def forward(
        self,
        base: Tensor,
        source: Tensor | None = None,
        feature_indices: list[int] | None = None,
    ) -> Tensor:
        del source
        self._align_to(base)
        f_base, _ = self._featurizer.featurize(base)
        if feature_indices is not None:
            return f_base[..., feature_indices]
        return f_base

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:  # pragma: no cover - unreachable, forward is overridden
        return f_base


class SteeringIntervention(FeatureIntervention):
    """Add a steering vector to the base's features.

    The source is a vector *already in feature space* (a steering direction, not
    another run's activation), so it is not featurized on the way in.
    """

    source_is_positionless = True

    def _featurize_source(self, source: Tensor | None, f_base: Tensor) -> Tensor | None:
        return source

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:
        assert f_src is not None, "steering requires a steering vector"
        steered = f_base + f_src.to(f_base.dtype)
        return _select(f_base, steered, feature_indices)


class ReplaceIntervention(FeatureIntervention):
    """Overwrite the base's features with a given vector.

    Unlike steering this discards the base features entirely; the reconstruction
    error is still preserved, so replacing with zeros ablates the feature-space
    contribution while leaving the orthogonal component intact.
    """

    source_is_positionless = True

    def _featurize_source(self, source: Tensor | None, f_base: Tensor) -> Tensor | None:
        return source

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:
        assert f_src is not None, "replacement requires a replacement vector"
        return _select(f_base, f_src.to(f_base.dtype), feature_indices)


class NoiseIntervention(FeatureIntervention):
    """Add seeded Gaussian noise to the base's features — ROME-style corruption.

    The source carries the noise *scale* (a scalar, or a tensor broadcastable to
    the feature shape such as ``3 * sigma``); the randomness is drawn here, so
    callers pass a magnitude rather than a pre-shaped noise tensor.

    One generator per device, seeded once and then *advanced* across calls, so
    consecutive batches get independent draws. Re-seeding every call would hand
    identically-shaped batches the same noise, making corruption depend on
    ``batch_size``. A fresh instance with the same seed reproduces the sequence,
    which is what makes a sweep's grid cells comparable.
    """

    source_is_positionless = True

    def __init__(self, featurizer: Any, seed: int = 0) -> None:
        super().__init__(featurizer)
        self._seed = seed
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(self._seed)
            self._generators[key] = generator
        return generator

    def reset_noise_rng(self) -> None:
        """Restart the corruption stream at the seed.

        Not needed by anything in causalab: a caller that wants a fresh stream
        builds a fresh intervention, which is what every runner does — each
        causal-tracing grid cell is its own ``run_steering_interventions`` call
        and so replays the same draws, which is what makes cells comparable.
        Kept for a caller driving an intervention directly across several runs.
        """
        self._generators.clear()

    def _featurize_source(self, source: Tensor | None, f_base: Tensor) -> Tensor | None:
        return source

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:
        assert f_src is not None, "noise corruption requires a scale"
        noise = torch.randn(
            f_base.shape,
            generator=self._generator(f_base.device),
            device=f_base.device,
            dtype=f_base.dtype,
        )
        noised = f_base + f_src.to(f_base.dtype) * noise
        return _select(f_base, noised, feature_indices)


class InterpolationIntervention(FeatureIntervention):
    """Apply an arbitrary ``fn(f_base, f_src, **params)`` in feature space.

    Canonically ``(1 - alpha) * f_base + alpha * f_src``: at ``alpha=1`` this is
    interchange, at ``alpha=0`` the identity.
    """

    def __init__(self, featurizer: Any) -> None:
        super().__init__(featurizer)
        self._fn: Callable[..., Tensor] | None = None
        self._params: dict[str, Any] = {}

    def set_interpolation(self, fn: Callable[..., Tensor], **params: Any) -> None:
        self._fn = fn
        self._params = params

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:
        if self._fn is None:
            raise ValueError(
                "Interpolation function not set. Call set_interpolation(fn, **params) "
                "before running the intervention."
            )
        assert f_src is not None, "interpolation requires a source activation"
        mixed = self._fn(f_base=f_base, f_src=f_src, **self._params)
        return _select(f_base, mixed, feature_indices)


class MaskIntervention(FeatureIntervention):
    """Differentiable binary masking between base and source features (DBM).

    Learns one gate per feature (or a single tied gate). Training uses a
    temperature-annealed sigmoid so the gate is differentiable; evaluation
    thresholds it at 0.5 so the learned mask is genuinely binary.
    """

    def __init__(
        self, featurizer: Any, n_features: int, tie_masks: bool = False
    ) -> None:
        super().__init__(featurizer)
        self._tie_masks = tie_masks
        self.mask = torch.nn.Parameter(
            torch.zeros(1 if tie_masks else n_features), requires_grad=True
        )
        self.temperature: Tensor | None = None

    def get_temperature(self) -> Tensor:
        if self.temperature is None:
            raise ValueError("Temperature has not been set.")
        return self.temperature

    def set_temperature(self, temp: float | Tensor) -> None:
        self.temperature = torch.as_tensor(temp, dtype=self.mask.dtype).to(
            self.mask.device
        )

    def get_sparsity_loss(self) -> Tensor:
        if self.temperature is None:
            raise ValueError("Temperature has not been set.")
        return torch.norm(torch.sigmoid(self.mask / self.temperature), p=1)

    def combine(
        self, f_base: Tensor, f_src: Tensor | None, feature_indices: list[int] | None
    ) -> Tensor:
        if self.temperature is None:
            raise ValueError("Cannot run a mask intervention without a temperature.")
        assert f_src is not None, "masking requires a source activation"
        mask = self.mask.to(f_base.device)
        temperature = self.temperature.to(f_base.device)
        base = f_base.to(mask.dtype)
        source = f_src.to(mask.dtype)
        gate = (
            torch.sigmoid(mask / temperature)
            if self.training
            else (torch.sigmoid(mask) > 0.5).to(mask.dtype)
        )
        gated = (1.0 - gate) * base + gate * source
        return _select(base, gated, feature_indices)


INTERVENTION_MODES = (
    "interchange",
    "collect",
    "mask",
    "add",
    "replace",
    "interpolation",
    "noise",
)


def build_intervention(
    featurizer: Any, mode: str, **kwargs: Any
) -> FeatureIntervention:
    """Construct the intervention for ``mode`` over ``featurizer``'s feature space.

    ``mode`` uses causalab's existing vocabulary, in which ``"add"`` is steering.
    ``noise`` accepts ``seed``; ``mask`` accepts ``tie_masks`` and requires the
    featurizer to declare ``n_features``.
    """
    if mode == "interchange":
        return InterchangeIntervention(featurizer)
    if mode == "collect":
        return CollectIntervention(featurizer)
    if mode == "add":
        return SteeringIntervention(featurizer)
    if mode == "replace":
        return ReplaceIntervention(featurizer)
    if mode == "interpolation":
        return InterpolationIntervention(featurizer)
    if mode == "noise":
        return NoiseIntervention(featurizer, seed=kwargs.get("seed", 0))
    if mode == "mask":
        if featurizer.n_features is None:
            raise ValueError(
                "`n_features` must be set on the Featurizer to build a mask "
                "intervention — the mask has one gate per feature."
            )
        return MaskIntervention(
            featurizer,
            n_features=featurizer.n_features,
            tie_masks=kwargs.get("tie_masks", featurizer.tie_masks),
        )
    raise ValueError(
        f"Unknown intervention mode {mode!r}. Known modes: {list(INTERVENTION_MODES)}"
    )
