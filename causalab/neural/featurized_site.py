"""The ``(featurize, inverse)`` wrap around Site reads and writes — ST3.

Pyvene applied a featurizer by generating one intervention *subclass per mode*
(the ``build_feature_*_intervention`` closure factories in
:mod:`causalab.neural.featurizer`). On nnsight the mode is just trace code, so
the whole axis collapses to one wrap: featurize what a :class:`~causalab.neural.
site.Site` reads, operate in feature space, inverse-featurize with the base
error, write back. :class:`FeaturizedSite` is that wrap — the design doc's

.. code-block:: text

    f   = featurize(read(site))      # read may be omitted (pure write)
    f'  = g(f, *aux)                 # aux = values from other sites / inputs
    write(site, inverse(f'))         # write may be omitted (pure collect)

with the existing :class:`~causalab.neural.featurizer.Featurizer` stack reused
unchanged (▣): the identity default, :class:`~causalab.neural.featurizer.
ComposedFeaturizer` per-stage error threading, and subspace indexing via
``feature_ids`` (the pyvene ``subspaces`` analog — causalab passes a static
per-site index list, ``pipeline.intervenable_generate``'s ``feature_indices``).

Notes
-----
* **Error-term contract.** Every write path featurizes the *base* activation
  first — the reconstruction error (the component outside the feature space)
  and any untouched feature columns always come from base; only the selected
  features are replaced or transformed. This is exactly the pyvene semantics
  (``FeatureReplaceIntervention`` / ``_do_intervention_by_swap``): a zero-vector
  :meth:`FeaturizedSite.write` ablates the feature contribution while leaving
  the orthogonal component intact, and interchange is ``base_fsite.write(model,
  src_fsite.read(model, ...), ...)``.
* **In-trace module calls.** ``featurize``/``inverse_featurize`` are ordinary
  module calls on trace proxies (`NNsight_overview.md` §3.9) — no subclassing,
  no closure factory. Concrete featurizer modules own their device/dtype
  hygiene (e.g. ``SubspaceFeaturizerModule`` moves its weight to the input's
  device); the final reconstruction is moved to the site's device and dtype by
  :meth:`Site.write`, so a featurizer computing in a wider dtype round-trips.
* **Gradients** flow through the wrap by construction (plain tensor ops), but
  the trainable-edit contract (mask/DAS training) is pinned by F6 and built by
  ED1/ED3 — not here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from causalab.neural.featurizer import Featurizer
from causalab.neural.site import Positions, WritableSite, collect_ordered, forward_key

__all__ = ["FeaturizedSite"]


@dataclasses.dataclass(frozen=True)
class FeaturizedSite:
    """A :class:`Site` (or any :class:`~causalab.neural.site.WritableSite`,
    e.g. a per-head :class:`~causalab.neural.head_view.HeadSite`) read/written
    through a feature space.

    Parameters
    ----------
    site :
        Where to read/write (component + layer, ST1; or a per-head location,
        ST4).
    featurizer :
        The feature space, as the existing causalab :class:`Featurizer`
        (identity by default — the wrap then reduces to plain site access).
    feature_ids :
        Optional static indices into the feature dimension (subspace
        indexing). Reads gather these columns; writes scatter into them,
        leaving the other feature columns (and the reconstruction error) from
        base. ``None`` addresses the full feature space.
    """

    site: WritableSite
    featurizer: Featurizer = dataclasses.field(default_factory=Featurizer)
    feature_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.feature_ids is None:
            return
        ids = tuple(int(i) for i in self.feature_ids)
        object.__setattr__(self, "feature_ids", ids)  # frozen: normalize in place
        if not ids:
            raise ValueError("feature_ids must be non-empty (use None for all)")
        if any(i < 0 for i in ids):
            raise ValueError(f"feature_ids must be non-negative, got {ids}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"feature_ids must be unique, got {ids}")
        n = self.featurizer.n_features
        if n is not None and max(ids) >= n:
            raise ValueError(
                f"feature_ids {ids} out of range for a {n}-feature featurizer"
            )

    # -- in-trace accessors (call inside `with model.trace(...)`) ---------------- #
    def read(self, model: Any, positions: Positions | None = None) -> Any:
        """In-trace featurized read: ``featurize(site.read(...))``, gathered to
        ``feature_ids`` when set. The reconstruction error is discarded — a
        read-only tap never needs it (write paths re-featurize base themselves).
        Call inside ``with model.trace(...):``."""
        features, _ = self.featurizer.featurize(self.site.read(model, positions))
        if self.feature_ids is None:
            return features
        return features[..., list(self.feature_ids)]

    def edit(
        self,
        model: Any,
        g: Callable[[Any], Any],
        positions: Positions | None = None,
    ) -> None:
        """In-trace read-modify-write in feature space: featurize base, apply
        ``g`` to the (selected) features, inverse-featurize with the base error,
        write back. ``g`` receives the ``feature_ids`` gather when set and only
        those columns are replaced. Feature-space code runs where the activation
        lives, so any auxiliary tensor ``g`` closes over (a steering vector, an
        interpolation target) must already sit on the features' device — the
        mode constructors (ED1) own that placement, the way pyvene placed
        ``source_representations``. Call inside ``with model.trace(...):``."""
        self._rewrite(model, g, positions)

    def write(
        self,
        model: Any,
        features: Any,
        positions: Positions | None = None,
    ) -> None:
        """In-trace pure write: replace the (selected) base features with
        ``features`` — a constant vector, or another site's :meth:`read` (the
        interchange pattern). Base still contributes the reconstruction error
        and, under ``feature_ids``, the untouched columns; a zero ``features``
        therefore ablates the feature contribution only. Call inside
        ``with model.trace(...):``."""
        self._rewrite(model, lambda f: features, positions)

    def _rewrite(
        self,
        model: Any,
        g: Callable[[Any], Any],
        positions: Positions | None,
    ) -> None:
        """Shared RMW core: featurize → transform (full vector or the
        ``feature_ids`` scatter) → inverse with base error → site write."""
        base = self.site.read(model, positions)
        f, err = self.featurizer.featurize(base)
        if self.feature_ids is None:
            f_out = self._coerce(g(f), f)
        else:
            ids = list(self.feature_ids)
            f_out = f.clone()  # keep the untouched columns from base
            f_out[..., ids] = self._coerce(g(f[..., ids]), f)
        self.site.write(model, self.featurizer.inverse_featurize(f_out, err), positions)

    @staticmethod
    def _coerce(value: Any, like: Any) -> Any:
        """Move an externally-supplied feature tensor to the featurized device
        and dtype (steering vectors and replacement constants arrive as the
        caller made them — typically CPU — while the features live on the
        site's GPU; the scatter and the inverse's error addition both need an
        exact match). Non-tensors (e.g. proxies) pass through — they are
        already the trace's."""
        if isinstance(value, torch.Tensor):
            return value.to(device=like.device, dtype=like.dtype)
        return value

    # -- one-shot read ------------------------------------------------------------ #
    def collect(
        self,
        model: Any,
        inputs: Any,
        positions: Positions | None = None,
    ) -> torch.Tensor:
        """Featurized :meth:`Site.collect`: read this site through the feature
        space in a single forward pass and return a concrete **CPU** tensor
        ``(batch, seq | len(positions), n_features | len(feature_ids))``. The
        forward stops right after the tap (:func:`collect_ordered`)."""
        return collect_ordered(
            model,
            inputs,
            [(forward_key(self.site, model), lambda m: self.read(m, positions))],
        )[0]
