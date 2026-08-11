"""The read-modify-write primitive over a :class:`FeaturizedSite` — ED1.

ST3 (:mod:`causalab.neural.featurized_site`) already ships the mechanics of the
design doc's

.. code-block:: text

    f   = featurize(read(site))      # read may be omitted (pure write)
    f'  = g(f, *aux)                 # aux = values from other sites / inputs
    write(site, inverse(f'))         # write may be omitted (pure collect)

as three plain methods: :meth:`FeaturizedSite.collect` (read-only),
:meth:`FeaturizedSite.write` (write-only / "replace"), and
:meth:`FeaturizedSite.edit` (general RMW). What is missing is the design doc's
abstraction-stack item 3: **``Edit`` — a site + a feature-space ``g`` + its
read-sources** — a declarative value, not a Python closure, so a later plan
compiler (PL1) can introspect what an edit depends on instead of reaching into
opaque closures. :class:`Edit` is that value; :class:`ReadSource` reifies one
entry of ``*aux``.

The seven pyvene modes (ED2, a later wave) become thin constructors over
:class:`Edit`:

* **collect** — ``Edit(site)`` (``g=None``), via :meth:`Edit.collect`.
* **replace** — ``Edit(site, g=lambda f: value)``, via :meth:`Edit.apply`.
* **steer / noise** — ``g`` closes over its own parameters (a factor, a
  seed); the vector/scale rides a constant ``ReadSource``.
* **interchange / interpolate / mask** — ``g`` combines the base features with
  one or more ``read_sources`` (another site's features, or a precomputed
  constant); mask's ``g`` is the learnable gate itself.

Scope
-----
:meth:`Edit.apply` composes reads and one write **within a single already-open
trace over one input** — the shape ST3's
``test_cross_site_feature_transplant_in_one_trace`` already exercises by hand
(``dst.write(model, src.read(model, [last]))``). Cross-*input* composition
(reading a counterfactual/source input to intervene on a base input, via
multiple ``invoke`` blocks or staged sequential traces) is the plan
compiler's job (PL1) — out of scope here. Trainable parameters (mask/DAS) and the gradient
contract are ED3's job, gated on the F6 spike; nothing here needs it; like
``FeaturizedSite``, gradients flow through by construction (plain tensor ops).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch
from nnterp import StandardizedTransformer

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.site import Positions, forward_key

__all__ = ["Edit", "ReadSource"]


@dataclasses.dataclass(frozen=True)
class ReadSource:
    """One auxiliary value :attr:`Edit.g` needs beyond its own site's base
    features.

    ``value`` is either another :class:`FeaturizedSite` — read in-trace,
    forward-order checked against the owning :class:`Edit`'s site, honoring
    nnsight's constraint that reads within one trace must be requested in
    forward-execution order — or any other value (a steering vector, an
    interpolation target), passed through unchanged except for the same
    device/dtype coercion a site-backed read gets. Resolving and placing this
    value is :class:`Edit`'s job, not each mode constructor's — mirroring how
    pyvene placed ``source_representations`` before an intervention ran.

    ``input`` names the *plan input* the site is read under (a key into
    ``Plan.inputs``, :mod:`causalab.neural.plan`). ``None`` — the only value
    :meth:`Edit.apply` accepts — means "the same input this Edit runs on".
    Setting it makes the read **cross-input** (the interchange pattern: read
    the source input's features to write the base input's site), which only
    the plan compiler can realize — it must stage the read in another
    ``tracer.invoke`` and synchronize with a barrier. A cross-input read is
    necessarily site-backed: a constant carries no notion of "under which
    input".
    """

    value: FeaturizedSite | Any
    positions: Positions | None = None
    input: str | None = None

    def __post_init__(self) -> None:
        if self.input is not None and not self.is_site:
            raise ValueError(
                "a cross-input ReadSource (input=...) must be site-backed — a "
                "non-site value is input-independent, so name no input for it"
            )

    @property
    def is_site(self) -> bool:
        return isinstance(self.value, FeaturizedSite)


@dataclasses.dataclass(frozen=True)
class Edit:
    """A :class:`FeaturizedSite` + a feature-space transform ``g`` + the extra
    sites/values ``g`` reads (:class:`ReadSource`) — the design doc's
    ``f' = g(f, *aux)``. One shape covers all three RMW cases:

    * read-only ("collect"): ``g=None``; call :meth:`collect`.
    * write-only ("replace"): ``g`` ignores its ``f`` argument, e.g.
      ``lambda f: value``.
    * general RMW: ``g`` combines ``f`` with the resolved ``read_sources``.

    ``read_sources`` stay a declarative field (not baked into ``g``'s closure)
    so a later plan compiler can inspect what this edit depends on.
    """

    site: FeaturizedSite
    g: Callable[..., Any] | None = None
    read_sources: tuple[ReadSource, ...] = ()
    positions: Positions | None = None

    def __post_init__(self) -> None:
        if self.g is None and self.read_sources:
            raise ValueError(
                "read_sources requires g — a read-only Edit (g=None) takes no aux"
            )

    # -- in-trace RMW -------------------------------------------------------- #
    def apply(self, model: StandardizedTransformer) -> None:
        """In-trace read-modify-write: resolve ``read_sources`` (forward-order
        checked, device/dtype coerced to the base features), then
        ``site.edit(model, g_wrapped, positions)``. Call inside
        ``with model.trace(...):``, over the same input this :class:`Edit`'s
        ``read_sources`` address — composing across a *different* input is the
        plan compiler's job (PL1), not this method's."""
        if self.g is None:
            raise ValueError(
                "a read-only Edit (g=None) has nothing to apply; use .collect(...)"
            )
        cross = [i for i, rs in enumerate(self.read_sources) if rs.input is not None]
        if cross:
            raise ValueError(
                f"read_sources{cross} address another plan input — a single "
                "Edit.apply() runs over one input and cannot stage a cross-input "
                "read; run this Edit through a Plan "
                "(causalab.neural.plan.run_plan)."
            )
        resolved = self._resolve_read_sources(model)
        g = self.g

        def g_wrapped(f: Any) -> Any:
            aux = tuple(FeaturizedSite._coerce(v, f) for v in resolved)
            return g(f, *aux)

        self.site.edit(model, g_wrapped, self.positions)

    # -- one-shot read --------------------------------------------------------- #
    def collect(self, model: StandardizedTransformer, inputs: Any) -> torch.Tensor:
        """The read-only shape: one-shot featurized read of this edit's site,
        no write. Valid regardless of ``g`` — a collect never applies it."""
        return self.site.collect(model, inputs, self.positions)

    # -- read_sources resolution ------------------------------------------------ #
    def _resolve_read_sources(self, model: StandardizedTransformer) -> tuple[Any, ...]:
        """Read every site-backed :class:`ReadSource` in ascending
        ``(layer, forward_rank_on(model))`` order — the order nnsight's
        forward-pass-interleaved trace requires — then reassemble the results
        (and pass through non-site values unchanged) in the declared
        ``read_sources`` order ``g`` expects. Raises if a site-backed source
        fires strictly after this edit's own site: that dependency needs a
        cross-pass plan (PL1), which a single ``Edit.apply()`` cannot express.
        """
        dst_rank = forward_key(self.site.site, model)
        order = sorted(
            (i for i, rs in enumerate(self.read_sources) if rs.is_site),
            key=lambda i: forward_key(self.read_sources[i].value.site, model),
        )
        resolved: list[Any] = [rs.value for rs in self.read_sources]
        for i in order:
            rs = self.read_sources[i]
            src_rank = forward_key(rs.value.site, model)
            if src_rank > dst_rank:
                raise ValueError(
                    f"read_sources[{i}] ({rs.value.site!r}, rank {src_rank}) fires "
                    f"after this Edit's site ({self.site.site!r}, rank {dst_rank}) "
                    "in forward order — reading a later site to write an earlier "
                    "one needs a cross-pass plan (PL1), not a single Edit.apply()."
                )
            resolved[i] = rs.value.read(model, rs.positions)
        return tuple(resolved)
