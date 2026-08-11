"""Corruption and clean-restoration vectors for causal tracing (sufficiency).

Causal tracing corrupts the information where it enters the network and then
restores one clean site at a time. This module produces the ``{spec.key: tensor}``
dicts the tracing runner feeds to the (possibly mixed) intervention pass.

Two kinds of corruption mechanism are supported, dispatched by
:func:`corruption_intervention_type`:

* ``zero`` / ``mean`` are **replace** interventions — a broadcast
  ``(n_features,)`` vector overwrites the entry span (the ablation reference
  vectors, reused verbatim), so they may span any number of tokens.
* ``noise`` is the seeded additive-Gaussian **noise** intervention of ROME-style
  tracing: each entry-token activation gets independent Gaussian noise drawn
  *inside* the forward pass (so it natively spans the whole multi-token subject).
  :func:`make_corruption_vectors` returns the per-site *noise scale*
  ``noise_scale * σ`` — ``noise_scale`` is the multiple of the subject-embedding
  standard deviation ``σ`` (ROME's ``ν = 3σ`` ⇒ ``noise_scale = 3``), and ``σ`` is
  estimated from the entry-span activations over the dataset.

**Clean restoration** (:func:`collect_clean_vectors`) re-injects each example's
own clean activation at every swept site; it is a per-example **replace** value
collected from one un-intervened pass and so requires single-position sites (one
token per restored ``(layer, token)`` cell).

Sites are :class:`~causalab.neural.specs.SiteSpec` values (WU4, #506).
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.ablation._spans import (
    group_by_position_count,
    spec_position_count,
)
from causalab.methods.ablation.reference_vectors import (
    _single_position_spec,  # pyright: ignore[reportPrivateUsage]
    make_mean_vectors,
    make_zero_vectors,
)
from causalab.neural.activations.collect import collect_features
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import Pipeline
from causalab.neural.specs import SiteSpec

CorruptionKind = Literal["zero", "mean", "noise"]
VALID_CORRUPTIONS: tuple[CorruptionKind, ...] = ("zero", "mean", "noise")


def _n_features(spec: SiteSpec) -> int:
    """Feature dimensionality of a site, falling back to its raw ``width``
    (identity featurizer)."""
    n_features = spec.fsite.featurizer.n_features
    if n_features is None:
        if spec.width is None:
            raise ValueError(
                f"Site {spec.key!r} has no n_features and no width; cannot size "
                "its noise scale vector."
            )
        n_features = spec.width
    return n_features


def corruption_intervention_type(kind: CorruptionKind) -> str:
    """The intervention type a corruption kind applies at the entry site.

    ``noise`` uses the dynamic, per-position ``noise`` intervention; ``zero`` and
    ``mean`` overwrite with a fixed vector via ``replace``.
    """
    if kind not in VALID_CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption kind {kind!r}; expected one of {VALID_CORRUPTIONS}."
        )
    return "noise" if kind == "noise" else "replace"


def _require_single_position(
    sites: Sequence[SiteSpec], reference_input: object, why: str
) -> None:
    """Raise unless every site gathers exactly one token position.

    Per-example *replace* collection (clean restoration) reads one activation row
    per example, so the site's span must resolve to a single token.
    """
    for spec in sites:
        n_pos = spec_position_count(spec, reference_input)
        if n_pos != 1:
            raise ValueError(
                f"{why} requires single-position spans (one token per site), but "
                f"site {spec.key!r} spans {n_pos} positions. Use a single named "
                "token position (e.g. a subject or last token) for the restore "
                "span."
            )


def collect_clean_vectors(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    sites: Sequence[SiteSpec],
    batch_size: int = 32,
) -> dict[str, Tensor]:
    """Per-example clean activations at every swept site (the restore values).

    Returns ``{spec.key: (n_examples, n_features)}`` collected on the un-intervened
    base inputs, in dataset order. Restoring a site with these values re-injects
    exactly the activation it held on the clean run. Sites must be
    single-position (validated up front so the error is actionable).
    """
    if not dataset:
        raise ValueError("Cannot collect clean vectors from an empty dataset.")
    _require_single_position(sites, dataset[0]["input"], "Site restoration")

    feats = collect_features(dataset, pipeline, sites, batch_size=batch_size)
    assert isinstance(feats, dict)  # collect_output_logits=False

    n = len(dataset)
    result: dict[str, Tensor] = {}
    for spec in sites:
        rows = feats[spec.key]
        if rows.shape[0] != n:
            raise ValueError(
                f"Site {spec.key!r} collected {rows.shape[0]} rows for {n} examples; "
                "site restoration requires a single token position per example."
            )
        result[spec.key] = rows.float().cpu()
    return result


def _entry_activation_std(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    entry_sites: Sequence[SiteSpec],
    batch_size: int,
) -> float:
    """Standard deviation of the entry-span activations over ``dataset`` (ROME's σ).

    Collects every gathered (example, position) activation at the entry sites and
    returns the scalar std across all of them — the subject-embedding σ ROME
    scales its corruption noise by. Multi-position residual/MLP spans are
    collected one position at a time (the same legacy-inherited slicing as
    :func:`make_mean_vectors` — the pooled std is invariant to it), bucketed so
    each gather stays rectangular.
    """
    entry_sites = list(entry_sites)
    chunks: list[Tensor] = []

    for _, sub in group_by_position_count(entry_sites, dataset):
        reference = sub[0]["input"]
        simple: list[SiteSpec] = []
        multi: list[tuple[SiteSpec, int]] = []
        for spec in entry_sites:
            n_pos = spec_position_count(spec, reference)
            if not isinstance(spec.fsite.site, HeadSite) and n_pos > 1:
                multi.append((spec, n_pos))
            else:
                simple.append(spec)

        if simple:
            feats = collect_features(sub, pipeline, simple, batch_size=batch_size)
            assert isinstance(feats, dict)
            chunks.extend(feats[s.key].float() for s in simple)

        max_n_pos = max((n for _, n in multi), default=0)
        for k in range(max_n_pos):
            clones = [_single_position_spec(spec, k) for spec, n in multi if k < n]
            if clones:
                feats = collect_features(sub, pipeline, clones, batch_size=batch_size)
                assert isinstance(feats, dict)
                chunks.extend(feats[c.key].float() for c in clones)

    if not chunks:
        raise ValueError(
            "No entry activations collected; cannot estimate the corruption-noise "
            "scale σ from an empty dataset."
        )
    return float(torch.cat([c.reshape(-1) for c in chunks]).std())


def make_corruption_vectors(
    kind: CorruptionKind,
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    entry_sites: Sequence[SiteSpec],
    *,
    noise_scale: float = 3.0,
    noise_seed: int = 0,
    batch_size: int = 32,
) -> dict[str, Tensor]:
    """Per-site reference vectors that corrupt the entry span (the floor).

    * ``zero``/``mean`` → broadcast ``(n_features,)`` *replace* vectors (any span).
    * ``noise`` → per-site *noise scale* ``(n_features,)`` filled with
      ``noise_scale * σ`` (σ = entry-span activation std), consumed by the dynamic
      noise intervention as the std of the Gaussian it adds per token. The
      ``noise_seed`` argument is accepted for signature symmetry but the seed is
      applied where the intervention is built (see ``run_causal_trace``); noise
      spans any number of tokens, so there is no single-position restriction.

    Keys cover every spec in ``entry_sites``.
    """
    entry_sites = list(entry_sites)
    if kind == "zero":
        return make_zero_vectors(entry_sites)
    if kind == "mean":
        return make_mean_vectors(pipeline, dataset, entry_sites, batch_size=batch_size)
    if kind == "noise":
        sigma = _entry_activation_std(pipeline, dataset, entry_sites, batch_size)
        scale = noise_scale * sigma
        return {
            spec.key: torch.full((_n_features(spec),), scale) for spec in entry_sites
        }
    raise ValueError(
        f"Unknown corruption kind {kind!r}; expected one of {VALID_CORRUPTIONS}."
    )
