"""Corruption and clean-restoration vectors for causal tracing (sufficiency).

Causal tracing corrupts the information where it enters the network and then
restores one clean site at a time. This module produces the ``{unit_id: tensor}``
dicts the tracing runner feeds to the (possibly mixed) intervention pass.

Two kinds of corruption mechanism are supported, dispatched by
:func:`corruption_intervention_type`:

* ``zero`` / ``mean`` are **replace** interventions — a broadcast
  ``(n_features,)`` vector overwrites the entry span (the ablation reference
  vectors, reused verbatim), so they may span any number of tokens.
* ``noise`` is the seeded additive-Gaussian **noise** intervention of ROME-style
  tracing: each entry-token activation gets independent Gaussian noise drawn
  *inside* the forward pass (so it natively spans the whole multi-token subject).
  :func:`make_corruption_vectors` returns the per-unit *noise scale*
  ``noise_scale * σ`` — ``noise_scale`` is the multiple of the subject-embedding
  standard deviation ``σ`` (ROME's ``ν = 3σ`` ⇒ ``noise_scale = 3``), and ``σ`` is
  estimated from the entry-span activations over the dataset.

**Clean restoration** (:func:`collect_clean_vectors`) re-injects each example's
own clean activation at every swept site; it is a per-example **replace** value
collected from one un-intervened pass and so requires single-position sites (one
token per restored ``(layer, token)`` cell).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.ablation._spans import unit_position_count
from causalab.methods.ablation.reference_vectors import (
    make_mean_vectors,
    make_zero_vectors,
)
from causalab.neural.activations.collect import collect_features
from causalab.neural.pipeline import Pipeline
from causalab.neural.units import AtomicModelUnit, InterchangeTarget

CorruptionKind = Literal["zero", "mean", "noise"]
VALID_CORRUPTIONS: tuple[CorruptionKind, ...] = ("zero", "mean", "noise")


def _n_features(unit: AtomicModelUnit) -> int:
    """Feature dimensionality of a unit, falling back to its shape (identity feat)."""
    n_features = unit.featurizer.n_features
    if n_features is None:
        if unit.shape is None:
            raise ValueError(
                f"Unit {unit.id!r} has no n_features and no shape; cannot size its "
                "noise scale vector."
            )
        n_features = unit.shape[0]
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
    units: list[AtomicModelUnit], reference_input: object, why: str
) -> None:
    """Raise unless every unit gathers exactly one token position.

    Per-example *replace* collection (clean restoration) reads one activation row
    per example, so the unit's span must resolve to a single token.
    """
    for unit in units:
        n_pos = unit_position_count(unit, reference_input)
        if n_pos != 1:
            raise ValueError(
                f"{why} requires single-position spans (one token per site), but "
                f"unit {unit.id!r} spans {n_pos} positions. Use a single named "
                "token position (e.g. a subject or last token) for the restore "
                "span."
            )


def collect_clean_vectors(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    units: list[AtomicModelUnit],
    batch_size: int = 32,
) -> dict[str, Tensor]:
    """Per-example clean activations at every swept site (the restore values).

    Returns ``{unit_id: (n_examples, n_features)}`` collected on the un-intervened
    base inputs, in dataset order. Restoring a unit with these values re-injects
    exactly the activation it held on the clean run. Sites must be
    single-position (validated up front so the error is actionable).
    """
    if not dataset:
        raise ValueError("Cannot collect clean vectors from an empty dataset.")
    _require_single_position(units, dataset[0]["input"], "Site restoration")

    feats = collect_features(dataset, pipeline, units, batch_size=batch_size)
    assert isinstance(feats, dict)  # collect_output_logits=False

    n = len(dataset)
    result: dict[str, Tensor] = {}
    for unit in units:
        rows = feats[unit.id]
        if rows.shape[0] != n:
            raise ValueError(
                f"Unit {unit.id!r} collected {rows.shape[0]} rows for {n} examples; "
                "site restoration requires a single token position per example."
            )
        result[unit.id] = rows.float().cpu()
    return result


def _entry_activation_std(
    pipeline: Pipeline,
    dataset: list[CounterfactualExample],
    entry_target: InterchangeTarget,
    batch_size: int,
) -> float:
    """Standard deviation of the entry-span activations over ``dataset`` (ROME's σ).

    Collects every gathered (example, position) activation at the entry units and
    returns the scalar std across all of them — the subject-embedding σ ROME
    scales its corruption noise by. One collect covers every unit and every
    position it selects, whatever each example's span width.
    """
    units = entry_target.flatten()
    chunks: list[Tensor] = []

    feats = collect_features(dataset, pipeline, units, batch_size=batch_size)
    assert isinstance(feats, dict)
    chunks.extend(feats[u.id].float() for u in units)

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
    entry_target: InterchangeTarget,
    *,
    noise_scale: float = 3.0,
    noise_seed: int = 0,
    batch_size: int = 32,
) -> dict[str, Tensor]:
    """Per-unit reference vectors that corrupt the entry span (the floor).

    * ``zero``/``mean`` → broadcast ``(n_features,)`` *replace* vectors (any span).
    * ``noise`` → per-unit *noise scale* ``(n_features,)`` filled with
      ``noise_scale * σ`` (σ = entry-span activation std), consumed by the dynamic
      noise intervention as the std of the Gaussian it adds per token. The
      ``noise_seed`` argument is accepted for signature symmetry but the seed is
      applied where the intervention is built (see ``run_causal_trace``); noise
      spans any number of tokens, so there is no single-position restriction.

    Keys cover every unit in ``entry_target``.
    """
    if kind == "zero":
        return make_zero_vectors(entry_target)
    if kind == "mean":
        return make_mean_vectors(pipeline, dataset, entry_target, batch_size=batch_size)
    if kind == "noise":
        sigma = _entry_activation_std(pipeline, dataset, entry_target, batch_size)
        scale = noise_scale * sigma
        return {
            unit.id: torch.full((_n_features(unit),), scale)
            for unit in entry_target.flatten()
        }
    raise ValueError(
        f"Unknown corruption kind {kind!r}; expected one of {VALID_CORRUPTIONS}."
    )
