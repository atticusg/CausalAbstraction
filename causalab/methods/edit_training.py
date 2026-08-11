"""The outer optimization loop over trainable edits — ED3's training loop.

:mod:`causalab.neural.trainable` owns the *grad contract* (freeze at load,
in-trace trainable modules, saved-logits backward via
:func:`~causalab.neural.trainable.traced_label_loss`), the training edit
shapes (:func:`~causalab.neural.trainable.das_edit` /
:func:`~causalab.neural.trainable.dbm_edit`), the differentiable loss slice,
and explicit device placement. This module owns what sits *above* those
primitives: the epochs/AdamW/anneal loop (:func:`train_edits`), its per-step
input contract (:class:`TrainBatch`), and the DBM temperature schedule
(:func:`temperature_schedule`) — a training loop over learned interventions,
which docs/CODEBASE.md §3 places in ``methods/`` (invariant 1), with no
hyperparameter defaults in the signatures (invariant 5: defaults live exactly
once, in ``causalab/configs/`` — see ``configs/train_config.py``'s
``DEFAULT_CONFIG``).

The production DAS/DBM harness (`causalab/methods/trained_subspace/train.py`)
composes the same neural primitives with its own orchestration (LR scheduler,
early stopping, memory hygiene); :func:`train_edits` is the lean loop the ED3
toolkit ships for direct use.

Scope
-----
LR schedulers, early stopping, and experiment-harness wiring stay with the
``trained_subspace`` harness; dataset-scale paired batching is PL3
(:mod:`causalab.neural.dataset`). Multiple edits in one trace are applied in
forward order of their *sites*; constant/raw sources (the training pattern)
impose no cross-edit ordering constraint.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence

import torch
from nnterp import StandardizedTransformer

from causalab.neural.edit import Edit
from causalab.neural.modes import MaskGate
from causalab.neural.trainable import (
    edit_parameters,
    freeze_model_parameters,
    place_edit_parameters,
    traced_label_loss,
)

__all__ = [
    "TrainBatch",
    "temperature_schedule",
    "train_edits",
]


def temperature_schedule(
    start: float,
    end: float,
    total_steps: int,
    annealing_fraction: float,
) -> torch.Tensor:
    """Per-step DBM gate temperatures: linear anneal ``start → end`` over the
    first ``annealing_fraction`` of steps, then constant at ``end`` (the
    ``train_interventions`` schedule; the canonical values live in
    ``configs/train_config.py``'s ``DEFAULT_CONFIG`` —
    ``masking.temperature_schedule`` / ``masking.temperature_annealing_fraction``).
    Length ``total_steps``."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    annealing_steps = min(int(total_steps * annealing_fraction), total_steps)
    ramp = torch.linspace(start, end, annealing_steps) if annealing_steps else None
    tail = torch.full((total_steps - annealing_steps,), end)
    return tail if ramp is None else torch.cat([ramp, tail])


@dataclasses.dataclass(frozen=True)
class TrainBatch:
    """One training step's worth of work: label-concatenated ``inputs`` (from
    :func:`~causalab.neural.trainable.concat_label_inputs`), the ``label_ids``
    it returned, and the edits to apply — rebuilt per batch via the
    mode/:func:`~causalab.neural.trainable.das_edit`/
    :func:`~causalab.neural.trainable.dbm_edit` constructors around **shared**
    featurizer/gate modules (the state that trains lives in the modules, not
    in the frozen :class:`~causalab.neural.edit.Edit` values)."""

    inputs: dict[str, torch.Tensor]
    label_ids: torch.Tensor
    edits: tuple[Edit, ...]


def train_edits(
    model: StandardizedTransformer,
    batches: Sequence[TrainBatch],
    *,
    pad_token_id: int,
    epochs: int,
    lr: float,
    temperature: tuple[float, float],
    annealing_fraction: float,
    regularization_coefficient: float,
    gates: Sequence[MaskGate] = (),
    shuffle: bool = True,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Train the parameters behind ``batches``' edits; returns per-step stats
    (``loss``, ``accuracy``).

    Hyperparameters are explicit keyword arguments with no defaults (CODEBASE
    §3 invariant 5) — the canonical values live in
    ``configs/train_config.py``'s ``DEFAULT_CONFIG`` (``training_epoch``,
    ``init_lr``, ``masking.temperature_schedule``,
    ``masking.temperature_annealing_fraction``,
    ``masking.regularization_coefficient``).

    The loop is deliberately lean — freeze, place, AdamW, anneal, step:

    * :func:`~causalab.neural.trainable.freeze_model_parameters` +
      :func:`~causalab.neural.trainable.place_edit_parameters` once at entry
      (idempotent; placement is explicit, never a ``get_device`` patch).
    * One AdamW (``weight_decay=0``, matching ``train_interventions``) over
      :func:`~causalab.neural.trainable.edit_parameters` of all batches —
      shared modules dedupe, so an epoch of batches trains the same
      gate/rotation.
    * DBM ``gates`` get :func:`temperature_schedule`'s per-step temperature set
      as an attribute, plus normalized L1 sparsity
      (``regularization_coefficient · Σ sparsity / Σ numel``) added to the loss.
    * Accuracy is exact label-token match over non-pad positions — the cheap
      in-loop metric; causal scoring stays with the scoring layer (MX1).

    LR schedulers, early stopping, and harness wiring (tensorboard, memory
    hygiene) live with the ``trained_subspace`` harness — not here.
    """
    if not batches:
        raise ValueError("train_edits needs at least one batch")
    freeze_model_parameters(model)
    all_edits = [e for b in batches for e in b.edits]
    place_edit_parameters(model, all_edits)
    params = edit_parameters(all_edits)
    if not params:
        raise ValueError(
            "no trainable parameters found on the batches' edits — pass a "
            "trainable featurizer (e.g. SubspaceFeaturizer(trainable=True)) "
            "or a MaskGate"
        )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)

    total_steps = epochs * len(batches)
    temps = temperature_schedule(*temperature, total_steps, annealing_fraction)
    mask_numel = sum(g.mask.numel() for g in gates)

    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    step = 0
    for _epoch in range(epochs):
        order = list(range(len(batches)))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            batch = batches[i]
            for gate in gates:
                gate.set_temperature(temps[step])
            loss, pred_ids = traced_label_loss(
                model, batch.inputs, batch.label_ids, batch.edits, pad_token_id
            )
            if gates and regularization_coefficient:
                sparsity = sum(g.sparsity_loss().to(loss.device) for g in gates)
                loss = loss + regularization_coefficient * (sparsity / mask_numel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            labels = batch.label_ids.cpu()
            real = labels != pad_token_id
            accuracy = float(
                ((pred_ids == labels) | ~real).all(dim=-1).float().mean().item()
            )
            history.append({"loss": float(loss.item()), "accuracy": accuracy})
            step += 1
    return history
