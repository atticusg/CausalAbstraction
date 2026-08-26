"""Counterfactual datasets for the hypotheses in models.py.

This is a SCAFFOLD. Replace the toy generators, but keep the three tiers:

  WIDE   : broad counterfactual datasets (random resampling under task-appropriate balancing, or a
           systematic manipulation -- swap order / shuffle / hold the template and
           resample infills). Robust to the causal model being wrong. A wide counterfactual dataset
           is EXPECTED to distinguish hypotheses only imperfectly; that is fine.
  NARROW : sharply targeted pairs that hold one variable fixed and flip another to
           separate two specific hypotheses.
  SINGLE-TOKEN : base and counterfactual differ by exactly one token realizing one
           input variable. Low separating power by design -- they exist to trace
           that variable's path through the network, bridging to the downstream
           interchange/localization experiments (build one per variable you want
           to follow).

``random_pairs`` is the canonical wide generator AND the source the
``develop_hypothesis`` analysis uses for the large confounding-detection run, so it must
sample broadly and validly. Every example must carry EXACTLY ONE counterfactual
input.

When critiquing an EXISTING task, replace the bodies here with thin wrappers over
that task's shipped ``COUNTERFACTUAL_GENERATORS`` and point ``random_pairs`` at
its random generator -- nothing else in the distinguishability step changes.
"""

from __future__ import annotations

import random

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample


def _t(model: CausalModel, x: int, y: int):
    return model.new_trace({"X": x, "Y": y})


def random_pairs(
    model: CausalModel, n: int, seed: int = 0
) -> list[CounterfactualExample]:
    """Independent uniform samples. Canonical wide counterfactual dataset + large-random-run source."""
    st = random.getstate()
    random.seed(seed)
    try:
        out = []
        for _ in range(n):
            base = _t(model, random.randint(0, 9), random.randint(0, 9))
            cf = _t(model, random.randint(0, 9), random.randint(0, 9))
            out.append(CounterfactualExample(input=base, counterfactual_inputs=[cf]))
        return out
    finally:
        random.setstate(st)


def _split_to_trace(model: CausalModel, total: int):
    x = random.randint(max(0, total - 9), min(9, total))
    return _t(model, x, total - x)


def narrow_flip_carry_fix_ones(model: CausalModel) -> CounterfactualExample:
    """Ones digit fixed, carry flipped -- isolates the carry hypothesis."""
    d = random.randint(0, 8)
    return CounterfactualExample(
        input=_split_to_trace(model, d),
        counterfactual_inputs=[_split_to_trace(model, d + 10)],
    )


def narrow_flip_ones_fix_carry(model: CausalModel) -> CounterfactualExample:
    """Carry fixed (both False), ones changed -- isolates the ones hypothesis."""
    d1, d2 = random.sample(range(0, 10), 2)
    return CounterfactualExample(
        input=_split_to_trace(model, d1),
        counterfactual_inputs=[_split_to_trace(model, d2)],
    )


NARROW_GENERATORS = {
    "narrow_flip_carry_fix_ones": narrow_flip_carry_fix_ones,
    "narrow_flip_ones_fix_carry": narrow_flip_ones_fix_carry,
}

# Wide/narrow/single-token classification and intended train/eval split per dataset. Downstream
# supervised localizers train on the train split and evaluate on the eval splits;
# never train and evaluate on the same pairs.
DATASET_ROLES = {
    "wide_random": {"width": "wide", "split": "train"},
    "narrow_flip_carry_fix_ones": {"width": "narrow", "split": "eval"},
    "narrow_flip_ones_fix_carry": {"width": "narrow", "split": "eval"},
}


def make_datasets(
    model: CausalModel, n: int = 300, seed: int = 0
) -> dict[str, list[CounterfactualExample]]:
    """Design datasets (modest n). The big random run is separate."""
    datasets = {"wide_random": random_pairs(model, n, seed)}
    st = random.getstate()
    random.seed(seed + 1)
    try:
        for name, gen in NARROW_GENERATORS.items():
            datasets[name] = [gen(model) for _ in range(n)]
    finally:
        random.setstate(st)
    return datasets
