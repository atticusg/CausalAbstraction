"""Counterfactual datasets for the unified-arithmetic shared-calculator test.

Two tiers:
  WITHIN-DOMAIN partial counterfactual datasets -- isolate which operand is localized (entity vs
    number vs the sum). These are the contests can_distinguish can actually settle.
  CROSS-DOMAIN -- base and counterfactual in DIFFERENT domains. Patching raw_sum
    here transplants one task's integer sum into another's reduce/decode. This is
    where raw_sum (domain-free) separates from reduced/result (domain-entangled),
    and the substrate for the shared-vs-split generalization test downstream.

Plus per-domain random sets, so a localizer can train on one domain and be
evaluated on another (the only way to tell shared from split).
"""

from __future__ import annotations

import random

from causalab.causal.counterfactual_dataset import CounterfactualExample
from models import DOMAINS


def _one(model, dom, rng):
    d = DOMAINS[dom]
    return model.new_trace(
        {
            "entity": rng.choice(d.entities),
            "number": rng.choice(d.numbers),
            "domain": dom,
        }
    )


def random_pairs(model, n, seed=0):
    """Independent random samples across ALL domains (base and cf domains vary
    freely, so the pairing includes cross-domain pairs)."""
    rng = random.Random(seed)
    doms = list(DOMAINS)
    return [
        CounterfactualExample(
            input=_one(model, rng.choice(doms), rng),
            counterfactual_inputs=[_one(model, rng.choice(doms), rng)],
        )
        for _ in range(n)
    ]


def _within(model, n, seed, vary):
    rng = random.Random(seed)
    doms = list(DOMAINS)
    out = []
    for _ in range(n):
        dom = rng.choice(doms)
        d = DOMAINS[dom]
        e, nv = rng.choice(d.entities), rng.choice(d.numbers)
        base = model.new_trace({"entity": e, "number": nv, "domain": dom})
        if vary == "entity":
            e2 = rng.choice([x for x in d.entities if x != e] or [e])
            cf = model.new_trace({"entity": e2, "number": nv, "domain": dom})
        else:
            n2 = rng.choice([x for x in d.numbers if x != nv] or [nv])
            cf = model.new_trace({"entity": e, "number": n2, "domain": dom})
        out.append(CounterfactualExample(input=base, counterfactual_inputs=[cf]))
    return out


def _cross_domain(model, n, seed):
    rng = random.Random(seed)
    doms = list(DOMAINS)
    out = []
    for _ in range(n):
        da, db = rng.sample(doms, 2)
        out.append(
            CounterfactualExample(
                input=_one(model, da, rng), counterfactual_inputs=[_one(model, db, rng)]
            )
        )
    return out


def _domain_only(model, n, seed, dom):
    rng = random.Random(seed)
    return [
        CounterfactualExample(
            input=_one(model, dom, rng), counterfactual_inputs=[_one(model, dom, rng)]
        )
        for _ in range(n)
    ]


DATASET_ROLES = {
    "wide_all_domains": {"width": "wide", "split": "train"},
    "change_entity_fix_number": {"width": "narrow", "split": "eval"},
    "change_number_fix_entity": {"width": "narrow", "split": "eval"},
    "cross_domain": {"width": "wide", "split": "eval"},
    "integer_only": {"width": "wide", "split": "train (one domain)"},
    "weekdays_only": {"width": "wide", "split": "eval (held-out domain)"},
}


def make_datasets(model, n=300, seed=0):
    return {
        "wide_all_domains": random_pairs(model, n, seed),
        "change_entity_fix_number": _within(model, n, seed + 1, "entity"),
        "change_number_fix_entity": _within(model, n, seed + 2, "number"),
        "cross_domain": _cross_domain(model, n, seed + 3),
        "integer_only": _domain_only(model, n, seed + 4, "integer"),
        "weekdays_only": _domain_only(model, n, seed + 5, "weekdays"),
    }
