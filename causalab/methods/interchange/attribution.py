"""Attribution-patching pre-scan for interchange grids — CAP3 (#456).

A one-backward **gradient × Δactivation** approximation of interchange
effects, used to prune (layer × position) grids before exact interchange runs
on the survivors. For a counterfactual pair and a candidate cell, the
first-order effect of patching the source activation into the base run on a
scalar readout ``m`` is::

    m(patched) − m(base)  ≈  ∇_a m(base) · (a_source − a_base)

so ONE forward over the counterfactual batch (activations at every candidate
site) plus ONE forward+backward over the base batch (activations *and*
gradients at every candidate site, through the Plan IR's
:class:`~causalab.neural.plan.GradientRequest`) scores the whole grid — versus
one exact interchange run *per cell*.

The scalar readout is the standard attribution-patching logit difference
toward the counterfactual answer: ``logit[cf_answer] − logit[base_answer]``
at each example's final prompt position. A large positive approximate score
means patching that cell is predicted to move the output toward the causal
model's expected counterfactual label; gates rank candidates by
**magnitude** (``select_top_k(..., by_abs=True)``) — the attribution-patching
convention, because the linearization's *sign* is unreliable through many
downstream non-linearities (measured on the weekdays golden: the causally
strongest early-layer cells carry the largest ``|approx|`` with a flipped
sign) while the magnitude still separates live cells from dead ones.

Scores are **approximations**: they linearize the network around the base
run, so saturated or strongly non-linear cells mis-rank. That is why this is
a *pre-scan gate* (exact interchange still runs on the top-k survivors, and
callers report both scores where both exist — see
``causalab/analyses/locate`` / ``causalab/analyses/subspace``), never a
replacement for the exact grid.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Mapping, Sequence

import torch

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import answer_token_forms, single_token_id
from causalab.neural.activations.site_grids import SiteGrid
from causalab.neural.dataset import (
    forward_inputs,
    resolve_spec_positions,
    _batches,  # pyright: ignore[reportPrivateUsage]
    _check_pairwise_widths,  # pyright: ignore[reportPrivateUsage]
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.plan import CollectOp, GradientRequest, Plan, run_plan
from causalab.neural.site import backbone_has_edits, collect_ordered
from causalab.neural.specs import SiteSpec

logger = logging.getLogger(__name__)

__all__ = [
    "counterfactual_logit_diff_ids",
    "run_attribution_prescan",
    "select_top_k",
    "spearman_rank_correlation",
    "top_k_agreement",
]


def counterfactual_logit_diff_ids(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    causal_model: CausalModel,
    target_variables: Sequence[str] = ("raw_output",),
    label_variable: str = "raw_output",
) -> list[tuple[int, int]]:
    """Per-example ``(cf_answer_id, base_answer_id)`` token ids for the
    pre-scan's logit-difference readout.

    The counterfactual answer is the causal model's expected label under the
    interchange (:meth:`CausalModel.label_counterfactual_data` — the same
    reference the exact ``causal_label`` metric scores against); the base
    answer is the example's own expected output
    (``example["input"][label_variable]``, the ``compute_base_accuracy``
    convention). Both are resolved to their emitted single-token form via
    :func:`causalab.methods.metric.single_token_id`; a multi-token answer
    falls back to its **first** emitted token — the standard
    attribution-patching readout for multi-token answers (answers sharing a
    first token then contribute zero difference, weakening — never flipping
    — the signal).
    """
    labeled = causal_model.label_counterfactual_data(
        copy.deepcopy(list(dataset)),
        list(target_variables),
        label_variable=label_variable,
    )
    pairs: list[tuple[int, int]] = []
    for example, labeled_example in zip(dataset, labeled):
        expected = labeled_example["label"]
        cf_answer = (
            expected.get("string", expected) if isinstance(expected, dict) else expected
        )
        base_answer = example["input"][label_variable]
        pairs.append(
            (
                _readout_token_id(pipeline, str(cf_answer)),
                _readout_token_id(pipeline, str(base_answer)),
            )
        )
    return pairs


def _readout_token_id(pipeline: LMPipeline, answer: str) -> int:
    """The answer's emitted single-token id, else its first emitted token."""
    try:
        return single_token_id(pipeline, answer)
    except ValueError:
        first_form = answer_token_forms(answer)[0]  # the space-prefixed form
        return pipeline.tokenizer.encode(first_form, add_special_tokens=False)[0]


def _grid_specs(
    grid: Mapping[tuple[Any, ...], Sequence[Sequence[SiteSpec]]],
) -> Dict[tuple[Any, ...], SiteSpec]:
    """One raw-space spec per grid cell — the shape the pre-scan approximates.

    A grid cell (WU2 builder output, ``one_target_per_unit``) holds exactly
    one group with one spec; anything else (multi-group cells whose groups
    read *different* counterfactual inputs, multi-site joint patches) has no
    single ``grad · Δ`` per cell and is refused. Featurized specs are refused
    too: candidates are scored in raw activation space — for the DAS gate the
    featurizer is *trained after* the gate, so raw-space attribution is the
    honest upper-bound signal.
    """
    specs: Dict[tuple[Any, ...], SiteSpec] = {}
    for key, groups in grid.items():
        flat = [spec for group in groups for spec in group]
        if len(groups) != 1 or len(flat) != 1:
            raise ValueError(
                f"attribution pre-scan expects one unit per grid cell, but "
                f"cell {key!r} has {len(groups)} group(s) / {len(flat)} "
                "site(s) — score joint patches with the exact grid instead."
            )
        spec = flat[0]
        if not spec.fsite.featurizer.is_trivial() or spec.fsite.feature_ids is not None:
            raise NotImplementedError(
                f"attribution pre-scan scores raw activation space, but site "
                f"{spec.key!r} (cell {key!r}) carries a featurizer or "
                "feature_ids — prune the grid before attaching subspaces."
            )
        specs[key] = spec
    return specs


def _last_real_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of each row's last non-pad token (left- or right-padded)."""
    mask = attention_mask.int()
    return mask.shape[1] - 1 - mask.flip(1).argmax(dim=1)


def run_attribution_prescan(
    grid: SiteGrid,
    dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    batch_size: int,
    pair_token_ids: Sequence[tuple[int, int]],
) -> Dict[tuple[Any, ...], float]:
    """Approximate every cell's interchange effect with one backward per batch.

    Per batch: ONE fused, early-stopped forward over the counterfactual side
    collects source activations at every unique site; ONE forward+backward
    over the base side (a :class:`~causalab.neural.plan.Plan` carrying a
    :class:`~causalab.neural.plan.GradientRequest`) collects base activations
    and ``d(logit-diff)/d(activation)``. Each cell's score is then the mean
    over examples of ``grad · (a_source − a_base)`` at the cell's resolved
    positions — no per-cell forwards.

    Args:
        grid: :data:`~causalab.neural.activations.site_grids.SiteGrid` of
            single-spec cells (keys e.g. ``(layer, pos_id)``), as built by the
            WU2 builders under ``one_target_per_unit``.
        dataset: Counterfactual examples; group 0's counterfactual input is
            the source side (the grid convention).
        pipeline: Target pipeline (single-model; cross-model patching has no
            first-order approximation on one model's gradients). Its model
            must carry no persistent edits — the pre-scan approximates the
            unedited interchange the gated grid measures (refused loudly).
        batch_size: Examples per forward/backward.
        pair_token_ids: Per-example ``(cf_answer_id, base_answer_id)`` for the
            logit-difference readout (:func:`counterfactual_logit_diff_ids`).

    Returns:
        Dict mapping each target key to its mean **signed** approximate
        effect: positive = patching is predicted to move the output toward
        the counterfactual answer.
    """
    if len(pair_token_ids) != len(dataset):
        raise ValueError(
            f"pair_token_ids has {len(pair_token_ids)} entries for "
            f"{len(dataset)} examples — build them from the same dataset."
        )
    if not dataset:
        return {key: 0.0 for key in grid}

    specs = _grid_specs(grid)
    model = pipeline.model
    if backbone_has_edits(model):
        # Persistent edits (causalab.neural.persistent) compose into every
        # trace — mechanically the gradients would flow through the edited
        # forward just fine — but this pre-scan approximates (and its
        # golden-pinned quality diagnostics are calibrated on) interchange
        # over the UNEDITED model, the quantity the locate/DAS grids it
        # gates measure. An installed edit at pre-scan time is almost
        # certainly leftover state, so refuse loudly rather than silently
        # rank cells against a different model.
        raise ValueError(
            "attribution pre-scan refused: the pipeline's model carries "
            "persistent edits (causalab.neural.persistent). The pre-scan "
            "approximates interchange on the unedited model — uninstall the "
            "edits (uninstall_edits) before gating a grid, or run the exact "
            "grid directly."
        )

    # Deduplicate sites: grid cells share (component, layer) across positions,
    # so full-sequence reads amortize over every position at that layer. The
    # dedup key is structural (read off the engine site), not parsed from any
    # identifier string.
    site_key_of: Dict[tuple[Any, ...], str] = {}
    sites: Dict[str, Any] = {}
    for key, spec in specs.items():
        site = spec.fsite.site
        if isinstance(site, HeadSite):
            skey = f"{site.kind}|{site.layer}|h{site.head}"
        else:
            skey = f"{site.component}|{site.layer}"
        site_key_of[key] = skey
        sites.setdefault(skey, site)

    sums: Dict[tuple[Any, ...], float] = {key: 0.0 for key in specs}
    site_items = list(sites.items())

    for lo, hi in _batches(len(dataset), batch_size):
        batch = dataset[lo:hi]
        base_traces = [example["input"] for example in batch]
        cf_traces = [example["counterfactual_inputs"][0] for example in batch]
        base_encoding = pipeline.load(base_traces, return_offsets_mapping=True)
        cf_encoding = pipeline.load(cf_traces, return_offsets_mapping=True)

        # Source pass: one fused, early-stopped, grad-free forward.
        taps = [
            (
                (site.layer, site.forward_rank_on(model)),
                lambda m, site=site: site.read(m, None),
            )
            for _, site in site_items
        ]
        with torch.no_grad():
            source_values = collect_ordered(model, forward_inputs(cf_encoding), taps)
        source_of = {skey: value for (skey, _), value in zip(site_items, source_values)}

        # Base pass: one forward+backward through the Plan IR.
        cf_ids = torch.tensor([pair_token_ids[i][0] for i in range(lo, hi)])
        base_ids = torch.tensor([pair_token_ids[i][1] for i in range(lo, hi)])
        last = _last_real_positions(base_encoding["attention_mask"])

        def loss(
            values: Mapping[str, torch.Tensor],
            cf_ids: torch.Tensor = cf_ids,
            base_ids: torch.Tensor = base_ids,
            last: torch.Tensor = last,
        ) -> torch.Tensor:
            logits = values["base"]  # (batch, seq, vocab), graph intact
            rows = torch.arange(logits.shape[0], device=logits.device)
            step = logits[rows, last.to(logits.device)]
            diff = (
                step[rows, cf_ids.to(logits.device)]
                - step[rows, base_ids.to(logits.device)]
            )
            return diff.sum()  # per-example scalars; grads never mix rows

        plan = Plan(
            inputs={"base": forward_inputs(base_encoding)},
            ops=tuple(
                CollectOp("base", FeaturizedSite(site), key=skey)
                for skey, site in site_items
            ),
            save_logits=("base",),
            gradients=GradientRequest(
                loss=loss, wrt=tuple(skey for skey, _ in site_items)
            ),
        )
        result = run_plan(model, plan)

        # Score every cell off the shared reads.
        for key, spec in specs.items():
            skey = site_key_of[key]
            base_full = result.collects[skey].float()
            grad_full = result.gradients[skey].float()
            source_full = source_of[skey].float()
            base_rows = resolve_spec_positions(
                spec, base_traces, base_encoding, is_original=True
            )
            source_rows = resolve_spec_positions(
                spec, cf_traces, cf_encoding, is_original=False
            )
            _check_pairwise_widths(spec.key, base_rows, source_rows)
            for i in range(len(batch)):
                delta = source_full[i, source_rows[i]] - base_full[i, base_rows[i]]
                sums[key] += float((delta * grad_full[i, base_rows[i]]).sum())

    return {key: total / len(dataset) for key, total in sums.items()}


def select_top_k(
    scores: Mapping[tuple[Any, ...], float], k: int, *, by_abs: bool = False
) -> list[tuple[Any, ...]]:
    """The ``k`` best-scoring keys, descending (ties broken by key order).

    ``by_abs`` ranks by magnitude instead of signed value — for gates that
    treat a strong away-from-counterfactual effect as causally interesting.
    """
    if k <= 0:
        raise ValueError(f"top_k must be positive, got {k}")
    rank = (lambda kv: abs(kv[1])) if by_abs else (lambda kv: kv[1])
    ordered = sorted(scores.items(), key=rank, reverse=True)
    return [key for key, _ in ordered[:k]]


def top_k_agreement(
    approx: Mapping[tuple[Any, ...], float],
    exact: Mapping[tuple[Any, ...], float],
    k: int,
    *,
    by_abs: bool = False,
) -> float:
    """``|top-j(approx) ∩ top-j(exact)| / j`` over the keys both scored.

    The approximation-quality readout for goldens and reports. ``j`` is ``k``
    capped at **half** the both-scored keys (rounded up): at ``j = n`` the
    two top-sets are the whole domain and the overlap is vacuously 1.0 —
    which is exactly the regime a pruning gate lands in (the both-scored set
    *is* the approx top-k), so the cap keeps the metric able to miss in both
    the pruned and the full-grid (golden) regime. ``by_abs`` ranks the approx
    side by magnitude — pass the same convention the gate pruned with.
    """
    common = set(approx) & set(exact)
    if not common:
        return float("nan")
    j = min(k, (len(common) + 1) // 2) if len(common) > 1 else 1
    approx_top = set(select_top_k({c: approx[c] for c in common}, j, by_abs=by_abs))
    exact_top = set(select_top_k({c: exact[c] for c in common}, j))
    return len(approx_top & exact_top) / j


def spearman_rank_correlation(
    approx: Mapping[tuple[Any, ...], float],
    exact: Mapping[tuple[Any, ...], float],
) -> float:
    """Spearman rank correlation between the two scores' shared keys.

    Ties rank by key order (deterministic; adequate for a visibility metric).
    ``nan`` when fewer than two shared keys or either ranking is constant.
    """
    common = sorted(set(approx) & set(exact))
    if len(common) < 2:
        return float("nan")

    def ranks(values: list[float]) -> torch.Tensor:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = torch.zeros(len(values))
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out

    a = ranks([approx[c] for c in common])
    e = ranks([exact[c] for c in common])
    a = a - a.mean()
    e = e - e.mean()
    denom = float(a.norm() * e.norm())
    if denom == 0.0:
        return float("nan")
    return max(-1.0, min(1.0, float((a * e).sum()) / denom))
