"""Empirical construction guards for the path-patching engine.

The analytic recipe is exact only for pre-LN additive-trunk decoders whose
capture points and block wiring the descriptor got right. None of that is
trusted: every engine runs these checks on its own caches at construction
and refuses to patch if any fails.

G1  additivity      final-position residual == embedding contribution +
                    sum of every branch's trunk contribution. Fails on
                    post-LN trunks and wrong capture points.
G2  branch wiring   re-evaluating each MLP branch on its derived input
                    residual reproduces the cached branch output. Fails on
                    a mis-declared block order (sequential vs parallel) and
                    on wrong pre-norm modules.
G3  patch-nothing   the reassembled tail on the untouched final residual
                    reproduces the model's own logits.
G4  patch-everything  patching every component's direct edge (embedding +
                    all heads + all MLPs) reconstructs the counterfactual
                    side's logits: the decomposition closes across inputs.

Default tolerances are per-dtype; measured errors are kept in
``engine.guard_report`` for reporting either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .edges import PathSpec

if TYPE_CHECKING:
    from .engine import PatchEngine

__all__ = ["GuardError", "run_construction_guards", "default_tolerances"]


class GuardError(RuntimeError):
    """A construction guard failed; the engine refuses to patch."""


def default_tolerances(model_dtype: torch.dtype) -> dict[str, float]:
    """Relative tolerances for G1/G2 (vs the residual's max magnitude) and
    absolute max-logit tolerances for G3/G4."""
    if model_dtype in (torch.float32, torch.float64):
        return {
            "additivity_rel": 1e-4,
            "branch_rel": 1e-4,
            "closure_logits": 2e-3,
        }
    # bf16 / fp16: the model's own trunk accumulates in half precision, so
    # a float32 re-sum legitimately differs at ~1e-2 relative scale.
    return {
        "additivity_rel": 3e-2,
        "branch_rel": 3e-2,
        "closure_logits": 1e-1,
    }


def _rel_err(err: torch.Tensor, ref: torch.Tensor) -> float:
    return (err.abs().max() / ref.abs().max().clamp_min(1e-12)).item()


@torch.no_grad()
def run_construction_guards(
    engine: "PatchEngine", tolerances: dict[str, float] | None = None
) -> dict[str, Any]:
    """Run G1-G4 on the engine's caches; raise :class:`GuardError` on any
    failure; return the measured errors."""
    tol = default_tolerances(engine.model_dtype)
    if tolerances:
        tol.update(tolerances)
    desc = engine.desc
    report: dict[str, Any] = {"tolerances": dict(tol)}
    failures: list[str] = []

    caches = {"clean": engine.clean, "cf": engine.cf}

    # ---- G1: additivity of trunk contributions ----
    for name, cache in caches.items():
        resid = cache.final_resid(engine.position).to(engine.device).float()
        total = engine._at(cache, "embed").clone()
        for layer in range(desc.n_layers):
            total = total + engine.attn_trunk_contribution(cache, layer)
            total = total + engine.mlp_trunk_contribution(cache, layer)
        err = _rel_err(total - resid, resid)
        report[f"G1_additivity_{name}"] = err
        if err > tol["additivity_rel"]:
            failures.append(
                f"G1 additivity ({name} cache): relative error {err:.2e} > "
                f"{tol['additivity_rel']:.0e}. The final-position residual is "
                f"not the sum of the captured contributions. Likely causes: a "
                f"post-LN trunk (this recipe is only exact for pre-LN "
                f"additive-trunk decoders) or wrong capture points in the "
                f"architecture descriptor."
            )

    # ---- G2: branch wiring (block order + pre-norm) ----
    worst = {"clean": 0.0, "cf": 0.0}
    for name, cache in caches.items():
        for layer in range(desc.n_layers):
            resid = engine.resid_for_mlp(cache, layer)
            recomputed = engine._mlp_branch_fn(layer, resid)
            cached = engine.mlp_trunk_contribution(cache, layer)
            err = _rel_err(
                recomputed - cached,
                cache.final_resid(engine.position).to(engine.device).float(),
            )
            worst[name] = max(worst[name], err)
        report[f"G2_branch_wiring_{name}"] = worst[name]
        if worst[name] > tol["branch_rel"]:
            failures.append(
                f"G2 branch wiring ({name} cache): worst per-layer relative "
                f"error {worst[name]:.2e} > {tol['branch_rel']:.0e}. "
                f"Re-evaluating an MLP branch on its derived input residual "
                f"does not reproduce the cached branch output. Likely causes: "
                f"the declared block order ({desc.block_order!r}) is wrong for "
                f"this model, or the descriptor's MLP pre-norm module is not "
                f"the one the block actually applies."
            )

    # ---- G3: patch-nothing closure ----
    zero = torch.zeros(
        engine.clean.n_examples, desc.d_model, device=engine.device
    )
    logits = engine.patched_logits(zero, PathSpec.cascade())
    err3 = (logits - engine.clean.logits[engine.position]).abs().max().item()
    report["G3_patch_nothing_max_logit_err"] = err3
    if err3 > tol["closure_logits"]:
        failures.append(
            f"G3 patch-nothing closure: max logit error {err3:.2e} > "
            f"{tol['closure_logits']:.0e}. The reassembled final norm + LM "
            f"head does not reproduce the model's own logits. Likely causes: "
            f"wrong final-norm/LM-head modules or a missing logit transform "
            f"(softcapping)."
        )

    # ---- G4: patch-everything closure ----
    everything = (
        "group",
        [("embed",)]
        + [("head", l, h) for l in range(desc.n_layers) for h in range(desc.n_heads)]
        + [("mlp", l) for l in range(desc.n_layers)],
    )
    logits = engine.patched_logits(everything, PathSpec.cascade())
    err4 = (logits - engine.cf.logits[engine.position]).abs().max().item()
    report["G4_patch_everything_max_logit_err"] = err4
    if err4 > tol["closure_logits"]:
        failures.append(
            f"G4 patch-everything closure: max logit error {err4:.2e} > "
            f"{tol['closure_logits']:.0e}. Patching every component's direct "
            f"edge does not reconstruct the counterfactual logits: the "
            f"residual decomposition does not close across inputs. Likely "
            f"causes: wrong block order, wrong capture points, or caches "
            f"built at inconsistent positions."
        )

    if failures:
        raise GuardError(
            "path-patching construction guards failed:\n- " + "\n- ".join(failures)
        )
    return report
