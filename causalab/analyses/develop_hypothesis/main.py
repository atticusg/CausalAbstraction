"""Certify how counterfactual datasets distinguish competing hypotheses (CPU).

develop_hypothesis answers: *do a session's counterfactual datasets actually
distinguish its competing causal-model hypotheses?* It loads the session-local
``models.py`` / ``counterfactuals.py`` scaffolds named by
``cfg.develop_hypothesis.hypotheses_dir``, runs the CPU distinguishability engine
in ``causalab.causal.causal_utils``, and writes the target-centric rate matrix
plus the always-confounded groups.

Like ``characterize_subspace``, this is a **task-less** analysis: its inputs are
the hand-authored causal model + counterfactual generators (the develop-causal-model
skill's scaffolds), not a runner-generated task dataset. It loads no neural model.

It reads, from the session modules:

* ``models``: ``MODELS`` (name -> CausalModel), ``DEFAULT_MODEL``,
  ``HYPOTHESES`` (name -> (model name, [targets])), optional ``TARGETS``.
* ``counterfactuals``: ``make_datasets(model, n=, seed=)``,
  ``random_pairs(model, n, seed=)``, optional ``DATASET_ROLES``.

The reference hypotheses ``"null"`` (intervene on nothing) and ``"all"``
(transplant the whole output) are injected automatically if absent.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from causalab.causal.causal_utils import distinguishability_report

ANALYSIS_NAME = "develop_hypothesis"


def load_session_modules(hypotheses_dir: str):
    """Import the session-local ``models`` / ``counterfactuals`` modules.

    The directory is prepended to ``sys.path`` (so the scaffolds may import each
    other by name, as the develop-causal-model contract allows) and the two
    modules are imported by their canonical names.
    """
    resolved = str(Path(hypotheses_dir).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    models = importlib.import_module("models")
    counterfactuals = importlib.import_module("counterfactuals")
    return models, counterfactuals


def build_hypotheses(models) -> dict[str, tuple[str, list[str]]]:
    """The session's hypotheses plus the auto-injected null/all references."""
    hyps = dict(models.HYPOTHESES)
    default = models.DEFAULT_MODEL
    hyps.setdefault("null", (default, []))
    hyps.setdefault("all", (default, ["raw_output"]))
    return hyps


def _resolve_output_dir(analysis_cfg: DictConfig) -> str:
    """Return the analysis output dir, creating it if needed."""
    out_dir = analysis_cfg.get("_output_dir") or os.path.join(
        os.getcwd(), ANALYSIS_NAME
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main(cfg: DictConfig) -> dict[str, Any]:
    acfg = cfg[ANALYSIS_NAME]
    seed = int(cfg.get("seed", 0))
    out_dir = _resolve_output_dir(acfg)

    models, counterfactuals = load_session_modules(acfg.hypotheses_dir)
    hyps = build_hypotheses(models)
    targets = list(getattr(models, "TARGETS", []))
    if not targets:
        targets = [n for n in hyps if n not in ("null", "all")]
    model = models.MODELS[models.DEFAULT_MODEL]

    datasets = counterfactuals.make_datasets(model, n=int(acfg.n), seed=seed)
    random_pairs = counterfactuals.random_pairs(model, int(acfg.random_n), seed=seed)

    core = distinguishability_report(
        models.MODELS, hyps, targets, datasets, random_pairs
    )
    report: dict[str, Any] = {
        "n": int(acfg.n),
        "random_n": int(acfg.random_n),
        "seed": seed,
        "hypotheses": {
            name: {"model": m, "targets": list(t)} for name, (m, t) in hyps.items()
        },
        "targets": targets,
        "dataset_roles": getattr(counterfactuals, "DATASET_ROLES", {}),
        **core,
    }

    out_path = os.path.join(out_dir, "distinguishability.json")
    Path(out_path).write_text(json.dumps(report, indent=2))
    _print(report, float(acfg.get("confound_threshold", 0.2)))
    print(f"\nWrote {out_path}")
    return report


def _print(report: dict, threshold: float) -> None:
    for ds_name, ds in report["datasets"].items():
        role = report["dataset_roles"].get(ds_name, {})
        tag = f" [{role.get('width', '?')}/{role.get('split', '?')}]" if role else ""
        print(f"\n=== {ds_name}{tag}  (n={ds['size']}) ===")
        for tgt, info in ds["per_target"].items():
            print(
                f"  target {tgt!r}: vs_null={info['vs_null']:.2f}  vs_all={info['vs_all']:.2f}"
            )
            for alt, r in sorted(
                info["alternatives"].items(), key=lambda kv: kv[1], reverse=True
            ):
                if alt in ("null", "all"):
                    continue
                flag = (
                    "   <-- confounded with target on this counterfactual dataset"
                    if r < threshold
                    else ""
                )
                print(f"      {r:.2f}  vs {alt}{flag}")

    print(
        "\n=== always-confounded groups (no sampled pair deconfounds them; "
        f"large random run, N={report['random_n']}) ==="
    )
    if report["always_confounded"]:
        for grp in report["always_confounded"]:
            print(
                f"  {{ {', '.join(grp)} }}  -- confounded everywhere; pick one representative"
            )
        print(
            "  (empirical: a rare pair that would deconfound them could be missed at this N)"
        )
    else:
        print(
            "  none -- every pair of hypotheses is deconfounded by some pair at this N"
        )
