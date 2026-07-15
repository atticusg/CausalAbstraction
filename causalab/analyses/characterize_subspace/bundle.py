"""Refined-hypothesis bundle assembly.

After the judge + reconcile path completes, this module writes the canonical
output bundle consumed downstream:

```
${out_dir}/
├── metadata.json
├── refined_hypothesis.json
├── evidence.safetensors + evidence.meta.json
├── reconciliation_trace.json
└── figures/{projection_distribution,step1_vs_webtext,
            projection_explorer}.html
```

The presence of ``refined_hypothesis.json`` is the downstream branching
signal. ``more_info_needed.json`` is written by ``reproduction.py`` on a
gate failure and is mutually exclusive with this bundle.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor

from causalab.analyses.characterize_subspace.schemas import (
    JudgeHypothesis,
    PeakRecord,
    ReconciliationResult,
    Significance,
    Step1Summary,
    WebtextEvidence,
)
from causalab.io.artifacts import save_tensors_with_meta

logger = logging.getLogger(__name__)


def _verdict_to_refined_form(verdict: str) -> str:
    """Map a reconcile verdict to the refined-hypothesis JSON's verdict field.

    All four verdicts are passed through verbatim; this indirection is here
    so the downstream consumer can rely on a stable enum even if the judge
    layer's vocabulary expands.
    """
    return verdict


def _evidence_summary(evidence: WebtextEvidence) -> dict[str, Any]:
    return {
        "corpus": evidence.corpus,
        "n_quantile_bins": len(evidence.quantile_bins),
        "n_topk_spans": len(evidence.topk_spans),
        "n_bottomk_spans": len(evidence.bottomk_spans),
        "stats": asdict(evidence.stats),
    }


def write_refined_hypothesis(
    out_dir: str,
    *,
    judge: JudgeHypothesis,
    reconciliation: ReconciliationResult,
    provided: Significance,
    evidence: WebtextEvidence,
    step1_summary: Step1Summary,
) -> str:
    """Write ``refined_hypothesis.json`` and return the path."""
    payload: dict[str, Any] = {
        "hypothesis_text": judge.hypothesis_text,
        "confidence": judge.confidence,
        "verdict": _verdict_to_refined_form(reconciliation.verdict),
        "provided_significance": asdict(provided),
        "evidence_summary": _evidence_summary(evidence),
        "step1_summary": asdict(step1_summary),
        "judge": {
            "model": judge.model,
            "framing": judge.framing,
            "supporting_spans": judge.supporting_spans,
        },
        "reconciliation": {
            "verdict": reconciliation.verdict,
            "rationale": reconciliation.rationale,
            "final_framing": reconciliation.final_framing,
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "refined_hypothesis.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote refined hypothesis to %s", out_path)
    return out_path


def write_reconciliation_trace(
    out_dir: str, reconciliation: ReconciliationResult
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reconciliation_trace.json")
    payload = {
        "verdict": reconciliation.verdict,
        "judge_hypothesis": reconciliation.judge_hypothesis,
        "provided_hypothesis": reconciliation.provided_hypothesis,
        "rationale": reconciliation.rationale,
        "final_framing": reconciliation.final_framing,
        "model": reconciliation.model,
        "iterations": [asdict(it) for it in reconciliation.iterations],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def write_evidence_artifact(
    out_dir: str,
    *,
    webtext_projections: Tensor,
    step1_projections: Tensor,
    evidence: WebtextEvidence,
    step1_summary: Step1Summary,
) -> tuple[str, str]:
    """Save projections as safetensors + meta JSON via the standard helper.

    The ``*_projections`` tensors are the **peak-token** ``k``-dim vectors
    (one per document, at its largest-subspace-norm token), not per-document
    means; the full ``(N, k)`` set is persisted uncapped.
    """
    tensors = {
        "webtext_projections": webtext_projections.detach().cpu().contiguous().float(),
        "step1_projections": step1_projections.detach().cpu().contiguous().float(),
    }
    meta = {
        "schema": "characterize_subspace.evidence",
        "representation": "peak_token_kdim",
        "webtext": {
            "corpus": evidence.corpus,
            "stats": asdict(evidence.stats),
            "quantile_bins": [
                {
                    "quantile": qb.quantile,
                    "projection_range": list(qb.projection_range),
                    "spans": [asdict(s) for s in qb.spans],
                }
                for qb in evidence.quantile_bins
            ],
            "topk_spans": [asdict(s) for s in evidence.topk_spans],
            "bottomk_spans": [asdict(s) for s in evidence.bottomk_spans],
        },
        "step1": asdict(step1_summary),
    }
    return save_tensors_with_meta(tensors, meta, out_dir, "evidence")


def _write_distribution_html(
    out_path: str,
    *,
    series: dict[str, list[float]],
    title: str,
    xaxis_title: str,
) -> None:
    """Render a small Plotly histogram comparing one or more series.

    Imported lazily so the analysis package import-time cost stays small —
    plotly is heavier than the rest of this module.
    """
    import plotly.graph_objects as go  # type: ignore[import]

    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(
            go.Histogram(
                x=values,
                name=name,
                opacity=0.6,
                nbinsx=40,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)


def _decile_stratified_indices(peak_value: Tensor, cap: int, n_bins: int) -> Tensor:
    """Return ≤``cap`` indices, evenly drawn across ``n_bins`` deciles.

    Used to bound the per-document JSON embedded in the interactive figures
    while preserving per-decile coverage. Order is preserved within deciles
    (sorted by projection); the returned indices are a subset of ``range(N)``.
    """
    n = int(peak_value.shape[0])
    if n <= cap:
        return torch.arange(n)
    sorted_idx = torch.argsort(peak_value)
    edges = torch.linspace(0, n, n_bins + 1, dtype=torch.long).tolist()
    per_bin = max(1, cap // n_bins)
    picks: list[int] = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        count = hi - lo
        if count <= 0:
            continue
        take = min(per_bin, count)
        sel = torch.linspace(0, count - 1, take).round().long()
        picks.extend(int(sorted_idx[lo + int(s)].item()) for s in sel)
    return torch.tensor(sorted(set(picks)), dtype=torch.long)


def write_figures(
    out_dir: str,
    *,
    webtext_peak_kdim: Tensor,
    webtext_peak_value: Tensor,
    webtext_records: list[PeakRecord],
    step1_peak_value: Tensor,
    hist_nbins: int,
    n_quantile_bins: int,
    max_docs_embedded: int,
) -> dict[str, str]:
    """Render the bundle's three figures and return ``{name: path}``.

    Two static overlay histograms (kept, now on max-token ``peak_value``) plus
    one interactive ``projection_explorer`` (clickable histogram linked to a
    side panel of context windows and a 3D PCA scatter). The per-document data
    embedded into the interactive figure is capped at ``max_docs_embedded`` via
    a decile-stratified even-sample (the full ``(N, k)`` peak tensor still
    persists to ``evidence.safetensors``).
    """
    figures_dir = os.path.join(out_dir, "figures")
    web_1d = webtext_peak_value.detach().cpu()
    s1_1d = step1_peak_value.detach().cpu()

    proj_html = os.path.join(figures_dir, "projection_distribution.html")
    _write_distribution_html(
        proj_html,
        series={"webtext": web_1d.tolist()},
        title="Webtext peak-token subspace-activation distribution",
        xaxis_title="Peak-token subspace activation ‖·‖₂",
    )
    overlay_html = os.path.join(figures_dir, "step1_vs_webtext.html")
    _write_distribution_html(
        overlay_html,
        series={"step1": s1_1d.tolist(), "webtext": web_1d.tolist()},
        title="Step-1 vs webtext peak-token subspace activation",
        xaxis_title="Peak-token subspace activation ‖·‖₂",
    )

    # Interactive figures — lazy import keeps the package import-time cost down.
    from causalab.analyses.characterize_subspace import figures

    keep = _decile_stratified_indices(web_1d, max_docs_embedded, n_quantile_bins)
    if keep.shape[0] < web_1d.shape[0]:
        logger.warning(
            "Embedding a decile-stratified sample of %d / %d webtext docs in the "
            "interactive figures (max_docs_embedded=%d); evidence.safetensors keeps "
            "all %d.",
            int(keep.shape[0]),
            int(web_1d.shape[0]),
            max_docs_embedded,
            int(web_1d.shape[0]),
        )
    capped_kdim = webtext_peak_kdim.detach().cpu()[keep]
    capped_value = web_1d[keep]
    capped_records = [webtext_records[int(i)] for i in keep.tolist()]

    explorer_html = os.path.join(figures_dir, "projection_explorer.html")
    figures.write_projection_explorer_html(
        explorer_html,
        peak_kdim=capped_kdim,
        peak_value=capped_value,
        records=capped_records,
        nbins=hist_nbins,
    )
    return {
        "projection_distribution": proj_html,
        "step1_vs_webtext": overlay_html,
        "projection_explorer": explorer_html,
    }


def write_metadata(
    out_dir: str,
    *,
    extra: dict[str, Any],
) -> str:
    """Standard metadata.json — analysis name, config snapshot, run info."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {"analysis": "characterize_subspace", **extra}
    out_path = os.path.join(out_dir, "metadata.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def write_bundle(
    out_dir: str,
    *,
    judge: JudgeHypothesis,
    reconciliation: ReconciliationResult,
    provided: Significance,
    evidence: WebtextEvidence,
    step1_summary: Step1Summary,
    webtext_peak_kdim: Tensor,
    webtext_peak_value: Tensor,
    webtext_records: list[PeakRecord],
    step1_peak_kdim: Tensor,
    step1_peak_value: Tensor,
    hist_nbins: int,
    n_quantile_bins: int,
    max_docs_embedded: int,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write the full bundle and return a map of artifact name → path.

    The caller composes a successful (judge + reconcile) path; failed-gate
    callers should use :func:`reproduction.write_more_info_needed` instead
    and skip this function. The ``*_peak_kdim`` tensors are the peak-token
    ``k``-dim vectors; ``*_peak_value`` are their dim-0 projections.
    """
    paths: dict[str, str] = {}
    paths["refined_hypothesis"] = write_refined_hypothesis(
        out_dir,
        judge=judge,
        reconciliation=reconciliation,
        provided=provided,
        evidence=evidence,
        step1_summary=step1_summary,
    )
    paths["reconciliation_trace"] = write_reconciliation_trace(out_dir, reconciliation)
    ev_st, ev_meta = write_evidence_artifact(
        out_dir,
        webtext_projections=webtext_peak_kdim,
        step1_projections=step1_peak_kdim,
        evidence=evidence,
        step1_summary=step1_summary,
    )
    paths["evidence_safetensors"] = ev_st
    paths["evidence_meta"] = ev_meta
    paths.update(
        write_figures(
            out_dir,
            webtext_peak_kdim=webtext_peak_kdim,
            webtext_peak_value=webtext_peak_value,
            webtext_records=webtext_records,
            step1_peak_value=step1_peak_value,
            hist_nbins=hist_nbins,
            n_quantile_bins=n_quantile_bins,
            max_docs_embedded=max_docs_embedded,
        )
    )
    paths["metadata"] = write_metadata(out_dir, extra=metadata_extra or {})
    return paths
