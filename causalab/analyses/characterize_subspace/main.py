"""Top-level orchestrator for the subspace-characterization analysis.

Pipeline:

1. Load subspace from ``.safetensors`` (adaptive shape inspection).
2. Load HF model + tokenizer, project phase-1 dataset, run reproduction gate.
   On gate failure, write ``more_info_needed.json`` and stop.
3. Collect broad-corpus webtext evidence via :mod:`webtext`.
4. Derive an independent hypothesis (judge call 1) — forbidden-substring
   guard ensures the rendered prompt is free of significance content.
5. Reconcile against the provided significance (judge call 2).
6. Assemble the output bundle.

Reads ``cfg.characterize_subspace.*`` only; this analysis intentionally
ignores ``cfg.task`` because phase-1 datasets here are arbitrary text
collections, not task-rendered prompts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, cast, get_args

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.analyses.characterize_subspace import bundle as bundle_mod
from causalab.analyses.characterize_subspace.judge import (
    derive_hypothesis,
    reconcile_hypotheses,
)
from causalab.analyses.characterize_subspace.loading import (
    SubspaceProjector,
    load_subspace,
)
from causalab.methods.llm_judge import Provider, resolve_credentials
from causalab.neural.pipeline import resolve_device
from causalab.analyses.characterize_subspace.reproduction import (
    ReproductionReport,
    run_gate,
    write_more_info_needed,
)
from causalab.analyses.characterize_subspace.schemas import (
    PeakRecord,
    Significance,
    Span,
)
from causalab.analyses.characterize_subspace.subspace_builder import (
    resolve_subspace_artifact,
)
from causalab.analyses.characterize_subspace.webtext import (
    collect_text_projections,
    collect_webtext_evidence,
)

logger = logging.getLogger(__name__)


ANALYSIS_NAME = "characterize_subspace"


def _load_step1_texts(step1_dataset: str) -> tuple[str, list[str]]:
    """Resolve the phase-1 dataset reference to a ``(name, texts)`` tuple.

    Supported shapes:

    - Path to a JSON file containing ``list[str]`` of documents.
    - Path to a JSON file containing ``{"texts": [str, ...]}``.

    Task-name resolution (rendering a task's prompts) is intentionally not
    supported in this first cut — phase-1 datasets are commonly free-form
    text collections, and routing through ``resolve_task`` would require
    target-variable wiring that's orthogonal to this analysis's goal.
    """
    if not os.path.isfile(step1_dataset):
        raise FileNotFoundError(
            f"step1_dataset must be a path to a JSON file. Not found: {step1_dataset!r}. "
            "Task-name resolution is not yet supported."
        )
    with open(step1_dataset, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        texts = [str(t) for t in payload if isinstance(t, str)]
    elif isinstance(payload, dict) and "texts" in payload:
        texts = [str(t) for t in payload["texts"] if isinstance(t, str)]
    else:
        raise ValueError(
            "step1_dataset JSON must be list[str] or {'texts': list[str]}. "
            f"Got top-level {type(payload).__name__}."
        )
    if not texts:
        raise ValueError(f"step1_dataset {step1_dataset!r} contained no strings.")
    return os.path.basename(step1_dataset), texts


def _load_hf_model(model_name: str, *, device: str | None, dtype: str | None) -> Any:
    """Load a raw HF model + tokenizer for activation collection.

    Bypasses ``causalab.io.pipelines.load_pipeline`` because the IntervenableModel
    wrapper is task-bound and unnecessary for the residual-stream forward
    pass this analysis needs.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    resolved_device = resolve_device(device)
    torch_dtype: torch.dtype | None
    if dtype is None:
        torch_dtype = None
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype!r}.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore[assignment]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    )
    model.to(torch.device(resolved_device))  # type: ignore[arg-type]
    model.eval()
    return model, tokenizer, resolved_device


def _significance_from_cfg(cfg_sig: DictConfig) -> Significance:
    return Significance(
        hypothesis_text=cfg_sig.get("hypothesis_text"),
        figure_path=cfg_sig.get("figure_path"),
        topology_description=cfg_sig.get("topology_description"),
    )


def _example_spans_from_records(
    records: list[PeakRecord],
    *,
    n: int = 8,
) -> list[Span]:
    """Pick ``n`` spans spread across the peak-projection range for ``Step1Summary``.

    Sorts records by peak projection value and samples evenly, carrying each
    document's peak-token context window. Empty input → empty list.
    """
    total = len(records)
    if total == 0:
        return []
    order = sorted(range(total), key=lambda i: records[i].projection_value)
    n = min(n, total)
    picks = [int(i) for i in torch.linspace(0, total - 1, n).round().tolist()]
    return [
        Span(
            text=records[order[p]].window_text,
            projection_value=records[order[p]].projection_value,
        )
        for p in picks
    ]


def _resolve_output_dir(analysis_cfg: DictConfig) -> str:
    """Return the analysis output dir, creating it if needed."""
    out_dir = analysis_cfg.get("_output_dir") or os.path.join(
        os.getcwd(), "characterize_subspace"
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _resolve_provider(judge_cfg: DictConfig) -> Provider:
    """Validate and return the judge provider from config."""
    provider_str = str(judge_cfg.get("provider", "openrouter"))
    valid_providers = get_args(Provider)
    if provider_str not in valid_providers:
        raise ValueError(
            f"Invalid judge.provider {provider_str!r}; "
            f"expected one of {valid_providers}."
        )
    return cast(Provider, provider_str)


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the subspace-characterization analysis end-to-end."""
    analysis = cfg[ANALYSIS_NAME]
    out_dir = _resolve_output_dir(analysis)
    logger.info("characterize_subspace output dir: %s", out_dir)

    # ---- 0. fail fast on missing judge credentials --------------------------
    # Validate the judge API key BEFORE the expensive model load + webtext
    # collection (issue #221). resolve_credentials raises a clear RuntimeError
    # naming the unset key, so a misconfigured run dies in seconds rather than
    # after minutes of GPU + streaming work. The runner has already loaded any
    # project .env (causalab.runner.env.load_project_dotenv) by this point.
    provider = _resolve_provider(analysis.judge)
    resolve_credentials(provider)

    # ---- 1. load subspace ---------------------------------------------------
    sub_cfg = analysis.subspace
    artifact_path, build_provenance = resolve_subspace_artifact(
        artifact=sub_cfg.get("artifact"),
        source=sub_cfg.get("source"),
        out_dir=out_dir,
    )
    projector: SubspaceProjector = load_subspace(
        artifact_path,
        k_features_hint=sub_cfg.get("k_features_hint", "auto"),
    )

    # ---- 2. load model + phase-1 dataset ------------------------------------
    model_name = sub_cfg.model
    layer = int(sub_cfg.layer)
    site = sub_cfg.get("site", "residual")
    device_cfg = cfg.model.get("device") if "model" in cfg else None
    dtype_cfg = cfg.model.get("dtype") if "model" in cfg else None
    model, tokenizer, device = _load_hf_model(
        model_name, device=device_cfg, dtype=dtype_cfg
    )

    dataset_name, step1_texts = _load_step1_texts(sub_cfg.step1_dataset)

    webtext_cfg = analysis.webtext
    batch_size = int(webtext_cfg.batch_size)
    max_seq_len = int(webtext_cfg.max_seq_len)
    context_window = int(webtext_cfg.get("context_window", 25))
    histogram_nbins = int(webtext_cfg.get("histogram_nbins", 40))
    max_docs_embedded = int(webtext_cfg.get("max_docs_embedded", 5000))
    n_quantile_bins = int(webtext_cfg.n_quantile_bins)

    logger.warning(
        "characterize_subspace now represents each document by its peak-token "
        "subspace-activation norm (issue #210; the token with the largest "
        "Euclidean norm over the whole subspace, BOS excluded), not a "
        "per-document mean. Reproduction-gate thresholds were tuned for "
        "mean-pooled distributions and may need recalibration."
    )

    step1_peak_kdim, step1_peak_value, step1_records = collect_text_projections(
        step1_texts,
        projector=projector,
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        site=site,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        window=context_window,
        device=device,
    )

    # ---- 3. reproduction gate ----------------------------------------------
    significance = _significance_from_cfg(analysis.significance)
    metric_cfg = analysis.get("metrics", {})
    report: ReproductionReport = run_gate(
        projections=step1_peak_kdim,
        example_spans=_example_spans_from_records(step1_records),
        significance=significance,
        dataset_name=dataset_name,
        pass_threshold=float(metric_cfg.get("reproduction_pass_threshold", 0.5)),
    )

    if not report.passed:
        missing_path = write_more_info_needed(
            out_dir, report=report, significance=significance
        )
        logger.warning("Reproduction gate failed; wrote %s and stopping.", missing_path)
        return {
            "passed": False,
            "more_info_needed": missing_path,
            "output_dir": out_dir,
            "metrics": [asdict(m) for m in report.metrics],
            "diagnostic": report.diagnostic,
        }

    # ---- 4. webtext collection ----------------------------------------------
    (
        evidence,
        webtext_peak_kdim,
        webtext_peak_value,
        webtext_records,
    ) = collect_webtext_evidence(
        projector=projector,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        layer=layer,
        site=site,
        corpus=webtext_cfg.corpus,
        split=webtext_cfg.get("split", "train"),
        n_tokens=int(webtext_cfg.n_tokens),
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        window=context_window,
        n_quantile_bins=n_quantile_bins,
        samples_per_bin=int(webtext_cfg.get("samples_per_bin", 3)),
        topk=int(webtext_cfg.topk),
        bottomk=int(webtext_cfg.bottomk),
        device=device,
        use_cache=bool(webtext_cfg.get("cache", True)),
    )

    # ---- 5. derive + reconcile ----------------------------------------------
    # provider was resolved and credential-checked in step 0.
    judge_cfg = analysis.judge
    framings = list(judge_cfg.get("framings", ["default"]))
    derived = derive_hypothesis(
        evidence=evidence,
        step1_summary=report.step1_summary,
        framing=framings[0],
        model=str(judge_cfg.model),
        provider=provider,
        max_tokens=int(judge_cfg.get("max_tokens", 4096)),
        forbidden_substrings=significance.non_empty_values(),
    )

    reconciliation = reconcile_hypotheses(
        judge=derived,
        provided=significance,
        framings=framings,
        model=str(judge_cfg.model),
        provider=provider,
        max_tokens=int(judge_cfg.get("max_tokens", 4096)),
        max_iterations=int(judge_cfg.get("max_reconciliation_iterations", 3)),
    )

    # ---- 6. assemble bundle -------------------------------------------------
    paths = bundle_mod.write_bundle(
        out_dir,
        judge=derived,
        reconciliation=reconciliation,
        provided=significance,
        evidence=evidence,
        step1_summary=report.step1_summary,
        webtext_peak_kdim=webtext_peak_kdim,
        webtext_peak_value=webtext_peak_value,
        webtext_records=webtext_records,
        step1_peak_kdim=step1_peak_kdim,
        step1_peak_value=step1_peak_value,
        hist_nbins=histogram_nbins,
        n_quantile_bins=n_quantile_bins,
        max_docs_embedded=max_docs_embedded,
        metadata_extra={
            "model": model_name,
            "layer": layer,
            "site": site,
            "subspace_artifact": artifact_path,
            "subspace_source": build_provenance,
            "metrics": [asdict(m) for m in report.metrics],
            "skipped_metrics": report.skipped_metrics,
            "config_snapshot": cast(
                dict[str, Any], OmegaConf.to_container(analysis, resolve=True)
            ),
        },
    )

    return {
        "passed": True,
        "verdict": reconciliation.verdict,
        "refined_hypothesis_path": paths["refined_hypothesis"],
        "output_dir": out_dir,
        "paths": paths,
    }
