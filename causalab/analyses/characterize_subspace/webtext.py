"""Broad-corpus activation collection and projection.

Streams text from a HuggingFace dataset (default FineWeb-Edu), forwards it
through the wrapped HF model, extracts activations at the configured
(layer, site), projects them through the loaded subspace, and assembles
:class:`WebtextEvidence` with quantile bins and top-/bottom-k spans.

Uses the HuggingFace model directly rather than the IntervenableModel path
(``causalab.neural.activations.collect``) because that one is task-bound,
and webtext has no task.

Each document is represented by its **peak-norm token**: the single (non-BOS)
token whose k-dim subspace projection has the largest Euclidean norm
(``‖proj‖₂`` over all subspace dimensions, not just the dim-0 coordinate). We
keep that token's subspace-activation norm (``peak_value``), its k-dim
projection vector (``peak_kdim``), and a ±W-token context *window* around it
with the peak token wrapped ``<<…>>``. The norm measures how strongly a token
fires in the subspace in any direction, so a concept living off the leading
axis is still captured; the BOS attention sink (huge subspace norm on every
doc) is excluded. This replaces the old per-document mean-pool, which washed
out the single token that actually fires on a sparse, concept-selective
subspace. Note the model is decoder-only, so the peak token's activation only
reflects tokens at or before its position; the right half of the displayed
window is human context that did not influence the projection.

Cache layout (v2): ``~/.cache/causalab/webtext/{key}/`` with
``peak_kdim.safetensors`` (``{peak_kdim:(N,k), peak_value:(N,)}``) and
``peaks.json`` (``list[{projection_value, peak_token_index, window_text}]``).
The key is sha256 over a ``v2|``-prefixed
``(corpus, split, n_tokens, model, layer, site, max_seq_len, projector_bytes)``
tuple; old v1 keys miss and recompute cleanly (old dirs are harmless orphans).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Iterable, cast

import torch
from torch import Tensor

from causalab.analyses.characterize_subspace.loading import SubspaceProjector, project
from causalab.analyses.characterize_subspace.schemas import (
    PeakRecord,
    ProjectionStats,
    QuantileBin,
    Span,
    WebtextEvidence,
)
from causalab.neural.pipeline import left_pad_position_ids

logger = logging.getLogger(__name__)


Site = str  # "residual" | "attn-out" | "mlp-out"


def _cache_root() -> str:
    return os.path.expanduser("~/.cache/causalab/webtext")


def _projector_fingerprint(projector: SubspaceProjector) -> str:
    """Stable hex digest of the rotation tensor for cache keying."""
    arr = (
        projector.rotation.detach()
        .to(torch.float32)
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(arr).hexdigest()[:16]


def _cache_key(
    *,
    corpus: str,
    split: str,
    n_tokens: int,
    model_name: str,
    layer: int,
    site: Site,
    max_seq_len: int,
    projector: SubspaceProjector,
) -> str:
    fp = _projector_fingerprint(projector)
    # ``v2|`` marks the max-token cache layout; old mean-pool keys miss and
    # recompute cleanly.
    parts = (
        f"v2|{corpus}|{split}|{n_tokens}|{model_name}|{layer}|{site}|{max_seq_len}|{fp}"
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:24]


def _stream_documents(
    corpus: str,
    split: str,
    n_tokens: int,
    *,
    tokenizer: Any,
    max_seq_len: int,
) -> Iterable[str]:
    """Yield documents from a streaming HF dataset until the token budget is hit.

    Counts against the model tokenizer, not the dataset's token count, so the
    budget is honest regardless of corpus tokenizer differences.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(corpus, split=split, streaming=True)
    budget = n_tokens
    for example in ds:
        if budget <= 0:
            break
        ex = cast("dict[str, Any]", example)
        text = str(ex.get("text") or ex.get("content") or "")
        if not text:
            continue
        ids = tokenizer(text, truncation=True, max_length=max_seq_len)["input_ids"]
        budget -= len(ids)
        yield text


def _residual_hidden_states(
    model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    layer: int,
) -> Tensor:
    """Return the residual stream at ``layer`` with shape ``(B, T, d_model)``.

    Uses HuggingFace's ``output_hidden_states=True`` interface. Index 0 of
    ``hidden_states`` is the embedding output; ``layer`` is interpreted as
    the residual stream *after* the layer-th block, matching causalab's
    convention.
    """
    # Supply position_ids for this plain forward. left_pad_position_ids is
    # padding-side-agnostic (cumsum over the mask), so this stays correct whether
    # the tokenizer right-pads (today's default here) or left-pads, and on an
    # absolute-position model where a wrong position corrupts the hidden states.
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=left_pad_position_ids(attention_mask),
            output_hidden_states=True,
            return_dict=True,
        )
    hs = out.hidden_states
    # hidden_states[0] = embeddings; hidden_states[i] = after block (i-1).
    # We index by "after block i" → hidden_states[i + 1].
    idx = layer + 1
    if idx >= len(hs):
        raise ValueError(f"Requested layer {layer} but model has {len(hs) - 1} layers.")
    return hs[idx]


def _sanitize_markers(text: str) -> str:
    """Neutralise any literal ``<<`` / ``>>`` so the peak marker is unambiguous."""
    return text.replace("<<", "‹‹").replace(">>", "››")


def _extract_window(
    input_ids_row: Tensor,
    attn_row: Tensor,
    peak_idx: int,
    *,
    tokenizer: Any,
    window: int,
) -> str:
    """Return a ±``window``-token context string around ``peak_idx``.

    The peak token is wrapped ``<<…>>``. The three pieces (left context, peak
    token, right context) are decoded *separately* to avoid BPE-merge
    ambiguity at the marker boundary, then any literal ``<<`` / ``>>`` already
    present in the source is sanitised so the marker is the only one. Padding
    is excluded by restricting to the row's true (non-padding) length.

    The displayed window is symmetric, but note that — for a decoder-only
    model — the peak token's activation only depends on tokens at or before
    ``peak_idx``; the right half is shown purely as human-readable context.
    """
    real_len = int(attn_row.sum().item())
    ids = input_ids_row[:real_len].tolist()
    peak_idx = max(0, min(peak_idx, real_len - 1)) if real_len else 0
    lo = max(0, peak_idx - window)
    hi = min(real_len, peak_idx + window + 1)
    left = _sanitize_markers(tokenizer.decode(ids[lo:peak_idx]))
    peak = _sanitize_markers(tokenizer.decode(ids[peak_idx : peak_idx + 1]))
    right = _sanitize_markers(tokenizer.decode(ids[peak_idx + 1 : hi]))
    return f"{left}<<{peak}>>{right}"


def collect_text_projections(
    texts: list[str],
    *,
    projector: SubspaceProjector,
    model: Any,
    tokenizer: Any,
    layer: int,
    site: Site,
    batch_size: int,
    max_seq_len: int,
    window: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, list[PeakRecord]]:
    """Project per-token activations and keep each document's peak-norm token.

    For every document the k-dim subspace projection is computed at every
    token; the token with the largest **Euclidean norm** over the whole
    subspace (``‖proj‖₂``, not just the dim-0 coordinate) is the document's
    *peak*. The norm measures how strongly a token fires in the subspace
    regardless of direction, so a concept living off the leading axis is still
    captured. The BOS token is excluded — it is an attention sink with a huge
    subspace norm on every document and would otherwise win the argmax
    everywhere. Padding is excluded too. Returns
    ``(peak_kdim, peak_value, records)`` where:

    - ``peak_kdim`` has shape ``(N, k)`` — the full k-dim projection vector at
      each document's peak token (persisted to the bundle / used for PCA).
    - ``peak_value`` has shape ``(N,)`` — the peak token's subspace-activation
      norm ``‖peak_kdim‖₂`` (non-negative); this is the per-document scalar the
      histogram bins and the judge sees.
    - ``records`` is row-aligned with the above and carries the peak token
      index and the ±``window``-token context window per document.

    Only ``site == "residual"`` is supported in this first cut. attn-out
    and mlp-out require forward hooks; raise NotImplementedError until a
    user needs them.
    """
    if site != "residual":
        raise NotImplementedError(
            f"site={site!r} is not yet supported. Only 'residual' is implemented "
            "in characterize_subspace.webtext; attn-out / mlp-out require "
            "forward hooks (TODO)."
        )
    if not texts:
        return torch.zeros(0, projector.k), torch.zeros(0), []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Many causal LMs ship without an explicit pad token. Use eos as a stand-in;
        # the attention mask still excludes padding from the argmax.
        pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError(
                "Tokenizer has no pad_token_id and no eos_token_id; cannot batch."
            )
    bos_id = getattr(tokenizer, "bos_token_id", None)

    kdim_chunks: list[Tensor] = []
    value_chunks: list[Tensor] = []
    records: list[PeakRecord] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        hidden = _residual_hidden_states(model, input_ids, attn_mask, layer=layer)
        proj = project(
            hidden, projector
        )  # (B, T, k) — project handles (...,d)->(...,k)
        # Peak token = largest Euclidean norm of the k-dim subspace projection
        # (the strongest-firing token in the subspace, in any direction), not
        # just the dim-0 coordinate. Exclude padding and the BOS attention sink
        # (huge subspace norm on every document) before the argmax.
        norms = proj.norm(dim=-1)  # (B, T)
        valid = attn_mask.bool()
        if bos_id is not None:
            valid = valid & (input_ids != bos_id)
        norms = norms.masked_fill(~valid, float("-inf"))
        peak = norms.argmax(dim=1)  # (B,)
        peak_kdim = proj.gather(
            1, peak[:, None, None].expand(-1, 1, proj.shape[-1])
        ).squeeze(1)  # (B, k)
        peak_val = peak_kdim.norm(dim=1)  # (B,) subspace-activation norm (>= 0)
        kdim_chunks.append(peak_kdim.detach().cpu().float())
        value_chunks.append(peak_val.detach().cpu().float())

        peak_cpu = peak.detach().cpu()
        ids_cpu = input_ids.detach().cpu()
        attn_cpu = attn_mask.detach().cpu()
        for i in range(len(batch_texts)):
            peak_idx = int(peak_cpu[i].item())
            records.append(
                PeakRecord(
                    projection_value=float(value_chunks[-1][i].item()),
                    peak_token_index=peak_idx,
                    window_text=_extract_window(
                        ids_cpu[i],
                        attn_cpu[i],
                        peak_idx,
                        tokenizer=tokenizer,
                        window=window,
                    ),
                )
            )

    peak_kdim_all = torch.cat(kdim_chunks, dim=0)
    peak_value_all = torch.cat(value_chunks, dim=0)
    return peak_kdim_all, peak_value_all, records


def _quantile_bin(
    projections_1d: Tensor,
    records: list[PeakRecord],
    *,
    n_bins: int,
    samples_per_bin: int,
) -> list[QuantileBin]:
    """Bin documents by peak projection value and sample within each bin.

    Spans carry the peak token's context window (``record.window_text``), not
    the full document.
    """
    n = projections_1d.shape[0]
    if n == 0:
        return []
    sorted_vals, sorted_idx = torch.sort(projections_1d)
    bins: list[QuantileBin] = []
    edges = torch.linspace(0, n, n_bins + 1, dtype=torch.long)
    for i in range(n_bins):
        lo, hi = int(edges[i].item()), int(edges[i + 1].item())
        if hi <= lo:
            continue
        bin_idx = sorted_idx[lo:hi]
        bin_vals = sorted_vals[lo:hi]
        # Even sample within the bin.
        step = max(1, (hi - lo) // samples_per_bin)
        picks = list(range(0, hi - lo, step))[:samples_per_bin]
        spans = [
            Span(
                text=records[int(bin_idx[p].item())].window_text,
                projection_value=float(bin_vals[p].item()),
            )
            for p in picks
        ]
        bins.append(
            QuantileBin(
                quantile=(i + 0.5) / n_bins,
                projection_range=(
                    float(bin_vals[0].item()),
                    float(bin_vals[-1].item()),
                ),
                spans=spans,
            )
        )
    return bins


def _topk_bottomk(
    projections_1d: Tensor,
    records: list[PeakRecord],
    *,
    topk: int,
    bottomk: int,
) -> tuple[list[Span], list[Span]]:
    """Return the strongest- and weakest-activating spans by subspace norm.

    ``projections_1d`` is the non-negative peak-token norm, so ``top_spans`` are
    the **strongest-firing** documents and ``bot_spans`` the **weakest-firing /
    most generic** ones. Direction is not represented, so the top set ranks by
    activation strength regardless of sign on any individual axis.
    """
    n = projections_1d.shape[0]
    if n == 0:
        return [], []
    topk = min(topk, n)
    bottomk = min(bottomk, n)
    top_vals, top_idx = torch.topk(projections_1d, topk, largest=True)
    bot_vals, bot_idx = torch.topk(projections_1d, bottomk, largest=False)
    top_spans = [
        Span(
            text=records[int(i.item())].window_text,
            projection_value=float(v.item()),
        )
        for v, i in zip(top_vals, top_idx)
    ]
    bot_spans = [
        Span(
            text=records[int(i.item())].window_text,
            projection_value=float(v.item()),
        )
        for v, i in zip(bot_vals, bot_idx)
    ]
    return top_spans, bot_spans


def collect_webtext_evidence(
    *,
    projector: SubspaceProjector,
    model: Any,
    tokenizer: Any,
    model_name: str,
    layer: int,
    site: Site,
    corpus: str,
    split: str,
    n_tokens: int,
    max_seq_len: int,
    batch_size: int,
    window: int,
    n_quantile_bins: int,
    samples_per_bin: int,
    topk: int,
    bottomk: int,
    device: torch.device | str,
    use_cache: bool = True,
) -> tuple[WebtextEvidence, Tensor, Tensor, list[PeakRecord]]:
    """Stream webtext, project, and assemble a :class:`WebtextEvidence`.

    Each document is reduced to its peak-norm token (see
    :func:`collect_text_projections`); binning is on that subspace-activation
    norm.

    Returns ``(evidence, peak_kdim, peak_value, records)`` where ``peak_kdim``
    has shape ``(N, k)`` (persisted to the bundle's ``evidence.safetensors``
    and used for the 3D-PCA figure), ``peak_value`` has shape ``(N,)``, and
    ``records`` is the row-aligned per-document peak metadata.
    """
    cache_dir = None
    if use_cache:
        key = _cache_key(
            corpus=corpus,
            split=split,
            n_tokens=n_tokens,
            model_name=model_name,
            layer=layer,
            site=site,
            max_seq_len=max_seq_len,
            projector=projector,
        )
        cache_dir = os.path.join(_cache_root(), key)
        cached_kdim = os.path.join(cache_dir, "peak_kdim.safetensors")
        cached_peaks = os.path.join(cache_dir, "peaks.json")
        if os.path.isfile(cached_kdim) and os.path.isfile(cached_peaks):
            logger.info("Reusing webtext cache (v2) at %s", cache_dir)
            from safetensors.torch import load_file

            loaded = load_file(cached_kdim)
            peak_kdim = loaded["peak_kdim"]
            peak_value = loaded["peak_value"]
            with open(cached_peaks, "r", encoding="utf-8") as fh:
                records = [PeakRecord(**rec) for rec in json.load(fh)]
            evidence = _assemble_evidence(
                peak_value=peak_value,
                records=records,
                corpus=corpus,
                n_quantile_bins=n_quantile_bins,
                samples_per_bin=samples_per_bin,
                topk=topk,
                bottomk=bottomk,
            )
            return evidence, peak_kdim, peak_value, records

    texts = list(
        _stream_documents(
            corpus, split, n_tokens, tokenizer=tokenizer, max_seq_len=max_seq_len
        )
    )
    if not texts:
        raise RuntimeError(
            f"No documents yielded from corpus {corpus!r} split={split!r}; "
            "check the dataset name and that it has a text field."
        )
    peak_kdim, peak_value, records = collect_text_projections(
        texts,
        projector=projector,
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        site=site,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        window=window,
        device=device,
    )

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        from safetensors.torch import save_file

        save_file(
            {"peak_kdim": peak_kdim, "peak_value": peak_value},
            os.path.join(cache_dir, "peak_kdim.safetensors"),
        )
        with open(os.path.join(cache_dir, "peaks.json"), "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in records], fh)
        logger.info("Cached webtext evidence (v2) to %s", cache_dir)

    evidence = _assemble_evidence(
        peak_value=peak_value,
        records=records,
        corpus=corpus,
        n_quantile_bins=n_quantile_bins,
        samples_per_bin=samples_per_bin,
        topk=topk,
        bottomk=bottomk,
    )
    return evidence, peak_kdim, peak_value, records


def _assemble_evidence(
    *,
    peak_value: Tensor,
    records: list[PeakRecord],
    corpus: str,
    n_quantile_bins: int,
    samples_per_bin: int,
    topk: int,
    bottomk: int,
) -> WebtextEvidence:
    proj_1d = peak_value
    n = proj_1d.shape[0]
    if n == 0:
        return WebtextEvidence(
            corpus=corpus,
            quantile_bins=[],
            topk_spans=[],
            bottomk_spans=[],
            stats=ProjectionStats(n_samples=0, mean=0.0, std=0.0, min=0.0, max=0.0),
        )
    stats = ProjectionStats(
        n_samples=int(n),
        mean=float(proj_1d.mean().item()),
        std=float(proj_1d.std().item()) if n > 1 else 0.0,
        min=float(proj_1d.min().item()),
        max=float(proj_1d.max().item()),
    )
    bins = _quantile_bin(
        proj_1d, records, n_bins=n_quantile_bins, samples_per_bin=samples_per_bin
    )
    top_spans, bot_spans = _topk_bottomk(proj_1d, records, topk=topk, bottomk=bottomk)
    return WebtextEvidence(
        corpus=corpus,
        quantile_bins=bins,
        topk_spans=top_spans,
        bottomk_spans=bot_spans,
        stats=stats,
    )
