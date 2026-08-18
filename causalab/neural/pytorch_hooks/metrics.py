"""Metric lowering (spec §2.10): the closed kinds over one lm_head read.

Every kind is gather-then-reduce over the read's logits and dataset
columns. Per-example results come back as plain floats (or small
structures for ``top_k``), ready for a parquet table.

Token resolution follows the repo's space-prefixed-first rule: a column
value resolves to the single token of ``" " + s`` when that is one token,
else of ``s`` itself; anything multi-token refuses — a metric over a
multi-token answer is not expressible in v1's closed vocabulary and must
not silently score the first piece.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import MetricSpec

__all__ = ["column_token_id", "compute_metric"]


def column_token_id(tokenizer: Any, value: str) -> int:
    """The single token id a metric column value names (module docstring)."""
    # space-prefixed first (BPE families make " one" one token); the stripped
    # form covers sentencepiece families, whose "one" IS the ▁one piece
    candidates = (
        [value, value.lstrip(" ")] if value.startswith(" ") else [" " + value, value]
    )
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ProtocolError(
        "P2",
        f"metric column value {value!r} is not a single token under this "
        "tokenizer (tried space-prefixed first) — multi-token answers have no "
        "closed metric kind in v1",
    )


def _last_pos_logits(value: torch.Tensor) -> torch.Tensor:
    """A read at one position arrives as (batch, 1, vocab); squeeze it."""
    if value.dim() == 3:
        if value.shape[1] != 1:
            raise ProtocolError(
                "P2",
                f"metric read spans {value.shape[1]} positions — metrics reduce "
                "one position per example",
            )
        return value[:, 0, :]
    return value


def compute_metric(
    metric: MetricSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
) -> list[Any]:
    """One metric over one read's value, per example."""
    logits = _last_pos_logits(of_value).float()
    kind = str(metric.kind)

    def column(field: str) -> list[str]:
        name = str(metric.fields[field])
        out: list[str] = []
        for i, row in enumerate(rows):
            if name not in row:
                raise ProtocolError(
                    "P2", f"metric column {name!r} missing from dataset row {i}"
                )
            out.append(str(row[name]))
        return out

    if kind == "logit_diff":
        a_ids = [column_token_id(tokenizer, v) for v in column("a")]
        b_ids = [column_token_id(tokenizer, v) for v in column("b")]
        return [
            float(logits[i, a] - logits[i, b])
            for i, (a, b) in enumerate(zip(a_ids, b_ids))
        ]
    if kind == "token_logit":
        ids = [column_token_id(tokenizer, v) for v in column("token")]
        return [float(logits[i, t]) for i, t in enumerate(ids)]
    if kind == "cross_entropy":
        ids = [column_token_id(tokenizer, v) for v in column("target")]
        log_probs = torch.log_softmax(logits, dim=-1)
        return [float(-log_probs[i, t]) for i, t in enumerate(ids)]
    if kind == "kl":
        if target_value is None:
            raise ProtocolError("P2", "kl needs its target read's value")
        p = torch.log_softmax(logits, dim=-1)
        q = torch.log_softmax(_last_pos_logits(target_value).float(), dim=-1)
        kl = (p.exp() * (p - q)).sum(dim=-1)
        return [float(v) for v in kl]
    if kind == "match":
        expected = [column_token_id(tokenizer, v) for v in column("expected")]
        argmax = logits.argmax(dim=-1)
        return [float(int(argmax[i]) == t) for i, t in enumerate(expected)]
    if kind == "top_k":
        k = metric.fields["k"]
        assert isinstance(k, int)  # parse guarantees the shape
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(k, dim=-1)
        return [
            {
                "tokens": [tokenizer.decode([int(t)]) for t in top.indices[i]],
                "probs": [float(p) for p in top.values[i]],
            }
            for i in range(logits.shape[0])
        ]
    if kind == "class_probs":
        groups = metric.fields["groups"]
        if not isinstance(groups, Mapping):
            raise ProtocolError(
                "P2", "class_probs groups is a {name: [tokens]} mapping"
            )
        probs = torch.softmax(logits, dim=-1)
        group_ids = {
            name: [column_token_id(tokenizer, str(v)) for v in members]
            for name, members in groups.items()
        }
        return [
            {name: float(probs[i, ids].sum()) for name, ids in group_ids.items()}
            for i in range(logits.shape[0])
        ]
    raise ProtocolError("P4", f"unknown metric kind {kind!r}")
