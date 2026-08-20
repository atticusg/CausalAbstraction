"""Metric lowering (spec §2.10): the closed kinds over one lm_head read.

Every kind is gather-then-reduce over the read's logits and dataset
columns. Per-example results come back as plain floats (or small
structures for ``top_k``), ready for a parquet table.

Token resolution defaults to the repo's space-prefixed-first rule
(``token_form: "auto"``): a column value resolves to the single token of
``" " + s`` when that is one token, else of ``s`` itself; anything
multi-token refuses — a metric over a multi-token answer is not expressible
in v1's closed vocabulary and must not silently score the first piece.

``auto`` is right whenever the answer follows a space in the prompt
(weekdays, IOI names, MCQA letters), which is the common case, and it is
**wrong** whenever the answer does not. Punctuation is the canonical
counterexample: gpt2 encodes ``"?"`` as 30 and ``" ?"`` as 5633, both single
tokens, so ``auto`` scores 5633 while the model emits 30 — a ``match`` metric
then reads a flat 0.000 with no error anywhere. That is why a §2.10 metric
carries ``token_form``: set ``"bare"`` or ``"space_prefixed"`` to pin the
form instead of letting the tokenizer's vocabulary decide. Under ``auto``,
:func:`column_token_ids` warns once per column when both forms are single
tokens and disagree — exactly the condition under which it can be silently
wrong.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import MetricSpec

__all__ = ["column_token_id", "column_token_ids", "compute_metric"]


def _single_token(tokenizer: Any, text: str) -> int | None:
    """The token id of ``text`` when it encodes to exactly one token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def column_token_id(tokenizer: Any, value: str, *, token_form: str = "auto") -> int:
    """The single token id a metric column value names (module docstring).

    ``token_form`` is the §2.10 knob: ``"auto"`` (the default, and what every
    document without the key gets) keeps the space-prefixed-first rule,
    ``"space_prefixed"`` pins ``" " + s``, ``"bare"`` pins ``s`` with leading
    spaces stripped.
    """
    bare = value.lstrip(" ")
    spaced = " " + bare
    # space-prefixed first (BPE families make " one" one token); the stripped
    # form covers sentencepiece families, whose "one" IS the ▁one piece
    candidates = {
        "auto": (spaced, bare),
        "space_prefixed": (spaced,),
        "bare": (bare,),
    }[token_form]

    resolved: int | None = None
    for candidate in candidates:
        resolved = _single_token(tokenizer, candidate)
        if resolved is not None:
            break
    if resolved is None:
        tried = ", ".join(repr(c) for c in candidates)
        raise ProtocolError(
            "P2",
            f"metric column value {value!r} is not a single token under this "
            f"tokenizer (token_form={token_form!r}, tried {tried}) — multi-token "
            "answers have no closed metric kind in v1",
        )
    return resolved


def _ambiguous_under_auto(tokenizer: Any, value: str) -> tuple[int, int] | None:
    """``(space_prefixed_id, bare_id)`` when both forms are single tokens and
    they name *different* rows — the condition under which ``auto`` picks one
    for the author and can silently pick the wrong one."""
    bare = value.lstrip(" ")
    spaced_id = _single_token(tokenizer, " " + bare)
    bare_id = _single_token(tokenizer, bare)
    if spaced_id is None or bare_id is None or spaced_id == bare_id:
        return None
    return spaced_id, bare_id


def column_token_ids(
    tokenizer: Any,
    values: Sequence[str],
    *,
    token_form: str = "auto",
    where: str = "metric column",
) -> list[int]:
    """Resolve a whole metric column, warning **once** if ``auto`` had to guess.

    Per-value warnings would fire on half the IOI name vocabulary every run, so
    the check is aggregated here rather than in :func:`column_token_id`: one
    message per column, naming a few examples. Staying silent is what let a
    punctuation ``match`` read a flat 0.000 at all 48 layers of a real gpt2-xl
    scan — a wrong answer a pipeline gate scores as a dead stage, not an error.
    """
    ids = [column_token_id(tokenizer, v, token_form=token_form) for v in values]
    if token_form != "auto":
        return ids
    ambiguous = {
        v: pair
        for v in dict.fromkeys(values)
        if (pair := _ambiguous_under_auto(tokenizer, v))
    }
    if ambiguous:
        examples = ", ".join(
            f"{v!r} → {spaced} (space-prefixed) vs {bare} (bare)"
            for v, (spaced, bare) in list(ambiguous.items())[:3]
        )
        warnings.warn(
            f"{where}: {len(ambiguous)} of {len(set(values))} distinct answers are "
            f"ambiguous under this tokenizer — both forms are single tokens and "
            f"they name different rows ({examples}). token_form='auto' took the "
            "space-prefixed form; set the metric's token_form to 'bare' or "
            "'space_prefixed' to say which one the model actually emits.",
            UserWarning,
            stacklevel=2,
        )
    return ids


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
    # §2.10: how this metric's string answers become token ids. `auto` is the
    # space-prefixed-first default every pre-token_form document gets.
    form = str(metric.token_form)

    def token_ids(values: Sequence[str], field: str) -> list[int]:
        return column_token_ids(
            tokenizer, values, token_form=form, where=f"metric {kind}.{field}"
        )

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
        a_ids = token_ids(column("a"), "a")
        b_ids = token_ids(column("b"), "b")
        return [
            float(logits[i, a] - logits[i, b])
            for i, (a, b) in enumerate(zip(a_ids, b_ids))
        ]
    if kind == "token_logit":
        ids = token_ids(column("token"), "token")
        return [float(logits[i, t]) for i, t in enumerate(ids)]
    if kind == "cross_entropy":
        ids = token_ids(column("target"), "target")
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
        expected = token_ids(column("expected"), "expected")
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
            name: token_ids([str(v) for v in members], f"groups.{name}")
            for name, members in groups.items()
        }
        return [
            {name: float(probs[i, ids].sum()) for name, ids in group_ids.items()}
            for i in range(logits.shape[0])
        ]
    raise ProtocolError("P4", f"unknown metric kind {kind!r}")
