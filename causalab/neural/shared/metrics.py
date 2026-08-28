"""Metric lowering (spec §2.10): the closed kinds over one read.

Every kind is gather-then-reduce over the read's value and dataset
columns. Per-example results come back as plain floats (or small
structures for ``top_k``), ready for a JSON metric table.

All but one kind name vocabulary entries, so validation binds them to an
``lm_head`` read and the value they reduce is a logit vector. ``top_k`` is
the exception: it ranks the entries of whatever axis its read has — a
vocabulary, a 4k-wide residual stream, a 100k-latent SAE code — and reduces
**where the rows are gathered**, which is the point of it (saving the whole
tensor just to argsort it later is the thing to avoid). Its mandatory ``by``
field says how to rank, and ``vocab_axis`` tells the reduction whether the
indices it found are token ids worth decoding.

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

📐 One limit of that warning, introduced by the transformers 5 bump: a
sentencepiece family that has dropped the legacy dummy prefix encodes ``" X"``
and ``"X"`` to the *same* id, so the two forms can never disagree there. On such
a tokenizer ``token_form`` has only one row to name, the warning is structurally
dark, and neither is a defect — but it does mean the punctuation trap above is a
BPE-family hazard, and a document cannot be checked against it by pinning
``token_form`` on a sentencepiece model.

``match`` is the one kind that can be told otherwise, and only explicitly
(§2.10): its ``expected`` column may hold a **list** of equivalent surface
forms (synonyms, casings), and ``"mode": "first_token"`` credits a form's
first token instead of demanding the form be one token. Both are
task-data decisions — the table says which forms count, the document says
whether a prefix counts — so neither can happen by accident.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    VOCAB_TOP_K_RANKING,
    WHOLE_WINDOW_METRIC_KINDS,
    MetricSpec,
)

__all__ = [
    "column_first_token_id",
    "column_token_id",
    "column_token_ids",
    "compute_metric",
    "compute_windowed_metric",
]


def _single_token(tokenizer: Any, text: str) -> int | None:
    """The token id of ``text`` when it encodes to exactly one token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def _candidates(value: str, token_form: str = "auto") -> tuple[str, ...]:
    """The surface forms to try, in order, for one metric answer.

    ``token_form`` is the §2.10 knob: ``auto`` tries the space-prefixed form
    first (BPE families make ``" one"`` one token) and falls back to the bare
    form (sentencepiece's ``"one"`` IS the ``\u2581one`` piece);
    ``space_prefixed`` and ``bare`` pin one form. A leading space in the
    authored value is normalized away rather than honored, so ``" ?"`` and
    ``"?"`` mean the same answer and only ``token_form`` decides the form.
    """
    bare = value.lstrip(" ")
    return {
        "auto": (" " + bare, bare),
        "space_prefixed": (" " + bare,),
        "bare": (bare,),
    }[token_form]


def column_token_id(tokenizer: Any, value: str, *, token_form: str = "auto") -> int:
    """The single token id a metric column value names (module docstring).

    ``token_form`` is the §2.10 knob: ``"auto"`` (the default, and what every
    document without the key gets) keeps the space-prefixed-first rule,
    ``"space_prefixed"`` pins ``" " + s``, ``"bare"`` pins ``s`` with leading
    spaces stripped.
    """
    candidates = _candidates(value, token_form)

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


def column_first_token_id(
    tokenizer: Any, value: str, *, token_form: str = "auto"
) -> int:
    """The first *content* token id of a value — ``match``'s ``first_token``
    mode.

    A single-token value resolves exactly as :func:`column_token_id` does, so
    ``first_token`` is a strict generalization of ``exact``. A multi-token
    value resolves to the first piece that carries text: a sentencepiece family
    can encode a leading space as its own ``▁`` piece, and crediting *that*
    would score every space-prefixed answer alike — the first piece an argmax
    can distinguish is the one after it, which is also what the model emits in
    context.

    📐 Which values trigger that is tokenizer- *and* version-dependent, so the
    skip is written as a property of the piece (does it decode to text?) rather
    than of a known value. Under transformers 4.x the tiny Llama tokenizer
    emitted the lone ``▁`` for any space-prefixed word (``" Thursday"`` →
    ``▁ Th urs day``); 5.16.1 dropped that legacy dummy prefix, so
    ``" Thursday"`` is now ``Th urs day`` — and the skip is what makes this
    function return the same id, ``Th``, across the bump. It is not dead code:
    5.16.1 still emits the lone ``▁`` whenever the first character has no
    merged ``▁X`` piece — digits, non-Latin scripts, emoji, ligatures
    (``" 3.14"`` → ``▁ 3 . 1 4``) — and byte-level BPE families still split a
    whitespace run off the front (gpt2 ``"  ?"`` → ``' '`` + ``' ?'``).

    What this cannot know is whether the table's answer space is
    first-token-distinct — two answers sharing a first piece would both score.
    That is a property of the dataset, checked where the dataset is built."""
    encoded = [
        tokenizer.encode(candidate, add_special_tokens=False)
        for candidate in _candidates(value, token_form)
    ]
    for ids in encoded:
        if len(ids) == 1:
            return int(ids[0])
    for ids in encoded:
        for token_id in ids:
            if tokenizer.decode([int(token_id)]).strip():
                return int(token_id)
    raise ProtocolError(
        "P2",
        f"metric column value {value!r} encodes to no content tokens under "
        "this tokenizer — nothing to compare an argmax against",
    )


def _last_pos_rows(value: torch.Tensor) -> torch.Tensor:
    """A read at one position arrives as (batch, 1, width); squeeze it.

    ``width`` is the vocabulary for an ``lm_head`` read and the site's own
    width otherwise — only ``top_k`` reduces the latter (every other kind is
    bound to a vocabulary projection by validation)."""
    if value.dim() == 3:
        if value.shape[1] != 1:
            raise ProtocolError(
                "P2",
                f"metric read spans {value.shape[1]} positions — metrics reduce "
                "one position per example",
            )
        return value[:, 0, :]
    return value


def _top_k(
    metric: MetricSpec,
    dense: torch.Tensor,
    tokenizer: Any,
    *,
    vocab_axis: bool,
) -> list[dict[str, Any]]:
    """``top_k`` over one read's rows — the reduction that happens **where the
    rows are gathered**, so a 100k-latent SAE code never reaches disk.

    ``by`` (mandatory, §2.10) is the ranking rule, and it is the author's call
    because only the author knows what the axis is: a vocabulary projection
    has no meaningful negative entries, a residual stream and a signed feature
    code do.

    The emitted columns have **fixed identities**, so a column never means one
    thing in one document and another in the next; a column is absent rather
    than reinterpreted:

    ==========  =====================================  ====================
    column      meaning                                emitted when
    ==========  =====================================  ====================
    ``indices`` index along the read's last axis       always
    ``tokens``  that index decoded as a token string   the read is a plain lm_head tap
    ``values``  the **raw** read value at that index   always
    ``probs``   softmax probability over the vocab     ``by == "prob"``
    ==========  =====================================  ====================

    ``values`` is always the raw value — a logit under ``by: "prob"``, not the
    probability — so a downstream reader never has to know the ranking rule to
    know what it is holding. The normalized number lives in its own column.
    """
    k = metric.fields["k"]
    assert isinstance(k, int)  # parse guarantees the shape
    by = str(metric.fields["by"])
    width = int(dense.shape[-1])
    if k < 1 or k > width:
        raise ProtocolError(
            "P2",
            f"top_k asks for k={k} of a read {width} wide — k must be in [1, width]",
        )
    if by == VOCAB_TOP_K_RANKING:
        # validation binds `prob` to an lm_head read (§2.10): a softmax across
        # neurons or latents normalizes over an axis that is not an event space
        scores = torch.softmax(dense, dim=-1)
    elif by == "abs_value":
        scores = dense.abs()
    else:
        scores = dense
    top = scores.topk(k, dim=-1)
    out: list[dict[str, Any]] = []
    for i in range(dense.shape[0]):
        indices = [int(j) for j in top.indices[i]]
        entry: dict[str, Any] = {"indices": indices}
        if vocab_axis:
            entry["tokens"] = [tokenizer.decode([j]) for j in indices]
        entry["values"] = [float(dense[i, j]) for j in indices]
        if by == VOCAB_TOP_K_RANKING:
            entry["probs"] = [float(p) for p in top.values[i]]
        out.append(entry)
    return out


def compute_metric(
    metric: MetricSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
    vocab_axis: bool = True,
) -> list[Any]:
    """One metric over one read's value, per example.

    ``vocab_axis`` says whether the read's last axis is the vocabulary — i.e.
    whether it is a plain ``lm_head`` tap, with no featurizer or ``dims``
    taking the value out of token-id space (:func:`~causalab.protocol.schema
    .metric_reads_vocabulary`). Every kind but ``top_k`` is bound to a
    vocabulary projection by validation, so the default is ``True``; ``top_k``
    is the one kind that also runs over a residual stream, an MLP activation
    or a featurizer's latents, and it needs to know because a token id is
    worth decoding and a neuron index is not."""
    # `dense` is the read's value at the addressed position, (batch, width).
    # Every kind but `top_k` is bound to an lm_head read, so for those it is
    # the vocabulary projection and reads as `logits` below.
    dense = _last_pos_rows(of_value).float()
    logits = dense
    kind = str(metric.kind)
    # §2.10: how this metric's string answers become token ids. `auto` is the
    # space-prefixed-first default every pre-token_form document gets.
    token_form = str(metric.token_form)

    def token_ids(values: Sequence[str], field: str) -> list[int]:
        return column_token_ids(
            tokenizer,
            values,
            token_form=token_form,
            where=f"metric {kind}.{field}",
        )

    def raw_column(field: str) -> list[Any]:
        name = str(metric.fields[field])
        out: list[Any] = []
        for i, row in enumerate(rows):
            if name not in row:
                raise ProtocolError(
                    "P2", f"metric column {name!r} missing from dataset row {i}"
                )
            out.append(row[name])
        return out

    def column(field: str) -> list[str]:
        return [str(value) for value in raw_column(field)]

    def form_groups(field: str) -> list[list[str]]:
        """One row's expected forms: a list column is a group of equivalent
        surface forms, a scalar is a group of one (§2.10)."""
        groups: list[list[str]] = []
        for i, value in enumerate(raw_column(field)):
            forms = [str(v) for v in value] if isinstance(value, list) else [str(value)]
            if not forms:
                raise ProtocolError(
                    "P2",
                    f"metric column {metric.fields[field]!r} is an empty form "
                    f"group on row {i} — nothing to match against",
                )
            groups.append(forms)
        return groups

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
        q = torch.log_softmax(_last_pos_rows(target_value).float(), dim=-1)
        kl = (p.exp() * (p - q)).sum(dim=-1)
        return [float(v) for v in kl]
    if kind == "match":
        # `mode` decides whether a form's first token counts (§2.10);
        # `token_form` decides which surface form resolves. Independent knobs.
        mode = str(metric.fields.get("mode", "exact"))
        resolve = column_first_token_id if mode == "first_token" else column_token_id
        argmax = logits.argmax(dim=-1)
        return [
            float(
                int(argmax[i])
                in {resolve(tokenizer, f, token_form=token_form) for f in forms}
            )
            for i, forms in enumerate(form_groups("expected"))
        ]
    if kind == "top_k":
        return _top_k(metric, dense, tokenizer, vocab_axis=vocab_axis)
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
    if kind == "decode":
        raise ProtocolError(
            "P2",
            "'decode' reduces the tokens a decode produced, so it binds to a "
            "read in the continuation frame (§2.3) — validation refuses it "
            "anywhere else",
        )
    raise ProtocolError("P4", f"unknown metric kind {kind!r}")


def compute_windowed_metric(
    metric: MetricSpec,
    windows: Sequence[torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_windows: Sequence[torch.Tensor] | None = None,
    generated_ids: Sequence[Sequence[int]] | None = None,
    vocab_axis: bool = True,
) -> list[list[Any]]:
    """One metric over a read that addresses **several** positions per row.

    ``windows[i]`` is example ``i``'s value at the positions it addresses,
    ``(positions_i, vocab)`` — empty when the row addressed none, which in
    the continuation frame is a result (§2.3), not a misalignment. Returns
    the same shape: one list of values per example.

    Every ``distribution`` kind reduces **per position**, and does so
    through :func:`compute_metric` on the flattened positions — one
    implementation of the kinds, not two. ``ids`` kinds never look at
    ``windows`` at all: they consume ``generated_ids``, which is why a text
    probe obliges no vocabulary projection (§8).
    """
    kind = str(metric.kind)
    if kind in WHOLE_WINDOW_METRIC_KINDS:
        if generated_ids is None:
            raise ProtocolError(
                "P2", f"metric kind {kind!r} needs the decode's token ids"
            )
        if kind == "decode":
            return [
                [tokenizer.decode(list(ids))] if len(ids) else []
                for ids in generated_ids
            ]
        raise ProtocolError("P4", f"unhandled whole-window metric kind {kind!r}")

    counts = [int(window.shape[0]) for window in windows]
    if not any(counts):
        return [[] for _ in windows]
    flat = torch.cat([w for w in windows if w.shape[0]], dim=0)
    flat_rows = [rows[i] for i, count in enumerate(counts) for _ in range(count)]
    flat_target = None
    if target_windows is not None:
        target_counts = [int(w.shape[0]) for w in target_windows]
        if target_counts != counts:
            raise ProtocolError(
                "P2",
                f"kl compares reads addressing different position counts "
                f"({counts} vs {target_counts}) — a comparison needs a "
                "position-for-position pairing",
            )
        flat_target = torch.cat([w for w in target_windows if w.shape[0]], dim=0)
    values = compute_metric(
        metric,
        flat,
        flat_rows,
        tokenizer,
        target_value=flat_target,
        vocab_axis=vocab_axis,
    )
    out: list[list[Any]] = []
    cursor = 0
    for count in counts:
        out.append(values[cursor : cursor + count])
        cursor += count
    return out
