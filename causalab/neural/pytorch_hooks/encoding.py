"""The position frame: one tokenization per batch, positions born padded.

The backend's ``PositionFrame`` (spec §2.3, §8) is the padded batch the
model actually runs, plus what position resolution needs to address it:
pad side (always left here), per-row content offsets, offset mappings for
char→token resolution, and per-row prefix lengths (0 without a chat
template; the chat seam is the field, not a code path yet).

Position rules implemented against this frame (spec §2.3, §6.1):

* ``{"index": n}`` — ``n < 0`` counts from the end of the row's real
  tokens; ``n ≥ 0`` counts from the row's content start (past padding and
  any chat prefix).
* ``{"variable": "x"}`` — all tokens overlapping the char span of the
  row's value for ``x``. The value comes from the dataset row: for a text
  column ``<col>`` the sibling ``<col>_variables`` mapping (aligned
  per-element for list columns), else a plain column named ``x``. The
  value must occur exactly once in the row's text — zero or several
  occurrences refuse loudly rather than address the wrong tokens.
* ``{"column": "c"}`` — the same token run, from the row's top-level
  column ``c`` only (never the ``<col>_variables`` sibling). The column is
  a property of the *row*, so it resolves to the same string whichever
  role reads it; that is what makes it the spelling for values a task
  computes per row (§2.3).
* ``{"span": [a, b]}`` — the content-frame window ``[a, b)``.
* ``scope`` — the index/span interpreted inside the anchor's token run;
  ``relative_to`` — an index offset from the run (``+1`` = first token
  after it, ``-1`` = last token before it; ``0`` is refused). The anchor is
  ``{"variable": …}`` or ``{"column": …}``.

Every resolved index is bounds-checked in the padded frame — a stale or
impossible position must fail here as a legible error, never reach a
gather (the #176 failure class the old resolver guarded the same way).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import PositionSpec, concrete_int

__all__ = ["EncodedBatch", "encode", "resolve_position"]


@dataclasses.dataclass(frozen=True)
class EncodedBatch:
    """One left-padded batch plus its position frame."""

    texts: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offset_mapping: tuple[tuple[tuple[int, int], ...], ...]
    prefix_lengths: tuple[int, ...]  # chat-prefix token counts; 0 = plain text

    @property
    def padded_len(self) -> int:
        return int(self.input_ids.shape[1])

    def content_start(self, row: int) -> int:
        """First real token of ``row`` in the padded frame, past any prefix."""
        mask = self.attention_mask[row].int()
        return int(torch.argmax(mask).item()) + self.prefix_lengths[row]

    def position_ids(self) -> torch.Tensor:
        """Left-pad position ids: ``cumsum(mask) - 1``, clamped at 0 — the
        plain-forward convention (RoPE is shift-blind, absolute embeddings
        like GPT-2's ``wpe`` are not, so this must always be passed)."""
        return (self.attention_mask.cumsum(dim=1) - 1).clamp(min=0)


def encode(
    tokenizer: Any, texts: Sequence[str], *, device: str = "cpu"
) -> EncodedBatch:
    """Tokenize one batch with the backend's single convention: left
    padding, special tokens as the tokenizer defines them, offset mapping
    kept for char→token position resolution."""
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=True,
    )
    return EncodedBatch(
        texts=tuple(texts),
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        offset_mapping=tuple(
            tuple((int(a), int(b)) for a, b in row) for row in enc["offset_mapping"]
        ),
        prefix_lengths=tuple(0 for _ in texts),
    )


_LIST_FIELD = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]$")


def select_field(row: Mapping[str, Any], field: str) -> Any:
    """Apply a data-role ``field`` selector (§2.2): a column name, with
    ``[j]`` indexing list-valued columns."""
    match = _LIST_FIELD.match(field)
    if match is None:
        if field not in row:
            raise ProtocolError(
                "P2", f"row has no column {field!r} (has {sorted(row)})"
            )
        return row[field]
    column, index = match.group(1), int(match.group(2))
    values = row.get(column)
    if not isinstance(values, list) or index >= len(values):
        raise ProtocolError("P2", f"column {column!r} has no element [{index}]")
    return values[index]


def variable_value(row: Mapping[str, Any], field: str, variable: str) -> str:
    """The row's value for a prompt variable, for the text selected by
    ``field`` (module docstring: ``<col>_variables`` sibling first, plain
    column fallback)."""
    match = _LIST_FIELD.match(field)
    column = match.group(1) if match else field
    sibling = row.get(f"{column}_variables")
    if match and isinstance(sibling, list):
        index = int(match.group(2))
        if index < len(sibling) and isinstance(sibling[index], Mapping):
            sibling = sibling[index]
        else:
            sibling = None
    if isinstance(sibling, Mapping) and variable in sibling:
        return str(sibling[variable])
    if variable in row:
        return str(row[variable])
    raise ProtocolError(
        "P2",
        f"no value for prompt variable {variable!r}: neither {column}_variables "
        f"nor a {variable!r} column exists in the dataset row",
    )


def column_value(row: Mapping[str, Any], column: str) -> str:
    """The row's value for a ``column`` position (§2.3) — a top-level column
    only, never the per-role ``<field>_variables`` sibling, so the same
    reference resolves to the same string whichever role reads it."""
    if column not in row:
        raise ProtocolError(
            "P2",
            f"position column {column!r} is not a column of the dataset row "
            f"(has {sorted(row)})",
        )
    value = row[column]
    if not isinstance(value, str):
        raise ProtocolError(
            "P2",
            f"position column {column!r} holds {type(value).__name__}, not a "
            "string — v1 column positions resolve a substring of the row's "
            "text (§2.3)",
        )
    return value


def _variable_token_run(batch: EncodedBatch, row: int, value: str) -> list[int]:
    """The padded-frame token indices covering the (unique) occurrence of
    ``value`` in the row's text, via the offset mapping ((0, 0) entries are
    specials/padding and never match)."""
    text = batch.texts[row]
    starts = [m.start() for m in re.finditer(re.escape(value), text)]
    if len(starts) != 1:
        raise ProtocolError(
            "P2",
            f"prompt variable value {value!r} occurs {len(starts)} times in "
            f"{text!r} — position resolution needs exactly one occurrence",
        )
    lo, hi = starts[0], starts[0] + len(value)
    run = [
        idx
        for idx, (a, b) in enumerate(batch.offset_mapping[row])
        if not (a == 0 and b == 0) and a < hi and b > lo
    ]
    if not run:
        raise ProtocolError(
            "P2", f"variable value {value!r} maps to no tokens in row {row}"
        )
    return run


def resolve_position(
    spec: PositionSpec,
    batch: EncodedBatch,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> list[int]:
    """Resolve one position spec for one row into padded-frame indices."""
    padded = batch.padded_len
    start = batch.content_start(row)

    def check(indices: list[int]) -> list[int]:
        bad = [
            i for i in indices if not start - batch.prefix_lengths[row] <= i < padded
        ]
        if bad:
            raise ProtocolError(
                "P2",
                f"resolved position(s) {bad} out of bounds for row {row} "
                f"(content [{start}, {padded}) in the padded frame) — refusing "
                "rather than addressing the wrong token",
            )
        return indices

    def row_value(name: str, *, from_column: bool) -> str:
        """The row's string for an anchor or an anchor-free reference —
        a top-level column (``column``) or a per-role prompt variable
        (``variable``), §2.3."""
        if dataset_row is None:
            raise ProtocolError("P2", "variable/column positions need a dataset row")
        if from_column:
            return column_value(dataset_row, name)
        if field is None:
            raise ProtocolError("P2", "variable positions need a dataset row")
        return variable_value(dataset_row, field, name)

    anchor_run: list[int] | None = None
    if spec.scope is not None or spec.relative_to is not None:
        anchor_name = str(spec.scope or spec.relative_to)
        anchor_value = row_value(
            anchor_name, from_column=spec.anchor_source == "column"
        )
        anchor_run = _variable_token_run(batch, row, anchor_value)

    if spec.variable is not None:
        return check(
            _variable_token_run(
                batch, row, row_value(str(spec.variable), from_column=False)
            )
        )

    if spec.column is not None:
        return check(
            _variable_token_run(
                batch, row, row_value(str(spec.column), from_column=True)
            )
        )

    if spec.index is not None:
        n = concrete_int(spec.index, "position index")
        if spec.relative_to is not None:
            assert anchor_run is not None
            if n == 0:
                raise ProtocolError(
                    "P2", "relative_to index 0 is ambiguous — use scope"
                )
            target = anchor_run[-1] + n if n > 0 else anchor_run[0] + n
            return check([target])
        if spec.scope is not None:
            assert anchor_run is not None
            if not -len(anchor_run) <= n < len(anchor_run):
                raise ProtocolError(
                    "P2",
                    f"index {n} outside the {len(anchor_run)}-token variable window",
                )
            return check([anchor_run[n]])
        if n < 0:
            return check([padded + n])
        return check([start + n])

    assert spec.span is not None
    span = spec.span
    if not isinstance(span, tuple) or len(span) != 2:
        raise ProtocolError("P2", f"span is not concrete: {span!r}")
    a, b = (int(v) for v in span)
    if spec.scope is not None:
        assert anchor_run is not None
        window = anchor_run[a:b]
        if not window:
            raise ProtocolError(
                "P2", f"span [{a}, {b}) is empty inside the variable window"
            )
        return check(window)
    if a < 0 or b <= a:
        raise ProtocolError("P2", f"span [{a}, {b}) is not a forward window")
    return check(list(range(start + a, start + b)))
