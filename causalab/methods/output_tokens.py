"""Resolve a variable's ``output_tokens`` declaration into the probability-path
score-token ids.

A causal model declares, per variable, the explicit surface forms of each value
(``{variable: {value: [form, ...]}}``; see
:func:`causalab.causal.causal_model.build_output_tokens`). This module holds the
*tokenizer-aware* resolvers over that declaration:

- :func:`form_groups` — the distinct, order-stable form groups (dedup: values
  that share a form group collapse, e.g. 28 ``(entity, group)`` tuples sharing 7
  entity forms → 7 groups).
- :func:`resolve_score_token_ids` — tokenizer-aware reduction to ``list[list[int]]``
  (reuses ``tokenize_variable_values``).
- :func:`form_group_labels` — one stripped label per distinct form group.
- :func:`form_group_values` — the first value (key) per distinct form group; the
  deduplicated value list the removed ``output_token_values`` used to hold.

The string match authority, :func:`causalab.causal.causal_model.derive_checker`,
lives in the lower ``causal/`` layer instead: it needs no tokenizer, so keeping
it there lets ``tasks/`` derive a checker without importing upward into
``methods/`` (#296 PR review). Part of #291.
"""

from __future__ import annotations

from typing import Any


def form_groups(var_map: dict[Any, list[str]]) -> list[list[str]]:
    """Distinct, order-stable form groups across a variable's values.

    ``var_map`` is ``{value: [form, ...]}``. Values whose form lists are
    identical collapse to a single group — this is where the dedup that
    ``output_token_values`` used to encode emerges for free (e.g. many
    ``(entity, group)`` tuples sharing one entity's forms).
    """
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for forms in var_map.values():
        key = tuple(forms)
        if key in seen:
            continue
        seen.add(key)
        groups.append(list(forms))
    return groups


def form_group_labels(var_map: dict[Any, list[str]]) -> list[str]:
    """One human-readable label per distinct form group (order-stable).

    The label is each group's first form, stripped — the surface string that
    names that score column. Aligns one-to-one with :func:`resolve_score_token_ids`'
    columns, so confusion / ground-truth plots and the answer-space relabel
    (#259) get a label per score token even when the score space is the deduped
    union of answer forms (e.g. entity_binding's 12 entity tokens), which the
    intervention variable's own values (the 2 positional classes) do not name.
    """
    return [group[0].strip() for group in form_groups(var_map)]


def form_group_values(var_map: dict[Any, list[str]]) -> list:
    """The first *value* of each distinct form group (order-stable).

    Where :func:`form_group_labels` returns each column's surface *string*, this
    returns each column's *value* — the key from ``var_map`` (e.g. a 2D
    coordinate tuple). This is the deduplicated value list the removed
    ``output_token_values`` used to hold (#291 phase 3): it aligns one-to-one
    with :func:`resolve_score_token_ids` / :func:`form_group_labels`, and is used
    where the value itself matters rather than its label — chiefly the grid-flow
    visualizations, which detect a 2D spatial layout from the value tuples.
    """
    values: list = []
    seen: set[tuple[str, ...]] = set()
    for value, forms in var_map.items():
        key = tuple(forms)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def resolve_score_token_ids(tokenizer, var_map: dict[Any, list[str]]):
    """Token ids for each distinct form group: ``list[list[int]]``.

    Keeps every single-token encoding across a group's forms (deduplicated);
    falls back to the first form's full sequence when no form is single-token
    (multi-token outputs). Delegates the reduction to ``tokenize_variable_values``
    so the probability path has exactly one tokenizer-aware code path.
    """
    from causalab.methods.metric import tokenize_variable_values

    groups = form_groups(var_map)
    # Each group already *is* the list of variants for one concept, so the
    # token_pattern is the identity over the group.
    return tokenize_variable_values(tokenizer, groups, lambda g: g)
