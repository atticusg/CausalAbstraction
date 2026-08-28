"""The component shape table, and the three defects it was built to close.

:func:`~causalab.protocol.registry.component_shape` is the one description
everything downstream derives from — the feature width, whether a featurizer may
attach, how many heads ``head`` selects among, and the native↔contract
conversion. Before it, those four answers lived in four places and had drifted:

* the head bound read ``info.num_heads`` no matter the component (§2.2);
* ``stream`` parsed as an integer while the only code that reads it wants a
  string, so no document could use the field (§2.1);
* ``router_scores`` accepted a basis-fitting featurizer over an axis that is a
  per-token ranking (D3);
* the MoE branch outputs were declared hidden-wide in one table and flat in
  another, and ``mlp_activation``'s width on the GPT-2 family came from a config
  key the modeling code ignores.

The last two were found by the width check itself, which is the argument for
having one.
"""

from __future__ import annotations

import dataclasses

import pytest

from causalab.protocol.canonical import canonicalize
from causalab.protocol.errors import ParseError, ValidationError
from causalab.protocol.registry import (
    ModelInfo,
    component_shape,
    component_width,
    get_model_info,
    register_model,
)
from causalab.protocol.plan import COMPONENT_RANK
from causalab.protocol.schema import (
    COMPONENTS,
    DEPRECATED_COMPONENTS,
    STREAMS,
    parse_document,
)

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit

#: A GQA model with every optional field populated, so one table can exercise
#: the whole vocabulary. ``num_kv_heads`` is deliberately half ``num_heads`` and
#: ``head_dim`` is deliberately *not* ``hidden_size // num_heads``: both are true
#: of the Qwen3.6 target, and both are the kind of coupling a table that assumed
#: them would silently get wrong.
GQA = ModelInfo(
    key="test/gqa",
    hidden_size=64,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    head_dim=16,
    intermediate_size=128,
    vocab_size=1000,
    num_experts=32,
    num_experts_per_tok=4,
    shared_expert_intermediate_size=48,
)


@pytest.fixture(autouse=True)
def _registered() -> None:
    """The document-level tests below name these keys, and the registry is the
    protocol layer's only source of static config (§6: never the network)."""
    register_model(GQA)
    register_model(dataclasses.replace(GQA, key="test/moe"))


#: ``component -> (width, head_space, is_feature_space)``. One row per name in
#: the vocabulary; the completeness test below fails if a component is added
#: without one.
EXPECTED: dict[str, tuple[int | None, int | None, bool]] = {
    "input_ids": (None, None, False),
    "embeddings": (64, None, True),
    "block_input": (64, None, True),
    "attention_input_norm": (64, None, True),
    "attention_probs": (None, 8, False),
    "attention_query_pre_rope": (128, 8, True),
    "attention_key_pre_rope": (64, 4, True),
    "attention_value_states": (64, 4, True),
    "attention_gate": (128, 8, True),
    "attention_premix": (128, 8, True),
    "attention_output": (64, None, True),
    "block_mid": (64, None, True),
    "mlp_input_norm": (64, None, True),
    "mlp_input": (64, None, True),
    "mlp_activation": (128, None, True),
    "router_logits": (32, None, True),
    "router_scores": (4, None, True),
    "expert_idx": (4, None, False),
    "expert_output": (64, None, True),
    "routed_output": (64, None, True),
    "shared_expert_gate_proj": (48, None, True),
    "shared_expert_up_proj": (48, None, True),
    "shared_expert_activation": (48, None, True),
    "shared_expert_output": (64, None, True),
    "shared_expert_gate": (1, None, True),
    "mlp_output": (64, None, True),
    "block_output": (64, None, True),
    "ln_final": (64, None, True),
    "lm_head": (1000, None, True),
}


def test_every_component_in_the_vocabulary_has_a_shape() -> None:
    """A new component must declare its axes, not inherit a default.

    ``(batch, position, hidden)`` is the right answer often enough that a
    default would be silently right most of the time and silently wrong for
    exactly the interior taps round 2 adds."""
    assert set(EXPECTED) == set(COMPONENTS)


@pytest.mark.parametrize("component", sorted(EXPECTED))
def test_the_shape_table(component: str) -> None:
    width, head_space, is_feature_space = EXPECTED[component]
    shape = component_shape(GQA, component)
    assert shape.width == width
    assert shape.head_space == head_space
    assert shape.is_feature_space is is_feature_space


@pytest.mark.parametrize("component", sorted(EXPECTED))
def test_width_and_the_shape_agree(component: str) -> None:
    """``component_width`` is a reading of the shape, so it must refuse exactly
    when the shape says there is nothing to measure."""
    shape = component_shape(GQA, component)
    if shape.is_feature_space:
        assert component_width(GQA, component) == shape.width
    else:
        with pytest.raises(ValidationError):
            component_width(GQA, component)


def test_the_pattern_is_the_only_shape_with_two_position_axes() -> None:
    """Which is what makes every one of the executor's refusals about it
    derivable rather than written by hand."""
    without_contract = [
        c for c in COMPONENTS if not component_shape(GQA, c).has_contract_form
    ]
    assert without_contract == ["attention_probs"]


# --------------------------------------------------------------------------- #
# §2.2 — the head bound is the component's, not the model's
# --------------------------------------------------------------------------- #


def test_head_on_a_component_with_no_head_axis_is_refused(env) -> None:
    """🐞 This used to validate and then be silently dropped by the backend —
    the same class as the ``expert`` sub-axis ``_moe_site`` refuses by name."""
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["sites"]["tgt"]["head"] = 2  # block_output has no head axis
    with pytest.raises(ValidationError, match="has no head axis"):
        canonicalize(raw, env)


def test_head_is_bounded_by_the_components_own_head_space(env) -> None:
    """And the bound is quoted with the shape it came from."""
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["sites"]["tgt"] = {"component": "attention_premix", "layer": 3, "head": 8}
    with pytest.raises(ValidationError, match="out of range"):
        canonicalize(raw, env)


def test_a_head_inside_the_components_head_space_is_accepted(env) -> None:
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["sites"]["tgt"] = {"component": "attention_premix", "layer": 3, "head": 7}
    assert canonicalize(raw, env)["sites"]["tgt"]["head"] == 7


def test_a_kv_space_head_bound_is_narrower_than_the_query_space_one() -> None:
    """The latent half of the defect, pinned before round 2 walks into it.

    📐 ``head_space`` is the *component's*, so a KV-space component under GQA
    admits half as many heads as a query-space one. Bounding the first by the
    second does not raise — python slices past the end silently — it yields an
    empty slice: a read of ``(b, n_pos, 0)`` and a write that changes nothing.
    """
    query_space = component_shape(GQA, "attention_premix").head_space
    assert query_space == GQA.num_heads == 8
    assert GQA.num_kv_heads == 4  # what round 2's `v`, `k` and `k_pre_rope` use
    # the bound that would have been applied to them, and the one that will be
    assert query_space != GQA.num_kv_heads


# --------------------------------------------------------------------------- #
# §2.1 — `stream` is a field a document can finally use
# --------------------------------------------------------------------------- #


def _doc_with_stream(value: object) -> dict[str, object]:
    raw = base_doc()
    raw["sites"]["tgt"]["stream"] = value
    return raw


@pytest.mark.parametrize("stream", STREAMS)
def test_a_document_may_name_the_stream_it_means(stream: str) -> None:
    """🐞 Every one of these was rejected at parse before round 2: the field
    parsed with ``_scalar_int``, while ``sites._check_stream`` reads it only
    when it is a *string* (``ModelBundle.stream_at`` returns one). The two
    halves of the feature each rejected what the other accepted, so no document
    could reach the check at all."""
    doc = parse_document(_doc_with_stream(stream))
    assert doc.sites["tgt"].stream == stream


def test_an_integer_stream_is_refused_rather_than_ignored() -> None:
    """The failure the ``expert`` refusal's comment already names: ``stream: 0``
    parsed, was stored, and was then silently skipped by the only code that
    reads the field."""
    with pytest.raises(ParseError, match="sites.tgt.stream"):
        parse_document(_doc_with_stream(0))


def test_a_stream_outside_the_vocabulary_is_refused() -> None:
    with pytest.raises(ParseError, match="sites.tgt.stream"):
        parse_document(_doc_with_stream("attention"))


# --------------------------------------------------------------------------- #
# D3 — a ranking axis is not a basis
# --------------------------------------------------------------------------- #


def _moe_doc_with_featurizer(kind: str) -> dict[str, object]:
    raw = base_doc()
    raw["model"]["key"] = "test/moe"
    raw["sites"]["tgt"] = {"component": "router_scores", "layer": 3}
    raw["featurizers"] = {"f": {"kind": kind, "k": 2}}
    raw["reads"]["v_cf"]["featurizer"] = "f"
    raw["writes"]["patch"]["featurizer"] = "f"
    return in_order(raw)


@pytest.mark.parametrize("kind", ["subspace", "pca"])
def test_a_basis_featurizer_on_a_ranking_axis_is_refused(env, kind: str) -> None:
    """📐 Dimensionally the axis has a width, so this used to be accepted — but
    column *k* is the *k*-th ranked expert, a different expert for different
    tokens. A subspace fitted across positions is fitted across a basis that is
    itself shuffled per position."""
    with pytest.raises(ValidationError, match="per-token ranking"):
        canonicalize(_moe_doc_with_featurizer(kind), env)


def test_a_per_column_featurizer_on_a_ranking_axis_still_works(env) -> None:
    """Only the kinds that *fit a basis* are refused. ``standardize`` computes a
    mean and a scale per column, which is meaningful on a ranking: 'how large is
    the top-ranked expert's score, typically'."""
    raw = _moe_doc_with_featurizer("standardize")
    raw["featurizers"]["f"] = {"kind": "standardize"}
    raw = in_order(raw)
    assert canonicalize(raw, env)["featurizers"]["f"]["width"] == 4


def test_a_plain_read_of_a_ranking_axis_is_untouched(env) -> None:
    """The refusal is about fitting a basis, not about reading the tensor."""
    raw = base_doc()
    raw["model"]["key"] = "test/moe"
    raw["sites"]["tgt"] = {"component": "router_scores", "layer": 3}
    assert canonicalize(raw, env)["sites"]["tgt"]["component"] == "router_scores"


# --------------------------------------------------------------------------- #
# the two defects the width check found on its own
# --------------------------------------------------------------------------- #


def test_the_moe_branch_outputs_are_hidden_wide_and_flat() -> None:
    """🐞 The width table said hidden-wide and the layout table said flat, in
    two different files, and the tensors are both — but nothing compared a
    declared width against a real tensor, so a shape that was only half right
    would have gone on being half right."""
    for component in ("routed_output", "shared_expert_output"):
        shape = component_shape(GQA, component)
        assert shape.width == GQA.hidden_size
        assert shape.flat_batch
        assert shape.native_rank == 2


def test_gpt2s_mlp_width_comes_from_n_inner() -> None:
    """🐞 ``GPT2Config`` spells the MLP's inner width ``n_inner`` and the block
    computes ``n_inner if n_inner is not None else 4 * hidden``
    (transformers ``models/gpt2/modeling_gpt2.py:250``). Reading
    ``intermediate_size`` instead reported 37 for the 128-wide MLP of
    ``hf-internal-testing/tiny-random-gpt2``, whose config carries that key even
    though nothing in the model reads it."""
    assert get_model_info("gpt2").intermediate_size == 4 * 768


# --------------------------------------------------------------------------- #
# D1/D6 — the rename, and the rank table it renumbered
# --------------------------------------------------------------------------- #


def test_the_retired_spelling_still_loads() -> None:
    """The one-release alias: a document written against the old vocabulary is
    not a hard error."""
    raw = base_doc()
    raw["sites"]["tgt"] = {"component": "attention_value", "layer": 3}
    assert parse_document(raw).sites["tgt"].component == "attention_premix"


def test_the_alias_folds_at_parse_so_nothing_downstream_sees_two_names(
    env,
) -> None:
    """Both spellings canonicalize identically, and therefore digest
    identically — the alias is a courtesy at the door, not a second vocabulary
    the tables have to know about."""
    old, new = base_doc(), base_doc()
    old["sites"]["tgt"] = {"component": "attention_value", "layer": 3}
    new["sites"]["tgt"] = {"component": "attention_premix", "layer": 3}
    assert canonicalize(old, env) == canonicalize(new, env)


def test_the_retired_name_is_not_reused_by_anything() -> None:
    """The rule that makes the alias safe.

    An alias that *redirects* is fine; one that *rebinds* would let a document
    written against the old vocabulary load and silently mean a different
    tensor. Round 2 introduces the real value vectors under their own name for
    exactly this reason (nnterp#51 is the same mistake, made after the fact).
    """
    for retired in DEPRECATED_COMPONENTS:
        assert retired not in COMPONENTS


def test_every_component_has_a_rank() -> None:
    """``COMPONENT_RANK`` drives group elision, and a component missing from it
    would silently sort as the deepest tap."""
    assert set(COMPONENT_RANK) == set(COMPONENTS)


def test_the_rank_table_is_written_in_forward_order() -> None:
    """It is read as a story about a block, so its source order and its values
    must agree — a table that sorts differently than it reads is one nobody
    checks against the architecture."""
    ranks = list(COMPONENT_RANK.values())
    assert ranks == sorted(ranks)


def test_the_attention_band_has_room_for_round_twos_insertions() -> None:
    """D6: renumber once, then never again. Each reserved slot named in
    ``plan.py`` must still be free, or a later PR is a re-pin rather than an
    insertion."""
    taken = set(COMPONENT_RANK.values())
    #: The slots `plan.py` still holds open, updated as each PR claims one —
    #: which is the point: a PR that takes a slot has to say so here.
    still_reserved = (200, 210, 220, 240, 350)
    assert not taken & set(still_reserved)
    # and each lands between the tap before it and the one after
    assert COMPONENT_RANK["attention_input_norm"] < min(still_reserved)
    assert max(still_reserved) < COMPONENT_RANK["attention_output"]
    # round 2.2 claimed its four, at the numbers plan.py reserved for them
    assert COMPONENT_RANK["attention_query_pre_rope"] == 160
    assert COMPONENT_RANK["attention_key_pre_rope"] == 170
    assert COMPONENT_RANK["attention_value_states"] == 180
    assert COMPONENT_RANK["attention_gate"] == 190


def test_a_kv_space_head_is_refused_at_load_not_just_at_the_tap(env) -> None:
    """The §2.2 defect in the form round 2 actually walks into.

    ``attention_value_states`` is KV-head space, so on this GQA model head 5 is
    valid in query space (8 heads) and not here (4). The bound has to come from
    the *component*: python does not raise on an over-wide slice, it returns an
    empty one, so the read would have saved ``(b, n_pos, 0)`` and the write
    would have changed nothing.
    """
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["sites"]["tgt"] = {
        "component": "attention_value_states",
        "layer": 3,
        "head": 5,
    }
    with pytest.raises(ValidationError, match="4 heads"):
        canonicalize(raw, env)


def test_the_query_space_twin_accepts_the_same_head(env) -> None:
    """Which is what makes the refusal above about the component and not about
    the number."""
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["sites"]["tgt"] = {
        "component": "attention_query_pre_rope",
        "layer": 3,
        "head": 5,
    }
    assert canonicalize(raw, env)["sites"]["tgt"]["head"] == 5
