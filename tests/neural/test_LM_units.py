"""Tests for ``causalab.neural.LM_units``.

``ResidualStream``, ``MLP`` and ``AttentionHead`` bind the generic
``AtomicModelUnit`` / ``Featurizer`` / ``ComponentIndexer`` abstractions
from ``causalab.neural.units`` to language-model-shaped activations.
Downstream, ``InterchangeTarget``, ``prepare_interchange_batch``,
``batched_interchange_intervention``, and the analysis-side mask plots
(``io/visualizations.py``) address these units by their ``id`` string and
their ``component_type``. The contracts pinned here — featurizer identity,
``component_type`` wiring, ``index_component`` shape, ``id`` string schema,
and ``load_modules`` round-trip — are load-bearing for every interchange
experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.neural.LM_units import AttentionHead, AttentionOutput, MLP, ResidualStream
from causalab.neural.units import ComponentIndexer, Featurizer
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer


# --------------------------------------------------------------------------- #
#  Local helpers                                                              #
# --------------------------------------------------------------------------- #
def _make_indexer(token_id: str = "last_token") -> ComponentIndexer:
    """Build a tiny ``ComponentIndexer`` whose ``.id`` matches *token_id*.

    The indexer returns a constant ``[0]`` per input; we only care about
    the ``id`` attribute, which is what the ``LM_units`` constructors
    embed into the unit ``id`` string and what ``load_modules`` parses
    back out.
    """
    return ComponentIndexer(lambda _input: [0], id=token_id)


def _write_trivial_featurizer(base_path: str, n_features: int | None = None) -> None:
    """Serialize a default ``Featurizer()`` to ``<base_path>_{featurizer,inverse_featurizer}``.

    Mirrors what ``LM_units.<Class>.load_modules`` expects on disk: a
    base-class identity featurizer saved through ``Featurizer.save_modules``.
    The optional ``n_features`` keyword pins ``feature_indices`` bounds
    checks for callers that want them.
    """
    Featurizer(n_features=n_features).save_modules(base_path)


# --------------------------------------------------------------------------- #
#  ResidualStream — unit tier                                                  #
# --------------------------------------------------------------------------- #
class TestResidualStreamUnit:
    """``ResidualStream(layer, token_indices)`` selects a hidden-state slice
    at a given layer + token position. The constructor must wire
    ``component_type`` from ``target_output``, mint a fresh featurizer per
    instance, and emit the canonical ``id`` schema that ``load_modules`` and
    the downstream heatmap pipeline parse back out.
    """

    pytestmark = pytest.mark.unit

    def test_featurizers_unique(self) -> None:
        rs1 = ResidualStream(layer=0, token_indices=[0])
        rs2 = ResidualStream(layer=0, token_indices=[0])
        assert rs1.featurizer is not rs2.featurizer

    def test_component_type_default_is_block_input(self) -> None:
        rs = ResidualStream(layer=2, token_indices=[0])
        assert rs.component_type == "block_input"

    def test_component_type_target_output_true_is_block_output(self) -> None:
        rs = ResidualStream(layer=2, token_indices=[0], target_output=True)
        assert rs.component_type == "block_output"

    def test_id_schema_with_list_token_indices(self) -> None:
        rs = ResidualStream(layer=3, token_indices=[7])
        assert rs.id == "ResidualStream(Layer-3,block_input,Token-[7])"

    def test_id_schema_with_componentindexer_token(self) -> None:
        idx = _make_indexer("last_token")
        rs = ResidualStream(layer=5, token_indices=idx, target_output=True)
        assert rs.id == "ResidualStream(Layer-5,block_output,Token-last_token)"

    def test_id_distinguishes_block_input_from_block_output(self) -> None:
        """Step 6 in the plan: ``block_input`` vs ``block_output`` for the
        same (layer, token) must produce distinct ``id`` strings so the
        downstream parser can address them independently."""
        idx = _make_indexer("last_token")
        rs_in = ResidualStream(layer=1, token_indices=idx, target_output=False)
        rs_out = ResidualStream(layer=1, token_indices=idx, target_output=True)
        assert rs_in.id != rs_out.id

    def test_load_modules_round_trip_block_input(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        rs = ResidualStream(layer=2, token_indices=idx, target_output=False)

        _write_trivial_featurizer(str(tmp_path / rs.id))
        loaded = ResidualStream.load_modules(rs.id, str(tmp_path), [idx])

        assert loaded.id == rs.id
        assert loaded.layer == rs.layer
        assert loaded.component_type == rs.component_type
        assert loaded.token_indices is idx

    def test_load_modules_round_trip_block_output(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        rs = ResidualStream(layer=2, token_indices=idx, target_output=True)

        _write_trivial_featurizer(str(tmp_path / rs.id))
        loaded = ResidualStream.load_modules(rs.id, str(tmp_path), [idx])

        assert loaded.id == rs.id
        assert loaded.component_type == "block_output"

    def test_load_modules_loads_feature_indices_json(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        rs = ResidualStream(layer=0, token_indices=idx)

        _write_trivial_featurizer(str(tmp_path / rs.id))
        indices_path = str(tmp_path / f"{rs.id}_indices")
        with open(indices_path, "w") as f:
            json.dump([1, 3], f)

        loaded = ResidualStream.load_modules(rs.id, str(tmp_path), [idx])
        assert loaded.feature_indices == [1, 3]

    def test_load_modules_missing_indices_is_none(self, tmp_path: Path) -> None:
        """Plan: the indices loader degrades to ``None`` when the JSON file
        is absent. The current ``except Exception`` is intentionally broad —
        flag as TODO for the source author, but pin the contract."""
        idx = _make_indexer("last_token")
        rs = ResidualStream(layer=0, token_indices=idx)

        _write_trivial_featurizer(str(tmp_path / rs.id))
        # No `_indices` file written.
        loaded = ResidualStream.load_modules(rs.id, str(tmp_path), [idx])
        assert loaded.feature_indices is None

    def test_load_modules_unknown_token_id_raises(self, tmp_path: Path) -> None:
        idx = _make_indexer("real_token")
        rs = ResidualStream(layer=0, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / rs.id))

        other = _make_indexer("other_token")
        with pytest.raises(ValueError, match="not found in provided list"):
            ResidualStream.load_modules(rs.id, str(tmp_path), [other])


# --------------------------------------------------------------------------- #
#  ResidualStream — property tier                                              #
# --------------------------------------------------------------------------- #
class TestResidualStreamProperty:
    """Invariants for ``ResidualStream``: every fresh construction owns a
    distinct featurizer, the ``id`` string is a pure function of
    ``(layer, target_output, token_indices.id)``, and the parser/serializer
    pair is the identity on freshly-constructed units.
    """

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_featurizers_are_pairwise_distinct(self, n: int) -> None:
        units = [ResidualStream(layer=0, token_indices=[0]) for _ in range(n)]
        assert len({id(u.featurizer) for u in units}) == n

    @pytest.mark.parametrize(
        "layer, target_output, token_id",
        [
            (0, False, "tok_a"),
            (3, True, "tok_a"),
            (7, False, "tok_b"),
            (11, True, "tok_c"),
        ],
    )
    def test_id_is_pure_function_of_inputs(
        self, layer: int, target_output: bool, token_id: str
    ) -> None:
        idx = _make_indexer(token_id)
        rs1 = ResidualStream(
            layer=layer, token_indices=idx, target_output=target_output
        )
        rs2 = ResidualStream(
            layer=layer, token_indices=idx, target_output=target_output
        )
        assert rs1.id == rs2.id

    @pytest.mark.parametrize(
        "layer, token_id",
        [
            (0, "tok_a"),
            (2, "tok_b"),
            (5, "tok_c"),
        ],
    )
    def test_load_modules_id_round_trip_block_input(
        self,
        tmp_path: Path,
        layer: int,
        token_id: str,
    ) -> None:
        """Round-trip the ``target_output=False`` (block_input) case across
        the parameter grid; the ``target_output=True`` (block_output) case is
        covered at the unit tier."""
        idx = _make_indexer(token_id)
        rs = ResidualStream(layer=layer, token_indices=idx, target_output=False)
        _write_trivial_featurizer(str(tmp_path / rs.id))

        loaded = ResidualStream.load_modules(rs.id, str(tmp_path), [idx])
        assert loaded.id == rs.id


# --------------------------------------------------------------------------- #
#  MLP — unit tier                                                             #
# --------------------------------------------------------------------------- #
class TestMLPUnit:
    """``MLP(layer, token_indices, location=...)`` selects an MLP-attached
    slice (input / output / per-neuron activation). The constructor must
    validate ``location`` against ``VALID_LOCATIONS``, mirror it onto
    ``component_type``, emit the canonical ``id``, and reconstruct cleanly
    from disk via ``load_modules``.
    """

    pytestmark = pytest.mark.unit

    def test_featurizers_unique(self) -> None:
        mlp1 = MLP(layer=0, token_indices=[0])
        mlp2 = MLP(layer=0, token_indices=[0])
        assert mlp1.featurizer is not mlp2.featurizer

    def test_default_location_is_mlp_output(self) -> None:
        mlp = MLP(layer=0, token_indices=[0])
        assert mlp.component_type == "mlp_output"

    @pytest.mark.parametrize(
        "location",
        ["mlp_input", "mlp_output", "mlp_activation"],
    )
    def test_component_type_mirrors_location(self, location: str) -> None:
        mlp = MLP(layer=0, token_indices=[0], location=location)
        assert mlp.component_type == location

    def test_invalid_location_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid location"):
            MLP(layer=0, token_indices=[0], location="invalid")

    def test_valid_locations_class_attr(self) -> None:
        """``VALID_LOCATIONS`` is part of the public surface (the constructor
        names it in its error message). Pin its membership."""
        assert set(MLP.VALID_LOCATIONS) == {
            "mlp_input",
            "mlp_output",
            "mlp_activation",
        }

    def test_id_schema_with_list_token_indices(self) -> None:
        mlp = MLP(layer=4, token_indices=[2], location="mlp_input")
        assert mlp.id == "MLP(Layer-4,mlp_input,Token-[2])"

    def test_id_schema_with_componentindexer_token(self) -> None:
        idx = _make_indexer("last_token")
        mlp = MLP(layer=6, token_indices=idx, location="mlp_activation")
        assert mlp.id == "MLP(Layer-6,mlp_activation,Token-last_token)"

    @pytest.mark.parametrize(
        "location",
        ["mlp_input", "mlp_output", "mlp_activation"],
    )
    def test_load_modules_round_trip(self, tmp_path: Path, location: str) -> None:
        idx = _make_indexer("last_token")
        mlp = MLP(layer=3, token_indices=idx, location=location)
        _write_trivial_featurizer(str(tmp_path / mlp.id))

        loaded = MLP.load_modules(mlp.id, str(tmp_path), [idx])
        assert loaded.id == mlp.id
        assert loaded.layer == mlp.layer
        assert loaded.component_type == location

    def test_load_modules_loads_feature_indices_json(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        mlp = MLP(layer=0, token_indices=idx)

        _write_trivial_featurizer(str(tmp_path / mlp.id))
        with open(str(tmp_path / f"{mlp.id}_indices"), "w") as f:
            json.dump([0, 2], f)

        loaded = MLP.load_modules(mlp.id, str(tmp_path), [idx])
        assert loaded.feature_indices == [0, 2]

    def test_load_modules_missing_indices_is_none(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        mlp = MLP(layer=0, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / mlp.id))

        loaded = MLP.load_modules(mlp.id, str(tmp_path), [idx])
        assert loaded.feature_indices is None

    def test_load_modules_unknown_token_id_raises(self, tmp_path: Path) -> None:
        idx = _make_indexer("real_token")
        mlp = MLP(layer=0, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / mlp.id))

        other = _make_indexer("other_token")
        with pytest.raises(ValueError, match="not found in provided list"):
            MLP.load_modules(mlp.id, str(tmp_path), [other])


# --------------------------------------------------------------------------- #
#  AttentionOutput — unit tier                                                 #
# --------------------------------------------------------------------------- #
class TestAttentionOutputUnit:
    """``AttentionOutput(layer, token_indices)`` selects the whole attention-block
    output at a layer (all heads), used as a path-patching restorer. The
    constructor must wire ``component_type='attention_output'``, mint a fresh
    featurizer per instance, emit the canonical ``id``, and reconstruct from disk
    via ``load_modules``.
    """

    pytestmark = pytest.mark.unit

    def test_featurizers_unique(self) -> None:
        a1 = AttentionOutput(layer=0, token_indices=[0])
        a2 = AttentionOutput(layer=0, token_indices=[0])
        assert a1.featurizer is not a2.featurizer

    def test_component_type_is_attention_output(self) -> None:
        ao = AttentionOutput(layer=0, token_indices=[0])
        assert ao.component_type == "attention_output"
        assert ao.unit == "pos"

    def test_id_schema_with_list_token_indices(self) -> None:
        ao = AttentionOutput(layer=4, token_indices=[2])
        assert ao.id == "AttentionOutput(Layer-4,attention_output,Token-[2])"

    def test_id_schema_with_componentindexer_token(self) -> None:
        idx = _make_indexer("last_token")
        ao = AttentionOutput(layer=6, token_indices=idx)
        assert ao.id == "AttentionOutput(Layer-6,attention_output,Token-last_token)"

    def test_load_modules_round_trip(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        ao = AttentionOutput(layer=3, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / ao.id))

        loaded = AttentionOutput.load_modules(ao.id, str(tmp_path), [idx])
        assert loaded.id == ao.id
        assert loaded.layer == ao.layer
        assert loaded.component_type == "attention_output"

    def test_load_modules_loads_feature_indices_json(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        ao = AttentionOutput(layer=0, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / ao.id))
        with open(str(tmp_path / f"{ao.id}_indices"), "w") as f:
            json.dump([0, 2], f)

        loaded = AttentionOutput.load_modules(ao.id, str(tmp_path), [idx])
        assert loaded.feature_indices == [0, 2]

    def test_load_modules_unknown_token_id_raises(self, tmp_path: Path) -> None:
        idx = _make_indexer("real_token")
        ao = AttentionOutput(layer=0, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / ao.id))

        other = _make_indexer("other_token")
        with pytest.raises(ValueError, match="not found in provided list"):
            AttentionOutput.load_modules(ao.id, str(tmp_path), [other])


# --------------------------------------------------------------------------- #
#  MLP — property tier                                                         #
# --------------------------------------------------------------------------- #
class TestMLPProperty:
    """``MLP.location`` is a bijection onto ``component_type`` across
    ``VALID_LOCATIONS``; freshly-constructed units own distinct featurizers;
    and ``load_modules(id) -> .id`` is the identity for every
    ``(layer, location, token)`` triple.
    """

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_featurizers_are_pairwise_distinct(self, n: int) -> None:
        units = [MLP(layer=0, token_indices=[0]) for _ in range(n)]
        assert len({id(u.featurizer) for u in units}) == n

    def test_location_bijects_onto_component_type(self) -> None:
        """For every ``VALID_LOCATIONS`` value, ``component_type`` equals it,
        and the set of produced ``component_type`` values is exactly
        ``VALID_LOCATIONS`` (a bijection)."""
        component_types = {
            MLP(layer=0, token_indices=[0], location=loc).component_type
            for loc in MLP.VALID_LOCATIONS
        }
        assert component_types == set(MLP.VALID_LOCATIONS)

    @pytest.mark.parametrize(
        "layer, location, token_id",
        [
            (0, "mlp_input", "tok_a"),
            (2, "mlp_output", "tok_b"),
            (5, "mlp_activation", "tok_c"),
            (11, "mlp_output", "tok_d"),
        ],
    )
    def test_load_modules_id_round_trip(
        self,
        tmp_path: Path,
        layer: int,
        location: str,
        token_id: str,
    ) -> None:
        idx = _make_indexer(token_id)
        mlp = MLP(layer=layer, token_indices=idx, location=location)
        _write_trivial_featurizer(str(tmp_path / mlp.id))

        loaded = MLP.load_modules(mlp.id, str(tmp_path), [idx])
        assert loaded.id == mlp.id


# --------------------------------------------------------------------------- #
#  AttentionHead — unit tier                                                   #
# --------------------------------------------------------------------------- #
class TestAttentionHeadUnit:
    """``AttentionHead(layer, head, token_indices)`` selects an attention
    head's value stream at a token position. The constructor must wire
    ``component_type`` from ``target_output``, mint a fresh featurizer per
    instance, validate feature-index bounds against ``set_featurizer``,
    expose the canonical ``[[[head]], [[token]]]`` index shape (with a
    batched variant), emit the canonical ``id``, and reconstruct cleanly
    from disk via ``load_modules``.
    """

    pytestmark = pytest.mark.unit

    def test_featurizers_unique(self) -> None:
        ah1 = AttentionHead(layer=0, head=5, token_indices=[0])
        ah2 = AttentionHead(layer=0, head=5, token_indices=[0])
        assert ah1.featurizer is not ah2.featurizer

    def test_index_component_scalar_shape(self) -> None:
        ah = AttentionHead(layer=1, head=7, token_indices=[3])
        assert ah.index_component("dummy") == [[[7]], [[3]]]

    def test_index_component_batched_shape(self) -> None:
        """Plan: the batched branch
        ``[[[head]] * N, [indices for each input]]`` is currently untested.
        Canonical layout per source: outer list of two — N copies of the
        head index, then N copies of the token index list."""
        ah = AttentionHead(layer=1, head=7, token_indices=[3])
        # attention_mask=None opts out of the padded-frame shift — these are
        # symbolic inputs, not real tokenizer output, so there's no padding
        # to correct for.
        result = ah.index_component(["a", "b", "c"], batch=True, attention_mask=None)
        assert result == [[[7], [7], [7]], [[3], [3], [3]]]

    def test_component_type_default_is_value_output(self) -> None:
        ah = AttentionHead(layer=0, head=0, token_indices=[0])
        assert ah.component_type == "head_attention_value_output"

    def test_component_type_target_output_false_is_value_input(self) -> None:
        ah = AttentionHead(layer=0, head=0, token_indices=[0], target_output=False)
        assert ah.component_type == "head_attention_value_input"

    def test_id_schema_with_list_token_indices(self) -> None:
        ah = AttentionHead(layer=2, head=4, token_indices=[8])
        assert ah.id == "AttentionHead(Layer-2,Head-4,Token-[8])"

    def test_id_schema_with_componentindexer_token(self) -> None:
        idx = _make_indexer("last_token")
        ah = AttentionHead(layer=5, head=3, token_indices=idx)
        assert ah.id == "AttentionHead(Layer-5,Head-3,Token-last_token)"

    def test_feature_bounds_violation(self) -> None:
        """Setting a smaller featurizer that no longer covers the unit's
        ``feature_indices`` must fail loudly."""
        big_feat = SubspaceFeaturizer(shape=(4, 4), trainable=False)
        small_feat = SubspaceFeaturizer(shape=(4, 2), trainable=False)

        ah = AttentionHead(
            layer=0,
            head=0,
            token_indices=[0],
            featurizer=big_feat,
            feature_indices=[3],  # valid in 4-dim space
        )
        with pytest.raises(ValueError):
            ah.set_featurizer(small_feat)

    def test_load_modules_round_trip(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        ah = AttentionHead(layer=2, head=4, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / ah.id))

        loaded = AttentionHead.load_modules(ah.id, str(tmp_path), [idx])
        assert loaded.id == ah.id
        assert loaded.layer == ah.layer
        assert loaded.head == ah.head

    def test_load_modules_loads_feature_indices_json(self, tmp_path: Path) -> None:
        """``AttentionHead.load_modules`` parses ``Layer`` / ``Head`` /
        ``Token`` and reads ``{id}_indices``."""
        idx = _make_indexer("last_token")
        ah = AttentionHead(layer=1, head=2, token_indices=idx)

        _write_trivial_featurizer(str(tmp_path / ah.id))
        with open(str(tmp_path / f"{ah.id}_indices"), "w") as f:
            json.dump([0, 1], f)

        loaded = AttentionHead.load_modules(ah.id, str(tmp_path), [idx])
        assert loaded.feature_indices == [0, 1]

    def test_load_modules_missing_indices_is_none(self, tmp_path: Path) -> None:
        idx = _make_indexer("last_token")
        ah = AttentionHead(layer=1, head=2, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / ah.id))

        loaded = AttentionHead.load_modules(ah.id, str(tmp_path), [idx])
        assert loaded.feature_indices is None


# --------------------------------------------------------------------------- #
#  AttentionHead — property tier                                               #
# --------------------------------------------------------------------------- #
class TestAttentionHeadProperty:
    """Invariants for ``AttentionHead``: pairwise-distinct featurizers per
    construction, canonical scalar / batched ``index_component`` shape across
    a grid of ``(layer, head, token)`` triples, and the parser/serializer
    pair is the identity on the unit ``id``.
    """

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_featurizers_are_pairwise_distinct(self, n: int) -> None:
        units = [AttentionHead(layer=0, head=0, token_indices=[0]) for _ in range(n)]
        assert len({id(u.featurizer) for u in units}) == n

    @pytest.mark.parametrize(
        "layer, head, token",
        [
            (0, 0, 0),
            (0, 7, 3),
            (5, 3, 2),
            (11, 7, 15),
        ],
    )
    def test_index_component_scalar_shape_invariant(
        self, layer: int, head: int, token: int
    ) -> None:
        ah = AttentionHead(layer=layer, head=head, token_indices=[token])
        assert ah.index_component("x") == [[[head]], [[token]]]

    @pytest.mark.parametrize(
        "layer, head, token, batch_size",
        [
            (0, 0, 0, 1),
            (1, 3, 2, 2),
            (5, 7, 4, 3),
            (11, 1, 0, 5),
        ],
    )
    def test_index_component_batched_shape_invariant(
        self, layer: int, head: int, token: int, batch_size: int
    ) -> None:
        ah = AttentionHead(layer=layer, head=head, token_indices=[token])
        # attention_mask=None: symbolic inputs, no padding to correct.
        result = ah.index_component(["x"] * batch_size, batch=True, attention_mask=None)
        # Canonical batched layout per source:
        # [[[head]] * batch_size, [indices(x) for x in input]]
        assert result == [[[head]] * batch_size, [[token]] * batch_size]

    @pytest.mark.parametrize(
        "layer, head, token_id",
        [
            (0, 0, "tok_a"),
            (2, 3, "tok_b"),
            (5, 7, "tok_c"),
        ],
    )
    def test_load_modules_id_round_trip(
        self,
        tmp_path: Path,
        layer: int,
        head: int,
        token_id: str,
    ) -> None:
        idx = _make_indexer(token_id)
        ah = AttentionHead(layer=layer, head=head, token_indices=idx)
        _write_trivial_featurizer(str(tmp_path / ah.id))

        loaded = AttentionHead.load_modules(ah.id, str(tmp_path), [idx])
        assert loaded.id == ah.id
