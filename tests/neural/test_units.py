"""Direct tests for ``causalab.neural.units``.

This module declares the core "where + what" intervention primitives that the
pyvene-backed interchange / collect / steer pipelines consume downstream:

* :class:`ComponentIndexer` wraps a per-input position-lookup callable (e.g.
  IOI / MCQA token positions) so the same indexer works in single-input and
  batched modes, with optional ``is_original`` plumbing.
* :class:`AtomicModelUnit` pairs a ``(layer, component_type, indices, unit)``
  location with a :class:`Featurizer` and an optional ``feature_indices``
  slice. ``create_intervention_config`` is the bridge into pyvene's
  intervention-class registry.
* :class:`InterchangeTarget` groups units into the
  ``List[List[AtomicModelUnit]]`` shape every interchange / DAS / PCA /
  steering analysis expects, and owns the consolidated
  ``units_metadata.json`` + ``featurizers.safetensors`` save/load round-trip.

If bounds-checking on ``set_feature_indices``, the static-vs-dynamic indexer
dispatch, the ``create_intervention_config`` enum mapping, or the JSON +
safetensors round-trip silently return wrong values, every interchange /
DAS / PCA / steering analysis that imports these symbols silently corrupts
its outputs.

Concrete featurizer subclasses (subspace, standardize, ...) live in
``causalab.methods``; tests here only need the base :class:`Featurizer`
plus :class:`SubspaceFeaturizer` for the bounds-revalidation and non-trivial
save/load paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.featurizer import Featurizer
from causalab.neural.units import (
    MODEL_UNIT_VERSION,
    AtomicModelUnit,
    ComponentIndexer,
    InterchangeTarget,
    _indexer_accepts_is_original,
)


# --------------------------------------------------------------------------- #
#  ComponentIndexer                                                            #
# --------------------------------------------------------------------------- #
class TestComponentIndexerUnit:
    """:class:`ComponentIndexer` wraps a per-input position-lookup callable
    used by every dynamic-indexed intervention site."""

    pytestmark = pytest.mark.unit

    def test_init_stores_indexer_and_id(self) -> None:
        ci = ComponentIndexer(lambda _: [0], id="idxID")
        assert ci.id == "idxID"
        assert callable(ci.indexer)

    def test_repr_carries_id(self) -> None:
        ci = ComponentIndexer(lambda _: [0], id="idxID")
        assert "idxID" in repr(ci)

    def test_index_single_input_delegates(self) -> None:
        ci = ComponentIndexer(lambda x: [x, x + 1], id="shift")
        assert ci.index(5) == [5, 6]

    def test_index_batch_preserves_order(self) -> None:
        ci = ComponentIndexer(lambda x: [x], id="echo")
        assert ci.index([3, 1, 2], batch=True) == [[3], [1], [2]]

    def test_index_passes_is_original_when_supported(self) -> None:
        seen: list[bool | None] = []

        def indexer(x, is_original):
            seen.append(is_original)
            return [int(is_original)]

        ci = ComponentIndexer(indexer, id="aware")
        out = ci.index("x", is_original=True)
        assert out == [1]
        assert seen == [True]

    def test_index_omits_is_original_for_indexers_without_it(self) -> None:
        # Indexer below does NOT accept ``is_original``; signature detection sees
        # that and calls it with no kwarg (no TypeError-swallowing fallback).
        ci = ComponentIndexer(lambda x: [42], id="legacy")
        assert ci.index("anything", is_original=True) == [42]

    def test_index_omits_is_original_when_none(self) -> None:
        # Indexer below has no ``is_original`` arg; with ``is_original=None``
        # the wrapper should call it directly (no TypeError fallback path).
        calls: list[tuple] = []

        def indexer(x):
            calls.append((x,))
            return [0]

        ci = ComponentIndexer(indexer, id="plain")
        ci.index("inp")
        assert calls == [("inp",)]


class TestComponentIndexerProperty:
    """Indexer wrapping must be transparent for the pipeline contract."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "inputs",
        [
            [0],
            [1, 2, 3],
            [5, 5, 5, 5],
        ],
    )
    def test_batch_index_equals_per_element_index(self, inputs: list[int]) -> None:
        ci = ComponentIndexer(lambda x: [x * 2], id="double")
        batched = ci.index(inputs, batch=True)
        per_element = [ci.index(x) for x in inputs]
        assert batched == per_element

    def test_index_returns_what_the_wrapped_callable_returns(self) -> None:
        def f(x):
            return list(range(x))

        ci = ComponentIndexer(f, id="range")
        for x in (0, 1, 4):
            assert ci.index(x) == f(x)


class TestComponentIndexerIsOriginalDispatch:
    """``is_original`` is threaded by signature detection at construction, never
    by catching a ``TypeError`` at call time (#430).

    The old code wrapped the flagged call in ``try/except TypeError`` and, on
    *any* ``TypeError``, re-invoked the indexer without the flag. That conflated
    "indexer does not take ``is_original``" with "indexer took it but a bug
    inside raised", silently re-running a flag-aware indexer without the flag —
    for a paired position, returning base positions on a counterfactual read.
    """

    pytestmark = pytest.mark.unit

    def test_internal_typeerror_propagates(self) -> None:
        """A ``TypeError`` raised *inside* an ``is_original``-accepting indexer
        must propagate — not be swallowed and the indexer silently re-run."""
        sentinel = "boom-from-inside-the-indexer"

        def buggy(x, is_original=True):
            raise TypeError(sentinel)

        ci = ComponentIndexer(buggy, id="buggy")
        with pytest.raises(TypeError, match=sentinel):
            ci.index("inp", is_original=False)

    def test_internal_typeerror_not_masked_as_base_positions(self) -> None:
        """The exact #430 hazard: a flag-aware indexer that raises on its CF
        branch must crash, not fall back to the (base) ``is_original=True``
        branch — which would be a silent wrong-position intervention."""

        def paired(x, is_original=True):
            if not is_original:
                raise TypeError("downstream bug on the counterfactual branch")
            return [0]  # base positions

        ci = ComponentIndexer(paired, id="paired")
        with pytest.raises(TypeError):
            ci.index("cf-input", is_original=False)

    def test_flag_threaded_when_accepted(self) -> None:
        seen: list[bool] = []

        def indexer(x, is_original=True):
            seen.append(is_original)
            return [0] if is_original else [1]

        ci = ComponentIndexer(indexer, id="aware")
        assert ci.index("x", is_original=True) == [0]
        assert ci.index("x", is_original=False) == [1]
        assert seen == [True, False]

    # -- signature detection over the four indexer shapes ------------------- #
    def test_detect_keyword_is_original(self) -> None:
        # POSITIONAL_OR_KEYWORD `is_original` → reachable by keyword.
        def kw(x, is_original=True):
            return [1] if is_original else [2]

        assert _indexer_accepts_is_original(kw) is True
        assert ComponentIndexer(kw, id="kw")._accepts_is_original is True

    def test_detect_keyword_only_is_original(self) -> None:
        def kwonly(x, *, is_original=True):
            return [1] if is_original else [2]

        assert _indexer_accepts_is_original(kwonly) is True
        assert ComponentIndexer(kwonly, id="kwonly")._accepts_is_original is True

    def test_detect_var_keyword(self) -> None:
        # A ``**kwargs`` catch-all absorbs ``is_original``.
        def varkw(x, **kwargs):
            return [1] if kwargs.get("is_original", True) else [2]

        assert _indexer_accepts_is_original(varkw) is True
        ci = ComponentIndexer(varkw, id="varkw")
        assert ci._accepts_is_original is True
        assert ci.index("x", is_original=False) == [2]

    def test_detect_positional_only_is_original_not_reachable(self) -> None:
        # A positional-only ``is_original`` cannot be passed by keyword, so it
        # counts as unsupported: the indexer is called with no kwarg (its
        # default wins) and, crucially, no ``TypeError`` from a bad keyword.
        def posonly(x, is_original=True, /):
            return [1] if is_original else [2]

        assert _indexer_accepts_is_original(posonly) is False
        ci = ComponentIndexer(posonly, id="posonly")
        assert ci._accepts_is_original is False
        assert ci.index("x", is_original=False) == [1]

    def test_detect_no_flag(self) -> None:
        def noflag(x):
            return [0]

        assert _indexer_accepts_is_original(noflag) is False
        assert ComponentIndexer(noflag, id="noflag")._accepts_is_original is False

    def test_detect_uninspectable_callable_is_unsupported(self) -> None:
        # Some C builtins have no introspectable signature (``inspect.signature``
        # raises ``ValueError``); the except branch treats them as no-flag.
        assert _indexer_accepts_is_original(print) is False


# --------------------------------------------------------------------------- #
#  AtomicModelUnit                                                             #
# --------------------------------------------------------------------------- #
class TestAtomicModelUnitUnit:
    """:class:`AtomicModelUnit` is the basic ``(where, feature-space)`` unit
    of intervention consumed by every interchange / steering analysis."""

    pytestmark = pytest.mark.unit

    @staticmethod
    def _make_unit(
        *,
        layer: int = 0,
        component_type: str = "block_input",
        indices_func: list[int] | ComponentIndexer | None = None,
        featurizer: Featurizer | None = None,
        feature_indices: list[int] | None = None,
        id: str = "null",
    ) -> AtomicModelUnit:
        return AtomicModelUnit(
            layer=layer,
            component_type=component_type,
            indices_func=indices_func if indices_func is not None else [0],
            unit="pos",
            featurizer=featurizer,
            feature_indices=feature_indices,
            id=id,
        )

    def test_init_sets_layer_and_component(self) -> None:
        u = self._make_unit(layer=1)
        assert u.layer == 1
        assert u.component_type == "block_input"
        assert u.unit == "pos"

    def test_default_featurizer_is_instance_unique(self) -> None:
        u1 = self._make_unit()
        u2 = self._make_unit()
        assert u1.featurizer is not u2.featurizer

    def test_repr_includes_id(self) -> None:
        u = self._make_unit(id="my_unit")
        assert "my_unit" in repr(u)

    def test_get_shape_returns_constructor_arg(self) -> None:
        u = AtomicModelUnit(
            layer=0,
            component_type="block_input",
            indices_func=[0],
            shape=(4, 2),
        )
        assert u.get_shape() == (4, 2)

    def test_set_and_get_layer(self) -> None:
        u = self._make_unit(layer=0)
        u.set_layer(3)
        assert u.get_layer() == 3

    def test_static_indices_dispatch(self) -> None:
        u = self._make_unit(indices_func=[2, 3])
        assert u.is_static()
        assert u.index_component("ignored") == [2, 3]

    def test_dynamic_indices_dispatch(self) -> None:
        indexer = ComponentIndexer(lambda x: [x], id="echo")
        u = self._make_unit(indices_func=indexer)
        assert not u.is_static()
        assert u.index_component(5) == [5]

    def test_index_component_batch_routes_through_indexer(self) -> None:
        u = self._make_unit(indices_func=[7])
        # attention_mask=None: symbolic inputs, no padding to correct.
        out = u.index_component(["a", "b"], batch=True, attention_mask=None)
        assert out == [[7], [7]]

    def test_get_index_id_reports_indexer_id(self) -> None:
        # Static-list case: id is derived from the constant value.
        u_static = self._make_unit(indices_func=[1, 2])
        assert "1" in u_static.get_index_id() and "2" in u_static.get_index_id()

        # Dynamic case: id is the wrapped indexer's id.
        indexer = ComponentIndexer(lambda x: [0], id="explicit")
        u_dynamic = self._make_unit(indices_func=indexer)
        assert u_dynamic.get_index_id() == "explicit"

    def test_feature_indices_within_bounds_accepted(self) -> None:
        feat = Featurizer(n_features=4)
        u = self._make_unit(featurizer=feat, feature_indices=[1, 2])
        assert u.get_feature_indices() == [1, 2]

    def test_feature_indices_out_of_bounds_raises(self) -> None:
        feat = SubspaceFeaturizer(shape=(4, 4), trainable=False)  # n_features == 4
        with pytest.raises(ValueError):
            self._make_unit(featurizer=feat, feature_indices=[0, 5])

    def test_set_feature_indices_can_clear(self) -> None:
        feat = Featurizer(n_features=4)
        u = self._make_unit(featurizer=feat, feature_indices=[0])
        u.set_feature_indices(None)
        assert u.get_feature_indices() is None

    @pytest.mark.parametrize(
        "intervention_type",
        ["collect", "interchange", "mask", "add", "replace", "interpolation"],
    )
    def test_every_mode_builds_over_the_unit_featurizer(
        self, intervention_type: str
    ) -> None:
        """Each mode is buildable over a unit's feature space.

        This replaces ``create_intervention_config``, which emitted the dict
        pyvene consumed (component / unit / layer / group_key /
        intervention_type-as-a-class). A unit no longer carries any backbone
        config: the engine resolves it to a site and pairs it with an
        intervention, so what a unit must still support is exactly this.
        """
        from causalab.neural.interventions import build_intervention

        feat = Featurizer(n_features=4)  # n_features required for ``mask``
        unit = self._make_unit(featurizer=feat)
        intervention = build_intervention(unit.featurizer, intervention_type)
        assert intervention._featurizer is feat

    def test_unknown_mode_raises(self) -> None:
        from causalab.neural.interventions import build_intervention

        unit = self._make_unit()
        with pytest.raises(ValueError, match="Unknown intervention mode"):
            build_intervention(unit.featurizer, "bogus")


class TestAtomicModelUnitProperty:
    """Invariants on the static/dynamic dispatch, featurizer swaps, and
    default-featurizer isolation."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("payload", [0, "anything", [1, 2, 3], object()])
    def test_static_index_constant_under_arbitrary_input(self, payload) -> None:
        u = AtomicModelUnit(
            layer=0,
            component_type="block_input",
            indices_func=[2, 3],
        )
        assert u.index_component(payload) == [2, 3]

    def test_set_featurizer_revalidates_bounds(self) -> None:
        big = SubspaceFeaturizer(shape=(4, 4), trainable=False)  # n_features = 4
        small = SubspaceFeaturizer(shape=(4, 2), trainable=False)  # n_features = 2
        # index 3 fits the 4-dim featurizer but overflows the 2-dim one.
        u = AtomicModelUnit(
            layer=0,
            component_type="block_input",
            indices_func=[0],
            featurizer=big,
            feature_indices=[3],
        )
        with pytest.raises(ValueError):
            u.set_featurizer(small)

    def test_default_featurizers_are_independent_instances(self) -> None:
        # Mutation isolation: poking one unit's featurizer must not bleed into
        # any other unit's featurizer.
        u1 = AtomicModelUnit(layer=0, component_type="block_input", indices_func=[0])
        u2 = AtomicModelUnit(layer=0, component_type="block_input", indices_func=[0])
        u1.featurizer.n_features = 4
        assert u2.featurizer.n_features is None


# --------------------------------------------------------------------------- #
#  InterchangeTarget                                                           #
# --------------------------------------------------------------------------- #
def _make_named_unit(unit_id: str, **kwargs) -> AtomicModelUnit:
    """Construct an AtomicModelUnit with an explicit id for InterchangeTarget tests."""
    return AtomicModelUnit(
        layer=kwargs.pop("layer", 0),
        component_type=kwargs.pop("component_type", "block_input"),
        indices_func=kwargs.pop("indices_func", [0]),
        unit=kwargs.pop("unit", "pos"),
        id=unit_id,
        **kwargs,
    )


class TestInterchangeTargetUnit:
    """:class:`InterchangeTarget` groups units into the
    ``List[List[AtomicModelUnit]]`` shape required by the pyvene interchange
    pipeline and the consolidated save/load format."""

    pytestmark = pytest.mark.unit

    def test_len_returns_group_count(self) -> None:
        target = InterchangeTarget(
            [
                [_make_named_unit("a"), _make_named_unit("b")],
                [_make_named_unit("c")],
            ]
        )
        assert len(target) == 2

    def test_getitem_returns_group(self) -> None:
        g0 = [_make_named_unit("a")]
        g1 = [_make_named_unit("b")]
        target = InterchangeTarget([g0, g1])
        assert target[0] is g0
        assert target[1] is g1

    def test_setitem_mutates_in_place(self) -> None:
        target = InterchangeTarget([[_make_named_unit("a")]])
        new_group = [_make_named_unit("b"), _make_named_unit("c")]
        target[0] = new_group
        assert target[0] is new_group

    def test_iter_yields_groups_in_order(self) -> None:
        groups = [
            [_make_named_unit("a")],
            [_make_named_unit("b"), _make_named_unit("c")],
        ]
        target = InterchangeTarget(groups)
        assert list(target) == groups

    def test_repr_reports_group_and_total_counts(self) -> None:
        target = InterchangeTarget(
            [
                [_make_named_unit("a"), _make_named_unit("b")],
                [_make_named_unit("c")],
            ]
        )
        rep = repr(target)
        assert "2 groups" in rep
        assert "3 total" in rep

    def test_flatten_walks_groups_in_order(self) -> None:
        u_a = _make_named_unit("a")
        u_b = _make_named_unit("b")
        u_c = _make_named_unit("c")
        target = InterchangeTarget([[u_a, u_b], [u_c]])
        assert target.flatten() == [u_a, u_b, u_c]

    def test_nest_to_match_shapes_flat_list_into_groups(self) -> None:
        target = InterchangeTarget(
            [
                [_make_named_unit("a"), _make_named_unit("b")],
                [_make_named_unit("c"), _make_named_unit("d"), _make_named_unit("e")],
            ]
        )
        nested = target.nest_to_match([1, 2, 3, 4, 5])
        assert nested == [[1, 2], [3, 4, 5]]

    def test_nest_to_match_length_mismatch_raises(self) -> None:
        target = InterchangeTarget([[_make_named_unit("a")]])
        with pytest.raises(ValueError):
            target.nest_to_match([1, 2])  # wrong length

    def test_set_featurizer_propagates_to_every_unit(self) -> None:
        target = InterchangeTarget([[_make_named_unit("a"), _make_named_unit("b")]])
        feat = Featurizer(n_features=4, id="shared")
        target.set_featurizer(feat)
        for unit in target.flatten():
            assert unit.featurizer is feat

    def test_set_feature_indices_propagates_to_every_unit(self) -> None:
        feat = Featurizer(n_features=4)
        target = InterchangeTarget(
            [
                [
                    _make_named_unit("a", featurizer=Featurizer(n_features=4)),
                    _make_named_unit("b", featurizer=feat),
                ]
            ]
        )
        target.set_feature_indices([1, 2])
        for unit in target.flatten():
            assert unit.get_feature_indices() == [1, 2]

    def test_get_feature_indices_returns_per_unit_dict(self) -> None:
        u_a = _make_named_unit("a", featurizer=Featurizer(n_features=4))
        u_b = _make_named_unit("b", featurizer=Featurizer(n_features=4))
        u_a.set_feature_indices([0])
        u_b.set_feature_indices([1, 2])
        target = InterchangeTarget([[u_a, u_b]])
        assert target.get_feature_indices() == {"a": [0], "b": [1, 2]}

    def test_load_raises_when_metadata_missing(self, tmp_path: Path) -> None:
        target = InterchangeTarget([[_make_named_unit("a")]])
        with pytest.raises(FileNotFoundError):
            target.load(str(tmp_path))


class TestInterchangeTargetProperty:
    """Round-trip and structural invariants for :class:`InterchangeTarget`.

    Note: ``InterchangeTarget.load`` modifies in place and requires the
    destination target to be pre-populated with units whose ``id``s match
    the saved metadata exactly. The round-trip property test below builds
    that companion target explicitly — a naive "load into empty target"
    would never match the saved ids.
    """

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "shape",
        [
            [1],
            [3],
            [1, 1],
            [2, 3],
            [1, 4, 2],
        ],
    )
    def test_nest_to_match_preserves_total_and_per_group_sizes(
        self, shape: list[int]
    ) -> None:
        groups: list[list[AtomicModelUnit]] = []
        counter = 0
        for n in shape:
            group = []
            for _ in range(n):
                group.append(_make_named_unit(f"u{counter}"))
                counter += 1
            groups.append(group)
        target = InterchangeTarget(groups)
        total = sum(shape)
        flat = list(range(total))
        nested = target.nest_to_match(flat)
        assert [len(g) for g in nested] == shape
        assert sum(len(g) for g in nested) == total
        # Order-preservation: flattening the nested result recovers the input.
        assert [x for g in nested for x in g] == flat

    def test_save_load_trivial_writes_only_metadata(self, tmp_path: Path) -> None:
        # All-trivial featurizers ⇒ no featurizers.safetensors written.
        target = InterchangeTarget([[_make_named_unit("a"), _make_named_unit("b")]])
        target.save(str(tmp_path))
        assert (tmp_path / "units_metadata.json").exists()
        assert not (tmp_path / "featurizers.safetensors").exists()

    def test_save_load_roundtrip_preserves_metadata(self, tmp_path: Path) -> None:
        # Build a target with a non-trivial featurizer so the safetensors path
        # is exercised; set explicit feature_indices to round-trip.
        feat_src = SubspaceFeaturizer(shape=(4, 2), trainable=False)
        u_src = AtomicModelUnit(
            layer=1,
            component_type="block_input",
            indices_func=[0],
            featurizer=feat_src,
            feature_indices=[0, 1],
            id="u1",
        )
        u_src_b = AtomicModelUnit(
            layer=2,
            component_type="block_input",
            indices_func=[0],
            id="u2",  # trivial featurizer
        )
        src = InterchangeTarget([[u_src, u_src_b]])
        src.save(str(tmp_path))

        # featurizers.safetensors must exist because at least one stage is
        # non-trivial.
        assert (tmp_path / "featurizers.safetensors").exists()

        # Build a companion target with matching ids; load mutates in place.
        u_dst = AtomicModelUnit(
            layer=0,  # deliberately wrong; load should NOT overwrite layer
            component_type="block_input",
            indices_func=[0],
            id="u1",
        )
        u_dst_b = AtomicModelUnit(
            layer=0,
            component_type="block_input",
            indices_func=[0],
            id="u2",
        )
        dst = InterchangeTarget([[u_dst, u_dst_b]])
        dst.load(str(tmp_path))

        # feature_indices round-trip
        assert dst.get_feature_indices() == {"u1": [0, 1], "u2": None}
        # featurizer was rebuilt for the non-trivial unit
        assert dst[0][0].featurizer.n_features == 2

    def test_load_raises_on_unknown_unit_id(self, tmp_path: Path) -> None:
        src = InterchangeTarget([[_make_named_unit("known")]])
        src.save(str(tmp_path))
        dst = InterchangeTarget([[_make_named_unit("different")]])
        with pytest.raises(ValueError, match="not found in saved metadata"):
            dst.load(str(tmp_path))

    def test_version_mismatch_raises_unless_ignored(self, tmp_path: Path) -> None:
        # Construct metadata with a synthetic version different from current.
        src = InterchangeTarget([[_make_named_unit("a")]])
        src.save(str(tmp_path))
        meta_path = tmp_path / "units_metadata.json"
        meta = json.loads(meta_path.read_text())
        for unit_id in meta:
            meta[unit_id]["version"] = "0.0-test-mismatch"
        assert meta["a"]["version"] != MODEL_UNIT_VERSION
        meta_path.write_text(json.dumps(meta))

        dst1 = InterchangeTarget([[_make_named_unit("a")]])
        with pytest.raises(ValueError, match="version"):
            dst1.load(str(tmp_path))

        # ``ignore_version_mismatch=True`` lets the load succeed.
        dst2 = InterchangeTarget([[_make_named_unit("a")]])
        dst2.load(str(tmp_path), ignore_version_mismatch=True)

    def test_save_load_units_metadata_keys_match_unit_ids(self, tmp_path: Path) -> None:
        target = InterchangeTarget(
            [
                [_make_named_unit("alpha"), _make_named_unit("beta")],
                [_make_named_unit("gamma")],
            ]
        )
        target.save(str(tmp_path))
        meta = json.loads((tmp_path / "units_metadata.json").read_text())
        assert set(meta.keys()) == {"alpha", "beta", "gamma"}
        for unit_id, entry in meta.items():
            assert entry["id"] == unit_id
            assert entry["version"] == MODEL_UNIT_VERSION

    def test_save_load_returns_parent_dir(self, tmp_path: Path) -> None:
        target = InterchangeTarget([[_make_named_unit("a")]])
        returned = target.save(str(tmp_path))
        assert returned == str(tmp_path)
        assert os.path.isdir(returned)
