"""
pytest unit-tests for the core abstractions in model_units.py
(no ResidualStream / AttentionHead).
"""

import pytest

import causalab.neural.model_units as MU
import causalab.neural.featurizers as F  # the module we just rewrote


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def make_test_unit():
    """Return a trivial AtomicModelUnit for tests."""
    return MU.AtomicModelUnit(
        layer=0,
        component_type="block_input",
        indices_func=[0],
        unit="pos",
    )


# --------------------------------------------------------------------------- #
#  1. Unique default featurizers                                               #
# --------------------------------------------------------------------------- #
def test_default_featurizers_are_unique():
    u1 = make_test_unit()
    u2 = make_test_unit()

    # distinct identity featurizers
    assert u1.featurizer is not u2.featurizer

    # mutate one — the other stays pristine
    u1.featurizer.n_features = 4
    assert u2.featurizer.n_features is None


# --------------------------------------------------------------------------- #
#  2. ComponentIndexer __repr__                                                #
# --------------------------------------------------------------------------- #
def test_componentindexer_repr():
    ci = MU.ComponentIndexer(lambda _: [0], id="idxID")
    assert "idxID" in repr(ci)


# --------------------------------------------------------------------------- #
#  3. AtomicModelUnit equality & hashing (based on layer, type, unit, index)  #
# --------------------------------------------------------------------------- #
def test_unit_properties():
    u1 = MU.AtomicModelUnit(layer=1, component_type="block_input", indices_func=[0])
    u2 = MU.AtomicModelUnit(layer=1, component_type="block_input", indices_func=[0])
    u3 = MU.AtomicModelUnit(layer=2, component_type="block_input", indices_func=[0])

    assert u1.layer == 1
    assert u1.component_type == "block_input"
    assert u1.unit == "pos"

    assert u2.layer == 1
    assert u3.layer == 2


# --------------------------------------------------------------------------- #
#  4. Feature-index bounds checking                                            #
# --------------------------------------------------------------------------- #
def test_feature_bounds_ok():
    feat = F.Featurizer(n_features=4)
    unit = MU.AtomicModelUnit(
        layer=0,
        component_type="block_input",
        indices_func=[0],
        featurizer=feat,
        feature_indices=[1, 2],
    )
    assert unit.get_feature_indices() == [1, 2]


def test_feature_bounds_violation():
    feat = F.SubspaceFeaturizer(shape=(4, 4), trainable=False)  # n_features = 4
    with pytest.raises(ValueError):
        MU.AtomicModelUnit(
            layer=0,
            component_type="block_input",
            indices_func=[0],
            featurizer=feat,
            feature_indices=[0, 5],
        )


def test_feature_bounds_after_featurizer_swap():
    big_feat = F.SubspaceFeaturizer(shape=(4, 4), trainable=False)  # n_features = 4
    small_feat = F.SubspaceFeaturizer(shape=(4, 2), trainable=False)  # n_features = 2

    # index 3 is valid for the 4-dim featurizer, but invalid for the 2-dim one
    unit = MU.AtomicModelUnit(
        layer=0,
        component_type="block_input",
        indices_func=[0],
        featurizer=big_feat,
        feature_indices=[3],
    )

    with pytest.raises(ValueError):
        unit.set_featurizer(small_feat)


# --------------------------------------------------------------------------- #
#  5. Static indices                                                          #
# --------------------------------------------------------------------------- #
def test_static_indices():
    unit = MU.AtomicModelUnit(
        layer=0,
        component_type="block_input",
        indices_func=[2, 3],
    )
    assert unit.index_component("ignored") == [2, 3]
    assert unit.is_static()


def test_dynamic_indices():
    indexer = MU.ComponentIndexer(lambda x: [x], id="test")
    unit = MU.AtomicModelUnit(
        layer=0,
        component_type="block_input",
        indices_func=indexer,
    )
    assert unit.index_component(5) == [5]
    assert not unit.is_static()


# --------------------------------------------------------------------------- #
#  6. Intervention config structure                                            #
# --------------------------------------------------------------------------- #
def test_create_intervention_config():
    unit = make_test_unit()
    cfg = unit.create_intervention_config(group_key="grp", intervention_type="collect")

    assert cfg["component"] == "block_input"
    assert cfg["unit"] == "pos"
    assert cfg["layer"] == 0
    assert cfg["group_key"] == "grp"
    assert callable(cfg["intervention_type"])
