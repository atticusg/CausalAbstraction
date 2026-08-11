"""Tests for :mod:`causalab.neural.specs` — WU1 (#503) spec vocabulary +
serialization (design-of-record: ``docs/REBASE_CAUSALAB_ON_NNTERP.md`` Part 6).

Tier (``causalab/neural`` owes ``unit`` + ``property``; this module's former
``property`` tier — the spec↔unit equivalence suite that gated the #491
migration waves — was retired with the unit surface itself in the WU6
deletion sweep, #508):

* ``unit`` — construction validation (literal-position normalization, key /
  width guards, the FeaturizedSite ``feature_ids`` validation inherited by
  composition), functional updates (``with_featurizer`` /
  ``with_feature_ids`` / ``with_positions``), ``EditSpec``'s five-mode
  validation messages, the ``save_site_specs`` / ``load_site_specs``
  round-trip (identity, ``SubspaceFeaturizer``, ``ComposedFeaturizer``),
  the fail-loudly contract for non-serializable non-trivial featurizers,
  the JSON+safetensors-only file set (no pickle / ``torch.save`` artifact),
  and the legacy ``units_metadata.json`` loader branch against a frozen
  module-local copy of the retired legacy writer's on-disk format
  (:func:`_write_legacy_units_bundle`).
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Sequence

import pytest
import torch

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.featurizer import ComposedFeaturizer, Featurizer
from causalab.neural.head_view import HeadSite
from causalab.neural.site import Site
from causalab.neural.specs import (
    SITE_SPECS_FORMAT_VERSION,
    EditSpec,
    SiteSpec,
    load_site_specs,
    save_site_specs,
)
from causalab.neural.token_positions import ComponentIndexer


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #
def _subspace(in_dim: int = 16, k: int = 4, id: str = "subspace") -> SubspaceFeaturizer:
    torch.manual_seed(0)
    return SubspaceFeaturizer(shape=(in_dim, k), trainable=False, id=id)


def _fsite(**kwargs: Any) -> FeaturizedSite:
    return FeaturizedSite(Site("block_output", 1), **kwargs)


def _spec(
    positions: Any = (0,), key: str = "spec0", width: int | None = None, **kwargs: Any
) -> SiteSpec:
    return SiteSpec(_fsite(**kwargs), positions, key, width=width)


def _indexer(id: str = "last") -> ComponentIndexer:
    return ComponentIndexer(lambda _input: [0], id=id)


def _assert_featurizers_equivalent(
    got: Featurizer, expected: Featurizer, in_dim: int
) -> None:
    """Structural + functional featurizer equality (reconstruction can never
    be the same *object*, so ``FeaturizedSite`` equality does not apply)."""
    assert type(got) is type(expected)
    assert got.id == expected.id
    assert got.n_features == expected.n_features
    torch.manual_seed(1)
    x = torch.randn(5, in_dim)
    got_f, _ = got.featurize(x)
    expected_f, _ = expected.featurize(x)
    torch.testing.assert_close(got_f, expected_f)


# --------------------------------------------------------------------------- #
#  frozen legacy writer — the retired InterchangeTarget.save's on-disk format  #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class _LegacyUnitRecord:
    """One legacy unit's metadata row, in the retired writer's vocabulary.

    ``uid`` is the legacy unit id string; a per-head unit encodes its head
    index ONLY here (e.g. ``"AttentionHead(Layer-0,Head-2,Token-last)"``;
    residual: ``"ResidualStream(Layer-1,block_output,Token-last_token)"``).
    ``component_type`` is the pyvene-era component string: whole-sublayer ones
    are the Site vocabulary (``"block_output"``, ``"block_input"``,
    ``"mlp_output"``, ...); per-head ones are
    ``"head_attention_value_output"`` (AttentionHead),
    ``"head_value_output"``, ``"head_query_output"``. ``unit`` is ``"pos"``
    (``"h.pos"`` for heads); ``index_id`` is the position resolver's id.
    """

    uid: str
    layer: int
    component_type: str
    index_id: str
    unit: str = "pos"
    featurizer: Featurizer = dataclasses.field(default_factory=Featurizer)
    feature_indices: Sequence[int] | None = None
    shape: tuple[int, ...] | None = None


def _write_legacy_units_bundle(
    records: Sequence[_LegacyUnitRecord], parent_dir: str
) -> str:
    """Frozen copy of the retired legacy writer's on-disk format (the deleted
    ``InterchangeTarget.save``) — the ``load_site_specs`` legacy branch's
    compatibility contract, reproduced byte-compatibly so the format cannot
    drift now that the writer itself is gone (WU6, #508).

    Writes ``units_metadata.json``: a dict keyed by unit id string, each value
    ``{"id", "feature_indices", "layer", "component_type", "unit",
    "index_id", "featurizer_info": {"id", "n_features", "is_trivial"},
    "shape", "version": "2.0"}``. Non-trivial featurizers go to the
    ``featurizers.safetensors`` + ``featurizers.meta.json`` pair via
    ``save_nested({uid: featurizer.to_dict()}, dir, "featurizers")`` — only
    when at least one featurizer's ``to_dict()`` is not ``None``. This
    deliberately reproduces the legacy silent-drop hazard: a non-trivial
    featurizer whose ``to_dict()`` returns ``None`` is recorded as
    ``is_trivial=false`` but never written to the payload.
    """
    os.makedirs(parent_dir, exist_ok=True)

    all_metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        all_metadata[record.uid] = {
            "id": record.uid,
            "feature_indices": (
                [int(i) for i in record.feature_indices]
                if record.feature_indices is not None
                else None
            ),
            "layer": record.layer,
            "component_type": record.component_type,
            "unit": record.unit,
            "index_id": record.index_id,
            "featurizer_info": {
                "id": record.featurizer.id,
                "n_features": record.featurizer.n_features,
                "is_trivial": record.featurizer.is_trivial(),
            },
            "shape": None if record.shape is None else list(record.shape),
            "version": "2.0",
        }
    with open(os.path.join(parent_dir, "units_metadata.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)

    featurizers: dict[str, dict[str, Any]] = {}
    for record in records:
        featurizer_data = record.featurizer.to_dict()
        if featurizer_data is not None:
            featurizers[record.uid] = featurizer_data
    if featurizers:
        from causalab.io.nested_artifacts import save_nested

        save_nested(featurizers, parent_dir, "featurizers")

    return parent_dir


# --------------------------------------------------------------------------- #
#  unit — SiteSpec construction + functional updates                           #
# --------------------------------------------------------------------------- #
class TestSiteSpecUnit:
    pytestmark = pytest.mark.unit

    def test_literal_positions_normalize_to_tuple(self) -> None:
        assert _spec(positions=[0, 2]).positions == (0, 2)
        assert _spec(positions=range(2)).positions == (0, 1)
        assert _spec(positions=torch.tensor([1, 3])).positions == (1, 3)
        assert _spec(positions=None).positions is None
        resolver = _indexer()
        assert _spec(positions=resolver).positions is resolver

    def test_invalid_positions_refused(self) -> None:
        with pytest.raises(TypeError, match="positions must be"):
            _spec(positions="last")  # a name is not a resolver
        with pytest.raises(ValueError, match="must be 1-D"):
            _spec(positions=torch.zeros(2, 2, dtype=torch.long))
        with pytest.raises(TypeError, match="positions must be"):
            _spec(positions=object())
        # bytes/bytearray carry an `index` METHOD — the resolver duck-type
        # must not accept them (they'd fail confusingly at resolve time).
        with pytest.raises(TypeError, match="positions must be"):
            _spec(positions=b"\x00\x01")
        with pytest.raises(TypeError, match="positions must be"):
            _spec(positions=bytearray(b"\x00\x01"))

    def test_key_and_width_validated(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            _spec(key="")
        with pytest.raises(ValueError, match="non-empty string"):
            SiteSpec(_fsite(), (0,), key=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="width must be positive"):
            _spec(width=0)
        with pytest.raises(TypeError, match="width must be an int"):
            _spec(width="16")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="fsite must be a FeaturizedSite"):
            SiteSpec(Site("block_output", 1), (0,), key="k")  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        spec = _spec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.key = "other"  # type: ignore[misc]

    def test_feature_ids_validation_inherited_from_featurized_site(self) -> None:
        # The spec composes FeaturizedSite, so its feature_ids contract
        # (non-empty, unique, non-negative, bounded) applies with no code here.
        with pytest.raises(ValueError, match="non-empty"):
            _spec(feature_ids=())
        with pytest.raises(ValueError, match="out of range"):
            _spec(featurizer=_subspace(16, 4), feature_ids=(1, 9))

    def test_with_featurizer_is_functional_and_revalidates(self) -> None:
        spec = _spec(feature_ids=(1, 3))
        feat = _subspace(16, 4)
        updated = spec.with_featurizer(feat)
        assert updated is not spec
        assert updated.fsite.featurizer is feat
        assert updated.fsite.site == spec.fsite.site
        assert updated.fsite.feature_ids == (1, 3)
        assert (updated.key, updated.width, updated.positions) == (
            spec.key,
            spec.width,
            spec.positions,
        )
        assert spec.fsite.featurizer is not feat  # original untouched
        # set_featurizer's re-check, inherited: stale ids fail at attach time.
        with pytest.raises(ValueError, match="out of range"):
            _spec(feature_ids=(1, 9)).with_featurizer(_subspace(16, 4))

    def test_with_feature_ids_is_functional_and_validated(self) -> None:
        spec = _spec(featurizer=_subspace(16, 4))
        updated = spec.with_feature_ids([0, 2])
        assert updated.fsite.feature_ids == (0, 2)
        assert updated.fsite.featurizer is spec.fsite.featurizer
        assert spec.fsite.feature_ids is None  # original untouched
        assert updated.with_feature_ids(None).fsite.feature_ids is None
        # The engine validation is NOT weakened: an empty selection is not a
        # valid FeaturizedSite (the no-op-skip contract lives at the dataset
        # layer, not here).
        with pytest.raises(ValueError, match="non-empty"):
            spec.with_feature_ids([])

    def test_with_positions_is_a_shallow_view(self) -> None:
        # A shallow view: the returned spec shares this spec's `fsite` by
        # identity (featurizer + site), keeps `key` (so rows collected
        # through the view accumulate under the real spec) and `width`, and
        # swaps only the positions; the original spec is untouched.
        original = _indexer("a")
        spec = SiteSpec(_fsite(), original, key="k", width=16)
        other = _indexer("b")
        view = spec.with_positions(other)
        assert view is not spec
        assert view.fsite is spec.fsite
        assert (view.key, view.width) == (spec.key, spec.width)
        assert view.positions is other
        assert spec.positions is original  # original untouched


# --------------------------------------------------------------------------- #
#  unit — EditSpec: five-mode validation + needs_source                        #
# --------------------------------------------------------------------------- #
class TestEditSpecUnit:
    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"mode": "ablate"},
                "unknown mode 'ablate'; expected one of "
                "('interchange', 'interpolate', 'replace', 'add', 'noise')",
            ),
            ({"mode": "replace"}, "mode 'replace' needs a vector"),
            ({"mode": "add"}, "mode 'add' needs a vector"),
            ({"mode": "interpolate"}, "mode 'interpolate' needs interpolate_fn"),
        ],
        ids=["unknown-mode", "replace-no-vector", "add-no-vector", "interpolate-no-fn"],
    )
    def test_validation_messages(self, kwargs: dict[str, Any], message: str) -> None:
        # EditSpec's own validation contract: same refusals, same messages
        # (string-identical to the retired legacy unit-edit's by design —
        # pinned here as EditSpec's own).
        with pytest.raises(ValueError) as excinfo:
            EditSpec(_spec(), **kwargs)
        assert str(excinfo.value) == message

    def test_valid_modes_construct(self) -> None:
        spec = _spec()
        vector = torch.zeros(3)
        assert EditSpec(spec).mode == "interchange"
        assert EditSpec(spec, mode="replace", vector=vector).scale == 1.0
        assert EditSpec(spec, mode="add", vector=vector, scale=2.0).scale == 2.0
        interp = EditSpec(
            spec, mode="interpolate", interpolate_fn=lambda f, f_src, alpha: f
        )
        assert interp.interpolate_params == {}

    def test_noise_seed_defaults_to_private_stream_seed_zero(self) -> None:
        # The noise mode defaults a missing seed to 0 — the data form of the
        # retired legacy private-stream default (SeededNoise(0)).
        assert EditSpec(_spec(), mode="noise").seed == 0
        assert EditSpec(_spec(), mode="noise", seed=7).seed == 7
        assert EditSpec(_spec()).seed is None  # only the noise mode defaults it

    def test_needs_source(self) -> None:
        spec = _spec()
        vector = torch.zeros(3)
        assert EditSpec(spec, mode="interchange").needs_source
        assert EditSpec(
            spec, mode="interpolate", interpolate_fn=lambda f, f_src: f
        ).needs_source
        assert not EditSpec(spec, mode="replace", vector=vector).needs_source
        assert not EditSpec(spec, mode="add", vector=vector).needs_source
        assert not EditSpec(spec, mode="noise").needs_source

    def test_frozen_and_site_typed(self) -> None:
        edit = EditSpec(_spec())
        with pytest.raises(dataclasses.FrozenInstanceError):
            edit.mode = "replace"  # type: ignore[misc]
        with pytest.raises(TypeError, match="site must be a SiteSpec"):
            EditSpec(_fsite())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
#  unit — save/load round-trip (new format)                                    #
# --------------------------------------------------------------------------- #
class TestSaveLoadUnit:
    pytestmark = pytest.mark.unit

    @staticmethod
    def _bundle() -> tuple[list[SiteSpec], ComponentIndexer]:
        resolver = _indexer("last")
        composed = ComposedFeaturizer(
            [
                _subspace(16, 4, id="stage_rot"),
                Featurizer(n_features=4, id="stage_mask"),
            ]
        )
        specs = [
            SiteSpec(
                FeaturizedSite(Site("block_output", 1)), (0, 2), "plain", width=16
            ),
            SiteSpec(
                FeaturizedSite(Site("mlp_output", 0), _subspace(16, 4), (1, 3)),
                resolver,
                "rotated",
                width=16,
            ),
            SiteSpec(
                FeaturizedSite(
                    HeadSite(kind="attention_value", layer=1, head=2), composed
                ),
                None,
                "head",
            ),
        ]
        return specs, resolver

    def test_roundtrip_restores_specs(self, tmp_path) -> None:
        specs, resolver = self._bundle()
        save_site_specs(specs, str(tmp_path))
        loaded = load_site_specs(str(tmp_path), token_positions={"last": resolver})

        assert [spec.key for spec in loaded] == ["plain", "rotated", "head"]
        for got, expected in zip(loaded, specs):
            assert got.fsite.site == expected.fsite.site
            assert got.fsite.feature_ids == expected.fsite.feature_ids
            assert got.width == expected.width
            _assert_featurizers_equivalent(
                got.fsite.featurizer, expected.fsite.featurizer, in_dim=16
            )
        assert loaded[0].positions == (0, 2)  # literal rows are data
        assert loaded[1].positions is resolver  # named → rebound
        assert loaded[2].positions is None

    def test_load_without_token_positions_keeps_name_in_record(self, tmp_path) -> None:
        specs, _resolver = self._bundle()
        save_site_specs(specs, str(tmp_path))
        loaded = load_site_specs(str(tmp_path))
        assert loaded[1].positions is None  # named position stays unbound...
        with open(tmp_path / "sites.json") as f:
            payload = json.load(f)
        assert payload["format_version"] == SITE_SPECS_FORMAT_VERSION
        by_key = {record["key"]: record for record in payload["specs"]}
        assert by_key["rotated"]["positions"] == {"kind": "named", "name": "last"}
        assert by_key["plain"]["positions"] == {"kind": "literal", "positions": [0, 2]}
        assert by_key["head"]["site"] == {
            "type": "head_site",
            "kind": "attention_value",
            "layer": 1,
            "head": 2,
        }
        assert loaded[0].positions == (0, 2)  # literal restores regardless

    def test_file_set_is_json_and_safetensors_only(self, tmp_path) -> None:
        # The no-pickle acceptance pin: the new path writes exactly the JSON
        # record + the safetensors/meta featurizer pair — no torch.save/.pt.
        specs, _resolver = self._bundle()
        save_site_specs(specs, str(tmp_path))
        assert set(os.listdir(tmp_path)) == {
            "sites.json",
            "featurizers.safetensors",
            "featurizers.meta.json",
        }

    def test_trivial_only_bundle_writes_single_json(self, tmp_path) -> None:
        save_site_specs([_spec(key="only")], str(tmp_path))
        assert set(os.listdir(tmp_path)) == {"sites.json"}
        loaded = load_site_specs(str(tmp_path))
        assert loaded[0].fsite.featurizer.is_trivial()

    def test_duplicate_keys_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="duplicate spec keys"):
            save_site_specs([_spec(key="k"), _spec(key="k")], str(tmp_path))

    def test_nameless_resolver_refused(self, tmp_path) -> None:
        class NoName:
            def index(self, input: Any, batch: bool = False, is_original=None):
                return [0]

        with pytest.raises(ValueError, match="no `id` name"):
            save_site_specs([_spec(positions=NoName())], str(tmp_path))

    def test_unserializable_nontrivial_featurizer_fails_loudly(self, tmp_path) -> None:
        # The verified legacy silent-drop hazard: ComposedFeaturizer.to_dict
        # returns None when ANY stage's does (here a trivial identity stage).
        # The legacy save skipped it silently; the constructive save must not.
        composed = ComposedFeaturizer([_subspace(16, 4), Featurizer()])
        assert composed.to_dict() is None and not composed.is_trivial()
        with pytest.raises(ValueError, match="cannot be saved"):
            save_site_specs([_spec(featurizer=composed)], str(tmp_path))
        assert not os.path.exists(tmp_path / "sites.json")  # nothing half-written

    def test_missing_bundle_refused(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="sites.json"):
            load_site_specs(str(tmp_path))

    def test_format_version_mismatch_refused(self, tmp_path) -> None:
        save_site_specs([_spec(key="k")], str(tmp_path))
        path = tmp_path / "sites.json"
        payload = json.loads(path.read_text())
        payload["format_version"] = "0.9"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="format version '0.9'"):
            load_site_specs(str(tmp_path))

    def test_named_position_missing_from_mapping_refused(self, tmp_path) -> None:
        save_site_specs([_spec(positions=_indexer("last"), key="k")], str(tmp_path))
        with pytest.raises(ValueError, match="'last' not found"):
            load_site_specs(str(tmp_path), token_positions={"other": _indexer("other")})

    def test_records_carry_featurizer_presence_flag(self, tmp_path) -> None:
        specs, _resolver = self._bundle()
        save_site_specs(specs, str(tmp_path))
        with open(tmp_path / "sites.json") as f:
            by_key = {record["key"]: record for record in json.load(f)["specs"]}
        assert by_key["plain"]["featurizer"] is False
        assert by_key["rotated"]["featurizer"] is True
        assert by_key["head"]["featurizer"] is True

    def test_truncated_bundle_missing_featurizer_payload_refused(
        self, tmp_path
    ) -> None:
        # The read-side of the silent-drop hazard: a bundle whose record
        # claims a non-trivial featurizer but whose payload files are gone
        # must refuse loudly, not silently reconstruct identity featurizers.
        specs, _resolver = self._bundle()
        save_site_specs(specs, str(tmp_path))
        os.remove(tmp_path / "featurizers.safetensors")
        with pytest.raises(ValueError, match="truncated or partially deleted"):
            load_site_specs(str(tmp_path))

    def test_save_writes_payload_before_commit_point(
        self, tmp_path, monkeypatch
    ) -> None:
        # sites.json is the bundle's commit point: a save that dies during
        # the featurizer payload write must leave no sites.json behind.
        import causalab.io.nested_artifacts as nested_artifacts

        def boom(payload, output_dir, stem, extra_meta=None):
            raise RuntimeError("simulated payload write failure")

        monkeypatch.setattr(nested_artifacts, "save_nested", boom)
        specs, _resolver = self._bundle()
        with pytest.raises(RuntimeError, match="simulated payload write"):
            save_site_specs(specs, str(tmp_path))
        assert not os.path.exists(tmp_path / "sites.json")

    def test_both_formats_present_new_format_wins(self, tmp_path) -> None:
        # Precedence pin: a directory holding BOTH bundle formats loads the
        # new sites.json, never the legacy metadata.
        _write_legacy_units_bundle(
            [
                _LegacyUnitRecord(
                    uid="ResidualStream(Layer-0,block_input,Token-[0])",
                    layer=0,
                    component_type="block_input",
                    index_id="constant_[0]",
                )
            ],
            str(tmp_path),
        )
        save_site_specs([_spec(key="new-format-key")], str(tmp_path))
        loaded = load_site_specs(str(tmp_path))
        assert [spec.key for spec in loaded] == ["new-format-key"]

    def test_unknown_site_record_type_refused(self, tmp_path) -> None:
        save_site_specs([_spec(key="k")], str(tmp_path))
        path = tmp_path / "sites.json"
        payload = json.loads(path.read_text())
        payload["specs"][0]["site"]["type"] = "weird"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="unknown site record type 'weird'"):
            load_site_specs(str(tmp_path))

    def test_unknown_positions_kind_refused(self, tmp_path) -> None:
        save_site_specs([_spec(key="k")], str(tmp_path))
        path = tmp_path / "sites.json"
        payload = json.loads(path.read_text())
        payload["specs"][0]["positions"] = {"kind": "weird"}
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="unknown positions record kind 'weird'"):
            load_site_specs(str(tmp_path))


# --------------------------------------------------------------------------- #
#  unit — the legacy units_metadata.json loader branch                         #
# --------------------------------------------------------------------------- #
class TestLegacyLoadUnit:
    pytestmark = pytest.mark.unit

    @staticmethod
    def _legacy_bundle(
        tmp_path,
    ) -> tuple[list[_LegacyUnitRecord], dict[str, Any]]:
        """A frozen-writer bundle covering every retired legacy unit class
        (residual / MLP / whole-attention / the three per-head kinds), a
        non-trivial featurizer + feature subset, and a shape."""
        records = [
            _LegacyUnitRecord(
                uid="ResidualStream(Layer-1,block_output,Token-last)",
                layer=1,
                component_type="block_output",
                index_id="last",
                featurizer=_subspace(16, 4),
                feature_indices=[1, 3],
                shape=(16,),
            ),
            _LegacyUnitRecord(
                uid="MLP(Layer-0,mlp_input,Token-[0])",
                layer=0,
                component_type="mlp_input",
                index_id="constant_[0]",
            ),
            _LegacyUnitRecord(
                uid="AttentionOutput(Layer-1,attention_output,Token-last)",
                layer=1,
                component_type="attention_output",
                index_id="last",
                shape=(16,),
            ),
            _LegacyUnitRecord(
                uid="AttentionHead(Layer-0,Head-2,Token-last)",
                layer=0,
                component_type="head_attention_value_output",
                index_id="last",
                unit="h.pos",
            ),
            _LegacyUnitRecord(
                uid="AttentionHeadValue(Layer-1,Head-0,Token-last)",
                layer=1,
                component_type="head_value_output",
                index_id="last",
                unit="h.pos",
            ),
            _LegacyUnitRecord(
                uid="AttentionHeadQuery(Layer-1,Head-1,Token-last)",
                layer=1,
                component_type="head_query_output",
                index_id="last",
                unit="h.pos",
            ),
        ]
        _write_legacy_units_bundle(records, str(tmp_path))
        token_positions = {
            "last": ComponentIndexer(lambda _input: [0], id="last"),
            "constant_[0]": ComponentIndexer(lambda _input: [0], id="constant_[0]"),
        }
        return records, token_positions

    #: The engine site each legacy record must load as — per-head components
    #: through the retired adapter's table (with the head index recovered
    #: from the id string), whole-sublayer components 1:1.
    _EXPECTED_SITES = [
        Site("block_output", 1),
        Site("mlp_input", 0),
        Site("attention_output", 1),
        HeadSite(kind="attention_value", layer=0, head=2),
        HeadSite(kind="value", layer=1, head=0),
        HeadSite(kind="query", layer=1, head=1),
    ]

    def test_legacy_bundle_loads_equivalent_specs(self, tmp_path) -> None:
        records, token_positions = self._legacy_bundle(tmp_path)
        specs = load_site_specs(str(tmp_path), token_positions=token_positions)

        assert [spec.key for spec in specs] == [record.uid for record in records]
        for spec, record, expected_site in zip(specs, records, self._EXPECTED_SITES):
            # The acceptance pin: component mapping matches the retired
            # adapter's exactly (incl. the head index, recovered from the
            # legacy id).
            assert spec.fsite.site == expected_site
            expected_ids = (
                None
                if record.feature_indices is None
                else tuple(record.feature_indices)
            )
            assert spec.fsite.feature_ids == expected_ids
            _assert_featurizers_equivalent(
                spec.fsite.featurizer, record.featurizer, in_dim=16
            )
            assert spec.positions is token_positions[record.index_id]
            expected_width = None if record.shape is None else record.shape[0]
            assert spec.width == expected_width

    def test_legacy_load_without_token_positions_gives_unbound(self, tmp_path) -> None:
        _records, _token_positions = self._legacy_bundle(tmp_path)
        specs = load_site_specs(str(tmp_path))
        assert all(spec.positions is None for spec in specs)

    def test_legacy_version_mismatch_refused(self, tmp_path) -> None:
        self._legacy_bundle(tmp_path)
        path = tmp_path / "units_metadata.json"
        metadata = json.loads(path.read_text())
        next(iter(metadata.values()))["version"] = "1.0"
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="version '1.0'"):
            load_site_specs(str(tmp_path))

    def test_legacy_empty_feature_indices_refused(self, tmp_path) -> None:
        # A DBM mask that switched every feature off could legally write
        # feature_indices=[] in a legacy bundle; the engine refuses empty
        # selections by design, so the constructive loader refuses with a
        # clear pointer (this behavior informs the WU6 artifact census).
        self._legacy_bundle(tmp_path)
        path = tmp_path / "units_metadata.json"
        metadata = json.loads(path.read_text())
        mlp_uid = next(uid for uid in metadata if uid.startswith("MLP("))
        metadata[mlp_uid]["feature_indices"] = []
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="empty feature_indices"):
            load_site_specs(str(tmp_path))

    def test_legacy_missing_featurizer_payload_refused(self, tmp_path) -> None:
        # featurizer_info.is_trivial=false with no payload entry means a
        # truncated bundle OR a historic silent-drop bundle (the legacy save
        # skipped featurizers whose to_dict() returned None) — refuse both.
        self._legacy_bundle(tmp_path)
        os.remove(tmp_path / "featurizers.safetensors")
        with pytest.raises(ValueError, match="is_trivial=false"):
            load_site_specs(str(tmp_path))

    def test_legacy_silent_drop_bundle_refused(self, tmp_path) -> None:
        # The reviewer-reproduced shape, exactly: the LEGACY save itself
        # silently dropped a non-trivial featurizer whose to_dict() returns
        # None (a ComposedFeaturizer with a trivial identity stage is the
        # in-tree case) — the frozen writer reproduces that hazard — so the
        # bundle records featurizer_info.is_trivial=false but NEVER writes
        # featurizer payload files. The constructive load must refuse —
        # before this guard it silently returned identity-featurizer specs.
        composed = ComposedFeaturizer([_subspace(16, 4), Featurizer()])
        record = _LegacyUnitRecord(
            uid="ResidualStream(Layer-1,block_input,Token-[0])",
            layer=1,
            component_type="block_input",
            index_id="constant_[0]",
            featurizer=composed,
        )
        _write_legacy_units_bundle([record], str(tmp_path))
        assert set(os.listdir(tmp_path)) == {"units_metadata.json"}  # no payload
        metadata = json.loads((tmp_path / "units_metadata.json").read_text())
        assert metadata[record.uid]["featurizer_info"]["is_trivial"] is False
        with pytest.raises(ValueError, match="silently dropped"):
            load_site_specs(str(tmp_path))

    def test_legacy_head_unit_without_head_in_id_refused(self, tmp_path) -> None:
        self._legacy_bundle(tmp_path)
        path = tmp_path / "units_metadata.json"
        metadata = json.loads(path.read_text())
        head_uid = next(uid for uid in metadata if uid.startswith("AttentionHead("))
        record = metadata.pop(head_uid)
        record["id"] = "Custom(no-head)"
        metadata["Custom(no-head)"] = record
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="does not encode a head index"):
            load_site_specs(str(tmp_path))

    def test_legacy_unmapped_head_component_refused(self, tmp_path) -> None:
        # The retired adapter's own refusal, reproduced by the legacy branch.
        self._legacy_bundle(tmp_path)
        path = tmp_path / "units_metadata.json"
        metadata = json.loads(path.read_text())
        head_uid = next(uid for uid in metadata if uid.startswith("AttentionHead("))
        metadata[head_uid]["component_type"] = "head_attention_value_input"
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="no HeadSite mapping"):
            load_site_specs(str(tmp_path))
