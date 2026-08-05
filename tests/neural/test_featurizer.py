"""Direct tests for ``causalab.neural.featurizer``.

The base :class:`Featurizer` wrapper pairs a forward feature map with an inverse
reconstruction module — a *feature space*. Concrete featurizers (subspace,
standardize, sae, umap, spline, ...) live in ``causalab.methods`` and override
the serialization hooks defined here.

This module covers the base wrapper, the identity modules, and the
:class:`ComposedFeaturizer`, without depending on ``causalab.methods``. What to
*do* in a feature space moved to ``causalab.neural.interventions`` when pyvene
was replaced, and is tested in ``tests/neural/test_interventions.py``; tests
that need a concrete subclass live in ``tests/methods/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from causalab.neural.featurizer import (
    ComposedFeaturizer,
    Featurizer,
    IdentityFeaturizerModule,
    IdentityInverseFeaturizerModule,
)


@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


# --------------------------------------------------------------------------- #
#  Featurizer (base wrapper)                                                  #
# --------------------------------------------------------------------------- #
class TestFeaturizerUnit:
    """Base :class:`Featurizer` wrapper construction, accessors, and I/O."""

    pytestmark = pytest.mark.unit

    def test_default_construction_uses_identity_modules(self) -> None:
        feat = Featurizer(n_features=4)
        assert isinstance(feat.featurizer, IdentityFeaturizerModule)
        assert isinstance(feat.inverse_featurizer, IdentityInverseFeaturizerModule)
        assert feat.n_features == 4
        assert feat.id == "null"
        assert feat.tie_masks is False

    def test_is_trivial_true_for_default(self) -> None:
        assert Featurizer().is_trivial() is True

    def test_is_trivial_false_when_id_set(self) -> None:
        assert Featurizer(id="some_feat").is_trivial() is False

    def test_identity_featurize_returns_input_and_none_error(
        self, rng: torch.Generator
    ) -> None:
        x = randn((2, 4), rng)
        feat = Featurizer(n_features=4)

        f, err = feat.featurize(x)
        assert err is None
        assert torch.equal(f, x)

    def test_to_dict_returns_none_for_trivial(self) -> None:
        feat = Featurizer(n_features=4)
        assert feat.to_dict() is None

    def test_to_dict_returns_payload_for_named_identity(self) -> None:
        feat = Featurizer(n_features=4, id="masked")
        data = feat.to_dict()
        assert data is not None
        assert data["model_info"]["featurizer_class"] == "IdentityFeaturizerModule"
        assert data["model_info"]["n_features"] == 4
        assert data["model_info"]["featurizer_id"] == "masked"

    def test_from_dict_reconstructs_identity(self) -> None:
        feat = Featurizer(n_features=4, id="masked")
        data = feat.to_dict()
        assert data is not None
        loaded = Featurizer.from_dict(data)
        assert isinstance(loaded.featurizer, IdentityFeaturizerModule)
        assert loaded.n_features == 4
        assert loaded.id == "masked"

    def test_from_dict_raises_on_unknown_class(self) -> None:
        with pytest.raises(ValueError, match="Unknown featurizer class"):
            Featurizer.from_dict(
                {
                    "model_info": {
                        "featurizer_class": "DefinitelyNotARealFeaturizerXYZ",
                        "n_features": 4,
                        "featurizer_id": "x",
                    }
                }
            )

    def test_save_load_identity_roundtrip(
        self, tmp_path: Path, rng: torch.Generator
    ) -> None:
        x = randn((2, 4), rng)
        feat = Featurizer(n_features=4)

        path_root = tmp_path / "unit"
        saved = feat.save_modules(str(path_root))
        assert saved == (
            f"{path_root}_featurizer",
            f"{path_root}_inverse_featurizer",
        )

        loaded = Featurizer.load_modules(str(path_root))
        f, err = loaded.featurize(x)
        x_rec = loaded.inverse_featurize(f, err)
        assert torch.equal(x, x_rec)


class TestFeaturizerProperty:
    """Round-trip and identity invariants on the base wrapper."""

    pytestmark = pytest.mark.property

    def test_identity_roundtrip_is_exact(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        feat = Featurizer(n_features=4)

        f, err = feat.featurize(x)
        x_rec = feat.inverse_featurize(f, err)
        assert torch.equal(x, x_rec)

    @pytest.mark.parametrize(
        "id_value,expected_trivial",
        [
            ("null", True),
            ("", False),
            ("subspace", False),
            ("sae", False),
        ],
    )
    def test_is_trivial_iff_id_is_null(
        self, id_value: str, expected_trivial: bool
    ) -> None:
        assert Featurizer(id=id_value).is_trivial() is expected_trivial


# --------------------------------------------------------------------------- #
#  Identity modules                                                           #
# --------------------------------------------------------------------------- #
class TestIdentityFeaturizerModuleUnit:
    """``IdentityFeaturizerModule`` is the default no-op forward map."""

    pytestmark = pytest.mark.unit

    def test_forward_returns_input_and_none(self, rng: torch.Generator) -> None:
        x = randn((3, 4), rng)
        mod = IdentityFeaturizerModule()
        out, err = mod(x)
        # Identity by object identity: the module returns the input tensor as-is.
        assert out is x
        assert err is None


class TestIdentityInverseFeaturizerModuleUnit:
    """``IdentityInverseFeaturizerModule`` is the paired no-op inverse."""

    pytestmark = pytest.mark.unit

    def test_forward_returns_input_unchanged(self, rng: torch.Generator) -> None:
        x = randn((3, 4), rng)
        mod = IdentityInverseFeaturizerModule()
        out = mod(x, None)
        assert out is x


# --------------------------------------------------------------------------- #
#  ComposedFeaturizer                                                         #
# --------------------------------------------------------------------------- #
class TestComposedFeaturizerUnit:
    """``ComposedFeaturizer`` chains stages and flattens nested compositions."""

    pytestmark = pytest.mark.unit

    def test_construction_from_identity_stages(self) -> None:
        a = Featurizer(n_features=4, id="a")
        b = Featurizer(n_features=4, id="b")
        composed = ComposedFeaturizer([a, b])
        assert len(composed.stages) == 2
        assert composed.n_features == 4
        assert composed.id == "a >> b"

    def test_rshift_flattens_nested_compositions(self) -> None:
        a = Featurizer(n_features=4, id="a")
        b = Featurizer(n_features=4, id="b")
        c = Featurizer(n_features=4, id="c")

        nested = (a >> b) >> c
        assert isinstance(nested, ComposedFeaturizer)
        assert len(nested.stages) == 3

        right = a >> (b >> c)
        assert len(right.stages) == 3

    def test_n_features_propagates_from_last_stage(self) -> None:
        a = Featurizer(n_features=8, id="a")
        b = Featurizer(n_features=4, id="b")
        composed = a >> b
        assert composed.n_features == 4

    def test_id_defaults_to_joined_stage_ids(self) -> None:
        a = Featurizer(n_features=4, id="alpha")
        b = Featurizer(n_features=4, id="beta")
        composed = a >> b
        assert composed.id == "alpha >> beta"

    @pytest.mark.xfail(
        reason="Source fix pending: ComposedFeaturizer.to_dict overloads `None` "
        "with two meanings (trivial vs. non-serializable stage).",
        strict=True,
    )
    def test_to_dict_raises_when_a_stage_is_non_serializable(self) -> None:
        # A trivial featurizer's to_dict() returns None — using one as a stage
        # currently makes the composed to_dict() return None, indistinguishable
        # from a trivial featurizer. The fix is to raise instead.
        trivial = Featurizer(n_features=4)  # id="null" → to_dict() returns None
        named = Featurizer(n_features=4, id="named")
        composed = trivial >> named
        with pytest.raises((NotImplementedError, ValueError)):
            composed.to_dict()


class TestComposedFeaturizerProperty:
    """Algebraic invariants of ``>>`` composition on identity stages."""

    pytestmark = pytest.mark.property

    def test_rshift_is_associative_on_stage_count(self) -> None:
        a = Featurizer(n_features=4, id="a")
        b = Featurizer(n_features=4, id="b")
        c = Featurizer(n_features=4, id="c")

        left = (a >> b) >> c
        right = a >> (b >> c)

        assert len(left.stages) == len(right.stages) == 3

    def test_error_list_length_equals_stage_count(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        stages = [Featurizer(n_features=4, id=f"s{i}") for i in range(3)]
        composed = stages[0] >> stages[1] >> stages[2]
        _, errors = composed.featurize(x)
        assert isinstance(errors, list)
        assert len(errors) == 3

    def test_identity_only_roundtrip_is_exact(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        a = Featurizer(n_features=4, id="a")
        b = Featurizer(n_features=4, id="b")
        composed = a >> b

        features, errors = composed.featurize(x)
        x_rec = composed.inverse_featurize(features, errors)
        assert torch.equal(x, x_rec)
