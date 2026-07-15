"""Direct tests for ``causalab.neural.featurizer``.

The base :class:`Featurizer` wrapper pairs a forward feature map with an
inverse reconstruction module and exposes intervention-class factories
(interchange / collect / mask / steering / replace / interpolation) that the
pyvene-backed intervention pipeline in ``causalab.neural.units`` and
``causalab.neural.activations`` consumes downstream. Concrete featurizers
(subspace, standardize, sae, umap, spline, ...) live in ``causalab.methods``
and override the serialization hooks defined here.

This module covers the base wrapper, the identity modules, the
:class:`ComposedFeaturizer`, and the six ``build_feature_*_intervention``
factories — all without depending on ``causalab.methods``. Tests that need a
concrete subclass (subspace, standardize) live in ``tests/methods/``.
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
    build_feature_collect_intervention,
    build_feature_interchange_intervention,
    build_feature_interpolation_intervention,
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

    def test_get_mask_intervention_raises_without_n_features(self) -> None:
        feat = Featurizer()  # n_features=None
        with pytest.raises(ValueError):
            _ = feat.get_mask_intervention()

    def test_get_collect_intervention_str_contains_id(self) -> None:
        feat = Featurizer(n_features=4, id="my_feat")
        Collect = feat.get_collect_intervention()
        col = Collect()
        assert "my_feat" in str(col)

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


# --------------------------------------------------------------------------- #
#  Intervention factories                                                     #
# --------------------------------------------------------------------------- #
class TestBuildFeatureInterchangeInterventionUnit:
    """``build_feature_interchange_intervention`` swaps in feature space."""

    pytestmark = pytest.mark.unit

    def test_subspaces_none_swaps_fully(self, rng: torch.Generator) -> None:
        x_base = randn((2, 4), rng)
        x_src = randn((2, 4), rng)

        feat = Featurizer(n_features=4)
        Interchange = feat.get_interchange_intervention()
        inter = Interchange()

        out = inter(x_base, x_src, subspaces=None)
        assert torch.equal(out, x_src)

    def test_str_carries_featurizer_id(self) -> None:
        Cls = build_feature_interchange_intervention(
            IdentityFeaturizerModule(),
            IdentityInverseFeaturizerModule(),
            featurizer_id="my_id_42",
        )
        assert "my_id_42" in str(Cls())


class TestBuildFeatureCollectInterventionUnit:
    """``build_feature_collect_intervention`` returns a CollectIntervention."""

    pytestmark = pytest.mark.unit

    def test_str_carries_featurizer_id(self) -> None:
        Cls = build_feature_collect_intervention(
            IdentityFeaturizerModule(), featurizer_id="collected"
        )
        assert "collected" in str(Cls())


class TestBuildFeatureMaskInterventionUnit:
    """``build_feature_mask_intervention`` and its forward/eval/training paths."""

    pytestmark = pytest.mark.unit

    def test_forward_raises_without_temperature(self, rng: torch.Generator) -> None:
        x_base = randn((1, 4), rng)
        x_src = randn((1, 4), rng)
        feat = Featurizer(n_features=4)
        mask = feat.get_mask_intervention()()

        with pytest.raises(ValueError, match="temperature"):
            mask(x_base, x_src)

    def test_get_sparsity_loss_raises_without_temperature(self) -> None:
        feat = Featurizer(n_features=4)
        mask = feat.get_mask_intervention()()
        with pytest.raises(ValueError, match="Temperature"):
            mask.get_sparsity_loss()

    def test_training_mode_saturated_mask_yields_source(
        self, rng: torch.Generator
    ) -> None:
        x_base = randn((1, 4), rng)
        x_src = randn((1, 4), rng)
        feat = Featurizer(n_features=4)
        mask = feat.get_mask_intervention()()

        mask.set_temperature(1.0)
        mask.train()
        mask.mask.data.fill_(20.0)  # sigmoid(20) ≈ 1 − 2e-9
        out = mask(x_base, x_src)
        assert torch.allclose(out, x_src, atol=1e-6)

    def test_eval_mode_is_binary_gate(self, rng: torch.Generator) -> None:
        x_base = randn((1, 4), rng)
        x_src = randn((1, 4), rng)
        feat = Featurizer(n_features=4)
        mask = feat.get_mask_intervention()()

        mask.set_temperature(1.0)
        mask.eval()
        mask.mask.data.fill_(1.0)  # > 0 → gate = 1 → output = source
        out = mask(x_base, x_src)
        assert torch.allclose(out, x_src, atol=1e-5)

    def test_tied_masks_produces_scalar_parameter(self) -> None:
        feat = Featurizer(n_features=8, tie_masks=True)
        mask = feat.get_mask_intervention()()
        assert mask.mask.shape == (1,)

    def test_untied_masks_produces_per_feature_parameter(self) -> None:
        feat = Featurizer(n_features=8, tie_masks=False)
        mask = feat.get_mask_intervention()()
        assert mask.mask.shape == (8,)

    def test_str_carries_id_and_tied_suffix(self) -> None:
        feat_tied = Featurizer(n_features=4, id="t", tie_masks=True)
        feat_untied = Featurizer(n_features=4, id="u", tie_masks=False)

        s_tied = str(feat_tied.get_mask_intervention()())
        s_untied = str(feat_untied.get_mask_intervention()())

        assert "t" in s_tied
        assert ",tied" in s_tied
        assert "u" in s_untied
        assert ",tied" not in s_untied


class TestBuildFeatureMaskInterventionProperty:
    """Algebraic invariants of the mask gate."""

    pytestmark = pytest.mark.property

    def test_train_eval_agree_at_saturated_limit(self, rng: torch.Generator) -> None:
        x_base = randn((1, 4), rng)
        x_src = randn((1, 4), rng)
        feat = Featurizer(n_features=4)
        MaskCls = feat.get_mask_intervention()

        mask_train = MaskCls()
        mask_train.set_temperature(1.0)
        mask_train.train()
        mask_train.mask.data.fill_(20.0)
        out_train = mask_train(x_base, x_src)

        mask_eval = MaskCls()
        mask_eval.set_temperature(1.0)
        mask_eval.eval()
        mask_eval.mask.data.fill_(20.0)
        out_eval = mask_eval(x_base, x_src)

        assert torch.allclose(out_train, out_eval, atol=1e-6)
        assert torch.allclose(out_train, x_src, atol=1e-6)

    def test_gate_monotone_in_mask_value(self, rng: torch.Generator) -> None:
        """As ``mask`` grows, training-mode output moves from base toward source."""
        x_base = randn((1, 4), rng)
        x_src = randn((1, 4), rng)
        feat = Featurizer(n_features=4)
        MaskCls = feat.get_mask_intervention()

        def output_for(mask_value: float) -> torch.Tensor:
            m = MaskCls()
            m.set_temperature(1.0)
            m.train()
            m.mask.data.fill_(mask_value)
            return m(x_base, x_src)

        # mask=-5 → gate ≈ 0 → output ≈ base
        # mask= 0 → gate = 0.5 → output halfway
        # mask= 5 → gate ≈ 1 → output ≈ source
        out_low = output_for(-5.0)
        out_mid = output_for(0.0)
        out_high = output_for(5.0)

        dist_low = torch.norm(out_low - x_base)
        dist_mid = torch.norm(out_mid - x_base)
        dist_high = torch.norm(out_high - x_base)
        assert dist_low < dist_mid < dist_high


class TestBuildFeatureSteeringInterventionUnit:
    """``build_feature_steering_intervention`` adds in feature space."""

    pytestmark = pytest.mark.unit

    def test_zero_steering_preserves_base(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        zero = torch.zeros(2, 4)
        feat = Featurizer(n_features=4)
        Steering = feat.get_steering_intervention()
        steer = Steering()
        out = steer(x, zero)
        assert torch.allclose(out, x, atol=1e-6)

    def test_nonzero_steering_shifts_by_exact_vector(
        self, rng: torch.Generator
    ) -> None:
        x = randn((2, 4), rng)
        vec = randn((2, 4), rng)
        feat = Featurizer(n_features=4)
        steer = feat.get_steering_intervention()()
        out = steer(x, vec)
        # Identity featurizer ⇒ output is base + vec verbatim.
        assert torch.allclose(out, x + vec, atol=1e-6)


class TestBuildFeatureNoiseInterventionUnit:
    """``build_feature_noise_intervention`` adds seeded scaled noise."""

    pytestmark = pytest.mark.unit

    def test_zero_scale_preserves_base(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        zero = torch.zeros(2, 4)
        feat = Featurizer(n_features=4)
        out = feat.get_noise_intervention(seed=0)()(x, zero)
        assert torch.allclose(out, x, atol=1e-6)

    def test_noise_is_seeded_and_reproducible(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        scale = torch.ones(2, 4)
        feat = Featurizer(n_features=4)
        out_a = feat.get_noise_intervention(seed=7)()(x, scale)
        out_b = feat.get_noise_intervention(seed=7)()(x, scale)
        assert torch.allclose(out_a, out_b, atol=1e-6)
        # Identity featurizer ⇒ out = base + scale * randn(seed=7).
        expected_noise = torch.randn(
            x.shape, generator=torch.Generator().manual_seed(7)
        )
        assert torch.allclose(out_a, x + expected_noise, atol=1e-6)

    def test_distinct_seeds_give_distinct_noise(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        scale = torch.ones(2, 4)
        feat = Featurizer(n_features=4)
        out_a = feat.get_noise_intervention(seed=1)()(x, scale)
        out_b = feat.get_noise_intervention(seed=2)()(x, scale)
        assert not torch.allclose(out_a, out_b, atol=1e-6)

    def test_consecutive_calls_advance_but_instances_reproduce(
        self, rng: torch.Generator
    ) -> None:
        """Within one instance the noise stream advances (independent batches);
        a fresh instance with the same seed replays the identical sequence.

        This is the property causal tracing relies on: corruption is i.i.d.
        across batches/examples within a sweep, yet identical across grid cells
        (each cell rebuilds the intervention) so cells stay comparable. A fixed
        re-seed every call would instead hand same-shape batches identical noise.
        """
        x = randn((2, 4), rng)
        scale = torch.ones(2, 4)
        feat = Featurizer(n_features=4)

        a = feat.get_noise_intervention(seed=5)()
        a1 = a(x, scale)
        a2 = a(x, scale)
        # Consecutive draws on the same instance are independent.
        assert not torch.allclose(a1, a2, atol=1e-6)

        # A fresh instance (same seed) reproduces the whole sequence.
        b = feat.get_noise_intervention(seed=5)()
        b1 = b(x, scale)
        b2 = b(x, scale)
        assert torch.allclose(a1, b1, atol=1e-6)
        assert torch.allclose(a2, b2, atol=1e-6)

    def test_scale_controls_magnitude(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng)
        feat = Featurizer(n_features=4)
        small = feat.get_noise_intervention(seed=3)()(x, torch.full((2, 4), 0.1))
        large = feat.get_noise_intervention(seed=3)()(x, torch.full((2, 4), 10.0))
        # Same seed ⇒ same noise direction; larger scale ⇒ larger deviation.
        assert (large - x).abs().sum() > (small - x).abs().sum()

    def test_output_dtype_matches_base_dtype(self, rng: torch.Generator) -> None:
        x = randn((2, 4), rng).to(torch.float64)
        scale = torch.ones(2, 4, dtype=torch.float32)
        feat = Featurizer(n_features=4)
        out = feat.get_noise_intervention(seed=0)()(x, scale)
        assert out.dtype == x.dtype


class TestBuildFeatureReplaceInterventionUnit:
    """``build_feature_replace_intervention`` overwrites in feature space."""

    pytestmark = pytest.mark.unit

    def test_replacement_becomes_output(self, rng: torch.Generator) -> None:
        x_base = randn((2, 4), rng)
        replacement = randn((2, 4), rng)
        feat = Featurizer(n_features=4)
        replace = feat.get_replace_intervention()()
        out = replace(x_base, replacement)
        # Identity featurizer ⇒ inverse(replacement, None) = replacement.
        assert torch.allclose(out, replacement, atol=1e-6)

    def test_base_is_ignored_at_feature_level(self, rng: torch.Generator) -> None:
        replacement = randn((2, 4), rng)
        feat = Featurizer(n_features=4)
        replace = feat.get_replace_intervention()()

        out_a = replace(randn((2, 4), rng), replacement)
        out_b = replace(randn((2, 4), rng), replacement)
        # Both invocations should yield the same output despite different bases.
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_output_dtype_matches_base_dtype(self, rng: torch.Generator) -> None:
        x_base = randn((2, 4), rng).to(torch.float64)
        replacement = randn((2, 4), rng).to(torch.float32)
        feat = Featurizer(n_features=4)
        replace = feat.get_replace_intervention()()
        out = replace(x_base, replacement)
        assert out.dtype == x_base.dtype


class TestBuildFeatureInterpolationInterventionUnit:
    """``build_feature_interpolation_intervention`` and ``set_interpolation``."""

    pytestmark = pytest.mark.unit

    def test_forward_raises_before_set_interpolation(
        self, rng: torch.Generator
    ) -> None:
        x = randn((2, 4), rng)
        feat = Featurizer(n_features=4)
        interp = feat.get_interpolation_intervention()()
        with pytest.raises(ValueError, match="Interpolation function not set"):
            interp(x, x)

    def test_linear_interp_alpha_one_yields_source(self, rng: torch.Generator) -> None:
        x_base = randn((2, 4), rng)
        x_src = randn((2, 4), rng)
        feat = Featurizer(n_features=4)
        interp = feat.get_interpolation_intervention()()

        def linear(
            *, f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            return (1 - alpha) * f_base + alpha * f_src

        interp.set_interpolation(linear, alpha=1.0)
        out = interp(x_base, x_src)
        assert torch.allclose(out, x_src, atol=1e-6)

    def test_linear_interp_alpha_zero_yields_base(self, rng: torch.Generator) -> None:
        x_base = randn((2, 4), rng)
        x_src = randn((2, 4), rng)
        feat = Featurizer(n_features=4)
        interp = feat.get_interpolation_intervention()()

        def linear(
            *, f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            return (1 - alpha) * f_base + alpha * f_src

        interp.set_interpolation(linear, alpha=0.0)
        out = interp(x_base, x_src)
        assert torch.allclose(out, x_base, atol=1e-6)

    def test_str_carries_featurizer_id(self) -> None:
        Cls = build_feature_interpolation_intervention(
            IdentityFeaturizerModule(),
            IdentityInverseFeaturizerModule(),
            featurizer_id="interp_id",
        )
        assert "interp_id" in str(Cls())
