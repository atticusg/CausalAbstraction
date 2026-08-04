"""Intervention modes: the plain-torch modes must equal the pyvene ones they replace.

``causalab.neural.interventions`` replaces the seven
``build_feature_*_intervention`` factories in ``causalab.neural.featurizer``,
which synthesized pyvene ``TrainableIntervention`` subclasses at runtime. The
first class below runs both implementations on the same inputs and asserts they
agree — the migration's behavioural contract for what gets *written* at a site,
independent of how the site is hooked.

**That class is scheduled for deletion**: it imports the pyvene builders, so it
goes when they do. The classes after it test the new modes on their own terms
(error preservation, feature selection, the noise stream, mask semantics) and are
the permanent coverage.
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.featurizer import (
    Featurizer,
    build_feature_collect_intervention,
    build_feature_interchange_intervention,
    build_feature_interpolation_intervention,
    build_feature_mask_intervention,
    build_feature_noise_intervention,
    build_feature_replace_intervention,
    build_feature_steering_intervention,
)
from causalab.neural.interventions import build_intervention

pytestmark = pytest.mark.unit

D = 8  # activation width
K = 3  # subspace rank
BATCH, POS = 2, 4


def _rotation_featurizer(seed: int = 0) -> SubspaceFeaturizer:
    """A rank-K subspace featurizer — non-trivial, so error preservation matters."""
    torch.manual_seed(seed)
    return SubspaceFeaturizer(shape=(D, K), trainable=False, id="test_subspace")


def _identity_featurizer(n_features: int = D) -> Featurizer:
    return Featurizer(n_features=n_features, id="test_identity")


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return torch.randn(BATCH, POS, D), torch.randn(BATCH, POS, D)


# --------------------------------------------------------------------------- #
#  Migration parity — delete with the pyvene builders                          #
# --------------------------------------------------------------------------- #
class TestMatchesPyveneIntervention:
    """Old and new must write the same tensor. Temporary: see the module docstring."""

    def test_interchange(self) -> None:
        featurizer = _rotation_featurizer()
        base, source = _inputs()
        old = build_feature_interchange_intervention(
            featurizer.featurizer, featurizer.inverse_featurizer, featurizer.id
        )(embed_dim=D)
        new = build_intervention(featurizer, "interchange")
        torch.testing.assert_close(new(base, source), old(base, source))

    def test_collect(self) -> None:
        featurizer = _rotation_featurizer()
        base, _ = _inputs()
        old = build_feature_collect_intervention(featurizer.featurizer, featurizer.id)(
            embed_dim=D
        )
        new = build_intervention(featurizer, "collect")
        torch.testing.assert_close(new(base), old(base))

    def test_steering(self) -> None:
        featurizer = _rotation_featurizer()
        base, _ = _inputs()
        vector = torch.randn(K)
        old = build_feature_steering_intervention(
            featurizer.featurizer, featurizer.inverse_featurizer, featurizer.id
        )(embed_dim=D)
        new = build_intervention(featurizer, "add")
        torch.testing.assert_close(new(base, vector), old(base, vector))

    def test_replace(self) -> None:
        featurizer = _rotation_featurizer()
        base, _ = _inputs()
        vector = torch.randn(K)
        old = build_feature_replace_intervention(
            featurizer.featurizer, featurizer.inverse_featurizer, featurizer.id
        )(embed_dim=D)
        new = build_intervention(featurizer, "replace")
        torch.testing.assert_close(new(base, vector), old(base, vector))

    def test_noise_same_seed_gives_same_draw(self) -> None:
        featurizer = _rotation_featurizer()
        base, _ = _inputs()
        scale = torch.tensor(0.5)
        old = build_feature_noise_intervention(
            featurizer.featurizer, featurizer.inverse_featurizer, featurizer.id, 7
        )(embed_dim=D)
        new = build_intervention(featurizer, "noise", seed=7)
        torch.testing.assert_close(new(base, scale), old(base, scale))

    def test_interpolation(self) -> None:
        featurizer = _rotation_featurizer()
        base, source = _inputs()

        def lerp(f_base: torch.Tensor, f_src: torch.Tensor, alpha: float):
            return (1 - alpha) * f_base + alpha * f_src

        old = build_feature_interpolation_intervention(
            featurizer.featurizer, featurizer.inverse_featurizer, featurizer.id
        )(embed_dim=D)
        old.set_interpolation(lerp, alpha=0.25)
        new = build_intervention(featurizer, "interpolation")
        new.set_interpolation(lerp, alpha=0.25)
        torch.testing.assert_close(new(base, source), old(base, source))

    @pytest.mark.parametrize("tie_masks", [False, True])
    @pytest.mark.parametrize("training", [False, True])
    def test_mask(self, tie_masks: bool, training: bool) -> None:
        featurizer = _identity_featurizer()
        base, source = _inputs()
        old = build_feature_mask_intervention(
            featurizer.featurizer,
            featurizer.inverse_featurizer,
            D,
            featurizer.id,
            tie_masks,
        )(embed_dim=D)
        new = build_intervention(featurizer, "mask", tie_masks=tie_masks)

        torch.manual_seed(1)
        weights = torch.randn(1 if tie_masks else D)
        with torch.no_grad():
            old.mask.copy_(weights)
            new.mask.copy_(weights)
        old.set_temperature(0.3)
        new.set_temperature(0.3)
        old.train(training)
        new.train(training)

        torch.testing.assert_close(new(base, source), old(base, source))
        torch.testing.assert_close(new.get_sparsity_loss(), old.get_sparsity_loss())


# --------------------------------------------------------------------------- #
#  Permanent coverage                                                          #
# --------------------------------------------------------------------------- #
class TestErrorPreservation:
    """Intervening in a rank-K subspace must leave the orthogonal complement alone.

    This is what makes a subspace result interpretable: if the write also
    perturbed the complement, a behavioural change could not be attributed to the
    subspace.
    """

    @pytest.mark.parametrize("mode", ["interchange", "add", "replace"])
    def test_orthogonal_complement_is_untouched(self, mode: str) -> None:
        featurizer = _rotation_featurizer()
        base, source = _inputs()
        payload = source if mode == "interchange" else torch.randn(K)
        out = build_intervention(featurizer, mode)(base, payload)

        # The complement is what the featurizer cannot represent: x - R Rᵀ x.
        _f_base, base_error = featurizer.featurize(base)
        _f_out, out_error = featurizer.featurize(out)
        torch.testing.assert_close(out_error, base_error, atol=1e-5, rtol=1e-4)
        # ...and the write actually changed the in-subspace part.
        assert not torch.allclose(out, base, atol=1e-4)


class TestFeatureIndices:
    """``feature_indices`` restricts the write to a subset of feature dimensions."""

    def test_only_selected_features_change(self) -> None:
        featurizer = _identity_featurizer()
        base, source = _inputs()
        selected = [0, 3]
        out = build_intervention(featurizer, "interchange")(base, source, selected)

        untouched = [i for i in range(D) if i not in selected]
        torch.testing.assert_close(out[..., selected], source[..., selected])
        torch.testing.assert_close(out[..., untouched], base[..., untouched])

    def test_none_swaps_every_feature(self) -> None:
        featurizer = _identity_featurizer()
        base, source = _inputs()
        out = build_intervention(featurizer, "interchange")(base, source, None)
        torch.testing.assert_close(out, source)

    def test_collect_returns_only_selected_features(self) -> None:
        featurizer = _identity_featurizer()
        base, _ = _inputs()
        out = build_intervention(featurizer, "collect")(base, None, [1, 2])
        assert out.shape[-1] == 2
        torch.testing.assert_close(out, base[..., [1, 2]])


class TestNoiseStream:
    """The noise generator advances across calls and restarts on request.

    Re-seeding every call would give identically-shaped consecutive batches the
    *same* corruption, making results depend on ``batch_size``; never restarting
    would make a sweep's grid cells incomparable.
    """

    def test_consecutive_calls_draw_independently(self) -> None:
        intervention = build_intervention(_rotation_featurizer(), "noise", seed=0)
        base, _ = _inputs()
        scale = torch.tensor(1.0)
        first, second = intervention(base, scale), intervention(base, scale)
        assert not torch.allclose(first, second)

    def test_same_seed_reproduces_the_sequence(self) -> None:
        base, _ = _inputs()
        scale = torch.tensor(1.0)
        a = build_intervention(_rotation_featurizer(), "noise", seed=11)
        b = build_intervention(_rotation_featurizer(), "noise", seed=11)
        torch.testing.assert_close(a(base, scale), b(base, scale))
        torch.testing.assert_close(a(base, scale), b(base, scale))

    def test_reset_restarts_the_stream(self) -> None:
        intervention = build_intervention(_rotation_featurizer(), "noise", seed=3)
        base, _ = _inputs()
        scale = torch.tensor(1.0)
        first = intervention(base, scale)
        intervention(base, scale)
        intervention.reset_noise_rng()
        torch.testing.assert_close(intervention(base, scale), first)

    def test_different_seeds_differ(self) -> None:
        base, _ = _inputs()
        scale = torch.tensor(1.0)
        a = build_intervention(_rotation_featurizer(), "noise", seed=0)
        b = build_intervention(_rotation_featurizer(), "noise", seed=1)
        assert not torch.allclose(a(base, scale), b(base, scale))


class TestMaskLearnability:
    """The mask's parameters must be reachable by an optimizer and differentiable."""

    def test_mask_is_a_registered_parameter(self) -> None:
        intervention = build_intervention(_identity_featurizer(), "mask")
        names = [name for name, _ in intervention.named_parameters()]
        assert "mask" in names
        assert intervention.mask.requires_grad

    def test_gradient_reaches_the_mask(self) -> None:
        intervention = build_intervention(_identity_featurizer(), "mask")
        intervention.set_temperature(0.5)
        intervention.train()
        base, source = _inputs()
        intervention(base, source).sum().backward()
        assert intervention.mask.grad is not None
        assert float(intervention.mask.grad.abs().sum()) > 0

    def test_eval_gate_is_binary(self) -> None:
        """Evaluation must threshold the gate, so the reported mask is the one
        the reported score was measured with."""
        intervention = build_intervention(_identity_featurizer(), "mask")
        with torch.no_grad():
            intervention.mask.copy_(torch.tensor([5.0, -5.0] * (D // 2)))
        intervention.set_temperature(0.5)
        intervention.eval()
        base, source = _inputs()
        out = intervention(base, source)
        on = list(range(0, D, 2))
        off = list(range(1, D, 2))
        torch.testing.assert_close(out[..., on], source[..., on])
        torch.testing.assert_close(out[..., off], base[..., off])

    def test_mask_without_n_features_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_features"):
            build_intervention(Featurizer(), "mask")


class TestUnknownMode:
    def test_unknown_mode_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="Unknown intervention mode"):
            build_intervention(_identity_featurizer(), "teleport")
