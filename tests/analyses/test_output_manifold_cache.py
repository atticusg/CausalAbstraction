"""Unit-tier tests for ``causalab.analyses.output_manifold.belief_cache``.

Regression coverage for GH #220: the belief-distribution cache used to validate
only the class dimension (``saved.shape[-1]``), so a debug pass (``n_train=16``)
and a full pass (``n_train=64``) sharing one ``experiment_root`` silently reused
the small debug cache and crashed downstream in ``load_natural_distributions``.

These exercise the real cache-management functions (identity, reuse decision,
archive) directly — no model required, so they live on the CPU ``unit`` tier
rather than the GPU golden tier that pins ``output_manifold``'s numerics.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

from causalab.analyses.output_manifold.belief_cache import (
    BELIEF_DISTS_STEM,
    archive_belief_artifacts,
    belief_cache_status,
    build_belief_identity,
)
from causalab.io.artifacts import save_tensor_results, save_tensors_with_meta


pytestmark = pytest.mark.unit


# W = 3 intervention classes → expected_dim = W + 1 = 4 ('other' bin).
_W = 3
_DIM = _W + 1


class _StubTask:
    """Minimal stand-in exposing only what ``build_belief_identity`` reads."""

    intervention_variable = "day"
    intervention_values = ["mon", "tue", "wed"]


def _cfg(
    *,
    n_train=64,
    seed=0,
    model_name="tiny-model",
    chat_template=False,
    target_variable="day",
):
    return OmegaConf.create(
        {
            "seed": seed,
            "task": {
                "name": "weekdays",
                "target_variable": target_variable,
                "n_train": n_train,
                "n_test": 8,
                "balanced": False,
                "enumerate_all": False,
                "resample_variable": "all",
                "max_new_tokens": 1,
            },
            "model": {
                "name": model_name,
                "dtype": "float32",
                "chat_template": chat_template,
                "chat_answer_directive": None,
            },
        }
    )


def _identity(**overrides):
    return build_belief_identity(_cfg(**overrides), _StubTask())


def _valid_dists(rows, cols=_DIM, seed=0):
    """Rows that sum to 1 (a valid distribution)."""
    torch.manual_seed(rows * 1000 + cols + seed)
    return torch.softmax(torch.randn(rows, cols), dim=-1)


def _write_cache(
    out_root, rows, *, cols=_DIM, identity=None, created_at=None, valid=True
):
    dists = _valid_dists(rows, cols) if valid else torch.ones(rows, cols)
    if identity is None:
        # Legacy artifact: only the .safetensors, no metadata sidecar.
        save_tensor_results(
            {"dists": dists}, out_root, f"{BELIEF_DISTS_STEM}.safetensors"
        )
    else:
        meta = {"identity": identity}
        if created_at is not None:
            meta["created_at"] = created_at
        save_tensors_with_meta({"dists": dists}, meta, out_root, BELIEF_DISTS_STEM)
    return dists


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")


# ── build_belief_identity ────────────────────────────────────────────────


class TestBuildBeliefIdentity:
    def test_deterministic(self):
        assert _identity() == _identity()

    def test_columns_recorded(self):
        ident = _identity()
        assert ident["intervention_variable"] == "day"
        assert ident["intervention_values"] == ["mon", "tue", "wed"]
        assert ident["model"]["name"] == "tiny-model"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"n_train": 16},
            {"seed": 1},
            {"model_name": "other-model"},
            {"chat_template": True},
            {"target_variable": "month"},
        ],
    )
    def test_relevant_change_changes_identity(self, overrides):
        assert _identity() != _identity(**overrides)


# ── belief_cache_status ──────────────────────────────────────────────────


class TestBeliefCacheStatus:
    def test_missing_artifact_not_reusable(self):
        with tempfile.TemporaryDirectory() as d:
            reuse, reason = belief_cache_status(d, _identity(), 64, _DIM)
            assert reuse is False
            assert "no cached" in reason

    def test_matching_identity_and_shape_reusable(self):
        ident = _identity(n_train=64)
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 64, identity=ident)
            reuse, reason = belief_cache_status(d, ident, 64, _DIM)
            assert reuse is True

    def test_row_count_mismatch_not_reusable(self):
        """The GH #220 failure: a 16-row debug cache must not satisfy an n=64 run."""
        ident = _identity()
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 16, identity=ident)
            reuse, reason = belief_cache_status(d, ident, 64, _DIM)
            assert reuse is False
            assert "row count" in reason

    def test_class_dim_mismatch_not_reusable(self):
        ident = _identity()
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 64, cols=_DIM + 1, identity=ident)
            reuse, reason = belief_cache_status(d, ident, 64, _DIM)
            assert reuse is False
            assert "last-dim" in reason

    def test_rows_not_summing_to_one_not_reusable(self):
        ident = _identity()
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 64, identity=ident, valid=False)
            reuse, reason = belief_cache_status(d, ident, 64, _DIM)
            assert reuse is False
            assert "sum" in reason

    def test_identity_mismatch_not_reusable(self):
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 64, identity=_identity(model_name="model-A"))
            reuse, reason = belief_cache_status(
                d, _identity(model_name="model-B"), 64, _DIM
            )
            assert reuse is False
            assert "identity mismatch" in reason

    def test_legacy_no_sidecar_structurally_valid_reusable(self):
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 64, identity=None)  # legacy: no .meta.json
            reuse, reason = belief_cache_status(d, _identity(), 64, _DIM)
            assert reuse is True
            assert "legacy" in reason

    def test_legacy_no_sidecar_row_mismatch_not_reusable(self):
        """Legacy dirs still get the #220 fix via the structural row check."""
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 16, identity=None)
            reuse, reason = belief_cache_status(d, _identity(), 64, _DIM)
            assert reuse is False
            assert "row count" in reason


# ── archive_belief_artifacts ─────────────────────────────────────────────


_PCA_VIZ_MEMBERS = (
    "hellinger_pca.safetensors",
    "hellinger_pca.meta.json",
    "hellinger_pca_3d.html",
    "hellinger_pca_2d.pdf",
)


class TestArchiveBeliefArtifacts:
    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            assert archive_belief_artifacts(d) is None

    def test_moves_full_canonical_set(self):
        with tempfile.TemporaryDirectory() as d:
            _write_cache(d, 16, identity=_identity(n_train=16))
            for m in _PCA_VIZ_MEMBERS:
                _touch(os.path.join(d, m))

            archive_dir = archive_belief_artifacts(d)

            assert archive_dir is not None and os.path.isdir(archive_dir)
            # Canonical paths are now empty…
            assert not os.path.exists(
                os.path.join(d, f"{BELIEF_DISTS_STEM}.safetensors")
            )
            for m in _PCA_VIZ_MEMBERS:
                assert not os.path.exists(os.path.join(d, m))
            # …and every member landed in the archive.
            assert os.path.exists(
                os.path.join(archive_dir, f"{BELIEF_DISTS_STEM}.safetensors")
            )
            assert os.path.exists(
                os.path.join(archive_dir, f"{BELIEF_DISTS_STEM}.meta.json")
            )
            for m in _PCA_VIZ_MEMBERS:
                assert os.path.exists(os.path.join(archive_dir, m))

    def test_archive_named_from_displaced_created_at(self):
        with tempfile.TemporaryDirectory() as d:
            _write_cache(
                d, 16, identity=_identity(), created_at="2026-06-11T14:02:09+00:00"
            )
            archive_dir = archive_belief_artifacts(d)
            assert archive_dir is not None
            # ':' and '+' are sanitized to '-' for a filesystem-safe folder name.
            assert os.path.basename(archive_dir).startswith("2026-06-11T14-02-09")


# ── round trip: debug pass → full pass (the acceptance scenario) ──────────


class TestDebugThenFullRoundTrip:
    def test_smaller_cache_recomputes_not_reused(self):
        ident16 = _identity(n_train=16)
        ident64 = _identity(n_train=64)
        with tempfile.TemporaryDirectory() as d:
            # Debug pass left a 16-row cache + its PCA artifacts.
            _write_cache(d, 16, identity=ident16, created_at="2026-06-11T14:02:09Z")
            for m in _PCA_VIZ_MEMBERS:
                _touch(os.path.join(d, m))

            # Full pass (n=64) must NOT reuse it.
            reuse, _ = belief_cache_status(d, ident64, 64, _DIM)
            assert reuse is False

            # It archives the displaced run instead of overwriting it.
            archive_dir = archive_belief_artifacts(d)
            assert archive_dir is not None
            assert os.path.exists(
                os.path.join(archive_dir, f"{BELIEF_DISTS_STEM}.safetensors")
            )
            assert not os.path.exists(
                os.path.join(d, f"{BELIEF_DISTS_STEM}.safetensors")
            )

            # A fresh 64-row collection is then reusable on the next run…
            _write_cache(d, 64, identity=ident64)
            reuse_after, _ = belief_cache_status(d, ident64, 64, _DIM)
            assert reuse_after is True

            # …and the archived 16-row artifact is preserved.
            archived = json.load(
                open(os.path.join(archive_dir, f"{BELIEF_DISTS_STEM}.meta.json"))
            )
            assert archived["identity"] == ident16


# ── _build_belief_artifacts: PCA-only refit must not re-run the model ──────


class TestBuildBeliefArtifactsRefitOnly:
    """Regression for the PR-review note: when the (GPU-collected) dists are valid
    but only the Hellinger PCA is missing, refit the cheap CPU PCA from the cached
    dists instead of archiving + re-running the whole model forward pass."""

    def test_pca_missing_refits_without_model_or_archive(self, monkeypatch):
        import causalab.analyses.output_manifold.main as om_main

        cfg = _cfg(n_train=8)
        task = _StubTask()
        n = 8

        # Keep the run model-free: stub task resolution + dataset generation, and
        # make any model load a hard failure so reaching the recollection branch
        # would fail the test loudly.
        monkeypatch.setattr(om_main, "resolve_task", lambda **kw: (task, None))
        monkeypatch.setattr(
            om_main,
            "prepare_datasets",
            lambda *a, **kw: ([{} for _ in range(n)], []),
        )

        def _no_model(*a, **kw):
            raise AssertionError("load_pipeline must not be called when dists cached")

        monkeypatch.setattr(om_main, "load_pipeline", _no_model)

        with tempfile.TemporaryDirectory() as d:
            # Valid dists cache with a matching identity, but NO PCA artifacts.
            _write_cache(d, n, identity=build_belief_identity(cfg, task))
            assert not os.path.exists(os.path.join(d, "hellinger_pca.safetensors"))

            om_main._build_belief_artifacts(cfg, d, batch_size=4)

            # PCA was refit from the cached dists — no model pass, no archive,
            # and the original dists artifact is untouched.
            assert os.path.exists(os.path.join(d, "hellinger_pca.safetensors"))
            assert os.path.exists(os.path.join(d, "hellinger_pca.meta.json"))
            assert os.path.exists(os.path.join(d, f"{BELIEF_DISTS_STEM}.safetensors"))
            assert not os.path.isdir(os.path.join(d, "archive"))
