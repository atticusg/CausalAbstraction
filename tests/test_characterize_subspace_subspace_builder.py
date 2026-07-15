"""Tests for the SAE-cluster -> subspace artifact producer.

Covers the pure primitive ``causalab.methods.sae.decoder_subspace``
(orientation resolution + orthonormality + span preservation) and the
analysis-layer ``build_subspace_artifact`` round-trip through the on-disk
``.safetensors`` that ``load_subspace`` consumes.

Tier split (one class per tier):

- ``TestDecoderSubspaceProperty`` (``property``) — orientation-independent
  geometric invariants of ``decoder_subspace``: orthonormality, span
  preservation, and transpose-invariant orientation.
- ``TestSubspaceBuilderUnit`` (``unit``) — orientation-resolution logic,
  input-validation error paths, and the IO round-trip through
  ``build_subspace_artifact`` / ``resolve_subspace_artifact`` /
  ``load_subspace``.
"""

from __future__ import annotations

import json

import pytest
import torch

from causalab.analyses.characterize_subspace.loading import load_subspace
from causalab.analyses.characterize_subspace.subspace_builder import (
    build_subspace_artifact,
    resolve_cluster_feature_ids,
    resolve_subspace_artifact,
)
from causalab.methods.sae import decoder_subspace


D_MODEL = 16
D_SAE = 40
FEATURE_IDS = [3, 7, 11, 29]
BLOCK_N_GROUPS = 6
BLOCK_K = 3


def _decoder_dmodel_by_dsae() -> torch.Tensor:
    """A ``(d_model, d_sae)`` decoder (the orientation our SAE ships)."""
    torch.manual_seed(0)
    return torch.randn(D_MODEL, D_SAE)


def _span_projector(basis: torch.Tensor) -> torch.Tensor:
    """Orthogonal projector onto the column span of ``basis`` (d_model, k)."""
    q, _ = torch.linalg.qr(basis)
    return q @ q.T


def _write_fake_sae(tmp_path) -> str:
    """A minimal checkpoint shaped like the real batchtopk SAE."""
    ckpt = {
        "model_config": {"d_model": D_MODEL, "d_sae": D_SAE},
        "state_dict": {"decoder.weight": _decoder_dmodel_by_dsae()},
    }
    path = tmp_path / "sae.pt"
    torch.save(ckpt, str(path))
    return str(path)


def _write_fake_block_sae(tmp_path, name: str = "grass.pt") -> str:
    """A minimal block/Grassmannian SAE checkpoint (orthonormal B_raw blocks)."""
    torch.manual_seed(1)
    b_raw, _ = torch.linalg.qr(torch.randn(BLOCK_N_GROUPS, D_MODEL, BLOCK_K))
    ckpt = {
        "tag": "grass_test",
        "scale": 3.744,
        "state_dict": {
            "B_raw": b_raw,
            "b_gate": torch.zeros(BLOCK_N_GROUPS),
            "dim_mask": torch.ones(BLOCK_N_GROUPS, BLOCK_K),
        },
        "model_kwargs": {
            "d": D_MODEL,
            "n_groups": BLOCK_N_GROUPS,
            "group_size": BLOCK_K,
        },
        "config": {"dataset": "model.layers.19", "d": D_MODEL},
    }
    path = tmp_path / name
    torch.save(ckpt, str(path))
    return str(path)


class TestDecoderSubspaceProperty:
    """Orientation-independent geometric invariants of ``decoder_subspace``."""

    pytestmark = pytest.mark.property

    def test_decoder_subspace_shape_and_orthonormality(self):
        dec = _decoder_dmodel_by_dsae()  # (d_model, d_sae)
        basis = decoder_subspace(dec, FEATURE_IDS)
        assert basis.shape == (D_MODEL, len(FEATURE_IDS))
        gram = basis.T @ basis
        assert torch.allclose(gram, torch.eye(len(FEATURE_IDS)), atol=1e-5)

    def test_decoder_subspace_preserves_span(self):
        dec = _decoder_dmodel_by_dsae()
        raw = decoder_subspace(dec, FEATURE_IDS, orthonormalize=False)
        basis = decoder_subspace(dec, FEATURE_IDS, orthonormalize=True)
        # Raw selection is exactly the decoder columns, in (d_model, k).
        assert torch.allclose(raw, dec[:, FEATURE_IDS].to(torch.float32))
        # Orthonormal basis spans the same subspace as the raw directions.
        assert torch.allclose(_span_projector(raw), _span_projector(basis), atol=1e-5)

    def test_decoder_subspace_orientation_is_transpose_invariant(self):
        """A (d_sae, d_model) decoder yields the same subspace as its transpose."""
        dec = _decoder_dmodel_by_dsae()  # (d_model, d_sae), d_sae > d_model
        basis_a = decoder_subspace(dec, FEATURE_IDS)
        basis_b = decoder_subspace(dec.t().contiguous(), FEATURE_IDS)
        assert basis_b.shape == (D_MODEL, len(FEATURE_IDS))
        assert torch.allclose(
            _span_projector(basis_a), _span_projector(basis_b), atol=1e-5
        )


class TestSubspaceBuilderUnit:
    """Orientation-resolution logic, error paths, and the artifact round-trip."""

    pytestmark = pytest.mark.unit

    # --- decoder_subspace: orientation resolution + input validation -------- #

    def test_decoder_subspace_dmodel_hint_overrides_heuristic(self):
        """When d_sae < d_model (under-complete), the larger-axis heuristic guesses
        the feature axis wrong; the d_model hint corrects it."""
        torch.manual_seed(1)
        dec = torch.randn(3, 12)  # (d_sae=3, d_model=12): smaller axis is features
        # Heuristic treats the larger (12) axis as features -> wrong d_model (3).
        assert decoder_subspace(dec, [0, 1, 2]).shape == (3, 3)
        # The hint pins d_model=12, so the size-3 axis is correctly the feature axis.
        assert decoder_subspace(dec, [0, 1, 2], d_model=12).shape == (12, 3)

    def test_decoder_subspace_square_is_ambiguous(self):
        """A square decoder cannot be oriented even with a d_model hint."""
        torch.manual_seed(2)
        dec = torch.randn(12, 12)
        with pytest.raises(ValueError, match="square"):
            decoder_subspace(dec, [1, 2, 3])
        with pytest.raises(ValueError, match="square"):
            decoder_subspace(dec, [1, 2, 3], d_model=12)

    def test_decoder_subspace_rejects_bad_inputs(self):
        dec = _decoder_dmodel_by_dsae()
        with pytest.raises(ValueError, match="non-empty"):
            decoder_subspace(dec, [])
        with pytest.raises(ValueError, match="out of range"):
            decoder_subspace(dec, [D_SAE + 1])
        with pytest.raises(ValueError, match="2-D"):
            decoder_subspace(torch.randn(D_SAE), FEATURE_IDS)

    # --- resolve_cluster_feature_ids --------------------------------------- #

    def test_resolve_cluster_feature_ids(self, tmp_path):
        clusters = {
            "950": [
                {"feature_id": 11, "label": "warmth"},
                {"feature_id": 3, "label": "cold"},
                {"feature_id": 11, "label": "dup"},  # de-duplicated
                {"feature": 7, "label": "alt key"},  # alternate key name
            ]
        }
        path = tmp_path / "clusters.json"
        path.write_text(json.dumps(clusters), encoding="utf-8")
        assert resolve_cluster_feature_ids(str(path), 950) == [3, 7, 11]
        assert resolve_cluster_feature_ids(str(path), "950") == [3, 7, 11]
        with pytest.raises(KeyError, match="999"):
            resolve_cluster_feature_ids(str(path), 999)

    # --- build_subspace_artifact: round-trip through load_subspace ---------- #

    def test_build_subspace_artifact_roundtrip(self, tmp_path):
        sae_path = _write_fake_sae(tmp_path)
        clusters = {"950": [{"feature_id": fid} for fid in FEATURE_IDS]}
        clusters_path = tmp_path / "clusters.json"
        clusters_path.write_text(json.dumps(clusters), encoding="utf-8")
        out = tmp_path / "subspace.safetensors"

        prov = build_subspace_artifact(
            sae_checkpoint=sae_path,
            clusters_path=str(clusters_path),
            cluster_id="950",
            out_path=str(out),
        )
        assert prov["feature_ids"] == sorted(FEATURE_IDS)
        assert prov["d_model"] == D_MODEL
        assert prov["k_features"] == len(FEATURE_IDS)
        assert out.is_file()

        # The artifact loads back through the consumer with the right geometry.
        projector = load_subspace(str(out))
        assert projector.d_model == D_MODEL
        assert projector.k == len(FEATURE_IDS)
        gram = projector.rotation.T @ projector.rotation
        assert torch.allclose(gram, torch.eye(len(FEATURE_IDS)), atol=1e-5)

    # --- resolve_subspace_artifact: artifact-vs-source branching ----------- #

    def test_resolve_subspace_artifact_passthrough(self, tmp_path):
        path, prov = resolve_subspace_artifact(
            artifact="/some/ready.safetensors", source=None, out_dir=str(tmp_path)
        )
        assert path == "/some/ready.safetensors"
        assert prov is None

    def test_resolve_subspace_artifact_builds_from_source(self, tmp_path):
        sae_path = _write_fake_sae(tmp_path)
        clusters = {"950": [{"feature_id": fid} for fid in FEATURE_IDS]}
        clusters_path = tmp_path / "clusters.json"
        clusters_path.write_text(json.dumps(clusters), encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        source = {
            "sae_checkpoint": sae_path,
            "clusters_path": str(clusters_path),
            "cluster_id": "950",
        }
        path, prov = resolve_subspace_artifact(
            artifact=None, source=source, out_dir=str(out_dir)
        )
        assert path == str(out_dir / "subspace.safetensors")
        assert prov is not None and prov["feature_ids"] == sorted(FEATURE_IDS)
        assert load_subspace(path).k == len(FEATURE_IDS)

    def test_resolve_subspace_artifact_rejects_ambiguous(self, tmp_path):
        with pytest.raises(ValueError, match="not both"):
            resolve_subspace_artifact(
                artifact="x.safetensors",
                source={"sae_checkpoint": "a", "clusters_path": "b", "cluster_id": "1"},
                out_dir=str(tmp_path),
            )
        with pytest.raises(ValueError, match="required"):
            resolve_subspace_artifact(artifact=None, source=None, out_dir=str(tmp_path))
        with pytest.raises(ValueError, match="missing required keys"):
            resolve_subspace_artifact(
                artifact=None, source={"sae_checkpoint": "a"}, out_dir=str(tmp_path)
            )
        with pytest.raises(ValueError, match="empty path values"):
            resolve_subspace_artifact(
                artifact=None,
                source={"sae_checkpoint": "", "clusters_path": "b", "cluster_id": "1"},
                out_dir=str(tmp_path),
            )

    def test_resolve_subspace_artifact_accepts_cluster_id_zero(self, tmp_path):
        """cluster_id=0 (int) is a legitimate key, not a "missing" field."""
        sae_path = _write_fake_sae(tmp_path)
        clusters = {"0": [{"feature_id": fid} for fid in FEATURE_IDS]}
        clusters_path = tmp_path / "clusters.json"
        clusters_path.write_text(json.dumps(clusters), encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        source = {
            "sae_checkpoint": sae_path,
            "clusters_path": str(clusters_path),
            "cluster_id": 0,
        }
        path, prov = resolve_subspace_artifact(
            artifact=None, source=source, out_dir=str(out_dir)
        )
        assert prov is not None and prov["cluster_id"] == "0"
        assert load_subspace(path).k == len(FEATURE_IDS)

    # --- resolve_subspace_artifact: block/Grassmannian-SAE source ----------- #

    def test_resolve_subspace_artifact_block_sae_source(self, tmp_path):
        """A {block_sae_checkpoint, block_id} source builds a rotation both
        consumers (characterize_subspace.loading + subspace/fixed.py) load."""
        from causalab.analyses.subspace.fixed import resolve_fixed_rotation

        block_path = _write_fake_block_sae(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        source = {"block_sae_checkpoint": block_path, "block_id": 1}

        # Consumer 1: characterize_subspace.loading.load_subspace via resolve.
        path, prov = resolve_subspace_artifact(
            artifact=None, source=source, out_dir=str(out_dir)
        )
        assert prov is not None
        assert prov["block_id"] == 1
        assert prov["k_features"] == BLOCK_K
        assert prov["block_sae_meta"]["layer"] == 19
        proj = load_subspace(path)
        assert proj.d_model == D_MODEL and proj.k == BLOCK_K

        # Consumer 2: subspace/fixed.py resolve_fixed_rotation reads rotation_matrix.
        rotation, _ = resolve_fixed_rotation(
            {"source": source}, out_dir=str(tmp_path / "fixed_out")
        )
        assert tuple(rotation.shape) == (D_MODEL, BLOCK_K)

    def test_resolve_subspace_artifact_block_sae_requires_block_id(self, tmp_path):
        """block_id=0 is legitimate, but a missing block_id is an error."""
        block_path = _write_fake_block_sae(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        with pytest.raises(ValueError, match="requires block_id"):
            resolve_subspace_artifact(
                artifact=None,
                source={"block_sae_checkpoint": block_path},
                out_dir=str(out_dir),
            )
        # block_id=0 builds fine.
        path, prov = resolve_subspace_artifact(
            artifact=None,
            source={"block_sae_checkpoint": block_path, "block_id": 0},
            out_dir=str(out_dir),
        )
        assert prov is not None and prov["block_id"] == 0
        assert load_subspace(path).k == BLOCK_K
