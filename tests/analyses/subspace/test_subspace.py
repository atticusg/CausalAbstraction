"""Property-tier tests for ``causalab.analyses.subspace``.

Covers the two subspace-discovery analysis wrappers that aren't exercised
by the smoke tier's function-level contract:

* ``find_pca_subspace`` — the smoke chain (``configs/smoke/weekdays_subspace.yaml``)
  runs the PCA path end-to-end through the *runner* and asserts only that
  ``subspace/pca_k8/result/metadata.json`` exists. These tests pin the
  *function's* contract directly: output shapes, the functional featurizer
  attach (the returned ``spec`` carries it; the input spec is unchanged),
  the rotation/feature artifacts it writes, and an equivalence check against
  a manual SVD reference (catches a transposed rotation or a dropped
  centering step, which the existence-only smoke run cannot).
* ``find_das_subspace`` — has **no** smoke coverage at all (the smoke
  subspace chain is PCA-only), so this is the sole run-coverage of the DAS
  analysis wrapper.

These are ``property`` tier (CPU, tiny ``tiny-random-gpt2``): they assert
shape/dtype contracts at the module boundary plus a computation
equivalence, not pinned numerical values. Relocated here from
``tests/tasks/graph_walk/test_subspace.py`` (the functions live in
``causalab/analyses/subspace/``, not under a task) when the ``slow`` marker
was retired.
"""

from __future__ import annotations

import os
import random
import tempfile

import pytest
import torch

from causalab.tasks.graph_walk.config import GraphWalkConfig
from causalab.tasks.graph_walk.causal_models import create_causal_model


pytestmark = pytest.mark.property


TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


def _build_last_token_sites(pipeline):
    from causalab.neural.activations.site_grids import build_residual_stream_sites
    from causalab.neural.token_positions import TokenPosition, get_last_token_index

    last_pos = TokenPosition(
        indexer=lambda inp: get_last_token_index(inp, pipeline),
        pipeline=pipeline,
        id="last",
    )
    targets = build_residual_stream_sites(
        pipeline=pipeline,
        layers=[2],
        token_positions=[last_pos],
        mode="one_target_per_unit",
    )
    return list(targets.values())[0]


def _make_dataset(causal_model, n=30, seed=42):
    rng = random.Random(seed)
    examples = []
    node_ids = causal_model.values["node_coordinates"]
    for i in range(n):
        node_id = node_ids[i % len(node_ids)]
        input_trace = causal_model.new_trace({"node_coordinates": node_id})
        cf_node = rng.choice([n for n in node_ids if n != node_id])
        cf_trace = causal_model.new_trace({"node_coordinates": cf_node})
        examples.append({"input": input_trace, "counterfactual_inputs": [cf_trace]})
    return examples


def _setup():
    from causalab.neural.pipeline import LMPipeline

    gw_config = GraphWalkConfig(
        graph_type="ring",
        graph_size=6,
        context_length=30,
        separator=",",
        seed=42,
    )
    causal_model = create_causal_model(gw_config)
    dataset = _make_dataset(causal_model, n=30)
    pipeline = LMPipeline(TINY_MODEL, max_new_tokens=1)
    sites = _build_last_token_sites(pipeline)
    return causal_model, dataset, pipeline, sites


class TestFindPcaSubspace:
    def test_returns_features_and_rotation(self):
        from causalab.analyses.subspace import find_pca_subspace

        causal_model, dataset, pipeline, sites = _setup()  # pyright: ignore[reportUnusedVariable]
        k = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_pca_subspace(
                sites,
                dataset,
                pipeline,
                k,
                batch_size=8,
                output_dir=tmpdir,
            )

            assert "rotation" in result
            assert "explained_variance_ratio" in result
            assert "features" in result

            assert result["rotation"].shape[1] == k
            assert result["features"].shape[1] == k
            assert len(result["explained_variance_ratio"]) == k
            assert result["features"].shape[0] == len(dataset)

            # Artifacts saved
            assert os.path.exists(os.path.join(tmpdir, "rotation.safetensors"))
            assert os.path.exists(
                os.path.join(tmpdir, "features", "training_features.safetensors")
            )

    def test_returns_spec_with_featurizer_functionally(self):
        from causalab.analyses.subspace import find_pca_subspace

        causal_model, dataset, pipeline, sites = _setup()  # pyright: ignore[reportUnusedVariable]
        input_spec = sites[0][0]
        assert input_spec.fsite.featurizer.id != "PCA"

        result = find_pca_subspace(sites, dataset, pipeline, 4, batch_size=8)

        # Functional attach: the RETURNED spec carries the PCA featurizer;
        # the caller's input spec is a frozen value and stays unchanged.
        assert result["spec"].fsite.featurizer.id == "PCA"
        assert input_spec.fsite.featurizer.id != "PCA"
        assert sites[0][0] is input_spec

    def test_matches_old_pipeline_pca(self):
        """PCA features match what the old fitting_pipeline produced."""
        from causalab.analyses.subspace import find_pca_subspace
        from causalab.neural.activations.collect import collect_features
        from causalab.methods.pca import compute_svd

        causal_model, dataset, pipeline, sites = _setup()  # pyright: ignore[reportUnusedVariable]
        spec = sites[0][0]
        k = 4

        # Manual PCA (old approach)
        raw_dict = collect_features(
            dataset=dataset,
            pipeline=pipeline,
            sites=[spec],
            batch_size=8,
        )
        # collect_features returns dict when collect_output_logits=False (default)
        svd = compute_svd(raw_dict, n_components=k, preprocess="center")  # pyright: ignore[reportArgumentType]
        manual_rotation = svd[spec.key]["rotation"]
        manual_features = (
            raw_dict[spec.key].detach().float() @ manual_rotation.float()  # pyright: ignore[reportCallIssue, reportArgumentType]
        ).detach()

        # Fresh sites for the subspace function
        sites2 = _build_last_token_sites(pipeline)
        result = find_pca_subspace(sites2, dataset, pipeline, k, batch_size=8)

        assert torch.allclose(result["rotation"], manual_rotation, atol=1e-5)
        assert torch.allclose(result["features"], manual_features, atol=1e-5)


class TestFindDasSubspace:
    def test_returns_features_and_das_result(self):
        from causalab.analyses.subspace import find_das_subspace

        causal_model, dataset, pipeline, sites = _setup()

        def metric(neural_output, causal_output):
            neural_str = neural_output["string"].strip().lower()
            if isinstance(causal_output, list):
                return any(c.strip().lower() in neural_str for c in causal_output)
            return neural_str == str(causal_output).strip().lower()

        k = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_das_subspace(
                sites,
                dataset,
                dataset,
                pipeline,
                causal_model,
                k,
                batch_size=8,
                output_dir=tmpdir,
                metric=metric,
                loss_config={"training_epoch": 1, "init_lr": 1e-3},
            )

            assert "das_result" in result
            assert "features" in result
            assert "avg_test_score" in result["das_result"]
            assert result["features"].shape[0] == len(dataset)
            assert result["features"].shape[1] == k

            # Artifacts saved
            assert os.path.isdir(os.path.join(tmpdir, "das"))
            assert os.path.exists(
                os.path.join(tmpdir, "features", "training_features.safetensors")
            )
