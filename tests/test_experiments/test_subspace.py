"""Tests for jobs/subspace/ package.

Verifies that find_pca_subspace and find_das_subspace produce correct
artifacts and match the behavior previously in fitting_pipeline steps 1-2.

Run with:
    uv run pytest tests/test_experiments/test_subspace.py -v
"""

from __future__ import annotations

import os
import random
import tempfile

import pytest
import torch

from causalab.tasks.graph_walk.config import GraphWalkConfig
from causalab.tasks.graph_walk.causal_models import create_causal_model


TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


def _build_last_token_target(pipeline):
    from causalab.neural.activations.targets import build_residual_stream_targets
    from causalab.neural.token_positions import TokenPosition, get_last_token_index

    last_pos = TokenPosition(
        indexer=lambda inp: get_last_token_index(inp, pipeline),
        pipeline=pipeline,
        id="last",
    )
    targets = build_residual_stream_targets(
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
    target = _build_last_token_target(pipeline)
    return causal_model, dataset, pipeline, target


@pytest.mark.slow
class TestFindPcaSubspace:
    def test_returns_features_and_rotation(self):
        from causalab.analyses.subspace import find_pca_subspace

        causal_model, dataset, pipeline, target = _setup()
        k = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_pca_subspace(
                target,
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

    def test_sets_featurizer_on_target(self):
        from causalab.analyses.subspace import find_pca_subspace

        causal_model, dataset, pipeline, target = _setup()
        unit = target.flatten()[0]
        assert unit.featurizer is None or unit.featurizer.id != "PCA"

        find_pca_subspace(target, dataset, pipeline, 4, batch_size=8)

        assert unit.featurizer is not None
        assert unit.featurizer.id == "PCA"

    def test_matches_old_pipeline_pca(self):
        """PCA features match what the old fitting_pipeline produced."""
        from causalab.analyses.subspace import find_pca_subspace
        from causalab.neural.activations.collect import collect_features
        from causalab.methods.pca import compute_svd

        causal_model, dataset, pipeline, target = _setup()
        unit = target.flatten()[0]
        k = 4

        # Manual PCA (old approach)
        raw_dict = collect_features(
            dataset=dataset,
            pipeline=pipeline,
            model_units=[unit],
            batch_size=8,
        )
        svd = compute_svd(raw_dict, n_components=k, preprocess="center")
        manual_rotation = svd[unit.id]["rotation"]
        manual_features = (
            raw_dict[unit.id].detach().float() @ manual_rotation.float()
        ).detach()

        # Fresh target for subspace function
        target2 = _build_last_token_target(pipeline)
        result = find_pca_subspace(target2, dataset, pipeline, k, batch_size=8)

        assert torch.allclose(result["rotation"], manual_rotation, atol=1e-5)
        assert torch.allclose(result["features"], manual_features, atol=1e-5)


@pytest.mark.slow
class TestFindDasSubspace:
    def test_returns_features_and_das_result(self):
        from causalab.analyses.subspace import find_das_subspace

        causal_model, dataset, pipeline, target = _setup()

        def metric(neural_output, causal_output):
            neural_str = neural_output["string"].strip().lower()
            if isinstance(causal_output, list):
                return any(c.strip().lower() in neural_str for c in causal_output)
            return neural_str == str(causal_output).strip().lower()

        k = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_das_subspace(
                target,
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
