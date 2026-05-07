"""Tests for method: boundless support in subspace analysis.

Covers:
1. YAML config parsing — boundless block accepted without error.
2. Dispatch in _run_grid() / _run_single_cell() — boundless branch reached,
   tie_masks=False always passed.
3. run_boundless_grid() — correct config structure forwarded to train_interventions.
4. find_boundless_subspace() — tie_masks=False used in config.
5. __init__.py exports — public symbols present.
6. Existing dbm method unchanged.
7. Slow end-to-end smoke test.

Run fast tests:
    uv run pytest tests/test_experiments/test_boundless.py -v -k "not slow"

Run all:
    uv run pytest tests/test_experiments/test_boundless.py -v
"""

from __future__ import annotations

import os
import random
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# 1. YAML config parsing
# ---------------------------------------------------------------------------


class TestBoundlessYAMLConfig:
    def test_boundless_yaml_parsed_correctly(self):
        """Boundless block is accepted by OmegaConf without error."""
        yaml_str = """
method: boundless
k_features: 32
boundless:
  training_epoch: 20
  lr: 0.001
  regularization_coefficient: 0.1
"""
        cfg = OmegaConf.create(yaml_str)
        assert cfg.method == "boundless"
        assert cfg.boundless.training_epoch == 20
        assert cfg.boundless.lr == 0.001
        assert cfg.boundless.regularization_coefficient == 0.1

    def test_subspace_yaml_file_has_boundless_block(self):
        """The shipped subspace.yaml includes a boundless: block."""
        import yaml

        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "../../causalab/configs/analysis/subspace.yaml",
        )
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        assert "boundless" in data, "subspace.yaml must contain a 'boundless:' block"
        assert "training_epoch" in data["boundless"]
        assert "lr" in data["boundless"]
        assert "regularization_coefficient" in data["boundless"]
        # tie_masks must NOT be a field (it's enforced in code)
        assert "tie_masks" not in data["boundless"]

    def test_subspace_yaml_method_comment_mentions_boundless(self):
        """The method field comment in subspace.yaml mentions boundless."""
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "../../causalab/configs/analysis/subspace.yaml",
        )
        with open(yaml_path) as f:
            content = f.read()

        assert "boundless" in content


# ---------------------------------------------------------------------------
# 2. Public API exports
# ---------------------------------------------------------------------------


class TestSubspacePackageExports:
    def test_find_boundless_subspace_exported(self):
        from causalab.analyses.subspace import find_boundless_subspace

        assert callable(find_boundless_subspace)

    def test_run_boundless_grid_exported(self):
        from causalab.analyses.subspace import run_boundless_grid

        assert callable(run_boundless_grid)

    def test_all_includes_new_symbols(self):
        import causalab.analyses.subspace as pkg

        assert "find_boundless_subspace" in pkg.__all__
        assert "run_boundless_grid" in pkg.__all__


# ---------------------------------------------------------------------------
# 3. run_boundless_grid — tie_masks=False forwarded to train_interventions
# ---------------------------------------------------------------------------


class TestRunBoundlessGrid:
    # grid.py uses lazy imports (train_interventions imported inside the function),
    # so we patch at the actual definition module.

    def test_tie_masks_false_forwarded(self, tmp_path):
        """run_boundless_grid always calls train_interventions with tie_masks=False."""
        from causalab.analyses.subspace.grid import run_boundless_grid

        mock_result = {
            "results_by_key": {
                (0, "last"): {"train_score": 0.8, "test_score": 0.7},
            },
            "avg_train_score": 0.8,
            "avg_test_score": 0.7,
        }

        with patch(
            "causalab.methods.trained_subspace.train.train_interventions",
            return_value=mock_result,
        ) as mock_train:
            run_boundless_grid(
                targets={(0, "last"): MagicMock()},
                train_dataset=[],
                test_dataset=[],
                pipeline=MagicMock(),
                causal_model=MagicMock(),
                k_features=32,
                batch_size=8,
                metric=lambda x, y: True,
                log_dir=str(tmp_path / "logs"),
            )

        assert mock_train.called
        config_used = mock_train.call_args.kwargs["config"]
        assert config_used["intervention_type"] == "mask"
        assert config_used["featurizer_kwargs"]["tie_masks"] is False

    def test_boundless_config_hyperparams_forwarded(self, tmp_path):
        """Custom training_epoch, lr, and regularization_coefficient are passed."""
        from causalab.analyses.subspace.grid import run_boundless_grid

        mock_result = {
            "results_by_key": {
                (0, "last"): {"train_score": 0.5, "test_score": 0.5},
            },
            "avg_train_score": 0.5,
            "avg_test_score": 0.5,
        }

        with patch(
            "causalab.methods.trained_subspace.train.train_interventions",
            return_value=mock_result,
        ) as mock_train:
            run_boundless_grid(
                targets={(0, "last"): MagicMock()},
                train_dataset=[],
                test_dataset=[],
                pipeline=MagicMock(),
                causal_model=MagicMock(),
                k_features=16,
                batch_size=4,
                metric=lambda x, y: True,
                boundless_config={
                    "training_epoch": 5,
                    "init_lr": 0.01,
                    "masking": {"regularization_coefficient": 0.5},
                },
                log_dir=str(tmp_path / "logs"),
            )

        config_used = mock_train.call_args.kwargs["config"]
        assert config_used["training_epoch"] == 5
        assert config_used["init_lr"] == 0.01
        assert config_used["masking"]["regularization_coefficient"] == 0.5
        # tie_masks still forced False regardless
        assert config_used["featurizer_kwargs"]["tie_masks"] is False

    def test_returns_train_and_test_scores(self, tmp_path):
        """run_boundless_grid returns train_scores, test_scores, train_result."""
        from causalab.analyses.subspace.grid import run_boundless_grid

        mock_result = {
            "results_by_key": {
                (0, "last"): {"train_score": 0.9, "test_score": 0.8},
                (1, "last"): {"train_score": 0.7, "test_score": 0.6},
            },
            "avg_train_score": 0.8,
            "avg_test_score": 0.7,
        }

        with patch(
            "causalab.methods.trained_subspace.train.train_interventions",
            return_value=mock_result,
        ):
            result = run_boundless_grid(
                targets={(0, "last"): MagicMock(), (1, "last"): MagicMock()},
                train_dataset=[],
                test_dataset=[],
                pipeline=MagicMock(),
                causal_model=MagicMock(),
                k_features=32,
                batch_size=8,
                metric=lambda x, y: True,
                log_dir=str(tmp_path / "logs"),
            )

        assert "train_scores" in result
        assert "test_scores" in result
        assert "train_result" in result
        assert result["train_scores"][(0, "last")] == 0.9
        assert result["test_scores"][(1, "last")] == 0.6


# ---------------------------------------------------------------------------
# 4. find_boundless_subspace — tie_masks=False used
# ---------------------------------------------------------------------------


class TestFindBoundlessSubspace:
    # boundless.py uses lazy imports, so patch at the actual definition module.

    def test_tie_masks_false_in_config(self, tmp_path):
        """find_boundless_subspace passes tie_masks=False to train_interventions."""
        from causalab.analyses.subspace.boundless import find_boundless_subspace

        mock_result = {
            "results_by_key": {
                ("single",): {
                    "train_score": 0.75,
                    "test_score": 0.65,
                    "feature_indices": {"unit_0": [0, 1, 2]},
                    "train_eval": {},
                    "test_eval": {},
                    "trained_target": MagicMock(),
                }
            },
            "avg_train_score": 0.75,
            "avg_test_score": 0.65,
            "metadata": {},
        }

        with patch(
            "causalab.methods.trained_subspace.train.train_interventions",
            return_value=mock_result,
        ) as mock_train:
            find_boundless_subspace(
                target=MagicMock(),
                train_dataset=[],
                test_dataset=[],
                pipeline=MagicMock(),
                causal_model=MagicMock(),
                k_features=32,
                batch_size=8,
                output_dir=str(tmp_path),
                metric=lambda x, y: True,
            )

        config_used = mock_train.call_args.kwargs["config"]
        assert config_used["intervention_type"] == "mask"
        assert config_used["featurizer_kwargs"]["tie_masks"] is False

    def test_returns_boundless_result_dict(self, tmp_path):
        """find_boundless_subspace wraps result in boundless_result key."""
        from causalab.analyses.subspace.boundless import find_boundless_subspace

        mock_result = {
            "results_by_key": {
                ("single",): {
                    "train_score": 0.8,
                    "test_score": 0.75,
                    "feature_indices": {"unit_0": [0, 2], "unit_1": None},
                    "train_eval": {},
                    "test_eval": {},
                    "trained_target": MagicMock(),
                }
            },
            "avg_train_score": 0.8,
            "avg_test_score": 0.75,
            "metadata": {},
        }

        with patch(
            "causalab.methods.trained_subspace.train.train_interventions",
            return_value=mock_result,
        ):
            result = find_boundless_subspace(
                target=MagicMock(),
                train_dataset=[],
                test_dataset=[],
                pipeline=MagicMock(),
                causal_model=MagicMock(),
                k_features=32,
                batch_size=8,
                output_dir=str(tmp_path),
                metric=lambda x, y: True,
            )

        assert "boundless_result" in result
        br = result["boundless_result"]
        assert br["train_score"] == 0.8
        assert br["test_score"] == 0.75
        # n_features_by_unit computed from feature_indices
        assert br["n_features_by_unit"]["unit_0"] == 2
        assert br["n_features_by_unit"]["unit_1"] == 0

    def test_target_wrapped_as_single_key(self, tmp_path):
        """find_boundless_subspace passes target as {('single',): target}."""
        from causalab.analyses.subspace.boundless import find_boundless_subspace

        mock_target = MagicMock()
        mock_result = {
            "results_by_key": {
                ("single",): {
                    "train_score": 0.5,
                    "test_score": 0.5,
                    "feature_indices": {},
                    "train_eval": {},
                    "test_eval": {},
                    "trained_target": MagicMock(),
                }
            },
            "avg_train_score": 0.5,
            "avg_test_score": 0.5,
            "metadata": {},
        }

        with patch(
            "causalab.methods.trained_subspace.train.train_interventions",
            return_value=mock_result,
        ) as mock_train:
            find_boundless_subspace(
                target=mock_target,
                train_dataset=[],
                test_dataset=[],
                pipeline=MagicMock(),
                causal_model=MagicMock(),
                k_features=32,
                batch_size=8,
                output_dir=str(tmp_path),
                metric=lambda x, y: True,
            )

        targets_passed = mock_train.call_args.kwargs["interchange_targets"]
        assert ("single",) in targets_passed
        assert targets_passed[("single",)] is mock_target


# ---------------------------------------------------------------------------
# 5. main.py dispatch — boundless branch is reached and dbm is unchanged
# ---------------------------------------------------------------------------


class TestMainDispatch:
    """Verify that _run_grid and _run_single_cell correctly dispatch method=boundless."""

    def _make_cfg(self, method: str, layers: list[int]) -> object:
        """Build a minimal DictConfig-like object for dispatch tests."""
        cfg = OmegaConf.create(
            {
                "analysis": {
                    "method": method,
                    "k_features": 32,
                    "batch_size": 8,
                    "layers": layers,
                    "token_positions": ["last"],
                    "das": {"training_epoch": 1, "lr": 0.001},
                    "dbm": {
                        "training_epoch": 1,
                        "lr": 0.001,
                        "regularization_coefficient": 0.1,
                    },
                    "boundless": {
                        "training_epoch": 1,
                        "lr": 0.001,
                        "regularization_coefficient": 0.1,
                    },
                    "visualization": {
                        "figure_format": "png",
                        "colormap": "viridis",
                        "vis_dims": None,
                        "detailed_hover": False,
                        "max_hover_chars": 50,
                    },
                    "_output_dir": "/tmp/test_output",
                    "_name_": "subspace",
                    "_subdir": "boundless_k32",
                },
                "model": {"name": "tiny", "device": "cpu"},
                "task": {
                    "name": "test_task",
                    "n_train": 10,
                    "n_test": 5,
                    "max_new_tokens": 1,
                    "enumerate_all": False,
                    "intervention_metric": "exact_match",
                    "colormap": "viridis",
                },
                "seed": 42,
            }
        )
        return cfg

    def test_run_grid_boundless_calls_run_boundless_grid(self, tmp_path):
        """_run_grid dispatches to run_boundless_grid for method=boundless."""
        from causalab.analyses.subspace.main import _run_grid

        cfg = self._make_cfg("boundless", [0, 1])

        mock_grid_result = {
            "results_by_key": {},
            "avg_train_score": 0.5,
            "avg_test_score": 0.5,
            "train_scores": {(0, "last"): 0.5, (1, "last"): 0.6},
            "test_scores": {(0, "last"): 0.4, (1, "last"): 0.5},
            "train_result": {"results_by_key": {}, "metadata": {}},
        }

        mock_task = MagicMock()
        mock_task.intervention_values = None
        mock_task.intervention_variable = "answer"

        mock_targets = {(0, "last"): MagicMock(), (1, "last"): MagicMock()}
        mock_token_positions = [MagicMock(id="last"), MagicMock(id="last")]

        # _run_grid imports run_boundless_grid from causalab.analyses.subspace.grid
        with (
            patch(
                "causalab.analyses.subspace.main.build_targets_for_grid",
                return_value=(mock_targets, mock_token_positions),
            ),
            patch(
                "causalab.analyses.subspace.grid.run_boundless_grid",
                return_value=mock_grid_result,
            ) as mock_boundless,
            patch(
                "causalab.methods.trained_subspace.train.save_train_results",
            ),
            patch(
                "causalab.analyses.subspace.main._save_grid_results",
                return_value={
                    "best_cell": None,
                    "best_layer": None,
                    "scores_per_cell": {},
                },
            ),
            patch(
                "causalab.io.counterfactuals.save_counterfactual_examples",
            ),
            patch("os.makedirs"),
        ):
            _run_grid(
                cfg=cfg,
                pipeline=MagicMock(),
                task=mock_task,
                train_dataset=[],
                test_dataset=[],
                out_dir=str(tmp_path),
                method="boundless",
                k_features=32,
                batch_size=8,
                intervention_metric="exact_match",
                figure_format="png",
                layers=[0, 1],
                token_positions=["last"],
                das_training_epoch=1,
                das_lr=1e-3,
                dbm_training_epoch=1,
                dbm_lr=1e-3,
                dbm_regularization_coefficient=0.1,
                dbm_tie_masks=False,
                boundless_training_epoch=1,
                boundless_lr=1e-3,
                boundless_regularization_coefficient=0.1,
                colormap=None,
            )

        assert mock_boundless.called, "run_boundless_grid should have been called"

    def test_run_grid_raises_for_unknown_method(self, tmp_path):
        """_run_grid raises ValueError for unknown method names."""
        from causalab.analyses.subspace.main import _run_grid

        cfg = self._make_cfg("unknown_method", [0, 1])
        mock_targets = {(0, "last"): MagicMock()}
        mock_token_positions = [MagicMock(id="last")]

        with (
            patch(
                "causalab.analyses.subspace.main.build_targets_for_grid",
                return_value=(mock_targets, mock_token_positions),
            ),
            patch("os.makedirs"),
        ):
            with pytest.raises(ValueError, match="Unknown subspace method"):
                _run_grid(
                    cfg=cfg,
                    pipeline=MagicMock(),
                    task=MagicMock(),
                    train_dataset=[],
                    test_dataset=[],
                    out_dir=str(tmp_path),
                    method="unknown_method",
                    k_features=32,
                    batch_size=8,
                    intervention_metric="exact_match",
                    figure_format="png",
                    layers=[0, 1],
                    token_positions=["last"],
                    das_training_epoch=1,
                    das_lr=1e-3,
                    dbm_training_epoch=1,
                    dbm_lr=1e-3,
                    dbm_regularization_coefficient=0.1,
                    dbm_tie_masks=False,
                    boundless_training_epoch=1,
                    boundless_lr=1e-3,
                    boundless_regularization_coefficient=0.1,
                    colormap=None,
                )

    def test_run_single_cell_boundless_calls_find_boundless_subspace(self, tmp_path):
        """_run_single_cell dispatches to find_boundless_subspace for method=boundless."""
        from causalab.analyses.subspace.main import _run_single_cell

        cfg = self._make_cfg("boundless", [3])

        mock_task = MagicMock()
        mock_task.intervention_values = None
        mock_task.intervention_variable = "answer"

        mock_target = MagicMock()
        mock_targets = {(3, "last"): mock_target}
        mock_token_positions = [MagicMock(id="last")]

        mock_sub = {
            "boundless_result": {
                "train_score": 0.8,
                "test_score": 0.7,
                "n_features_by_unit": {"unit_0": 5},
            }
        }

        # _run_single_cell does: from causalab.analyses.subspace import find_boundless_subspace
        # We must patch the symbol in the package namespace.
        with (
            patch(
                "causalab.analyses.subspace.main.build_targets_for_grid",
                return_value=(mock_targets, mock_token_positions),
            ),
            patch(
                "causalab.analyses.subspace.find_boundless_subspace",
                return_value=mock_sub,
            ) as mock_find,
        ):
            result = _run_single_cell(
                cfg=cfg,
                pipeline=MagicMock(),
                task=mock_task,
                train_dataset=[],
                test_dataset=[],
                layer=3,
                out_dir=str(tmp_path),
                method="boundless",
                k_features=32,
                batch_size=8,
                intervention_metric="exact_match",
                figure_format="png",
                token_positions=["last"],
                colormap=None,
                vis_dims=None,
                detailed_hover=False,
                max_hover_chars=200,
                das_training_epoch=1,
                das_lr=1e-3,
            )

        assert mock_find.called
        assert "boundless_result" in result
        assert result["boundless_result"]["train_score"] == 0.8

    def test_dbm_dispatch_unchanged_in_single_cell(self, tmp_path):
        """method=dbm still dispatches to find_dbm_subspace (not boundless)."""
        from causalab.analyses.subspace.main import _run_single_cell

        cfg = self._make_cfg("dbm", [3])
        mock_task = MagicMock()
        mock_task.intervention_values = None
        mock_task.intervention_variable = "answer"

        mock_targets = {(3, "last"): MagicMock()}
        mock_token_positions = [MagicMock(id="last")]

        mock_sub = {
            "dbm_result": {
                "train_score": 0.7,
                "test_score": 0.6,
                "n_features_by_unit": {},
            }
        }

        # _run_single_cell does: from causalab.analyses.subspace import find_dbm_subspace
        # We must patch the symbols in the package namespace.
        with (
            patch(
                "causalab.analyses.subspace.main.build_targets_for_grid",
                return_value=(mock_targets, mock_token_positions),
            ),
            patch(
                "causalab.analyses.subspace.find_dbm_subspace",
                return_value=mock_sub,
            ) as mock_dbm,
            patch(
                "causalab.analyses.subspace.find_boundless_subspace",
            ) as mock_boundless,
        ):
            result = _run_single_cell(
                cfg=cfg,
                pipeline=MagicMock(),
                task=mock_task,
                train_dataset=[],
                test_dataset=[],
                layer=3,
                out_dir=str(tmp_path),
                method="dbm",
                k_features=32,
                batch_size=8,
                intervention_metric="exact_match",
                figure_format="png",
                token_positions=["last"],
                colormap=None,
                vis_dims=None,
                detailed_hover=False,
                max_hover_chars=200,
                das_training_epoch=1,
                das_lr=1e-3,
            )

        assert mock_dbm.called, "find_dbm_subspace should have been called"
        assert not mock_boundless.called, (
            "find_boundless_subspace should NOT have been called"
        )
        assert "dbm_result" in result


# ---------------------------------------------------------------------------
# 6. Config merge: boundless config merges correctly with defaults
# ---------------------------------------------------------------------------


class TestBoundlessConfigMerge:
    def test_merge_with_defaults_accepts_tie_masks_false(self):
        """merge_with_defaults accepts tie_masks=False without error."""
        from causalab.configs.train_config import merge_with_defaults

        config = merge_with_defaults(
            {
                "intervention_type": "mask",
                "featurizer_kwargs": {"tie_masks": False},
                "DAS": {"n_features": 32},
                "train_batch_size": 8,
                "evaluation_batch_size": 8,
                "training_epoch": 20,
                "init_lr": 0.001,
                "masking": {"regularization_coefficient": 0.1},
            }
        )

        assert config["featurizer_kwargs"]["tie_masks"] is False
        assert config["intervention_type"] == "mask"
        assert config["training_epoch"] == 20


# ---------------------------------------------------------------------------
# 7. Slow end-to-end smoke test (requires model download)
# ---------------------------------------------------------------------------

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


def _make_dataset(causal_model, n=20, seed=42):
    rng = random.Random(seed)
    node_ids = causal_model.values["node_coordinates"]
    examples = []
    for i in range(n):
        node_id = node_ids[i % len(node_ids)]
        input_trace = causal_model.new_trace({"node_coordinates": node_id})
        cf_node = rng.choice([nd for nd in node_ids if nd != node_id])
        cf_trace = causal_model.new_trace({"node_coordinates": cf_node})
        examples.append({"input": input_trace, "counterfactual_inputs": [cf_trace]})
    return examples


def _setup():
    from causalab.neural.pipeline import LMPipeline
    from causalab.tasks.graph_walk.config import GraphWalkConfig
    from causalab.tasks.graph_walk.causal_models import create_causal_model

    gw_config = GraphWalkConfig(
        graph_type="ring",
        graph_size=6,
        context_length=30,
        separator=",",
        seed=42,
    )
    causal_model = create_causal_model(gw_config)
    dataset = _make_dataset(causal_model, n=20)
    pipeline = LMPipeline(TINY_MODEL, max_new_tokens=1)
    target = _build_last_token_target(pipeline)
    return causal_model, dataset, pipeline, target


@pytest.mark.slow
class TestFindBoundlessSubspaceSmoke:
    def test_find_boundless_subspace_runs_and_returns_scores(self):
        """Smoke test: find_boundless_subspace runs end-to-end without error."""
        from causalab.analyses.subspace import find_boundless_subspace

        causal_model, dataset, pipeline, target = _setup()

        def metric(neural_output, causal_output):
            neural_str = neural_output["string"].strip().lower()
            if isinstance(causal_output, list):
                return any(c.strip().lower() in neural_str for c in causal_output)
            return neural_str == str(causal_output).strip().lower()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_boundless_subspace(
                target,
                dataset,
                dataset,
                pipeline,
                causal_model,
                k_features=4,
                batch_size=8,
                output_dir=tmpdir,
                metric=metric,
            )

        assert "boundless_result" in result
        br = result["boundless_result"]
        assert "train_score" in br
        assert "test_score" in br
        assert "n_features_by_unit" in br
        assert isinstance(br["train_score"], float)
        assert isinstance(br["test_score"], float)
        assert 0.0 <= br["train_score"] <= 1.0
        assert 0.0 <= br["test_score"] <= 1.0

    def test_boundless_produces_different_result_from_interchange(self):
        """Boundless (mask, tie_masks=False) config differs from DAS (interchange) config."""
        from causalab.configs.train_config import merge_with_defaults

        boundless_config = merge_with_defaults(
            {
                "intervention_type": "mask",
                "featurizer_kwargs": {"tie_masks": False},
            }
        )
        das_config = merge_with_defaults(
            {
                "intervention_type": "interchange",
            }
        )

        assert boundless_config["intervention_type"] == "mask"
        assert das_config["intervention_type"] == "interchange"
        assert boundless_config["featurizer_kwargs"]["tie_masks"] is False
