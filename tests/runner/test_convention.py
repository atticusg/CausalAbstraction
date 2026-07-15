"""Tests for run_exp.py — discovery helpers and Hydra config composition."""

import json
import os

import pytest

from causalab.runner.helpers import (
    resolve_intervention_metric,
)
from causalab.runner.run_exp import _iter_analysis_steps  # pyright: ignore[reportPrivateUsage]
from causalab.io.pipelines import (
    find_activation_manifold_dirs,
    find_subspace_dirs,
    load_locate_result,
    load_activation_manifold_metadata,
    load_subspace_metadata,
)


def _exact_checker(neural_output: dict, causal_output: str) -> bool:
    """Stand-in task checker for resolver tests (the resolver wraps it)."""
    return neural_output["string"].strip() == causal_output.strip()


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_find_subspace_dirs_empty(self, tmp_path):
        assert find_subspace_dirs(str(tmp_path)) == []

    def test_find_subspace_dirs(self, tmp_path):
        subspace_root = tmp_path / "subspace"
        (subspace_root / "pca_k3").mkdir(parents=True)
        (subspace_root / "das_k5").mkdir(parents=True)
        # Files should be ignored
        (subspace_root / "readme.txt").touch()

        result = find_subspace_dirs(str(tmp_path))
        assert result == ["das_k5", "pca_k3"]

    def test_find_activation_manifold_dirs_empty(self, tmp_path):
        assert find_activation_manifold_dirs(str(tmp_path), "pca_k3") == []

    def test_find_activation_manifold_dirs(self, tmp_path):
        manifold_root = tmp_path / "activation_manifold" / "pca_k3"
        (manifold_root / "spline_s0.0").mkdir(parents=True)
        (manifold_root / "spline_s0.0" / "metadata.json").write_text("{}")
        (manifold_root / "spline_s0.1").mkdir(parents=True)
        (manifold_root / "spline_s0.1" / "metadata.json").write_text("{}")

        result = find_activation_manifold_dirs(str(tmp_path), "pca_k3")
        assert result == ["spline_s0.0", "spline_s0.1"]

    def test_load_locate_result_empty(self, tmp_path):
        assert load_locate_result(str(tmp_path)) == {}

    def test_load_locate_result(self, tmp_path):
        locate_root = tmp_path / "locate" / "interchange"
        locate_root.mkdir(parents=True)
        meta = {"best_layer": 17, "best_position": "last"}
        (locate_root / "metadata.json").write_text(json.dumps(meta))

        result = load_locate_result(str(tmp_path))
        assert result["best_layer"] == 17
        assert result["best_position"] == "last"

    def test_load_subspace_metadata(self, tmp_path):
        ss_dir = tmp_path / "subspace" / "pca_k3"
        ss_dir.mkdir(parents=True)
        meta = {"method": "pca", "k_features": 3, "layer": 5}
        (ss_dir / "metadata.json").write_text(json.dumps(meta))

        result = load_subspace_metadata(str(tmp_path), "pca_k3")
        assert result["method"] == "pca"
        assert result["k_features"] == 3

    def test_load_subspace_metadata_missing(self, tmp_path):
        assert load_subspace_metadata(str(tmp_path), "nonexistent") == {}

    def test_load_activation_manifold_metadata(self, tmp_path):
        m_dir = tmp_path / "activation_manifold" / "pca_k3" / "spline_s0.0"
        m_dir.mkdir(parents=True)
        meta = {"method": "spline", "smoothness": 0.0}
        (m_dir / "metadata.json").write_text(json.dumps(meta))

        result = load_activation_manifold_metadata(
            str(tmp_path), "pca_k3", "spline_s0.0"
        )
        assert result["method"] == "spline"

    def test_load_activation_manifold_metadata_missing(self, tmp_path):
        assert load_activation_manifold_metadata(str(tmp_path), "x", "y") == {}


# ---------------------------------------------------------------------------
# Integration: full directory tree
# ---------------------------------------------------------------------------


class TestFullTree:
    """Test that naming + discovery compose correctly."""

    def test_roundtrip(self, tmp_path):
        root = str(tmp_path)

        ss_sub = "pca_k3"
        m_sub = "spline_s0.0"

        ss_path = os.path.join(root, "subspace", ss_sub)
        os.makedirs(ss_path)
        meta = {"method": "pca", "k_features": 3, "layer": 5}
        with open(os.path.join(ss_path, "metadata.json"), "w") as f:
            json.dump(meta, f)

        m_path = os.path.join(root, "activation_manifold", ss_sub, m_sub)
        os.makedirs(m_path)
        with open(os.path.join(m_path, "metadata.json"), "w") as f:
            json.dump({}, f)

        assert find_subspace_dirs(root) == [ss_sub]
        assert find_activation_manifold_dirs(root, ss_sub) == [m_sub]

        ss_meta = load_subspace_metadata(root, ss_sub)
        assert ss_meta["layer"] == 5


# ---------------------------------------------------------------------------
# Hydra compose integration
# ---------------------------------------------------------------------------


class TestHydraCompose:
    """Each runner config pulls analyses via ``- analysis/<name>`` defaults.

    Each ``analysis/<name>.yaml`` declares ``# @package <name>``, so its body
    lands at ``cfg.<name>``. Order of execution is the order of those entries
    in the defaults list, recovered at runtime via OmegaConf insertion order.
    """

    @staticmethod
    def _compose(overrides=None):
        """Compose config with optional overrides."""
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../causalab/configs")
        )
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            return hydra.compose(config_name="config", overrides=overrides or [])

    # --- Root config defaults ---

    def test_default_config_no_analyses(self):
        cfg = self._compose()
        assert cfg.task.name == "natural_domains_arithmetic"
        assert cfg.model.name == "meta-llama/Llama-3.1-8B"
        assert list(_iter_analysis_steps(cfg)) == []

    def test_override_task_and_model(self):
        cfg = self._compose(
            overrides=["task.name=natural_domains_arithmetic", "model=llama31_8b"]
        )
        assert cfg.task.name == "natural_domains_arithmetic"
        assert cfg.model.name == "meta-llama/Llama-3.1-8B"
        assert cfg.model.dtype == "bfloat16"

    def test_experiment_root_override(self):
        cfg = self._compose(overrides=["experiment_root=/tmp/my_experiment"])
        assert cfg.experiment_root == "/tmp/my_experiment"

    # --- Step override and merge ---

    def test_step_override_params(self):
        cfg = self._compose(
            overrides=[
                "+analysis@subspace=subspace",
                "subspace.k_features=5",
            ],
        )
        assert cfg.subspace._name_ == "subspace"
        assert cfg.subspace.k_features == 5

    def test_dataset_params_in_task(self):
        cfg = self._compose(overrides=["+analysis@subspace=subspace"])
        # Dataset construction params live in task config, not analysis
        assert cfg.task.n_train == 1000
        assert cfg.task.n_test == 50
        assert "n_train" not in cfg.subspace
        assert "n_test" not in cfg.subspace

    def test_step_locate_defaults(self):
        cfg = self._compose(overrides=["+analysis@locate=locate"])
        assert cfg.locate.method == "interchange"
        assert cfg.locate.mode == "pairwise"

    def test_step_activation_manifold_defaults(self):
        cfg = self._compose(
            overrides=["+analysis@activation_manifold=activation_manifold"]
        )
        assert cfg.activation_manifold.method == "spline"
        assert cfg.activation_manifold.smoothness == 0.0

    def test_step_activation_manifold_override(self):
        cfg = self._compose(
            overrides=[
                "+analysis@activation_manifold=activation_manifold",
                "activation_manifold.smoothness=0.5",
            ],
        )
        assert cfg.activation_manifold.smoothness == 0.5

    def test_step_path_steering_defaults(self):
        cfg = self._compose(overrides=["+analysis@path_steering=path_steering"])
        assert cfg.path_steering.eval_criteria == [
            "isometry",
            "coherence",
            "distance_from_behavior_manifold",
        ]
        assert "geometric" in cfg.path_steering.path_modes
        assert "linear" in cfg.path_steering.path_modes

    # --- Task config tests ---

    def test_max_new_tokens_in_task(self):
        cfg = self._compose()
        assert cfg.task.max_new_tokens == 1
        assert "max_new_tokens" not in cfg.model

    def test_max_new_tokens_months(self):
        cfg = self._compose(overrides=["task.name=natural_domains_arithmetic"])
        assert cfg.task.max_new_tokens == 1

    def test_intervention_metric_default(self):
        cfg = self._compose()
        assert cfg.task.intervention_metric == "kl"
        assert cfg.task.reference_source == "empirical"

    def test_intervention_metric_graph_walk(self):
        cfg = self._compose(overrides=["task.name=graph_walk_grid_5x5"])
        assert cfg.task.intervention_metric == "kl"
        assert cfg.task.reference_source == "empirical"

    def test_intervention_metric_override_to_string_match(self):
        cfg = self._compose(overrides=["task.intervention_metric=string_match"])
        assert cfg.task.intervention_metric == "string_match"

    def test_intervention_metric_hellinger(self):
        cfg = self._compose(overrides=["task.intervention_metric=hellinger"])
        assert cfg.task.intervention_metric == "hellinger"

    def test_resolve_metric_kl(self):
        cfg = self._compose()
        _string_fn, cmp_fn = resolve_intervention_metric(
            cfg.task.intervention_metric, checker=_exact_checker
        )
        # Divergence metrics are wrapped to negate (higher-is-better convention).
        assert cmp_fn is not None
        assert callable(cmp_fn)

    def test_resolve_metric_string_match(self):
        cfg = self._compose(overrides=["task.intervention_metric=string_match"])
        string_fn, cmp_fn = resolve_intervention_metric(
            cfg.task.intervention_metric, checker=_exact_checker
        )
        # string_match returns (string_fn, _argmax_accuracy)
        assert string_fn is not None
        assert cmp_fn is not None

    def test_resolve_metric_hellinger(self):
        cfg = self._compose(overrides=["task.intervention_metric=hellinger"])
        _string_fn, cmp_fn = resolve_intervention_metric(
            cfg.task.intervention_metric, checker=_exact_checker
        )
        assert cmp_fn is not None
        assert callable(cmp_fn)

    def test_resolve_metric_invalid(self):
        cfg = self._compose(overrides=["task.intervention_metric=bogus"])
        with pytest.raises(ValueError, match="Unknown intervention_metric"):
            resolve_intervention_metric(
                cfg.task.intervention_metric, checker=_exact_checker
            )

    # --- Step name and output dir resolution ---

    def test_step_name_resolves(self):
        cfg = self._compose(overrides=["+analysis@locate=locate"])
        assert cfg.locate._name_ == "locate"

    def test_step_output_dir_resolves(self):
        cfg = self._compose(
            overrides=[
                "+analysis@subspace=subspace",
                "subspace.k_features=5",
            ],
        )
        assert "subspace/pca_k5" in cfg.subspace._output_dir

    # --- Relative interpolation tests ---

    def test_step_relative_interpolation_locate(self):
        """${.method} resolves correctly within an analysis step."""
        cfg = self._compose(overrides=["+analysis@locate=locate"])
        assert cfg.locate._subdir == "interchange"
        assert "locate/interchange" in cfg.locate._output_dir

    def test_step_relative_interpolation_subspace(self):
        cfg = self._compose(
            overrides=[
                "+analysis@subspace=subspace",
                "subspace.k_features=5",
            ],
        )
        assert cfg.subspace._subdir == "pca_k5"
        assert "subspace/pca_k5" in cfg.subspace._output_dir

    def test_step_relative_interpolation_activation_manifold(self):
        cfg = self._compose(
            overrides=["+analysis@activation_manifold=activation_manifold"]
        )
        assert cfg.activation_manifold._subdir == "spline_s0.0"
        assert "activation_manifold/spline_s0.0" in cfg.activation_manifold._output_dir

    def test_step_relative_interpolation_output_manifold(self):
        cfg = self._compose(overrides=["+analysis@output_manifold=output_manifold"])
        assert cfg.output_manifold._subdir == "spline_s0.0"
        assert "output_manifold/spline_s0.0" in cfg.output_manifold._output_dir

    def test_step_relative_interpolation_path_steering(self):
        cfg = self._compose(overrides=["+analysis@path_steering=path_steering"])
        # ${..visualization.figure_format} -> ${figure_format} (global default png; #163)
        assert cfg.path_steering.isometry_visualization.figure_format == "png"

    # --- Multi-step composition / order ---

    def test_multi_step_compose_and_order(self):
        """Defaults-list order is preserved by _iter_analysis_steps."""
        cfg = self._compose(
            overrides=[
                "+analysis@locate=locate",
                "+analysis@subspace=subspace",
            ],
        )
        assert "locate" in cfg
        assert "subspace" in cfg
        assert cfg.locate._name_ == "locate"
        assert cfg.subspace._name_ == "subspace"
        assert cfg.locate.method == "interchange"
        assert cfg.subspace.method == "pca"

        step_order = [name for name, _ in _iter_analysis_steps(cfg)]
        assert step_order == ["locate", "subspace"]

    def test_step_cli_override(self):
        """CLI overrides on top-level analysis keys work."""
        cfg = self._compose(
            overrides=[
                "+analysis@locate=locate",
                "locate.layers=[15,16,17]",
            ],
        )
        assert list(cfg.locate.layers) == [15, 16, 17]

    # --- Runner preset tests ---

    def test_pipeline_runner_preset(self):
        """he_pipeline runner preset composes a two-step sequence."""
        from causalab.io.configs import load_runner_config

        cfg = load_runner_config("runners/hierarchical_equality/he_pipeline")
        assert cfg.locate._name_ == "locate"
        assert cfg.subspace._name_ == "subspace"
        assert cfg.subspace.k_features == 8
        assert list(cfg.locate.layers) == [10, 20]
        step_order = [name for name, _ in _iter_analysis_steps(cfg)]
        assert step_order == ["locate", "subspace"]

    def test_pipeline_post_config(self):
        """Pipeline preset includes post-step configuration."""
        from causalab.io.configs import load_runner_config

        cfg = load_runner_config("runners/hierarchical_equality/he_pipeline")
        assert "post" in cfg
        assert len(cfg.post) == 1
        assert cfg.post[0].type == "variable_localization_heatmap"

    def test_he_visual_runner_preset(self):
        """he_visual defines a single analysis + post."""
        from causalab.io.configs import load_runner_config

        cfg = load_runner_config("runners/hierarchical_equality/he_visual")
        assert cfg.locate._name_ == "locate"
        assert "post" in cfg
        assert cfg.post[0].type == "variable_localization_heatmap"

    def test_single_step_runner(self):
        """Single-step runner presets yield exactly one analysis step."""
        from causalab.io.configs import load_runner_config

        cfg = load_runner_config("runners/hierarchical_equality/he_locate")
        steps = list(_iter_analysis_steps(cfg))
        assert len(steps) == 1
        assert steps[0][0] == "locate"
        assert steps[0][1]._name_ == "locate"


# ---------------------------------------------------------------------------
# Runner YAML conventions
# ---------------------------------------------------------------------------


class TestRunnerYamlConventions:
    """Guard against shapes that silently break analysis composition.

    The convention is: each analysis is composed via ``- analysis/<name>``
    in the defaults list and lives at ``cfg.<name>`` (because the file
    declares ``# @package <name>``). Overrides go in a top-level
    ``<name>:`` body block. The legacy ``analysis_sequence:`` key is no
    longer recognized.
    """

    @staticmethod
    def _runner_yaml_paths():
        import glob

        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../causalab/configs")
        )
        # Top-level *.yaml files only — exclude analysis/, model/, task/ subdirs
        return sorted(glob.glob(os.path.join(config_dir, "*.yaml")))

    def test_no_legacy_analysis_sequence(self):
        """No runner YAML may still use the legacy ``analysis_sequence`` key."""
        import yaml

        offenders = []
        for path in self._runner_yaml_paths():
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            if "analysis_sequence" in data:
                offenders.append(f"{path}: contains legacy 'analysis_sequence' key")

        assert not offenders, (
            "Runner YAMLs use removed analysis_sequence shape:\n  "
            + "\n  ".join(offenders)
        )

    def test_body_keys_match_defaults(self):
        """Each top-level body key matching an analysis name must have a corresponding
        ``- analysis/<name>`` entry in the defaults list (otherwise the body has nothing
        to override and would create an orphan key)."""
        import yaml

        analyses_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../causalab/configs/analysis")
        )
        analysis_names = {
            os.path.splitext(f)[0]
            for f in os.listdir(analyses_dir)
            if f.endswith(".yaml")
        }

        offenders = []
        for path in self._runner_yaml_paths():
            name = os.path.basename(path)
            if name in {"base.yaml", "config.yaml"}:
                continue
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue

            defaults = data.get("defaults") or []
            composed = set()
            for entry in defaults:
                if isinstance(entry, str) and entry.startswith("analysis/"):
                    composed.add(entry.split("/", 1)[1])

            for key in data:
                if key in analysis_names and key not in composed:
                    offenders.append(
                        f"{path}: body has '{key}:' but no '- analysis/{key}' in defaults"
                    )

        assert not offenders, (
            "Runner YAML bodies do not match defaults:\n  " + "\n  ".join(offenders)
        )
