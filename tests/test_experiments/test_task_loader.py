"""Tests for the unified Task dataclass and load_task() loader."""

import dataclasses

import pytest

from causalab.causal.causal_model import CausalModel
from causalab.tasks.loader import Task, load_task


# ---------------------------------------------------------------------------
# Factory task (parametric)
# ---------------------------------------------------------------------------


class TestLoadNaturalDomainsArithmetic:
    @pytest.fixture
    def weekdays_cfg(self):
        from causalab.tasks.natural_domains_arithmetic import NaturalDomainConfig

        return NaturalDomainConfig(domain_type="weekdays")

    def test_load(self, weekdays_cfg):
        task = load_task("natural_domains_arithmetic", task_cfg=weekdays_cfg)
        assert isinstance(task, Task)
        assert task.name == "natural_domains_arithmetic"
        assert isinstance(task.causal_model, CausalModel)
        assert "entity" in task.causal_model.values
        assert len(task.causal_model.values["entity"]) == 7
        assert task.is_cyclic
        assert task.causal_model.embeddings

    def test_periodic_info(self, weekdays_cfg):
        task = load_task("natural_domains_arithmetic", task_cfg=weekdays_cfg)
        assert task.causal_model.periods


# ---------------------------------------------------------------------------
# Factory task (graph_walk)
# ---------------------------------------------------------------------------


class TestLoadGraphWalk:
    @pytest.fixture
    def ring6_cfg(self):
        from causalab.tasks.graph_walk.config import GraphWalkConfig

        return GraphWalkConfig(graph_type="ring", graph_size=6)

    def test_load(self, ring6_cfg):
        task = load_task("graph_walk", task_cfg=ring6_cfg)
        assert isinstance(task, Task)
        assert task.name == "graph_walk"
        assert isinstance(task.causal_model, CausalModel)
        assert len(task.intervention_values) == 6

    def test_requires_config(self):
        with pytest.raises(ValueError, match="task_cfg is required"):
            load_task("graph_walk")

    def test_random_unsupported(self, ring6_cfg):
        with pytest.raises(ValueError, match="random"):
            load_task("graph_walk", task_cfg=ring6_cfg, random=True)

    def test_intervention_variable(self, ring6_cfg):
        task = load_task("graph_walk", task_cfg=ring6_cfg)
        assert task.intervention_variable == "node_coordinates"

    def test_intervention_value_index(self, ring6_cfg):
        task = load_task("graph_walk", task_cfg=ring6_cfg)
        # Graph walk uses EXAMPLE_TO_CLASS which maps by walk_sequence[-1]
        # The intervention_value_index method uses intervention_variable lookup,
        # so this test just verifies the method exists and is callable
        assert callable(task.intervention_value_index)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestLoaderErrors:
    def test_nonexistent_task(self):
        with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
            load_task("nonexistent_task_xyz")

    def test_task_is_dataclass(self):
        assert dataclasses.is_dataclass(Task)


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------


class TestModuleLoaders:
    def test_load_counterfactuals_natural_domains_arithmetic(self):
        from causalab.tasks.loader import load_task_counterfactuals

        mod = load_task_counterfactuals("natural_domains_arithmetic")
        assert hasattr(mod, "generate_dataset")

    def test_load_counterfactuals_graph_walk(self):
        from causalab.tasks.loader import load_task_counterfactuals

        mod = load_task_counterfactuals("graph_walk")
        assert hasattr(mod, "generate_dataset")

    def test_load_token_positions(self):
        from causalab.tasks.loader import load_task_token_positions

        mod = load_task_token_positions("natural_domains_arithmetic")
        assert hasattr(mod, "create_token_positions")

    def test_load_counterfactuals_nonexistent(self):
        from causalab.tasks.loader import load_task_counterfactuals

        with pytest.raises(ImportError):
            load_task_counterfactuals("nonexistent_task_xyz")
