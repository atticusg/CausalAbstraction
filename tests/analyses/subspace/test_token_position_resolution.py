"""Regression tests for token-position resolution (issue #257).

``subspace`` / ``activation_manifold`` used to ship a global default
``token_positions: [last_token]`` — a position name **most tasks do not
define** (many, e.g. ``graph_walk``, expose ``last``). An explicit single-cell
run (``layers: [L]`` with ``token_positions`` left at the default) therefore
died with ``Unknown token positions ['last_token']``.

The fix defaults ``token_positions`` to ``null`` and resolves positions from the
task instead of a hard-coded name:

* the shared grid resolver (``runner/helpers.py::build_targets_for_grid``) treats
  ``None`` as "all positions the task declares", and raises a clear, *name-listing*
  error for an unknown name;
* single-cell mode (``analyses/subspace/main.py::_run_single_cell``) uses the
  task's sole declared position when exactly one exists, and otherwise raises a
  name-listing error rather than assuming ``last_token``.

These are ``property`` tier (CPU, tiny ``tiny-random-gpt2`` + the ``graph_walk``
task, which declares the single position ``last``): they pin the *resolution
behaviour*, which is the real contract — not a brittle "config contains null"
check.
"""

from __future__ import annotations

import random
import tempfile

import pytest

from causalab.tasks.graph_walk.config import GraphWalkConfig
from causalab.tasks.loader import load_task


pytestmark = pytest.mark.property


TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


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


@pytest.fixture(scope="module")
def gw_setup():
    """A real ``graph_walk`` Task + tiny pipeline. graph_walk declares only `last`."""
    from causalab.neural.pipeline import LMPipeline

    cfg = GraphWalkConfig(
        graph_type="ring",
        graph_size=6,
        context_length=30,
        separator=",",
        seed=42,
    )
    task = load_task("graph_walk", task_cfg=cfg)
    pipeline = LMPipeline(TINY_MODEL, max_new_tokens=1)
    dataset = _make_dataset(task.causal_model, n=30)
    return task, pipeline, dataset


def test_graph_walk_declares_last_not_last_token(gw_setup):
    """Premise of #257: graph_walk exposes `last`, not the old default `last_token`."""
    task, pipeline, _ = gw_setup
    names = set(task.create_token_positions(pipeline))
    assert names == {"last"}


def test_grid_resolver_null_uses_all_task_positions(gw_setup):
    """``position_names=None`` (the new default) resolves to the task's positions."""
    from causalab.runner.helpers import build_targets_for_grid

    task, pipeline, _ = gw_setup
    _targets, positions = build_targets_for_grid(pipeline, task, [2], None)
    assert [tp.id for tp in positions] == ["last"]


def test_grid_resolver_unknown_name_lists_available(gw_setup):
    """An unknown name (e.g. the old `last_token`) raises listing the real names."""
    from causalab.runner.helpers import build_targets_for_grid

    task, pipeline, _ = gw_setup
    with pytest.raises(ValueError) as exc:
        build_targets_for_grid(pipeline, task, [2], ["last_token"])
    msg = str(exc.value)
    assert "last_token" in msg  # the bad name is echoed
    assert "last" in msg  # ...and the available name is listed


def test_single_cell_null_resolves_sole_position(gw_setup):
    """Single-cell + unset token_positions resolves graph_walk's sole `last` cell.

    This is the exact #257 failure: with the old `[last_token]` default this
    raised ``Unknown token positions ['last_token']``. It must now run and record
    the resolved position in the result (so single-cell metadata stays accurate).
    """
    from causalab.analyses.subspace.main import _run_single_cell

    task, pipeline, dataset = gw_setup
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_single_cell(
            None,  # cfg — unused on the PCA path
            pipeline,
            task,
            dataset,
            dataset,
            2,  # layer
            tmpdir,
            "pca",
            4,  # k_features
            8,  # batch_size
            None,  # intervention_metric — unused on the PCA path
            "pdf",  # figure_format
            token_positions=None,  # the new default
            colormap=None,
            vis_dims=None,
            detailed_hover=False,
            max_hover_chars=50,
            das_training_epoch=1,
            das_lr=1e-3,
        )
    assert result["token_position"] == "last"
    assert result["layer"] == 2


def test_single_cell_ambiguous_position_lists_names():
    """Single-cell + unset token_positions on a multi-position task lists the names.

    A single cell needs exactly one position; when the task declares several and
    none was chosen, the error must enumerate them rather than guess. Raises
    before any fitting, so dummy collaborators are fine.
    """
    from causalab.analyses.subspace.main import _run_single_cell

    class _TwoPosTask:
        name = "two_pos_fake"

        def create_token_positions(self, pipeline):
            return {"alpha": object(), "beta": object()}

    with pytest.raises(ValueError) as exc:
        _run_single_cell(
            None,
            object(),  # pipeline — never used; resolution raises first
            _TwoPosTask(),
            [],
            [],
            2,
            "/tmp/unused",
            "pca",
            4,
            8,
            None,
            "pdf",
            token_positions=None,
            colormap=None,
            vis_dims=None,
            detailed_hover=False,
            max_hover_chars=50,
            das_training_epoch=1,
            das_lr=1e-3,
        )
    msg = str(exc.value)
    assert "alpha" in msg and "beta" in msg
