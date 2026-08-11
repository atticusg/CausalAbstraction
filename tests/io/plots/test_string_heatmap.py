"""Tests for ``causalab.io.plots.string_heatmap``.

Pins the cell-form contract of the shared string-heatmap entry point
``plot_residual_stream_intervention_heatmap`` (review #492 F8): each cell of
``intervention_results`` is either the legacy ``raw_results`` dict view of a
run (:meth:`~causalab.neural.pipeline.GenerationResult.to_raw_results` —
``"string"`` is ONE synthetic batch, ``[[...strings...]]``) or a hand-built
flat ``{"string": ["<decoded>"]}`` (the logit-lens heatmap,
``causalab/analyses/logit_lens/prompts.py``), with empty/missing strings
falling back to the ``"∅"`` placeholder and absent keys to ``"?"``.

The cell-parsing loop is asserted through the public function with the
private renderer spied out (the renderer itself is exercised once, headless
Agg, in the save-path test)."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from causalab.io.plots import string_heatmap as string_heatmap_module
from causalab.io.plots.string_heatmap import (
    plot_residual_stream_intervention_heatmap,
)
from causalab.neural.pipeline import GenerationResult, LMPipeline
from causalab.neural.token_positions import TokenPosition

# Force Agg backend so nothing opens a window in headless CI (the precedent
# in tests/io/plots/test_causal_graph.py).
matplotlib.use("Agg")

pytestmark = pytest.mark.unit

_PROMPT = "hello world"
_KEY = (0, "first_token")


@pytest.fixture(autouse=True)
def _no_window():
    """Close all matplotlib figures after each test to keep state clean."""
    yield
    plt.close("all")


@pytest.fixture
def positions(mock_tiny_lm: LMPipeline) -> List[TokenPosition]:
    return [TokenPosition(lambda _x: [0], mock_tiny_lm, id="first_token")]


def _fill_matrices(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: LMPipeline,
    positions: List[TokenPosition],
    cells: Dict[tuple[Any, ...], Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the public heatmap fn with the renderer spied out; return the
    kwargs it would have rendered (``token_matrix`` is the direct product of
    the cell-parsing loop under test)."""
    captured: Dict[str, Any] = {}

    def spy(**kw: Any) -> None:
        captured.update(kw)

    monkeypatch.setattr(string_heatmap_module, "_render_string_heatmap", spy)
    plot_residual_stream_intervention_heatmap(
        intervention_results=cells,
        prompt=_PROMPT,
        layers=[0],
        token_positions=positions,
        pipeline=pipeline,
        **kwargs,
    )
    assert captured, "the renderer was never reached"
    return captured


class TestCellForms:
    def test_synthetic_batch_cell_from_to_raw_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tiny_lm: LMPipeline,
        positions: List[TokenPosition],
    ) -> None:
        """A cell fed straight from ``GenerationResult.to_raw_results()`` —
        ``"string"`` nested as ONE synthetic batch — decodes to its first
        example's stripped string, and the parsed token feeds correctness
        scoring."""
        result = GenerationResult(
            sequences=torch.zeros((1, 2), dtype=torch.long),
            strings=[" decoded "],
        )
        cells = {_KEY: result.to_raw_results()}
        assert cells[_KEY]["string"] == [[" decoded "]]  # the nesting under test

        got = _fill_matrices(
            monkeypatch, mock_tiny_lm, positions, cells, correct_answer="decoded"
        )
        assert got["token_matrix"][0][0] == "decoded"
        assert got["score_matrix"][0][0] == 1.0

    def test_flat_hand_built_cell(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tiny_lm: LMPipeline,
        positions: List[TokenPosition],
    ) -> None:
        """The hand-built flat form (``{"string": ["<decoded>"]}`` — how the
        logit-lens heatmap wraps its top-1 tokens) decodes identically."""
        cells = {_KEY: {"string": [" decoded "]}}
        got = _fill_matrices(monkeypatch, mock_tiny_lm, positions, cells)
        assert got["token_matrix"][0][0] == "decoded"

    @pytest.mark.parametrize(
        "cell",
        [
            {"string": []},  # empty string list
            {"string": [[]]},  # empty synthetic batch
            {"sequences": []},  # "string" missing entirely
        ],
        ids=["empty-list", "empty-batch", "missing-key"],
    )
    def test_empty_or_missing_string_falls_back_to_placeholder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tiny_lm: LMPipeline,
        positions: List[TokenPosition],
        cell: Dict[str, Any],
    ) -> None:
        got = _fill_matrices(monkeypatch, mock_tiny_lm, positions, {_KEY: cell})
        assert got["token_matrix"][0][0] == "∅"

    def test_absent_key_renders_question_mark(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_tiny_lm: LMPipeline,
        positions: List[TokenPosition],
    ) -> None:
        got = _fill_matrices(monkeypatch, mock_tiny_lm, positions, {})
        assert got["token_matrix"][0][0] == "?"
        assert got["score_matrix"][0][0] == 0.5  # gray for missing data


class TestHeadlessRender:
    def test_renders_and_saves_png(
        self, mock_tiny_lm: LMPipeline, positions: List[TokenPosition], tmp_path
    ) -> None:
        """The full public path (real renderer, Agg backend) survives
        headless and writes the figure file."""
        cells = {_KEY: {"string": [["decoded"]]}}
        out = os.path.join(str(tmp_path), "heatmap")
        plot_residual_stream_intervention_heatmap(
            intervention_results=cells,
            prompt=_PROMPT,
            layers=[0],
            token_positions=positions,
            pipeline=mock_tiny_lm,
            save_path=out,
            figure_format="png",
        )
        assert os.path.exists(out + ".png")
