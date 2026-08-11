"""Tests for :mod:`causalab.neural.positions` — the ST2 position-resolver bridge (#397).

Tiers (``causalab/neural`` owes ``unit`` + ``property``, docs/TESTS.md):

* ``unit`` — the shift math on hand-built masks (left-pad per-row offsets,
  right-pad identity, ragged rows, the zero-shift bounds check that catches
  #176-style stale positions) and the resolver dispatch in
  :func:`resolve_positions` (indexer vs. static broadcast, the
  ``attention_mask=None`` opt-out, ``is_original`` threading). The parity
  pin against ``AtomicModelUnit.index_component`` is historical: the WU6
  sweep (#508) deleted that pyvene-era consumer, so
  ``shift_to_padded_frame`` here is the *only* shift implementation and
  there is no second path to hold in sync.
* ``property`` — with a real tokenizer (the tiny-random pipelines from
  ``tests/neural/conftest.py``): ``Template``-driven variable positions on a
  ragged-length left-padded batch land on the variable's tokens *in the
  padded frame*, and paired positions route by ``is_original``. Then the full
  bridge on a fresh (nnsight-traceable) tiny Llama: resolved rows handed to
  ``Site.read`` / ``Site.write`` inside a trace match the raw
  ``register_forward_hook`` oracle gathered/edited at the same padded
  indices — resolver → padded frame → in-trace slice, end to end.

The in-trace tests build a private fresh model (never the session-cached
``tiny_random_model`` singleton, whose leftover pyvene forward hooks break a
later nnsight trace — see ``tests/_helpers/tiny.py``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.pipeline import LMPipeline
from causalab.neural.positions import resolve_positions, shift_to_padded_frame
from causalab.neural.site import Site
from causalab.neural.token_positions import (
    ComponentIndexer,
    build_token_positions,
    paired_token_position,
)

from tests._helpers.tiny import fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import (
    capture_component,
    component_edited_logits,
    component_module,
)


def _const_indexer(positions: list[int]) -> ComponentIndexer:
    """ComponentIndexer that returns ``positions`` regardless of input."""
    return ComponentIndexer(lambda _x: list(positions), id=f"const_{positions}")


def _input_sample(text: str, **vars: Any) -> CausalTrace:
    """A ``CausalTrace`` carrying ``raw_input`` plus template variables, the
    shape every ``Template``-driven factory indexes into (mirrors the helper in
    ``tests/neural/test_token_positions.py``)."""
    mechanisms: dict[str, Mechanism] = {
        "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"]),
    }
    for name in vars:
        mechanisms[name] = Mechanism(parents=[], compute=lambda t, n=name: t[n])
    return CausalTrace(mechanisms=mechanisms, inputs={"raw_input": text, **vars})


# --------------------------------------------------------------------------- #
#  unit — shift math on hand-built masks                                       #
# --------------------------------------------------------------------------- #
class TestShiftUnit:
    pytestmark = pytest.mark.unit

    def test_left_pad_shifts_each_row_by_its_pad_count(self) -> None:
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        assert shift_to_padded_frame([[0, 1], [0, 1]], mask) == [[2, 3], [0, 1]]

    def test_right_pad_is_identity(self) -> None:
        mask = torch.tensor([[1, 1, 0, 0]])
        assert shift_to_padded_frame([[0, 1]], mask) == [[0, 1]]

    def test_ragged_rows_shift_independently(self) -> None:
        # Resolution stays ragged-safe — only the batched *consumption* of
        # ragged rows is deferred (to PL3).
        mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        assert shift_to_padded_frame([[0], [0, 2]], mask) == [[1], [0, 2]]

    def test_out_of_bounds_after_shift_raises(self) -> None:
        # Unpadded index 3 is valid for the row's 2 real tokens' frame only if
        # < padded_len after the +2 shift; 3+2=5 ≥ 4 must fail loudly.
        mask = torch.tensor([[0, 0, 1, 1]])
        with pytest.raises(ValueError, match="out of bounds"):
            shift_to_padded_frame([[3]], mask)

    def test_bounds_checked_even_with_zero_shift(self) -> None:
        # The #176 guard: a stale position computed for a differently-shaped
        # input passes through unshifted (no left-padding) but must still raise
        # a catchable ValueError, never reach a CUDA gather.
        mask = torch.tensor([[1, 1, 1]])
        with pytest.raises(ValueError, match="out of bounds"):
            shift_to_padded_frame([[7]], mask)

    def test_row_count_mismatch_raises(self) -> None:
        mask = torch.tensor([[1, 1], [1, 1]])
        with pytest.raises(ValueError, match="position rows"):
            shift_to_padded_frame([[0]], mask)


# --------------------------------------------------------------------------- #
#  unit — resolve_positions dispatch                                           #
# --------------------------------------------------------------------------- #
class TestResolveUnit:
    pytestmark = pytest.mark.unit

    _MASK = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    def test_indexer_rows_land_in_padded_frame(self) -> None:
        got = resolve_positions(_const_indexer([0, 1]), ["a", "b"], self._MASK)
        assert got == [[2, 3], [0, 1]]

    def test_static_list_broadcasts_to_every_row(self) -> None:
        assert resolve_positions([1], ["a", "b"], self._MASK) == [[3], [1]]

    def test_static_1d_tensor_broadcasts_to_every_row(self) -> None:
        got = resolve_positions(torch.tensor([1]), ["a", "b"], self._MASK)
        assert got == [[3], [1]]

    def test_static_2d_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            resolve_positions(torch.tensor([[1], [2]]), ["a", "b"], self._MASK)

    def test_none_mask_is_the_unpadded_frame_opt_out(self) -> None:
        got = resolve_positions(_const_indexer([0, 1]), ["a", "b"], None)
        assert got == [[0, 1], [0, 1]]

    def test_is_original_threads_to_the_indexer(self) -> None:
        def indexer(_x: Any, is_original: bool = True) -> list[int]:
            return [0] if is_original else [1]

        paired = ComponentIndexer(indexer, id="paired")
        assert resolve_positions(paired, ["a"], None, is_original=True) == [[0]]
        assert resolve_positions(paired, ["a"], None, is_original=False) == [[1]]


# --------------------------------------------------------------------------- #
#  property — real tokenizer: Template positions survive the padded frame      #
# --------------------------------------------------------------------------- #
_TEMPLATE = "The sum of {x} and {y} is "


def _sum_sample(x: str, y: str) -> CausalTrace:
    return _input_sample(f"The sum of {x} and {y} is ", x=x, y=y)


class TestResolutionProperty:
    pytestmark = pytest.mark.property

    def test_variable_positions_survive_the_padded_frame(self, tiny_pipeline) -> None:
        """On a ragged-length left-padded batch, the resolved rows decode back
        to the variable's value in the *padded* ``input_ids`` — the whole point
        of the frame shift."""
        tp = build_token_positions(
            {"x": {"type": "variable", "name": "x"}}, _TEMPLATE, tiny_pipeline
        )["x"]
        # y makes the second prompt strictly longer, so the first row is
        # left-padded and its x-position must shift; x stays one digit so the
        # rows are equal-width.
        traces = [_sum_sample("5", "7"), _sum_sample("9", "7777777")]
        enc = tiny_pipeline.load(traces)
        mask = enc["attention_mask"]
        assert int(mask[0].sum()) < int(mask[1].sum())  # genuinely ragged

        rows = resolve_positions(tp, traces, mask)
        unshifted = resolve_positions(tp, traces, None)
        assert rows[0] != unshifted[0]  # the padded row moved ...
        assert rows[1] == unshifted[1]  # ... the full-length row did not

        for i, (row, value) in enumerate(zip(rows, ["5", "9"])):
            decoded = tiny_pipeline.tokenizer.decode(
                [enc["input_ids"][i][p].item() for p in row]
            )
            assert value in decoded

    def test_paired_positions_route_by_frame(self, tiny_pipeline) -> None:
        """``paired_token_position`` resolves the original-side spec under
        ``is_original=True`` and the counterfactual-side spec under ``False`` —
        threaded through the bridge, not around it."""
        built = build_token_positions(
            {
                "x": {"type": "variable", "name": "x"},
                "last": {"type": "index", "position": -1},
            },
            _TEMPLATE,
            tiny_pipeline,
        )
        paired = paired_token_position(built["last"], built["x"], id="last<-x")
        traces = [_sum_sample("5", "7")]

        as_original = resolve_positions(paired, traces, None, is_original=True)
        as_counterfactual = resolve_positions(paired, traces, None, is_original=False)
        assert as_original == resolve_positions(built["last"], traces, None)
        assert as_counterfactual == resolve_positions(built["x"], traces, None)
        assert as_original != as_counterfactual


# --------------------------------------------------------------------------- #
#  property — the full bridge: resolver → padded frame → in-trace Site slice   #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class BridgeCase:
    pipeline: LMPipeline  # fresh tiny Llama; .model is the StandardizedTransformer

    def resolved_batch(self) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        """A ragged-length left-padded batch + the x-variable rows resolved
        into its padded frame (equal-width, so they batch as one Site slice)."""
        tp = build_token_positions(
            {"x": {"type": "variable", "name": "x"}}, _TEMPLATE, self.pipeline
        )["x"]
        traces = [_sum_sample("5", "7"), _sum_sample("9", "7777777")]
        enc = self.pipeline.load(traces)
        inputs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        return inputs, resolve_positions(tp, traces, enc["attention_mask"])


@pytest.fixture(scope="module")
def bridge_case() -> BridgeCase:
    # Fresh (uncached) model: safe to trace. LMPipeline wraps it in a
    # dispatched StandardizedTransformer and owns the left-pad convention the
    # resolver side assumes.
    raw, _tok = fresh_tiny_random_llama()
    return BridgeCase(pipeline=LMPipeline(raw, max_new_tokens=1, padding_side="left"))


class TestBridgeToSiteProperty:
    pytestmark = pytest.mark.property

    def test_resolved_rows_read_like_the_oracle(self, bridge_case: BridgeCase) -> None:
        """``Site.read`` at the resolved per-row positions equals the raw-hook
        capture gathered at the same padded indices, row by row."""
        inputs, rows = bridge_case.resolved_batch()
        st = bridge_case.pipeline.model

        got = Site("block_output", 0).collect(st, inputs, positions=rows)

        module, kind = component_module(bridge_case.pipeline, 0, "block_output")
        full = capture_component(bridge_case.pipeline, module, kind, inputs)
        for i, row in enumerate(rows):
            torch.testing.assert_close(got[i], full[i, row, :], atol=1e-5, rtol=1e-4)

    def test_resolved_rows_write_like_the_oracle(self, bridge_case: BridgeCase) -> None:
        """``Site.write`` at the resolved per-row positions reproduces a
        hand-rolled hook edit at the same padded indices — the interchange
        write path over bridge-resolved positions."""
        inputs, rows = bridge_case.resolved_batch()
        st = bridge_case.pipeline.model
        site = Site("block_output", 0)
        hidden = int(bridge_case.pipeline.model.config.hidden_size)
        delta = torch.linspace(-10.0, 10.0, hidden)

        with st.trace(inputs):
            clean = st.logits[:, -1, :].cpu().save()
        with st.trace(inputs):
            site.write(st, site.read(st, rows) + delta, positions=rows)
            edited = st.logits[:, -1, :].cpu().save()

        module, kind = component_module(bridge_case.pipeline, 0, "block_output")

        def edit(h: torch.Tensor) -> None:
            for i, row in enumerate(rows):
                h[i, row, :] = h[i, row, :] + delta

        manual = component_edited_logits(
            bridge_case.pipeline, inputs, module, kind, edit
        )

        # The edit is non-vacuous (the oracle itself moves the logits) ...
        assert not torch.allclose(manual, clean, atol=1e-4)
        # ... and Site.write at the bridged positions reproduces it exactly.
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  batch-first resolution on the run encoding (PL3, #405)                      #
# --------------------------------------------------------------------------- #
from causalab.neural.positions import resolve_positions_batched  # noqa: E402
from causalab.neural.token_positions import (  # noqa: E402
    PromptTemplateMismatchError,
    TokenPosition,
    combined_token_position,
)


def _all_specs() -> dict[str, Any]:
    """One spec of every declarative type over ``_TEMPLATE``."""
    return {
        "last": {"type": "index", "position": -1},
        "first": {"type": "index", "position": 0},
        "x": {"type": "variable", "name": "x"},
        "y_last": {"type": "index", "position": -1, "scope": {"variable": "y"}},
        "after_x": {"type": "index", "position": 1, "relative_to": {"variable": "x"}},
        "dynamic": lambda setting: {
            "type": "variable",
            "name": "x" if str(setting["x"]) == "5" else "y",
        },
    }


def _ragged_traces() -> list[CausalTrace]:
    # y widths differ, so the batch is genuinely ragged and the first rows
    # are left-padded in the run encoding.
    return [
        _sum_sample("5", "7"),
        _sum_sample("9", "7777777"),
        _sum_sample("5", "123456789"),
    ]


class TestBatchFirstUnit:
    pytestmark = pytest.mark.unit

    def test_hand_built_token_position_returns_none(self, tiny_pipeline) -> None:
        # No declarative structure → no batch-first path; the caller falls
        # back to the legacy per-example resolve + shift.
        tp = TokenPosition(lambda _x: [0], tiny_pipeline, id="custom")
        enc = tiny_pipeline.load(_ragged_traces(), return_offsets_mapping=True)
        assert tp.index_on_encoding(_ragged_traces(), enc) is None

    def test_fallback_resolves_custom_indexer_against_run_mask(
        self, tiny_pipeline
    ) -> None:
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        tp = TokenPosition(lambda _x: [0], tiny_pipeline, id="custom")
        got = resolve_positions_batched(tp, traces, enc)
        assert got == resolve_positions(tp, traces, enc["attention_mask"])

    def test_static_list_takes_the_legacy_frame_semantics(self, tiny_pipeline) -> None:
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        assert resolve_positions_batched([1], traces, enc) == resolve_positions(
            [1], traces, enc["attention_mask"]
        )

    def test_variable_spec_requires_offsets(self, tiny_pipeline) -> None:
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces)  # no return_offsets_mapping
        tp = build_token_positions(
            {"x": {"type": "variable", "name": "x"}}, _TEMPLATE, tiny_pipeline
        )["x"]
        with pytest.raises(ValueError, match="return_offsets_mapping"):
            resolve_positions_batched(tp, traces, enc)

    def test_absolute_spec_resolves_without_offsets(self, tiny_pipeline) -> None:
        # Pure index arithmetic on the attention mask — offsets not needed.
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces)
        tp = build_token_positions(
            {"last": {"type": "index", "position": -1}}, _TEMPLATE, tiny_pipeline
        )["last"]
        got = resolve_positions_batched(tp, traces, enc)
        assert got == resolve_positions(tp, traces, enc["attention_mask"])

    def test_raw_input_mismatch_still_guards(self, tiny_pipeline) -> None:
        trace = _input_sample("something the template never produced", x="5", y="7")
        enc = tiny_pipeline.load([trace], return_offsets_mapping=True)
        tp = build_token_positions(
            {"x": {"type": "variable", "name": "x"}}, _TEMPLATE, tiny_pipeline
        )["x"]
        with pytest.raises(PromptTemplateMismatchError):
            resolve_positions_batched(tp, [trace], enc)

    def test_absolute_out_of_range_raises(self, tiny_pipeline) -> None:
        traces = [_sum_sample("5", "7")]
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        tp = build_token_positions(
            {"far": {"type": "index", "position": 10_000}}, _TEMPLATE, tiny_pipeline
        )["far"]
        with pytest.raises(ValueError, match="out of range"):
            resolve_positions_batched(tp, traces, enc)

    def test_combined_with_spec_less_member_falls_back(self, tiny_pipeline) -> None:
        built = build_token_positions(
            {"last": {"type": "index", "position": -1}}, _TEMPLATE, tiny_pipeline
        )
        custom = TokenPosition(lambda _x: [0], tiny_pipeline, id="custom")
        combo = combined_token_position([built["last"], custom], id="combo")
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        assert combo.index_on_encoding(traces, enc) is None
        got = resolve_positions_batched(combo, traces, enc)
        assert got == resolve_positions(combo, traces, enc["attention_mask"])


class TestBatchFirstParityProperty:
    """The batch-first rows equal the legacy per-example resolve + shift rows
    for every declarative spec type, on a genuinely ragged left-padded batch —
    the two paths must describe the same padded frame until SH2 deletes the
    legacy one."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("name", sorted(_all_specs(), key=str))
    def test_spec_parity_on_ragged_batch(self, tiny_pipeline, name: str) -> None:
        tp = build_token_positions(_all_specs(), _TEMPLATE, tiny_pipeline)[name]
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        assert int(enc["attention_mask"][0].sum()) < int(
            enc["attention_mask"][-1].sum()
        )
        batched = resolve_positions_batched(tp, traces, enc)
        legacy = resolve_positions(tp, traces, enc["attention_mask"])
        assert batched == legacy

    def test_paired_routes_and_matches_legacy(self, tiny_pipeline) -> None:
        built = build_token_positions(_all_specs(), _TEMPLATE, tiny_pipeline)
        paired = paired_token_position(built["last"], built["x"], id="last<-x")
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        for is_original in (True, False, None):
            batched = resolve_positions_batched(
                paired, traces, enc, is_original=is_original
            )
            legacy = resolve_positions(
                paired, traces, enc["attention_mask"], is_original=is_original
            )
            assert batched == legacy

    def test_combined_concatenates_and_matches_legacy(self, tiny_pipeline) -> None:
        built = build_token_positions(_all_specs(), _TEMPLATE, tiny_pipeline)
        combo = combined_token_position([built["x"], built["last"]], id="x+last")
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        batched = resolve_positions_batched(combo, traces, enc)
        assert batched == resolve_positions(combo, traces, enc["attention_mask"])

    def test_variable_rows_decode_to_the_value_born_padded(self, tiny_pipeline) -> None:
        # The point of batch-first: indices index the run encoding directly.
        tp = build_token_positions(_all_specs(), _TEMPLATE, tiny_pipeline)["x"]
        traces = _ragged_traces()
        enc = tiny_pipeline.load(traces, return_offsets_mapping=True)
        rows = resolve_positions_batched(tp, traces, enc)
        for i, (row, value) in enumerate(zip(rows, ["5", "9", "5"])):
            decoded = tiny_pipeline.tokenizer.decode(
                [enc["input_ids"][i][p].item() for p in row]
            )
            assert value in decoded
