"""Output-token scoring helpers in ``metric.py`` (issue #208).

Covers the space/case-agnostic token-form expansion that aligns the
probability grader with the strip-tolerant string grader, so a task shipping a
bare ``raw_output`` (e.g. ``"blue"``) no longer reads ``prob_accuracy ≈ 0``
while the model is in fact emitting the space-prefixed ``" blue"`` token.
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.metric import (
    answer_token_forms,
    compute_base_accuracy,
    make_logit_metric,
    single_token_id,
    tokenize_variable_values,
)

pytestmark = pytest.mark.unit


# The mechanical ``[" v", v]`` form map (both BPE spacings) that
# ``build_output_tokens`` declares per value — exercised here through the
# low-level ``tokenize_variable_values`` primitive.
def _mechanical_pattern(v: str) -> list[str]:
    return [f" {v}", v]


class _StubTokenizer:
    """Maps a few strings to fixed token ids; everything else encodes to a
    2-token sequence so it is filtered out as multi-token. Lets the tests pin
    exactly which forms the scoring code probes without loading a real model.
    """

    _TABLE = {
        " blue": [5],
        "blue": [6],
        " Blue": [5],  # leading-space form lowercases to the same emitted id
        "Blue": [6],
        "green": [3],  # a second single-token answer (for logit-diff tests)
        " green": [4],
        " dark": [7, 8],  # compound modifier: multi-token both spacings
        "dark": [9, 10],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._TABLE.get(text, [98, 99]))


class TestAnswerTokenForms:
    def test_bare_lowercase(self):
        # Space-prefixed first: it is the token the model emits at a word boundary.
        assert answer_token_forms("blue") == [" blue", "blue"]

    def test_capitalized_expands_case_and_spacing(self):
        assert answer_token_forms("Blue") == [" Blue", " blue", "Blue", "blue"]

    def test_leading_space_is_stripped_first(self):
        # A space-prefixed answer yields the same forms as its bare counterpart.
        assert answer_token_forms(" orange") == [" orange", "orange"]

    def test_order_stable_and_deduped(self):
        forms = answer_token_forms("RED")
        assert forms == [" RED", " red", "RED", "red"]
        assert len(forms) == len(set(forms))


class TestMechanicalPatternIntegration:
    """The figure/scoring path runs declared values through
    ``tokenize_variable_values`` with the mechanical ``[" v", v]`` forms — it
    must surface the bare *and* space-prefixed single-token ids, not just ``" v"``.
    """

    def test_tokenize_collects_both_spacings(self):
        ids = tokenize_variable_values(_StubTokenizer(), ["blue"], _mechanical_pattern)
        assert ids == [[5, 6]]

    def test_multi_token_value_falls_back_to_first_variant(self):
        # No single-token form: fall back to the first variant's full sequence
        # (" dark"). Strictly additive.
        ids = tokenize_variable_values(_StubTokenizer(), ["dark"], _mechanical_pattern)
        assert ids == [[7, 8]]


def _stub_pipeline(emitted: str, logits_row: list[float], max_new_tokens: int = 1):
    tok = _StubTokenizer()
    _max_new_tokens = max_new_tokens

    class _StubPipeline:
        tokenizer = tok
        max_new_tokens = _max_new_tokens

        def generate(self, batch_inputs):
            n = len(batch_inputs)
            scores = torch.tensor([logits_row], dtype=torch.float32).repeat(n, 1)
            return {"string": [emitted] * n, "scores": [scores]}

    return _StubPipeline()


def _exact_checker(neural_output: dict, expected: str) -> bool:
    """Exact stripped equality — the canonical single-token task checker."""
    return neural_output["string"].strip() == expected.strip()


class TestComputeBaseAccuracy:
    def test_prob_accuracy_captures_space_prefixed_token(self):
        # raw_output ships bare "blue"; the model emits " blue" (id 5). Pre-#208
        # the grader only encoded "blue" (id 6) and read prob_accuracy ~0.
        logits = [0.0] * 8
        logits[5] = 12.0
        pipeline = _stub_pipeline(" blue", logits)
        dataset = [{"input": {"raw_output": "blue"}}]

        result = compute_base_accuracy(dataset, pipeline, checker=_exact_checker)

        assert result["accuracy"] == 1.0  # strip-tolerant string match
        assert result["prob_accuracy"] is not None
        assert result["prob_accuracy"] > 0.99  # " blue" mass now counted

    def test_prob_accuracy_is_case_agnostic(self):
        # Declared answer "Blue" (capital); model emits lowercase " blue".
        logits = [0.0] * 8
        logits[5] = 12.0
        pipeline = _stub_pipeline(" blue", logits)
        dataset = [{"input": {"raw_output": "Blue"}}]

        result = compute_base_accuracy(dataset, pipeline, checker=_exact_checker)

        assert result["prob_accuracy"] > 0.99

    def test_prob_accuracy_low_when_mass_off_answer_tokens(self):
        # Sanity floor: mass parked on an unrelated id keeps prob_accuracy low,
        # so the space-agnostic expansion is not trivially saturating.
        logits = [0.0] * 8
        logits[0] = 12.0
        pipeline = _stub_pipeline(" blue", logits)
        dataset = [{"input": {"raw_output": "blue"}}]

        result = compute_base_accuracy(dataset, pipeline, checker=_exact_checker)

        assert result["prob_accuracy"] < 0.01


def _startswith_checker(neural_output: dict, expected: str) -> bool:
    """The shipped entity_binding checker: ``startswith`` accepts continuation."""
    actual = neural_output["string"].strip()
    want = expected.strip()
    return actual.startswith(want) or actual == want


class TestComputeBaseAccuracyChecker:
    """``checker`` lets a ``max_new_tokens > 1`` task score the answer that
    precedes its continuation tokens — the strict-equality scoring that ignored
    ``task.checker`` reported 0% on such tasks (issue #167).
    """

    def test_multi_token_continuation_scored_via_checker(self):
        # max_new_tokens=4: model emits the answer ("bread") then keeps going.
        pipeline = _stub_pipeline("bread\n\nAnn loves", [0.0] * 8, max_new_tokens=4)
        dataset = [{"input": {"raw_output": "bread"}}]

        result = compute_base_accuracy(dataset, pipeline, checker=_startswith_checker)

        assert result["accuracy"] == 1.0
        assert result["prob_accuracy"] is None  # not single-token

    def test_exact_checker_rejects_continuation(self):
        # Same generation scored by an exact-match checker → the continuation is
        # rejected (0%). This is why a continuation task must ship a lenient
        # (``startswith``) checker; the choice is the task's, not a global default.
        pipeline = _stub_pipeline("bread\n\nAnn loves", [0.0] * 8, max_new_tokens=4)
        dataset = [{"input": {"raw_output": "bread"}}]

        result = compute_base_accuracy(dataset, pipeline, checker=_exact_checker)

        assert result["accuracy"] == 0.0

    def test_checker_still_rejects_a_wrong_answer(self):
        # The checker is not a blanket pass: a wrong leading answer still fails.
        pipeline = _stub_pipeline("cheese\n\nAnn loves", [0.0] * 8, max_new_tokens=4)
        dataset = [{"input": {"raw_output": "bread"}}]

        result = compute_base_accuracy(dataset, pipeline, checker=_startswith_checker)

        assert result["accuracy"] == 0.0

    def test_checker_handles_multi_answer_lists(self):
        # graph_walk-style list expected: any valid answer (via checker) counts.
        pipeline = _stub_pipeline("east, then north", [0.0] * 8, max_new_tokens=4)
        dataset = [{"input": {"raw_output": ["north", "east"]}}]

        result = compute_base_accuracy(dataset, pipeline, checker=_startswith_checker)

        assert result["accuracy"] == 1.0


class _PipelineWithTokenizer:
    """Just enough of an ``LMPipeline`` for :func:`single_token_id`."""

    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()


class TestSingleTokenId:
    def test_prefers_emitted_leading_space_form(self):
        # answer_token_forms("blue") == [" blue", "blue"]; the space-prefixed
        # " blue"→[5] is the first single-token form and wins over bare "blue"→[6].
        # This is the readout-token fix: the metrics must read the id the model
        # actually emits after a word boundary, not the bare-word id.
        assert single_token_id(_PipelineWithTokenizer(), "blue") == 5

    def test_falls_through_to_space_form_when_bare_multitoken(self):
        # Stub: "red" bare encodes multi-token ([98, 99]); " red" is not in the
        # table either, so every form is multi-token and the resolver raises.
        with pytest.raises(ValueError, match="single-token"):
            single_token_id(_PipelineWithTokenizer(), "red")

    def test_raises_when_no_single_token_form(self):
        # "dark" → [9, 10] and " dark" → [7, 8] are both multi-token.
        with pytest.raises(ValueError, match="single-token"):
            single_token_id(_PipelineWithTokenizer(), "dark")


class TestMakeLogitMetric:
    """``make_logit_metric`` reads the raw logit of a fixed answer token, per
    example, optionally relative to the base (un-intervened) run.
    """

    @staticmethod
    def _dataset():
        return [{"input": {"raw_output": "blue"}}]

    @staticmethod
    def _answer_of(ex):
        return ex["input"]["raw_output"]

    def _metric(self, **kwargs):
        return make_logit_metric(
            _PipelineWithTokenizer(), self._dataset(), self._answer_of, **kwargs
        )

    def test_flags_relative_to_base_default(self):
        metric = self._metric()
        assert metric.needs_scores is True
        assert metric.needs_causal_expected is False
        assert metric.needs_original_output is True  # default relative_to_base=True

    def test_raw_logit_when_not_relative(self):
        metric = self._metric(relative_to_base=False)
        assert metric.needs_original_output is False
        # blue → emitted id 5 (" blue"); park 3.0 there in the patched run's
        # first-step logits.
        patched = torch.zeros(1, 8)
        patched[0, 5] = 3.0
        intervention_output = {"scores": [patched], "example_idx": 0}
        assert metric.fn(intervention_output, {}, {}) == pytest.approx(3.0)

    def test_relative_to_base_is_base_minus_patched(self):
        metric = self._metric(relative_to_base=True)
        patched = torch.zeros(1, 8)
        patched[0, 5] = 3.0  # ablation pushed the answer logit down to 3.0 …
        base_row = torch.zeros(8)
        base_row[5] = 5.0  # … from 5.0 in the intact run.
        intervention_output = {"scores": [patched], "example_idx": 0}
        original = {"scores": [base_row]}
        # impact = base − patched = 5.0 − 3.0 = 2.0 (positive: ablation hurt the answer)
        assert metric.fn(intervention_output, {}, original) == pytest.approx(2.0)

    def test_score_token_index_selects_generation_step(self):
        # score_token_index=1 reads the *second* generated position; the first
        # is decoy mass that must be ignored.
        metric = self._metric(relative_to_base=False, score_token_index=1)
        step0 = torch.zeros(1, 8)
        step0[0, 5] = 99.0  # decoy on the answer token at step 0
        step1 = torch.zeros(1, 8)
        step1[0, 5] = 2.0  # the value actually read
        intervention_output = {"scores": [step0, step1], "example_idx": 0}
        assert metric.fn(intervention_output, {}, {}) == pytest.approx(2.0)

    def test_missing_scores_raises(self):
        metric = self._metric(relative_to_base=False)
        with pytest.raises(ValueError, match="output_scores=True"):
            metric.fn({"example_idx": 0}, {}, {})

    def test_relative_without_base_raises(self):
        metric = self._metric(relative_to_base=True)
        patched = torch.zeros(1, 8)
        intervention_output = {"scores": [patched], "example_idx": 0}
        with pytest.raises(ValueError, match="base outputs"):
            metric.fn(intervention_output, {}, {})


class TestMakeLogitDiffMetric:
    """``make_logit_diff_metric`` reads ``logit[correct] − logit[distractor]`` per
    example, optionally relative to the base run (the direct-effect readout).
    """

    @staticmethod
    def _dataset():
        return [{"input": {"correct": "blue", "distractor": "green"}}]

    def _metric(self, **kwargs):
        from causalab.methods.metric import make_logit_diff_metric

        return make_logit_diff_metric(
            _PipelineWithTokenizer(),
            self._dataset(),
            lambda ex: ex["input"]["correct"],  # blue → emitted id 5 (" blue")
            lambda ex: ex["input"]["distractor"],  # green → emitted id 4 (" green")
            **kwargs,
        )

    def test_raw_diff_when_not_relative(self):
        metric = self._metric(relative_to_base=False)
        assert metric.needs_original_output is False
        patched = torch.zeros(1, 8)
        patched[0, 5] = 4.0  # correct
        patched[0, 4] = 1.0  # distractor
        intervention_output = {"scores": [patched], "example_idx": 0}
        # patched_diff = 4.0 − 1.0 = 3.0
        assert metric.fn(intervention_output, {}, {}) == pytest.approx(3.0)

    def test_relative_is_base_diff_minus_patched_diff(self):
        metric = self._metric(relative_to_base=True)
        assert metric.needs_original_output is True
        patched = torch.zeros(1, 8)
        patched[0, 5] = 4.0
        patched[0, 4] = 1.0  # patched_diff = 3.0
        base_row = torch.zeros(8)
        base_row[5] = 5.0
        base_row[4] = 1.0  # base_diff = 4.0
        intervention_output = {"scores": [patched], "example_idx": 0}
        original = {"scores": [base_row]}
        # direct effect = base_diff − patched_diff = 4.0 − 3.0 = 1.0
        assert metric.fn(intervention_output, {}, original) == pytest.approx(1.0)

    def test_score_token_index_selects_generation_step(self):
        metric = self._metric(relative_to_base=False, score_token_index=1)
        step0 = torch.zeros(1, 8)
        step0[0, 5] = 99.0  # decoy step ignored
        step0[0, 4] = -99.0
        step1 = torch.zeros(1, 8)
        step1[0, 5] = 4.0  # correct
        step1[0, 4] = 1.0  # distractor
        intervention_output = {"scores": [step0, step1], "example_idx": 0}
        # patched_diff at step 1 = 4.0 − 1.0 = 3.0
        assert metric.fn(intervention_output, {}, {}) == pytest.approx(3.0)

    def test_reexport_from_path_patching_is_same_object(self):
        # Back-compat: path_patching keeps exporting the (now relocated) metric.
        from causalab.methods.metric import make_logit_diff_metric as canonical
        from causalab.methods.path_patching import make_logit_diff_metric as reexport

        assert reexport is canonical
