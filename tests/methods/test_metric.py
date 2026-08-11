"""Output-token scoring helpers in ``metric.py`` (issue #208).

Covers the space/case-agnostic token-form expansion that aligns the
probability grader with the strip-tolerant string grader, so a task shipping a
bare ``raw_output`` (e.g. ``"blue"``) no longer reads ``prob_accuracy ≈ 0``
while the model is in fact emitting the space-prefixed ``" blue"`` token.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.pipeline import GenerationResult
from causalab.methods.metric import (
    InterchangeMetric,
    answer_token_forms,
    as_label_checker,
    as_generation_result,
    compute_base_accuracy,
    make_logit_metric,
    outputs_from_logits,
    score_base_outputs,
    score_intervention_outputs,
    score_label_predictions,
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
    # First single-token surface form per id — enough decode for the adapter
    # tests (argmax id → string), the inverse of the encode table above.
    _DECODE = {5: " blue", 6: "blue", 3: "green", 4: " green"}

    def encode(self, text, add_special_tokens=False):
        return list(self._TABLE.get(text, [98, 99]))

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [
            "".join(self._DECODE.get(int(t), "?") for t in seq) for seq in sequences
        ]


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
            return GenerationResult(
                sequences=torch.zeros((n, _max_new_tokens), dtype=torch.long),
                strings=[emitted] * n,
                scores=[scores],
            )

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
    """Just enough of an ``LMPipeline`` for :func:`single_token_id`, the
    Plan-logits adapter (``tokenizer.batch_decode``) and the label-prediction
    scorer (``dump``)."""

    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()

    def dump(self, token_ids):
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return decoded[0] if len(decoded) == 1 else decoded


class TestOutputsFromLogits:
    """The MX1 scoring adapter (#408): Plan-saved logits → per-example
    ``{"string", "sequences", "scores"}`` outputs, the shape every scorer
    consumes."""

    def _logits(self) -> torch.Tensor:
        # Two examples, three positions, vocab 8. Argmax at the LAST position
        # is id 5 (" blue") / id 3 ("green"); an earlier position carries a
        # decoy peak that must be ignored (prefill logits ≠ next-token step).
        logits = torch.zeros(2, 3, 8)
        logits[0, -1, 5] = 9.0
        logits[1, -1, 3] = 9.0
        logits[0, 0, 6] = 99.0  # decoy at a non-final position
        return logits

    def test_last_position_argmax_decodes_to_string(self):
        outputs = outputs_from_logits(_PipelineWithTokenizer(), self._logits())
        assert [o["string"] for o in outputs] == [" blue", "green"]
        # Each output carries its argmax token id as a (1, 1) sequences row
        # (EU5b, #487) — what as_generation_result stacks.
        assert [o["sequences"].tolist() for o in outputs] == [[[5]], [[3]]]

    def test_scores_are_the_last_position_row(self):
        logits = self._logits()
        outputs = outputs_from_logits(_PipelineWithTokenizer(), logits)
        assert len(outputs[0]["scores"]) == 1  # prefill-only → ONE step
        assert torch.equal(outputs[0]["scores"][0], logits[0, -1])
        assert torch.equal(outputs[1]["scores"][0], logits[1, -1])

    def test_accepts_per_example_rows_of_ragged_length(self):
        # collect_dataset_features returns one (seq, vocab) row per example;
        # rows may differ in seq length — only the last position matters.
        short = torch.zeros(2, 8)
        short[-1, 6] = 1.0  # "blue"
        long = torch.zeros(5, 8)
        long[-1, 4] = 1.0  # " green"
        outputs = outputs_from_logits(_PipelineWithTokenizer(), [short, long])
        assert [o["string"] for o in outputs] == ["blue", " green"]


class TestAsGenerationResult:
    def _outputs(self) -> list[dict]:
        logits = torch.zeros(2, 3, 8)
        logits[0, -1, 5] = 9.0  # " blue"
        logits[1, -1, 3] = 9.0  # "green"
        return outputs_from_logits(_PipelineWithTokenizer(), logits)

    def test_flattens_per_example_outputs(self):
        result = as_generation_result(self._outputs())
        assert isinstance(result, GenerationResult)
        assert result.strings == [" blue", "green"]
        assert result.sequences.shape == (2, 1)  # per-example (1, 1) rows, stacked
        assert result.sequences.squeeze(1).tolist() == [5, 3]  # the argmax ids
        assert result.scores is not None
        assert len(result.scores) == 1  # prefill-only → ONE step
        assert result.scores[0].shape == (2, 8)  # tok0 (N, V)

    def test_string_only_when_step_counts_are_mixed(self):
        # Mixed per-example step counts cannot form flat (N, V) steps — the
        # flattened value scores strings only (the retired as_raw_results
        # contract, preserved).
        seq = torch.zeros(1, 1, dtype=torch.long)
        result = as_generation_result(
            [
                {"string": "a", "sequences": seq},
                {"string": "b", "sequences": seq, "scores": []},
            ]
        )
        assert result.strings == ["a", "b"]
        assert result.scores is None

    def test_roundtrip_through_score_intervention_outputs(self):
        # End-to-end: Plan logits → adapter → the shipped scorer, both the
        # string path (checker) and the scores path (example_idx-addressed).
        outputs = self._outputs()

        def fn(intervention_output, expected, original):
            row = intervention_output["scores"][0][intervention_output["example_idx"]]
            assert row.shape == (8,)
            return float(intervention_output["string"] == " blue")

        metric = InterchangeMetric(fn=fn, needs_causal_expected=False)
        scores = score_intervention_outputs(
            results={("k",): as_generation_result(outputs)},
            dataset=[{"input": {}}, {"input": {}}],  # length only (no causal labels)
            metric=metric,
        )
        assert scores[("k",)] == pytest.approx(0.5)

    def test_score_intervention_outputs_refuses_top_k(self):
        # Metrics consume full-vocab per-step tensors; a top-k-compressed
        # result cannot be scored and refuses loudly (legacy crashed with an
        # opaque TypeError inside torch.cat).
        result = GenerationResult(
            sequences=torch.zeros(1, 1, dtype=torch.long),
            strings=["a"],
            scores_top_k=[{"top_k_logits": torch.zeros(1, 2)}],
        )
        metric = InterchangeMetric(fn=lambda o, e, g: 1.0, needs_causal_expected=False)
        with pytest.raises(ValueError, match="top-k"):
            score_intervention_outputs(
                results={("k",): result},
                dataset=[{"input": {}}],
                metric=metric,
            )


class TestScoreBaseOutputs:
    """The grading semantics behind ``compute_base_accuracy``, off saved
    outputs — one implementation for generation runs and Plan-saved logits."""

    def test_matches_compute_base_accuracy(self):
        logits = [0.0] * 8
        logits[5] = 12.0
        pipeline = _stub_pipeline(" blue", logits)
        dataset = [{"input": {"raw_output": "blue"}}]
        via_pipeline = compute_base_accuracy(dataset, pipeline, checker=_exact_checker)

        outputs = [{"string": " blue", "scores": [torch.tensor(logits)]}]
        via_outputs = score_base_outputs(
            outputs,
            dataset,
            _exact_checker,
            tokenizer=pipeline.tokenizer,
            single_token=True,
        )
        assert via_outputs == via_pipeline

    def test_single_token_inferred_from_one_score_row(self):
        logits = torch.zeros(8)
        logits[5] = 12.0
        outputs = [{"string": " blue", "scores": [logits]}]
        result = score_base_outputs(
            outputs,
            [{"input": {"raw_output": "blue"}}],
            _exact_checker,
            tokenizer=_StubTokenizer(),
        )
        assert result["accuracy"] == 1.0
        assert result["prob_accuracy"] > 0.99

    def test_no_tokenizer_skips_prob_accuracy(self):
        outputs = [{"string": " blue", "scores": [torch.zeros(8)]}]
        result = score_base_outputs(
            outputs, [{"input": {"raw_output": "blue"}}], _exact_checker
        )
        assert result["accuracy"] == 1.0
        assert result["prob_accuracy"] is None

    def test_misaligned_outputs_raise(self):
        with pytest.raises(ValueError, match="misaligned"):
            score_base_outputs(
                [{"string": "blue"}],
                [{"input": {"raw_output": "blue"}}] * 2,
                _exact_checker,
            )


class TestScoreLabelPredictions:
    """The answer-scoring half of ``LM_loss_and_metric_fn``, consuming the ED3
    loss slice's ``pred_ids`` (MX1 owns scoring; the forward is trainable.py's)."""

    def test_scores_decoded_predictions_via_checker(self):
        pred_ids = torch.tensor([[5], [3]])  # " blue", "green"
        result = score_label_predictions(
            _PipelineWithTokenizer(), pred_ids, ["blue", "blue"], _exact_checker
        )
        assert result["scores"] == [1.0, 0.0]
        assert result["accuracy"] == pytest.approx(0.5)

    def test_dict_labels_via_as_label_checker(self):
        pred_ids = torch.tensor([[5]])
        result = score_label_predictions(
            _PipelineWithTokenizer(),
            pred_ids,
            [{"string": "blue"}],
            as_label_checker(_exact_checker),
        )
        assert result["accuracy"] == 1.0

    def test_misaligned_labels_raise(self):
        with pytest.raises(ValueError, match="misaligned"):
            score_label_predictions(
                _PipelineWithTokenizer(),
                torch.tensor([[5]]),
                ["a", "b"],
                _exact_checker,
            )


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
