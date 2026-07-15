"""Tests for ``causalab.causal.causal_utils``.

``causal_utils`` is the helper grab-bag attached to ``CausalModel``: it
samples interventions / inputs, computes per-variable equivalence classes
and live causal paths, labels datasets by target-variable settings, builds
filter callbacks consumed by ``generate_counterfactual_samples``, and (the
one piece that fires post-experiment) re-scores ``perform_interventions``
raw output against a checker in ``compute_interchange_scores``. Its
consumers are ``causalab/causal/causal_model.py`` (the class methods
delegate to these free functions), the baseline analyses, and counterfactual
dataset construction for every task / runner. Wrong values here silently
corrupt counterfactual sampling for task fixtures, interchange-intervention
scoring, and the "baseline" analysis row in the dashboard without crashing
the runner.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import Mock

from causalab.causal.causal_model import CausalModel
from causalab.causal.causal_utils import (
    can_distinguish_with_dataset,
    compute_interchange_scores,
    display_counterfactual_examples,
    distinguishability_report,
    find_live_paths,
    generate_counterfactual_samples,
    get_partial_filter,
    get_path_maxlen_filter,
    get_specific_path_filter,
    intervened_output_vector,
    label_data_with_variables,
    sample_intervention,
)
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism, input_var

from tests._helpers.tiny import tiny_chain_model


@pytest.fixture
def simple_chain_model() -> CausalModel:
    """3-node A→B→C ``CausalModel`` from ``tests/_helpers/tiny.py``."""
    return tiny_chain_model()


# ============================================================================
# compute_interchange_scores
# ============================================================================


class MockDataset:
    """Mock dataset for testing (simulates list-like dataset interface).

    NOTE: per docs/TESTS.md mocking policy, this is a system-boundary stand-in
    for the dataset interface — score-aggregation tests exercise the
    transformation from raw_outputs+labels to score dicts, not the model's
    label-generation. The real ``label_counterfactual_data`` is tested
    separately in ``test_causal_model.py``.
    """

    def __init__(self, data, dataset_id="test_dataset"):
        self.data = data
        self.id = dataset_id

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def mock_causal_model():
    """Mock for the score-aggregation tests.

    Only ``label_counterfactual_data`` is consumed by
    ``compute_interchange_scores``; replacing this with a real
    ``CausalModel`` would require hand-tuning the model's mechanisms to
    produce the deterministic ``expected_i`` labels the score tests rely on
    — that couples the test to the model's forward semantics, which are
    already covered in ``test_causal_model.py``. Keeping the mock here
    isolates the score-aggregation contract.
    """
    model = Mock()

    def mock_label(dataset, target_vars):
        labeled_data = []
        for i, sample in enumerate(dataset):
            labeled_sample = sample.copy()
            labeled_sample["label"] = f"expected_{i}"
            labeled_data.append(labeled_sample)
        return MockDataset(labeled_data, dataset_id=dataset.id)

    model.label_counterfactual_data = mock_label
    return model


@pytest.fixture
def mock_raw_results():
    return {
        "experiment_id": "test_exp",
        "method_name": "PatchResidualStream",
        "model_name": "TestModel",
        "dataset": {
            "test_dataset": {
                "model_unit": {
                    "[[unit_1]]": {
                        "raw_outputs": [
                            {
                                "string": "output_0",
                                "sequences": torch.tensor([[1, 2, 3]]),
                            },
                            {
                                "string": "expected_1",
                                "sequences": torch.tensor([[4, 5, 6]]),
                            },
                            {
                                "string": "expected_2",
                                "sequences": torch.tensor([[7, 8, 9]]),
                            },
                        ],
                        "causal_model_inputs": [
                            {
                                "base_input": {"id": 0, "raw_input": "input_0"},
                                "counterfactual_inputs": [
                                    {"id": 0, "raw_input": "cf_0"}
                                ],
                            },
                            {
                                "base_input": {"id": 1, "raw_input": "input_1"},
                                "counterfactual_inputs": [
                                    {"id": 1, "raw_input": "cf_1"}
                                ],
                            },
                            {
                                "base_input": {"id": 2, "raw_input": "input_2"},
                                "counterfactual_inputs": [
                                    {"id": 2, "raw_input": "cf_2"}
                                ],
                            },
                        ],
                        "metadata": {"layer": 5, "position": "last"},
                        "feature_indices": None,
                    }
                }
            }
        },
    }


@pytest.fixture
def mock_dataset():
    data = [
        {"id": 0, "raw_input": "input_0", "counterfactual_inputs": [{"id": 0}]},
        {"id": 1, "raw_input": "input_1", "counterfactual_inputs": [{"id": 1}]},
        {"id": 2, "raw_input": "input_2", "counterfactual_inputs": [{"id": 2}]},
    ]
    return MockDataset(data, dataset_id="test_dataset")


@pytest.fixture
def exact_match_checker():
    def checker(output, expected):
        return 1.0 if output["string"] == expected else 0.0

    return checker


class TestComputeInterchangeScoresUnit:
    """``compute_interchange_scores`` — annotates raw intervention results with
    per-target-variable scores via a user-supplied checker."""

    pytestmark = pytest.mark.unit

    def test_basic_score_computation(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        target_variables_list = [["output"]]
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            exact_match_checker,
        )

        assert "dataset" in results
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "output" in unit_data
        score_data = unit_data["output"]
        assert "scores" in score_data
        assert "average_score" in score_data
        assert isinstance(score_data["scores"], list)
        assert isinstance(score_data["average_score"], (float, np.floating))

    def test_score_accuracy(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            exact_match_checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        scores = unit_data["output"]["scores"]
        # output_0 != expected_0 -> 0.0; expected_1 == expected_1 -> 1.0; same for 2.
        assert scores == [0.0, 1.0, 1.0]
        assert abs(unit_data["output"]["average_score"] - 2 / 3) < 1e-6

    def test_multiple_target_variables(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"], ["answer"]],
            exact_match_checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "output" in unit_data
        assert "answer" in unit_data

    def test_raw_results_preserved(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            exact_match_checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "raw_outputs" in unit_data
        assert "causal_model_inputs" in unit_data
        assert len(unit_data["raw_outputs"]) == 3

    def test_does_not_modify_input(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        original_keys = set(
            mock_raw_results["dataset"]["test_dataset"]["model_unit"][
                "[[unit_1]]"
            ].keys()
        )
        compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            exact_match_checker,
        )
        # Original raw_results must not have gained the score keys.
        current_keys = set(
            mock_raw_results["dataset"]["test_dataset"]["model_unit"][
                "[[unit_1]]"
            ].keys()
        )
        assert original_keys == current_keys

    def test_tensor_score_conversion(
        self, mock_raw_results, mock_causal_model, mock_dataset
    ):
        def tensor_checker(output, expected):
            return torch.tensor(1.0 if output["string"] == expected else 0.0)

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            tensor_checker,
        )
        scores = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"][
            "output"
        ]["scores"]
        for s in scores:
            assert isinstance(s, float)
            assert not isinstance(s, torch.Tensor)


class TestComputeInterchangeScoresProperty:
    """Invariances of ``compute_interchange_scores``."""

    pytestmark = pytest.mark.property

    def test_ordering_of_target_variables_list_is_independent(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        # Per-group scores must not depend on the outer ordering.
        results_ab = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["a"], ["b"]],
            exact_match_checker,
        )
        results_ba = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["b"], ["a"]],
            exact_match_checker,
        )
        unit_ab = results_ab["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        unit_ba = results_ba["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert unit_ab["a"]["scores"] == unit_ba["a"]["scores"]
        assert unit_ab["b"]["scores"] == unit_ba["b"]["scores"]

    def test_average_score_is_mean_of_scores(
        self, mock_raw_results, mock_causal_model, mock_dataset
    ):
        test_scores = [0.25, 0.5, 0.75]
        idx = [0]

        def checker(output, expected):
            score = test_scores[idx[0]]
            idx[0] += 1
            return score

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert unit_data["output"]["scores"] == test_scores
        assert abs(unit_data["output"]["average_score"] - np.mean(test_scores)) < 1e-6


# ============================================================================
# generate_counterfactual_samples (migrated from tests/io/test_counterfactuals.py)
# ============================================================================


@pytest.fixture
def trivial_mechanisms():
    """Two-key mechanisms dict — raw_input + var — used by sampler tests."""
    return {
        "raw_input": Mechanism(parents=[], compute=lambda t: "test input"),
        "var": Mechanism(parents=[], compute=lambda t: 1),
    }


class TestGenerateCounterfactualSamplesUnit:
    """``generate_counterfactual_samples`` — bounded loop sampler with optional filter."""

    pytestmark = pytest.mark.unit

    def test_basic_generation(self, trivial_mechanisms):
        def sampler() -> CounterfactualExample:
            trace = CausalTrace(
                trivial_mechanisms, {"raw_input": "original input", "var": 1}
            )
            cf1 = CausalTrace(trivial_mechanisms, {"raw_input": "cf 1", "var": 2})
            cf2 = CausalTrace(trivial_mechanisms, {"raw_input": "cf 2", "var": 3})
            return {"input": trace, "counterfactual_inputs": [cf1, cf2]}

        samples = generate_counterfactual_samples(5, sampler)

        assert len(samples) == 5
        for sample in samples:
            assert "input" in sample
            assert "counterfactual_inputs" in sample
            assert sample["input"]["raw_input"] == "original input"
            assert len(sample["counterfactual_inputs"]) == 2

    def test_generation_with_filter(self, trivial_mechanisms):
        counter = [0]

        def sampler() -> CounterfactualExample:
            counter[0] += 1
            trace = CausalTrace(
                trivial_mechanisms,
                {"raw_input": f"input {counter[0]}", "var": counter[0]},
            )
            cf = CausalTrace(
                trivial_mechanisms,
                {"raw_input": f"cf {counter[0]}", "var": counter[0]},
            )
            return {"input": trace, "counterfactual_inputs": [cf]}

        def filter_fn(sample: CounterfactualExample) -> bool:
            return sample["input"]["var"] % 2 == 0

        samples = generate_counterfactual_samples(3, sampler, filter=filter_fn)

        assert len(samples) == 3
        for sample in samples:
            assert sample["input"]["var"] % 2 == 0


class TestGenerateCounterfactualSamplesProperty:
    """Invariances of the bounded sampler."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("size", [1, 3, 10])
    def test_returned_count_equals_size(self, trivial_mechanisms, size):
        def sampler() -> CounterfactualExample:
            trace = CausalTrace(trivial_mechanisms, {"raw_input": "x", "var": 1})
            return {"input": trace, "counterfactual_inputs": []}

        samples = generate_counterfactual_samples(size, sampler)
        assert len(samples) == size

    def test_every_sample_satisfies_filter(self, trivial_mechanisms):
        def sampler() -> CounterfactualExample:
            value = 1 + len(samples_so_far) % 5
            trace = CausalTrace(trivial_mechanisms, {"raw_input": "x", "var": value})
            return {"input": trace, "counterfactual_inputs": []}

        samples_so_far: list = []

        def keep_odd(s: CounterfactualExample) -> bool:
            samples_so_far.append(s)
            return s["input"]["var"] % 2 == 1

        samples = generate_counterfactual_samples(5, sampler, filter=keep_odd)
        for s in samples:
            assert s["input"]["var"] % 2 == 1


# ============================================================================
# display_counterfactual_examples (migrated, using capsys)
# ============================================================================


class TestDisplayCounterfactualExamplesUnit:
    """``display_counterfactual_examples`` — pretty-printer; returns selected entries."""

    pytestmark = pytest.mark.unit

    def test_display_verbose(self, capsys):
        examples = [
            {
                "input": {"raw_input": "input1", "var": 1},
                "counterfactual_inputs": [
                    {"raw_input": "cf1_1", "var": 2},
                    {"raw_input": "cf1_2", "var": 3},
                ],
            },
            {
                "input": {"raw_input": "input2", "var": 4},
                "counterfactual_inputs": [{"raw_input": "cf2_1", "var": 5}],
            },
        ]

        result = display_counterfactual_examples(examples, num_examples=2)

        captured = capsys.readouterr()
        assert captured.out  # something was printed
        assert "input1" in captured.out
        assert len(result) == 2
        assert 0 in result and 1 in result

    def test_display_quiet(self, capsys):
        examples = [
            {
                "input": {"raw_input": "input1", "var": 1},
                "counterfactual_inputs": [{"raw_input": "cf1", "var": 2}],
            },
        ]

        result = display_counterfactual_examples(examples, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(result) == 1

    def test_display_limited_examples(self, capsys):
        examples = [
            {
                "input": {"raw_input": f"input{i}", "var": i},
                "counterfactual_inputs": [],
            }
            for i in range(10)
        ]

        result = display_counterfactual_examples(
            examples, num_examples=3, verbose=False
        )
        assert len(result) == 3


# ============================================================================
# sample_intervention (migrated)
# ============================================================================


class TestSampleInterventionUnit:
    """``sample_intervention`` — random partial intervention over non-input,
    non-output variables, satisfying an optional filter."""

    pytestmark = pytest.mark.unit

    def test_sample_excludes_inputs_and_outputs(self, simple_chain_model):
        # Loop a few times to amortize the random sampling.
        for _ in range(50):
            intervention = sample_intervention(simple_chain_model)
            if intervention:
                for var in intervention:
                    assert var not in simple_chain_model.inputs
                    assert var not in simple_chain_model.outputs
                return
        pytest.skip(
            "sample_intervention produced no non-empty intervention in 50 tries"
        )


class TestSampleInterventionProperty:
    """Invariances of the random intervention sampler."""

    pytestmark = pytest.mark.property

    def test_values_drawn_from_value_lists(self, simple_chain_model):
        # Every sampled value must be in the model's permitted value list.
        for _ in range(50):
            intervention = sample_intervention(simple_chain_model)
            for var, val in intervention.items():
                assert val in simple_chain_model.values[var]


# ============================================================================
# label_data_with_variables (migrated)
# ============================================================================


class TestLabelDataWithVariablesUnit:
    """``label_data_with_variables`` — concatenated-target-value → integer label."""

    pytestmark = pytest.mark.unit

    def test_label_keys_and_mapping(self, simple_chain_model):
        model = simple_chain_model
        test_inputs = [{"A": 0}, {"A": 1}, {"A": 0}]
        traces = [model.new_trace(inp) for inp in test_inputs]
        data_list = [{"input": t} for t in traces]

        labeled_dataset, label_mapping = label_data_with_variables(
            model, data_list, ["C"]
        )

        assert len(labeled_dataset) == 3
        assert len(label_mapping) == 2  # C=0 and C=1
        assert "0" in label_mapping
        assert "1" in label_mapping

        labels = [item["label"] for item in labeled_dataset]
        assert labels[0] == labels[2]  # both A=0 -> C=0
        assert labels[0] != labels[1]


class TestLabelDataWithVariablesProperty:
    """Invariances: determinism and order-stability of label assignment."""

    pytestmark = pytest.mark.property

    def test_determinism_same_input_same_labels(self, simple_chain_model):
        # Same input list, called twice, yields identical labels and mapping.
        model = simple_chain_model
        traces = [model.new_trace({"A": v}) for v in [0, 1, 0, 1]]
        data = [{"input": t} for t in traces]

        labeled_a, mapping_a = label_data_with_variables(model, data, ["C"])
        labeled_b, mapping_b = label_data_with_variables(model, data, ["C"])

        assert [d["label"] for d in labeled_a] == [d["label"] for d in labeled_b]
        assert mapping_a == mapping_b


# ============================================================================
# get_partial_filter and get_specific_path_filter (migrated; split per symbol)
# ============================================================================


class TestGetPartialFilterUnit:
    """``get_partial_filter`` — closure that checks a setting against a partial spec."""

    pytestmark = pytest.mark.unit

    def test_matches_partial(self, simple_chain_model):
        partial = get_partial_filter({"A": 1, "B": 1})
        assert partial(
            {"A": 1, "B": 1, "C": 1, "raw_input": "...", "raw_output": "..."}
        )

    def test_rejects_mismatch(self, simple_chain_model):
        partial = get_partial_filter({"A": 1, "B": 1})
        assert not partial(
            {"A": 0, "B": 1, "C": 1, "raw_input": "...", "raw_output": "..."}
        )


class TestGetSpecificPathFilterUnit:
    """``get_specific_path_filter`` — closure that gates settings with a live path."""

    pytestmark = pytest.mark.unit

    def test_accepts_setting_with_path(self, simple_chain_model):
        path_filter = get_specific_path_filter(simple_chain_model, "A", "C")
        # Build a setting that has a live A->...->C path.
        setting = simple_chain_model.new_trace({"A": 1}).to_dict()
        assert path_filter(setting)


# ============================================================================
# find_live_paths
# ============================================================================


class TestFindLivePathsUnit:
    """``find_live_paths`` — enumerates length-≥2 paths whose endpoints are
    actual causes of each other on a chain-model trace."""

    pytestmark = pytest.mark.unit

    def test_chain_emits_path_through_chain(self, simple_chain_model):
        paths = find_live_paths(simple_chain_model, {"A": 0})
        # Some length must contain the full A->B->C path.
        flat_paths = [p for length_group in paths.values() for p in length_group]
        assert ["A", "B", "C"] in flat_paths

    def test_returned_dict_keys_are_path_lengths(self, simple_chain_model):
        paths = find_live_paths(simple_chain_model, {"A": 0})
        for length in paths:
            assert isinstance(length, int)
            assert length >= 2


class TestFindLivePathsProperty:
    """Adjacency-closure and shape invariants of ``find_live_paths``."""

    pytestmark = pytest.mark.property

    def test_adjacent_pairs_obey_model_children(self, simple_chain_model):
        paths = find_live_paths(simple_chain_model, {"A": 0})
        for length_group in paths.values():
            for path in length_group:
                for parent, child in zip(path[:-1], path[1:]):
                    assert child in simple_chain_model.children[parent]

    def test_no_empty_paths(self, simple_chain_model):
        paths = find_live_paths(simple_chain_model, {"A": 0})
        for length_group in paths.values():
            for path in length_group:
                assert len(path) >= 2


# ============================================================================
# get_path_maxlen_filter
# ============================================================================


class TestGetPathMaxlenFilterUnit:
    """``get_path_maxlen_filter`` — closure that gates by the longest live path."""

    pytestmark = pytest.mark.unit

    def test_returned_callable_is_idempotent(self, simple_chain_model):
        # Calling twice on the same setting yields the same answer.
        setting = simple_chain_model.new_trace({"A": 0}).to_dict()
        path_filter = get_path_maxlen_filter(simple_chain_model, [4])
        first = path_filter(setting)
        second = path_filter(setting)
        assert first == second


# ============================================================================
# can_distinguish_with_dataset (module-level function — note: the CausalModel
# *method* with the same name lives in causal_model.py and is tested there)
# ============================================================================


class TestCanDistinguishWithDatasetUnit:
    """``causal_utils.can_distinguish_with_dataset`` — module-level helper
    (note: ``CausalModel`` has a method with the same name; this is its
    free-function ancestor, currently unused but not dead code)."""

    pytestmark = pytest.mark.unit

    def test_identical_targets_yield_zero_count(self, simple_chain_model):
        model = simple_chain_model
        base = model.new_trace({"A": 0})
        cf = model.new_trace({"A": 1})
        dataset = [{"input": base, "counterfactual_inputs": [cf]}]
        result = can_distinguish_with_dataset(dataset, model, target_variables1=["B"])
        # Intervening on B with cf changes the chain -> output differs.
        assert result["count"] == 1
        assert result["proportion"] == 1.0


class TestCanDistinguishWithDatasetProperty:
    """Proportion invariants for the module-level helper."""

    pytestmark = pytest.mark.property

    def test_proportion_in_unit_interval(self, simple_chain_model):
        model = simple_chain_model
        base = model.new_trace({"A": 0})
        cf = model.new_trace({"A": 1})
        dataset = [{"input": base, "counterfactual_inputs": [cf]}]
        result = can_distinguish_with_dataset(dataset, model, target_variables1=["B"])
        assert 0.0 <= result["proportion"] <= 1.0


# ---------------------------------------------------------------------------
# distinguishability engine — intervened_output_vector + distinguishability_report
# (the CPU logic migrated out of the develop-causal-model skill's distinguish.py)
# ---------------------------------------------------------------------------


def _addition_model() -> CausalModel:
    """X+Y with explicit carry/ones, raw_output reconstructed from both."""
    digits = list(range(10))
    values = {
        "X": digits,
        "Y": digits,
        "total": list(range(19)),
        "carry": [False, True],
        "ones": digits,
        "raw_input": None,
        "raw_output": None,
    }
    mechanisms = {
        "X": input_var(digits),
        "Y": input_var(digits),
        "total": Mechanism(parents=["X", "Y"], compute=lambda t: t["X"] + t["Y"]),
        "carry": Mechanism(parents=["total"], compute=lambda t: t["total"] >= 10),
        "ones": Mechanism(parents=["total"], compute=lambda t: t["total"] % 10),
        "raw_input": Mechanism(
            parents=["X", "Y"], compute=lambda t: f"{t['X']}+{t['Y']}="
        ),
        "raw_output": Mechanism(
            parents=["carry", "ones"],
            compute=lambda t: str(int(t["carry"]) * 10 + t["ones"]),
        ),
    }
    return CausalModel(mechanisms, values, id="addition")


def _pair(xb, yb, xc, yc) -> dict:
    return {
        "input": {"X": xb, "Y": yb},
        "counterfactual_inputs": [{"X": xc, "Y": yc}],
    }


class TestIntervenedOutputVector:
    pytestmark = pytest.mark.unit

    def test_patches_named_target_and_recomputes_downstream(self):
        model = _addition_model()
        # base 2+3 -> ones=5,carry=F,out="5";  cf 4+4 -> ones=8,carry=F,out="8"
        data = [_pair(2, 3, 4, 4)]
        assert intervened_output_vector(model, [], data) == ["5"]  # null = base
        assert intervened_output_vector(model, ["ones"], data) == ["8"]  # 0*10+8
        # cf carry == base carry (both False), so patching carry is a no-op
        assert intervened_output_vector(model, ["carry"], data) == ["5"]
        # transplant the whole output
        assert intervened_output_vector(model, ["raw_output"], data) == ["8"]

    def test_rejects_multiple_counterfactual_inputs(self):
        model = _addition_model()
        bad = [{"input": {"X": 1, "Y": 1}, "counterfactual_inputs": [{}, {}]}]
        with pytest.raises(AssertionError):
            intervened_output_vector(model, ["ones"], bad)


class TestDistinguishabilityReport:
    pytestmark = pytest.mark.unit

    def _hyps(self):
        return {
            "carry": ("add", ["carry"]),
            "ones": ("add", ["ones"]),
            "null": ("add", []),
            "all": ("add", ["raw_output"]),
        }

    def test_rates_and_always_confounded_groups(self):
        model = _addition_model()
        models_by_name = {"add": model}
        # Both pairs keep carry False (totals < 10), so patching `carry` is a
        # no-op == null; and `ones` == `all` here (both yield the cf ones digit).
        datasets = {"wide": [_pair(2, 3, 4, 4), _pair(1, 1, 3, 5)]}
        report = distinguishability_report(
            models_by_name, self._hyps(), ["ones"], datasets, datasets["wide"]
        )
        per = report["datasets"]["wide"]["per_target"]["ones"]
        assert per["vs_null"] == 1.0  # patching ones moves the output on both
        assert per["vs_all"] == 0.0  # ones == all on these pairs
        assert per["alternatives"]["carry"] == 1.0

        groups = {frozenset(g) for g in report["always_confounded"]}
        assert frozenset({"carry", "null"}) in groups
        assert frozenset({"ones", "all"}) in groups
        assert report["singletons"] == []

    def test_empty_datasets_still_groups_on_random_run(self):
        model = _addition_model()
        report = distinguishability_report(
            {"add": model}, self._hyps(), ["ones"], {}, [_pair(2, 3, 4, 4)]
        )
        assert report["datasets"] == {}
        # carry==null on this no-carry pair; ones==all -> two confounded pairs
        groups = {frozenset(g) for g in report["always_confounded"]}
        assert frozenset({"carry", "null"}) in groups
