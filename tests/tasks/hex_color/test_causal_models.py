"""Property-tier invariants for the ``hex_color`` causal model.

``hex_color`` casts perceptual colour naming as a tiny causal DAG
``hex → color → raw_output`` over 600 bundled stimuli (100 per colour × 6;
``indigo`` dropped — see the package docstrings).
The mechanisms are pure-symbolic — no LM, no analysis, no runner — so
these tests run sub-second and cover the *invariants* (things that hold
for every stimulus, every seed). Numerical pinning of specific symbolic
outputs lives in the sibling ``test_hex_color_numerical.py`` +
``pinned_samples.json``.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke comes from
``tests/end_to_end/configs/smoke/hex_color.yaml``; this file satisfies
property-direct; the sibling numerical-tier test satisfies numerical-direct.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.tasks.hex_color.causal_models import (
    CAUSAL_MODEL,
    HEX_TO_LABEL,
    HEXES,
    HEXES_BY_COLOR,
    STIMULI,
)
from causalab.tasks.hex_color.config import COLORS, HUE_CENTERS_DEG, HUE_PERIOD
from causalab.tasks.loader import load_task
from causalab.tasks.hex_color.counterfactuals import (
    COUNTERFACTUAL_GENERATORS,
    different_color,
    generate_dataset,
    random_counterfactual,
    same_color_different_hex,
)

_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestHexColorStructureProperty:
    """Static structural invariants of the causal DAG."""

    pytestmark = pytest.mark.property

    def test_required_variables_present(self) -> None:
        assert set(CAUSAL_MODEL.variables) == {
            "hex",
            "color",
            "raw_input",
            "raw_output",
        }

    def test_hex_is_the_only_input(self) -> None:
        assert CAUSAL_MODEL.inputs == ["hex"]

    def test_color_depends_only_on_hex(self) -> None:
        assert CAUSAL_MODEL.parents["color"] == ["hex"]

    def test_raw_input_depends_only_on_hex(self) -> None:
        assert CAUSAL_MODEL.parents["raw_input"] == ["hex"]

    def test_raw_output_depends_only_on_color(self) -> None:
        assert CAUSAL_MODEL.parents["raw_output"] == ["color"]

    def test_target_variable_values_are_the_six_colours(self) -> None:
        assert CAUSAL_MODEL.values["color"] == COLORS
        assert len(COLORS) == 6
        assert "indigo" not in COLORS


class TestHexColorOutputTokensProperty:
    """The answer's surface-form declaration drives scoring."""

    pytestmark = pytest.mark.property

    def test_output_tokens_declared_for_color(self) -> None:
        assert CAUSAL_MODEL.output_tokens is not None
        assert set(CAUSAL_MODEL.output_tokens) == {"color"}

    def test_six_way_output_tokens(self) -> None:
        assert set(CAUSAL_MODEL.output_tokens["color"]) == set(COLORS)
        assert len(CAUSAL_MODEL.output_tokens["color"]) == 6

    def test_each_colour_has_space_and_bare_forms(self) -> None:
        for c in COLORS:
            assert CAUSAL_MODEL.output_tokens["color"][c] == [f" {c}", c]


class TestHexColorPeriodicEmbeddingProperty:
    """The hue-circle geometry of the ``color`` variable."""

    pytestmark = pytest.mark.property

    def test_color_has_360_period(self) -> None:
        assert CAUSAL_MODEL.periods == {"color": HUE_PERIOD}
        assert HUE_PERIOD == 360.0

    def test_embedding_is_the_hue_centre(self) -> None:
        assert set(CAUSAL_MODEL.embeddings) == {"color"}
        for c in COLORS:
            assert CAUSAL_MODEL.embeddings["color"](c) == [HUE_CENTERS_DEG[c]]

    def test_hue_centres_within_circle(self) -> None:
        for c in COLORS:
            assert 0.0 <= HUE_CENTERS_DEG[c] < HUE_PERIOD


class TestHexColorBundledDataProperty:
    """Invariants of the committed 600-stimulus set (100 per colour × 6)."""

    pytestmark = pytest.mark.property

    def test_six_hundred_stimuli(self) -> None:
        assert len(STIMULI) == 600
        assert len(HEXES) == 600

    def test_hex_domain_matches_bundled_hexes(self) -> None:
        assert CAUSAL_MODEL.values["hex"] == HEXES

    def test_hundred_per_colour(self) -> None:
        assert set(HEXES_BY_COLOR) == set(COLORS)
        for c in COLORS:
            assert len(HEXES_BY_COLOR[c]) == 100

    def test_every_hex_maps_to_a_declared_colour(self) -> None:
        assert set(HEX_TO_LABEL.values()) == set(COLORS)
        assert len(HEX_TO_LABEL) == 600

    def test_no_indigo_in_the_bundle(self) -> None:
        # ``indigo`` was excluded at build time (7 → 6 colours).
        assert all(row["label"] != "indigo" for row in STIMULI)

    def test_data_carries_model_agnostic_fields_only(self) -> None:
        # No tokenizer/position/model-specific fields leaked into the bundle.
        assert set(STIMULI[0]) == {"hex", "r", "g", "b", "h", "s", "v", "label"}


class TestHexColorSampleProperty:
    """Sampling invariants that must hold for every seed."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_color_is_the_stimulus_label(self, seed: int) -> None:
        random.seed(seed)
        s = CAUSAL_MODEL.sample_input()
        assert s["color"] == HEX_TO_LABEL[s["hex"]]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_input_substitutes_hex(self, seed: int) -> None:
        random.seed(seed)
        s = CAUSAL_MODEL.sample_input()
        assert "{hex}" not in s["raw_input"]
        assert s["hex"] in s["raw_input"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_output_is_space_prefixed_colour(self, seed: int) -> None:
        random.seed(seed)
        s = CAUSAL_MODEL.sample_input()
        assert s["raw_output"] == " " + s["color"]


class TestHexColorCounterfactualProperty:
    """Invariants the three counterfactual generators must preserve."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_different_color_changes_the_label(self, seed: int) -> None:
        random.seed(seed)
        cf = different_color()
        assert cf["input"]["color"] != cf["counterfactual_inputs"][0]["color"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_same_color_different_hex_keeps_label_changes_hex(self, seed: int) -> None:
        random.seed(seed)
        cf = same_color_different_hex()
        base, ctf = cf["input"], cf["counterfactual_inputs"][0]
        assert base["color"] == ctf["color"]
        assert base["hex"] != ctf["hex"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_yields_two_well_formed_stimuli(self, seed: int) -> None:
        random.seed(seed)
        cf = random_counterfactual()
        for trace in (cf["input"], cf["counterfactual_inputs"][0]):
            assert trace["hex"] in HEX_TO_LABEL
            assert trace["color"] == HEX_TO_LABEL[trace["hex"]]

    def test_generate_dataset_is_deterministic_under_fixed_seed(self) -> None:
        """``generate_dataset(model, n, seed)`` is reproducible via its local
        ``random.Random(seed)`` (global RNG untouched). The zero-arg wrappers are
        intentionally *independent* single samples (fresh ``random.Random()``), so
        determinism is asserted at the seeded ``generate_dataset`` boundary — the
        contract that actually matters for the pipeline."""
        first = generate_dataset(None, 8, seed=123)
        second = generate_dataset(None, 8, seed=123)
        assert len(first) == len(second) == 8
        for a, b in zip(first, second):
            assert a["input"].to_dict() == b["input"].to_dict()
            assert (
                a["counterfactual_inputs"][0].to_dict()
                == b["counterfactual_inputs"][0].to_dict()
            )

    def test_generator_registry_names(self) -> None:
        assert set(COUNTERFACTUAL_GENERATORS) == {
            "different_color",
            "same_color_different_hex",
            "random",
        }


class TestHexColorDerivedCheckerProperty:
    """Scoring relies on the loader-*derived* checker (no bespoke checker.py).

    With ``indigo`` (the only multi-token colour) dropped, all six colours are
    single-token, so ``derive_checker`` over ``output_tokens`` (exact stripped
    match) grades correctly and the bespoke ``checker.py`` is retired.
    """

    pytestmark = pytest.mark.property

    def test_no_bespoke_checker_module(self) -> None:
        import importlib.util

        assert importlib.util.find_spec("causalab.tasks.hex_color.checker") is None, (
            "bespoke checker.py should be retired (6 single-token colours)"
        )

    def test_derived_checker_grades_colour_words(self) -> None:
        task = load_task("hex_color")
        for c in COLORS:
            assert task.checker({"string": c}, " " + c)
            assert task.checker({"string": " " + c}, c)  # spacing/strip tolerant

    def test_derived_checker_rejects_wrong_colour(self) -> None:
        task = load_task("hex_color")
        assert not task.checker({"string": "red"}, " blue")
        assert not task.checker({"string": "green"}, " purple")
