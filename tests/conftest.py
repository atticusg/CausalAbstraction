# tests/conftest.py

import gc

import pytest
import torch
import random
import numpy as np

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

# Tier markers recognised by the warn-mode collection hook below. Mirrors the
# tier taxonomy in docs/TESTS.md. Tests must declare exactly one of
# {smoke, numerical_unit, golden, property, unit} explicitly. `golden` is the
# sole GPU tier (value-pinned coherent-model runners); the rest are CPU. The
# default tier (`unit`) is the most common; tag with
# `pytestmark = pytest.mark.unit` at module scope.
_TIER_MARKERS = frozenset({"smoke", "numerical_unit", "golden", "property", "unit"})


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Reset the smoke flag so off-test composition (e.g. fixture teardown
    that happens to call ``load_runner_config``) defaults to non-smoke.

    After teardown, release GPU memory leaked by golden-tier tests (#442).
    Golden tests load multi-GB coherent backbones as function locals or
    module-scoped fixtures with no teardown of their own; the nnsight/nnterp
    object graphs are cyclic, so those models survive the end of the test
    until a full ``gc.collect()`` pass — which CPython triggers on object
    counts, not GPU bytes, i.e. effectively never under this workload. In a
    single-process ``pytest -m golden`` run the dead models accumulate until
    whichever test loads a fresh backbone last OOMs. Running post-``yield``
    puts the collection after fixture finalization, so module-scoped model
    fixtures are reclaimed at module boundaries too. Gated on the ``golden``
    marker (the sole GPU tier) so CPU tiers don't pay for per-test GC.
    """
    try:
        return (yield)
    finally:
        if "golden" in {m.name for m in item.iter_markers()}:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def pytest_collection_modifyitems(config, items):
    """Fail-mode tier-marker enforcement (Phase 7).

    Collects every item lacking any of ``smoke / numerical / property /
    unit`` markers and raises :class:`pytest.UsageError` if any are
    found. The message lists the offending nodeids so the operator can
    paste a ``pytestmark = pytest.mark.<tier>`` line into each module.

    Phase 2 shipped this as a warn-mode hook; Phase 7 promotes it to
    fail-mode alongside a backfill that tags every existing test file.
    The xdist master-process gate stays — only the master process
    decides whether the suite is well-formed; worker collections are
    a strict subset of the master's so they'd surface the same untagged
    set redundantly.
    """
    if getattr(config, "workerinput", None) is not None:
        # xdist worker — let the master enforce.
        return

    untagged = [
        item.nodeid
        for item in items
        if not _TIER_MARKERS.intersection(m.name for m in item.iter_markers())
    ]
    if not untagged:
        return

    # Truncate the offending list so a global drift doesn't dump
    # thousands of lines into the terminal. The first ten and a count
    # are enough to see what's going on.
    preview = "\n  ".join(untagged[:10])
    suffix = ""
    if len(untagged) > 10:
        suffix = f"\n  ... and {len(untagged) - 10} more"
    raise pytest.UsageError(
        f"{len(untagged)} tests lack a tier marker "
        f"(smoke/numerical_unit/golden/property/unit). Tag them with "
        f"`pytestmark = pytest.mark.<tier>` at module scope:\n"
        f"  {preview}{suffix}"
    )


@pytest.fixture(scope="session")
def seed_everything():
    """Set random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session")
def mcqa_causal_model():
    """
    Create a simple MCQA causal model fixture.

    This model represents a simplified version of the multiple-choice question answering task
    where questions have 4 choices and 1 correct answer.
    """
    # Define model variables
    NUM_CHOICES = 4
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Define object/color pairs for the questions
    COLOR_OBJECTS = [
        ("red", "apple"),
        ("yellow", "banana"),
        ("green", "leaf"),
        ("blue", "sky"),
        ("brown", "chocolate"),
        ("white", "snow"),
        ("black", "coal"),
        ("purple", "grape"),
        ("orange", "carrot"),
        ("pink", "flamingo"),
        ("gray", "elephant"),
        ("gold", "coin"),
    ]

    COLORS = [item[0] for item in COLOR_OBJECTS]

    # Define values
    values = {f"choice{x}": COLORS for x in range(NUM_CHOICES)}
    values.update({f"symbol{x}": list(ALPHABET) for x in range(NUM_CHOICES)})
    values.update(
        {"answer_pointer": list(range(NUM_CHOICES)), "answer": list(ALPHABET)}
    )
    values.update({"question": COLOR_OBJECTS})
    values.update({"raw_input": None, "raw_output": None})

    def _fill_prompt(t):
        color, obj = t["question"]
        lines = [f"Which color is the {obj}?"]
        for i in range(NUM_CHOICES):
            lines.append(f"{t[f'symbol{i}']}: {t[f'choice{i}']}")
        lines.append("Answer:")
        return "\n".join(lines)

    # Define mechanisms using new API
    mechanisms: dict[str, Mechanism] = {
        "question": input_var(COLOR_OBJECTS),
        **{f"symbol{i}": input_var(list(ALPHABET)) for i in range(NUM_CHOICES)},
        **{f"choice{i}": input_var(COLORS) for i in range(NUM_CHOICES)},
        "answer_pointer": Mechanism(
            parents=["question"] + [f"choice{x}" for x in range(NUM_CHOICES)],
            compute=lambda t: next(
                (i for i in range(NUM_CHOICES) if t[f"choice{i}"] == t["question"][0]),
                random.randint(0, NUM_CHOICES - 1),
            ),
        ),
        "answer": Mechanism(
            parents=["answer_pointer"] + [f"symbol{x}" for x in range(NUM_CHOICES)],
            compute=lambda t: " " + t[f"symbol{t['answer_pointer']}"],
        ),
        "raw_input": Mechanism(
            parents=["question"]
            + [f"symbol{x}" for x in range(NUM_CHOICES)]
            + [f"choice{x}" for x in range(NUM_CHOICES)],
            compute=_fill_prompt,
        ),
        "raw_output": Mechanism(
            parents=["answer"],
            compute=lambda t: t["answer"],
        ),
    }

    # Create and return the model
    return CausalModel(
        mechanisms=mechanisms,
        values=values,
        id="4_answer_MCQA_test",
    )


@pytest.fixture(scope="session")
def mcqa_counterfactual_datasets(mcqa_causal_model, seed_everything):
    """
    Generate test counterfactual datasets for the MCQA task.

    Returns a dictionary with 3 types of counterfactual datasets:
    1. random_letter - Swapping the letter symbols while keeping choices same
    2. random_position - Moving the correct answer to a different position
    3. combined - Both letter and position changes

    Each type has small train and test sets.
    """
    model = mcqa_causal_model
    NUM_CHOICES = 4

    # Helper to check if inputs are well-formed
    def is_input_valid(x):
        # Check that the question color appears in the choices
        question_color = x["question"][0]
        choice_colors = [x[f"choice{i}"] for i in range(NUM_CHOICES)]
        symbols = [x[f"symbol{i}"] for i in range(NUM_CHOICES)]

        # The color must be in the choices and symbols must be unique
        return question_color in choice_colors and len(symbols) == len(set(symbols))

    # Counterfactual generator functions
    def random_letter_counterfactual():
        """Generate counterfactual with new random letters."""
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy

        # Get current symbols to avoid duplicates
        used_symbols = [input_setting[f"symbol{i}"] for i in range(NUM_CHOICES)]

        # Generate new set of symbols
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        available_symbols = [s for s in alphabet if s not in used_symbols]
        new_symbols = random.sample(available_symbols, NUM_CHOICES)

        # Update symbols in counterfactual
        for i in range(NUM_CHOICES):
            counterfactual[f"symbol{i}"] = new_symbols[i]

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    def random_position_counterfactual():
        """Generate counterfactual with answer moved to new position."""
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy

        # Get current answer position
        answer_position = model.new_trace(input_setting)["answer_pointer"]

        # Choose a different position
        available_positions = [i for i in range(NUM_CHOICES) if i != answer_position]
        new_position = random.choice(available_positions)

        # Swap choices to move correct answer
        correct_color = counterfactual[f"choice{answer_position}"]
        counterfactual[f"choice{answer_position}"] = counterfactual[
            f"choice{new_position}"
        ]
        counterfactual[f"choice{new_position}"] = correct_color

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    def combined_counterfactual():
        """Generate counterfactual with both letter and position changes."""
        letter_cf = random_letter_counterfactual()

        # Start with the letter-changed counterfactual
        input_setting = letter_cf["input"]
        counterfactual = letter_cf["counterfactual_inputs"][0]

        # Now change position too
        answer_position = model.new_trace(input_setting)["answer_pointer"]
        available_positions = [i for i in range(NUM_CHOICES) if i != answer_position]
        new_position = random.choice(available_positions)

        # Swap choices
        correct_color = counterfactual[f"choice{answer_position}"]
        counterfactual[f"choice{answer_position}"] = counterfactual[
            f"choice{new_position}"
        ]
        counterfactual[f"choice{new_position}"] = correct_color

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    # Generate the datasets
    datasets = {}

    # Small size for tests
    train_size = 10
    test_size = 5

    # Generate datasets for each counterfactual type
    for name, generator in [
        ("random_letter", random_letter_counterfactual),
        ("random_position", random_position_counterfactual),
        ("combined", combined_counterfactual),
    ]:
        # Train dataset
        train_data = {"input": [], "counterfactual_inputs": []}
        for _ in range(train_size):
            sample = generator()
            train_data["input"].append(sample["input"])
            train_data["counterfactual_inputs"].append(sample["counterfactual_inputs"])

        # Test dataset
        test_data = {"input": [], "counterfactual_inputs": []}
        for _ in range(test_size):
            sample = generator()
            test_data["input"].append(sample["input"])
            test_data["counterfactual_inputs"].append(sample["counterfactual_inputs"])

        # Create list[CounterfactualExample]
        datasets[f"{name}_train"] = [
            {"input": inp, "counterfactual_inputs": cf}
            for inp, cf in zip(train_data["input"], train_data["counterfactual_inputs"])
        ]
        datasets[f"{name}_test"] = [
            {"input": inp, "counterfactual_inputs": cf}
            for inp, cf in zip(test_data["input"], test_data["counterfactual_inputs"])
        ]

    return datasets
