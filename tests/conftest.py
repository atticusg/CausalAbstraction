# tests/conftest.py

import pytest
import torch
import random
import numpy as np

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var
from causalab.neural.pipeline import LMPipeline
from causalab.neural.LM_units import ResidualStream
from causalab.neural.token_positions import TokenPosition

# Re-export so tests can do "from tests.conftest import assert_runner_completed".
from tests.end_to_end._helpers.runner_completion import (  # noqa: F401
    assert_runner_completed,
)

# Tier markers recognised by the warn-mode collection hook below. Mirrors the
# tier taxonomy in docs/TESTS.md. Tests must declare exactly one of
# {smoke, numerical_unit, golden, property, unit} explicitly. `golden` is the
# sole GPU tier (value-pinned coherent-model runners); the rest are CPU. The
# default tier (`unit`) is the most common; tag with
# `pytestmark = pytest.mark.unit` at module scope.
_TIER_MARKERS = frozenset({"smoke", "numerical_unit", "golden", "property", "unit"})


# --- ``model: tiny-random`` smoke-only guardrail ----------------------------
#
# ``causalab/configs/model/tiny-random.yaml`` carries ``is_smoke_only: true``
# and a fat-finger warning header. The hook below is the pytest-side runtime
# half of that guardrail: it wraps :func:`causalab.io.configs.load_runner_config`
# in :func:`pytest_configure` (i.e. *before* any test module imports the
# function by name) so that any test which composes a cfg with
# ``model.is_smoke_only`` set must also carry the ``smoke`` pytest marker.
#
# (``chat-coherent`` = Qwen3-4B-Instruct is the *trained* stub used by the
# golden tier — it is NOT flagged ``is_smoke_only`` because its outputs are
# meaningful, not random noise.)
#
# Why ``pytest_configure`` and not an autouse fixture? Test modules
# typically do ``from causalab.io.configs import load_runner_config``,
# which binds the function-name *at import time*. An autouse fixture that
# rebinds the module attribute after import is a no-op for those callers.
# Patching in ``pytest_configure`` runs before any test-module collection /
# import, so subsequent ``from`` imports pick up the wrapped function.
#
# This is pytest-only — it does not protect against fat-fingering
# ``model=tiny-random`` in a real Hydra CLI invocation. The yaml-header
# banner plus the random-init weights producing visibly nonsense numbers
# handle that path.

_CURRENT_TEST_IS_SMOKE: list[bool] = [False]


def pytest_configure(config):
    """Install the ``is_smoke_only`` guardrail before any test imports."""
    from causalab.io import configs as _configs_module

    if getattr(_configs_module, "_load_runner_config_guardrail_installed", False):
        return  # idempotent; already patched (e.g. nested pytest invocation)

    real_loader = _configs_module.load_runner_config

    def _guarded_loader(*args, **kwargs):
        cfg = real_loader(*args, **kwargs)
        model = getattr(cfg, "model", None)
        if model is not None and bool(getattr(model, "is_smoke_only", False)):
            if not _CURRENT_TEST_IS_SMOKE[0]:
                raise AssertionError(
                    f"Composed a model config flagged ``is_smoke_only: true`` "
                    f"(model.id={getattr(model, 'id', '?')!r}) outside the "
                    f"smoke tier. ``tiny-random`` weights are random-init and "
                    f"produce nonsense numbers — pick a real model config "
                    f"(e.g. ``chat-coherent``) or tag "
                    f"this test with @pytest.mark.smoke."
                )
        return cfg

    _configs_module.load_runner_config = _guarded_loader
    _configs_module._load_runner_config_guardrail_installed = True


def pytest_runtest_setup(item):
    """Track whether the currently-running test is in the smoke tier."""
    _CURRENT_TEST_IS_SMOKE[0] = "smoke" in {m.name for m in item.iter_markers()}


def pytest_runtest_teardown(item, nextitem):
    """Reset the smoke flag so off-test composition (e.g. fixture teardown
    that happens to call ``load_runner_config``) defaults to non-smoke."""
    del item, nextitem
    _CURRENT_TEST_IS_SMOKE[0] = False


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


@pytest.fixture(scope="function")
def mock_tiny_lm():
    """Return an :class:`LMPipeline` backed by the tiny-random Llama stub.

    Tests on this fixture want a placeholder pipeline that exercises real
    HF shapes / dtypes / tokenizer behaviour without the cost of loading a
    production-sized model. The underlying model is cached in-process via
    :func:`tests._helpers.tiny.tiny_random_model`, so repeated fixture
    construction is effectively free after the first pytest session warm-up.
    """
    from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME

    return LMPipeline(model_or_name=TINY_RANDOM_MODEL_NAME, max_new_tokens=3)


@pytest.fixture(scope="function")
def token_positions(mock_tiny_lm, mcqa_causal_model):
    """Create token position identifiers for the MCQA task."""

    # Define a function to get the last token
    def get_last_token(prompt):
        token_ids = mock_tiny_lm.load(prompt)["input_ids"][0]
        return [len(token_ids) - 1]

    # Define a function to get the position of the answer symbol
    def get_answer_symbol_position(prompt):
        # Parse the prompt to get the causal input
        causal_input = mcqa_causal_model.input_loader(prompt)

        # Get the answer pointer
        output = mcqa_causal_model.new_trace(causal_input)
        answer_position = output["answer_pointer"]

        # Get the symbol at that position
        answer_symbol = causal_input[f"symbol{answer_position}"]

        # Find the token position of this symbol
        mock_tiny_lm.load(prompt)["input_ids"][0]
        text = prompt.split("\n")

        for i, line in enumerate(text[1:]):  # Skip the question line
            if line.startswith(answer_symbol):
                # Found the line with the answer
                # Count tokens up to this point
                substring = "\n".join(text[: i + 1]) + "\n" + answer_symbol
                position_tokens = mock_tiny_lm.load(substring)["input_ids"][0]
                return [
                    len(position_tokens) - 1
                ]  # Return the last token position (the symbol)

        # Fallback to last token if the symbol isn't found
        return get_last_token(prompt)

    # Create TokenPosition objects
    return [
        TokenPosition(get_last_token, mock_tiny_lm, id="last_token"),
        TokenPosition(get_answer_symbol_position, mock_tiny_lm, id="answer_symbol"),
    ]


@pytest.fixture(scope="function")
def model_units_list(mock_tiny_lm, token_positions):
    """Create model units list for testing."""
    # Create a list of ResidualStream units for different layers and positions
    units = []

    # Use a subset of layers for testing
    layers = [0, 2]

    for layer in layers:
        for token_position in token_positions:
            # Create a ResidualStream unit
            unit = ResidualStream(
                layer=layer,
                token_indices=token_position,
                shape=(mock_tiny_lm.model.config.hidden_size,),
                target_output=True,  # Use block output
            )
            units.append([unit])  # Each unit is wrapped in its own list

    return units


@pytest.fixture(scope="function")
def mock_intervenable_model(mock_tiny_lm, model_units_list):
    """Mock intervenable model for testing."""

    class MockIntervenableModel:
        def __init__(self):
            self.model = mock_tiny_lm.model
            self.interventions = {}

            # Create mock interventions
            for i, units in enumerate(model_units_list):
                key = f"intervention_{i}"
                self.interventions[key] = type(
                    "MockIntervention",
                    (object,),
                    {
                        "forward": lambda x, y, **kwargs: torch.randn_like(x),
                        "parameters": lambda: [torch.nn.Parameter(torch.randn(10))],
                        "state_dict": lambda: {"weight": torch.randn(10)},
                        "load_state_dict": lambda x: None,
                        "get_sparsity_loss": lambda: torch.tensor(0.1),
                        "set_temperature": lambda x: None,
                        "mask": torch.nn.Parameter(torch.randn(10)),
                    },
                )

        def disable_model_gradients(self):
            pass

        def eval(self):
            pass

        def set_device(self, device, set_model=True):
            pass

        def set_zero_grad(self):
            pass

        def count_parameters(self):
            return 100

        def __call__(self, inputs, unit_locations=None, **kwargs):
            # Mock forward pass
            batch_size = inputs["input_ids"].shape[0]
            hidden_size = self.model.config.hidden_size

            # Create mock activations
            activations = [torch.randn(1, hidden_size) for _ in range(10)]

            # Create mock outputs
            outputs = type(
                "obj",
                (object,),
                {
                    "hidden_states": torch.randn(batch_size, 10, hidden_size),
                    "logits": torch.randn(batch_size, 10, 100),
                },
            )

            return [(outputs, activations), None]

        def generate(
            self, inputs, sources=None, unit_locations=None, subspaces=None, **kwargs
        ):
            # Mock generation with interventions
            batch_size = inputs["input_ids"].shape[0]
            seq_len = 5  # Fixed output sequence length

            if kwargs.get("output_scores", False):
                # Return mock scores
                scores = [torch.randn(batch_size, 100) for _ in range(seq_len)]
                return [
                    type(
                        "GenerationOutput",
                        (object,),
                        {"sequences": None, "scores": scores},
                    )
                ]

            # Return mock sequences
            sequences = torch.randint(2, 99, (batch_size, seq_len))
            return [
                type(
                    "GenerationOutput",
                    (object,),
                    {"sequences": sequences, "scores": None},
                )
            ]

    return MockIntervenableModel()
