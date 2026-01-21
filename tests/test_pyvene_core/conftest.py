# tests/test_experiments/conftest.py

import pytest
import torch
import random
from typing import Any
from unittest.mock import MagicMock

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var
from causalab.neural.pipeline import LMPipeline
from causalab.neural.LM_units import ResidualStream
from causalab.neural.token_position_builder import TokenPosition


@pytest.fixture(scope="session")
def mcqa_causal_model():
    """Create a simple MCQA causal model fixture."""
    NUM_CHOICES = 4
    ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

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
    ]

    COLORS = [item[0] for item in COLOR_OBJECTS]

    # Define values for each variable
    values: dict[str, Any] = {f"choice{x}": COLORS for x in range(NUM_CHOICES)}
    values.update({f"symbol{x}": ALPHABET for x in range(NUM_CHOICES)})
    values.update({"answer_pointer": list(range(NUM_CHOICES)), "answer": ALPHABET})
    values.update({"question": COLOR_OBJECTS})
    values.update({"raw_input": [""], "raw_output": [""]})

    # Define mechanisms using new Mechanism API
    mechanisms: dict[str, Mechanism] = {}

    # Input variables (no parents)
    mechanisms["raw_input"] = input_var([""])
    mechanisms["question"] = input_var(COLOR_OBJECTS)
    for i in range(NUM_CHOICES):
        mechanisms[f"symbol{i}"] = input_var(ALPHABET)
        mechanisms[f"choice{i}"] = input_var(COLORS)

    # answer_pointer depends on question and all choices
    choice_parents = [f"choice{i}" for i in range(NUM_CHOICES)]
    mechanisms["answer_pointer"] = Mechanism(
        parents=["question"] + choice_parents,
        compute=lambda t: next(
            (
                i
                for i, c in enumerate([t[f"choice{j}"] for j in range(4)])
                if c == t["question"][0]
            ),
            random.randint(0, 3),
        ),
    )

    # answer depends on answer_pointer and all symbols
    symbol_parents = [f"symbol{i}" for i in range(NUM_CHOICES)]
    mechanisms["answer"] = Mechanism(
        parents=["answer_pointer"] + symbol_parents,
        compute=lambda t: " "
        + [t[f"symbol{j}"] for j in range(4)][t["answer_pointer"]],
    )

    # raw_output depends on answer
    mechanisms["raw_output"] = Mechanism(
        parents=["answer"], compute=lambda t: t["answer"]
    )

    # Create and return the model
    return CausalModel(mechanisms, values, id="4_answer_MCQA_test")


@pytest.fixture(scope="session")
def mcqa_counterfactual_datasets(mcqa_causal_model):
    """Generate test counterfactual datasets for the MCQA task."""
    model = mcqa_causal_model
    NUM_CHOICES = 4

    def is_input_valid(x):
        question_color = x["question"][0]
        choice_colors = [x[f"choice{i}"] for i in range(NUM_CHOICES)]
        symbols = [x[f"symbol{i}"] for i in range(NUM_CHOICES)]
        return question_color in choice_colors and len(symbols) == len(set(symbols))

    def random_letter_counterfactual():
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy

        used_symbols = [input_setting[f"symbol{i}"] for i in range(NUM_CHOICES)]
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        available_symbols = [s for s in alphabet if s not in used_symbols]
        new_symbols = random.sample(available_symbols, NUM_CHOICES)

        for i in range(NUM_CHOICES):
            counterfactual[f"symbol{i}"] = new_symbols[i]

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    def random_position_counterfactual():
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy

        answer_position = model.new_trace(input_setting)["answer_pointer"]
        available_positions = [i for i in range(NUM_CHOICES) if i != answer_position]
        new_position = random.choice(available_positions)

        correct_color = counterfactual[f"choice{answer_position}"]
        counterfactual[f"choice{answer_position}"] = counterfactual[
            f"choice{new_position}"
        ]
        counterfactual[f"choice{new_position}"] = correct_color

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    # Generate small datasets for testing
    datasets = {}
    train_size, test_size = 5, 3

    for name, generator in [
        ("random_letter", random_letter_counterfactual),
        ("random_position", random_position_counterfactual),
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


class MockModel:
    """Mock model implementation for testing."""

    def __init__(self):
        self.config = type(
            "MockConfig",
            (object,),
            {
                "name_or_path": "mock_model",
                "num_hidden_layers": 4,
                "hidden_size": 32,
                "n_head": 4,
            },
        )
        self.device = "cpu"
        self.dtype = torch.float32

    def to(self, device=None, dtype=None):
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    def generate(self, **kwargs):
        # Create mock outputs
        batch_size = kwargs.get("input_ids", torch.ones(1, 1)).shape[0]
        max_new = kwargs.get("max_new_tokens", 3)

        # Create sequences
        sequences = torch.randint(2, 99, (batch_size, max_new))

        # Create scores if needed
        scores = None
        if kwargs.get("output_scores", False):
            scores = [torch.rand(batch_size, 100) for _ in range(max_new)]

        if kwargs.get("return_dict_in_generate", False):
            return type(
                "GenerationOutput",
                (object,),
                {"sequences": sequences, "scores": scores},
            )
        return sequences

    def prepare_inputs_for_generation(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
        return {"position_ids": pos}


class MockTokenizer:
    """Mock tokenizer implementation for testing."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1
        self.padding_side = "right"

    def __call__(self, texts, **kwargs):
        # Simple tokenization
        if isinstance(texts, str):
            texts = [texts]

        batch = []
        for text in texts:
            # Just use character codes
            tokens = [ord(c) % 100 + 2 for c in text]
            batch.append(tokens)

        # Pad if needed
        max_len = kwargs.get("max_length", max(len(seq) for seq in batch))
        batch = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in batch]

        # Create tensors
        input_ids = torch.tensor(batch, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(chr((t - 2) + 97) if t >= 2 else "_" for t in token_ids)

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(seq, skip_special_tokens) for seq in sequences]

    def convert_tokens_to_ids(self, token):
        return 0  # Always return pad token id for simplicity


@pytest.fixture
def mock_tiny_lm(monkeypatch):
    """Create a mock LMPipeline that doesn't try to load from HuggingFace."""
    # Create mock model and tokenizer
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    # Patch AutoModelForCausalLM.from_pretrained and AutoTokenizer.from_pretrained
    import transformers

    def mock_model_from_pretrained(*args, **kwargs):
        return mock_model

    def mock_tokenizer_from_pretrained(*args, **kwargs):
        return mock_tokenizer

    # Apply patches
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", mock_model_from_pretrained
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", mock_tokenizer_from_pretrained
    )

    # Create pipeline with mocked components
    pipeline = LMPipeline("mock_model", max_new_tokens=3)

    return pipeline


@pytest.fixture
def token_positions(mock_tiny_lm, mcqa_causal_model):
    """Create token position identifiers for the MCQA task."""

    # Define a function to get the last token
    def get_last_token(prompt):
        token_ids = mock_tiny_lm.load(prompt)["input_ids"][0]
        return [len(token_ids) - 1]

    # Create TokenPosition objects
    return [TokenPosition(get_last_token, mock_tiny_lm, id="last_token")]


@pytest.fixture
def model_units_list(mock_tiny_lm, token_positions):
    """Create model units list for testing."""
    units = []
    layers = [0, 2]

    for layer in layers:
        for token_position in token_positions:
            unit = ResidualStream(
                layer=layer,
                token_indices=token_position,
                shape=(mock_tiny_lm.model.config.hidden_size,),
                target_output=True,
            )
            units.append([[unit]])

    return units


@pytest.fixture
def mock_intervenable_model():
    """Create a mock intervenable model with the necessary methods."""

    class MockIntervenableModel:
        def __init__(self):
            self.model = MagicMock()
            self.interventions = {"test_intervention": MagicMock()}

        def set_device(self, device, set_model=False):
            pass

    return MockIntervenableModel()
