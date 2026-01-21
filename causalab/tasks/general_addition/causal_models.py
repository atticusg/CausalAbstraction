"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Causal models for general addition tasks.

This module defines two causal models:
1. Basic input-output model: Maps input digits directly to output digits
2. Intermediate structure model: Includes explicit sum and carry variables (2 addends only)
"""

import random
from typing import Dict, Any
from causalab.causal.causal_model import CausalModel
from .config import AdditionTaskConfig
from .templates import AdditionTemplateProcessor


def create_basic_addition_model(config: AdditionTaskConfig) -> CausalModel:
    """
    Create the basic addition causal model (input-output only).

    This model directly computes the output digits from input digits
    without explicit intermediate variables for sum and carry.

    Variables:
    - Input digits: digit_{k}_{d} for k=0..K-1, d=0..D-1 (each 0-9)
    - Control: num_addends (how many numbers), num_digits (digits per number), template
    - Output digits: output_digit_{d} for d=0..D (D+1 output digits)
    - Special: raw_input, raw_output

    Args:
        config: The task configuration

    Returns:
        A CausalModel instance for basic addition
    """
    K = config.max_numbers
    D = config.max_digits

    # Build variable list
    variables = []

    # Input digit variables
    for k in range(K):
        for d in range(D):
            variables.append(f"digit_{k}_{d}")

    # Control variables
    variables.extend(
        [
            "num_addends",  # How many numbers are being added (1 to K)
            "num_digits",  # How many digits per number (1 to D)
            "template",  # Which template string to use
        ]
    )

    # Output digit variables (D+1 digits for potential carry)
    for d in range(D + 1):
        variables.append(f"output_digit_{d}")

    # Special variables
    variables.extend(["raw_input", "raw_output"])

    # Build values dictionary
    values = {}

    # Input digits can be 0-9
    for k in range(K):
        for d in range(D):
            values[f"digit_{k}_{d}"] = list(range(10))

    # Control values
    values.update(
        {
            "num_addends": list(range(1, K + 1)),
            "num_digits": list(range(1, D + 1)),
            "template": config.templates,
        }
    )

    # Output digits can be 0-9
    for d in range(D + 1):
        values[f"output_digit_{d}"] = list(range(10))

    # Special variables are computed
    values.update(
        {
            "raw_input": None,
            "raw_output": None,
        }
    )

    # Build parents dictionary
    parents = {}

    # Input digits are independent
    for k in range(K):
        for d in range(D):
            parents[f"digit_{k}_{d}"] = []

    # Control variables are independent
    parents.update(
        {
            "num_addends": [],
            "num_digits": [],
            "template": [],
        }
    )

    # Output digits depend on all input digits and controls
    input_digit_vars = [f"digit_{k}_{d}" for k in range(K) for d in range(D)]
    for d in range(D + 1):
        parents[f"output_digit_{d}"] = input_digit_vars + ["num_addends", "num_digits"]

    # raw_input depends on input digits, controls, and template
    parents["raw_input"] = input_digit_vars + ["num_addends", "num_digits", "template"]

    # raw_output depends on output digits
    output_digit_vars = [f"output_digit_{d}" for d in range(D + 1)]
    parents["raw_output"] = output_digit_vars

    # Build mechanisms dictionary
    mechanisms = {}

    # Input digit sampling
    for k in range(K):
        for d in range(D):
            key = f"digit_{k}_{d}"
            mechanisms[key] = lambda t: random.randint(0, 9)

    # Control mechanisms
    mechanisms.update({
        "num_addends": lambda t: random.randint(1, K),
        "num_digits": lambda t: random.randint(1, D),
        "template": lambda t: random.choice(config.templates),
    })

    # Output computation mechanism
    def compute_output_digits(trace):
        """Compute all output digits from input digits."""
        num_addends = trace["num_addends"]
        num_digits = trace["num_digits"]

        # Extract numbers
        numbers = []
        for k in range(num_addends):
            digits = []
            for d in range(num_digits):
                digits.append(trace[f"digit_{k}_{d}"])
            numbers.append(digits)

        # Compute sum
        processor = AdditionTemplateProcessor(config)
        output_digits = processor.compute_sum(numbers)

        return output_digits

    # Create mechanism for each output digit
    for d in range(D + 1):

        def make_output_digit_mechanism(digit_idx):
            def mechanism(trace):
                output_digits = compute_output_digits(trace)
                if digit_idx < len(output_digits):
                    return output_digits[digit_idx]
                return 0

            return mechanism

        mechanisms[f"output_digit_{d}"] = make_output_digit_mechanism(d)

    # raw_input generation mechanism
    def generate_raw_input(trace):
        """Generate the complete prompt text."""
        num_addends = trace["num_addends"]
        num_digits = trace["num_digits"]
        template = trace["template"]

        # Build digit dictionary
        digit_dict = {}
        for k in range(K):
            for d in range(D):
                digit_dict[f"digit_{k}_{d}"] = trace[f"digit_{k}_{d}"]

        # Generate prompt
        processor = AdditionTemplateProcessor(config)
        return processor.generate_prompt(digit_dict, num_addends, num_digits, template)

    mechanisms["raw_input"] = generate_raw_input

    # raw_output generation mechanism
    def generate_raw_output(trace):
        """Generate the expected output text."""
        output_digits = [trace[f"output_digit_{d}"] for d in range(D + 1)]
        processor = AdditionTemplateProcessor(config)
        return " " + processor.format_output(output_digits)

    mechanisms["raw_output"] = generate_raw_output

    # Create the causal model
    model_id = f"addition_basic_{K}n_{D}d"
    return CausalModel(variables, values, parents, mechanisms, id=model_id)


def create_intermediate_addition_model(config: AdditionTaskConfig) -> CausalModel:
    """
    Create the intermediate structure addition causal model (2 addends only).

    This model includes explicit intermediate variables where index i refers to
    position i counting from the RIGHT starting at 1 (i=1 is ones, i=2 is tens, etc.):
    - C_1: 1 if ones digits of X and Y sum to >= 10, else 0
    - C_i (i>1): 1 if (digit_i of X + digit_i of Y + C_{i-1}) >= 10, else 0
    - O_1: (ones digit of X + ones digit of Y) % 10
    - O_i (i>1): (digit_i of X + digit_i of Y + C_{i-1}) % 10

    Note: digit_{k}_{d} uses indexing from LEFT (d=0 is most significant),
    but C_i and O_i use indexing from RIGHT starting at 1 (i=1 is ones place).

    This is only valid for 2-addend addition.

    Args:
        config: The task configuration (must have max_numbers >= 2)

    Returns:
        A CausalModel instance with intermediate structure
    """
    if config.max_numbers < 2:
        raise ValueError("Intermediate model requires at least 2 addends")

    K = 2  # Fixed to 2 addends for intermediate model
    D = config.max_digits

    # Build variable list
    variables = []

    # Input digit variables (only for 2 numbers)
    # digit_{k}_{d} where d=0 is leftmost (most significant)
    for k in range(K):
        for d in range(D):
            variables.append(f"digit_{k}_{d}")

    # Control variables
    variables.extend(
        [
            "num_digits",  # How many digits per number
            "template",  # Which template string to use
        ]
    )

    # Intermediate carry variables: C_1, C_2, ..., C_D
    # where C_i is the carry from position i (1-indexed from right)
    for i in range(1, D + 1):
        variables.append(f"C_{i}")

    # Output digit variables: O_1, O_2, ..., O_D, O_{D+1}
    # where O_i is the output at position i (1-indexed from right)
    for i in range(1, D + 2):
        variables.append(f"O_{i}")

    # Special variables
    variables.extend(["raw_input", "raw_output"])

    # Build values dictionary
    values = {}

    # Input digits can be 0-9
    for k in range(K):
        for d in range(D):
            values[f"digit_{k}_{d}"] = list(range(10))

    # Control values
    values.update(
        {
            "num_digits": list(range(1, D + 1)),
            "template": config.templates,
        }
    )

    # Intermediate carry values (binary)
    for i in range(1, D + 1):
        values[f"C_{i}"] = [0, 1]

    # Output digits can be 0-9
    for i in range(1, D + 2):
        values[f"O_{i}"] = list(range(10))

    # Special variables are computed
    values.update(
        {
            "raw_input": None,
            "raw_output": None,
        }
    )

    # Build parents dictionary
    parents = {}

    # Input digits are independent
    for k in range(K):
        for d in range(D):
            parents[f"digit_{k}_{d}"] = []

    # Control variables are independent
    parents.update(
        {
            "num_digits": [],
            "template": [],
        }
    )

    # Carry variables: C_i depends on digits at position i and previous carry
    # Position i (1-indexed from right) maps to digit index d = D - i (from left)
    for i in range(1, D + 1):
        d = D - i  # Convert right-indexed position to left-indexed digit
        if i == 1:
            # C_1 depends only on the ones digits
            parents[f"C_{i}"] = [f"digit_0_{d}", f"digit_1_{d}"]
        else:
            # C_i depends on digits at position i and previous carry C_{i-1}
            parents[f"C_{i}"] = [f"digit_0_{d}", f"digit_1_{d}", f"C_{i - 1}"]

    # Output variables: O_i depends on digits at position i and previous carry
    for i in range(1, D + 2):
        if i == 1:
            # O_1 depends only on the ones digits
            d = D - i  # d = D - 1 (ones place in left-indexing)
            parents[f"O_{i}"] = [f"digit_0_{d}", f"digit_1_{d}"]
        elif i <= D:
            # O_i for i=2 to D depends on digits and previous carry
            d = D - i
            parents[f"O_{i}"] = [f"digit_0_{d}", f"digit_1_{d}", f"C_{i - 1}"]
        else:
            # O_{D+1} is just the final carry C_D
            parents[f"O_{i}"] = [f"C_{D}"]

    # raw_input depends on input digits, num_digits, and template
    input_digit_vars = [f"digit_{k}_{d}" for k in range(K) for d in range(D)]
    parents["raw_input"] = input_digit_vars + ["num_digits", "template"]

    # raw_output depends on output digits O_1, O_2, ..., O_{D+1}
    output_vars = [f"O_{i}" for i in range(1, D + 2)]
    parents["raw_output"] = output_vars

    # Build mechanisms dictionary
    mechanisms = {}

    # Input digit sampling
    for k in range(K):
        for d in range(D):
            key = f"digit_{k}_{d}"
            mechanisms[key] = lambda t: random.randint(0, 9)

    # Control mechanisms
    mechanisms.update({
        "num_digits": lambda t: random.randint(1, D),
        "template": lambda t: random.choice(config.templates),
    })

    # Carry mechanisms
    for i in range(1, D + 1):
        d = D - i  # Convert right-indexed position to left-indexed digit
        if i == 1:
            # C_1 = 1 if (digit_0 + digit_1) >= 10, else 0
            def mechanism(trace, d=d):
                digit_0 = trace[f"digit_0_{d}"]
                digit_1 = trace[f"digit_1_{d}"]
                return 1 if (digit_0 + digit_1) >= 10 else 0

            mechanisms[f"C_{i}"] = mechanism
        else:
            # C_i = 1 if (digit_0 + digit_1 + C_{i-1}) >= 10, else 0
            def make_carry_mechanism(pos, d=d):
                def mechanism(trace):
                    digit_0 = trace[f"digit_0_{d}"]
                    digit_1 = trace[f"digit_1_{d}"]
                    carry_prev = trace[f"C_{pos-1}"]
                    return 1 if (digit_0 + digit_1 + carry_prev) >= 10 else 0

                return mechanism
            mechanisms[f"C_{i}"] = make_carry_mechanism(i, d)

    # Output mechanisms
    for i in range(1, D + 2):
        if i == 1:
            # O_1 = (digit_0 + digit_1) % 10
            d = D - i
            def mechanism(trace, d=d):
                digit_0 = trace[f"digit_0_{d}"]
                digit_1 = trace[f"digit_1_{d}"]
                return (digit_0 + digit_1) % 10

            mechanisms[f"O_{i}"] = mechanism
        elif i <= D:
            # O_i = (digit_0 + digit_1 + C_{i-1}) % 10
            d = D - i
            def make_output_mechanism(pos, d=d):
                def mechanism(trace):
                    digit_0 = trace[f"digit_0_{d}"]
                    digit_1 = trace[f"digit_1_{d}"]
                    carry_prev = trace[f"C_{pos-1}"]
                    return (digit_0 + digit_1 + carry_prev) % 10

                return mechanism
            mechanisms[f"O_{i}"] = make_output_mechanism(i, d)
        else:
            # O_{D+1} = C_D (final carry out)
            def mechanism(trace):
                return trace[f"C_{D}"]
            mechanisms[f"O_{i}"] = mechanism

    # raw_input generation mechanism
    def generate_raw_input(trace):
        """Generate the complete prompt text."""
        num_digits = trace["num_digits"]
        template = trace["template"]

        # Build digit dictionary
        digit_dict = {}
        for k in range(K):
            for d in range(D):
                digit_dict[f"digit_{k}_{d}"] = trace[f"digit_{k}_{d}"]

        # Generate prompt (always 2 addends for this model)
        processor = AdditionTemplateProcessor(config)
        return processor.generate_prompt(digit_dict, 2, num_digits, template)

    mechanisms["raw_input"] = generate_raw_input

    # raw_output generation mechanism
    def generate_raw_output(trace):
        """Generate the expected output text.

        Args are O_1, O_2, ..., O_{D+1} (1-indexed from right)
        where 1 is ones place, so we need to reverse for formatting.
        """
        output_digits_right_to_left = [trace[f"O_{i}"] for i in range(1, D + 2)]
        # Reverse to get left-to-right (most significant first)
        output_digits_left_to_right = list(reversed(output_digits_right_to_left))
        processor = AdditionTemplateProcessor(config)
        return " " + processor.format_output(output_digits_left_to_right)

    mechanisms["raw_output"] = generate_raw_output

    # Create the causal model
    model_id = f"addition_intermediate_2n_{D}d"
    return CausalModel(variables, values, parents, mechanisms, id=model_id)


def sample_valid_addition_input(
    config: AdditionTaskConfig, num_addends: int, num_digits: int
) -> Dict[str, Any]:
    """
    Sample a valid input for addition causal models.

    Args:
        config: Task configuration
        num_addends: Number of numbers to add (must be <= config.max_numbers)
        num_digits: Number of digits per number (must be <= config.max_digits)

    Returns:
        Dictionary with sampled input values
    """
    if num_addends > config.max_numbers:
        raise ValueError(
            f"num_addends {num_addends} exceeds max_numbers {config.max_numbers}"
        )
    if num_digits > config.max_digits:
        raise ValueError(
            f"num_digits {num_digits} exceeds max_digits {config.max_digits}"
        )

    input_sample = {
        "num_addends": num_addends,
        "num_digits": num_digits,
        "template": random.choice(config.templates),
    }

    # Sample digits for active addends
    # Ensure the most significant digit is non-zero so we get exactly num_digits digits
    for k in range(config.max_numbers):
        for d in range(config.max_digits):
            if k < num_addends and d < num_digits:
                if d == 0:
                    # Most significant digit: sample 1-9 (no leading zeros)
                    # This ensures 2-digit numbers are always 10-99, 3-digit are 100-999, etc.
                    input_sample[f"digit_{k}_{d}"] = random.randint(1, 9)
                else:
                    # Other digits: sample 0-9
                    input_sample[f"digit_{k}_{d}"] = random.randint(0, 9)
            else:
                # Inactive digit: set to 0
                input_sample[f"digit_{k}_{d}"] = 0

    return input_sample


def sample_sum_to_nine_input(
    config: AdditionTaskConfig, digit_position: int
) -> Dict[str, Any]:
    """
    Sample an input where the specified digit position sums to 9 across all addends.

    This is used for the specialized sampler mentioned in the seed document.

    Args:
        config: Task configuration
        digit_position: Which digit position should sum to 9

    Returns:
        Dictionary with sampled input values where digits at digit_position sum to 9
    """
    # Use 2 addends for simplicity
    num_addends = 2
    num_digits = config.max_digits

    if digit_position >= num_digits:
        raise ValueError(
            f"digit_position {digit_position} exceeds num_digits {num_digits}"
        )

    input_sample = {
        "num_addends": num_addends,
        "num_digits": num_digits,
        "template": random.choice(config.templates),
    }

    # Sample digits
    for k in range(config.max_numbers):
        for d in range(config.max_digits):
            if k < num_addends and d < num_digits:
                if d == digit_position and k == 0:
                    # For the target position, sample first digit
                    if d == 0:
                        # Most significant digit: 1-9 (no leading zeros)
                        digit_0 = random.randint(1, 9)
                    else:
                        digit_0 = random.randint(0, 9)
                    input_sample[f"digit_{k}_{d}"] = digit_0
                elif d == digit_position and k == 1:
                    # Second digit at target position: make sum = 9
                    digit_0 = input_sample[f"digit_0_{d}"]
                    target_value = 9 - digit_0
                    # If this is most significant digit, ensure it's non-zero
                    if d == 0 and target_value == 0:
                        # Adjust: pick different digit_0 that allows non-zero result
                        digit_0 = random.randint(1, 8)  # 1-8 so that 9-digit_0 is 1-8
                        input_sample[f"digit_0_{d}"] = digit_0
                        target_value = 9 - digit_0
                    input_sample[f"digit_{k}_{d}"] = target_value
                else:
                    # Other positions: ensure most significant digit is non-zero
                    if d == 0:
                        input_sample[f"digit_{k}_{d}"] = random.randint(1, 9)
                    else:
                        input_sample[f"digit_{k}_{d}"] = random.randint(0, 9)
            else:
                input_sample[f"digit_{k}_{d}"] = 0

    return input_sample
