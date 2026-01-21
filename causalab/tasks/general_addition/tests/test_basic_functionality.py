"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Basic functionality tests for general addition task.

This script tests the core functionality without using pytest,
just running checks and printing results.
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

from causalab.tasks.general_addition.config import (
    create_two_number_two_digit_config,
    create_general_config,
)
from causalab.tasks.general_addition.templates import AdditionTemplateProcessor
from causalab.tasks.general_addition.causal_models import (
    create_basic_addition_model,
    create_intermediate_addition_model,
    sample_valid_addition_input,
    sample_sum_to_nine_input,
)
from causalab.tasks.general_addition.counterfactuals import (
    random_counterfactual,
    sum_to_nine_counterfactual,
)


def test_config():
    """Test configuration creation."""
    print("\n" + "=" * 70)
    print("TEST: Configuration Creation")
    print("=" * 70)

    config1 = create_two_number_two_digit_config()
    print(
        f"✓ Created 2-number, 2-digit config: {config1.max_numbers}n, {config1.max_digits}d"
    )
    assert config1.max_numbers == 2
    assert config1.max_digits == 2

    config2 = create_general_config(3, 4)
    print(
        f"✓ Created 3-number, 4-digit config: {config2.max_numbers}n, {config2.max_digits}d"
    )
    assert config2.max_numbers == 3
    assert config2.max_digits == 4

    print("✓ All config tests passed")


def test_templates():
    """Test template processing."""
    print("\n" + "=" * 70)
    print("TEST: Template Processing")
    print("=" * 70)

    config = create_two_number_two_digit_config()
    processor = AdditionTemplateProcessor(config)

    # Test digit-to-number conversion
    result = processor.digits_to_number([1, 2, 3])
    print(f"✓ digits_to_number([1,2,3]) = {result}")
    assert result == 123

    # Test number-to-digits conversion
    digits = processor.number_to_digits(23, 3)
    print(f"✓ number_to_digits(23, 3) = {digits}")
    assert digits == [0, 2, 3]

    # Test format_number
    formatted = processor.format_number([0, 2, 3])
    print(f"✓ format_number([0,2,3]) = '{formatted}'")
    assert formatted == "23"

    # Test fill_template
    filled = processor.fill_template(
        "The sum of {num0} and {num1} is", [[2, 3], [4, 5]]
    )
    print(f"✓ fill_template: '{filled}'")
    assert filled == "The sum of 23 and 45 is"

    # Test compute_sum
    sum_digits = processor.compute_sum([[2, 3], [4, 5]])
    print(f"✓ compute_sum([[2,3], [4,5]]) = {sum_digits}")
    assert processor.digits_to_number(sum_digits) == 68

    print("✓ All template tests passed")


def test_basic_causal_model():
    """Test basic addition causal model."""
    print("\n" + "=" * 70)
    print("TEST: Basic Causal Model")
    print("=" * 70)

    config = create_two_number_two_digit_config()
    model = create_basic_addition_model(config)
    print(f"✓ Created model: {model.id}")

    # Sample input
    input_sample = sample_valid_addition_input(config, 2, 2)
    print(
        f"✓ Sampled input with {input_sample['num_addends']} addends, {input_sample['num_digits']} digits"
    )

    # Run model forward to get all outputs
    output = model.new_trace(input_sample)
    prompt = output['raw_input']
    answer = output['raw_output']
    print(f"✓ Generated prompt: '{prompt}'")
    print(f"✓ Generated answer: '{answer}'")

    # Verify the math
    num1_digits = [input_sample["digit_0_0"], input_sample["digit_0_1"]]
    num2_digits = [input_sample["digit_1_0"], input_sample["digit_1_1"]]
    processor = AdditionTemplateProcessor(config)
    num1 = processor.digits_to_number(num1_digits)
    num2 = processor.digits_to_number(num2_digits)
    expected_sum = num1 + num2
    actual_sum = int(answer.strip())

    print(f"  {num1} + {num2} = {expected_sum}")
    print(f"  Model output: {actual_sum}")
    assert actual_sum == expected_sum, f"Expected {expected_sum}, got {actual_sum}"
    print("✓ Math is correct!")

    print("✓ All basic model tests passed")


def test_intermediate_causal_model():
    """Test intermediate addition causal model with carry variables."""
    print("\n" + "=" * 70)
    print("TEST: Intermediate Causal Model (with carry)")
    print("=" * 70)

    config = create_two_number_two_digit_config()
    model = create_intermediate_addition_model(config)
    print(f"✓ Created intermediate model: {model.id}")

    # Test case 1: No carry (23 + 45 = 68)
    # digit_0_0=2 (tens), digit_0_1=3 (ones)
    # digit_1_0=4 (tens), digit_1_1=5 (ones)
    input_sample = {
        "digit_0_0": 2,
        "digit_0_1": 3,
        "digit_1_0": 4,
        "digit_1_1": 5,
        "num_digits": 2,
        "template": config.templates[0],
    }

    output = model.new_trace(input_sample)
    print(f"\nTest 1: 23 + 45 = 68")
    print(f"  C_1 = {output['C_1']} (carry from ones: 3+5=8 < 10, no carry)")
    print(f"  O_1 = {output['O_1']} (ones place: (3+5)%10 = 8)")
    print(f"  C_2 = {output['C_2']} (carry from tens: 2+4+0=6 < 10, no carry)")
    print(f"  O_2 = {output['O_2']} (tens place: (2+4+0)%10 = 6)")
    print(f"  O_3 = {output['O_3']} (hundreds place: final carry = 0)")
    print(f"  raw_output = '{output['raw_output']}'")

    assert output["C_1"] == 0, "no carry from ones"
    assert output["O_1"] == 8, "ones output = 8"
    assert output["C_2"] == 0, "no carry from tens"
    assert output["O_2"] == 6, "tens output = 6"
    assert output["O_3"] == 0, "no hundreds digit"
    assert output["raw_output"].strip() == "68", "final answer is 68"
    print("✓ Test 1 passed (no carry)")

    # Test case 2: With carry (27 + 48 = 75)
    # digit_0_0=2 (tens), digit_0_1=7 (ones)
    # digit_1_0=4 (tens), digit_1_1=8 (ones)
    input_sample = {
        "digit_0_0": 2,
        "digit_0_1": 7,
        "digit_1_0": 4,
        "digit_1_1": 8,
        "num_digits": 2,
        "template": config.templates[0],
    }

    output = model.new_trace(input_sample)
    print(f"\nTest 2: 27 + 48 = 75")
    print(f"  C_1 = {output['C_1']} (carry from ones: 7+8=15 >= 10, carry!)")
    print(f"  O_1 = {output['O_1']} (ones place: (7+8)%10 = 15%10 = 5)")
    print(f"  C_2 = {output['C_2']} (carry from tens: 2+4+1=7 < 10, no carry)")
    print(f"  O_2 = {output['O_2']} (tens place: (2+4+1)%10 = 7)")
    print(f"  O_3 = {output['O_3']} (hundreds place: final carry = 0)")
    print(f"  raw_output = '{output['raw_output']}'")

    assert output["C_1"] == 1, "carry from ones"
    assert output["O_1"] == 5, "ones output = 15%10 = 5"
    assert output["C_2"] == 0, "no carry from tens"
    assert output["O_2"] == 7, "tens output = (2+4+1)%10 = 7"
    assert output["O_3"] == 0, "no hundreds digit"
    assert output["raw_output"].strip() == "75", "final answer is 75"
    print("✓ Test 2 passed (with carry)")

    print("✓ All intermediate model tests passed")


def test_counterfactuals():
    """Test counterfactual generation."""
    print("\n" + "=" * 70)
    print("TEST: Counterfactual Generation")
    print("=" * 70)

    config = create_two_number_two_digit_config()

    # Test random counterfactual
    cf_data = random_counterfactual(config, 2, 2)
    print("✓ Random counterfactual:")
    print(f"  Input:  {cf_data['input']['raw_input']}")
    print(f"  Counter: {cf_data['counterfactual_inputs'][0]['raw_input']}")
    assert "input" in cf_data and "counterfactual_inputs" in cf_data

    # Test sum to nine (tens digit position)
    cf_data = sum_to_nine_counterfactual(config, digit_position=1)
    print("\n✓ Sum to nine counterfactual (digit position 1):")
    print(f"  Input:  {cf_data['input']['raw_input']}")
    print(f"  Counter: {cf_data['counterfactual_inputs'][0]['raw_input']}")

    # Verify that the specified digit position sums to 9
    # For 2-digit numbers, digit_position=1 is the tens digit (position 0 would be ones)
    input_digit_0 = cf_data["input"]["digit_0_1"]  # tens digit of first number
    input_digit_1 = cf_data["input"]["digit_1_1"]  # tens digit of second number
    digit_sum = input_digit_0 + input_digit_1
    print(f"  Tens digits: {input_digit_0} + {input_digit_1} = {digit_sum}")
    assert digit_sum == 9, f"Digits at position 1 should sum to 9, but got {digit_sum}"

    print("✓ All counterfactual tests passed")


def test_sum_to_nine_sampler():
    """Test the specialized sum-to-nine sampler."""
    print("\n" + "=" * 70)
    print("TEST: Sum-to-Nine Sampler")
    print("=" * 70)

    config = create_two_number_two_digit_config()

    # Test for tens position (position 0 in 2-digit numbers)
    for i in range(5):
        input_sample = sample_sum_to_nine_input(config, 0)
        digit_0 = input_sample["digit_0_0"]
        digit_1 = input_sample["digit_1_0"]
        total = digit_0 + digit_1
        print(
            f"  Sample {i + 1}: digit_0_0={digit_0}, digit_1_0={digit_1}, sum={total}"
        )
        assert total == 9, f"Expected sum of 9, got {total}"

    print("✓ All samples correctly sum to 9")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GENERAL ADDITION TASK - FUNCTIONALITY TESTS")
    print("=" * 70)

    try:
        test_config()
        test_templates()
        test_basic_causal_model()
        test_intermediate_causal_model()
        test_counterfactuals()
        test_sum_to_nine_sampler()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
