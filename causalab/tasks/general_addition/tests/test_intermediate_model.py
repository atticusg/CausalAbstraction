"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Tests for the intermediate addition causal model.

Tests verify:
1. The model performs addition correctly
2. Carry variables have expected values
3. Output variables have expected values
4. Interventions on variables have expected effects
"""

import pytest
import sys

sys.path.append("../..")

from causalab.tasks.general_addition.config import create_general_config
from causalab.tasks.general_addition.causal_models import create_intermediate_addition_model


class TestIntermediateModelCorrectness:
    """Test that the intermediate model performs addition correctly."""

    def test_simple_no_carry(self):
        """Test 23 + 45 = 68 (no carries)."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 2,
            "digit_0_1": 3,  # 23
            "digit_1_0": 4,
            "digit_1_1": 5,  # 45
            "num_digits": 2,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)

        # Check raw output
        assert output["raw_output"].strip() == "68"

        # Check carries (should all be 0)
        assert output["C_1"] == 0  # 3 + 5 = 8 < 10
        assert output["C_2"] == 0  # 2 + 4 = 6 < 10

        # Check output digits (1-indexed from right)
        assert output["O_1"] == 8  # ones place: (3 + 5) % 10
        assert output["O_2"] == 6  # tens place: (2 + 4) % 10
        assert output["O_3"] == 0  # hundreds: carry out

    def test_carry_from_ones(self):
        """Test 27 + 45 = 72 (carry from ones place)."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 2,
            "digit_0_1": 7,  # 27
            "digit_1_0": 4,
            "digit_1_1": 5,  # 45
            "num_digits": 2,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)

        # Check raw output
        assert output["raw_output"].strip() == "72"

        # Check carries
        assert output["C_1"] == 1  # 7 + 5 = 12 >= 10
        assert output["C_2"] == 0  # 2 + 4 + 1 = 7 < 10

        # Check output digits
        assert output["O_1"] == 2  # (7 + 5) % 10 = 12 % 10
        assert output["O_2"] == 7  # (2 + 4 + 1) % 10
        assert output["O_3"] == 0

    def test_multiple_carries(self):
        """Test 58 + 67 = 125 (carry propagates)."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 5,
            "digit_0_1": 8,  # 58
            "digit_1_0": 6,
            "digit_1_1": 7,  # 67
            "num_digits": 2,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)

        # Check raw output
        assert output["raw_output"].strip() == "125"

        # Check carries
        assert output["C_1"] == 1  # 8 + 7 = 15 >= 10
        assert output["C_2"] == 1  # 5 + 6 + 1 = 12 >= 10

        # Check output digits
        assert output["O_1"] == 5  # (8 + 7) % 10 = 15 % 10
        assert output["O_2"] == 2  # (5 + 6 + 1) % 10 = 12 % 10
        assert output["O_3"] == 1  # carry out from C_2

    def test_three_digit_addition(self):
        """Test 123 + 456 = 579 (3 digits, no carries)."""
        config = create_general_config(2, 3)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 1,
            "digit_0_1": 2,
            "digit_0_2": 3,  # 123
            "digit_1_0": 4,
            "digit_1_1": 5,
            "digit_1_2": 6,  # 456
            "num_digits": 3,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)

        # Check raw output
        assert output["raw_output"].strip() == "579"

        # Check carries (no carries)
        assert output["C_1"] == 0  # 3 + 6 = 9 < 10
        assert output["C_2"] == 0  # 2 + 5 = 7 < 10
        assert output["C_3"] == 0  # 1 + 4 = 5 < 10

        # Check output digits
        assert output["O_1"] == 9  # ones
        assert output["O_2"] == 7  # tens
        assert output["O_3"] == 5  # hundreds
        assert output["O_4"] == 0  # thousands (carry out)

    def test_three_digit_with_carries(self):
        """Test 789 + 876 = 1665 (3 digits with carries)."""
        config = create_general_config(2, 3)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 7,
            "digit_0_1": 8,
            "digit_0_2": 9,  # 789
            "digit_1_0": 8,
            "digit_1_1": 7,
            "digit_1_2": 6,  # 876
            "num_digits": 3,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)

        # Check raw output
        assert output["raw_output"].strip() == "1665"

        # Check carries
        assert output["C_1"] == 1  # 9 + 6 = 15 >= 10
        assert output["C_2"] == 1  # 8 + 7 + 1 = 16 >= 10
        assert output["C_3"] == 1  # 7 + 8 + 1 = 16 >= 10

        # Check output digits
        assert output["O_1"] == 5  # (9 + 6) % 10
        assert output["O_2"] == 6  # (8 + 7 + 1) % 10
        assert output["O_3"] == 6  # (7 + 8 + 1) % 10
        assert output["O_4"] == 1  # carry out


class TestIntermediateModelInterventions:
    """Test that interventions on intermediate variables have expected effects."""

    def test_intervene_on_C_1(self):
        """Test intervening on C_1 (ones place carry)."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # Original: 23 + 45 = 68 (no carry from ones)
        input_sample = {
            "digit_0_0": 2,
            "digit_0_1": 3,
            "digit_1_0": 4,
            "digit_1_1": 5,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output_original = model.new_trace(input_sample)
        assert output_original['C_1'] == 0
        assert output_original['O_2'] == 6  # 2 + 4 + 0 = 6

        # Intervene: force C_1 = 1 (as if ones place generated a carry)
        output_intervened = model.new_trace({**input_sample, 'C_1': 1})
        assert output_intervened['C_1'] == 1
        assert output_intervened['O_1'] == 8  # ones unchanged
        assert output_intervened['O_2'] == 7  # 2 + 4 + 1 = 7 (affected by carry)
        assert output_intervened['raw_output'].strip() == '78'

    def test_intervene_on_O_1(self):
        """Test intervening on O_1 (ones place output)."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 2,
            "digit_0_1": 3,
            "digit_1_0": 4,
            "digit_1_1": 5,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output_original = model.new_trace(input_sample)
        assert output_original['O_1'] == 8
        assert output_original['raw_output'].strip() == '68'

        # Intervene: force O_1 = 9
        output_intervened = model.new_trace({**input_sample, 'O_1': 9})
        assert output_intervened['O_1'] == 9
        # Note: O_2 is not affected because it doesn't depend on O_1
        assert output_intervened["O_2"] == 6
        assert output_intervened["raw_output"].strip() == "69"

    def test_carry_propagation_intervention(self):
        """Test that intervening on C_1 propagates through the model."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # Setup where tens place is close to carrying
        # 29 + 40 = 69 (C_1=0, C_2=0)
        input_sample = {
            "digit_0_0": 2,
            "digit_0_1": 9,
            "digit_1_0": 4,
            "digit_1_1": 0,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output_original = model.new_trace(input_sample)
        assert output_original['C_1'] == 0  # 9 + 0 = 9 < 10
        assert output_original['C_2'] == 0  # 2 + 4 + 0 = 6 < 10
        assert output_original['raw_output'].strip() == '69'

        # Intervene: force C_1 = 1
        output_intervened = model.new_trace({**input_sample, 'C_1': 1})
        assert output_intervened['C_1'] == 1
        assert output_intervened['O_2'] == 7  # 2 + 4 + 1 = 7
        assert output_intervened['C_2'] == 0  # 2 + 4 + 1 = 7 < 10
        assert output_intervened['raw_output'].strip() == '79'

    def test_intervene_triggers_further_carry(self):
        """Test that forcing a carry can trigger additional carries."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # 29 + 70 = 99 (C_1=0, C_2=0)
        input_sample = {
            "digit_0_0": 2,
            "digit_0_1": 9,
            "digit_1_0": 7,
            "digit_1_1": 0,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output_original = model.new_trace(input_sample)
        assert output_original['C_1'] == 0  # 9 + 0 = 9 < 10
        assert output_original['C_2'] == 0  # 2 + 7 + 0 = 9 < 10
        assert output_original['O_3'] == 0  # no carry out
        assert output_original['raw_output'].strip() == '99'

        # Intervene: force C_1 = 1
        output_intervened = model.new_trace({**input_sample, 'C_1': 1})
        assert output_intervened['C_1'] == 1
        assert output_intervened['O_2'] == 0  # (2 + 7 + 1) % 10 = 10 % 10 = 0
        assert output_intervened['C_2'] == 1  # 2 + 7 + 1 = 10 >= 10
        assert output_intervened['O_3'] == 1  # carry out!
        assert output_intervened['raw_output'].strip() == '109'

    def test_interchange_intervention(self):
        """Test interchange intervention between two examples."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # Input: 23 + 45 = 68
        input_1 = {
            "digit_0_0": 2,
            "digit_0_1": 3,
            "digit_1_0": 4,
            "digit_1_1": 5,
            "num_digits": 2,
            "template": config.templates[0],
        }

        # Counterfactual: 27 + 45 = 72 (has carry from ones)
        input_2 = {
            "digit_0_0": 2,
            "digit_0_1": 7,
            "digit_1_0": 4,
            "digit_1_1": 5,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output_1 = model.new_trace(input_1)
        output_2 = model.new_trace(input_2)

        # Original outputs
        assert output_1["C_1"] == 0
        assert output_1["raw_output"].strip() == "68"
        assert output_2["C_1"] == 1
        assert output_2["raw_output"].strip() == "72"

        # Interchange C_1 from input_2 into input_1
        intervened = input_1.copy()
        intervened["C_1"] = input_2["C_1"]
        assert intervened["C_1"] == 1  # took from input_2
        assert intervened["O_2"] == 7  # 2 + 4 + 1 = 7
        assert intervened["raw_output"].strip() == "78"


class TestIntermediateModelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros(self):
        """Test 00 + 00 = 00."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 0,
            "digit_0_1": 0,
            "digit_1_0": 0,
            "digit_1_1": 0,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)
        assert output['raw_output'].strip() == '0'
        assert output['C_1'] == 0
        assert output['C_2'] == 0

    def test_maximum_two_digit(self):
        """Test 99 + 99 = 198."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        input_sample = {
            "digit_0_0": 9,
            "digit_0_1": 9,
            "digit_1_0": 9,
            "digit_1_1": 9,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)
        assert output['raw_output'].strip() == '198'
        assert output['C_1'] == 1  # 9 + 9 = 18 >= 10
        assert output['C_2'] == 1  # 9 + 9 + 1 = 19 >= 10
        assert output['O_1'] == 8
        assert output['O_2'] == 9
        assert output['O_3'] == 1

    def test_sum_to_ten_exactly(self):
        """Test when digits sum to exactly 10."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # 19 + 81 = 100 (ones: 9+1=10, tens: 1+8+1=10)
        input_sample = {
            "digit_0_0": 1,
            "digit_0_1": 9,
            "digit_1_0": 8,
            "digit_1_1": 1,
            "num_digits": 2,
            "template": config.templates[0],
        }

        output = model.new_trace(input_sample)
        assert output['raw_output'].strip() == '100'
        assert output['C_1'] == 1  # 9 + 1 = 10 >= 10
        assert output['C_2'] == 1  # 1 + 8 + 1 = 10 >= 10
        assert output['O_1'] == 0  # 10 % 10
        assert output['O_2'] == 0  # 10 % 10
        assert output['O_3'] == 1  # carry out


class TestVariableNaming:
    """Test that variable names follow the specification."""

    def test_carry_variables_start_at_1(self):
        """Verify carry variables are C_1, C_2, etc."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # Check variable names
        assert "C_1" in model.variables
        assert "C_2" in model.variables
        assert "C_0" not in model.variables  # Should not exist

    def test_output_variables_start_at_1(self):
        """Verify output variables are O_1, O_2, etc."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # Check variable names
        assert "O_1" in model.variables
        assert "O_2" in model.variables
        assert "O_3" in model.variables  # D+1
        assert "O_0" not in model.variables  # Should not exist

    def test_no_sum_variables(self):
        """Verify there are no sum variables (only C and O)."""
        config = create_general_config(2, 2)
        model = create_intermediate_addition_model(config)

        # Check that old sum variables don't exist
        for var in model.variables:
            assert not var.startswith("sum_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
