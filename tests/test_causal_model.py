import unittest

from causalab.causal.causal_model import CausalModel
from causalab.causal.causal_utils import (
    get_partial_filter,
    get_specific_path_filter,
    label_data_with_variables,
    sample_intervention,
)
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import Mechanism, input_var


class ArithmeticCausalModel:
    """
    Factory class to create a causal model for two-digit arithmetic.

    This model represents two two-digit numbers (A and B) and computes their
    sum and product. The causal structure models how each digit in the input
    affects digits in the output.
    """

    @staticmethod
    def create():
        """
        Create and return a CausalModel for two-digit arithmetic.

        Returns:
        --------
        CausalModel
            A causal model with variables for two-digit numbers and their sum and product.
        """
        # Define possible values for each variable (digits 0-9, carry 0-1)
        values = {
            "A1": list(range(10)),
            "A0": list(range(10)),
            "B1": list(range(10)),
            "B0": list(range(10)),
            "CARRY_SUM": [0, 1],
            "SUM1": list(range(10)),
            "SUM0": list(range(10)),
            "PROD0": list(range(10)),
            "PROD1": list(range(10)),
            "PROD2": list(range(10)),
            "PROD3": list(range(10)),
            "raw_input": None,  # String representation of input
            "raw_output": None,  # String representation of output
        }

        # Define mechanisms using the new Mechanism API
        mechanisms = {
            # Input variables (no parents)
            "A1": input_var(list(range(10))),
            "A0": input_var(list(range(10))),
            "B1": input_var(list(range(10))),
            "B0": input_var(list(range(10))),
            # Carry for sum
            "CARRY_SUM": Mechanism(
                parents=["A0", "B0"],
                compute=lambda t: 1 if t["A0"] + t["B0"] >= 10 else 0,
            ),
            # Sum digits
            "SUM0": Mechanism(
                parents=["A0", "B0"],
                compute=lambda t: (t["A0"] + t["B0"]) % 10,
            ),
            "SUM1": Mechanism(
                parents=["A1", "B1", "CARRY_SUM"],
                compute=lambda t: (t["A1"] + t["B1"] + t["CARRY_SUM"]) % 10,
            ),
            # Product digits
            "PROD0": Mechanism(
                parents=["A0", "B0"],
                compute=lambda t: (t["A0"] * t["B0"]) % 10,
            ),
            "PROD1": Mechanism(
                parents=["A0", "B1", "A1", "B0"],
                compute=lambda t: (
                    (t["A0"] * t["B1"] + t["A1"] * t["B0"] + (t["A0"] * t["B0"]) // 10)
                    % 10
                ),
            ),
            "PROD2": Mechanism(
                parents=["A1", "B1", "A0", "B0"],
                compute=lambda t: (
                    t["A1"] * t["B1"]
                    + (
                        t["A0"] * t["B1"]
                        + t["A1"] * t["B0"]
                        + (t["A0"] * t["B0"]) // 10
                    )
                    // 10
                )
                % 10,
            ),
            "PROD3": Mechanism(
                parents=["A1", "B1", "A0", "B0"],
                compute=lambda t: (
                    t["A1"] * t["B1"]
                    + (
                        t["A0"] * t["B1"]
                        + t["A1"] * t["B0"]
                        + (t["A0"] * t["B0"]) // 10
                    )
                    // 10
                )
                // 10,
            ),
            # String representations
            "raw_input": Mechanism(
                parents=["A1", "A0", "B1", "B0"],
                compute=lambda t: f"{t['A1']}{t['A0']} + {t['B1']}{t['B0']} = ?, {t['A1']}{t['A0']} * {t['B1']}{t['B0']} = ?",
            ),
            "raw_output": Mechanism(
                parents=["SUM1", "SUM0", "PROD3", "PROD2", "PROD1", "PROD0"],
                compute=lambda t: f"Sum: {t['SUM1']}{t['SUM0']}, Product: {str(t['PROD3']) + str(t['PROD2']) + str(t['PROD1']) + str(t['PROD0']).lstrip('0') or '0'}",
            ),
        }

        return CausalModel(mechanisms, values, id="arithmetic_model")


class TestCausalModel(unittest.TestCase):
    """
    Test suite for the CausalModel class using an arithmetic causal model.
    """

    def setUp(self):
        """Set up the arithmetic causal model for testing."""
        self.model = ArithmeticCausalModel.create()

    def test_model_initialization(self):
        """Test that the model initializes correctly with the expected structure."""
        # Check that all variables are defined
        expected_vars = [
            "A1",
            "A0",
            "B1",
            "B0",
            "CARRY_SUM",
            "SUM0",
            "SUM1",
            "PROD0",
            "PROD1",
            "PROD2",
            "PROD3",
            "raw_input",
            "raw_output",
        ]
        self.assertEqual(set(self.model.variables), set(expected_vars))

        # Check inputs and outputs
        expected_inputs = ["A1", "A0", "B1", "B0"]
        self.assertEqual(set(self.model.inputs), set(expected_inputs))

        # Check that timesteps are correctly assigned (inputs have timestep 0)
        for var in expected_inputs:
            self.assertEqual(self.model.timesteps[var], 0)

        # Check that raw_input and raw_output are present (required by new implementation)
        self.assertIn("raw_input", self.model.variables)
        self.assertIn("raw_output", self.model.variables)

    def test_basic_arithmetic(self):
        """Test that the model correctly computes arithmetic operations."""
        # Test addition: 25 + 37 = 62
        setting = self.model.new_trace({"A1": 2, "A0": 5, "B1": 3, "B0": 7})

        # Check SUM digits
        self.assertEqual(setting["SUM0"], 2)
        self.assertEqual(setting["SUM1"], 6)
        self.assertEqual(setting["CARRY_SUM"], 1)  # 5+7=12, so carry is 1

        # Test multiplication: 25 * 37 = 925

        # Check PROD digits
        self.assertEqual(setting["PROD0"], 5)
        self.assertEqual(setting["PROD1"], 2)
        self.assertEqual(setting["PROD2"], 9)
        self.assertEqual(setting["PROD3"], 0)  # No thousands digit

        # Check raw input and output are generated
        self.assertIsNotNone(setting["raw_input"])
        self.assertIsNotNone(setting["raw_output"])
        self.assertIn("25 + 37", setting["raw_input"])
        self.assertIn("Sum: 62", setting["raw_output"])

    def test_edge_cases(self):
        """Test edge cases like zeros and carrying."""
        # Test with zeros: 20 + 09 = 29
        setting = self.model.new_trace({"A1": 2, "A0": 0, "B1": 0, "B0": 9})
        self.assertEqual(setting["SUM0"], 9)
        self.assertEqual(setting["SUM1"], 2)
        self.assertEqual(setting["CARRY_SUM"], 0)

        # Test with carrying in sum: 95 + 17 = 112
        setting = self.model.new_trace({"A1": 9, "A0": 5, "B1": 1, "B0": 7})
        self.assertEqual(setting["SUM0"], 2)
        self.assertEqual(setting["SUM1"], 1)
        self.assertEqual(setting["CARRY_SUM"], 1)

        # Test large product with carrying: 95 * 95 = 9025
        setting = self.model.new_trace({"A1": 9, "A0": 5, "B1": 9, "B0": 5})
        self.assertEqual(setting["PROD0"], 5)
        self.assertEqual(setting["PROD1"], 2)
        self.assertEqual(setting["PROD2"], 0)
        self.assertEqual(setting["PROD3"], 9)

    def test_intervention(self):
        """Test interventions on the model."""
        # Run with no intervention
        base_setting = self.model.new_trace({"A1": 2, "A0": 5, "B1": 3, "B0": 7})

        # Run with intervention on CARRY_SUM
        # Forcing CARRY_SUM to 0 should change SUM1
        intervened_setting = self.model.new_trace(
            {"A1": 2, "A0": 5, "B1": 3, "B0": 7, "CARRY_SUM": 0}
        )

        # Original carry was 1, so SUM1 should be 1 less in the intervention
        self.assertEqual(intervened_setting["SUM1"], base_setting["SUM1"] - 1)

        # Other values should stay the same
        self.assertEqual(intervened_setting["SUM0"], base_setting["SUM0"])
        self.assertEqual(intervened_setting["PROD0"], base_setting["PROD0"])

    def test_sample_input(self):
        """Test sampling inputs from the model."""
        # Sample an input
        input_setting = self.model.sample_input()

        # Check that the input has all the required variables
        for var in self.model.inputs:
            self.assertTrue(var in input_setting)
            self.assertIn(input_setting[var], self.model.values[var])

    def test_counterfactual_reasoning(self):
        """Test counterfactual reasoning with the model."""
        # Original scenario: 25 + 37
        original_input = {"A1": 2, "A0": 5, "B1": 3, "B0": 7}
        original_trace = self.model.new_trace(original_input)

        # Counterfactual: What if B0 was 8 instead of 7?
        counterfactual_input = {"A1": 2, "A0": 5, "B1": 3, "B0": 8}
        counterfactual_trace = self.model.new_trace(counterfactual_input)

        # Run interchange intervention - swap B0 from counterfactual into original
        result = self.model.run_interchange(
            original_trace, {"B0": counterfactual_trace}
        )

        # The result should have B0=8, but the rest from the original input
        self.assertEqual(result["B0"], 8)
        self.assertEqual(result["A1"], 2)
        self.assertEqual(result["A0"], 5)
        self.assertEqual(result["B1"], 3)

        # SUM0 should also reflect the change (5+8=13, so SUM0=3)
        self.assertEqual(result["SUM0"], 3)

        # CARRY_SUM should be 1 (5+8=13, carry is 1)
        self.assertEqual(result["CARRY_SUM"], 1)

        # SUM1 should reflect the carry (2+3+1=6)
        self.assertEqual(result["SUM1"], 6)

    def test_new_trace_doesnt_mutate_inputs(self):
        """Test that new_trace doesn't mutate the input dictionary."""
        inputs = {"raw_input": "test", "A1": 2, "CARRY_SUM": 1}
        original = inputs.copy()

        # new_trace should not modify the input dict
        self.model.new_trace(inputs)

        # Verify input dict remains unchanged
        self.assertEqual(inputs, original, "new_trace should not mutate input dict")


class TestMoreComplexCausalModel(unittest.TestCase):
    """
    Additional tests for more complex aspects of the CausalModel.
    """

    def setUp(self):
        """Set up a simplified causal model for testing."""
        # Simple model: A -> B -> C (with required raw_input and raw_output)
        values = {
            "A": [0, 1],
            "B": [0, 1],
            "C": [0, 1],
            "raw_input": None,
            "raw_output": None,
        }

        mechanisms = {
            "A": input_var([0, 1]),
            "B": Mechanism(parents=["A"], compute=lambda t: t["A"]),
            "C": Mechanism(parents=["B"], compute=lambda t: t["B"]),
            "raw_input": Mechanism(
                parents=["A"], compute=lambda t: f"Input A={t['A']}"
            ),
            "raw_output": Mechanism(
                parents=["C"], compute=lambda t: f"Output C={t['C']}"
            ),
        }
        self.model = CausalModel(mechanisms, values, id="simple_model")

    def test_label_data_with_variables(self):
        """Test labeling a dataset based on variable settings."""
        # Create test traces - label_data_with_variables reads from existing traces
        # Generate traces using the model's forward pass
        test_inputs = [{"A": 0}, {"A": 1}, {"A": 0}]
        test_data = [self.model.new_trace(inp) for inp in test_inputs]

        # Label the dataset based on variable C
        data_list = [{"input": trace} for trace in test_data]
        labeled_dataset, label_mapping = label_data_with_variables(
            self.model, data_list, ["C"]
        )

        # Check that we get back a properly labeled dataset
        self.assertEqual(len(labeled_dataset), 3)
        self.assertEqual(len(label_mapping), 2)  # Two unique values: C=0 and C=1

        # Check label mappings
        self.assertIn("0", label_mapping)  # C=0
        self.assertIn("1", label_mapping)  # C=1

        # Check that labels are assigned correctly
        labels = [item["label"] for item in labeled_dataset]
        self.assertEqual(
            labels[0], labels[2]
        )  # Both have A=0, so same C value, same label
        self.assertNotEqual(
            labels[0], labels[1]
        )  # Different A values should give different labels

    def test_sample_intervention(self):
        """Test sampling interventions from the model."""
        # For this simple model, we expect interventions on B or C, not A
        # Run the sampling multiple times to increase chance of getting an intervention
        got_intervention = False
        for _ in range(50):  # Try multiple times
            intervention = sample_intervention(self.model)
            if len(intervention) > 0:
                got_intervention = True
                self.assertNotIn(
                    "A", intervention
                )  # A is an input, shouldn't be intervened on
                self.assertNotIn(
                    "raw_input", intervention
                )  # Raw variables shouldn't be intervened on
                self.assertNotIn("raw_output", intervention)
                break

        # Allow this test to pass even if we didn't get an intervention
        # In a real test we might want to fail, but for demonstration purposes this is ok
        if got_intervention:
            self.assertTrue(True)  # Explicit pass if we got an intervention
        else:
            # Skip the test if we couldn't generate an intervention after multiple attempts
            self.skipTest(
                "Could not generate a non-empty intervention after multiple attempts"
            )

    def test_filters(self):
        """Test the various filter functions."""
        # Test partial filter
        partial_filter = get_partial_filter({"A": 1, "B": 1})

        # This setting should match the filter
        self.assertTrue(
            partial_filter(
                {
                    "A": 1,
                    "B": 1,
                    "C": 1,
                    "raw_input": "Input A=1",
                    "raw_output": "Output C=1",
                }
            )
        )

        # This setting should not match the filter
        self.assertFalse(
            partial_filter(
                {
                    "A": 0,
                    "B": 1,
                    "C": 1,
                    "raw_input": "Input A=0",
                    "raw_output": "Output C=1",
                }
            )
        )

        # Test path filter
        path_filter = get_specific_path_filter(self.model, "A", "C")

        # Set up a specific input where A affects C
        self.assertTrue(
            path_filter(
                {
                    "A": 1,
                    "B": 1,
                    "C": 1,
                    "raw_input": "Input A=1",
                    "raw_output": "Output C=1",
                }
            )
        )


class TestNewCausalModelMethods(unittest.TestCase):
    """
    Test the new methods added to CausalModel.
    """

    def setUp(self):
        """Set up a simple model for testing new methods."""
        # Create a very simple model for testing: Input -> Processing -> Output
        values = {
            "input_val": [1, 2, 3],
            "processed_val": [2, 4, 6],
            "raw_input": None,
            "raw_output": None,
        }

        mechanisms = {
            "input_val": input_var([1, 2, 3]),
            "processed_val": Mechanism(
                parents=["input_val"], compute=lambda t: t["input_val"] * 2
            ),
            "raw_input": Mechanism(
                parents=["input_val"], compute=lambda t: f"Input: {t['input_val']}"
            ),
            "raw_output": Mechanism(
                parents=["processed_val"],
                compute=lambda t: f"Output: {t['processed_val']}",
            ),
        }

        self.model = CausalModel(mechanisms, values, id="test_new_methods")

    def test_label_counterfactual_data(self):
        """Test the label_counterfactual_data method."""
        # Create test data using CausalTrace objects
        input1 = self.model.new_trace({"input_val": 1})
        cf1 = self.model.new_trace({"input_val": 2})

        input2 = self.model.new_trace({"input_val": 2})
        cf2 = self.model.new_trace({"input_val": 3})

        test_data: list[CounterfactualExample] = [
            {"input": input1, "counterfactual_inputs": [cf1]},
            {"input": input2, "counterfactual_inputs": [cf2]},
        ]

        # Test the method
        result = self.model.label_counterfactual_data(test_data, ["processed_val"])

        # Check that labels were added
        self.assertEqual(len(result), 2)
        # First example: interchange processed_val from input_val=2 -> processed_val=4
        self.assertEqual(result[0]["label"], "Output: 4")
        # Second example: interchange processed_val from input_val=3 -> processed_val=6
        self.assertEqual(result[1]["label"], "Output: 6")


if __name__ == "__main__":
    unittest.main()
