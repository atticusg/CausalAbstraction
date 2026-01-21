import unittest
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var


class TestRunInterchangeArrowSyntax(unittest.TestCase):
    """
    Test suite for the run_interchange method with "<-" arrow syntax.

    The arrow syntax "original_var<-counterfactual_var" allows specifying
    different variable names in the original and counterfactual inputs.
    """

    def setUp(self):
        """Set up a simple causal model for testing."""
        # Create a simple model with two separate chains
        # Chain 1: X -> Y -> Z
        # Chain 2: A -> B -> C

        values = {
            "X": [0, 1, 2],
            "A": [0, 1, 2],
            "Y": [0, 1, 2, 3],
            "B": [0, 1, 2, 3],
            "Z": [0, 1, 2, 3, 4],
            "C": [0, 1, 2, 3, 4],
            "raw_input": None,
            "raw_output": None,
        }

        mechanisms = {
            "X": input_var([0, 1, 2]),
            "A": input_var([0, 1, 2]),
            "Y": Mechanism(parents=["X"], compute=lambda t: t["X"] + 1),
            "B": Mechanism(parents=["A"], compute=lambda t: t["A"] + 1),
            "Z": Mechanism(parents=["Y"], compute=lambda t: t["Y"] + 1),
            "C": Mechanism(parents=["B"], compute=lambda t: t["B"] + 1),
            "raw_input": Mechanism(
                parents=["X", "A"], compute=lambda t: f"X={t['X']}, A={t['A']}"
            ),
            "raw_output": Mechanism(
                parents=["Z", "C"], compute=lambda t: f"Z={t['Z']}, C={t['C']}"
            ),
        }

        self.model = CausalModel(mechanisms, values, id="test_model")

    def test_original_syntax_still_works(self):
        """Test that the original syntax (without arrow) still works correctly."""
        # Original input: X=1, A=2
        # This should produce: Y=2, B=3, Z=3, C=4
        original_trace = self.model.new_trace({"X": 1, "A": 2})

        # Counterfactual input: X=0, A=0
        # This should produce: Y=1, B=1, Z=2, C=2
        counterfactual_trace = self.model.new_trace({"X": 0, "A": 0})

        # Interchange on Y (using original syntax)
        # Take Y from counterfactual (Y=1) and use it in original
        result = self.model.run_interchange(original_trace, {"Y": counterfactual_trace})

        # Y should be 1 (from counterfactual), Z should be 2 (Y+1)
        # C should be 4 (from original path)
        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["Z"], 2)
        self.assertEqual(result["C"], 4)

    def test_arrow_syntax_basic(self):
        """Test basic arrow syntax functionality."""
        # Original input: X=1, A=2
        # This produces: Y=2, B=3, Z=3, C=4
        original_trace = self.model.new_trace({"X": 1, "A": 2})

        # Counterfactual input: X=0, A=0
        # This produces: Y=1, B=1, Z=2, C=2
        counterfactual_trace = self.model.new_trace({"X": 0, "A": 0})

        # Use arrow syntax: "Y<-B"
        # This should take B from counterfactual (B=1) and use it for Y in original
        result = self.model.run_interchange(
            original_trace, {"Y<-B": counterfactual_trace}
        )

        # Y should be 1 (from counterfactual B)
        # Z should be 2 (Y+1)
        # B and C should be from original (B=3, C=4)
        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["Z"], 2)
        self.assertEqual(result["B"], 3)
        self.assertEqual(result["C"], 4)

    def test_arrow_syntax_with_spaces(self):
        """Test that arrow syntax works with spaces around the arrow."""
        original_trace = self.model.new_trace({"X": 1, "A": 2})
        counterfactual_trace = self.model.new_trace({"X": 0, "A": 0})

        # Test with spaces: "Y <- B"
        result = self.model.run_interchange(
            original_trace, {"Y <- B": counterfactual_trace}
        )

        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["Z"], 2)
        self.assertEqual(result["B"], 3)
        self.assertEqual(result["C"], 4)

    def test_arrow_syntax_swap_chains(self):
        """Test swapping values between two independent chains."""
        # Original: X=1 -> Y=2 -> Z=3, A=2 -> B=3 -> C=4
        original_trace = self.model.new_trace({"X": 1, "A": 2})

        # Counterfactual: X=0 -> Y=1 -> Z=2, A=0 -> B=1 -> C=2
        counterfactual_trace = self.model.new_trace({"X": 0, "A": 0})

        # Swap: Use B's value from counterfactual for Y in original
        # and Y's value from counterfactual for B in original
        result = self.model.run_interchange(
            original_trace,
            {"Y<-B": counterfactual_trace, "B<-Y": counterfactual_trace},
        )

        # Y should be 1 (from counterfactual B=1)
        # B should be 1 (from counterfactual Y=1)
        # Z should be 2 (Y+1=1+1)
        # C should be 2 (B+1=1+1)
        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["B"], 1)
        self.assertEqual(result["Z"], 2)
        self.assertEqual(result["C"], 2)

    def test_arrow_syntax_mixed_with_regular(self):
        """Test mixing arrow syntax with regular syntax in the same interchange."""
        original_trace = self.model.new_trace({"X": 1, "A": 2})
        counterfactual_trace = self.model.new_trace({"X": 0, "A": 0})

        # Mix arrow syntax and regular syntax
        result = self.model.run_interchange(
            original_trace,
            {
                "Y<-B": counterfactual_trace,  # Arrow syntax
                "B": counterfactual_trace,  # Regular syntax
            },
        )

        # Y should be 1 (from counterfactual B=1)
        # B should be 1 (from counterfactual B=1, regular syntax)
        # Z should be 2 (Y+1)
        # C should be 2 (B+1)
        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["B"], 1)
        self.assertEqual(result["Z"], 2)
        self.assertEqual(result["C"], 2)

    def test_arrow_syntax_different_counterfactuals(self):
        """Test arrow syntax with different counterfactual inputs for different variables."""
        original_trace = self.model.new_trace({"X": 1, "A": 2})

        # Two different counterfactual inputs
        counterfactual_1 = self.model.new_trace({"X": 0, "A": 0})  # Y=1, B=1
        counterfactual_2 = self.model.new_trace({"X": 2, "A": 2})  # Y=3, B=3

        # Use different counterfactuals for different interventions
        result = self.model.run_interchange(
            original_trace,
            {
                "Y<-B": counterfactual_1,  # Y gets B from counterfactual_1 (B=1)
                "B<-Y": counterfactual_2,  # B gets Y from counterfactual_2 (Y=3)
            },
        )

        # Y should be 1 (from counterfactual_1 B=1)
        # B should be 3 (from counterfactual_2 Y=3)
        # Z should be 2 (Y+1=1+1)
        # C should be 4 (B+1=3+1)
        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["B"], 3)
        self.assertEqual(result["Z"], 2)
        self.assertEqual(result["C"], 4)

    def test_arrow_syntax_input_variable(self):
        """Test arrow syntax with input variables."""
        original_trace = self.model.new_trace({"X": 1, "A": 2})
        counterfactual_trace = self.model.new_trace({"X": 0, "A": 0})

        # Use arrow syntax to swap input variables
        result = self.model.run_interchange(
            original_trace, {"X<-A": counterfactual_trace}
        )

        # X should be 0 (from counterfactual A=0)
        # A should be 2 (original)
        # Y should be 1 (X+1=0+1)
        # B should be 3 (A+1=2+1)
        # Z should be 2 (Y+1=1+1)
        # C should be 4 (B+1=3+1)
        self.assertEqual(result["X"], 0)
        self.assertEqual(result["A"], 2)
        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["B"], 3)
        self.assertEqual(result["Z"], 2)
        self.assertEqual(result["C"], 4)

    def test_arrow_syntax_same_variable_name(self):
        """Test that arrow syntax works even when using the same variable name on both sides."""
        original_trace = self.model.new_trace({"X": 1, "A": 2})
        counterfactual_trace = self.model.new_trace({"X": 0, "A": 0})

        # Use "Y<-Y" which should behave like the regular syntax
        result = self.model.run_interchange(
            original_trace, {"Y<-Y": counterfactual_trace}
        )

        # Should behave the same as regular syntax "Y"
        # Y should be 1 (from counterfactual)
        # Z should be 2 (Y+1)
        self.assertEqual(result["Y"], 1)
        self.assertEqual(result["Z"], 2)


if __name__ == "__main__":
    unittest.main()
