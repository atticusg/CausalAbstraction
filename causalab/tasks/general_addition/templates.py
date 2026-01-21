"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Template processing for addition tasks.

This module handles the conversion from digit values to formatted addition prompts.
"""

from typing import List, Dict, Optional
from .config import AdditionTaskConfig


class AdditionTemplateProcessor:
    """
    Processes templates to generate addition prompts from digit values.

    Handles variable numbers of addends and digits per number.
    """

    def __init__(self, config: AdditionTaskConfig):
        self.config = config

    def digits_to_number(self, digits: List[int]) -> int:
        """
        Convert a list of digits to an integer.

        Args:
            digits: List of digit values, where digits[0] is the most significant digit

        Returns:
            Integer value

        Example:
            [1, 2, 3] -> 123
        """
        result = 0
        for digit in digits:
            result = result * 10 + digit
        return result

    def number_to_digits(self, number: int, num_digits: int) -> List[int]:
        """
        Convert an integer to a list of digits with padding.

        Args:
            number: Integer value
            num_digits: Number of digits to output (with leading zeros if needed)

        Returns:
            List of digit values

        Example:
            (23, 3) -> [0, 2, 3]
        """
        digits = []
        n = number
        for _ in range(num_digits):
            digits.append(n % 10)
            n //= 10
        return list(reversed(digits))

    def format_number(self, digits: List[int]) -> str:
        """
        Format a list of digits as a number string (removing leading zeros).

        Args:
            digits: List of digit values

        Returns:
            String representation of the number

        Example:
            [0, 2, 3] -> "23"
            [1, 2, 3] -> "123"
        """
        # Convert to integer and back to string to remove leading zeros
        number = self.digits_to_number(digits)
        return str(number)

    def fill_template(
        self,
        template: str,
        numbers: List[List[int]]
    ) -> str:
        """
        Fill a template with formatted numbers.

        Args:
            template: Template string with {num0}, {num1}, etc. placeholders
            numbers: List of numbers, where each number is a list of digits

        Returns:
            Filled template string

        Example:
            template="The sum of {num0} and {num1} is"
            numbers=[[2, 3], [4, 5]]
            -> "The sum of 23 and 45 is"
        """
        # Build substitution dictionary
        substitutions = {}
        for i, num_digits in enumerate(numbers):
            if i < len(numbers):
                substitutions[f"num{i}"] = self.format_number(num_digits)
            else:
                substitutions[f"num{i}"] = "0"  # Default for unused placeholders

        # Fill template
        try:
            return template.format(**substitutions)
        except KeyError as e:
            raise ValueError(f"Template requires placeholder {e} that's not provided")

    def generate_prompt(
        self,
        digit_values: Dict[str, int],
        num_addends: int,
        num_digits: int,
        template: Optional[str] = None
    ) -> str:
        """
        Generate a complete addition prompt from digit values.

        Args:
            digit_values: Dictionary with keys like "digit_0_0", "digit_0_1", etc.
            num_addends: Number of numbers being added
            num_digits: Number of digits per number
            template: Template to use (if None, uses first template from config)

        Returns:
            Complete prompt string with prefix and suffix

        Example:
            digit_values={"digit_0_0": 2, "digit_0_1": 3, "digit_1_0": 4, "digit_1_1": 5}
            num_addends=2, num_digits=2
            -> "The sum of 23 and 45 is"
        """
        # Extract numbers from digit values
        numbers = []
        for k in range(num_addends):
            digits = []
            for d in range(num_digits):
                key = f"digit_{k}_{d}"
                if key in digit_values:
                    digits.append(digit_values[key])
                else:
                    digits.append(0)  # Default to 0 if missing
            numbers.append(digits)

        # Select template
        if template is None:
            template = self.config.templates[0]

        # Fill template
        core_text = self.fill_template(template, numbers)

        # Apply prefix and suffix
        return f"{self.config.prompt_prefix}{core_text}{self.config.prompt_suffix}"

    def compute_sum(self, numbers: List[List[int]]) -> List[int]:
        """
        Compute the sum of multiple numbers.

        Args:
            numbers: List of numbers, where each number is a list of digits

        Returns:
            List of digits representing the sum

        Example:
            [[2, 3], [4, 5]] -> [6, 8] (representing 68)
        """
        # Convert to integers
        int_numbers = [self.digits_to_number(num) for num in numbers]

        # Compute sum
        total = sum(int_numbers)

        # Convert back to digits (with one extra digit for potential carry)
        num_digits = len(numbers[0]) if numbers else 1
        return self.number_to_digits(total, num_digits + 1)

    def format_output(self, output_digits: List[int]) -> str:
        """
        Format output digits as a string (removing leading zeros).

        Args:
            output_digits: List of output digit values

        Returns:
            String representation of the result
        """
        return self.format_number(output_digits)
