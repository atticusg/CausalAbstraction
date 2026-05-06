"""
Task-specific configuration and constants.
"""

TASK_NAME = "foo_bar"

FOOS = ["baz", "qux", "quux", "corge"]
BARS = ["grault", "garply", "waldo", "fred"]

# TODO: Token length requirements
# TODO (human): tests to encourage minimal token length requirements
MAX_TASK_TOKENS = 32  # How many tokens are needed for the input to fit?
MAX_NEW_TOKENS = 1  # How many tokens does the model need to generate?
