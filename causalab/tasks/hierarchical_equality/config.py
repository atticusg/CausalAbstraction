"""
Task-specific configuration and constants.
"""

TASK_NAME = "hierarchical_equality"

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

NUM_ICL_EXAMPLES = 60

PATTERNS = ["AABB", "ABCD", "ABCC", "AABC"]

MAX_TASK_TOKENS = 2048
MAX_NEW_TOKENS = 1

PROMPT_MODE = "minimal_function"  # "code", "algorithmic", or "minimal_function"
