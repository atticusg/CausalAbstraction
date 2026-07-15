"""Shared random word pool for random baseline tasks.

These words replace the steered variable's real values (days, months, years)
to test whether good directness scores reflect genuine causal structure
or are artifacts. All words are single-token under Llama-3.1-8B with space prefix.
"""

RANDOM_WORD_POOL: list[str] = [
    "apple",
    "stone",
    "blade",
    "crown",
    "pearl",
    "flame",
    "river",
    "storm",
    "chair",
    "glass",
    "cloud",
    "dream",
    "tiger",
    "maple",
    "frost",
    "piano",
    "lemon",
    "coral",
    "steel",
    "patch",
    "torch",
    "wheat",
    "brick",
    "lodge",
]


def get_random_words(n: int) -> list[str]:
    """Return the first *n* words from the shared pool."""
    if n > len(RANDOM_WORD_POOL):
        raise ValueError(
            f"Requested {n} random words but pool only has {len(RANDOM_WORD_POOL)}"
        )
    return RANDOM_WORD_POOL[:n]
