"""Tests for the coherence metric.

Pure math tests — no GPU required.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Coherence Score
# ---------------------------------------------------------------------------


class TestCoherence:
    """Coherence = mean concept mass along a path (1 - non-concept mass)."""

    def test_fully_coherent(self):
        """Distributions summing to 1 should have coherence 1.0."""
        from causalab.methods.scores.coherence import (
            compute_score_single_path,
        )

        distributions = torch.tensor(
            [
                [0.3, 0.7],
                [0.5, 0.5],
                [0.7, 0.3],
            ]
        )
        assert compute_score_single_path(distributions) == pytest.approx(1.0, abs=1e-6)

    def test_half_coherent(self):
        """If concept tokens get 50% of mass, coherence ≈ 0.5."""
        from causalab.methods.scores.coherence import (
            compute_score_single_path,
        )

        distributions = torch.tensor(
            [
                [0.25, 0.25],
                [0.25, 0.25],
                [0.25, 0.25],
            ]
        )
        assert compute_score_single_path(distributions) == pytest.approx(0.5, abs=1e-6)

    def test_varying_coherence(self):
        """Concept mass varies along path — check averaging."""
        from causalab.methods.scores.coherence import (
            compute_score_single_path,
        )

        distributions = torch.tensor(
            [
                [0.9, 0.1],  # 100% concept
                [0.45, 0.05],  # 50% concept
                [0.0, 0.0],  # 0% concept
            ]
        )
        # Mean concept mass: (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert compute_score_single_path(distributions) == pytest.approx(0.5, abs=1e-6)

    def test_concept_mass_above_one_raises(self):
        """If concept_mass > 1 + tol, the assertion should fire."""
        from causalab.methods.scores.coherence import (
            compute_score_single_path,
        )

        distributions = torch.tensor(
            [
                [0.6, 0.6],  # sum = 1.2 — should trip the guard
            ]
        )
        with pytest.raises(AssertionError):
            compute_score_single_path(distributions)
