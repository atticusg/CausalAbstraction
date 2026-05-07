"""
Test suite for attention pattern analysis.

Tests the attention pattern extraction and visualization functions.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from causalab.causal.trace import CausalTrace, Mechanism

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from causalab.methods.attention_pattern_analysis import (
    get_attention_patterns,
    compute_average_attention,
    analyze_attention_statistics,
)
from causalab.io.plots.attention_pattern import (
    plot_attention_heatmap,
    plot_attention_comparison,
    plot_attention_statistics,
    plot_layer_head_attention_grid,
)


# ---------------------- Fixtures ---------------------- #


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing attention extraction."""

    class MockConfig:
        num_hidden_layers = 4
        num_attention_heads = 8
        hidden_size = 64

    class MockTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0

        def __call__(
            self, text: str, return_tensors: str | None = None, **kwargs: Any
        ) -> Dict[str, torch.Tensor]:
            # Simple tokenization - 10 tokens per input
            seq_len = 10
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            }

        def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
            return [f"tok_{i}" for i in ids]

    class MockModel:
        config = MockConfig()
        device = torch.device("cpu")

        def __call__(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            output_attentions: bool = False,
            **kwargs: Any,
        ) -> MagicMock:
            assert input_ids is not None
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            num_heads = self.config.num_attention_heads
            num_layers = self.config.num_hidden_layers

            # Create mock attention outputs
            attentions = []
            for _ in range(num_layers):
                # Random attention weights (batch, heads, seq, seq)
                attn = torch.softmax(
                    torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1
                )
                attentions.append(attn)

            return MagicMock(attentions=tuple(attentions))

    class MockPipeline:
        model = MockModel()
        tokenizer = MockTokenizer()

        def load(self, prompts: List[CausalTrace]) -> Dict[str, torch.Tensor]:
            # Match real LMPipeline.load signature - takes list of CausalTrace
            text = prompts[0]["raw_input"]
            return self.tokenizer(text)

    return MockPipeline()


def _make_trace(raw_input: str) -> CausalTrace:
    """Create a simple CausalTrace with just raw_input for testing."""
    mechanisms = {"raw_input": Mechanism(parents=[], compute=lambda t: raw_input)}
    return CausalTrace(mechanisms, inputs={"raw_input": raw_input})


@pytest.fixture
def sample_prompts() -> List[CausalTrace]:
    """Sample prompts for testing."""
    return [
        _make_trace("The capital of France is"),
        _make_trace("The capital of Germany is"),
        _make_trace("The capital of England is"),
    ]


@pytest.fixture
def sample_attention_results():
    """Sample attention results for testing visualization and statistics."""
    seq_len = 10
    results = []
    for i in range(3):
        # Create random attention pattern
        pattern = np.random.rand(seq_len, seq_len)
        # Make it causal (lower triangular)
        pattern = np.tril(pattern)
        # Normalize rows
        pattern = pattern / (pattern.sum(axis=1, keepdims=True) + 1e-10)

        results.append(
            {
                "prompt": {"raw_input": f"Test prompt {i}"},
                "layer": 5,
                "head": 0,
                "attention_pattern": pattern,
                "token_positions": list(range(seq_len)),
                "filtered_attention": pattern,
                "seq_len": seq_len,
            }
        )
    return results


@pytest.fixture
def sample_tokens():
    """Sample tokens for axis labels."""
    return [
        "<s>",
        "The",
        "capital",
        "of",
        "France",
        "is",
        "Paris",
        ".",
        "</s>",
        "<pad>",
    ]


# ---------------------- Tests for get_attention_patterns ---------------------- #


class TestGetAttentionPatterns:
    """Tests for the get_attention_patterns function."""

    def test_basic_extraction(
        self, mock_pipeline: Any, sample_prompts: List[CausalTrace]
    ) -> None:
        """Test basic attention pattern extraction."""
        results = get_attention_patterns(
            pipeline=mock_pipeline,
            layer=0,
            head=0,
            prompts=sample_prompts,
        )

        assert len(results) == len(sample_prompts)
        for result in results:
            assert "attention_pattern" in result
            assert "layer" in result
            assert "head" in result
            assert "seq_len" in result
            assert result["layer"] == 0
            assert result["head"] == 0

    def test_attention_pattern_shape(
        self, mock_pipeline: Any, sample_prompts: List[CausalTrace]
    ) -> None:
        """Test that attention patterns have correct shape."""
        results = get_attention_patterns(
            pipeline=mock_pipeline,
            layer=0,
            head=0,
            prompts=sample_prompts[:1],
        )

        pattern = results[0]["attention_pattern"]
        seq_len = results[0]["seq_len"]
        assert pattern.shape == (seq_len, seq_len)

    def test_invalid_layer_raises_error(
        self, mock_pipeline: Any, sample_prompts: List[CausalTrace]
    ) -> None:
        """Test that invalid layer index raises ValueError."""
        with pytest.raises(ValueError, match="Layer index .* out of range"):
            get_attention_patterns(
                pipeline=mock_pipeline,
                layer=100,  # Invalid layer
                head=0,
                prompts=sample_prompts,
            )

    def test_invalid_head_raises_error(
        self, mock_pipeline: Any, sample_prompts: List[CausalTrace]
    ) -> None:
        """Test that invalid head index raises ValueError."""
        with pytest.raises(ValueError, match="Head index .* out of range"):
            get_attention_patterns(
                pipeline=mock_pipeline,
                layer=0,
                head=100,  # Invalid head
                prompts=sample_prompts,
            )

    def test_with_fixed_token_positions(
        self, mock_pipeline: Any, sample_prompts: List[CausalTrace]
    ) -> None:
        """Test extraction with fixed token positions."""
        positions = [0, 1, 2]
        results = get_attention_patterns(
            pipeline=mock_pipeline,
            layer=0,
            head=0,
            prompts=sample_prompts[:1],
            token_positions=positions,
        )

        assert results[0]["token_positions"] == positions
        assert results[0]["filtered_attention"].shape[0] == len(positions)

    def test_verbose_output(
        self,
        mock_pipeline: Any,
        sample_prompts: List[CausalTrace],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test verbose logging."""
        with caplog.at_level(logging.INFO):
            get_attention_patterns(
                pipeline=mock_pipeline,
                layer=0,
                head=0,
                prompts=sample_prompts,
                verbose=True,
            )


# ---------------------- Tests for compute_average_attention ---------------------- #


class TestComputeAverageAttention:
    """Tests for the compute_average_attention function."""

    def test_basic_average(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test basic averaging of attention patterns."""
        result = compute_average_attention(sample_attention_results)

        assert "average_pattern" in result
        assert "seq_len" in result
        assert "num_samples" in result
        assert result["num_samples"] == len(sample_attention_results)

    def test_average_pattern_shape(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test that averaged pattern has correct shape."""
        result = compute_average_attention(sample_attention_results)

        seq_len = sample_attention_results[0]["seq_len"]
        # With ignore_first_token=True (default), shape should be (seq_len-1, seq_len-1)
        assert result["average_pattern"].shape == (seq_len - 1, seq_len - 1)

    def test_ignore_first_token_false(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test averaging without ignoring first token."""
        result = compute_average_attention(
            sample_attention_results,
            ignore_first_token=False,
        )

        seq_len = sample_attention_results[0]["seq_len"]
        assert result["average_pattern"].shape == (seq_len, seq_len)
        assert result["ignored_first_token"] is False

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_average_attention([])

    def test_mismatched_lengths_raises_error(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test that mismatched sequence lengths raise ValueError."""
        # Modify one result to have different length
        modified_results = sample_attention_results.copy()
        modified_results[1] = {
            **modified_results[1],
            "seq_len": 15,  # Different length
            "attention_pattern": np.random.rand(15, 15),
        }

        with pytest.raises(ValueError, match="same sequence length"):
            compute_average_attention(modified_results)


# ---------------------- Tests for analyze_attention_statistics ---------------------- #


class TestAnalyzeAttentionStatistics:
    """Tests for the analyze_attention_statistics function."""

    def test_basic_statistics(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test basic statistics computation."""
        stats = analyze_attention_statistics(sample_attention_results)

        assert "avg_entropy" in stats
        assert "avg_max_attention" in stats
        assert "avg_diagonal" in stats
        assert "avg_previous" in stats

    def test_statistics_are_floats(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test that all statistics are floats."""
        stats = analyze_attention_statistics(sample_attention_results)

        for key, value in stats.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_statistics_ranges(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test that statistics are in expected ranges."""
        stats = analyze_attention_statistics(sample_attention_results)

        # Entropy should be non-negative
        assert stats["avg_entropy"] >= 0

        # Attention weights should be in [0, 1]
        assert 0 <= stats["avg_max_attention"] <= 1
        assert 0 <= stats["avg_diagonal"] <= 1
        assert 0 <= stats["avg_previous"] <= 1

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_attention_statistics([])


# ---------------------- Tests for plot_attention_heatmap ---------------------- #


class TestPlotAttentionHeatmap:
    """Tests for the plot_attention_heatmap function."""

    @pytest.fixture
    def sample_pattern(self) -> NDArray[np.floating[Any]]:
        """Sample attention pattern for visualization."""
        seq_len = 10
        pattern = np.random.rand(seq_len, seq_len)
        pattern = np.tril(pattern)
        return pattern / (pattern.sum(axis=1, keepdims=True) + 1e-10)

    def test_basic_heatmap(self, sample_pattern: NDArray[np.floating[Any]]) -> None:
        """Test basic heatmap creation."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_heatmap(
                attention_pattern=sample_pattern,
                title="Test Heatmap",
            )
        plt.close("all")

    def test_heatmap_with_tokens(
        self, sample_pattern: NDArray[np.floating[Any]], sample_tokens: List[str]
    ) -> None:
        """Test heatmap with token labels."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_heatmap(
                attention_pattern=sample_pattern,
                tokens=sample_tokens,
                title="Test Heatmap with Tokens",
            )
        plt.close("all")

    def test_heatmap_with_pad_token(
        self, sample_pattern: NDArray[np.floating[Any]], sample_tokens: List[str]
    ) -> None:
        """Test heatmap with pad token filtering."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_heatmap(
                attention_pattern=sample_pattern,
                tokens=sample_tokens,
                pad_token="<pad>",
                title="Test Heatmap with Pad Token Filter",
            )
        plt.close("all")

    def test_heatmap_ignore_first_token(
        self, sample_pattern: NDArray[np.floating[Any]], sample_tokens: List[str]
    ) -> None:
        """Test heatmap with first token ignored."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_heatmap(
                attention_pattern=sample_pattern,
                tokens=sample_tokens,
                ignore_first_token=True,
                title="Test Heatmap without First Token",
            )
        plt.close("all")

    def test_heatmap_with_save(self, sample_pattern: NDArray[np.floating[Any]]) -> None:
        """Test heatmap saving to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_heatmap.png")
            with patch("matplotlib.pyplot.show"):
                plot_attention_heatmap(
                    attention_pattern=sample_pattern,
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")


# ---------------------- Tests for plot_attention_comparison ---------------------- #


class TestPlotAttentionComparison:
    """Tests for the plot_attention_comparison function."""

    @pytest.fixture
    def sample_patterns(self) -> List[NDArray[np.floating[Any]]]:
        """Sample attention patterns for comparison."""
        seq_len = 10
        patterns = []
        for _ in range(4):
            pattern = np.random.rand(seq_len, seq_len)
            pattern = np.tril(pattern)
            patterns.append(pattern / (pattern.sum(axis=1, keepdims=True) + 1e-10))
        return patterns

    def test_basic_comparison(
        self, sample_patterns: List[NDArray[np.floating[Any]]]
    ) -> None:
        """Test basic comparison plot."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_comparison(
                attention_patterns=sample_patterns,
                labels=["Head 0", "Head 1", "Head 2", "Head 3"],
                title="Test Comparison",
            )
        plt.close("all")

    def test_comparison_with_tokens(
        self,
        sample_patterns: List[NDArray[np.floating[Any]]],
        sample_tokens: List[str],
    ) -> None:
        """Test comparison with token labels."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_comparison(
                attention_patterns=sample_patterns,
                labels=["Head 0", "Head 1", "Head 2", "Head 3"],
                tokens=sample_tokens,
                title="Test Comparison with Tokens",
            )
        plt.close("all")

    def test_comparison_with_pad_token(
        self,
        sample_patterns: List[NDArray[np.floating[Any]]],
        sample_tokens: List[str],
    ) -> None:
        """Test comparison with pad token filtering."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_comparison(
                attention_patterns=sample_patterns,
                labels=["Head 0", "Head 1", "Head 2", "Head 3"],
                tokens=sample_tokens,
                pad_token="<pad>",
                title="Test Comparison with Pad Filter",
            )
        plt.close("all")

    def test_comparison_mismatched_lengths_raises_error(
        self, sample_patterns: List[NDArray[np.floating[Any]]]
    ) -> None:
        """Test that mismatched patterns and labels raise error."""
        with pytest.raises(ValueError, match="must match"):
            plot_attention_comparison(
                attention_patterns=sample_patterns,
                labels=["Head 0", "Head 1"],  # Wrong number of labels
            )

    def test_comparison_with_save(
        self, sample_patterns: List[NDArray[np.floating[Any]]]
    ) -> None:
        """Test comparison saving to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_comparison.png")
            with patch("matplotlib.pyplot.show"):
                plot_attention_comparison(
                    attention_patterns=sample_patterns,
                    labels=["Head 0", "Head 1", "Head 2", "Head 3"],
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")


# ---------------------- Tests for plot_attention_statistics ---------------------- #


class TestPlotAttentionStatistics:
    """Tests for the plot_attention_statistics function."""

    @pytest.fixture
    def sample_stats(self) -> Dict[str, float]:
        """Sample statistics for visualization."""
        return {
            "avg_entropy": 2.5,
            "avg_max_attention": 0.8,
            "avg_diagonal": 0.3,
            "avg_previous": 0.4,
        }

    def test_basic_statistics_plot(self, sample_stats: Dict[str, float]) -> None:
        """Test basic statistics plot."""
        with patch("matplotlib.pyplot.show"):
            plot_attention_statistics(
                statistics=sample_stats,
                title="Test Statistics",
            )
        plt.close("all")

    def test_statistics_plot_with_save(self, sample_stats: Dict[str, float]) -> None:
        """Test statistics plot saving to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_stats.png")
            with patch("matplotlib.pyplot.show"):
                plot_attention_statistics(
                    statistics=sample_stats,
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")


# ---------------------- Tests for plot_layer_head_attention_grid ---------------------- #


class TestPlotLayerHeadAttentionGrid:
    """Tests for the plot_layer_head_attention_grid function."""

    def test_basic_grid(self, sample_attention_results: List[Dict[str, Any]]) -> None:
        """Test basic grid plot."""
        with patch("matplotlib.pyplot.show"):
            plot_layer_head_attention_grid(
                attention_results=sample_attention_results,
                title="Test Grid",
            )
        plt.close("all")

    def test_grid_with_tokens(
        self, sample_attention_results: List[Dict[str, Any]], sample_tokens: List[str]
    ) -> None:
        """Test grid with token labels."""
        with patch("matplotlib.pyplot.show"):
            plot_layer_head_attention_grid(
                attention_results=sample_attention_results,
                tokens=sample_tokens,
                title="Test Grid with Tokens",
            )
        plt.close("all")

    def test_grid_with_pad_token(
        self, sample_attention_results: List[Dict[str, Any]], sample_tokens: List[str]
    ) -> None:
        """Test grid with pad token filtering."""
        with patch("matplotlib.pyplot.show"):
            plot_layer_head_attention_grid(
                attention_results=sample_attention_results,
                tokens=sample_tokens,
                pad_token="<pad>",
                title="Test Grid with Pad Filter",
            )
        plt.close("all")

    def test_grid_with_save(
        self, sample_attention_results: List[Dict[str, Any]]
    ) -> None:
        """Test grid saving to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_grid.png")
            with patch("matplotlib.pyplot.show"):
                plot_layer_head_attention_grid(
                    attention_results=sample_attention_results,
                    save_path=save_path,
                    figure_format="png",
                )
            assert os.path.exists(save_path)
        plt.close("all")
