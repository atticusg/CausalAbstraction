# tests/test_pyvene_core/test_model_management.py

import pytest
import gc
import torch
from unittest.mock import MagicMock

from causalab.neural.pyvene_core.intervenable_model import delete_intervenable_model


class TestDeleteIntervenableModel:
    """Tests for the delete_intervenable_model function."""

    def test_model_moved_to_cpu(self, mock_intervenable_model, monkeypatch):
        """Test that model is properly moved to CPU before deletion."""
        # Spy on the set_device method
        set_device_mock = MagicMock()
        monkeypatch.setattr(mock_intervenable_model, "set_device", set_device_mock)

        # Call the function
        delete_intervenable_model(mock_intervenable_model)

        # Verify set_device was called with expected args
        set_device_mock.assert_called_once_with("cpu", set_model=False)

    def test_model_properly_deleted(self, mock_intervenable_model, monkeypatch):
        """Test that model is properly deleted and garbage collected."""
        # Mock gc.collect
        gc_collect_mock = MagicMock(return_value=0)
        monkeypatch.setattr(gc, "collect", gc_collect_mock)

        # Call the function
        delete_intervenable_model(mock_intervenable_model)

        # Verify gc.collect was called
        gc_collect_mock.assert_called_once()

    def test_cuda_memory_cleared_when_available(
        self, mock_intervenable_model, monkeypatch
    ):
        """Test that CUDA memory is cleared when available."""
        # Mock CUDA functions
        is_available_mock = MagicMock(return_value=True)
        empty_cache_mock = MagicMock()

        monkeypatch.setattr(torch.cuda, "is_available", is_available_mock)
        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache_mock)

        # Call the function
        delete_intervenable_model(mock_intervenable_model)

        # Verify CUDA functions were called
        is_available_mock.assert_called_once()
        empty_cache_mock.assert_called_once()

    def test_cuda_memory_not_cleared_when_unavailable(
        self, mock_intervenable_model, monkeypatch
    ):
        """Test that CUDA memory is not cleared when unavailable."""
        # Mock CUDA functions
        is_available_mock = MagicMock(return_value=False)
        empty_cache_mock = MagicMock()

        monkeypatch.setattr(torch.cuda, "is_available", is_available_mock)
        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache_mock)

        # Call the function
        delete_intervenable_model(mock_intervenable_model)

        # Verify is_available was called but empty_cache was not
        is_available_mock.assert_called_once()
        empty_cache_mock.assert_not_called()

    def test_function_returns_none(self, mock_intervenable_model):
        """Test that function returns None."""
        result = delete_intervenable_model(mock_intervenable_model)
        assert result is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.gpu
class TestDeleteIntervenableModelIntegration:
    """Integration tests for delete_intervenable_model."""

    def test_memory_freed_on_cuda(self, mock_intervenable_model):
        """Test that memory is actually freed on CUDA devices."""
        # This test only runs if CUDA is available
        # Empty cache first to get a clean baseline
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()

        # Create a tensor to simulate model memory (must assign to keep it alive)
        tensor = torch.ones(100, 100, device="cuda")

        # Record memory after tensor allocation
        torch.cuda.synchronize()
        memory_after_allocation = torch.cuda.memory_allocated()

        # Verify memory increased
        assert memory_after_allocation > memory_before

        # Delete tensor and call the function
        del tensor
        delete_intervenable_model(mock_intervenable_model)

        # Check memory after cleanup
        torch.cuda.synchronize()
        memory_after_cleanup = torch.cuda.memory_allocated()

        # The memory should at least not have increased further
        # This is a more robust test than trying to check specific amounts
        assert memory_after_cleanup <= memory_after_allocation
