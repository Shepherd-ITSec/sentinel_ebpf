"""CUDA availability tests for PyTorch. Skip when using CPU-only torch or no GPU."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch CUDA not available (CPU-only build or no GPU)")
def test_pytorch_cuda_available():
  """Ensure PyTorch sees CUDA for GPU acceleration."""
  assert torch.cuda.is_available()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch CUDA not available (CPU-only build or no GPU)")
def test_pytorch_cuda_device_count():
  """Ensure at least one CUDA device is visible."""
  assert torch.cuda.device_count() > 0
