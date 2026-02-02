"""CUDA availability tests for PyTorch."""

import torch


def test_pytorch_cuda_available():
  """Ensure PyTorch sees CUDA for GPU acceleration."""
  assert torch.cuda.is_available(), (
    "PyTorch CUDA is not available. "
    "In WSL2, ensure NVIDIA drivers and CUDA are installed, "
    "and that the CUDA-enabled PyTorch build is in use."
  )


def test_pytorch_cuda_device_count():
  """Ensure at least one CUDA device is visible."""
  assert torch.cuda.device_count() > 0, (
    "No CUDA devices detected by PyTorch. "
    "Verify GPU passthrough in WSL2 and NVIDIA driver setup."
  )
