"""Runtime configuration helpers for LPDM."""

from __future__ import annotations

import torch


def select_torch_device() -> torch.device:
    """Choose the best available PyTorch execution device.

    Priority order:
    1. CUDA (NVIDIA GPU, cloud deployments)
    2. MPS (Apple Silicon local development)
    3. CPU fallback
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


DEVICE = select_torch_device()
