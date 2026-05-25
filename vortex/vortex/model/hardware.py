# Copyright (c) 2024, Michael Poli.
# Hardware capability detection for Evo2 portable runtime.

"""
Centralized hardware capability detection for adaptive backend dispatch.

This module provides cached, device-aware capability queries that enable
the Evo2 runtime to dynamically select execution paths based on available
hardware features. All functions are safe to call on any device and will
gracefully return False on unsupported or unavailable hardware.

Capability thresholds:
    - FP8 Tensor Cores:      compute capability >= 8.9 (Ada Lovelace, Hopper, Blackwell)
    - Flash Attention 2:     compute capability >= 8.0 (Ampere+)
    - Triton kernels:        probed at import time

Design notes:
    - Results are cached per (device_type, device_index) since GPU capabilities
      are immutable for the lifetime of a process.
    - All functions catch exceptions internally — callers never need try/except.
    - This module has no side effects; it only performs read-only capability queries.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-device capability cache
# ---------------------------------------------------------------------------
# Key: (device_type: str, device_index: int)
# Value: (major: int, minor: int)
_CAPABILITY_CACHE: Dict[Tuple[str, int], Tuple[int, int]] = {}

# Triton availability (probed once at import time)
_TRITON_AVAILABLE: Optional[bool] = None


def _normalize_device(device: Union[str, torch.device, None]) -> Optional[torch.device]:
    """Normalize a device specification to a torch.device, or None on failure."""
    try:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
            return None
        if isinstance(device, str):
            return torch.device(device)
        return device
    except Exception:
        return None


def get_device_capability(device: Union[str, torch.device, None] = None) -> Tuple[int, int]:
    """
    Return the compute capability of the given CUDA device as (major, minor).

    Results are cached per device index. Returns (0, 0) for non-CUDA devices
    or if the query fails for any reason.

    Args:
        device: CUDA device specification. If None, uses the current device.

    Returns:
        Tuple of (major, minor) compute capability. (0, 0) on failure.
    """
    dev = _normalize_device(device)
    if dev is None or dev.type != "cuda":
        return (0, 0)

    device_index = dev.index
    if device_index is None:
        try:
            device_index = torch.cuda.current_device()
        except Exception:
            return (0, 0)

    cache_key = (dev.type, device_index)
    if cache_key in _CAPABILITY_CACHE:
        return _CAPABILITY_CACHE[cache_key]

    try:
        capability = torch.cuda.get_device_capability(device_index)
        _CAPABILITY_CACHE[cache_key] = capability
        return capability
    except Exception:
        _CAPABILITY_CACHE[cache_key] = (0, 0)
        return (0, 0)


def supports_fp8(device: Union[str, torch.device, None] = None) -> bool:
    """
    Check whether the given device supports FP8 execution via Transformer Engine.

    FP8 Tensor Cores are available on:
        - Ada Lovelace (SM 8.9): e.g. RTX 4090, L40
        - Hopper (SM 9.0): e.g. H100, H200
        - Blackwell (SM 10.0+): e.g. B100, B200

    Args:
        device: CUDA device to query. If None, uses current device.

    Returns:
        True if FP8 is supported, False otherwise.
    """
    return get_device_capability(device) >= (8, 9)


def supports_flash_attention(device: Union[str, torch.device, None] = None) -> bool:
    """
    Check whether the given device supports Flash Attention 2.

    Flash Attention 2 requires Ampere or newer (SM 8.0+).

    Args:
        device: CUDA device to query. If None, uses current device.

    Returns:
        True if Flash Attention 2 is supported, False otherwise.
    """
    return get_device_capability(device) >= (8, 0)


def supports_triton(device: Union[str, torch.device, None] = None) -> bool:
    """
    Check whether Triton kernels are available on this system.

    This probes whether the `triton` package can be imported. The result is
    cached globally (Triton availability is a system-level property, not
    per-device).

    Args:
        device: Currently unused, included for API consistency.

    Returns:
        True if Triton is importable, False otherwise.
    """
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import triton  # noqa: F401
            _TRITON_AVAILABLE = True
        except (ImportError, RuntimeError):
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def get_runtime_info() -> Dict[str, object]:
    """
    Collect diagnostic runtime information for logging and debugging.

    Returns a dictionary with:
        - cuda_available: bool
        - device_count: int
        - devices: list of per-device info dicts
        - triton_available: bool
        - transformer_engine_available: bool

    This function is safe to call at any time and will not raise exceptions.
    """
    info: Dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "triton_available": supports_triton(),
        "transformer_engine_available": False,
    }

    try:
        import transformer_engine  # noqa: F401
        info["transformer_engine_available"] = True
    except ImportError:
        pass

    if not torch.cuda.is_available():
        return info

    try:
        device_count = torch.cuda.device_count()
        info["device_count"] = device_count

        for i in range(device_count):
            cap = get_device_capability(torch.device("cuda", i))
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = "unknown"

            info["devices"].append({
                "index": i,
                "name": name,
                "compute_capability": f"{cap[0]}.{cap[1]}",
                "supports_fp8": cap >= (8, 9),
                "supports_flash_attn": cap >= (8, 0),
            })
    except Exception as e:
        log.debug(f"Error collecting device info: {e}")

    return info
