"""Best-effort hardware discovery for strategy evaluation."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any

import psutil

from stratumllm.exceptions import HardwareDiscoveryError
from stratumllm.hardware import AcceleratorKind, GpuDevice, HardwareProfile


def discover_hardware_profile(
    *,
    gpu_utilization_target: float = 0.90,
    ram_utilization_target: float = 0.75,
    kv_cache_reserve_gib: float = 2.0,
) -> HardwareProfile:
    """
    Probe host RAM and accelerators, returning a validated ``HardwareProfile``.

    NVIDIA GPUs use ``nvidia-smi`` when available; optional ``pynvml`` is not required.
    On Apple Silicon without a discrete GPU, synthesizes a Metal residency cap and
    sets ``unified_memory=True`` (single pool; see ``HardwareProfile.unified_weight_budget_gib``).
    """
    mem = psutil.virtual_memory()
    system_ram_gib = mem.total / (1024**3)
    system_ram_available_gib = mem.available / (1024**3)

    gpus = _discover_gpus()
    accelerator = _infer_accelerator(gpus)
    unified = _is_apple_silicon() and not gpus

    if unified:
        # Nominal accelerator sizing = unified pool (feasibility uses unified_weight_budget).
        gpus = [
            GpuDevice(
                index=0,
                name=platform.processor() or "Apple GPU",
                vram_gib=system_ram_gib,
                vram_free_gib=system_ram_available_gib,
            )
        ]
        accelerator = AcceleratorKind.METAL

    return HardwareProfile(
        system_ram_gib=system_ram_gib,
        system_ram_available_gib=system_ram_available_gib,
        gpus=gpus,
        accelerator=accelerator,
        gpu_utilization_target=gpu_utilization_target,
        ram_utilization_target=ram_utilization_target,
        kv_cache_reserve_gib=kv_cache_reserve_gib,
        unified_memory=unified,
    )


def _discover_gpus() -> list[GpuDevice]:
    nvidia = _nvidia_smi_gpus()
    if nvidia:
        return nvidia
    return _pynvml_gpus()


def _nvidia_smi_gpus() -> list[GpuDevice]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []

    devices: list[GpuDevice] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        index_s, name, total_mib, free_mib = parts[0], parts[1], parts[2], parts[3]
        try:
            index = int(index_s)
            total_gib = float(total_mib) / 1024.0
            free_gib = float(free_mib) / 1024.0
        except ValueError:
            continue
        if total_gib <= 0:
            continue
        devices.append(
            GpuDevice(
                index=index,
                name=name,
                vram_gib=total_gib,
                vram_free_gib=free_gib,
            )
        )
    return devices


def _pynvml_gpus() -> list[GpuDevice]:
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return []

    try:
        pynvml.nvmlInit()
    except Exception:
        return []

    devices: list[GpuDevice] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode() if isinstance(name_raw, bytes) else str(name_raw)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gib = mem.total / (1024**3)
            free_gib = mem.free / (1024**3)
            devices.append(
                GpuDevice(
                    index=index,
                    name=name,
                    vram_gib=total_gib,
                    vram_free_gib=free_gib,
                )
            )
    except Exception:
        return devices
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return devices


def _infer_accelerator(gpus: list[GpuDevice]) -> AcceleratorKind:
    if gpus:
        return AcceleratorKind.CUDA
    if _is_apple_silicon():
        return AcceleratorKind.METAL
    return AcceleratorKind.CPU


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def hardware_from_dict(data: dict[str, Any]) -> HardwareProfile:
    """
    Build a profile from JSON/YAML-compatible dicts (integration hook).

    Raises:
        HardwareDiscoveryError: On validation failure with context.
    """
    try:
        return HardwareProfile.model_validate(data)
    except Exception as exc:
        raise HardwareDiscoveryError(f"Invalid hardware profile: {exc}") from exc
