from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Awaitable, Callable

import psutil
from pynvml import (
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024**3
_BYTES_PER_MB = 1024**2

_nvml_lock = threading.Lock()
_nvml_initialized = False
_nvml_unavailable = False
_nvml_handle = None


def _init_nvml() -> None:
    global _nvml_initialized, _nvml_unavailable,_nvml_handle

    if _nvml_initialized:
        return
    if _nvml_unavailable:
        raise RuntimeError("NVIDIA NVML is unavailable")
    
    with _nvml_lock:
        if _nvml_initialized:
            return
        if _nvml_unavailable:
            raise RuntimeError("NVIDIA NVML is unavailable")
        try:
            nvmlInit()
            _nvml_handle = nvmlDeviceGetHandleByIndex(0)
            _nvml_initialized = True
        except Exception as e:
            logger.warning("Failed to initialize NVIDIA NVML: %s", e)
            _nvml_unavailable = True
            raise RuntimeError("NVIDIA NVML is unavailable") from e

class SystemMonitor:
    @staticmethod
    def read_stats() -> dict[str, Any]:
        vm = psutil.virtual_memory()
        stats: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": vm.percent,
            "ram_used_gb": round(vm.used / _BYTES_PER_GB, 2),
            "ram_total_gb": round(vm.total / _BYTES_PER_GB, 2),
            "gpu": SystemMonitor._read_nvidia(),
        }
        return stats

    @staticmethod
    def _read_nvidia() -> dict[str, Any] | None:
        try:
            _init_nvml()

            util = nvmlDeviceGetUtilizationRates(_nvml_handle)
            mem = nvmlDeviceGetMemoryInfo(_nvml_handle)
            temp = nvmlDeviceGetTemperature(_nvml_handle, NVML_TEMPERATURE_GPU)

            try:
                raw = nvmlDeviceGetPowerUsage(_nvml_handle)
                power = raw / 1000.0 if raw > 0 else None
            except Exception:
                power = None
                logger.debug("NVIDIA power usage not available", exc_info=True)
            
            name = nvmlDeviceGetName(_nvml_handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            return {
                "util_percent": float(util.gpu),
                "mem_used_mb": round(mem.used / _BYTES_PER_MB, 2),
                "mem_total_mb": round(mem.total / _BYTES_PER_MB, 2),
                "temp_c": float(temp),
                "power_w": float(power) if power is not None else None,
                "name": name,
            }

        except Exception:
            if _nvml_unavailable:
                logger.debug("NVIDIA monitoring unavailable", exc_info=True)
            else:
                logger.warning("Failed to read NVIDIA stats", exc_info=True)
            return None

    @staticmethod
    async def stream(
        publish: Callable[[dict[str, Any]], Awaitable[None]],
        run_id: str,
        stop_event: asyncio.Event,
    ) -> None:
        logger.info("System monitor stream started for %s", run_id)
        while not stop_event.is_set():
            stats = await asyncio.to_thread(SystemMonitor.read_stats)
            stats["ts"] = time.time()
            try:
                await publish({"event": "system_stats", "run_id": run_id, "payload": stats})
            except Exception:
                logger.warning("Failed to publish system stats for %s", run_id, exc_info=True)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
