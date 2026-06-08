from __future__ import annotations
import asyncio
import logging
import time
import psutil
import threading
import torch
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel
from mlp_observatory.core.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

_start_time = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    device: DeviceInfo
    system: SystemInfo
    storage: StorageInfo
    
    
class DeviceInfo(BaseModel):
    cuda_available: bool
    mps_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_memory_allocated_mb: float | None
    cuda_memory_reserved_mb: float | None
    
class SystemInfo(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    async_tasks: int
    python_threads: int
    
class StorageInfo(BaseModel):
    runs_dir: str
    runs_dir_exists: bool
    disk_free_gb: float

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cuda_available = torch.cuda.is_available()
    cuda_name: str | None = None
    cuda_alloc: float | None = None
    cuda_reserved: float | None = None
    
    if cuda_available:
        try:
            cuda_name = torch.cuda.get_device_name(0)
            cuda_alloc = torch.cuda.memory_allocated(0) / 1024 ** 2
            cuda_reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
        except Exception:
            pass
    
    mem = psutil.virtual_memory()
    cpu = await asyncio.to_thread(psutil.cpu_percent, interval=0.1)
    
    runs_dir = Path(settings.artifacts_dir)
    try:
        disk = psutil.disk_usage(str(runs_dir) if runs_dir.exists() else ".")
        disk_free_gb = disk.free / 1024 ** 3
    except Exception:
        disk_free_gb = -1.0
    
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        device=DeviceInfo(
            cuda_available=cuda_available,
            mps_available=torch.backends.mps.is_available(),
            cuda_device_count=torch.cuda.device_count(),
            cuda_device_name=cuda_name,
            cuda_memory_allocated_mb=cuda_alloc,
            cuda_memory_reserved_mb=cuda_reserved,
        ),
        system=SystemInfo(
            cpu_percent=cpu,
            ram_used_gb=round(mem.used / 1024 ** 3, 2),
            ram_total_gb=round(mem.total / 1024 ** 3, 2),
            ram_percent=mem.percent,
            async_tasks=len(asyncio.all_tasks()),
            python_threads=threading.active_count(),
        ),
        storage=StorageInfo(
            runs_dir=str(runs_dir),
            runs_dir_exists=runs_dir.exists(),
            disk_free_gb=round(disk_free_gb, 2),
        ),
    )
