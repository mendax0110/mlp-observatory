from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from mlp_observatory.api.deps import get_container
from mlp_observatory.api.schemas import RunListResponse, RunSummaryResponse, StartRunRequest, StartRunResponse
from mlp_observatory.app.container import Container
from mlp_observatory.domain.models import DataConfig, LayerConfig, ModelConfig, TaskType, TrainConfig

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=StartRunResponse)
async def start_run(payload: StartRunRequest, container: Container = Depends(get_container)) -> StartRunResponse:
    model_data = payload.model.model_dump()
    layer_payloads = model_data.pop("layers", [])
    model_cfg = ModelConfig(
        **model_data,
        layers=[LayerConfig(**layer) for layer in layer_payloads],
    )

    run_id = container.run_service.start_run(
        task=TaskType(payload.task),
        data=DataConfig(**payload.data.model_dump()),
        model=model_cfg,
        train=TrainConfig(**payload.train.model_dump()),
    )
    return StartRunResponse(run_id=run_id)


@router.post("/{run_id}/stop")
async def stop_run(run_id: str, container: Container = Depends(get_container)) -> dict[str, str]:
    success = container.run_service.stop_run(run_id)
    if success:
        return {"message": f"Run {run_id} stopped successfully."}
    raise HTTPException(status_code=404, detail=f"Run {run_id} not found or already completed.")


@router.get("/{run_id}", response_model=RunSummaryResponse)
async def get_summary(run_id: str, container: Container = Depends(get_container)) -> RunSummaryResponse:
    summary = container.run_service.get_summary(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Run summary not available")
    return RunSummaryResponse(**summary)


@router.get("/{run_id}/trace")
async def get_forward_trace(run_id: str, container: Container = Depends(get_container)) -> dict[str, object]:
    trace = container.run_service.get_forward_trace(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Forward trace not available")
    return {"run_id": run_id, "trace": trace}


@router.get("/{run_id}/history")
async def get_history(run_id: str, container: Container = Depends(get_container)) -> dict[str, object]:
    history = container.run_service.get_history(run_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Run history not available")
    return {"run_id": run_id, "history": history}


@router.get("/{run_id}/system-stats")
async def get_system_stats(run_id: str, container: Container = Depends(get_container)) -> dict[str, object]:
    stats = container.run_service.get_system_stats(run_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="System stats not available")
    return {"run_id": run_id, "system_stats": stats}


@router.get("", response_model=RunListResponse)
async def list_runs(container: Container = Depends(get_container)) -> RunListResponse:
    return RunListResponse(runs=container.run_service.list_runs())
