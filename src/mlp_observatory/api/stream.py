from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mlp_observatory.api.deps import get_container

router = APIRouter(tags=["stream"])


@router.websocket("/ws/runs/{run_id}")
async def stream_run(run_id: str, websocket: WebSocket) -> None:
    container = get_container(websocket)
    hub = container.run_service.hub
    await hub.connect(run_id, websocket)

    for event in container.run_service.get_events(run_id):
        await websocket.send_json(event)

    started = container.run_service.get_run_started_payload(run_id)
    if started is not None and not container.run_service.get_events(run_id):
        await websocket.send_json({"event": "run_started", "run_id": run_id, "payload": started})

    summary = container.run_service.get_summary(run_id)
    if summary is not None and not container.run_service.get_events(run_id):
        await websocket.send_json({"event": "run_finished", "run_id": run_id, "payload": summary})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(run_id, websocket)
