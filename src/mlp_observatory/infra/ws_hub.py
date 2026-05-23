from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class WebSocketHub:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, run_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections[run_id].add(ws)

    async def disconnect(self, run_id: str, ws: WebSocket) -> None:
        async with self._lock:
            if run_id in self._connections and ws in self._connections[run_id]:
                self._connections[run_id].remove(ws)

    async def publish(self, run_id: str, message: dict[str, Any]) -> None:
        async with self._lock:
            peers = list(self._connections.get(run_id, set()))

        dead: list[WebSocket] = []
        for peer in peers:
            try:
                await peer.send_json(message)
            except Exception:
                dead.append(peer)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections[run_id].discard(ws)
