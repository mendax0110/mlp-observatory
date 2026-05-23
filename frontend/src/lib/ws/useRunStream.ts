import { useEffect } from "react";
import { WS_BASE } from "../api/config";

export function useRunStream(runId: string | null, onMessage: (msg: any) => void): void {
  useEffect(() => {
    if (!runId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/runs/${runId}`);

    ws.onopen = () => {
      ws.send("subscribe");
    };

    ws.onmessage = (event) => {
      onMessage(JSON.parse(event.data));
      ws.send("ack");
    };

    return () => {
      ws.close();
    };
  }, [runId, onMessage]);
}
