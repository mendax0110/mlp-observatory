from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mlp_observatory.core.settings import settings
from mlp_observatory.domain.events import RunEvent
from mlp_observatory.domain.models import DataConfig, ModelConfig, ProjectConfig, TaskType, TrainConfig
from mlp_observatory.infra.run_repository import LocalRunRepository
from mlp_observatory.infra.system_monitor import SystemMonitor
from mlp_observatory.infra.ws_hub import WebSocketHub
from mlp_observatory.services.evaluator import RunEvaluator
from mlp_observatory.training.data import make_dataset
from mlp_observatory.training.model import ModelFactory
from mlp_observatory.training.trainer import Trainer
from mlp_observatory.visualization.network_graph import NetworkGraphBuilder

logger = logging.getLogger(__name__)


class RunService:
    def __init__(self, repo: LocalRunRepository, hub: WebSocketHub) -> None:
        self.repo = repo
        self.hub = hub
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._summaries: dict[str, dict[str, Any]] = {}
        self._run_started_payload: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._run_configs: dict[str, ProjectConfig] = {}
        self._latest_trace: dict[str, list[dict[str, object]]] = {}
        self._system_stats_history: dict[str, list[dict[str, Any]]] = {}
        self._queue_maxsize = 256
        self._queue_sentinel = object()

    def start_run(self, task: TaskType, data: DataConfig, model: ModelConfig, train: TrainConfig) -> str:
        run_id, run_dir = self.repo.create_run_dir()
        cfg = ProjectConfig(task=task, data=data, model=model, train=train)

        self._events[run_id] = []
        self._run_configs[run_id] = cfg
        self._system_stats_history[run_id] = []
        task_handle = asyncio.create_task(self._run_training(run_id, run_dir, cfg))
        self._tasks[run_id] = task_handle
        self.repo.save_config(run_dir, cfg)
        return run_id

    def stop_run(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def list_runs(self) -> list[dict[str, Any]]:
        runs = self.repo.list_runs()
        for item in runs:
            task = self._tasks.get(item["run_id"])
            if task and not task.done():
                item["status"] = "running"
        return runs

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        summary = self._summaries.get(run_id)
        if summary is None:
            summary = self.repo.load_summary(run_id)
            if summary is not None:
                self._summaries[run_id] = summary
        return summary

    def get_history(self, run_id: str) -> list[dict[str, Any]] | None:
        return self.repo.load_history(run_id)

    def get_run_started_payload(self, run_id: str) -> dict[str, Any] | None:
        payload = self._run_started_payload.get(run_id)
        if payload is None:
            payload = self.repo.load_started(run_id)
            if payload is not None:
                self._run_started_payload[run_id] = payload
        return payload

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(run_id, []))

    def get_forward_trace(self, run_id: str) -> list[dict[str, object]] | None:
        return self._latest_trace.get(run_id)

    def get_system_stats(self, run_id: str) -> list[dict[str, Any]] | None:
        stats = self._system_stats_history.get(run_id)
        if stats is None:
            stats = self.repo.load_system_stats(run_id)
            if stats is not None:
                self._system_stats_history[run_id] = stats
        return stats

    async def _send_event(self, run_id: str, queue: asyncio.Queue[dict[str, Any] | object]) -> None:
        while True:
            item = await queue.get()
            if item is self._queue_sentinel:
                break
            await self.hub.publish(run_id, item)

    async def _run_training(self, run_id: str, run_dir: Path, cfg: ProjectConfig) -> None:
        device, warning = self._select_device(cfg.train.device)
        dataset, input_dim = make_dataset(cfg)
        cfg.data.features = input_dim
        model = ModelFactory.build(cfg.model, input_dim=input_dim)
        model_param_count = sum(p.numel() for p in model.parameters())

        queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue(maxsize=self._queue_maxsize)
        sender_task = asyncio.create_task(self._send_event(run_id, queue))

        async def enqueue(raw: dict[str, Any], drop_if_full: bool) -> None:
            if drop_if_full:
                try:
                    queue.put_nowait(raw)
                except asyncio.QueueFull:
                    return
            else:
                await queue.put(raw)

        async def publish(event: RunEvent) -> None:
            payload = event.to_dict()
            if event.event == "batch_update":
                trace = payload.get("payload", {}).get("forward_trace")
                if isinstance(trace, list):
                    self._latest_trace[run_id] = trace
            self._events.setdefault(run_id, []).append(payload)
            self._events[run_id] = self._events[run_id][-1000:]
            await enqueue(payload, drop_if_full=(event.event == "batch_update"))

        async def publish_raw(raw: dict[str, Any]) -> None:
            self._events.setdefault(run_id, []).append(raw)
            self._events[run_id] = self._events[run_id][-1000:]
            if raw.get("event") == "system_stats":
                stats = raw.get("payload")
                if isinstance(stats, dict):
                    self._system_stats_history.setdefault(run_id, []).append(stats)
                    self._system_stats_history[run_id] = self._system_stats_history[run_id][-2000:]
            await enqueue(raw, drop_if_full=True)

        graph = NetworkGraphBuilder.build(cfg.data, cfg.model)
        started_payload = {
            "task": cfg.task.value,
            "model_family": cfg.model.family,
            "device": str(device),
            "requested_device": cfg.train.device,
            "device_warning": warning,
            "graph": asdict(graph),
            "train": {"epochs": cfg.train.epochs},
            "parameter_count": model_param_count,
        }
        self.repo.save_started(run_dir, started_payload)
        self._run_started_payload[run_id] = started_payload
        await publish(RunEvent("run_started", run_id, started_payload))

        stop_monitor = asyncio.Event()
        monitor_task = asyncio.create_task(SystemMonitor.stream(publish_raw, run_id, stop_monitor))

        try:
            trainer = Trainer(model=model, config=cfg, device=device)
            history, diagnostics, final = await trainer.fit(
                run_id=run_id,
                dataset=dataset,
                publish=publish,
                update_every_steps=settings.update_every_steps,
            )

            best = min(history, key=lambda x: x.val_loss)
            recommendations = RunEvaluator.evaluate(cfg, history, diagnostics)
            suggestions = RunEvaluator.suggest_config(cfg, history, diagnostics)
            summary = {
                "run_id": run_id,
                "best_val_loss": best.val_loss,
                "best_val_accuracy": best.val_accuracy,
                "epochs": len(history),
                "diagnostics": diagnostics,
                "recommendations": recommendations,
                "suggestions": [
                    {
                        "message": s.message,
                        "config_patch": asdict(s.config_patch) if s.config_patch is not None else None,
                    }
                    for s in suggestions
                ],
                "feature_importance": final.get("feature_importance"),
            }
            self.repo.save_history(run_dir, history)
            self.repo.save_summary(run_dir, summary)
            self.repo.update_status(run_dir, "finished")
            self._summaries[run_id] = summary

            await publish(RunEvent("run_finished", run_id, summary))
        except asyncio.CancelledError:
            self.repo.update_status(run_dir, "stopped")
            await publish(RunEvent("run_failed", run_id, {"message": "Run cancelled"}))
            raise
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            self.repo.update_status(run_dir, "failed", error=str(exc))
            await publish(RunEvent("run_failed", run_id, {"message": str(exc)}))
            raise
        finally:
            stop_monitor.set()
            await monitor_task
            stats = self._system_stats_history.get(run_id)
            if stats is not None:
                self.repo.save_system_stats(run_dir, stats)
            await queue.put(self._queue_sentinel)
            await sender_task

    @staticmethod
    def _select_device(requested: str) -> tuple[torch.device, str | None]:
        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda"), None
            return torch.device("cpu"), "CUDA requested but not available. Falling back to CPU."

        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps"), None
            return torch.device("cpu"), "MPS requested but not available. Falling back to CPU."

        if requested == "cpu":
            return torch.device("cpu"), None

        if torch.cuda.is_available():
            return torch.device("cuda"), None
        if torch.backends.mps.is_available():
            return torch.device("mps"), None
        return torch.device("cpu"), None
