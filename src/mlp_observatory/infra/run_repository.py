from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlp_observatory.domain.models import EpochMetrics


class LocalRunRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._normalize_existing_runs()

    def save_history(self, run_dir: Path, history: list[EpochMetrics]) -> None:
        payload = [asdict(x) for x in history]
        (run_dir / "history.json").write_text(json.dumps(payload, indent=2))

    def save_summary(self, run_dir: Path, summary: dict[str, Any]) -> None:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def save_system_stats(self, run_dir: Path, stats: list[dict[str, Any]]) -> None:
        (run_dir / "system_stats.json").write_text(json.dumps(stats, indent=2))

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2))

    def _parse_created_at(self, run_id: str, fallback_path: Path) -> str:
        try:
            dt = datetime.strptime(run_id, "run-%Y%m%d-%H%M%S").replace(tzinfo=UTC)
            return dt.isoformat()
        except ValueError:
            return datetime.fromtimestamp(fallback_path.stat().st_mtime, UTC).isoformat()

    def _infer_status(self, run_dir: Path) -> str:
        if (run_dir / "summary.json").exists():
            return "finished"
        if (run_dir / "history.json").exists():
            return "stopped"
        return "unknown"

    def _normalize_existing_runs(self) -> None:
        for path in self.root.glob("run-*"):
            if not path.is_dir():
                continue
            run_id = path.name
            meta_path = path / "meta.json"
            meta = self._read_json(meta_path) or {}
            changed = False

            if "run_id" not in meta:
                meta["run_id"] = run_id
                changed = True
            if "created_at" not in meta:
                meta["created_at"] = self._parse_created_at(run_id, path)
                changed = True
            if "status" not in meta:
                meta["status"] = self._infer_status(path)
                changed = True

            if changed:
                self._write_json(meta_path, meta)

    def create_run_dir(self) -> tuple[str, Path]:
        for _ in range(10):
            now = datetime.now(UTC)
            run_id = now.strftime("run-%Y%m%d-%H%M%S-%f") + f"-{random.randint(0, 9999):04d}"
            path = self.root / run_id
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue

            self._write_json(
                path / "meta.json",
                {"run_id": run_id, "created_at": now.isoformat(), "status": "running"},
            )
            return run_id, path

        raise RuntimeError("Unable to create unique run directory after multiple attempts")

    def get_run_dir(self, run_id: str) -> Path | None:
        path = self.root / run_id
        return path if path.exists() else None

    def save_config(self, run_dir: Path, cfg: Any) -> None:
        self._write_json(run_dir / "config.json", asdict(cfg))

    def save_started(self, run_dir: Path, payload: dict[str, Any]) -> None:
        self._write_json(run_dir / "started.json", payload)

    def update_status(self, run_dir: Path, status: str, error: str | None = None) -> None:
        meta = self._read_json(run_dir / "meta.json") or {}
        meta["run_id"] = run_dir.name
        meta["created_at"] = meta.get("created_at") or self._parse_created_at(run_dir.name, run_dir)
        meta["status"] = status
        if error:
            meta["error"] = error
        self._write_json(run_dir / "meta.json", meta)

    def load_summary(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self.get_run_dir(run_id)
        if not run_dir:
            return None
        return self._read_json(run_dir / "summary.json")

    def load_started(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self.get_run_dir(run_id)
        if not run_dir:
            return None
        return self._read_json(run_dir / "started.json")

    def load_history(self, run_id: str) -> list[dict[str, Any]] | None:
        run_dir = self.get_run_dir(run_id)
        if not run_dir:
            return None
        path = run_dir / "history.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            return raw
        return None

    def load_system_stats(self, run_id: str) -> list[dict[str, Any]] | None:
        run_dir = self.get_run_dir(run_id)
        if not run_dir:
            return None
        path = run_dir / "system_stats.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            return raw
        return None

    def list_runs(self) -> list[dict[str, Any]]:
        self._normalize_existing_runs()
        runs: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("run-*"), reverse=True):
            if not path.is_dir():
                continue
            run_id = path.name
            meta = self._read_json(path / "meta.json") or {}
            status = str(meta.get("status") or self._infer_status(path))
            created_at = str(meta.get("created_at") or self._parse_created_at(run_id, path))
            runs.append(
                {
                    "run_id": run_id,
                    "created_at": created_at,
                    "status": status,
                    "has_summary": (path / "summary.json").exists(),
                }
            )
        return runs
