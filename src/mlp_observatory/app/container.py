from __future__ import annotations

from dataclasses import dataclass

from mlp_observatory.core.settings import settings
from mlp_observatory.infra.run_repository import LocalRunRepository
from mlp_observatory.infra.ws_hub import WebSocketHub
from mlp_observatory.services.run_service import RunService


@dataclass(slots=True)
class Container:
    run_service: RunService


def create_container() -> Container:
    repo = LocalRunRepository(settings.artifacts_dir)
    hub = WebSocketHub()
    run_service = RunService(repo=repo, hub=hub)
    return Container(run_service=run_service)
