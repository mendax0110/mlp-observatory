from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLPO_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    artifacts_dir: Path = Path("runs")
    update_every_steps: int = 2
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])


settings = Settings()
