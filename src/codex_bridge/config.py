"""Centralised configuration for codex-bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_MODELS_URL = "https://api.openai.com/v1/models"
DEFAULT_MODEL = "openai-codex/gpt-5.1-codex"
DEFAULT_ORIGINATOR = "nanobot"


@dataclass(frozen=True)
class Settings:
    """Application-wide settings, resolved once at startup."""

    host: str = "127.0.0.1"
    port: int = 8000
    default_model: str = DEFAULT_MODEL
    models: list[str] = field(default_factory=list)
    codex_url: str = DEFAULT_CODEX_URL
    models_url: str = DEFAULT_MODELS_URL
    stream_idle_timeout: float = 90.0
    models_timeout: float = 10.0
    originator: str = DEFAULT_ORIGINATOR

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from environment variables, with keyword overrides taking precedence."""
        models_raw = os.environ.get("CODEX_API_MODELS", "")
        models = [m.strip() for m in models_raw.split(",") if m.strip()] if models_raw else []

        defaults: dict[str, object] = {
            "host": os.environ.get("CODEX_API_HOST", "127.0.0.1"),
            "port": int(os.environ.get("CODEX_API_PORT", "8000")),
            "default_model": os.environ.get("CODEX_API_MODEL", DEFAULT_MODEL),
            "models": models,
            "codex_url": os.environ.get("CODEX_API_CODEX_URL", DEFAULT_CODEX_URL),
            "models_url": os.environ.get("CODEX_API_MODELS_URL", DEFAULT_MODELS_URL),
            "stream_idle_timeout": float(os.environ.get("CODEX_STREAM_IDLE_TIMEOUT_S", "90")),
            "models_timeout": float(os.environ.get("CODEX_MODELS_TIMEOUT_S", "10")),
        }
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**defaults)  # type: ignore[arg-type]
