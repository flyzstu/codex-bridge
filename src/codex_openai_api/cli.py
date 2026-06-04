"""Command-line interface for codex-openai-api."""

from __future__ import annotations

import os

import typer
from aiohttp import web
from loguru import logger

from .auth import login as oauth_login
from .auth import logout as oauth_logout
from .config import DEFAULT_MODEL, Settings
from .server import create_app

app = typer.Typer(help="OpenAI-compatible gateway for Codex OAuth.")


@app.command()
def login() -> None:
    """Authenticate with OpenAI Codex OAuth."""
    try:
        token = oauth_login(
            print_fn=lambda msg: logger.info(msg),
            prompt_fn=lambda prompt: typer.prompt(prompt),
        )
    except RuntimeError as exc:
        logger.error(f"Error: {exc}")
        raise typer.Exit(1) from exc
    logger.info(f"Authenticated with OpenAI Codex account {token.account_id}")


@app.command()
def logout() -> None:
    """Clear local OpenAI Codex OAuth credentials."""
    try:
        path = oauth_logout()
    except RuntimeError as exc:
        logger.error(f"Error: {exc}")
        raise typer.Exit(1) from exc
    logger.info(f"Removed Codex OAuth token at {path}")


@app.command()
def serve(
    host: str = typer.Option(
        os.environ.get("CODEX_API_HOST", "127.0.0.1"), help="Bind host."
    ),
    port: int = typer.Option(
        int(os.environ.get("CODEX_API_PORT", "8000")), help="Bind port."
    ),
    model: str = typer.Option(
        os.environ.get("CODEX_API_MODEL", DEFAULT_MODEL), help="Default model."
    ),
    models: str = typer.Option(
        os.environ.get("CODEX_API_MODELS", ""),
        help="Comma-separated list of supported model IDs. "
        "When set, /v1/models returns this list and requests for unlisted models are rejected.",
    ),
) -> None:
    """Run the local OpenAI-compatible HTTP server."""
    model_list = [m.strip() for m in models.split(",") if m.strip()] if models else []
    settings = Settings.from_env(
        host=host,
        port=port,
        default_model=model,
        models=model_list or None,
    )
    logger.info(
        "Starting gateway: host={} port={} default_model={} models={}",
        settings.host,
        settings.port,
        settings.default_model,
        settings.models or "(discover from API)",
    )
    web.run_app(create_app(settings=settings), host=settings.host, port=settings.port)
