"""Command-line interface for codex-openai-api."""

from __future__ import annotations

import os

import typer
from aiohttp import web

from .auth import login as oauth_login
from .auth import logout as oauth_logout
from .codex import DEFAULT_MODEL
from .server import create_app

app = typer.Typer(help="OpenAI-compatible gateway for Codex OAuth.")


@app.command()
def login() -> None:
    """Authenticate with OpenAI Codex OAuth."""
    try:
        token = oauth_login(print_fn=typer.echo, prompt_fn=lambda prompt: typer.prompt(prompt))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Authenticated with OpenAI Codex account {token.account_id}")


@app.command()
def logout() -> None:
    """Clear local OpenAI Codex OAuth credentials."""
    try:
        path = oauth_logout()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Removed Codex OAuth token at {path}")


@app.command()
def serve(
    host: str = typer.Option(os.environ.get("CODEX_API_HOST", "127.0.0.1"), help="Bind host."),
    port: int = typer.Option(int(os.environ.get("CODEX_API_PORT", "8000")), help="Bind port."),
    model: str = typer.Option(os.environ.get("CODEX_API_MODEL", DEFAULT_MODEL), help="Default model."),
) -> None:
    """Run the local OpenAI-compatible HTTP server."""
    web.run_app(create_app(default_model=model), host=host, port=port)
