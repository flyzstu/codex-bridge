"""Command-line interface for codex-bridge."""

from __future__ import annotations

import os
import sys

import typer
from aiohttp import web
from loguru import logger

from .auth import login as oauth_login
from .auth import logout as oauth_logout
from .config import DEFAULT_MODEL, Settings
from .server import create_app

app = typer.Typer(help="OpenAI-compatible gateway for Codex OAuth.")


def _configure_logging(level: str) -> None:
    """Reset loguru and configure a single stderr sink at the given level."""
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


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
    model: str | None = typer.Option(
        None, help="Default model. If not specified, defaults to the first model in --models."
    ),
    models: str | None = typer.Option(
        None,
        help="Comma-separated list of supported model IDs. "
        "When set, /v1/models returns this list and requests for unlisted models are rejected.",
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug logging. Logs full Codex API request/response payloads.",
    ),
) -> None:
    """Run the local OpenAI-compatible HTTP server."""
    log_level = "DEBUG" if debug else os.environ.get("CODEX_LOG_LEVEL", "INFO").upper()
    _configure_logging(log_level)

    model_list = [m.strip() for m in models.split(",") if m.strip()] if models is not None else None
    settings = Settings.from_env(
        host=host,
        port=port,
        default_model=model,
        models=model_list,
        log_level=log_level,
    )
    logger.info(
        "Starting gateway: host={} port={} default_model={} models={} log_level={}",
        settings.host,
        settings.port,
        settings.default_model,
        settings.models or "(discover from API)",
        settings.log_level,
    )
    web.run_app(create_app(settings=settings), host=settings.host, port=settings.port, print=None)


@app.command()
def systemd(
    user: str = typer.Option(
        None, help="User to run the service as. Defaults to the current user."
    ),
    group: str = typer.Option(
        None, help="Group to run the service as. Defaults to the current group."
    ),
    host: str = typer.Option(
        "127.0.0.1", help="Host for the service to bind to."
    ),
    port: int = typer.Option(
        8000, help="Port for the service to bind to."
    ),
    write: bool = typer.Option(
        False,
        help="Write the file directly to /etc/systemd/system/codex-bridge.service (requires sudo).",
    ),
) -> None:
    """Generate or install a systemd service unit file for codex-bridge."""
    import getpass
    import shutil

    executable = shutil.which("codex-bridge")
    if not executable:
        executable = sys.argv[0]
        executable = os.path.abspath(executable)

    resolved_user = user or getpass.getuser()

    resolved_group = group
    if not resolved_group:
        try:
            import grp

            resolved_group = grp.getgrgid(os.getgid()).gr_name
        except Exception:
            resolved_group = resolved_user

    service_content = f"""[Unit]
Description=codex-bridge - OpenAI-compatible gateway for Codex OAuth
After=network.target

[Service]
Type=simple
User={resolved_user}
Group={resolved_group}
WorkingDirectory={os.getcwd()}
ExecStart={executable} serve --host {host} --port {port}
Restart=always
RestartSec=5
Environment=PATH={os.environ.get('PATH', '/usr/bin:/bin')}

[Install]
WantedBy=multi-user.target
"""
    if write:
        service_path = "/etc/systemd/system/codex-bridge.service"
        try:
            with open(service_path, "w") as f:
                f.write(service_content)
            logger.info(f"Successfully wrote systemd unit to {service_path}")
            logger.info("To enable and start the service, run:")
            logger.info("  sudo systemctl daemon-reload")
            logger.info("  sudo systemctl enable --now codex-bridge")
        except PermissionError as exc:
            logger.error(
                f"Permission denied: cannot write to {service_path}. Try running with sudo."
            )
            raise typer.Exit(1) from exc
        except Exception as exc:
            logger.error(f"Failed to write service file: {exc}")
            raise typer.Exit(1) from exc
    else:
        typer.echo(service_content)

