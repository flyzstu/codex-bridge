"""Codex OAuth helpers."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CodexToken:
    account_id: str
    access: str


def load_token() -> CodexToken | None:
    """Load a usable Codex OAuth token, returning ``None`` when unavailable."""
    try:
        from oauth_cli_kit import get_token
    except ImportError:
        return None

    with suppress(Exception):
        token = get_token()
        access = getattr(token, "access", None)
        account_id = getattr(token, "account_id", None)
        if access and account_id:
            return CodexToken(account_id=str(account_id), access=str(access))
    return None


def login(print_fn: Callable[[str], None], prompt_fn: Callable[[str], str]) -> CodexToken:
    """Run oauth-cli-kit's interactive Codex OAuth flow."""
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
    except ImportError as exc:
        raise RuntimeError("oauth-cli-kit is not installed") from exc

    token = None
    with suppress(Exception):
        token = get_token()
    if not (getattr(token, "access", None) and getattr(token, "account_id", None)):
        token = login_oauth_interactive(print_fn=print_fn, prompt_fn=prompt_fn)
    if not (getattr(token, "access", None) and getattr(token, "account_id", None)):
        raise RuntimeError("Codex OAuth login failed")
    return CodexToken(account_id=str(token.account_id), access=str(token.access))


def logout() -> Path:
    """Remove the local Codex OAuth token file and return its path."""
    try:
        from oauth_cli_kit.providers import OPENAI_CODEX_PROVIDER
        from oauth_cli_kit.storage import FileTokenStorage
    except ImportError as exc:
        raise RuntimeError("oauth-cli-kit is not installed") from exc

    storage = FileTokenStorage(token_filename=OPENAI_CODEX_PROVIDER.token_filename)
    path = Path(storage.get_token_path())
    with suppress(FileNotFoundError):
        path.unlink()
    return path
