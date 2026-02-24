"""Configuration loaded entirely from environment variables.

Every secret and tuneable is read from the environment (or a .env file
loaded by the caller).  Nothing is ever hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated string of integers, ignoring blanks."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _default_model(provider: str) -> str:
    """Return the default model name for a given LLM provider."""
    defaults = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
    }
    return defaults.get(provider, "claude-sonnet-4-20250514")


@dataclass
class Config:
    """Application configuration – every field sourced from env vars."""

    # --- Telegram -----------------------------------------------------------
    telegram_bot_token: str = ""
    allowed_telegram_ids: list[int] = field(default_factory=list)

    # --- LLM ----------------------------------------------------------------
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_provider: str = "anthropic"
    llm_model: str = ""

    # --- Google Calendar ----------------------------------------------------
    google_credentials_file: str = "credentials.json"
    google_token_file: str = "token.json"
    google_calendar_id: str = "primary"

    # --- Gmail (SMTP/IMAP) --------------------------------------------------
    gmail_address: str = ""
    gmail_app_password: Optional[str] = None

    # --- Storage / logging --------------------------------------------------
    database_path: str = "nestor.db"
    log_level: str = "INFO"

    # --- Factory ------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from the current environment."""
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        model = os.environ.get("LLM_MODEL", "") or _default_model(provider)

        raw_ids = os.environ.get("ALLOWED_TELEGRAM_IDS", "")

        return cls(
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            allowed_telegram_ids=_parse_int_list(raw_ids) if raw_ids else [],
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            llm_provider=provider,
            llm_model=model,
            gmail_address=os.environ.get("GMAIL_ADDRESS", ""),
            gmail_app_password=os.environ.get("GMAIL_APP_PASSWORD"),
            google_credentials_file=os.environ.get(
                "GOOGLE_CREDENTIALS_FILE", "credentials.json"
            ),
            google_token_file=os.environ.get("GOOGLE_TOKEN_FILE", "token.json"),
            google_calendar_id=os.environ.get("GOOGLE_CALENDAR_ID", "primary"),
            database_path=os.environ.get("DATABASE_PATH", "nestor.db"),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        )

    # --- Validation ---------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`ValueError` if any required variable is missing."""
        errors: list[str] = []

        if not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN is required")

        if not self.allowed_telegram_ids:
            errors.append(
                "ALLOWED_TELEGRAM_IDS is required "
                "(comma-separated Telegram user IDs)"
            )

        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            errors.append(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
            )
        elif self.llm_provider == "openai" and not self.openai_api_key:
            errors.append(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        elif self.llm_provider not in ("anthropic", "openai"):
            errors.append(
                f"LLM_PROVIDER must be 'anthropic' or 'openai', "
                f"got '{self.llm_provider}'"
            )

        if errors:
            raise ValueError(
                "Configuration errors:\n  • " + "\n  • ".join(errors)
            )
