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


def _parse_bool(raw: str | None, default: bool) -> bool:
    """Parse common boolean env var forms."""
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


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
    db_backup_to_drive: bool = True
    db_backup_interval_hours: int = 24
    db_backup_filename: str = "nestor-backup-latest.sqlite3"
    db_backup_drive_folder_id: Optional[str] = None
    db_backup_on_start: bool = True
    db_restore_from_drive: bool = True

    # --- Factory ------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from the current environment."""
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        model = os.environ.get("LLM_MODEL", "") or _default_model(provider)

        raw_ids = os.environ.get("ALLOWED_TELEGRAM_IDS", "")
        backup_interval_raw = os.environ.get("DB_BACKUP_INTERVAL_HOURS", "24")
        try:
            backup_interval = int(backup_interval_raw)
        except ValueError:
            backup_interval = 24
        backup_interval = max(1, backup_interval)

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
            db_backup_to_drive=_parse_bool(
                os.environ.get("DB_BACKUP_TO_DRIVE"),
                True,
            ),
            db_backup_interval_hours=backup_interval,
            db_backup_filename=os.environ.get(
                "DB_BACKUP_FILENAME",
                "nestor-backup-latest.sqlite3",
            ),
            db_backup_drive_folder_id=os.environ.get("DB_BACKUP_DRIVE_FOLDER_ID"),
            db_backup_on_start=_parse_bool(
                os.environ.get("DB_BACKUP_ON_START"),
                True,
            ),
            db_restore_from_drive=_parse_bool(
                os.environ.get("DB_RESTORE_FROM_DRIVE"),
                True,
            ),
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

        if self.db_backup_interval_hours < 1:
            errors.append("DB_BACKUP_INTERVAL_HOURS must be >= 1")

        if not self.db_backup_filename.strip():
            errors.append("DB_BACKUP_FILENAME must not be empty")

        if errors:
            raise ValueError(
                "Configuration errors:\n  • " + "\n  • ".join(errors)
            )
