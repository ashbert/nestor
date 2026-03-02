"""Configuration loaded entirely from environment variables.

Every secret and tuneable is read from the environment (or a .env file
loaded by the caller).  Nothing is ever hardcoded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated string of integers, ignoring blanks."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_str_list(raw: str) -> list[str]:
    """Parse a comma-separated string of non-empty strings."""
    return [p.strip() for p in raw.split(",") if p.strip()]


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


def _default_model(provider: str, tier: str = "deep") -> str:
    """Return the default model name for a provider and tier."""
    defaults = {
        "anthropic": {
            "fast": "claude-3-5-haiku-latest",
            "deep": "claude-sonnet-4-20250514",
        },
        "openai": {
            "fast": "gpt-4o-mini",
            "deep": "gpt-4o",
        },
    }
    provider_defaults = defaults.get(provider, defaults["anthropic"])
    return provider_defaults.get(tier, provider_defaults["deep"])


def _parse_json_object(raw: str | None) -> dict[str, str]:
    """Parse JSON object env var to ``dict[str, str]`` (best effort)."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, str):
            out[key] = value
    return out


@dataclass
class Config:
    """Application configuration – every field sourced from env vars."""

    # --- Telegram -----------------------------------------------------------
    telegram_bot_token: str = ""
    allowed_telegram_ids: list[int] = field(default_factory=list)
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_signing_secret: Optional[str] = None
    allowed_slack_user_ids: list[str] = field(default_factory=list)
    allowed_slack_channel_ids: list[str] = field(default_factory=list)
    slack_require_mention: bool = True
    slack_allow_thread_followups: bool = True

    # --- LLM ----------------------------------------------------------------
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_model_fast: str = ""
    llm_model_deep: str = ""
    channel_model_overrides: dict[str, str] = field(default_factory=dict)
    enable_parallel_research: bool = True

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
    db_backup_hmac_key: Optional[str] = None

    # --- Factory ------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from the current environment."""
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        model = os.environ.get("LLM_MODEL", "") or _default_model(provider, "deep")
        model_fast = os.environ.get("LLM_MODEL_FAST", "") or _default_model(provider, "fast")
        model_deep = os.environ.get("LLM_MODEL_DEEP", "") or model

        raw_ids = os.environ.get("ALLOWED_TELEGRAM_IDS", "")
        raw_slack_users = os.environ.get("ALLOWED_SLACK_USER_IDS", "")
        raw_slack_channels = os.environ.get("ALLOWED_SLACK_CHANNEL_IDS", "")
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
            llm_model_fast=model_fast,
            llm_model_deep=model_deep,
            channel_model_overrides=_parse_json_object(
                os.environ.get("CHANNEL_MODEL_OVERRIDES")
            ),
            enable_parallel_research=_parse_bool(
                os.environ.get("ENABLE_PARALLEL_RESEARCH"),
                True,
            ),
            slack_bot_token=os.environ.get("SLACK_BOT_TOKEN", ""),
            slack_app_token=os.environ.get("SLACK_APP_TOKEN", ""),
            slack_signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
            allowed_slack_user_ids=_parse_str_list(raw_slack_users),
            allowed_slack_channel_ids=_parse_str_list(raw_slack_channels),
            slack_require_mention=_parse_bool(
                os.environ.get("SLACK_REQUIRE_MENTION"),
                True,
            ),
            slack_allow_thread_followups=_parse_bool(
                os.environ.get("SLACK_ALLOW_THREAD_FOLLOWUPS"),
                True,
            ),
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
            db_backup_hmac_key=os.environ.get("DB_BACKUP_HMAC_KEY"),
        )

    # --- Validation ---------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`ValueError` if any required variable is missing."""
        errors: list[str] = []

        telegram_enabled = bool(self.telegram_bot_token and self.allowed_telegram_ids)
        slack_enabled = bool(self.slack_bot_token and self.slack_app_token)

        if not telegram_enabled and not slack_enabled:
            errors.append(
                "At least one transport must be configured: "
                "Telegram (TELEGRAM_BOT_TOKEN + ALLOWED_TELEGRAM_IDS) or "
                "Slack (SLACK_BOT_TOKEN + SLACK_APP_TOKEN)"
            )

        if self.telegram_bot_token and not self.allowed_telegram_ids:
            errors.append(
                "ALLOWED_TELEGRAM_IDS is required when TELEGRAM_BOT_TOKEN is set"
            )

        if self.slack_bot_token and not self.slack_app_token:
            errors.append(
                "SLACK_APP_TOKEN is required when SLACK_BOT_TOKEN is set"
            )

        if self.slack_app_token and not self.slack_bot_token:
            errors.append(
                "SLACK_BOT_TOKEN is required when SLACK_APP_TOKEN is set"
            )

        if slack_enabled and not self.allowed_slack_user_ids:
            errors.append(
                "ALLOWED_SLACK_USER_IDS is required when Slack is enabled"
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

        if not self.llm_model_fast.strip():
            errors.append("LLM_MODEL_FAST must not be empty")

        if not self.llm_model_deep.strip():
            errors.append("LLM_MODEL_DEEP must not be empty")

        if errors:
            raise ValueError(
                "Configuration errors:\n  • " + "\n  • ".join(errors)
            )
