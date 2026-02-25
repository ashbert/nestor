#!/usr/bin/env python3
"""Nestor – Telegram butler bot.

Entry point: load configuration, wire up components, and run the
Telegram polling loop until interrupted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from nestor.backup import (
    has_backup_file,
    restore_database_if_missing,
    run_periodic_drive_backup,
)
from nestor.config import Config
from nestor.memory import MemoryStore
from nestor.llm import create_provider
from nestor.brain import NestorBrain
from nestor.tools import ToolRegistry
from nestor.tools.datetime_tool import GetCurrentDateTimeTool
from nestor.telegram_handler import create_bot

logger = logging.getLogger("nestor")

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "nestor" / "prompts" / "system.txt"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def _register_tools(config: Config, memory: MemoryStore) -> ToolRegistry:
    """Build the tool registry with all available tools.

    Google tools are only registered if credentials are available.
    """
    registry = ToolRegistry()

    # Always available
    registry.register(GetCurrentDateTimeTool())
    try:
        from nestor.tools.memory_tool import RememberThoughtTool, RecallThoughtsTool
        registry.register(RememberThoughtTool(memory))
        registry.register(RecallThoughtsTool(memory))
        logger.info("Registered local memory tools")
    except Exception:
        logger.warning("Local memory tools unavailable", exc_info=True)

    # Web search / fetch
    try:
        from nestor.tools.search_tool import WebSearchTool, FetchWebPageTool
        registry.register(WebSearchTool())
        registry.register(FetchWebPageTool())
        logger.info("Registered web search tools")
    except Exception:
        logger.warning("Web search tools unavailable", exc_info=True)

    # Google Calendar
    creds_file = config.google_credentials_file
    token_file = config.google_token_file

    if Path(creds_file).exists() or Path(token_file).exists():
        try:
            from nestor.tools.calendar_tool import (
                build_calendar_service,
                CreateEventTool,
                ListEventsTool,
                DeleteEventTool,
                SearchEventsTool,
            )
            cal_service = build_calendar_service(creds_file, token_file)
            cal_id = config.google_calendar_id
            registry.register(CreateEventTool(cal_service, cal_id))
            registry.register(ListEventsTool(cal_service, cal_id))
            registry.register(DeleteEventTool(cal_service, cal_id))
            registry.register(SearchEventsTool(cal_service, cal_id))
            logger.info("Registered Google Calendar tools (calendar=%s)", cal_id)
        except Exception:
            logger.warning("Google Calendar tools unavailable", exc_info=True)

        # Google Drive / Docs
        try:
            from nestor.tools.drive_tool import (
                build_drive_service,
                CreateNoteTool,
                ListNotesTool,
                ReadNoteTool,
                AppendNoteTool,
            )
            drive_svc, docs_svc = build_drive_service(creds_file, token_file)
            registry.register(CreateNoteTool(drive_svc, docs_svc))
            registry.register(ListNotesTool(drive_svc, docs_svc))
            registry.register(ReadNoteTool(drive_svc, docs_svc))
            registry.register(AppendNoteTool(drive_svc, docs_svc))
            logger.info("Registered Google Drive tools")
        except Exception:
            logger.warning("Google Drive tools unavailable", exc_info=True)

    # Gmail (SMTP/IMAP — independent of Google Cloud OAuth)
    if config.gmail_app_password:
        try:
            from nestor.tools.email_tool import (
                SendEmailTool,
                SearchEmailTool,
                ReadEmailTool,
            )
            registry.register(SendEmailTool(config.gmail_address, config.gmail_app_password))
            registry.register(SearchEmailTool(config.gmail_address, config.gmail_app_password))
            registry.register(ReadEmailTool(config.gmail_address, config.gmail_app_password))
            logger.info("Registered Gmail tools (SMTP/IMAP)")
        except Exception:
            logger.warning("Gmail tools unavailable", exc_info=True)
    else:
        logger.info("GMAIL_APP_PASSWORD not set — email tools disabled")

    if not (Path(creds_file).exists() or Path(token_file).exists()):
        logger.info(
            "Google credentials not found (%s / %s) — calendar and drive tools disabled",
            creds_file,
            token_file,
        )

    logger.info("Tool registry: %s", registry)
    return registry


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

_shutdown_event: asyncio.Event | None = None


def _request_shutdown(sig: signal.Signals) -> None:
    logger.info("Received %s – shutting down…", sig.name)
    if _shutdown_event is not None:
        _shutdown_event.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run() -> None:
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # -- Env & config -------------------------------------------------------
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    config = Config.from_env()
    _setup_logging(config.log_level)
    logger.info("Nestor starting (provider=%s, model=%s)", config.llm_provider, config.llm_model)
    config.validate()

    # -- Signal handlers ----------------------------------------------------
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown, sig)

    restore_status = "local_exists"
    if config.db_restore_from_drive:
        if Path(config.google_credentials_file).exists() or Path(config.google_token_file).exists():
            try:
                restore_status = await asyncio.to_thread(
                    restore_database_if_missing,
                    db_path=config.database_path,
                    credentials_file=config.google_credentials_file,
                    token_file=config.google_token_file,
                    filename=config.db_backup_filename,
                    folder_id=config.db_backup_drive_folder_id,
                )
                if restore_status == "restored":
                    logger.info(
                        "Restored local database from Google Drive backup (%s)",
                        config.db_backup_filename,
                    )
                elif restore_status == "backup_not_found":
                    logger.info(
                        "No Google Drive DB backup found (%s) — starting with a fresh local DB",
                        config.db_backup_filename,
                    )
            except Exception:
                restore_status = "restore_error"
                logger.exception("Database restore from Google Drive failed")
        else:
            logger.info(
                "DB restore enabled but Google credentials are unavailable (%s / %s)",
                config.google_credentials_file,
                config.google_token_file,
            )

    # -- Database -----------------------------------------------------------
    memory = MemoryStore(config.database_path)
    logger.info("Database: %s", config.database_path)

    # -- LLM ----------------------------------------------------------------
    api_key = (
        config.anthropic_api_key
        if config.llm_provider == "anthropic"
        else config.openai_api_key
    )
    assert api_key, "API key missing (should have been caught by validate)"

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    llm = create_provider(
        provider=config.llm_provider,
        api_key=api_key,
        model=config.llm_model,
        system_prompt=system_prompt,
    )
    logger.info("LLM: %s / %s", config.llm_provider, config.llm_model)

    # -- Tools --------------------------------------------------------------
    tools = _register_tools(config, memory)

    # -- Brain --------------------------------------------------------------
    brain = NestorBrain(
        llm=llm,
        tool_registry=tools,
        memory=memory,
        system_prompt=system_prompt,
    )

    # -- Telegram bot -------------------------------------------------------
    allowed_ids = set(config.allowed_telegram_ids)

    app = create_bot(
        token=config.telegram_bot_token,
        allowed_ids=allowed_ids,
        message_handler=brain.handle_message,
        today_handler=brain.get_today_summary,
        week_handler=brain.get_week_summary,
    )

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)  # type: ignore[union-attr]
    logger.info("Telegram polling started – Nestor is ready.")

    backup_task: asyncio.Task[None] | None = None
    if config.db_backup_to_drive:
        if Path(config.google_credentials_file).exists() or Path(config.google_token_file).exists():
            run_on_start = config.db_backup_on_start
            db_path = Path(config.database_path)
            local_db_exists = db_path.exists() and db_path.stat().st_size > 0
            if local_db_exists:
                try:
                    remote_exists = await asyncio.to_thread(
                        has_backup_file,
                        credentials_file=config.google_credentials_file,
                        token_file=config.google_token_file,
                        filename=config.db_backup_filename,
                        folder_id=config.db_backup_drive_folder_id,
                    )
                    if not remote_exists:
                        run_on_start = True
                        logger.info(
                            "No Drive DB backup found for %s — forcing immediate startup backup",
                            config.db_backup_filename,
                        )
                except Exception:
                    logger.exception("Could not verify Drive DB backup presence")
            if restore_status == "restore_error":
                run_on_start = False
                logger.warning(
                    "Skipping immediate DB backup on startup because restore failed. "
                    "Scheduled backups will continue."
                )
            backup_task = asyncio.create_task(
                run_periodic_drive_backup(
                    db_path=config.database_path,
                    credentials_file=config.google_credentials_file,
                    token_file=config.google_token_file,
                    stop_event=_shutdown_event,
                    interval_hours=config.db_backup_interval_hours,
                    filename=config.db_backup_filename,
                    folder_id=config.db_backup_drive_folder_id,
                    run_on_start=run_on_start,
                )
            )
            logger.info(
                "Drive DB backups enabled (every %sh, file=%s)",
                config.db_backup_interval_hours,
                config.db_backup_filename,
            )
        else:
            logger.warning(
                "Drive DB backups enabled but Google credentials are unavailable "
                "(%s / %s)",
                config.google_credentials_file,
                config.google_token_file,
            )

    # Block until shutdown signal.
    await _shutdown_event.wait()

    # Graceful teardown
    logger.info("Stopping Telegram polling…")
    await app.updater.stop()  # type: ignore[union-attr]
    await app.stop()
    await app.shutdown()

    if backup_task:
        try:
            await asyncio.wait_for(backup_task, timeout=15)
        except asyncio.TimeoutError:
            backup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await backup_task

    logger.info("Nestor stopped.")


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
