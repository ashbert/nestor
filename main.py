#!/usr/bin/env python3
"""Nestor – Telegram butler bot.

Entry point: load configuration, wire up components, and run the
Telegram polling loop until interrupted.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

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

def _register_tools(config: Config) -> ToolRegistry:
    """Build the tool registry with all available tools.

    Google tools are only registered if credentials are available.
    """
    registry = ToolRegistry()

    # Always available
    registry.register(GetCurrentDateTimeTool())

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
    tools = _register_tools(config)

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

    # Block until shutdown signal.
    await _shutdown_event.wait()

    # Graceful teardown
    logger.info("Stopping Telegram polling…")
    await app.updater.stop()  # type: ignore[union-attr]
    await app.stop()
    await app.shutdown()
    logger.info("Nestor stopped.")


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
