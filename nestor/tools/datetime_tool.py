"""Date/time tool for Nestor."""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from nestor.tools import BaseTool

logger = logging.getLogger(__name__)

__all__ = ["GetCurrentDateTimeTool"]

_DEFAULT_TIMEZONE = "America/New_York"


class GetCurrentDateTimeTool(BaseTool):
    """Returns the current date, time, and day of week."""

    name = "get_current_datetime"
    description = (
        "Get the current date, time, and day of week. "
        "Timezone is configurable via the NESTOR_TIMEZONE environment variable "
        "(defaults to America/New_York)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self, **kwargs: Any) -> str:
        try:
            tz_name = os.environ.get("NESTOR_TIMEZONE", _DEFAULT_TIMEZONE)
            try:
                tz = ZoneInfo(tz_name)
            except (KeyError, Exception):
                logger.warning(
                    "Invalid timezone %r, falling back to %s",
                    tz_name,
                    _DEFAULT_TIMEZONE,
                )
                tz = ZoneInfo(_DEFAULT_TIMEZONE)

            now = datetime.now(tz)
            return (
                f"Current date and time:\n"
                f"  Date: {now.strftime('%Y-%m-%d')}\n"
                f"  Time: {now.strftime('%H:%M:%S')}\n"
                f"  Day:  {now.strftime('%A')}\n"
                f"  Timezone: {tz_name}"
            )
        except Exception as exc:
            logger.exception("Failed to get current datetime")
            return f"Error getting current date/time: {exc}"
