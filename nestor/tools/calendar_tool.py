"""Google Calendar tools for Nestor."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from functools import partial
from typing import Any
from zoneinfo import ZoneInfo

from googleapiclient.discovery import Resource, build

from nestor.google_auth import get_google_credentials
from nestor.tools import BaseTool

logger = logging.getLogger(__name__)

__all__ = [
    "build_calendar_service",
    "CreateEventTool",
    "ListEventsTool",
    "DeleteEventTool",
    "SearchEventsTool",
]


# ---------------------------------------------------------------------------
# Service builder
# ---------------------------------------------------------------------------

def build_calendar_service(
    credentials_file: str,
    token_file: str,
) -> Resource:
    """Build an authenticated Google Calendar API *Resource*."""
    creds = get_google_credentials(credentials_file, token_file)
    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking call in the default executor."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


def _format_event(event: dict[str, Any]) -> str:
    """Format a single Calendar event into a readable string."""
    title = event.get("summary", "(no title)")
    event_id = event.get("id", "")
    start = event.get("start", {})
    end = event.get("end", {})

    start_str = start.get("dateTime", start.get("date", "?"))
    end_str = end.get("dateTime", end.get("date", ""))
    description = event.get("description", "")

    parts = [f"• {title}"]
    parts.append(f"  ID: {event_id}")
    parts.append(f"  Start: {start_str}")
    if end_str:
        parts.append(f"  End: {end_str}")
    if description:
        parts.append(f"  Description: {description}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class CreateEventTool(BaseTool):
    """Creates a Google Calendar event."""

    name = "create_calendar_event"
    description = (
        "Create a new event on Google Calendar. Supports timed events and "
        "all-day events."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Event title / summary.",
            },
            "date": {
                "type": "string",
                "description": "Event date in YYYY-MM-DD format.",
            },
            "start_time": {
                "type": "string",
                "description": "Start time in HH:MM (24-h). Omit for all-day events.",
            },
            "end_time": {
                "type": "string",
                "description": "End time in HH:MM (24-h). Omit for all-day events.",
            },
            "description": {
                "type": "string",
                "description": "Optional event description / notes.",
            },
            "all_day": {
                "type": "boolean",
                "description": "If true, create an all-day event (default false).",
            },
        },
        "required": ["title", "date"],
    }

    def __init__(self, calendar_service: Resource, calendar_id: str = "primary") -> None:
        self._service = calendar_service
        self._calendar_id = calendar_id

    async def execute(self, **kwargs: Any) -> str:
        title: str = kwargs["title"]
        date: str = kwargs["date"]
        start_time: str | None = kwargs.get("start_time")
        end_time: str | None = kwargs.get("end_time")
        description: str = kwargs.get("description", "")
        all_day: bool = kwargs.get("all_day", False)

        try:
            body: dict[str, Any] = {"summary": title}
            if description:
                body["description"] = description

            if all_day or (not start_time and not end_time):
                # All-day event: end date is exclusive, so add 1 day.
                start_date = datetime.strptime(date, "%Y-%m-%d").date()
                end_date = start_date + timedelta(days=1)
                body["start"] = {"date": str(start_date)}
                body["end"] = {"date": str(end_date)}
            else:
                st = start_time or "09:00"
                et = end_time or _default_end(st)
                tz = os.environ.get("NESTOR_TIMEZONE", "America/Los_Angeles")
                body["start"] = {"dateTime": f"{date}T{st}:00", "timeZone": tz}
                body["end"] = {"dateTime": f"{date}T{et}:00", "timeZone": tz}

            event = await _run_sync(
                self._service.events()
                .insert(calendarId=self._calendar_id, body=body)
                .execute
            )
            return (
                f"Event created: \"{event.get('summary')}\" on {date}. "
                f"Event ID: {event.get('id')}. "
                f"Link: {event.get('htmlLink', 'N/A')}"
            )
        except Exception as exc:
            logger.exception("Failed to create calendar event")
            return f"Error creating event: {exc}"


def _default_end(start_hhmm: str) -> str:
    """Return an end time 1 hour after *start_hhmm* (HH:MM)."""
    h, m = map(int, start_hhmm.split(":"))
    end = datetime(2000, 1, 1, h, m) + timedelta(hours=1)
    return end.strftime("%H:%M")


class ListEventsTool(BaseTool):
    """Lists Google Calendar events in a date range."""

    name = "list_calendar_events"
    description = "List events on Google Calendar between two dates (inclusive)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Range start in YYYY-MM-DD.",
            },
            "end_date": {
                "type": "string",
                "description": "Range end in YYYY-MM-DD (inclusive).",
            },
        },
        "required": ["start_date", "end_date"],
    }

    def __init__(self, calendar_service: Resource, calendar_id: str = "primary") -> None:
        self._service = calendar_service
        self._calendar_id = calendar_id

    async def execute(self, **kwargs: Any) -> str:
        start_date: str = kwargs["start_date"]
        end_date: str = kwargs["end_date"]

        try:
            # Use local timezone for accurate day boundaries.
            tz_name = os.environ.get("NESTOR_TIMEZONE", "America/Los_Angeles")
            tz = ZoneInfo(tz_name)
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz)
            end_dt = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=tz)
            time_min = start_dt.isoformat()
            time_max = end_dt.isoformat()

            result = await _run_sync(
                self._service.events()
                .list(
                    calendarId=self._calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=50,
                )
                .execute
            )
            events = result.get("items", [])
            if not events:
                return f"No events found between {start_date} and {end_date}."
            formatted = "\n\n".join(_format_event(e) for e in events)
            return f"Found {len(events)} event(s) from {start_date} to {end_date}:\n\n{formatted}"
        except Exception as exc:
            logger.exception("Failed to list calendar events")
            return f"Error listing events: {exc}"


class DeleteEventTool(BaseTool):
    """Deletes a Google Calendar event by ID."""

    name = "delete_calendar_event"
    description = "Delete an event from Google Calendar by its event ID."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The Google Calendar event ID to delete.",
            },
        },
        "required": ["event_id"],
    }

    def __init__(self, calendar_service: Resource, calendar_id: str = "primary") -> None:
        self._service = calendar_service
        self._calendar_id = calendar_id

    async def execute(self, **kwargs: Any) -> str:
        event_id: str = kwargs["event_id"]
        try:
            await _run_sync(
                self._service.events()
                .delete(calendarId=self._calendar_id, eventId=event_id)
                .execute
            )
            return f"Event {event_id} deleted successfully."
        except Exception as exc:
            logger.exception("Failed to delete calendar event %s", event_id)
            return f"Error deleting event {event_id}: {exc}"


class SearchEventsTool(BaseTool):
    """Searches Google Calendar events by text query."""

    name = "search_calendar_events"
    description = (
        "Search for upcoming Google Calendar events matching a text query."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search query.",
            },
            "days_ahead": {
                "type": "integer",
                "description": "How many days ahead to search (default 30).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, calendar_service: Resource, calendar_id: str = "primary") -> None:
        self._service = calendar_service
        self._calendar_id = calendar_id

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        days_ahead: int = kwargs.get("days_ahead", 30)

        try:
            tz_name = os.environ.get("NESTOR_TIMEZONE", "America/Los_Angeles")
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz=tz)
            time_min = now.isoformat()
            time_max = (now + timedelta(days=days_ahead)).isoformat()

            result = await _run_sync(
                self._service.events()
                .list(
                    calendarId=self._calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    q=query,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=25,
                )
                .execute
            )
            events = result.get("items", [])
            if not events:
                return f'No events matching "{query}" in the next {days_ahead} days.'
            formatted = "\n\n".join(_format_event(e) for e in events)
            return (
                f'Found {len(events)} event(s) matching "{query}":\n\n{formatted}'
            )
        except Exception as exc:
            logger.exception("Failed to search calendar events")
            return f"Error searching events: {exc}"
