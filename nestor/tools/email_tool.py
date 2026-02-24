"""Gmail tools for Nestor — send, read, and search emails via SMTP/IMAP.

Uses an App Password instead of OAuth/Gmail API to avoid
Google Cloud sensitive-scope review.
"""

from __future__ import annotations

import asyncio
import email
import email.utils
import imaplib
import logging
import smtplib
from email.mime.text import MIMEText
from functools import partial
from typing import Any

from nestor.tools import BaseTool

logger = logging.getLogger(__name__)

__all__ = [
    "SendEmailTool",
    "SearchEmailTool",
    "ReadEmailTool",
]

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587
_IMAP_HOST = "imap.gmail.com"
_IMAP_PORT = 993


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


def _parse_email_message(raw_bytes: bytes) -> dict[str, str]:
    """Parse raw email bytes into a dict with headers and body."""
    msg = email.message_from_bytes(raw_bytes)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="replace")

    return {
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "subject": msg.get("Subject", "(no subject)"),
        "date": msg.get("Date", ""),
        "body": body,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class SendEmailTool(BaseTool):
    """Send an email via Gmail SMTP."""

    name = "send_email"
    description = (
        "Send an email from Nestor's Gmail account. "
        "Can send to any email address."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line.",
            },
            "body": {
                "type": "string",
                "description": "Plain text email body.",
            },
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, gmail_address: str, app_password: str) -> None:
        self._address = gmail_address
        self._password = app_password

    def _send(self, to: str, subject: str, body: str) -> str:
        msg = MIMEText(body)
        msg["From"] = self._address
        msg["To"] = to
        msg["Subject"] = subject

        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self._address, self._password)
            server.sendmail(self._address, [to], msg.as_string())

        return f'Email sent to {to} (subject: "{subject}").'

    async def execute(self, **kwargs: Any) -> str:
        try:
            return await _run_sync(
                self._send, kwargs["to"], kwargs["subject"], kwargs["body"]
            )
        except Exception as exc:
            logger.exception("Failed to send email")
            return f"Error sending email: {exc}"


class SearchEmailTool(BaseTool):
    """Search Gmail messages via IMAP."""

    name = "search_email"
    description = (
        "Search for emails in Nestor's Gmail. Uses IMAP search criteria. "
        "Common searches: FROM \"someone@example.com\", SUBJECT \"keyword\", "
        "UNSEEN, SINCE \"01-Jan-2025\", ALL. "
        "Returns sender, subject, date, and preview for each match."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "IMAP search criteria. Examples: "
                    'FROM "john@example.com", '
                    'SUBJECT "meeting", '
                    'UNSEEN, '
                    'SINCE "01-Feb-2025", '
                    'OR FROM "alice" FROM "bob"'
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
            },
            "folder": {
                "type": "string",
                "description": 'IMAP folder to search (default "INBOX").',
            },
        },
        "required": ["query"],
    }

    def __init__(self, gmail_address: str, app_password: str) -> None:
        self._address = gmail_address
        self._password = app_password

    def _search(self, query: str, max_results: int, folder: str) -> str:
        with imaplib.IMAP4_SSL(_IMAP_HOST, _IMAP_PORT) as imap:
            imap.login(self._address, self._password)
            imap.select(folder, readonly=True)

            status, data = imap.search(None, query)
            if status != "OK" or not data[0]:
                return f'No emails found matching: {query}'

            msg_ids = data[0].split()
            # Take the most recent N
            msg_ids = msg_ids[-max_results:]
            msg_ids.reverse()  # newest first

            summaries: list[str] = []
            for msg_id in msg_ids:
                status, msg_data = imap.fetch(msg_id, "(BODY.PEEK[HEADER] BODY.PEEK[TEXT]<0.200>)")
                if status != "OK":
                    continue

                # Parse header
                header_bytes = b""
                snippet_bytes = b""
                for part in msg_data:
                    if isinstance(part, tuple):
                        desc = part[0].decode("utf-8", errors="replace").upper()
                        if "HEADER" in desc:
                            header_bytes = part[1]
                        elif "TEXT" in desc:
                            snippet_bytes = part[1]

                msg = email.message_from_bytes(header_bytes)
                snippet = snippet_bytes.decode("utf-8", errors="replace").strip()[:150]
                uid = msg_id.decode()

                summaries.append(
                    f"\u2022 UID: {uid}\n"
                    f"  From: {msg.get('From', '')}\n"
                    f"  Subject: {msg.get('Subject', '(no subject)')}\n"
                    f"  Date: {msg.get('Date', '')}\n"
                    f"  Preview: {snippet}"
                )

            return (
                f"Found {len(summaries)} email(s):\n\n"
                + "\n\n".join(summaries)
            )

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        max_results: int = kwargs.get("max_results", 10)
        folder: str = kwargs.get("folder", "INBOX")

        try:
            return await _run_sync(self._search, query, max_results, folder)
        except Exception as exc:
            logger.exception("Failed to search emails")
            return f"Error searching emails: {exc}"


class ReadEmailTool(BaseTool):
    """Read the full content of an email by its UID."""

    name = "read_email"
    description = (
        "Read the full content of a specific email by its UID. "
        "Use search_email first to find the UID."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "uid": {
                "type": "string",
                "description": "The email UID to read.",
            },
            "folder": {
                "type": "string",
                "description": 'IMAP folder (default "INBOX").',
            },
        },
        "required": ["uid"],
    }

    def __init__(self, gmail_address: str, app_password: str) -> None:
        self._address = gmail_address
        self._password = app_password

    def _read(self, uid: str, folder: str) -> str:
        with imaplib.IMAP4_SSL(_IMAP_HOST, _IMAP_PORT) as imap:
            imap.login(self._address, self._password)
            imap.select(folder, readonly=True)

            status, msg_data = imap.fetch(uid.encode(), "(RFC822)")
            if status != "OK":
                return f"Could not fetch email UID {uid}."

            raw = b""
            for part in msg_data:
                if isinstance(part, tuple):
                    raw = part[1]
                    break

            if not raw:
                return f"Empty message for UID {uid}."

            parsed = _parse_email_message(raw)
            body = parsed["body"]
            if len(body) > 4000:
                body = body[:4000] + "\n\n[...truncated]"

            return (
                f"From: {parsed['from']}\n"
                f"To: {parsed['to']}\n"
                f"Date: {parsed['date']}\n"
                f"Subject: {parsed['subject']}\n"
                f"{'=' * 40}\n"
                f"{body}"
            )

    async def execute(self, **kwargs: Any) -> str:
        uid: str = kwargs["uid"]
        folder: str = kwargs.get("folder", "INBOX")

        try:
            return await _run_sync(self._read, uid, folder)
        except Exception as exc:
            logger.exception("Failed to read email UID %s", uid)
            return f"Error reading email: {exc}"
