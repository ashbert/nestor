"""Google Drive / Docs tools for Nestor."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

from googleapiclient.discovery import Resource, build

from nestor.google_auth import get_google_credentials
from nestor.tools import BaseTool

logger = logging.getLogger(__name__)

__all__ = [
    "build_drive_service",
    "CreateNoteTool",
    "ListNotesTool",
    "ReadNoteTool",
    "AppendNoteTool",
]


# ---------------------------------------------------------------------------
# Service builder
# ---------------------------------------------------------------------------

def build_drive_service(
    credentials_file: str,
    token_file: str,
) -> tuple[Resource, Resource]:
    """Build authenticated Google Drive **and** Docs API resources.

    Returns ``(drive_service, docs_service)``.
    """
    creds = get_google_credentials(credentials_file, token_file)
    drive_service = build("drive", "v3", credentials=creds)
    docs_service = build("docs", "v1", credentials=creds)
    return drive_service, docs_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking call in the default executor."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


def _extract_doc_text(doc: dict[str, Any]) -> str:
    """Walk the Docs body content and extract plain text."""
    body = doc.get("body", {})
    content = body.get("content", [])
    parts: list[str] = []
    for element in content:
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for pe in paragraph.get("elements", []):
            text_run = pe.get("textRun")
            if text_run:
                parts.append(text_run.get("content", ""))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class CreateNoteTool(BaseTool):
    """Creates a new Google Doc."""

    name = "create_note"
    description = (
        "Create a new Google Doc with a title and optional initial content. "
        "Optionally place it in a specific Drive folder."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Document title.",
            },
            "content": {
                "type": "string",
                "description": "Initial plain-text content for the document.",
            },
            "folder_id": {
                "type": "string",
                "description": "Google Drive folder ID to place the doc in (optional).",
            },
        },
        "required": ["title", "content"],
    }

    def __init__(self, drive_service: Resource, docs_service: Resource) -> None:
        self._drive = drive_service
        self._docs = docs_service

    async def execute(self, **kwargs: Any) -> str:
        title: str = kwargs["title"]
        content: str = kwargs["content"]
        folder_id: str | None = kwargs.get("folder_id")

        try:
            # 1. Create the document via the Docs API.
            doc = await _run_sync(
                self._docs.documents()
                .create(body={"title": title})
                .execute
            )
            doc_id: str = doc["documentId"]

            # 2. Insert initial content if provided.
            if content:
                requests = [
                    {
                        "insertText": {
                            "location": {"index": 1},
                            "text": content,
                        }
                    }
                ]
                await _run_sync(
                    self._docs.documents()
                    .batchUpdate(
                        documentId=doc_id,
                        body={"requests": requests},
                    )
                    .execute
                )

            # 3. Move to folder if requested.
            if folder_id:
                await _run_sync(
                    self._drive.files()
                    .update(
                        fileId=doc_id,
                        addParents=folder_id,
                        fields="id, parents",
                    )
                    .execute
                )

            url = f"https://docs.google.com/document/d/{doc_id}/edit"
            return f'Created document "{title}". URL: {url}'
        except Exception as exc:
            logger.exception("Failed to create note")
            return f"Error creating note: {exc}"


class ListNotesTool(BaseTool):
    """Lists Google Docs in a Drive folder."""

    name = "list_notes"
    description = (
        "List Google Docs, optionally filtered to a specific folder and/or "
        "a search query."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder_id": {
                "type": "string",
                "description": "Drive folder ID to scope the listing (optional).",
            },
            "query": {
                "type": "string",
                "description": "Search query to filter documents by name (optional).",
            },
        },
        "required": [],
    }

    def __init__(self, drive_service: Resource, docs_service: Resource) -> None:
        self._drive = drive_service
        self._docs = docs_service

    async def execute(self, **kwargs: Any) -> str:
        folder_id: str | None = kwargs.get("folder_id")
        query: str | None = kwargs.get("query")

        try:
            q_parts: list[str] = ["mimeType='application/vnd.google-apps.document'"]
            q_parts.append("trashed=false")
            if folder_id:
                q_parts.append(f"'{folder_id}' in parents")
            if query:
                # Drive search: name contains
                escaped = query.replace("'", "\\'")
                q_parts.append(f"name contains '{escaped}'")

            result = await _run_sync(
                self._drive.files()
                .list(
                    q=" and ".join(q_parts),
                    pageSize=50,
                    fields="files(id, name, modifiedTime, webViewLink)",
                    orderBy="modifiedTime desc",
                )
                .execute
            )
            files = result.get("files", [])
            if not files:
                return "No documents found."

            lines: list[str] = [f"Found {len(files)} document(s):"]
            for f in files:
                link = f.get("webViewLink", "N/A")
                lines.append(
                    f"\u2022 {f['name']}  (ID: {f['id']}, modified: {f.get('modifiedTime', '?')})\n  {link}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.exception("Failed to list notes")
            return f"Error listing notes: {exc}"


class ReadNoteTool(BaseTool):
    """Reads the text content of a Google Doc."""

    name = "read_note"
    description = "Read and return the plain-text content of a Google Doc."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The Google Docs document ID.",
            },
        },
        "required": ["document_id"],
    }

    def __init__(self, drive_service: Resource, docs_service: Resource) -> None:
        self._drive = drive_service
        self._docs = docs_service

    async def execute(self, **kwargs: Any) -> str:
        document_id: str = kwargs["document_id"]

        try:
            doc = await _run_sync(
                self._docs.documents()
                .get(documentId=document_id)
                .execute
            )
            title = doc.get("title", "Untitled")
            text = _extract_doc_text(doc)
            if not text.strip():
                return f'Document "{title}" is empty.'
            return f'--- {title} ---\n{text}'
        except Exception as exc:
            logger.exception("Failed to read note %s", document_id)
            return f"Error reading note: {exc}"


class AppendNoteTool(BaseTool):
    """Appends text to an existing Google Doc."""

    name = "append_note"
    description = "Append text content to the end of an existing Google Doc."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The Google Docs document ID.",
            },
            "content": {
                "type": "string",
                "description": "Text to append to the document.",
            },
        },
        "required": ["document_id", "content"],
    }

    def __init__(self, drive_service: Resource, docs_service: Resource) -> None:
        self._drive = drive_service
        self._docs = docs_service

    async def execute(self, **kwargs: Any) -> str:
        document_id: str = kwargs["document_id"]
        content: str = kwargs["content"]

        try:
            # Fetch current doc to find the end index.
            doc = await _run_sync(
                self._docs.documents()
                .get(documentId=document_id)
                .execute
            )
            body = doc.get("body", {})
            body_content = body.get("content", [])

            # The end index of the body content (subtract 1 for the trailing newline).
            end_index = 1
            if body_content:
                end_index = body_content[-1].get("endIndex", 1) - 1
            end_index = max(end_index, 1)

            text_to_insert = f"\n{content}" if end_index > 1 else content

            requests = [
                {
                    "insertText": {
                        "location": {"index": end_index},
                        "text": text_to_insert,
                    }
                }
            ]
            await _run_sync(
                self._docs.documents()
                .batchUpdate(
                    documentId=document_id,
                    body={"requests": requests},
                )
                .execute
            )

            title = doc.get("title", "Untitled")
            return f'Appended {len(content)} characters to "{title}".'
        except Exception as exc:
            logger.exception("Failed to append to note %s", document_id)
            return f"Error appending to note: {exc}"
