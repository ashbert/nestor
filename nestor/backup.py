"""Periodic SQLite backups to Google Drive."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import tempfile
from pathlib import Path

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from nestor.tools.drive_tool import build_drive_service

logger = logging.getLogger(__name__)

_SQLITE_MIME = "application/vnd.sqlite3"


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _snapshot_sqlite_database(db_path: str, snapshot_path: Path) -> None:
    source = sqlite3.connect(db_path)
    dest = sqlite3.connect(str(snapshot_path))
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()


def _find_backup_file_id(
    *,
    drive_service,
    filename: str,
    folder_id: str | None,
) -> str | None:
    q_parts = [f"name = '{_escape_drive_query(filename)}'", "trashed = false"]
    if folder_id:
        q_parts.append(f"'{_escape_drive_query(folder_id)}' in parents")
    result = (
        drive_service.files()
        .list(
            q=" and ".join(q_parts),
            pageSize=1,
            fields="files(id)",
            orderBy="modifiedTime desc",
        )
        .execute()
    )
    files = result.get("files", [])
    return files[0]["id"] if files else None


def has_backup_file(
    *,
    credentials_file: str,
    token_file: str,
    filename: str,
    folder_id: str | None,
) -> bool:
    """Return True when a matching backup file exists in Google Drive."""
    drive_service, _ = build_drive_service(credentials_file, token_file)
    file_id = _find_backup_file_id(
        drive_service=drive_service,
        filename=filename,
        folder_id=folder_id,
    )
    return file_id is not None


def restore_database_if_missing(
    *,
    db_path: str,
    credentials_file: str,
    token_file: str,
    filename: str,
    folder_id: str | None,
) -> str:
    """Restore local DB from Drive backup when the local DB file is missing.

    Returns one of:
    - ``"local_exists"``: local DB is already present (no restore attempt).
    - ``"backup_not_found"``: no matching backup file exists in Drive.
    - ``"restored"``: backup file was downloaded and restored locally.
    """
    local_path = Path(db_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        return "local_exists"

    local_path.parent.mkdir(parents=True, exist_ok=True)

    drive_service, _ = build_drive_service(credentials_file, token_file)
    file_id = _find_backup_file_id(
        drive_service=drive_service,
        filename=filename,
        folder_id=folder_id,
    )
    if not file_id:
        return "backup_not_found"

    with tempfile.NamedTemporaryFile(
        prefix="nestor-db-restore-",
        suffix=".sqlite3",
        delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        request = drive_service.files().get_media(fileId=file_id)
        with temp_path.open("wb") as out:
            downloader = MediaIoBaseDownload(out, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        if temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded backup file is empty")

        temp_path.replace(local_path)
        return "restored"
    finally:
        temp_path.unlink(missing_ok=True)


def _upload_backup_snapshot(
    *,
    snapshot_path: Path,
    filename: str,
    credentials_file: str,
    token_file: str,
    folder_id: str | None,
) -> str:
    drive_service, _ = build_drive_service(credentials_file, token_file)
    existing_id = _find_backup_file_id(
        drive_service=drive_service,
        filename=filename,
        folder_id=folder_id,
    )

    media = MediaFileUpload(str(snapshot_path), mimetype=_SQLITE_MIME, resumable=False)
    if existing_id:
        response = (
            drive_service.files()
            .update(fileId=existing_id, media_body=media, fields="id")
            .execute()
        )
    else:
        body: dict[str, object] = {"name": filename}
        if folder_id:
            body["parents"] = [folder_id]
        response = (
            drive_service.files()
            .create(body=body, media_body=media, fields="id")
            .execute()
        )
    return response["id"]


def _backup_once(
    *,
    db_path: str,
    filename: str,
    credentials_file: str,
    token_file: str,
    folder_id: str | None,
) -> str:
    with tempfile.NamedTemporaryFile(
        prefix="nestor-db-backup-",
        suffix=".sqlite3",
        delete=False,
    ) as tmp:
        snapshot_path = Path(tmp.name)

    try:
        _snapshot_sqlite_database(db_path, snapshot_path)
        return _upload_backup_snapshot(
            snapshot_path=snapshot_path,
            filename=filename,
            credentials_file=credentials_file,
            token_file=token_file,
            folder_id=folder_id,
        )
    finally:
        snapshot_path.unlink(missing_ok=True)


async def run_periodic_drive_backup(
    *,
    db_path: str,
    credentials_file: str,
    token_file: str,
    stop_event: asyncio.Event,
    interval_hours: int = 24,
    filename: str = "nestor-backup-latest.sqlite3",
    folder_id: str | None = None,
    run_on_start: bool = True,
) -> None:
    """Continuously back up SQLite to Google Drive until *stop_event* is set."""
    interval_seconds = max(1, interval_hours * 3600)
    should_run_now = run_on_start

    logger.info(
        "Database backup task started (interval=%sh, filename=%s)",
        interval_hours,
        filename,
    )

    while not stop_event.is_set():
        if should_run_now:
            try:
                file_id = await asyncio.to_thread(
                    _backup_once,
                    db_path=db_path,
                    filename=filename,
                    credentials_file=credentials_file,
                    token_file=token_file,
                    folder_id=folder_id,
                )
                logger.info(
                    "Database backup uploaded to Google Drive (file_id=%s, filename=%s)",
                    file_id,
                    filename,
                )
            except Exception:
                logger.exception("Database backup failed")

        should_run_now = True

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue

    logger.info("Database backup task stopped")
