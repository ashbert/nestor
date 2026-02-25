"""Periodic SQLite backups to Google Drive."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import sqlite3
import tempfile
from pathlib import Path

from googleapiclient.http import (
    MediaFileUpload,
    MediaInMemoryUpload,
    MediaIoBaseDownload,
)

from nestor.tools.drive_tool import build_drive_service

logger = logging.getLogger(__name__)

_SQLITE_MIME = "application/vnd.sqlite3"
_MANIFEST_MIME = "application/json"
_MANIFEST_SUFFIX = ".manifest.json"


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _manifest_filename(filename: str) -> str:
    return f"{filename}{_MANIFEST_SUFFIX}"


def _file_digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_hmac_sha256(path: Path, key: str) -> str:
    sig = hmac.new(key.encode("utf-8"), digestmod=hashlib.sha256)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sig.update(chunk)
    return sig.hexdigest()


def _snapshot_sqlite_database(db_path: str, snapshot_path: Path) -> None:
    source = sqlite3.connect(db_path)
    dest = sqlite3.connect(str(snapshot_path))
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()


def _find_drive_file(
    *,
    drive_service,
    filename: str,
    folder_id: str | None,
    fields: str = "id",
) -> dict | None:
    q_parts = [f"name = '{_escape_drive_query(filename)}'", "trashed = false"]
    if folder_id:
        q_parts.append(f"'{_escape_drive_query(folder_id)}' in parents")
    result = (
        drive_service.files()
        .list(
            q=" and ".join(q_parts),
            pageSize=1,
            fields=f"files({fields})",
            orderBy="modifiedTime desc",
        )
        .execute()
    )
    files = result.get("files", [])
    return files[0] if files else None


def has_backup_file(
    *,
    credentials_file: str,
    token_file: str,
    filename: str,
    folder_id: str | None,
) -> bool:
    """Return True when a matching backup file exists in Google Drive."""
    drive_service, _ = build_drive_service(credentials_file, token_file)
    file_meta = _find_drive_file(
        drive_service=drive_service,
        filename=filename,
        folder_id=folder_id,
    )
    return file_meta is not None


def _download_drive_file_to_path(drive_service, file_id: str, out_path: Path) -> None:
    request = drive_service.files().get_media(fileId=file_id)
    with out_path.open("wb") as out:
        downloader = MediaIoBaseDownload(out, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _download_drive_file_bytes(drive_service, file_id: str) -> bytes:
    request = drive_service.files().get_media(fileId=file_id)
    out = io.BytesIO()
    downloader = MediaIoBaseDownload(out, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return out.getvalue()


def _load_manifest(
    *,
    drive_service,
    filename: str,
    folder_id: str | None,
) -> dict | None:
    meta = _find_drive_file(
        drive_service=drive_service,
        filename=_manifest_filename(filename),
        folder_id=folder_id,
        fields="id,name,modifiedTime",
    )
    if not meta:
        return None
    raw = _download_drive_file_bytes(drive_service, meta["id"])
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Backup manifest JSON is invalid") from exc


def _verify_restored_file(
    *,
    local_path: Path,
    backup_meta: dict,
    manifest: dict | None,
    hmac_key: str | None,
) -> None:
    local_md5 = _file_digest(local_path, "md5")
    remote_md5 = (backup_meta.get("md5Checksum") or "").lower()
    if remote_md5 and local_md5 != remote_md5:
        raise RuntimeError("Drive backup checksum mismatch (md5)")

    if manifest:
        expected_sha256 = manifest.get("sha256")
        if expected_sha256:
            local_sha256 = _file_digest(local_path, "sha256")
            if local_sha256 != str(expected_sha256).lower():
                raise RuntimeError("Backup manifest SHA-256 mismatch")

        manifest_hmac = manifest.get("hmac_sha256")
        if manifest_hmac:
            if not hmac_key:
                raise RuntimeError("Backup manifest requires HMAC key but none is configured")
            local_hmac = _file_hmac_sha256(local_path, hmac_key)
            if local_hmac != str(manifest_hmac).lower():
                raise RuntimeError("Backup manifest HMAC mismatch")
    elif hmac_key:
        raise RuntimeError("HMAC key is configured but no backup manifest was found")


def restore_database_if_missing(
    *,
    db_path: str,
    credentials_file: str,
    token_file: str,
    filename: str,
    folder_id: str | None,
    hmac_key: str | None = None,
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
    backup_meta = _find_drive_file(
        drive_service=drive_service,
        filename=filename,
        folder_id=folder_id,
        fields="id,name,md5Checksum,size",
    )
    if not backup_meta:
        return "backup_not_found"

    with tempfile.NamedTemporaryFile(
        prefix="nestor-db-restore-",
        suffix=".sqlite3",
        delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        _download_drive_file_to_path(
            drive_service=drive_service,
            file_id=backup_meta["id"],
            out_path=temp_path,
        )

        if temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded backup file is empty")

        manifest = _load_manifest(
            drive_service=drive_service,
            filename=filename,
            folder_id=folder_id,
        )
        _verify_restored_file(
            local_path=temp_path,
            backup_meta=backup_meta,
            manifest=manifest,
            hmac_key=hmac_key,
        )

        temp_path.replace(local_path)
        return "restored"
    finally:
        temp_path.unlink(missing_ok=True)


def _upload_backup_snapshot(
    *,
    drive_service,
    snapshot_path: Path,
    filename: str,
    folder_id: str | None,
) -> str:
    existing = _find_drive_file(
        drive_service=drive_service,
        filename=filename,
        folder_id=folder_id,
    )

    media = MediaFileUpload(str(snapshot_path), mimetype=_SQLITE_MIME, resumable=False)
    if existing:
        response = (
            drive_service.files()
            .update(fileId=existing["id"], media_body=media, fields="id")
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


def _upload_backup_manifest(
    *,
    drive_service,
    filename: str,
    folder_id: str | None,
    backup_file_id: str,
    snapshot_path: Path,
    hmac_key: str | None,
) -> str:
    manifest_payload: dict[str, object] = {
        "version": 1,
        "backup_filename": filename,
        "backup_file_id": backup_file_id,
        "size_bytes": snapshot_path.stat().st_size,
        "sha256": _file_digest(snapshot_path, "sha256"),
    }
    if hmac_key:
        manifest_payload["hmac_sha256"] = _file_hmac_sha256(snapshot_path, hmac_key)

    manifest_bytes = json.dumps(
        manifest_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    media = MediaInMemoryUpload(
        manifest_bytes, mimetype=_MANIFEST_MIME, resumable=False
    )
    manifest_name = _manifest_filename(filename)

    existing = _find_drive_file(
        drive_service=drive_service,
        filename=manifest_name,
        folder_id=folder_id,
    )
    if existing:
        response = (
            drive_service.files()
            .update(fileId=existing["id"], media_body=media, fields="id")
            .execute()
        )
    else:
        body: dict[str, object] = {"name": manifest_name}
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
    hmac_key: str | None,
) -> str:
    with tempfile.NamedTemporaryFile(
        prefix="nestor-db-backup-",
        suffix=".sqlite3",
        delete=False,
    ) as tmp:
        snapshot_path = Path(tmp.name)

    try:
        _snapshot_sqlite_database(db_path, snapshot_path)
        drive_service, _ = build_drive_service(credentials_file, token_file)
        backup_file_id = _upload_backup_snapshot(
            drive_service=drive_service,
            snapshot_path=snapshot_path,
            filename=filename,
            folder_id=folder_id,
        )
        _upload_backup_manifest(
            drive_service=drive_service,
            filename=filename,
            folder_id=folder_id,
            backup_file_id=backup_file_id,
            snapshot_path=snapshot_path,
            hmac_key=hmac_key,
        )
        return backup_file_id
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
    hmac_key: str | None = None,
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
                    hmac_key=hmac_key,
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
