"""Migration readiness preflight checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from nestor.backup import has_backup_file
from nestor.config import Config


@dataclass(slots=True)
class PreflightResult:
    passed: bool
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _permission_mode(path: Path) -> str:
    return oct(path.stat().st_mode & 0o777)


def _is_owner_only_file(path: Path) -> bool:
    return (path.stat().st_mode & 0o777) == 0o600


async def run_migration_readiness_preflight(config: Config) -> PreflightResult:
    """Validate migration-critical settings and file states."""
    checks: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []

    db_path = Path(config.database_path)
    token_path = Path(config.google_token_file)
    creds_path = Path(config.google_credentials_file)

    if db_path.exists():
        checks.append(f"Local DB exists at {db_path}")
        if _is_owner_only_file(db_path):
            checks.append(f"Local DB permissions are restrictive ({_permission_mode(db_path)})")
        else:
            warnings.append(
                f"Local DB permissions are {_permission_mode(db_path)} (recommended: 0o600)"
            )
    else:
        warnings.append(f"Local DB not found at startup ({db_path})")

    if token_path.exists():
        if _is_owner_only_file(token_path):
            checks.append(f"Google token permissions are restrictive ({_permission_mode(token_path)})")
        else:
            failures.append(
                f"Google token permissions are {_permission_mode(token_path)} (required: 0o600)"
            )
    else:
        warnings.append(f"Google token file not found ({token_path})")

    creds_present = creds_path.exists()
    token_present = token_path.exists()
    if config.db_backup_to_drive or config.db_restore_from_drive:
        if not (creds_present or token_present):
            failures.append(
                "Drive backup/restore enabled but neither credentials nor token file is available"
            )
        else:
            checks.append("Drive backup/restore has usable Google auth material")

    if config.db_backup_hmac_key:
        checks.append("Backup manifest HMAC verification is enabled")
    else:
        warnings.append("DB_BACKUP_HMAC_KEY is unset (integrity is checksum-only)")

    if (config.db_backup_to_drive or config.db_restore_from_drive) and (creds_present or token_present):
        try:
            remote_exists = await asyncio.to_thread(
                has_backup_file,
                credentials_file=config.google_credentials_file,
                token_file=config.google_token_file,
                filename=config.db_backup_filename,
                folder_id=config.db_backup_drive_folder_id,
            )
            if remote_exists:
                checks.append(f"Found remote DB backup file ({config.db_backup_filename})")
            else:
                warnings.append(f"Remote DB backup not found ({config.db_backup_filename})")
        except Exception as exc:
            warnings.append(f"Could not verify remote backup presence: {exc}")

    return PreflightResult(
        passed=not failures,
        checks=checks,
        warnings=warnings,
        failures=failures,
    )
