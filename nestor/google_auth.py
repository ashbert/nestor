"""Shared Google OAuth2 authentication for Nestor.

All Google services share a single token with combined scopes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

# Combined scopes for all Google services Nestor uses.
ALL_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/documents",
]

_cached_creds: Credentials | None = None


def _set_owner_only_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.warning("Could not set secure permissions on %s", path, exc_info=True)


def get_google_credentials(
    credentials_file: str,
    token_file: str,
) -> Credentials:
    """Get valid Google OAuth2 credentials.

    Uses a module-level cache so the OAuth flow runs at most once per
    process lifetime.

    - If *token_file* exists with valid/refreshable creds, reuse them.
    - Otherwise run the installed-app browser flow.
    - Always persist the (refreshed) token back to *token_file*.
    """
    global _cached_creds

    if _cached_creds and _cached_creds.valid:
        return _cached_creds

    creds: Credentials | None = None
    token_path = Path(token_file)

    if token_path.exists():
        _set_owner_only_permissions(token_path)
        creds = Credentials.from_authorized_user_file(str(token_path), ALL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired Google token")
            creds.refresh(Request())
        else:
            logger.info("Starting Google OAuth browser flow")
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, ALL_SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        _set_owner_only_permissions(token_path)
        logger.info("Google token saved to %s", token_path)

    _cached_creds = creds
    return creds
