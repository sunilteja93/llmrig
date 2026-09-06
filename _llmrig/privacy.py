"""Privacy checks shared by internal public-result representations."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Optional, Tuple


_PATH_SHAPED = re.compile(
    r"(?:^|[\s\"'(\[{:;,=])(?:/(?!/)[^\s]+|~[/\\][^\s]+|\.{1,2}[/\\][^\s]+|"
    r"[A-Za-z]:[\\/][^\s]+|\\\\[^\s]+)"
)


def private_identity_values() -> Tuple[str, ...]:
    """Return local identity values only for rejecting accidental serialization."""
    values = (
        str(os.environ.get("USER") or "").strip(),
        str(os.environ.get("USERNAME") or "").strip(),
        str(platform.node() or "").strip(),
    )
    return tuple(value for value in values if len(value) >= 3)


def contains_private_path(value: str) -> bool:
    """Detect local path-shaped text without interpreting public repository IDs."""
    text = str(value)
    home = str(Path.home())
    return bool(
        "\x00" in text
        or "\n" in text
        or "\r" in text
        or (home and home in text)
        or _PATH_SHAPED.search(text)
    )


def contains_private_identity(value: str) -> bool:
    """Detect a local username or hostname as a distinct public-text token."""
    text = str(value).strip()
    return any(
        re.search(
            rf"(?<![A-Za-z0-9_.-]){re.escape(item)}(?![A-Za-z0-9_.-])",
            text,
            re.IGNORECASE,
        )
        is not None
        for item in private_identity_values()
    )


def privacy_safe_public_id(runtime: str, value: str) -> str:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > 500
        or contains_private_path(text)
        or contains_private_identity(text)
    ):
        return f"local:{runtime}:private-identity"
    return text


def safe_optional_public_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if (
        not text
        or contains_private_path(text)
        or contains_private_identity(text)
    ):
        return None
    return text


def validate_public_text(
    value: Optional[str], label: str, *, identity: bool = False
) -> Optional[str]:
    """Return safe public text or fail without echoing its private contents."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 500 or contains_private_path(text):
        raise ValueError(f"{label} is not safe public metadata")
    if contains_private_identity(text):
        raise ValueError(f"{label} is not safe public metadata")
    return text
