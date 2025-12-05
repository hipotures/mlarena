"""Time utility stubs (Phase 1 placeholder)."""

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
