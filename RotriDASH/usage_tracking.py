#type: ignore
"""
Usage tracking module for Rotrix Dashboard.

Logs user actions (login, upload, report generation, plot creation)
into the usage_events table for admin analytics.

Usage:
    from usage_tracking import track_login, track_file_upload, track_report_generated
"""

import logging
import streamlit as st

logger = logging.getLogger(__name__)

# Deduplication: skip same event type within this many seconds (avoids duplicate events on reruns)
_TRACK_DEDUP_SECONDS = 5


def _get_db():
    """Return a SQLAlchemy session."""
    from models import SessionLocal
    return SessionLocal()


# ---------------------------------------------------------------------------
# Core event logger
# ---------------------------------------------------------------------------
def track_event(user_id: str, org_id: str, event_type: str,
                metadata: dict | None = None) -> bool:
    """Insert a usage event into the usage_events table.

    Args:
        user_id: UUID of the user
        org_id: UUID of the organization
        event_type: One of 'login', 'file_uploaded', 'report_generated',
                    'plot_created', 'file_downloaded'
        metadata: Optional dict with extra info (stored as JSONB)

    Returns:
        True on success, False on failure.
    """
    try:
        import time as _t
        now = _t.time()
        dedup_key = "_usage_track_last"
        last = st.session_state.get(dedup_key, {})
        if last.get("event_type") == event_type and (now - last.get("t", 0)) < _TRACK_DEDUP_SECONDS:
            return True

        from models import UsageEvent
        db = _get_db()
        try:
            event = UsageEvent(
                user_id=user_id,
                organization_id=org_id,
                event_type=event_type,
                event_metadata=metadata or {},
            )
            db.add(event)
            db.commit()
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()

        st.session_state[dedup_key] = {"event_type": event_type, "t": now}
        return True
    except Exception:
        # Usage tracking is non-critical; silently ignore failures
        # (e.g. row-level security violations) so they do not clutter logs.
        return False


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def track_login(user_id: str, org_id: str) -> bool:
    """Log a user login event."""
    return track_event(user_id, org_id, "login")


def track_file_upload(user_id: str, org_id: str,
                      filename: str, file_size: int = 0) -> bool:
    """Log a file upload event."""
    return track_event(user_id, org_id, "file_uploaded", {
        "filename": filename,
        "file_size": file_size,
    })


def track_report_generated(user_id: str, org_id: str,
                           report_name: str,
                           source_file: str = "") -> bool:
    """Log a report generation event."""
    return track_event(user_id, org_id, "report_generated", {
        "report_name": report_name,
        "source_file": source_file,
    })


def track_plot_created(user_id: str, org_id: str,
                       plot_type: str = "multi_param") -> bool:
    """Log a plot creation event."""
    return track_event(user_id, org_id, "plot_created", {
        "plot_type": plot_type,
    })


def track_file_download(user_id: str, org_id: str,
                        filename: str) -> bool:
    """Log a file download event."""
    return track_event(user_id, org_id, "file_downloaded", {
        "filename": filename,
    })


# ---------------------------------------------------------------------------
# Helpers for getting current user context
# ---------------------------------------------------------------------------
def _get_user_context() -> tuple[str | None, str | None]:
    """Get (user_id, org_id) from session state for convenience. Treats empty string as missing."""
    user_id = st.session_state.get("user_id")
    org_id = st.session_state.get("organization_id")
    if user_id == "":
        user_id = None
    if org_id == "":
        org_id = None
    return user_id, org_id


def auto_track_login():
    """Track login using current session state. Call after successful login."""
    uid, oid = _get_user_context()
    if uid is not None and oid is not None:
        track_login(uid, oid)


def auto_track_file_upload(filename: str, file_size: int = 0):
    """Track file upload using current session state."""
    uid, oid = _get_user_context()
    if uid is not None and oid is not None:
        track_file_upload(uid, oid, filename, file_size)


def auto_track_report(report_name: str, source_file: str = ""):
    """Track report generation using current session state."""
    uid, oid = _get_user_context()
    if uid is not None and oid is not None:
        track_report_generated(uid, oid, report_name, source_file)


def auto_track_plot(plot_type: str = "multi_param"):
    """Track plot creation using current session state."""
    uid, oid = _get_user_context()
    if uid is not None and oid is not None:
        track_plot_created(uid, oid, plot_type)
