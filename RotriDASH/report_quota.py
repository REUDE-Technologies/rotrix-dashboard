#type: ignore
"""
Daily report-generation quota system for Rotrix Dashboard.

Uses local Postgres (SQLAlchemy) for quota storage.

Quota resolution: per-user override (profiles.daily_report_quota) > system default (100).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    DEFAULT_DAILY_REPORT_QUOTA = int(os.getenv("DEFAULT_DAILY_REPORT_QUOTA", "100"))
except Exception:
    DEFAULT_DAILY_REPORT_QUOTA = 100
REPORT_COOLDOWN_SECONDS = 10


def _get_db():
    """Return a SQLAlchemy session."""
    from models import SessionLocal
    return SessionLocal()


def get_daily_report_count(user_id: str) -> int:
    """Count reports generated today (UTC) by this user."""
    if not user_id:
        return 0

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    from models import ReportMetadata
    from sqlalchemy import func as sa_func
    db = _get_db()
    try:
        count = (
            db.query(sa_func.count(ReportMetadata.id))
            .filter(
                ReportMetadata.user_id == user_id,
                ReportMetadata.generated_at >= today_start,
            )
            .scalar()
        )
        return int(count or 0)
    except Exception as exc:
        logger.warning("Failed to count daily reports from PG: %s", exc)
        return 0
    finally:
        db.close()


def _read_user_quota(user_id: str) -> int | None:
    """Read per-user quota override from profiles.daily_report_quota."""
    if not user_id:
        return None

    from models import Profile
    db = _get_db()
    try:
        profile = db.get(Profile, user_id)
        if profile is None:
            return None
        return profile.daily_report_quota
    except Exception:
        return None
    finally:
        db.close()


def get_user_quota(user_id: str, org_id: str) -> int:
    """Get the effective daily report quota for a user."""
    user_q = _read_user_quota(user_id)
    if user_q is not None:
        return user_q
    import streamlit as st
    try:
        if st.session_state.get("user_role") == "super_admin":
            return 100
    except Exception:
        pass

    # For now we use a single system-wide default; org-level overrides can be added later.
    return DEFAULT_DAILY_REPORT_QUOTA


def get_remaining_quota(user_id: str, org_id: str) -> int:
    """Get remaining number of reports this user can generate today."""
    quota = get_user_quota(user_id, org_id)
    used = get_daily_report_count(user_id)
    return max(0, quota - used)


def check_quota(user_id: str, org_id: str, num_files: int = 1) -> Tuple[bool, int, int, int]:
    """Check if user can generate num_files reports.

    Applies both a daily quota and a short-term cooldown to avoid bursts
    of many heavy reports in quick succession.
    """
    import time
    import streamlit as st

    # Short-term cooldown (per-session, best-effort)
    last_gen = st.session_state.get("_last_report_generated_at", 0)
    if last_gen:
        elapsed = time.time() - float(last_gen)
        if elapsed < REPORT_COOLDOWN_SECONDS:
            quota = get_user_quota(user_id, org_id)
            used = get_daily_report_count(user_id)
            remaining = max(0, quota - used)
            return False, remaining, quota, used

    quota = get_user_quota(user_id, org_id)
    used = get_daily_report_count(user_id)
    remaining = max(0, quota - used)
    if remaining >= num_files:
        # Only stamp cooldown when a generation is actually allowed
        try:
            st.session_state["_last_report_generated_at"] = time.time()
        except Exception:
            pass
    return remaining >= num_files, remaining, quota, used


def set_user_quota(user_id: str, quota: int | None) -> Tuple[bool, str]:
    """Set daily report quota override for a specific user."""
    from models import Profile
    db = _get_db()
    try:
        profile = db.get(Profile, user_id)
        if profile is None:
            return False, "User profile not found."
        profile.daily_report_quota = quota
        db.commit()
        if quota is None:
            return True, "User quota override removed (using system default)."
        return True, f"User quota set to {quota}/day."
    except Exception as exc:
        db.rollback()
        return False, f"Failed to set user quota: {exc}"
    finally:
        db.close()
